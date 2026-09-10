"""편성(compose) 라우트 — 질의 1건 → 편성 flow → t_compose 저장 + 결과.

계약:
- 선곡 LLM 콜이 1~3분이라 POST 는 202 + 폴링 패턴 (단일 워커 전제).
  같은 v_id 동시 편성은 허용 — 읽기 전용 + comp_id 신규 발급이라 충돌이 없다.
- 폴링 식별자는 접수 시 선-INSERT 로 발급되는 comp_id 다 — 별도 잡 ID 가 없다.
- GET /compose?v_id=&comp_id= 로 진행(progress)·결과를 폴링한다 — 실행 중이면
  인메모리 잡을, 아니면 저장된 편성(t_compose)을 돌려준다. comp_id 가 v_id 안
  시퀀스라 둘 다 받는다.
- 렌더는 여기서 다루지 않는다 — 편성 완료 후 별도 POST /render 로만 요청한다.
"""

import asyncio
import time

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

from api.errors import ComposeNotFoundError
from api.jobs import JobStore, RunningGuard
from domains.baseball import callback
from log import bind_v_id, get_logger
from pipeline import dispatch
from rdb.composes import ComposeRepo, ComposeStatus, result_of
from rdb.videos import VideoRepo

log = get_logger(__name__)
router = APIRouter(tags=["compose"])

_jobs = JobStore()      # (v_id, comp_id) → 상태·결과 (프로세스 수명 캐시)

# 편성은 한 번에 한 건만 받는다 — 슬롯이 하나뿐이라 키를 고정한다
_SLOT = "compose"
_guard = RunningGuard("compose")


class ComposeRequest(BaseModel):
    """편성 요청."""

    v_id: int
    query: str
    stream_id: str = "VOD"   # 보관 전용 — 편성 범위를 좁히지 않는다
    # 목표 분량(초). 없으면 절단하지 않는다 — 선곡이 곧 편성이다.
    # 예산은 마감 단계의 **덜어내기 전용**이다: 예산을 채우려고 선곡에 없던 장면을
    # 끌어오는 통로는 열지 않는다 (질의를 규칙이 덮어쓰게 된다 — 설계 결정).
    budget_sec: int | None = None
    callback_url: str | None = None


class ComposeAccepted(BaseModel):
    """접수 응답 — 접수 가부를 code(0=성공)로 알린다. 실패면 search_id 가 없다."""

    v_id: int
    stream_id: str | None = None
    search_id: str | None = None
    code: int
    result: str


@router.post("/compose", response_model=ComposeAccepted,
             response_model_exclude_none=True)
async def post_compose(req: ComposeRequest, request: Request,
                       background: BackgroundTasks) -> ComposeAccepted:
    """편성 접수 — 완료는 GET /compose?v_id=&comp_id= 로 폴링.

    요청 형식이 맞으면 200 이고, 접수 가부는 본문 code 가 알린다 (0=성공 / -1=실패).
    접수와 동시에 t_compose 헤더를 선-INSERT 한다 (comp_id·search_id 즉시 발급,
    status=PLAN) — 진행 국면이 status_code 로 드러나고, 실패도 ERROR 행으로 남는다.
    """
    video = await VideoRepo(request.app.state.db).get(req.v_id)
    if video is None:
        return ComposeAccepted(v_id=req.v_id, code=-1, result="영상이 없습니다.")

    flow = dispatch.resolve(video.cate_id)
    if flow is None:
        return ComposeAccepted(v_id=req.v_id, code=-1,
                               result=f"지원하지 않는 카테고리입니다. (cate_id={video.cate_id})")

    if not _guard.try_acquire(_SLOT):
        return ComposeAccepted(v_id=req.v_id, code=-1,
                               result="편성이 진행 중입니다. 잠시 후 다시 요청해 주세요.")

    try:
        comp_id, search_id = await ComposeRepo(request.app.state.db).create(
            req.v_id, req.query, req.budget_sec, req.stream_id, req.callback_url)
    except Exception:
        _guard.release(_SLOT)   # 백그라운드가 못 떴으니 여기서 놓는다
        raise

    _jobs.create((req.v_id, comp_id),
                 v_id=req.v_id, comp_id=comp_id, query=req.query, progress=[])
    background.add_task(_run, request, comp_id, search_id, flow, req)
    log.info("편성 접수: v_id=%s comp_id=%s search_id=%s cate_id=%s(%s) %r",
             req.v_id, comp_id, search_id, video.cate_id, flow.__module__, req.query)
    return ComposeAccepted(v_id=req.v_id, stream_id=req.stream_id,
                           search_id=search_id, code=0, result="OK")


# 노드 완료 → 다음 국면 코드. 코드는 노드가 아니라 국면이라 전 노드를 다 적지 않는다
# (PLAN 은 create() 가 접수 시 찍고, 종결 OK/EMPTY 는 finish() 가 찍는다).
_PHASE_AFTER = {
    "select_clips": ComposeStatus.CUT,        # 선곡 끝 → 클립 구간 확정 국면
    "select_end_point": ComposeStatus.VERIFY,  # 끝점 확정 끝 → 검수·절단 국면
}


async def _run(request: Request, comp_id: int, search_id: str, flow,
               req: ComposeRequest) -> None:
    """백그라운드 본체 — flow 실행 → 잡 갱신. 실패는 잡의 error 로 드러낸다.

    편성 국면은 t_compose.status_code 에 기록한다 (t_video 는 안 건드린다).
    """
    st = request.app.state
    repo = ComposeRepo(st.db)
    job_key = (req.v_id, comp_id)
    progress: list[dict] = _jobs.get(job_key)["progress"]
    started = time.monotonic()

    async def on_node(node: str, elapsed: float) -> None:
        """노드 완료마다 진행 목록에 이름·소요 초를 쌓고, 국면 전환을 DB 에 찍는다."""
        progress.append({"node": node, "sec": elapsed})
        phase = _PHASE_AFTER.get(node)
        if phase is not None:
            await repo.set_status(req.v_id, comp_id, phase)

    try:
        with bind_v_id(req.v_id):
            state = await flow(req.v_id, comp_id, req.query, req.budget_sec,
                               st.db, st.llm, st.embedder, st.vector, st.settings,
                               on_node=on_node)

            # 종결 — 클립 저장 + 최종 코드 (empty 도 이력으로 남긴다)
            final = (ComposeStatus.EMPTY if state.get("status") == "empty"
                     else ComposeStatus.OK)
            rows = _clip_rows(state)
            await repo.finish(req.v_id, comp_id, final, rows)

        _jobs.replace(job_key, {
            "comp_id": comp_id,
            "status": state.get("status", "ok"),
            "v_id": req.v_id, "query": req.query, "progress": progress,
            "budget_sec": req.budget_sec,
            "elapsed_sec": round(time.monotonic() - started, 1),
            # 응답은 JSON 직렬화 가능한 요약만 — 그래프 확장에 맞춰 채워 간다
            "scene_count": len(state.get("scenes", [])),
            "spec": state.get("spec"),
            "evidence": state.get("evidence"),
            "evidence_orphan": state.get("evidence_orphan"),
            "candidates": state.get("candidates"),
            "picked": state.get("picked"),
            "clips": state.get("clips"),
            "dropped": state.get("dropped"),
            "duration_sec": sum(c["sec"] for c in (state.get("clips") or [])),
        })
        await _notify(req, search_id, 0, "OK", rows, st.settings)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("compose 실패: v_id=%s %r", req.v_id, req.query)
        # 실패도 행으로 남긴다 — 사유는 잡·로그 소유 (set_status 는 실패를 삼킨다:
        # DB 가 원인인 실패에서 스탬프까지 또 터져 원래 예외를 가리면 안 된다)
        await repo.set_status(req.v_id, comp_id, ComposeStatus.ERROR)
        _jobs.replace(job_key, {
            "comp_id": comp_id,
            "status": "error", "v_id": req.v_id, "query": req.query,
            "progress": progress, "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": round(time.monotonic() - started, 1),
        })
        await _notify(req, search_id, -1, "fail", [], st.settings)
    finally:
        _guard.release(_SLOT)


async def _notify(req: ComposeRequest, search_id: str, code: int, result: str,
                  rows: list[dict], settings) -> None:
    """callback_url 이 있을 때만 결과를 통보한다 (없으면 아무 것도 하지 않는다)."""
    if not req.callback_url:
        return
    payload = callback.build(req.v_id, req.stream_id, search_id, req.query,
                             code, result, rows)
    await callback.send(req.callback_url, payload, settings.callback_timeout)


def _union(lists) -> str:
    """여러 목록 → 등장 순서를 지킨 합집합 콤마 문자열."""
    out = []
    for items in lists:
        for item in items:
            if item not in out:
                out.append(item)
    return ",".join(out)


def _clip_rows(state: dict) -> list[dict]:
    """최종 클립 → 저장 행. 태그·라벨·이닝은 인벤토리(Scene)에서 scene_seq 로 되찾는다.

    병합 클립은 담은 구간 전부의 태그·라벨을 합치고, 흡수한 구간 번호를 merge_seqs 에 남긴다.
    """
    by_no = {}
    for scene in state.get("scenes") or []:
        by_no[scene.scene_seq] = scene

    rows = []
    for clip in state.get("clips") or []:
        seqs = clip.get("scene_seqs") or [clip["scene_seq"]]
        scenes = []
        for seq in seqs:
            scene = by_no.get(seq)
            if scene is not None:
                scenes.append(scene)
        absorbed = []
        for seq in seqs[1:]:
            scene = by_no.get(seq)
            if scene is not None:
                absorbed.append(str(scene.scene_stream_seq))
        rows.append({
            "stream_id": clip["stream_id"],
            "scene_stream_seq": clip["scene_stream_seq"],
            "scene_seq": clip["scene_seq"],
            "merge_seqs": ",".join(absorbed) or None,
            "start": clip["start"],
            "end": clip["end"],
            "start_whole": clip["start_whole"],
            "end_whole": clip["end_whole"],
            "tags": _union(s.tags for s in scenes),
            "labels": _union(s.labels for s in scenes),
            "inning": scenes[0].inning if scenes else "",
        })
    return rows


@router.get("/compose")
async def get_compose(v_id: int, search_id: str, request: Request) -> dict:
    """편성 조회 — 본문은 통보(callback)와 같은 모양이다.

    상태는 t_compose.status_code 하나로 판정한다 — 인메모리 잡을 보지 않으므로
    재기동해도 응답 모양이 바뀌지 않는다. 진행 중이면 scenes 가 빈 목록이다.
    """
    repo = ComposeRepo(request.app.state.db)
    comp_id = await repo.find_comp_id(v_id, search_id)
    row = await repo.fetch(v_id, comp_id) if comp_id is not None else None
    if row is None:
        return {"v_id": v_id, "search_id": search_id, "query": "", "result":
                "편성을 찾을 수 없습니다.", "code": -1, "scenes": []}

    code, result = result_of(row["status_code"])
    clips = [{"stream_id": c["stream_id"], "start": c["start_stream_sec"],
              "end": c["end_stream_sec"], "inning": c["inning"]}
             for c in row["clips"]]
    return callback.build(v_id, row["stream_id"], search_id, row["query"],
                          code, result, clips)


@router.get("/inter-compose")
async def get_inter_compose(v_id: int, comp_id: int, request: Request) -> dict:
    """편성 조회(테스트·디버깅용) — 인메모리 잡(진행·노드별 상세) 우선, 없으면 저장분.

    연동 규격이 아니다 — 그래프 중간 산출(spec·evidence·picked 등)을 그대로 노출한다.
    """
    job = _jobs.get((v_id, comp_id))
    if job is not None:
        return job
    row = await ComposeRepo(request.app.state.db).fetch(v_id, comp_id)
    if row is None:
        raise ComposeNotFoundError(v_id=v_id, comp_id=comp_id)
    return row
