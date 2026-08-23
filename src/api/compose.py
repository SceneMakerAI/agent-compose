"""편성(compose) 라우트 — 질의 1건 → LangGraph flow → t_compose 저장 + 결과.

select_clips(thinking)이 1~3분이라 202 + job 폴링 패턴 (단일 워커 전제 — ingest 와 동일).
같은 v_id 동시 편성은 허용 — 읽기 전용 + comp_id 신규 발급이라 충돌이 없다.
render=True 원샷: 편성 저장 후 ok 면 worker-render 까지 이어 호출 (기본 False —
렌더 실패는 잡의 render 필드로만 드러나고 편성 성공을 뒤집지 않는다).
"""

import asyncio
import uuid
from trace import Trace

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from db.repos import SourceRepo
from db.status_repo import (
    COMPOSE_CUT,
    COMPOSE_EMPTY,
    COMPOSE_ERROR,
    COMPOSE_ERROR_RENDER,
    COMPOSE_ERROR_SOURCE,
    COMPOSE_ERROR_STAMP,
    COMPOSE_OK,
    COMPOSE_PLAN,
    COMPOSE_RENDER,
    COMPOSE_VERIFY,
)
from flow import plan, players
from flow.graph import run_compose
from flow.state import Inventory
from log import bind_v_id, get_logger
from render.payload import build_request

log = get_logger(__name__)
router = APIRouter()

_JOBS: dict[str, dict] = {}     # job_id → {status, v_id, query, result?, error?}
_JOBS_MAX = 200                 # 오래된 완료 잡 정리 상한 (프로세스 수명 캐시)

# 그래프 노드 → t_video 상태 코드 (UI 진행 표시 — 코드가 바뀌는 노드에서만 기록).
# **국면 순서대로 단조 증가해야 한다** — 화면 진행이 뒤로 가면 안 된다.
_NODE_CODE = {"rephrase_query": COMPOSE_PLAN, "retrieve_evidence": COMPOSE_PLAN,
              "select_clips": COMPOSE_PLAN, "retry_select": COMPOSE_PLAN,
              "refine_end_bound": COMPOSE_CUT, "refine_start_bound": COMPOSE_CUT,
              "finish": COMPOSE_VERIFY}


class ComposeRequest(BaseModel):
    """편성 요청."""

    v_id: int
    query: str
    # 목표 분량(초). 없으면 절단하지 않는다 — 선곡이 곧 편성이다.
    # 94b58dc 에서 없앴던 인자를 되살린다. 그때 뺀 이유는 예산이 필요 없어서가 아니라
    # fill_budget 이 예산을 채우려고 선곡에 없던 장면을 끌어왔기 때문이다. 이번 예산은
    # finish 에서 **덜어내기만** 한다 — 그 통로는 다시 열지 않는다.
    budget_sec: int | None = None
    render: bool = False         # 원샷 옵션 — 편성 ok 면 이어서 mp4 렌더까지 (동기)
    bumper: bool = True          # render=True 일 때만 — 이닝 그룹 사이 범퍼


@router.post("/compose", status_code=202)
async def post_compose(req: ComposeRequest, request: Request, background: BackgroundTasks) -> dict:
    """편성 접수 — 202 {job_id} 반환, 완료는 GET /compose/{job_id} 로 폴링."""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "running", "v_id": req.v_id, "query": req.query, "progress": []}
    if len(_JOBS) > _JOBS_MAX:
        for k in [k for k, j in _JOBS.items() if j["status"] != "running"][:50]:
            _JOBS.pop(k, None)
    background.add_task(_run, request, job_id, req)
    
    return {"job_id": job_id, "status": "running"}


@router.get("/compose/{job_id}")
async def get_compose_job(job_id: str) -> dict:
    """잡 상태·결과 조회."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "job_id": job_id})
    return job


@router.get("/compose")
async def get_compose(request: Request, comp_id: int) -> dict:
    """저장된 편성 재조회 (t_compose·t_compose_clip)."""
    row = await request.app.state.compose_repo.fetch(comp_id)
    if not row:
        raise HTTPException(404, detail={"code": "COMPOSE_NOT_FOUND", "comp_id": comp_id})
    return row


async def _run(request: Request, job_id: str, req: ComposeRequest) -> None:
    """백그라운드 본체 — 인벤토리 로드 → 그래프 실행 → 저장 → 잡 갱신."""
    st = request.app.state
    try:
        with bind_v_id(req.v_id):
            result = await _compose_once(st, req, _JOBS[job_id]["progress"])
        _JOBS[job_id] = {"status": result["status"], "v_id": req.v_id, "query": req.query, **result}
        
    except asyncio.CancelledError:
        raise
    
    except Exception as e:
        log.exception("compose 실패: v_id=%s %r", req.v_id, req.query)
        code = COMPOSE_ERROR_SOURCE if isinstance(e, ValueError) else COMPOSE_ERROR
        await st.status.set(req.v_id, code, f"{type(e).__name__}: {e}")
        _JOBS[job_id] = {
            "status": "error",
            "v_id": req.v_id,
            "query": req.query,
            "error": f"{type(e).__name__}: {e}"
        }


async def _compose_once(st, req: ComposeRequest, progress: list[str]) -> dict:
    """인벤토리 스냅샷 → run_compose → 클립 요약 + t_compose 저장."""
    repo: SourceRepo = st.repo
    scenes = await repo.fetch_scenes(req.v_id)

    if not scenes:
        raise ValueError(
            f"t_scene_baseball 이 비어 있음 — vision3 scene 선행 필요 (v_id={req.v_id})")

    segs = [{"seg_id": i, **r} for i, r in enumerate(await repo.fetch_shots_all(req.v_id), 1)]
    utts = tuple(await repo.fetch_utterances(req.v_id))
    # 타자 이름은 하단 자막에서 — 선수 질의를 벡터 검색이 아니라 인벤토리로 풀기 위한 재료
    players.annotate_batters(scenes, await repo.fetch_etc_rows(req.v_id))
    # 팀명은 전광판 자막이 유일한 출처다 (발행본의 score 컬럼 폐기 — repos.fetch_teams).
    # 못 읽으면 팀 없이 간다 — 관점("초=원정") 규칙은 이닝만으로도 선다.
    teams = await repo.fetch_teams(req.v_id)
    game_line = (f"v_id={req.v_id}  {teams[0]}(원정) vs {teams[1]}(홈)" if teams
                 else f"v_id={req.v_id}  (팀명 판독 없음)")
    inv = Inventory(
        v_id=req.v_id, scenes=tuple(scenes), segs=tuple(segs), utts=utts,
        game_line=game_line,
        pitches=tuple(await repo.fetch_pitch_windows(req.v_id)),  # refine_bounds 후보 재료
    )
    tr = Trace(st.settings.trace_dir, req.v_id, req.query)

    await st.status.set(req.v_id, COMPOSE_PLAN)
    last_code = COMPOSE_PLAN

    async def on_node(node: str) -> None:
        nonlocal last_code
        progress.append(node)
        code = _NODE_CODE.get(node)
        if code and code != last_code:
            last_code = code
            await st.status.set(req.v_id, code)

    state = await run_compose(st.graph, inv, req.query, req.budget_sec,
                              on_node=on_node, trace=tr)

    clips = [_clip_row(r) for r in state.get("picked", [])]
    comp_id = await st.compose_repo.save(
        req.v_id, req.query, state.get("spec"), state["status"], clips)
    tr.finish(comp_id, state["status"], clips=len(clips), total=state.get("total"))

    render_result = None
    stamp_error = None
    if req.render:
        progress.append("render")
        render_result = await _render_after(st, req, comp_id, state["status"], clips)
        if render_result.get("status") not in ("error", "skipped"):
            # 렌더 성공만 t_compose 에 스탬프 — 뷰어의 중복 렌더 차단 근거.
            # 기록 실패를 조용히 넘기면 중복 렌더를 부르므로 4960 으로 드러낸다.
            try:
                await st.compose_repo.mark_rendered(comp_id)
            except Exception as e:      # 실패 사실을 상태 코드로 남기고 계속
                log.exception("렌더 완료 기록 실패: comp_id=%s", comp_id)
                stamp_error = f"{type(e).__name__}: {e}"
    # 최종 상태 코드 — 렌더 실패 > 기록 실패 > empty > 완료 순으로 판정
    if render_result and render_result.get("status") == "error":
        await st.status.set(req.v_id, COMPOSE_ERROR_RENDER, render_result.get("error"))
    elif stamp_error:
        await st.status.set(req.v_id, COMPOSE_ERROR_STAMP, stamp_error)
    elif state["status"] == "empty":
        await st.status.set(req.v_id, COMPOSE_EMPTY)
    else:
        await st.status.set(req.v_id, COMPOSE_OK)
    return {
        **({"render": render_result} if render_result else {}),
        "comp_id": comp_id,
        "spec": {k: v for k, v in (state.get("spec") or {}).items() if k != "raw"},
        "clips": clips,
        "duration": round(sum(c["end"] - c["start"] for c in clips), 1),
        "end_moved": state.get("end_moved", []),
        "start_moved": state.get("start_moved", []),
        "orphans": [{"s": o["s"], "text": o["text"][:80]}
                    for o in state.get("evidence_orphan", [])],
        "status": state["status"],
    }


async def _render_after(st, req: ComposeRequest, comp_id: int, status: str,
                        clips: list[dict]) -> dict:
    """원샷 렌더 — 편성이 ok 일 때만 worker-render 동기 호출(sync_yn=True).

    잡이 이미 백그라운드라 여기서는 완주까지 기다린다 — 잡 하나로 편성·렌더가 함께
    끝나는 게 원샷의 취지 (단독 POST /render 는 비동기 접수 + 폴러 감시).
    렌더 실패가 편성 성공을 뒤집지 않는다 — 편성은 이미 저장됐으므로 잡의
    render 필드에 사유만 남긴다 (empty·이닝 결손은 호출 전 생략 = 사전 차단).
    """
    if status != "ok" or not clips:
        return {"status": "skipped", "reason": f"편성 status={status} 클립 {len(clips)}건 — 렌더 생략"}
    try:
        payload = build_request(req.v_id, comp_id, clips, req.bumper, sync=True)
    except ValueError as e:
        log.warning("렌더 생략(comp_id=%s): %s", comp_id, e)
        return {"status": "skipped", "reason": str(e)}
    try:
        await st.status.set(req.v_id, COMPOSE_RENDER)
        await st.compose_repo.mark_render_started(comp_id, req.bumper)  # 실제 사용한 범퍼 값
        result = await st.render.render(payload)
    except httpx.HTTPError as e:
        log.error("원샷 렌더 실패: comp_id=%s %s: %s", comp_id, type(e).__name__, e)
        await st.compose_repo.mark_render_failed(comp_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    log.info("원샷 렌더 완료: comp_id=%s %s", comp_id, result)
    return result


def _clip_row(r: dict) -> dict:
    """채택 행 → 저장·응답용 클립 요약 (초 — **소수 유지**).

    int() 로 내리지 않는다. t_segment 경계가 소수라 4369.3s('투구' 샷 시작)를 내리면
    4369s 가 되어 직전 '리액션' 샷 꼬리에서 시작한다 — 파이프라인이 투구에 맞춰 둔
    시작을 저장 직전에 되돌리는 셈이었다 (v201 comp24 실측: 16클립 중 '투구' 시작 0건).
    컬럼은 time(3) 이고 SEC_TO_TIME 이 밀리초를 그대로 받는다.
    """
    return {
        "scene_id": r["scene_id"],
        # h_id 는 더 이상 채우지 않는다 — 상류 원장 키가 h_id→p_id 로 바뀌었고
        # p_id 는 scene_id 와 같은 값이라 별도 칸에 담을 것이 없다. 컬럼은 하류
        # 소비자를 위해 남겨 둔다 (nullable).
        "h_id": None,
        "start": r["cut"]["cs"], "end": r["cut"]["ce"],
        # 판세(game_context)를 라벨 끝에 잇는다 — 구 스키마에선 '역전'·'동점'이
        # t_scene.labels 안에 있어 이 칸으로 그대로 흘렀다. 축이 갈린 뒤 안 이으면
        # 편성 결과에서 가장 신호가 센 표기가 통째로 사라진다. 이 칸은 표시용이라
        # (compose_repo.fetch 가 원문 그대로 돌려주고 코드가 다시 파싱하지 않는다)
        # 두 축을 한 문자열에 담아도 되는 유일한 자리다.
        "label": ",".join(r["tags"]),
        "labels": ",".join(r["label_list"] + ([r["game_context"]]
                                              if r.get("game_context") else [])),
        "inning": r["inning"] or "",
        "score_before": r["score_before"] or "?", "score_after": plan.score_after(r),
        "cut_mode": r["cut"]["mode"],
    }
