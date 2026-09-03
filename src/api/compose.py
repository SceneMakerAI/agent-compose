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

from fastapi import APIRouter, BackgroundTasks, Request, status
from pydantic import BaseModel

from api.errors import (
    ComposeNotFoundError,
    UnsupportedCategoryError,
    VideoNotFoundError,
)
from api.jobs import JobStore
from log import bind_v_id, get_logger
from pipeline import dispatch
from rdb.composes import ComposeRepo, ComposeStatus
from rdb.videos import VideoRepo

log = get_logger(__name__)
router = APIRouter(tags=["compose"])

_jobs = JobStore()      # (v_id, comp_id) → 상태·결과 (프로세스 수명 캐시)


class ComposeRequest(BaseModel):
    """편성 요청."""

    v_id: int
    query: str
    # 목표 분량(초). 없으면 절단하지 않는다 — 선곡이 곧 편성이다.
    # 예산은 마감 단계의 **덜어내기 전용**이다: 예산을 채우려고 선곡에 없던 장면을
    # 끌어오는 통로는 열지 않는다 (질의를 규칙이 덮어쓰게 된다 — 설계 결정).
    budget_sec: int | None = None


@router.post("/compose", status_code=status.HTTP_202_ACCEPTED)
async def post_compose(req: ComposeRequest, request: Request,
                       background: BackgroundTasks) -> dict:
    """편성 접수 — 202 {comp_id} 반환, 완료는 GET /compose?v_id=&comp_id= 로 폴링.

    접수 시점에 검증해 오류를 4xx 로 돌려준다 — 백그라운드에서 터지면 호출자가 모른다:
    t_video 부재 404 / 미등록 카테고리 422.
    접수와 동시에 t_compose 헤더를 선-INSERT 한다 (comp_id 즉시 발급, status=PLAN) —
    진행 국면이 status_code 로 드러나고, 실패도 ERROR 행으로 남는다.
    """
    video = await VideoRepo(request.app.state.db).get(req.v_id)
    if video is None:
        raise VideoNotFoundError(v_id=req.v_id)

    flow = dispatch.resolve(video.cate_id)
    if flow is None:
        raise UnsupportedCategoryError(v_id=req.v_id, cate_id=video.cate_id)

    comp_id = await ComposeRepo(request.app.state.db).create(
        req.v_id, req.query, req.budget_sec)

    _jobs.create((req.v_id, comp_id),
                 v_id=req.v_id, comp_id=comp_id, query=req.query, progress=[])
    background.add_task(_run, request, comp_id, flow, req)
    log.info("편성 접수: v_id=%s comp_id=%s cate_id=%s(%s) %r",
             req.v_id, comp_id, video.cate_id, flow.__module__, req.query)
    return {"v_id": req.v_id, "comp_id": comp_id, "status": "running"}


# 노드 완료 → 다음 국면 코드. 코드는 노드가 아니라 국면이라 전 노드를 다 적지 않는다
# (PLAN 은 create() 가 접수 시 찍고, 종결 OK/EMPTY 는 finish() 가 찍는다).
_PHASE_AFTER = {
    "select_clips": ComposeStatus.CUT,        # 선곡 끝 → 클립 구간 확정 국면
    "select_end_point": ComposeStatus.VERIFY,  # 끝점 확정 끝 → 검수·절단 국면
}


async def _run(request: Request, comp_id: int, flow,
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
            await repo.finish(req.v_id, comp_id, final, _clip_rows(state))

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


def _clip_rows(state: dict) -> list[dict]:
    """최종 클립 → 저장 행. 태그·라벨·이닝은 인벤토리(Scene)에서 scene_no 로 되찾는다."""
    by_no = {}
    for scene in state.get("scenes") or []:
        by_no[scene.scene_no] = scene

    rows = []
    for clip in state.get("clips") or []:
        scene = by_no.get(clip["scene_no"])
        rows.append({
            "scene_no": clip["scene_no"],
            "start": clip["start"],
            "end": clip["end"],
            "tags": ",".join(scene.tags) if scene else "",
            "labels": ",".join(scene.labels) if scene else "",
            "inning": scene.inning if scene else "",
        })
    return rows


@router.get("/compose")
async def get_compose(v_id: int, comp_id: int, request: Request) -> dict:
    """편성 조회 — 인메모리 잡(진행·결과 상세) 우선, 없으면 저장분(t_compose).

    running 이면 progress(완료 노드 목록)가 진행 표시다. 프로세스 재시작으로 잡이
    유실돼도 저장 행으로 폴백한다 — 그마저 없으면 404.
    comp_id 는 v_id 안에서 1부터 발급되는 시퀀스라 v_id 없이는 특정할 수 없다.
    """
    job = _jobs.get((v_id, comp_id))
    if job is not None:
        return job
    row = await ComposeRepo(request.app.state.db).fetch(v_id, comp_id)
    if row is None:
        raise ComposeNotFoundError(v_id=v_id, comp_id=comp_id)
    return row
