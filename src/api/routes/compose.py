"""편성(compose) 라우트 — 연동 규격 API. 접수 가부·조회 결과는 본문 code(0/-1)로 알린다.

백그라운드 본체는 pipeline.compose_run — 여기는 접수 검사·잡/가드 관리·응답 조립만.
"""

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

from api.errors import ComposeNotFoundError
from domains.baseball import callback
from jobs import JobStore, RunningGuard
from log import get_logger
from pipeline import compose_run, dispatch
from rdb.composes import ComposeRepo, result_of
from rdb.videos import VideoRepo

log = get_logger(__name__)
router = APIRouter(tags=["compose"])

_jobs = JobStore()      # (v_id, comp_id) → 진행·결과 — /inter-compose 전용
_SLOT = "compose"       # 편성은 한 번에 한 건
_guard = RunningGuard("compose")


class ComposeRequest(BaseModel):
    v_id: int
    query: str
    stream_id: str | None = None   # 주면 그 청크만, 없으면 영상 전체
    budget_sec: int | None = None
    callback_url: str | None = None


class ComposeAccepted(BaseModel):
    v_id: int
    stream_id: str | None = None
    search_id: str | None = None
    code: int
    result: str


@router.post("/compose", response_model=ComposeAccepted, response_model_exclude_none=True)
async def post_compose(
    req: ComposeRequest, request: Request, background: BackgroundTasks
) -> ComposeAccepted:
    """편성 접수 — 완료는 GET /compose?v_id=&search_id= 로 폴링."""
    videos = VideoRepo(request.app.state.db)
    video = await videos.get(req.v_id)
    if video is None:
        return ComposeAccepted(v_id=req.v_id, code=-1, result="영상이 없습니다.")

    if req.stream_id is not None and not await videos.has_stream(req.v_id, req.stream_id):
        return ComposeAccepted(
            v_id=req.v_id, stream_id=req.stream_id, code=-1,
            result="해당 스트림이 없습니다.")

    flow = dispatch.resolve(video.cate_id)
    if flow is None:
        return ComposeAccepted(
            v_id=req.v_id, code=-1,
            result=f"지원하지 않는 카테고리입니다. (cate_id={video.cate_id})")

    if not _guard.try_acquire(_SLOT):
        return ComposeAccepted(
            v_id=req.v_id, code=-1,
            result="편성이 진행 중입니다. 잠시 후 다시 요청해 주세요.")

    try:
        comp_id, search_id = await ComposeRepo(request.app.state.db).create(
            req.v_id, req.query, req.budget_sec, req.stream_id, req.callback_url)
    except Exception:
        _guard.release(_SLOT)
        raise

    _jobs.create(
        (req.v_id, comp_id),
        v_id=req.v_id, comp_id=comp_id, query=req.query, progress=[])

    order = compose_run.Order(**req.model_dump())
    background.add_task(_run, request, comp_id, search_id, flow, order)
    log.info(
        "편성 접수: v_id=%s comp_id=%s search_id=%s cate_id=%s(%s) %r",
        req.v_id, comp_id, search_id, video.cate_id, flow.__module__, req.query)

    return ComposeAccepted(
        v_id=req.v_id, stream_id=req.stream_id, search_id=search_id,
        code=0, result="OK")


async def _run(request: Request, comp_id: int, search_id: str, flow,
               order: compose_run.Order) -> None:
    try:
        await compose_run.execute(
            request.app.state, _jobs, comp_id, search_id, flow, order)
    finally:
        _guard.release(_SLOT)


@router.get("/compose")
async def get_compose(v_id: int, search_id: str, request: Request) -> dict:
    """편성 조회 — 본문은 통보(callback)와 같은 모양. 진행 중이면 scenes 가 빈 목록."""
    repo = ComposeRepo(request.app.state.db)
    comp_id = await repo.find_comp_id(v_id, search_id)
    row = await repo.fetch(v_id, comp_id) if comp_id is not None else None
    if row is None:
        return {"v_id": v_id, "search_id": search_id, "query": "", "result":
                "편성을 찾을 수 없습니다.", "code": -1, "scenes": []}

    code, result = result_of(row["status_code"])
    clips = [{
        "stream_id": c["stream_id"], "start": c["start_stream_sec"],
        "end": c["end_stream_sec"], "inning": c["inning"]} for c in row["clips"]]

    return callback.build(v_id, search_id, row["query"], code, result, clips)


@router.get("/inter-compose")
async def get_inter_compose(v_id: int, comp_id: int, request: Request) -> dict:
    """편성 조회(디버깅용) — 인메모리 잡 우선, 없으면 저장분. 연동 규격 아님."""
    job = _jobs.get((v_id, comp_id))
    if job is not None:
        return job
    row = await ComposeRepo(request.app.state.db).fetch(v_id, comp_id)
    if row is None:
        raise ComposeNotFoundError(v_id=v_id, comp_id=comp_id)
    return row
