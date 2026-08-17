"""색인(ingest) 라우트 — vision3 발행 직후 호출되는 진입점.

POST 는 202 접수 후 백그라운드 색인 (임베딩 수십 초 — 호출자를 잡아두지 않는다).
같은 v_id 동시 요청은 409 — 색인이 delete-insert 라 겹치면 서로를 지운다.
GET 은 색인 현황 조회 (운영 확인용).
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from log import bind_v_id, get_logger
from vector.ingest import ingest

log = get_logger(__name__)
router = APIRouter()

_RUNNING: set[int] = set()      # 프로세스 내 중복 방지 (단일 워커 전제 — vision3 과 동일)


class IngestRequest(BaseModel):
    """색인 요청 — 대상 v_id 하나."""

    v_id: int


class IngestAccepted(BaseModel):
    """접수 응답."""

    v_id: int
    status: str = "accepted"


@router.post("/ingest", status_code=202, response_model=IngestAccepted)
async def post_ingest(req: IngestRequest, request: Request,
                      background: BackgroundTasks) -> IngestAccepted:
    """색인 접수 — 202 반환 후 백그라운드에서 수집→임베딩→교체."""
    if req.v_id in _RUNNING:
        raise HTTPException(409, detail={
            "code": "ALREADY_RUNNING",
            "message": f"v_id={req.v_id} 색인이 이미 진행 중입니다."})
    _RUNNING.add(req.v_id)
    background.add_task(_run, request, req.v_id)
    return IngestAccepted(v_id=req.v_id)


async def _run(request: Request, v_id: int) -> None:
    """백그라운드 본체 — 실패는 로그로 (조용히 삼키지 않고 원인 기록)."""
    try:
        with bind_v_id(v_id):
            await ingest(v_id, request.app.state.repo, request.app.state.embedder, request.app.state.vector)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("ingest 실패: v_id=%s", v_id)
    finally:
        _RUNNING.discard(v_id)


@router.get("/ingest")
async def get_ingest(request: Request, v_id: int | None = None) -> dict:
    """색인 현황 — 전체(또는 v_id) 행 수 + 진행 중 목록."""
    stats = await request.app.state.vector.stats(v_id)
    return {**stats, "running": sorted(_RUNNING)}
