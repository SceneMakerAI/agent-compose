"""헬스 체크 라우터.

라이브니스: 프로세스 생존. 레디니스: 의존 자원(DB·Milvus·embed)까지 받을 준비가 됐는가.
LLM·render 는 레디니스에서 제외한다 — GPU 야간 자동 중지로 서비스 전체가 not-ready 로
뒤집히는 오탐 방지. fail-open 금지 원칙: 접속 불가는 여기서 503 으로 드러낸다.
"""

import asyncio

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """라이브니스 — 프로세스 생존만 확인(의존 자원 미검사)."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request, response: Response):
    """
    Summary:
        레디니스 — 의존 자원 프로브. 하나라도 미준비면 503.
    Returns:
        dict: {"status", "db", "milvus", "embed"}.
            LLM 은 readyz 에 넣지 않는다 — 부팅 로그 프로브만 (편성 시 폴백 경로 존재).
    """
    st = request.app.state
    db_ok, milvus_ok, embed_ok = await asyncio.gather(
        st.db.ping(), st.vector.ready(), st.embedder.ready())
    ready = db_ok and milvus_ok and embed_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not ready",
        "db": "ok" if db_ok else "fail",
        "milvus": "ok" if milvus_ok else "fail",
        "embed": "ok" if embed_ok else "fail",
    }
