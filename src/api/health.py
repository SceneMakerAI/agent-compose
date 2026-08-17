"""헬스 체크 라우터.

라이브니스: 프로세스 생존. 레디니스: 의존 자원(DB + embed vLLM + Milvus)까지
받을 준비가 됐는가. bench4 의 fail-open 폐기 — 의존 자원 상태를 여기서 드러낸다.
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
        레디니스 — DB + embed vLLM + Milvus 3종 프로브. 하나라도 미준비면 503.
    Returns:
        dict: {"status", "db", "embed", "milvus"}.
    """
    st = request.app.state
    db_ok, embed_ok, milvus_ok = await asyncio.gather(
        st.db.ping(), st.embedder.ready(), st.vector.ready())
    ready = db_ok and embed_ok and milvus_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not ready",
        "db": "ok" if db_ok else "fail",
        "embed": "ok" if embed_ok else "fail",
        "milvus": "ok" if milvus_ok else "fail",
    }
