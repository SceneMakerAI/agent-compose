"""agent-compose API 서버 애플리케이션.

앱 조립 지점: lifespan(공유 자원 수명) + 라우터 등록. 편성·색인 로직은 여기 두지 않는다.
공유 자원 원칙: 클라이언트(DB 풀·Embedder·VectorStore 등)는 lifespan 1식 공유 —
호출마다 생성 금지.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors import ApiError
from api.router import api_router
from config import Settings, get_settings
from infer.chat import ChatLLM
from infer.embedder import Embedder
from log import get_logger, setup_logging
from rdb.pool import Database
from render.client import RenderClient
from vector.client import VectorClient

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Summary:
        FastAPI 수명 관리 — 부팅 시 설정·로깅 준비, 종료 시 정리.
    Description:
        - 공유 자원은 앱 수명 동안 재사용(요청마다 만들지 않음). app.state 에 보관.
        - DB 는 필수 자원 — 부팅 시 접속·ping 실패면 즉시 종료한다(fail-fast).
          DB 없이 받을 수 있는 요청이 없어 반쯤 뜬 서버를 허용하지 않는다.
    """
    settings: Settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_path)
    log.info("== startup AGENT COMPOSE ==")
    log.debug("Loaded settings: %s", settings)

    app.state.settings = settings
    app.state.db = await Database.connect(settings)

    if not await app.state.db.ping():
        await app.state.db.close()
        raise RuntimeError("DB 접속 실패 — 부팅을 중단한다 (접속 정보·네트워크 확인)")
    log.info("접속 테스트 — DB: OK")

    # Milvus·LLM 은 원격 자원 — 다운이어도 부팅은 계속한다 (수용 여부는 /readyz 가 판정)
    app.state.vector = VectorClient(settings)
    log.info("접속 테스트 — Milvus: %s", "OK" if await app.state.vector.ready() else "FAIL")
    app.state.llm = ChatLLM(settings)
    log.info("접속 테스트 — LLM: %s", "OK" if await app.state.llm.ready() else "FAIL")
    app.state.embedder = Embedder(settings)
    log.info("접속 테스트 — embed: %s", "OK" if await app.state.embedder.ready() else "FAIL")
    app.state.render = RenderClient(settings)
    log.info("AGENT COMPOSE 준비 완료.")

    try:
        yield
    finally:
        await app.state.render.aclose()
        await app.state.embedder.aclose()
        await app.state.llm.aclose()
        await app.state.vector.aclose()
        await app.state.db.close()
        log.info("== shutdown AGENT COMPOSE ==")


app = FastAPI(title="Agent Compose", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """API 오류 예외를 계약 응답({detail: {code, message, ...}})으로 변환한다."""
    log.info("API 오류: %s %s → %s %s", request.method, request.url.path, exc.code, exc.ctx)
    return JSONResponse(status_code=exc.http_status, content={"detail": exc.detail()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """미처리 예외를 일관된 JSON 500 으로 변환(스택트레이스는 로그만, 응답 미노출)."""
    log.exception("처리되지 않은 오류: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "INTERNAL_ERROR", "message": "서버 내부 오류가 발생했습니다."}},
    )


# 집계 라우터 — 라우터 추가/변경은 api/router.py 에서.
app.include_router(api_router)
