"""agent-compose API 서버 애플리케이션.

앱 조립 지점: lifespan(공유 자원 수명) + 라우터 등록. 편성 로직은 flow/,
색인 로직은 vector/ 소유. 자원: DB 풀 + Embedder + VectorStore
(bench4 는 호출마다 클라이언트를 만들었다 — 서비스는 lifespan 1식 공유).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.router import api_router
from config import Settings, get_settings
from db.compose_repo import ComposeRepo
from db.pool import Database
from db.repos import SourceRepo
from db.status_repo import StatusRepo
from flow import vocab
from flow.graph import build_graph
from flow.llm import ChatLLM
from log import get_logger, setup_logging
from render.client import RenderClient
from vector.embedder import Embedder
from vector.store import VectorStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Summary:
        FastAPI 수명 관리 — 부팅 시 설정·로깅·DB·임베딩·Milvus 준비, 종료 시 정리.
    Description:
        - 부팅 시 3종 프로브를 로그로 남긴다(진단 목적 — 미응답이어도 부팅은 계속,
          수용 여부는 /readyz 가 판정).
    """
    settings: Settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_path)
    log.info("== startup AGENT COMPOSE ==")
    log.debug("Loaded settings: %s", settings)

    vocab.validate()    # 어휘 드리프트는 부팅 실패로 (레시피 키 ↔ 태그 어휘)

    app.state.settings = settings
    app.state.db = await Database.connect(settings)
    app.state.repo = SourceRepo(app.state.db)
    app.state.compose_repo = ComposeRepo(app.state.db)
    app.state.status = StatusRepo(app.state.db)     # t_video.status_code 4000번대 기록
    app.state.embedder = Embedder(settings)
    app.state.vector = VectorStore(settings)
    app.state.llm = ChatLLM(settings)
    app.state.render = RenderClient(settings)   # readyz 미포함 — GPU 야간 중지 오탐 회피
    # 그래프는 무상태 배선 — 프로세스당 1회 컴파일 (요청마다 재컴파일 금지)
    app.state.graph = build_graph(app.state.llm, app.state.embedder, app.state.vector)

    db_ok = await app.state.db.ping()
    embed_ok = await app.state.embedder.ready()
    milvus_ok = await app.state.vector.ready()
    llm_ok = await app.state.llm.ready()
    log.info("접속 테스트 — DB: %s / embed: %s / Milvus: %s / LLM: %s", db_ok, embed_ok, milvus_ok, llm_ok)
    log.info("AGENT COMPOSE 준비 완료.")

    try:
        yield
    finally:
        await app.state.render.aclose()
        await app.state.llm.aclose()
        await app.state.embedder.aclose()
        await app.state.vector.aclose()
        await app.state.db.close()
        log.info("== shutdown AGENT COMPOSE ==")


app = FastAPI(title="Agent Compose", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    """미처리 예외를 일관된 JSON 500 으로 변환(스택트레이스는 로그만, 응답 미노출)."""
    log.exception("처리되지 않은 오류: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "INTERNAL_ERROR", 
                "message": "서버 내부 오류가 발생했습니다."
            }
        },
    )


app.include_router(api_router)
