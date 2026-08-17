"""라우터 집계 — 라우터 추가/변경은 여기서만."""

from fastapi import APIRouter

from api import compose, health, ingest

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
api_router.include_router(compose.router, prefix="/api/v1", tags=["compose"])
