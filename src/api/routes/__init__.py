"""API 집계 라우터 — 하위 라우터를 한곳에 모은다.

app.py 는 이 api_router 하나만 등록한다. 새 API = 이 패키지에 모듈 + 아래 include_router 한 줄.
"""

from fastapi import APIRouter

from api.routes import compose, health

api_router = APIRouter()

# 비즈니스 API
api_router.include_router(compose.router, prefix="/api/v1")

# 인프라 프로브
api_router.include_router(health.router)
