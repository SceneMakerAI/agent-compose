"""worker-render 호출 클라이언트 — 렌더 접수·상태 조회 1식 (lifespan 공유 전제).

단독 렌더는 sync_yn=False 로 접수만 하고 status() 로 폴링한다. 원샷(compose
render=true)만 sync_yn=True 로 완주까지 기다리므로 타임아웃은 LLM 보다 길게
(설정 render_timeout).
readyz 프로브에는 넣지 않는다 — GPU 야간 자동 중지 시간대마다 서비스 전체가
not-ready 로 뒤집히는 오탐을 피한다 (렌더는 편성의 하류 옵션 기능).
"""

import httpx

from config import Settings
from log import get_logger

log = get_logger(__name__)


class RenderClient:
    """
    Summary:
        worker-render 호출 객체 — POST /render/sports/baseball (동기 완주).
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 엔드포인트·타임아웃을 받는다 (테스트 기간: GA 경유)."""
        self._client = httpx.AsyncClient(
            base_url=settings.render_base_url, timeout=settings.render_timeout)

    async def render(self, payload: dict) -> dict:
        """
        Summary:
            렌더 요청 1건 — 완주(또는 실패)까지 대기 후 워커 응답을 그대로 돌려준다.
        Args:
            payload (dict): render.payload.build_request 산출 본문.
        Returns:
            dict: {status, output_path, error} (worker-render 응답).
        Raises:
            httpx.HTTPError: 접속 불가·타임아웃·4xx/5xx — 호출부가 502 로 변환.
        """
        resp = await self._client.post("/render/sports/baseball", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def status(self, v_id: int, c_id: int) -> dict:
        """
        Summary:
            렌더 1건의 현재 상태 조회 — GET /render/{v_id}/{c_id}.
        Returns:
            dict: {status, output_path, error}. status 는
                accepted(큐 대기) / running / done / error.
        Raises:
            httpx.HTTPError: 접속 불가·타임아웃·4xx/5xx.
        """
        resp = await self._client.get(f"/render/{v_id}/{c_id}", timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await self._client.aclose()
