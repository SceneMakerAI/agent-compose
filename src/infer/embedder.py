"""임베딩 클라이언트 — openai 호환 /v1/embeddings (vLLM Qwen3-Embedding).

역할은 전송뿐이다 — instruction 문구(도메인 언어)는 domains/* 가 갖는다.
색인(agent-vision)과 **같은 서버·모델**을 바라봐야 한다 — 다르면 벡터 공간이 어긋나
검색이 조용히 무너진다 (config 주석 참조).
lifespan 1식 공유 전제 (호출마다 생성 금지).
"""

import httpx

from config import Settings
from log import get_logger

log = get_logger(__name__)


class Embedder:
    """
    Summary:
        임베딩 호출 객체 — 질의 단건 임베딩 (검색 전용 — 색인은 agent-vision 소유).
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 엔드포인트·모델을 받는다 (lifespan 공유 전제)."""
        self._client = httpx.AsyncClient(
            base_url=settings.embed_base_url, timeout=settings.embed_timeout)
        self._model = settings.embed_model
        self._base_url = settings.embed_base_url    # 오류 로그용 — 어느 서버인지 드러낸다

    async def embed_query(self, instruct: str, query: str) -> list[float]:
        """
        Summary:
            질의 임베딩 — instruction 프리픽스 부착 (Qwen3-Embedding 비대칭 검색 권장:
            질의에만 instruct, 문서는 원문 그대로).
        Args:
            instruct (str): instruction 프리픽스 (도메인 소유 문구).
            query (str): 검색어.
        Returns:
            list[float]: 임베딩 벡터.
        Raises:
            httpx.HTTPError: 접속 불가·타임아웃·4xx/5xx — 폴백 판단은 호출자 몫.
        """
        try:
            resp = await self._client.post(
                "/embeddings", json={"model": self._model, "input": [instruct + query]})
        except httpx.HTTPError as e:
            # ReadError 처럼 메시지가 빈 전송 오류도 어느 서버인지 바로 보이게
            log.warning("임베딩 호출 실패: %s — %s(%s)",
                        self._base_url, type(e).__name__, e)
            raise
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def ready(self) -> bool:
        """모델 서빙 여부 (readyz 프로브) — 짧은 타임아웃, 예외는 False."""
        try:
            resp = await self._client.get("/models", timeout=3.0)
            return resp.is_success
        except httpx.HTTPError as e:
            log.warning("embed 프로브 실패: %s — %s(%s)", self._base_url, type(e).__name__, e)
            return False

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await self._client.aclose()
