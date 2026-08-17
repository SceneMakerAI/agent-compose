"""임베딩 클라이언트 — openai 호환 /v1/embeddings (vLLM Qwen3-Embedding).

httpx 비동기 직행 (openai SDK 는 동기/스레드 오버헤드만 더한다 — 엔드포인트 1개,
호출 형태 1가지). 배치 크기는 서버 max_num_seqs 소형 설정에 맞춰 보수적으로.
"""

import httpx

from config import Settings
from log import get_logger

log = get_logger(__name__)


class Embedder:
    """
    Summary:
        임베딩 호출 객체 — 문서 배치 임베딩 + 질의 단건 임베딩.
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 엔드포인트·모델·배치 크기를 받는다 (lifespan 공유 전제)."""
        self._client = httpx.AsyncClient(
            base_url=settings.embed_base_url,
            timeout=settings.embed_timeout,
        )
        self._model = settings.embed_model
        self._batch = settings.embed_batch

    async def embed_docs(self, texts: list[str]) -> list[list[float]]:
        """
        Summary:
            문서 임베딩 (원문 그대로 — instruction 없음, Qwen3-Embedding 권장).
        Args:
            texts (list[str]): 색인할 텍스트들.
        Returns:
            list[list[float]]: 입력 순서 유지 벡터.
        """
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            chunk = texts[i:i + self._batch]
            resp = await self._client.post(
                "/embeddings", json={"model": self._model, "input": chunk})
            resp.raise_for_status()
            data = resp.json()["data"]
            vecs += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        return vecs

    async def embed_query(self, instruct: str, query: str) -> list[float]:
        """질의 임베딩 — instruction 프리픽스 부착 (문서와 비대칭)."""
        resp = await self._client.post(
            "/embeddings", json={"model": self._model, "input": [instruct + query]})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def ready(self) -> bool:
        """모델 서빙 여부 (readyz 프로브) — 짧은 타임아웃, 예외는 False."""
        try:
            resp = await self._client.get("/models", timeout=3.0)
            return resp.is_success
        except httpx.HTTPError as e:
            log.warning("embed 프로브 실패: %s", e)
            return False

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await self._client.aclose()
