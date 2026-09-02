"""Milvus 연결 계층 — 클라이언트 래퍼 (rdb.pool.Database 와 같은 역할).

이 모듈은 도메인을 모른다 — 접속·조회 통로만 내어주고, 어느 컬렉션의 어떤 필드를
어떻게 읽는지는 도메인 repo(domains/*/repo/)가 갖는다 (전송/도메인 분리).
색인(쓰기)은 agent-vision 소유 — 여기는 읽기 통로만 둔다.

pymilvus 는 동기 — 이벤트 루프 블로킹 방지로 to_thread 경유.
클라이언트는 lifespan 1개 공유 (호출마다 생성·미반납 금지).
"""

import asyncio

from pymilvus import MilvusClient

from config import Settings
from log import get_logger

log = get_logger(__name__)


class VectorClient:
    """
    Summary:
        Milvus 클라이언트 래퍼 — 조회 통로·프로브·반납 (도메인 무관).
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 URI·DB 를 받아 클라이언트를 만든다 (lifespan 공유 전제)."""
        self._mc = MilvusClient(settings.milvus_uri)
        self._mc.use_database(settings.milvus_db)

    async def query(self, collection: str, filter: str,
                    output_fields: list[str], limit: int) -> list[dict]:
        """스칼라 필터 조회 — pymilvus query 의 to_thread 통로 (실패는 전파)."""
        return await asyncio.to_thread(
            self._mc.query, collection, filter=filter,
            output_fields=output_fields, limit=limit)

    async def search(self, collection: str, vector: list[float], filter: str,
                     output_fields: list[str], limit: int) -> list[dict]:
        """벡터 유사도 검색 — 상위 limit 히트를 {distance, ...엔티티} 로 돌려준다."""
        def _search() -> list[dict]:
            hits = self._mc.search(
                collection, data=[vector], filter=filter,
                output_fields=output_fields, limit=limit)[0]
            rows = []
            for hit in hits:
                rows.append({"distance": hit["distance"], **hit["entity"]})
            return rows

        return await asyncio.to_thread(_search)

    async def ready(self) -> bool:
        """접속 가능 여부 (readyz 프로브) — 예외는 False."""
        try:
            await asyncio.to_thread(self._mc.list_collections)
            return True
        except Exception as e:                       # noqa: BLE001 — 프로브는 원인 무관 False
            log.warning("Milvus 프로브 실패: %s", e)
            return False

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await asyncio.to_thread(self._mc.close)
