"""Milvus 접근 — 컬렉션 스키마·교체 적재·검색 (bench4 index_evidence/vector 이식).

bench4 와 다른 점 (서비스 승격 결정):
- **fail-open 폐기** — 접속 불가·검색 실패는 예외로 전파하고 /readyz 가 상태를 드러낸다.
  POC 에선 "벡터는 보강 재료"라 조용히 통과했지만, 서비스에선 품질 저하가 무음으로
  숨는 게 더 위험하다.
- 클라이언트는 lifespan 1개 공유 (POC 는 호출마다 생성·미반납 — 커넥션 누수).
- pymilvus 는 동기 — 이벤트 루프 블로킹 방지로 to_thread 경유.
"""

import asyncio

from pymilvus import DataType, MilvusClient

from config import Settings
from log import get_logger

log = get_logger(__name__)

# Qwen3-Embedding-4B 출력 차원 — 모델 교체 시 컬렉션 재생성 필요 (스키마 결합 상수)
EMBED_DIM = 2560

# Qwen3-Embedding 권장: 질의에만 instruction 프리픽스, 문서는 원문 그대로.
# 실측(bench4): "모음 3분" 같은 편성 문구가 섞여도 순위 유지 — 질의 전처리 없이 원문.
QUERY_INSTRUCT = ("Instruct: 야구 중계의 해설 대사와 장면 설명에서 질의와 관련된 "
                  "구간을 찾는다\nQuery: ")


class VectorStore:
    """
    Summary:
        sm_scene_evidence 컬렉션 접근 객체 — 생성·v_id 교체 적재·검색.
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 URI·DB·컬렉션명을 받아 클라이언트를 만든다 (lifespan 공유 전제).

        데이터베이스(sm_db)가 없으면 만든다 — 컬렉션은 팀 관례상 sm_db 아래
        (default 에 만들면 Attu 등에서 못 찾는 혼선 실측 2026-08-18).
        """
        self._mc = MilvusClient(settings.milvus_uri)
        if settings.milvus_db not in self._mc.list_databases():
            self._mc.create_database(settings.milvus_db)
        self._mc.use_database(settings.milvus_db)
        self._col = settings.milvus_collection
        self._top_k = settings.vector_top_k

    async def ensure_collection(self) -> None:
        """컬렉션 없으면 생성 — 스키마는 bench4 docs/VECTOR.md 표와 1:1."""
        def _ensure() -> bool:
            if self._mc.has_collection(self._col):
                return False
            schema = self._mc.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("v_id", DataType.INT64)
            schema.add_field("kind", DataType.VARCHAR, max_length=8)
            schema.add_field("s", DataType.FLOAT)
            schema.add_field("e", DataType.FLOAT)
            schema.add_field("scene_id", DataType.INT64)
            schema.add_field("h_id", DataType.INT64)
            schema.add_field("shot_type", DataType.VARCHAR, max_length=16)
            schema.add_field("tags", DataType.VARCHAR, max_length=64)
            schema.add_field("labels", DataType.VARCHAR, max_length=64)
            schema.add_field("score_delta", DataType.INT16)
            schema.add_field("inning", DataType.VARCHAR, max_length=8)
            schema.add_field("text", DataType.VARCHAR, max_length=1024)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
            idx = self._mc.prepare_index_params()
            idx.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
            self._mc.create_collection(self._col, schema=schema, index_params=idx)
            return True

        if await asyncio.to_thread(_ensure):
            log.info("컬렉션 생성: %s (dim=%d)", self._col, EMBED_DIM)

    async def replace(self, v_id: int, rows: list[dict]) -> int:
        """
        Summary:
            v_id 의 색인을 통째로 교체한다 (delete-insert 멱등 — DB 관례와 동일).
        Args:
            v_id (int): 대상 영상 id.
            rows (list[dict]): vector 필드까지 채워진 색인 행들.
        Returns:
            int: 적재 행 수.
        """
        def _replace() -> None:
            self._mc.delete(self._col, filter=f"v_id == {v_id}")
            for i in range(0, len(rows), 500):
                self._mc.insert(self._col, rows[i:i + 500])
            self._mc.flush(self._col)

        await self.ensure_collection()
        await asyncio.to_thread(_replace)
        log.info("색인 교체: v_id=%s %d건 → %s", v_id, len(rows), self._col)
        return len(rows)

    async def search(self, query_vec: list[float], v_id: int) -> list[dict]:
        """
        Summary:
            질의 벡터로 v_id 범위 검색 — 상위 top_k 히트 (엔티티 필드 포함).
        Returns:
            list[dict]: {distance, kind, s, e, scene_id, shot_type, tags, labels, text}.
        """
        def _search() -> list[dict]:
            hits = self._mc.search(
                self._col, data=[query_vec], limit=self._top_k,
                filter=f"v_id == {v_id}",
                output_fields=["kind", "s", "e", "scene_id", "shot_type",
                               "tags", "labels", "text"])[0]
            return [{"distance": h["distance"], **h["entity"]} for h in hits]

        return await asyncio.to_thread(_search)

    async def ready(self) -> bool:
        """접속 가능 여부 (readyz 프로브) — 예외는 False."""
        try:
            await asyncio.to_thread(self._mc.list_collections)
            return True
        except Exception as e:                       # noqa: BLE001 — 프로브는 원인 무관 False
            log.warning("Milvus 프로브 실패: %s", e)
            return False

    async def stats(self, v_id: int | None = None) -> dict:
        """색인 현황 — 전체(또는 v_id) 행 수. 운영 확인용."""
        def _stats() -> dict:
            if not self._mc.has_collection(self._col):
                return {"collection": self._col, "exists": False}
            if v_id is None:
                st = self._mc.get_collection_stats(self._col)
                return {"collection": self._col, "exists": True,
                        "rows": int(st.get("row_count", 0))}
            n = self._mc.query(self._col, filter=f"v_id == {v_id}",
                               output_fields=["count(*)"])
            return {"collection": self._col, "exists": True,
                    "v_id": v_id, "rows": int(n[0]["count(*)"])}

        return await asyncio.to_thread(_stats)

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await asyncio.to_thread(self._mc.close)
