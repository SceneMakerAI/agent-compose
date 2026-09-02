"""영상 repository — t_video 읽기 전용 (전 도메인 공통 테이블).

DB 연결(풀)은 Database(rdb.pool)가 쥐고, 여기서는 SQL·행 매핑만 담당한다.
편성 국면은 t_compose.status_code 가 소유한다 — 이 서비스는 t_video 를 쓰지 않는다.
"""

from dataclasses import dataclass

from asyncmy.cursors import DictCursor

from rdb.pool import Database


@dataclass
class Video:
    """t_video 1행 — 편성 접수·도메인 분기에 필요한 컬럼만 담는다."""

    v_id: int
    cate_id: int | None
    name: str
    status_code: int | None
    comment: str | None


class VideoRepo:
    """t_video 조회 전담 (읽기 전용 — 도메인 분기 재료)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def get(self, v_id: int) -> Video | None:
        """
        Summary:
            v_id 로 영상 1건을 조회한다.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            Video | None: 영상 행. 없으면 None (호출부가 404 로 변환).
        Description:
            - 도메인 플로우 분기는 cate_id 로 한다 (pipeline.dispatch — 5100=야구).
        """
        sql = ("SELECT v_id, cate_id, name, status_code, comment "
               "FROM t_video WHERE v_id = %s")
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            row = await cur.fetchone()
        return Video(**row) if row else None
