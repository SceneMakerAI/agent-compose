"""t_compose 저장·조회 repository (bench4 db.save_compose 이식 — async).

요청마다 새 comp_id (이력 보존 — t_compose_clip 스냅샷 컬럼은 ui 가 재조인 없이
렌더하는 실사용 계약, design.md §2-⑤). score 전/후는 문자열 되파싱 대신
클립 dict 의 개별 필드(score_before/score_after)로 받는다 — bench4 의
"report 문자열을 db 가 되파싱" 결합(실사 지적) 제거.
"""

from db.pool import Database
from log import get_logger

log = get_logger(__name__)


class ComposeRepo:
    """편성 결과 저장·조회 전담."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def save(self, v_id: int, query: str, spec: dict | None,
                   status: str, clips: list[dict]) -> int:
        """
        Summary:
            편성 헤더 1행 + 클립 N행 저장 — comp_id 반환.
        Args:
            clips (list[dict]): {scene_id, h_id, start, end, label, labels, inning,
                score_before, score_after} (시간은 초 — SEC_TO_TIME 으로 저장).
        """
        spec = spec or {}
        duration = sum(c["end"] - c["start"] for c in clips)
        async with self._db.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO t_compose (v_id, query, budget_sec, status, mode, "
                    "  view_side, targets, duration, clip_cnt) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (v_id, query, spec.get("budget", 0), status, spec.get("mode"),
                     spec.get("view"), ",".join(spec.get("targets", [])),
                     int(duration), len(clips)))
                comp_id = cur.lastrowid
                if clips:
                    await cur.executemany(
                        "INSERT INTO t_compose_clip (comp_id, clip_seq, v_id, scene_id, "
                        "  h_id, start_time, end_time, scene_type, labels, inning, "
                        "  score_before, score_after) "
                        "VALUES (%s, %s, %s, %s, %s, SEC_TO_TIME(%s), SEC_TO_TIME(%s), "
                        "  %s, %s, %s, %s, %s)",
                        [(comp_id, i, v_id, c["scene_id"], c["h_id"],
                          c["start"], c["end"], c["label"], c["labels"], c["inning"],
                          c["score_before"], c["score_after"])
                         for i, c in enumerate(clips, 1)])
            await conn.commit()
        log.info("t_compose 저장: comp_id=%s (%d클립)", comp_id, len(clips))
        return comp_id

    async def fetch(self, comp_id: int) -> dict | None:
        """저장된 편성 재조회 — 헤더 + 클립 목록 (초 단위 환산)."""
        from asyncmy.cursors import DictCursor
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute("SELECT * FROM t_compose WHERE comp_id = %s", (comp_id,))
            head = await cur.fetchone()
            if not head:
                return None
            await cur.execute(
                "SELECT clip_seq, scene_id, h_id, TIME_TO_SEC(start_time) AS start, "
                "       TIME_TO_SEC(end_time) AS end, scene_type, labels, inning, "
                "       score_before, score_after "
                "FROM t_compose_clip WHERE comp_id = %s ORDER BY clip_seq", (comp_id,))
            clips = list(await cur.fetchall())
        head["reg_datetime"] = str(head.get("reg_datetime"))
        for c in clips:
            c["start"], c["end"] = int(c["start"]), int(c["end"])
        return {**head, "clips": clips}
