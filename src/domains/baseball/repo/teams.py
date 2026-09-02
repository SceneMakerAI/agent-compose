"""팀명 사전 repository — t_team_baseball 읽기 전용.

질의·중계·자막의 다양한 팀 표기(엘지·트윈스·랜더스…)를 전광판 표기(team_id)로
정규화하는 재료다. team_id 는 구간의 home_team/away_team 값과 직접 일치한다.
"""

from asyncmy.cursors import DictCursor

from log import get_logger
from rdb.pool import Database

log = get_logger(__name__)


class TeamRepo:
    """t_team_baseball 조회 전담 (읽기 전용)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def fetch(self) -> dict[str, list[str]]:
        """
        Summary:
            팀명 사전 전량 — {team_id: [별칭…]} (alias 콤마 분해, 등장 순서 유지).
        Returns:
            dict[str, list[str]]: 대표(전광판 표기) → 별칭 목록.
                별칭에 team_id 자신이 포함돼 있어도 그대로 둔다 (표시는 호출부 몫).
        """
        sql = "SELECT team_id, alias FROM t_team_baseball ORDER BY team_id"
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql)
            rows = list(await cur.fetchall())

        teams: dict[str, list[str]] = {}
        for row in rows:
            aliases = []
            for piece in (row["alias"] or "").split(","):
                piece = piece.strip()
                if piece:
                    aliases.append(piece)
            teams[row["team_id"]] = aliases
        log.debug("팀명 사전 로드: %d팀", len(teams))
        return teams
