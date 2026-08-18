"""읽기 repository — 편성·색인이 소비하는 상류 산출 조회 (raw SQL 은 여기에만).

이 모듈은 상류 계약의 소비 지점이다: t_scene(source='board', 발행본) / t_segment
(상류 prep-vision 분할, 샷+caption) / t_dialogue(STT) / t_frame_board_detail
(kind='ETC', 하단 자막 OCR). seg_id 로 t_scene↔t_segment 를 조인하지 않는다 —
양쪽 다 delete-insert 라 재실행 시 ID 가 어긋난다 (bench4 관례 계승).
t_scene 시간은 start_ms/end_ms(밀리초 INT) — 초 단위로 환산해 돌려준다.
"""

from asyncmy.cursors import DictCursor

from db.pool import Database
from log import get_logger

log = get_logger(__name__)


class SourceRepo:
    """색인·편성 재료 조회 전담 (읽기 전용)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def fetch_scenes(self, v_id: int) -> list[dict]:
        """
        Summary:
            발행본(t_scene, source='board') 전량 — 선곡 인벤토리·색인 귀속 기준.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            list[dict]: {scene_id, h_id, s, e, tags(list), label_list(list),
                scene_type, inning, score, score_before, score_delta, pitch_sec,
                obs_sec}.
        Description:
            - obs_sec = t_play.end_sec(전광판 관측 시각) — 전이 원장이 보증하는 플레이
              결과 시점. cut 의 관측 하한(컷은 이 시점 이전에 끝날 수 없다) 재료.
        """
        sql = (
            "SELECT sc.scene_id, sc.h_id, sc.scene_type, sc.inning, sc.score, "
            "       sc.score_before, sc.score_delta, sc.labels, sc.pitch_sec, "
            "       sc.start_ms / 1000 AS s, sc.end_ms / 1000 AS e, "
            "       p.end_sec AS obs_sec "
            "FROM t_scene sc "
            "LEFT JOIN t_play p ON p.v_id = sc.v_id AND p.h_id = sc.h_id "
            "WHERE sc.v_id = %s AND sc.source = 'board' ORDER BY sc.scene_id"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())
        for r in rows:
            r["s"], r["e"] = float(r["s"]), float(r["e"])
            r["tags"] = (r["scene_type"] or "").split(",")
            r["label_list"] = r["labels"].split(",") if r["labels"] else []
        return rows

    async def fetch_shots(self, v_id: int) -> list[dict]:
        """
        Summary:
            분할 샷(t_segment, 상류 prep-vision 산출) — caption(summary) 있는 행만 (색인 재료).
        Returns:
            list[dict]: {s, e, shot_type, summary}.
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, "
            "       shot_type, summary "
            "FROM t_segment WHERE v_id = %s "
            "AND summary IS NOT NULL ORDER BY start_time"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())
        for r in rows:
            r["s"], r["e"] = float(r["s"]), float(r["e"])
        return rows

    async def fetch_shots_all(self, v_id: int) -> list[dict]:
        """
        Summary:
            분할 샷 전량 (caption 유무 무관) — compose cut 레시피 재료.
        Returns:
            list[dict]: {s, e, shot_type} 시간순.
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, shot_type "
            "FROM t_segment WHERE v_id = %s "
            "ORDER BY start_time"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())
        for r in rows:
            r["s"], r["e"] = float(r["s"]), float(r["e"])
        return rows

    async def fetch_utterances(self, v_id: int) -> list[tuple[float, float, str]]:
        """
        Summary:
            STT 발화 전량 (t_dialogue) — 색인 청크·컷 꼬리 스냅·endfix 재료.
        Returns:
            list[tuple]: (s, e, text) 시간순.
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, dialogue "
            "FROM t_dialogue WHERE v_id = %s ORDER BY start_time"
        )
        async with self._db.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, (v_id,))
            return [(float(s), float(e), (t or "").strip())
                    for s, e, t in await cur.fetchall()]

    async def fetch_etc_rows(self, v_id: int) -> list[tuple[int, str]]:
        """
        Summary:
            하단 자막 OCR(t_frame_board_detail, kind='ETC') — 매치업·선수 기록 재료.
        Returns:
            list[tuple]: (idx(초), txt) 시간순.
        """
        sql = (
            "SELECT idx, txt FROM t_frame_board_detail "
            "WHERE v_id = %s AND kind = 'ETC' AND txt <> '' ORDER BY idx"
        )
        async with self._db.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, (v_id,))
            return [(int(i), t.strip()) for i, t in await cur.fetchall()]
