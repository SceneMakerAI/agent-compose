"""읽기 repository — 편성·색인이 소비하는 상류 산출 조회 (raw SQL 은 여기에만).

이 모듈은 상류 계약의 소비 지점이다: t_scene_baseball(source='board', 발행본) /
t_segment(상류 prep-vision 분할, 샷+caption) / t_dialogue(STT) /
t_frame_baseball_board_detail(kind='ETC', 하단 자막 OCR) / t_play_baseball(전이 원장).
seg_id 로 scene↔segment 를 조인하지 않는다 — 양쪽 다 delete-insert 라 재실행 시
ID 가 어긋난다 (bench4 관례 계승). scene 시간은 start_ms/end_ms(밀리초 INT) —
초 단위로 환산해 돌려준다. (테이블명 *_baseball 접미는 2026-08 DB 정리 리네임.)
"""

import json

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
            발행본(t_scene_baseball, source='board') 전량 — 선곡 인벤토리·색인 귀속 기준.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            list[dict]: {scene_id, h_id, s, e, tags(list), label_list(list),
                scene_type, inning, score, score_before, score_delta, pitch_sec,
                obs_sec, outs, bases, away_team, home_team}.
        Description:
            - obs_sec = t_play_baseball.end_sec(전광판 관측 시각) — 전이 원장이 보증하는 플레이
              결과 시점. cut 의 관측 하한(컷은 이 시점 이전에 끝날 수 없다) 재료.
            - outs·bases·팀명은 t_transition_baseball(전광판 상태) — 인벤토리에 "상황"을 주는
              유일한 재료다. 태그·라벨로는 2사 만루 위기 탈출과 무사 주자없음 평범한 아웃이
              똑같이 보인다(둘 다 범타·score_delta=0). 볼카운트는 넣지 않는다: 관측이 플레이
              종료 후라 실측 330건 중 314건이 0-0 이라 변별력이 없다.
            - 병합 장면은 h_id 가 앞쪽 것을 유지해 board_sec 이 실제 결과보다 이를 수 있다
              (audit 6-1) — 상황이 병합 전 값일 여지가 있다.
        """
        sql = (
            "SELECT sc.scene_id, sc.h_id, sc.scene_type, sc.inning, sc.score, "
            "       sc.score_before, sc.score_delta, sc.labels, sc.pitch_sec, "
            "       sc.start_ms / 1000 AS s, sc.end_ms / 1000 AS e, "
            "       p.end_sec AS obs_sec, "
            "       tr.outs, tr.bases, tr.away_team, tr.home_team "
            "FROM t_scene_baseball sc "
            "LEFT JOIN t_play_baseball p ON p.v_id = sc.v_id AND p.h_id = sc.h_id "
            "LEFT JOIN t_transition_baseball tr "
            "       ON tr.v_id = sc.v_id AND tr.sec = p.board_sec "
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

    async def fetch_pitch_windows(self, v_id: int) -> dict[int, list[tuple[int, int]]]:
        """
        Summary:
            장면별 투구 후보 구간 — bounds 의 시작 후보 재료.
        Returns:
            dict: {scene_id: [(시작초, 끝초), …]}.
        Description:
            - t_transition_baseball.pitches 는 보드 검출이 찾은 투구 구간 목록이라
              **샷 분류(shot_type)와 독립적인 재료**다. 장면 시작 이전은 분류가 비어 있는
              경우가 많아(하이라이트 구간만 분류) 이쪽이 유일한 단서일 때가 있다.
              v201 실측: 전이 293건 중 246건(84%)에 값이 있다.
        """
        sql = (
            "SELECT sc.scene_id, tr.pitches "
            "FROM t_scene_baseball sc "
            "JOIN t_play_baseball p ON p.v_id = sc.v_id AND p.h_id = sc.h_id "
            "JOIN t_transition_baseball tr ON tr.v_id = sc.v_id AND tr.sec = p.board_sec "
            "WHERE sc.v_id = %s AND sc.source = 'board' AND tr.pitches IS NOT NULL"
        )
        out: dict[int, list[tuple[int, int]]] = {}
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            for r in await cur.fetchall():
                try:
                    wins = json.loads(r["pitches"])
                except (TypeError, ValueError):
                    continue                      # 형식이 깨진 행은 조용히 건너뛴다
                out[r["scene_id"]] = [
                    (int(w[0]), int(w[1])) for w in wins if isinstance(w, list) and len(w) == 2
                ]
        return out

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
            하단 자막 OCR(t_frame_baseball_board_detail, kind='ETC') — 매치업·선수 기록 재료.
        Returns:
            list[tuple]: (초, txt) 시간순.
        Description:
            - idx_sec 사용 — idx 는 프레임 인덱스라 샘플링 주기가 바뀌면 초와 어긋난다
              (현재 1fps 라 값이 같지만 idx_sec 이 의미상 정본, DB 정리 2026-08).
        """
        sql = (
            "SELECT idx_sec, txt FROM t_frame_baseball_board_detail "
            "WHERE v_id = %s AND kind = 'ETC' AND txt <> '' ORDER BY idx_sec"
        )
        async with self._db.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, (v_id,))
            return [(int(i), t.strip()) for i, t in await cur.fetchall()]
