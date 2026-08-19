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

    async def fetch_pitch_windows(self, v_id: int) -> list[tuple[int, int]]:
        """
        Summary:
            보드 검출 투구 구간 **전량** — bounds 의 시작 후보 재료 (시간순).
        Description:
            장면 기준으로 조인하면 안 된다. 예전에는 장면의 board_sec 행 하나만 붙여
            그 장면 소유 투구만 실었는데, bounds 가 찾는 건 **클립 경계 부근의 투구**라
            앞선 전이의 투구가 통째로 빠졌다 — v202 장면11(cs=2558)은 창(2518~2558)
            안에 2532·2550 이 있었는데 소유 행(2627)의 2583 만 실려 후보가 0건이었다.

            보드 검출은 shot_type 과 **독립**이라 상류 분류가 비어도 살아 있는 유일한
            단서다. 장면 이전 구간은 분류가 NULL 인 경우가 많다(v202 장면11 앞 40초:
            12샷 전부 NULL) — 그래서 이 재료가 중요하다.
        Returns:
            list[tuple[int, int]]: [(투구 시작, 끝)…] 시간순, 중복 제거.
        """
        sql = ("SELECT pitches FROM t_transition_baseball "
               "WHERE v_id = %s AND pitches IS NOT NULL ORDER BY sec")
        out: set[tuple[int, int]] = set()
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            for r in await cur.fetchall():
                try:
                    wins = json.loads(r["pitches"])
                except (TypeError, ValueError):
                    continue                      # 형식이 깨진 행은 조용히 건너뛴다
                out.update((int(w[0]), int(w[1]))
                           for w in wins if isinstance(w, list) and len(w) == 2)
        return sorted(out)

    async def fetch_transitions(self, v_id: int) -> list[dict]:
        """
        Summary:
            전광판 전이 전량 — verify 가 클립 구간에 겹치는 것만 골라 쓴다.
        Description:
            해설(STT)은 음성 인식이라 오인식이 섞인다 — v201 장면9 대사에는 2회인데
            "볼카운트는 5회초"가 들어 있다. 전이는 판독으로 확정된 사실이라
            그 오인식을 이기는 근거가 된다. hint 는 board 가 붙인 변화 요약
            (득점+1·아웃+1 등)이다.
        Returns:
            list[dict]: {sec, inning, half, outs, balls, strikes, bases,
                         away_score, home_score, hint} 시간순.
        """
        sql = (
            "SELECT sec, inning, half, outs, balls, strikes, bases, "
            "       away_score, home_score, hint "
            "FROM t_transition_baseball WHERE v_id = %s ORDER BY sec"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            return list(await cur.fetchall())

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
        Description:
            summary 를 함께 싣는다. bounds·verify 가 "이 시각에 무슨 화면인가"를
            프롬프트에 붙이는 재료다 — shot_type 만으로는 '타구·수비'가 그 플레이인지
            앞 플레이 잔상인지 갈리지 않는다 (v201 장면9 실측: 첫 샷이 앞 타구의
            "야수가 공을 쫓아 걷고 있다"인데 완결성 '정상' 판정을 받았다).
            caption 없는 행도 유지 — 컷 레시피는 유형만 보므로 빼면 경계가 달라진다.
        Returns:
            list[dict]: {s, e, shot_type, summary} 시간순.
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, "
            "       shot_type, summary "
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
