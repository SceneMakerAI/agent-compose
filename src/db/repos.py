"""읽기 repository — 편성·색인이 소비하는 상류 산출 조회 (raw SQL 은 여기에만).

이 모듈은 상류 계약의 소비 지점이다: t_scene_baseball(vision3 scene 단계 발행본) /
t_play_baseball(사실 원장) / t_scoreboard_baseball(전광판 관측) / t_segment(상류
prep-vision 분할, 샷+caption) / t_dialogue(STT) / t_frame_baseball_board_detail
(하단 자막 OCR + 팀명).

**2026-08-23 상류 개편 대응** — 바뀐 계약을 여기서 흡수해 flow 는 옛 이름 그대로 쓴다:
- `t_transition_baseball` **삭제**. 대체는 t_scoreboard_baseball(관측 전량)이다.
  전이 조회(`fetch_transitions`)는 재작성하지 않고 없앴다 — 플로우 재설계로
  score_match 가 사라지면서 `Inventory.trans` 를 읽는 코드가 남지 않았다.
- t_scene_baseball: source·seg_start·seg_end·h_id·score·pitch_sec·scene_type 제거,
  start_ms/end_ms(int ms) → start_time/end_time('HH:MM:SS.0'), tags(전광판 사실)·
  game_context·pitch_idxs·description 신설, scene_type → labels 로 통합.
- t_play_baseball: h_id → p_id (= scene_id), board_sec → board_time, end_sec 소멸.
- t_segment: 코드가 쓰던 `shot_type` 컬럼은 운영 DB 에 없었고 `scene_type` 으로
  신설됐다. **여기서 `shot_type` 으로 되돌려 실어 준다** — 샷의 유형이라는 뜻엔
  그 이름이 맞고, 상류도 t_scene_baseball.scene_type 과의 이름 충돌을 인정했다
  (vision3 migration_20260823h 주석). 상류 컬럼명을 찾을 땐 이 주석이 다리다.

seg_id 로 scene↔segment 를 조인하지 않는다 — 양쪽 다 delete-insert 라 재실행 시
ID 가 어긋난다 (bench4 관례 계승). 시간은 전부 초(float/int)로 환산해 돌려준다 —
compose 내부 좌표는 초 하나뿐이고, 문자열 시각은 이 경계를 넘지 않는다.
"""

from collections import Counter

from asyncmy.cursors import DictCursor

from db.pool import Database
from flow import vocab
from log import get_logger

log = get_logger(__name__)

# 팀명 자막(kind='TEAM') 형식 — "KIA 3: 삼성 2" (원정 먼저). 숫자·구분자를 걷어내고
# 팀 이름 두 개만 남긴다. 중계 자막엔 **다른 경기 스코어**도 섞여 흐르므로
# (v201 실측: 삼성:롯데 3,619프레임 vs KIA:NC 103·한화:KT 84) 최빈 쌍을 경기로 본다.
_TEAM_MIN_FRAMES = 30


def _sec(text: str | None) -> float | None:
    """'HH:MM:SS.0' → 초. 값이 없거나 형식이 어긋나면 None (합성값 금지)."""
    if not text:
        return None
    try:
        h, m, rest = text.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    except ValueError:
        return None


def parse_pitch_idxs(text: str | None) -> list[tuple[int, int]]:
    """'708-714,715-718' → [(708, 714), (715, 718)] (순수 함수).

    상류 규약상 끝은 **포함**이다. 소비처(bounds.start_candidates)는 시작만 쓰므로
    포함/배타가 결과를 가르지 않지만, 값을 바꿔 담지는 않는다 — 원문 그대로 옮긴다.
    형식이 깨진 조각은 조용히 건너뛴다 (부분 파손이 전량 손실이 되지 않게).
    """
    out = []
    for part in (text or "").split(","):
        lo, _, hi = part.strip().partition("-")
        try:
            out.append((int(lo), int(hi)))
        except ValueError:
            continue
    return out


class SourceRepo:
    """색인·편성 재료 조회 전담 (읽기 전용)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def fetch_scenes(self, v_id: int) -> list[dict]:
        """
        Summary:
            발행본(t_scene_baseball) 전량 — 선곡 인벤토리·색인 귀속 기준.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            list[dict]: {scene_id, s, e, tags(list), label_list(list), board_tags(list),
                game_context, inning, score_before, score_delta, pitch_sec, obs_sec,
                outs, bases, description}.
        Description:
            - `tags`·`label_list` 는 상류 **labels 한 칸**을 어휘로 분해한 것이다
              (vocab.split_labels — 행위 태그 / 파생 라벨). 상류가 두 축을 한 칸에
              담기로 한 뒤에도 compose 는 축마다 다른 표를 보므로 나눠 쓴다.
            - `board_tags` 는 상류 `tags` 컬럼 = **전광판 사실**(아웃·1루·주자득점…)
              이고 위 두 축과 다른 물건이다. 해석(labels)이 비어도 사실은 남는다는 게
              이 컬럼의 존재 이유라(vision3 migration_20260823k), 실측 v200~203 에서
              labels 가 NULL 인 12행이 정확히 그 자리다 — 인벤토리에서 그 행이
              태그 없는 빈칸으로 보이지 않게 하는 유일한 재료다.
            - obs_sec = t_play_baseball.board_time(전광판에 결과가 뜬 시각). cut 의
              관측 하한(컷은 이 시점 이전에 끝날 수 없다) 재료. 원장의 end_time 은
              쓰지 않는다 — DDL 주석과 달리 실측 337행 전부 빈 문자열이다(미구현).
            - outs·bases 는 그 관측 시각의 t_scoreboard_baseball 행. 인벤토리에
              "상황"을 주는 유일한 재료다 — 태그·라벨로는 2사 만루 위기 탈출과
              무사 주자없음 평범한 아웃이 똑같이 보인다(둘 다 범타·score_delta=0).
              볼카운트는 넣지 않는다: 관측이 플레이 종료 후라 변별력이 없다.
            - 시간이 비어 있는 행은 **버린다**. 상류 scene 은 끝을 못 정하면 NULL 을
              쓰는데(합성값 금지), 구간이 없는 클립은 편성도 색인도 할 수 없다.
        """
        sql = (
            "SELECT sc.scene_id, sc.start_time, sc.end_time, sc.tags AS board_tags, "
            "       sc.labels, sc.inning, sc.score_before, sc.score_delta, "
            "       sc.game_context, sc.pitch_time, sc.description, "
            "       p.board_time AS obs_time, sb.`out` AS outs, sb.base AS bases "
            "FROM t_scene_baseball sc "
            "LEFT JOIN t_play_baseball p "
            "       ON p.v_id = sc.v_id AND p.p_id = sc.scene_id "
            "LEFT JOIN t_scoreboard_baseball sb "
            "       ON sb.v_id = sc.v_id AND sb.idx_time = p.board_time "
            "WHERE sc.v_id = %s ORDER BY sc.scene_id"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())

        out, dropped = [], []
        for r in rows:
            s, e = _sec(r["start_time"]), _sec(r["end_time"])
            if s is None or e is None or e <= s:
                dropped.append(r["scene_id"])
                continue
            tags, label_list = vocab.split_labels(r["labels"])
            out.append({
                "scene_id": r["scene_id"], "s": s, "e": e,
                "tags": tags, "label_list": label_list,
                "board_tags": [t for t in (r["board_tags"] or "").split(",") if t],
                "game_context": r["game_context"],
                "inning": r["inning"], "score_before": r["score_before"],
                "score_delta": r["score_delta"] or 0,
                "pitch_sec": _sec(r["pitch_time"]),
                "obs_sec": _sec(r["obs_time"]),
                # 전광판 미인식 센티널(-1)은 여기서 None 으로 — 그대로 새면
                # '-1사 주자없음' 같은 값이 프롬프트에 실린다.
                "outs": None if r["outs"] is None or r["outs"] < 0 else int(r["outs"]),
                "bases": r["bases"] or None,
                "description": r["description"],
            })
        if dropped:
            log.warning("발행본 구간 결손으로 제외: v_id=%s 장면 %s", v_id, dropped)
        return out

    async def fetch_pitch_windows(self, v_id: int) -> list[tuple[int, int]]:
        """
        Summary:
            검출 투구 구간 **전량** — bounds 의 시작 후보 재료 (시간순).
        Description:
            원장(t_play_baseball.pitch_idxs)에서 읽는다. 발행본이 아니라 원장인 이유:
            bounds 가 찾는 건 **클립 경계 부근의 투구**라 그 장면 소유분만으로는
            모자란다. 원장에는 대표 투구를 못 골라 발행에서 빠진 사건의 후보까지
            남아 있어 범위가 더 넓다.

            투구 검출은 샷 분류와 **독립**이라 상류 분류가 비어도 살아 있는 유일한
            단서다 — 장면 이전 구간은 분류가 NULL 인 경우가 많다.
        Returns:
            list[tuple[int, int]]: [(투구 시작, 끝)…] 시간순, 중복 제거.
        """
        sql = ("SELECT pitch_idxs FROM t_play_baseball "
               "WHERE v_id = %s AND pitch_idxs <> '' ORDER BY p_id")
        out: set[tuple[int, int]] = set()
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            for r in await cur.fetchall():
                out.update(parse_pitch_idxs(r["pitch_idxs"]))
        return sorted(out)

    async def fetch_teams(self, v_id: int) -> tuple[str, str] | None:
        """
        Summary:
            (원정, 홈) 팀명 — 전광판 팀 자막(kind='TEAM')의 최빈 쌍.
        Returns:
            tuple[str, str] | None: 판독이 모자라면 None (지어내지 않는다).
        Description:
            발행본이 팀명을 잃어 여기가 유일한 출처가 됐다 (구 t_scene.score
            'KIA 0-0 삼성' 제거 — vision3 migration_20260823i: 전이 원장 폐기로
            팀명 출처가 사라졌다).

            자막 원문은 "KIA 3: 삼성 2" 로 **원정이 먼저**다. 점수는 매 프레임 달라도
            팀 이름은 안 바뀌므로 숫자를 걷어낸 뒤 최빈 쌍을 고른다 — 중계 자막엔
            다른 경기 스코어가 섞여 흐르기 때문에 단순 최초/최종 값은 못 쓴다
            (v201 실측: 삼성:롯데 3,619 vs KIA:NC 103 · 한화:KT 84).
        """
        sql = ("SELECT txt FROM t_frame_baseball_board_detail "
               "WHERE v_id = %s AND kind = 'TEAM' AND txt <> ''")
        pairs: Counter[tuple[str, str]] = Counter()
        async with self._db.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, (v_id,))
            for (txt,) in await cur.fetchall():
                away, sep, home = txt.partition(":")
                if not sep:
                    continue
                away = away.rsplit(" ", 1)[0].strip()      # 뒤에 붙은 점수 절단
                home = home.strip().rsplit(" ", 1)[0].strip()
                if away and home:
                    pairs[(away, home)] += 1
        if not pairs:
            return None
        (away, home), n = pairs.most_common(1)[0]
        if n < _TEAM_MIN_FRAMES:
            log.warning("팀명 판독 부족: v_id=%s 최빈 %s-%s %d프레임", v_id, away, home, n)
            return None
        return away, home

    async def fetch_shots(self, v_id: int) -> list[dict]:
        """
        Summary:
            분할 샷(t_segment, 상류 prep-vision 산출) — caption(summary) 있는 행만 (색인 재료).
        Returns:
            list[dict]: {s, e, shot_type, summary}. shot_type 은 상류 `scene_type` 이다
                (모듈 docstring — 이름만 이 경계에서 되돌린다).
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, "
            "       scene_type AS shot_type, summary "
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
            summary 를 함께 싣는다. bounds 가 "이 시각에 무슨 화면인가"를 프롬프트에
            붙이는 재료다 — shot_type 만으로는 '타구·수비'가 그 플레이인지 앞 플레이
            잔상인지 갈리지 않는다 (v201 장면9 실측: 첫 샷이 앞 타구의 "야수가 공을
            쫓아 걷고 있다"인데 완결성 '정상' 판정을 받았다).
            caption 없는 행도 유지 — 컷 레시피는 유형만 보므로 빼면 경계가 달라진다.
        Returns:
            list[dict]: {s, e, shot_type, summary} 시간순.
        """
        sql = (
            "SELECT TIME_TO_SEC(start_time) AS s, TIME_TO_SEC(end_time) AS e, "
            "       scene_type AS shot_type, summary "
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
            STT 발화 전량 (t_dialogue) — 색인 청크·컷 꼬리 스냅·끝 보정 재료.
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
