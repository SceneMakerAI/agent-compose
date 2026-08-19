"""ETC 자막 → 장면별 타자 이름 (순수 함수 — DB·LLM 무관).

하단 자막에는 타석마다 그 타자의 성적이 뜬다 ("2026 강민호 .267 / 7홈런 40타점").
그런데 자막은 **타석 준비 중**에 뜨고 발행 장면은 **플레이 순간**만 담아 시간이 겹치지
않는다 — v201 실측 865건 중 장면에 겹치는 건 153건(18%)뿐이다. 그래서 겹침이 아니라
**장면 시작 직전 구간**에서 이름을 모은다 (STT 의 여운 폴백과 반대 방향).

판정은 **타석 소개 자막**("2026 {이름} .267 …")을 1순위로 본다. 타석이 시작될 때 한 번
뜨므로 장면 직전의 마지막 소개가 곧 그 타자다. 빈도·최근성·최초 기준은 모두 오답을 냈다
(주석 참조). 소개 자막이 없으면 일반 후보의 최빈으로 떨어진다.

한계: 한 타석 안에서 여러 장면이 발행되거나 연속 타석이 촘촘하면 앞 타자를 물 수 있다
(실측 v201 장면 5). 근본 해결은 전광판 카운트 리셋으로 타석 경계를 받는 것이다 — 후속.
"""

import re
from collections import Counter

# 자막에 이름과 나란히 나오는 비-이름 토큰. 이름 후보에서 제외한다.
_STOP = frozenset({
    "시즌", "통산", "최근", "오늘", "어제", "득점권", "전적", "잠시", "이후", "당시",
    "주중", "마지막", "만루", "홈런", "안타", "타점", "득점", "출루율", "장타율",
    "피안타율", "피홈런", "자책", "실점", "이닝", "선발", "등판", "구원", "세이브",
    "뜬공", "삼진", "땅볼", "볼넷", "사구", "희생타", "병살", "도루", "실책",
    "라운드", "순위", "우세", "열세", "연승", "연패", "타수", "무안타", "타율",
    "캐스터", "해설", "중계", "심판", "감독", "코치", "선수", "타자", "투수",
})
# 팀명(중계 자막에 상대팀·소속팀으로 자주 등장) — 이름으로 오인하면 안 된다.
_TEAMS = frozenset({
    "삼성", "롯데", "기아", "두산", "한화", "키움", "엘지", "에스에스지", "엔씨",
    "KIA", "KT", "LG", "NC", "SSG", "라이온즈", "자이언츠", "위즈", "다이노스",
    "이글스", "히어로즈", "베어스", "타이거즈", "랜더스", "트윈스",
})
_HANGUL = re.compile(r"[가-힣]{2,4}")
# 타석 소개 자막: "2026 강민호 .267 / 7홈런 40타점 OPS.796" — 연도 + 이름으로 시작
_SEASON = re.compile(r"^\d{4}\s+([가-힣]{2,4})\b")
# 투수/타자가 한 줄에 함께 나오는 형태: "P 정철원 / 7 전병우 .229 2/3"
_PAIR = re.compile(r"^P\s+([가-힣]{2,4})\s*/\s*\d+\s+([가-힣]{2,4})")
# 주어가 맨 앞에 오는 형태: "2026 강민호 …" · "▶ 구자욱 …" · "7 노진혁 오늘 …"
_LEAD = re.compile(r"^(?:\d{4}|▶|\d{1,2})\s*([가-힣]{2,4})\b")
# 투수 성적 줄 표지 — 이 줄의 주어는 타자가 아니다.
_PITCHER_MARK = re.compile(
    r"피안타율|피홈런|등판|이닝|실점|자책|구원|세이브|탈삼진|방어율|평균자책|"
    r"연승|연패|\d+승|\d+패|\d+홀|\d+세")
# 타자 성적 줄 표지 — **이게 있어야만** 이름을 받는다. 배제만으로는 부족했다:
# 투수 소개 자막이 타석 직전에도 뜨는데 표지가 없는 형태가 있어 최근성 기준이 그걸
# 집었다 (실측 v201 장면 5 = 원태인, 삼성 선발투수).
_BATTER_MARK = re.compile(
    r"타점|안타|출루율|장타율|타율|OPS|타수|득점권|뜬공|땅볼|볼넷|사구|오늘\s*\d+/\d+|\d+/\d+")


def _candidates(txt: str) -> list[str]:
    """자막 한 줄에서 타자 이름 후보 — 앞자리 우선, 없으면 한글 토큰 전부.

    'A vs B' 는 앞이 주어(타자)다 — 뒤는 상대 투수라 후보에서 뺀다.
    """
    if not txt:
        return []
    if txt.startswith(("캐스터", "해설")):
        return []                                   # 중계진 소개 — 선수가 아니다
    if m := _PAIR.match(txt):
        return [m.group(2)]                         # 쌍이면 타자(뒤쪽)만
    if _PITCHER_MARK.search(txt) or not _BATTER_MARK.search(txt):
        return []                                   # 투수 줄이거나 타자 성적 줄이 아니다
    head = txt.split(" vs ")[0]                     # 맞대결이면 앞쪽만
    if m := _LEAD.match(head):
        name = m.group(1)
        if name not in _STOP and name not in _TEAMS:
            return [name]
    return [t for t in _HANGUL.findall(head) if t not in _STOP and t not in _TEAMS]


def batter_of(etc_rows: list[tuple[int, str]], start: float,
              lookback: int = 240, lookahead: int = 5,
              since: float | None = None) -> str | None:
    """장면 시작 직전 구간의 ETC 자막에서 타자 이름.

    Args:
        etc_rows: (초, 자막) 시간순. start: 장면 시작 초.
        lookback: 몇 초 전까지 볼 것인가 (타석 준비 자막이 뜨는 범위).
        lookahead: 장면 시작 직후 몇 초까지 포함할 것인가.
        since: 직전 발행 장면의 끝 — 이 이후만 본다. 앞 타석 자막이 플레이 뒤에도
            한동안 남아, 창을 넓게 잡으면 직전 타자를 물고 온다 (실측 v201 장면 5:
            황성빈이 출루한 장면 4 의 자막이 다음 타석까지 이어짐).
    Returns:
        str | None: 이름. 후보가 없으면 None (억지로 채우지 않는다).
    """
    lo, hi = start - lookback, start + lookahead
    if since is not None:
        lo = max(lo, since)

    # 1순위: 타석 소개 자막 ("2026 {이름} .267 / 7홈런 …"). 타석이 시작될 때 한 번 뜨므로
    # 장면 직전의 **마지막** 소개가 곧 그 타자다. 빈도·최근성 규칙은 둘 다 오답을 냈다 —
    # 최빈은 앞 타석 자막이 많으면 지고(장면 65), 최근은 출루한 앞 타자의 주자 자막에
    # 지고(장면 5), 최초는 앞 이닝 잔여를 물었다(장면 4·7).
    intro: str | None = None
    for sec, txt in etc_rows:
        if not (lo <= sec <= hi) or _PITCHER_MARK.search(txt):
            continue
        if (m := _SEASON.match(txt)) and m.group(1) not in _STOP | _TEAMS:
            intro = m.group(1)
    if intro:
        return intro

    # 2순위: 소개 자막이 없으면 일반 후보의 최빈 (1회성 토큰은 잡음으로 배제)
    c: Counter[str] = Counter()
    for sec, txt in etc_rows:
        if lo <= sec <= hi:
            c.update(_candidates(txt))
    cands = [n for n, hits in c.items() if hits >= 2]
    return max(cands, key=lambda n: c[n]) if cands else None


def annotate_batters(scenes: list[dict], etc_rows: list[tuple[int, str]]) -> None:
    """각 장면 행에 batter 키를 채운다 (제자리 수정 — 인벤토리 렌더 직전용).

    직전 장면의 끝을 하한으로 준다 — 앞 타석 자막을 물지 않게 (batter_of 의 since).
    """
    prev_end: float | None = None
    for r in sorted(scenes, key=lambda x: x["s"]):
        r["batter"] = batter_of(etc_rows, r["s"], since=prev_end)
        prev_end = r["e"]
