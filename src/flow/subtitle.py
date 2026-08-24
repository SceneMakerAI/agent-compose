"""장면별 하단 자막(ETC) 원문 → 인벤토리 '기타정보' 줄 (순수 함수 — DB·LLM 무관).

하단 자막에는 타석마다 그 타석의 정보가 뜬다 ("1.최원준 .355 2-2 P.레일러 5" ·
"2026 강민호 .267 / 7홈런 40타점"). 그런데 자막은 **타석 준비 중**에 뜨고 발행 장면은
**플레이 순간**만 담아 시간이 겹치지 않는다 — v201 실측 865건 중 장면에 겹치는 건
153건(18%)뿐이다. 그래서 겹침이 아니라 **장면 시작 직전 구간**에서 마지막 자막을 집는다.

집은 줄은 **원문 그대로** 싣는다. 예전에는 정규식으로 타자 이름 한 칸만 뽑았는데
자막 포맷이 영상마다 달라 오답을 냈다 — v203 실측: 83장면 중 48장면에 이름이 붙었으나
장면 23·25·27 은 '오원석', 28 은 '레일러'로 **투수**를 타자로 읽었다(이 영상 포맷
"1.최원준 .355 2-2 P.레일러 5" 가 규칙이 가정한 "2026 강민호 .267 …" 와 다르다).
줄을 통째로 주면 타순·타율·볼카운트·상대 투수가 다 남고 포맷 판정은 모델 몫이 된다.

한계: 한 타석 안에서 여러 장면이 발행되거나 연속 타석이 촘촘하면 앞 타석 자막을 물 수
있다 (실측 v201 장면 5). 근본 해결은 전광판 카운트 리셋으로 타석 경계를 받는 것 — 후속.
"""

# 로고 조각·OCR 파편 배제 기준. vector.ingest.merge_etc 의 ETC_MIN_CHARS 와 같은 값이나
# 소유 모듈이 달라 각자 둔다 (실측 v203 은 최단 21자라 이 문턱에 걸리는 행이 없다).
ETC_MIN_CHARS = 5


def etc_at(etc_rows: list[tuple[int, str]], start: float,
           lookback: int = 240, lookahead: int = 5,
           since: float | None = None) -> str | None:
    """
    Summary:
        장면 시작 직전 구간의 **마지막** ETC 자막 원문 1줄.
    Args:
        etc_rows (list[tuple]): (초, 자막) 시간순 — 1fps 프레임 단위라 같은 줄이 반복된다.
        start (float): 장면 시작 초.
        lookback (int): 몇 초 전까지 볼 것인가 (타석 준비 자막이 뜨는 범위).
        lookahead (int): 장면 시작 직후 몇 초까지 포함할 것인가.
        since (float | None): 직전 발행 장면의 끝 — 이 이후만 본다. 앞 타석 자막이
            플레이 뒤에도 한동안 남아, 창을 넓게 잡으면 직전 타석 줄을 물고 온다
            (실측 v201 장면 5: 장면 4 의 자막이 다음 타석까지 이어짐).
    Returns:
        str | None: 자막 원문. 구간에 자막이 없으면 None (억지로 채우지 않는다).
    Description:
        프레임 반복은 마지막 것만 남기면 그만이라 별도 병합을 하지 않는다 — 같은 줄이
        연속하면 마지막 프레임의 텍스트가 곧 그 런의 텍스트다.
    """
    lo, hi = start - lookback, start + lookahead
    if since is not None:
        lo = max(lo, since)
    last: str | None = None
    for sec, txt in etc_rows:
        if lo <= sec <= hi and len(txt) >= ETC_MIN_CHARS:
            last = txt
    return last


def annotate_etc(scenes: list[dict], etc_rows: list[tuple[int, str]]) -> None:
    """
    Summary:
        각 장면 행에 etc 키(자막 원문)를 채운다 — 제자리 수정, 인벤토리 렌더 직전용.
    Description:
        직전 장면의 끝을 하한으로 준다 (etc_at 의 since — 앞 타석 자막 방지).
    """
    prev_end: float | None = None
    for r in sorted(scenes, key=lambda x: x["s"]):
        r["etc"] = etc_at(etc_rows, r["s"], since=prev_end)
        prev_end = r["e"]
