"""bounds — 클립 시작·끝 후보 생성과 처분 (순수 계산, LLM 무관).

시작과 끝을 한 곳에서 다루는 이유: 따로 물으면 서로를 모른다. 시작을 25초 당기면
끝의 여유도 달라진다.

후보는 여기서 **결정적으로** 만들고 LLM 은 고르기만 한다 — 검증기(apply)가 제시 목록
밖의 값을 기각하므로 모델이 시각을 지어낼 수 없다.

시작 후보를 여러 층에서 뽑는 근거(실측 2026-08-19):
- 편성 클립 121건 중 114건(94%)은 이미 투구 근처에서 시작한다. 문제는 나머지 7건이
  하필 역전 홈런 2건을 포함한다는 것 — 홈런은 FULL_CLIP_TAGS 라 앵커 로직을 통째로
  건너뛰고, 그 장면들은 pitch_sec 도 NULL 이다.
- v202 장면 11(역전 홈런)은 장면이 2558s 에 시작하는데 투구는 2583s 다. 25초 앞의
  '주자가 1루에 슬라이딩' 샷부터 시작해 무슨 일인지 알 수 없었다.
- shot_type 은 하이라이트 구간과 겹치는 샷만 채워져 장면 이전은 NULL 이 많다.
  그래서 보드 검출(t_transition.pitches)을 독립 재료로 함께 쓴다 — v201 전이 293건 중
  246건(84%)에 값이 있다.
"""

import math
import re

from log import get_logger

log = get_logger(__name__)

# 시작을 되돌릴 수 있는 최대 폭 — 이보다 멀면 직전 타석을 물고 온다.
START_MAX_BACK_SEC = 40
# 끝을 늘릴 수 있는 최대 폭 (구 ENDFIX_MAX_EXT_SEC 계승).
END_MAX_EXT_SEC = 12
# 후보 개수 상한 — 많으면 모델이 고르지 못하고 프롬프트만 커진다.
CAND_MAX = 4
# 발화 끝과 샷 경계가 이 안에 함께 오면 "둘 다"로 표시한다 (가장 깔끔한 끝점).
COINCIDE_SEC = 2.0


def start_candidates(clip: dict, segs: list[dict], pitches: list[tuple[int, int]]) -> list:
    """시작 후보 [(초, 설명)] — 앞쪽으로만 되돌린다(뒤로 미루면 플레이를 잘라 먹는다)."""
    cs = clip["cut"]["cs"]
    lo = cs - START_MAX_BACK_SEC
    out: list[tuple[float, str]] = []
    for ps, _pe in pitches:                       # ① 보드 검출 투구 (분류 없이도 있다)
        if lo <= ps < cs:
            out.append((float(ps), "보드 검출 투구"))
    for s in segs:                                 # ② 분류된 투구 샷 시작
        if lo <= s["s"] < cs and s.get("shot_type") == "투구":
            out.append((s["s"], "'투구' 샷 시작"))
    if not out:                                    # ③ 투구를 못 찾으면 타구·수비 시작
        for s in segs:
            if lo <= s["s"] < cs and s.get("shot_type") == "타구·수비":
                out.append((s["s"], "'타구·수비' 샷 시작"))
    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in out:
        uniq.setdefault(int(sec), (sec, why))
    return sorted(uniq.values(), key=lambda x: -x[0])[:CAND_MAX]


def end_candidates(clip: dict, segs: list[dict], utts: list) -> list:
    """끝 후보 [(초, 설명)] — 발화 끝과 샷 경계를 함께 준다.

    둘이 COINCIDE_SEC 안에 겹치면 그렇게 표시한다: 말도 끝나고 화면도 바뀌는 지점이
    가장 깔끔한 끝점인데, 지금까지는 그 정보가 프롬프트에 없어 모델이 알 수 없었다.
    """
    ce = clip["cut"]["ce"]
    hi = ce + END_MAX_EXT_SEC
    utt_ends = [math.ceil(ue) for _us, ue, _t in utts if ce < ue <= hi]
    shot_ends = [s["e"] for s in segs if ce < s["e"] <= hi]
    out: list[tuple[float, str]] = []
    for ue in utt_ends:
        near = any(abs(ue - se) <= COINCIDE_SEC for se in shot_ends)
        out.append((float(ue), "발화 끝 + 화면 전환" if near else "발화 끝"))
    for se in shot_ends:
        if not any(abs(ue - se) <= COINCIDE_SEC for ue in utt_ends):
            out.append((float(se), "화면 전환"))
    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in sorted(out):
        uniq.setdefault(int(sec), (sec, why))
    return sorted(uniq.values())[:CAND_MAX]


def build_rows(clips: list[dict], segs: list[dict], utts: list,
               pitches_of: dict[int, list]) -> list[dict]:
    """LLM 에 낼 행 — 후보가 하나도 없는 클립은 물어볼 게 없으므로 뺀다."""
    rows = []
    for c in clips:
        starts = start_candidates(c, segs, pitches_of.get(c["scene_id"], []))
        ends = end_candidates(c, segs, utts)
        if starts or ends:
            rows.append({"scene_id": c["scene_id"], "tags": c["tags"],
                         "cs": c["cut"]["cs"], "ce": c["cut"]["ce"],
                         "starts": starts, "ends": ends})
    return rows


_LINE = re.compile(r"장면\s*(\d+)\s*[:：]\s*시작\s*(유지|\d+)\s*끝\s*(유지|\d+)")


def apply(clips: list[dict], rows: list[dict], text: str) -> list[str]:
    """제안 처분 — 제시한 후보의 초 값과 일치할 때만 적용. 그 외는 무시(임의 초 기각)."""
    shown = {r["scene_id"]: r for r in rows}
    by_id = {c["scene_id"]: c for c in clips}
    moved: list[str] = []
    for line in text.splitlines():
        m = _LINE.search(line.strip())
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in shown or sid not in by_id:
            continue
        row, cut_ = shown[sid], by_id[sid]["cut"]
        note = []
        if m.group(2) != "유지":
            new = int(m.group(2))
            if any(int(sec) == new for sec, _ in row["starts"]):
                note.append(f"시작 {cut_['cs']:.0f}→{new}")
                cut_["cs"] = float(new)
            else:
                log.info("bounds 기각: 장면%d 시작 %s (후보 밖)", sid, new)
        if m.group(3) != "유지":
            new = int(m.group(3))
            if any(int(sec) == new for sec, _ in row["ends"]):
                note.append(f"끝 {cut_['ce']:.0f}→{new}")
                cut_["ce"] = float(new)
            else:
                log.info("bounds 기각: 장면%d 끝 %s (후보 밖)", sid, new)
        if note:
            cut_["mode"] += "+경계보정"
            moved.append(f"장면{sid} " + " ".join(note))
    return moved
