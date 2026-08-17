"""cut — 샷 레시피로 클립 세부 구간 확정 (규칙, LLM 무관 — bench4 compose/cut.py 이식).

앵커(투구 샷) 선택 체인:
① ★pitch_sec 이 속한 샷 (유형 무관 — 긴 혼합 샷의 중간 프레임 오분류 대비)
② 구간 안 첫 '투구' 샷 (pitch_sec 없는 시작 보정 건은 첫 샷이 투구)
③ 둘 다 없으면 클립 통째 (FULL 폴백 = 완화 L2 내장 — 함부로 좁히지 않는다)

앵커부터 레시피 유형이 이어지는 동안 샷을 붙이고, 벗어나는 첫 샷에서 멈춘다.
끝 확정 후 대사 꼬리 스냅: 끝에 걸친 해설 발화는 상한 안에서 발화 끝까지 연장.
입력(r·segs·utts)은 읽기만 한다 — 결과는 새 dict (B4 불변 규약).
"""

import math

from flow import vocab
from log import get_logger

log = get_logger(__name__)


def clip(r: dict, segs: list[dict], utts: list[tuple[float, float, str]] = ()) -> dict:
    """t_scene 행 1건 → {cs, ce, shots:[(s,e,type)], anchor, mode}."""
    lo, hi = r["s"], r["e"]
    shots = [s for s in segs if s["s"] < hi and s["e"] > lo]
    if not shots or set(r["tags"]) & vocab.FULL_CLIP_TAGS:
        return _full(r, shots, "통째(레시피 제외 태그)" if shots else "통째(샷 없음)", utts)

    anchor = None
    if r["pitch_sec"] is not None:
        anchor = next((s for s in shots if s["s"] <= r["pitch_sec"] < s["e"]), None)
    if anchor is None:
        anchor = next((s for s in shots if s["shot_type"] == "투구"), None)
    if anchor is None:
        return _full(r, shots, "통째(투구 앵커 없음 — L2 폴백)", utts)

    allow = set().union(*(vocab.CUT_RECIPE.get(t, set()) for t in r["tags"]))
    # 큰 플레이(병살·적시타 등)는 라벨 기준 추가 유형까지 — 단 초 상한 (리플레이 차단)
    extra = set().union(*(vocab.LABEL_EXTRA_SHOTS.get(lab, set()) for lab in r["label_list"]),
                        set()) - allow
    picked = [anchor]
    extra_used = 0
    rest = shots[shots.index(anchor) + 1:]
    for i, s in enumerate(rest):
        if s["shot_type"] in allow:
            picked.append(s)
        elif (s["shot_type"] in extra
              and extra_used + s["e"] - s["s"] <= vocab.LABEL_EXTRA_MAX_SEC):
            picked.append(s)
            extra_used += s["e"] - s["s"]
        elif (s["shot_type"] == "투구"
              and i + 1 < len(rest) and rest[i + 1]["shot_type"] in allow):
            # 앵커 연장 — 중계 전경이 '투구'로 분류돼 체인이 끊기는 실측(장면 60:
            # 전경 샷 하나 때문에 다이빙 캐치가 클립에서 빠짐). 바로 뒤가 레시피
            # 샷일 때만 흡수 — 삼진처럼 레시피가 빈 태그는 조건이 성립하지 않아
            # 꽉 조임이 유지된다.
            picked.append(s)
        else:
            break
    cs, ce = max(picked[0]["s"], lo), min(picked[-1]["e"], hi)
    ce, snapped = _snap_tail(ce, utts)
    return {"cs": cs, "ce": ce, "anchor": (anchor["s"], anchor["e"]),
            "shots": [(s["s"], s["e"], s["shot_type"]) for s in picked],
            "mode": "레시피" + ("+대사꼬리" if snapped else "")}


def _full(r: dict, shots: list[dict], why: str, utts=()) -> dict:
    ce, snapped = _snap_tail(r["e"], utts)
    return {"cs": r["s"], "ce": ce, "anchor": None,
            "shots": [(s["s"], s["e"], s["shot_type"]) for s in shots],
            "mode": why + ("+대사꼬리" if snapped else "")}


def _snap_tail(ce: float, utts) -> tuple[float, bool]:
    """끝에 걸친(us ≤ ce < ue) 발화가 상한 안에 끝나면 발화 끝(올림)까지 연장.

    us == ce(끝과 동시 시작)도 포함 — 이벤트 설명이 딱 그 순간 시작하는 실측
    (장면 35) 대응. 연장은 1회 스캔 — 다음 발화로 재연쇄하지 않는다 (기어 나감 방지).
    """
    best = ce
    for us, ue, _ in utts:
        if us <= ce < ue and ue - ce <= vocab.DIALOGUE_TAIL_MAX_SEC:
            best = max(best, math.ceil(ue))
    return best, best != ce
