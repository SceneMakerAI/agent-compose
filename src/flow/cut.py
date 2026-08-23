"""set_bounds 노드의 본체 — 샷 레시피로 클립 구간 확정 (규칙, LLM 무관).

앵커(투구 샷) 선택 체인:
① ★pitch_sec 이 속한 샷 (유형 무관 — 긴 혼합 샷의 중간 프레임 오분류 대비)
② 구간 안 첫 '투구' 샷 (pitch_sec 없는 시작 보정 건은 첫 샷이 투구)
③ 둘 다 없으면 클립 통째 (FULL 폴백 = 완화 L2 내장 — 함부로 좁히지 않는다)

**시작과 끝의 정책을 분리한다** (2026-08-20). FULL_CLIP_TAGS(홈런·비디오 판독)는
"레시피로 **끝**을 좁히지 마라"는 뜻이다 — 홈런의 베이스 도는 장면·더그아웃 리액션은
레시피 유형에 없어 체인이 끊긴다. 그런데 그 태그가 **시작**까지 포기시켜 앵커 없는
통째 클립이 됐고, 그 클립이 refine_bounds 로 넘어가 LLM 이 시작을 다시 찾고 있었다 —
코드가 결정적으로 답할 수 있는 질문을 모델에게 물은 셈이다.

실측(5경기, 앵커 없이 refine_bounds 로 가던 16건): 10건은 구간 안에 '투구' 샷이 멀쩡히
있었고, 그 위치는 8건이 장면 시작과 정확히 일치했다(vision3 가 이미 맞춘 것).
나머지 1건이 v202 장면11 — 장면 2558s 인데 첫 투구 샷은 2583s 로, refine_bounds 를 만든
원인이던 바로 그 케이스가 규칙 하나로 풀린다. 남는 6건은 '투구' 샷 자체가 없는
진짜 미해결이다.

앵커부터 레시피 유형이 이어지는 동안 샷을 붙이고, 벗어나는 첫 샷에서 멈춘다.
끝 확정 후 대사 꼬리 스냅: 끝에 걸친 해설 발화는 상한 안에서 발화 끝까지 연장.
입력(r·segs·utts)은 읽기만 한다 — 결과는 새 dict (B4 불변 규약).
"""

import math

from flow import vocab
from log import get_logger

log = get_logger(__name__)


def _anchor(r: dict, shots: list[dict]) -> dict | None:
    """앵커 샷 — ★pitch_sec 이 속한 샷, 없으면 구간 안 첫 '투구' 샷 (모듈 docstring ①②).

    앵커가 잡혔다고 그 시작이 투구인 건 아니다. 상류가 고른 대표 투구(pitch_time)가
    든 샷을 그대로 쓰므로, 그 샷의 유형이 '투구'가 아닐 수 있다 — 실측 406장면 중
    61%(리액션 219·타구/수비 16·광고 4·기타 4·주루 3·미분류 1)가 그렇다. 그래서
    호출자가 판단할 수 있게 `anchor_type` 을 함께 낸다 (bounds.start_rows 의 게이트).
    """
    anchor = None
    if r["pitch_sec"] is not None:
        anchor = next((s for s in shots if s["s"] <= r["pitch_sec"] < s["e"]), None)
        if anchor is not None and anchor["shot_type"] is None:
            # 앵커 교정 — pitch_sec 이 장면 경계의 미분류 부스러기 샷 꼬리에 걸리는 실측
            # (v203 장면 6: 0.2초 볼넷 클립). 바로 뒤 '투구' 샷이 실제 투구다.
            nxt = next((s for s in shots[shots.index(anchor) + 1:]
                        if s["shot_type"] == "투구"), None)
            if nxt is not None:
                anchor = nxt
    if anchor is None:
        anchor = next((s for s in shots if s["shot_type"] == "투구"), None)
    return anchor


def clip(r: dict, segs: list[dict], utts: list[tuple[float, float, str]] = ()) -> dict:
    """t_scene_baseball 행 1건 → {cs, ce, shots:[(s,e,type)], anchor, mode}."""
    lo, hi = r["s"], r["e"]
    shots = [s for s in segs if s["s"] < hi and s["e"] > lo]
    if not shots:
        return _full(r, shots, "통째(샷 없음)", utts)

    anchor = _anchor(r, shots)
    full_tag = bool(set(r["tags"]) & vocab.FULL_CLIP_TAGS)
    if anchor is None:
        return _full(r, shots,
                     "통째(레시피 제외 태그)" if full_tag else "통째(투구 앵커 없음 — L2 폴백)",
                     utts)
    if full_tag:
        # 끝은 장면 그대로(레시피로 좁히지 않는다), 시작만 앵커로 — 정책 분리.
        picked = shots[shots.index(anchor):]
        ce, snapped = _snap_tail(hi, utts)
        return {"cs": max(anchor["s"], lo), "ce": ce, "anchor": (anchor["s"], anchor["e"]),
                "anchor_type": anchor["shot_type"],
                "shots": [(s["s"], s["e"], s["shot_type"]) for s in picked],
                "mode": "끝 통째(레시피 제외 태그)" + ("+대사꼬리" if snapped else "")}

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
    ce, floored = _obs_floor(ce, r.get("obs_sec"), shots, hi)
    ce, snapped = _snap_tail(ce, utts)
    return {"cs": cs, "ce": ce, "anchor": (anchor["s"], anchor["e"]),
            "anchor_type": anchor["shot_type"],
            "shots": [(s["s"], s["e"], s["shot_type"]) for s in picked],
            "mode": "레시피" + ("+관측하한" if floored else "")
                    + ("+대사꼬리" if snapped else "")}


def _obs_floor(ce: float, obs: float | None, shots: list[dict],
               hi: float) -> tuple[float, bool]:
    """관측 하한 — 컷은 전이 원장이 보증하는 결과 시점(obs_sec) 이전에 끝날 수 없다.

    레시피 체인은 샷 오분류 하나로 결과 전에 끊긴다(실측 v203 도루: '주루' 샷이
    없어 도루 실물·리플레이가 잘림). 원장 경계 [투구→관측]은 결정적이므로,
    끝을 관측이 속한 샷의 끝(장면 끝 상한)까지 보장한다 — 시간 상수 없음.
    """
    if obs is None or ce >= obs:
        return ce, False
    osh = next((x for x in shots if x["s"] <= obs < x["e"]), None)
    floor = min(osh["e"] if osh else obs, hi)
    return (floor, True) if floor > ce else (ce, False)


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
