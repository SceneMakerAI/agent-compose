"""fill_budget 노드의 본체 — select — 예산 확정 (순수 계산, LLM 무관). 경계가 다 정해진 **뒤에** 자른다.

왜 마지막인가: 예전에는 cutrank 가 경계 확정 전에 예산을 잘랐고, 그 뒤 끝 보정이 클립
끝을 최대 12초씩 늘려 예산 보장이 그 자리에서 무효가 됐다 (실측: 900초 요청에
947·949·964·977·1005·1018초).

왜 층인가: 이닝 균형과 득점 보존은 하나의 점수로 동시에 표현되지 않는다. 예산 배선을
고치자 이닝 커버리지가 100%가 되면서 득점 포함이 53%로 떨어진 것이 그 증거다
(v201 comp 3). 어느 쪽으로 프롬프트를 밀어도 반대쪽이 무너지므로, 균형점을 코드가
층으로 강제한다.

층 순서 (2026-08-20 재편 — 넷에서 둘로):
  ① 필수   득점 장면 + 결정 라벨 — score_match 가 0점을 줘도 유지한다(사실이 소견을 이긴다)
  ② 나머지 rank 노드가 준 **LLM 우선순위** 순서대로. 순서가 없으면 점수순 폴백.

구 ②질의·③커버·④잔여는 폐기했다. 층 순서와 규칙이 코드에 고정돼 있어
"이닝별로 1개씩은 꼭 넣어줘" 같은 질의별 요구가 도달할 통로가 없었다 — spec 에
담기지도 않고 select 는 mode 와 targets 만 봤다. v201 comp9 에서는 ①이 예산을 다
써서 ③커버층이 한 번도 돌지 않았다(질의가 "이닝별"인데도).

**순위는 LLM, 합산은 코드.** 예산 절단은 산술이고 LLM 은 산술을 못 한다
(실측: 900초 요청에 947~1018초, 득점 포함률 53%). LLM 이 내놓는 건 최종 선곡이
아니라 줄 세운 번호라, 비결정성은 "어느 클립이 먼저냐"까지만 번진다.
"""

import re

from flow import rank
from log import get_logger

log = get_logger(__name__)

# 사실이 보증하는 장면 — 이 라벨이 붙으면 필수층이다.
MUST_LABELS = frozenset({"역전", "동점", "끝내기", "경기 종료"})
# score_match 가 이 점수를 주면 필수층이 아닌 한 넣지 않는다 — 무관한 클립으로 예산을
# 채우는 건 채운 게 아니라 희석하는 것이다.
DROP_SCORE = 0
# 점수가 없는 클립(파싱 실패·verify 미실행)의 기본값 — 중립.
DEFAULT_SCORE = 2
# 필수층이 예산을 넘길 수 있는 한도 (방침 2026-08-20). 하이라이트에서 득점 장면이
# 빠지는 건 취향이 아니라 결함이라 예산을 넘겨서라도 담되, 무한정은 아니다 —
# 900초 요청에 1170초까지가 오차범위. 그 위로는 요청한 물건이 아니게 된다.
MUST_BUDGET_SLACK = 0.30


def _dur(c: dict) -> float:
    return c["cut"]["ce"] - c["cut"]["cs"]


def is_must(c: dict) -> bool:
    """사실이 보증하는 장면인가 — 득점이 났거나 결정 라벨이 붙었다."""
    return c["score_delta"] > 0 or bool(set(c["label_list"]) & MUST_LABELS)


def _score(c: dict, scores: dict[int, dict]) -> int:
    return scores.get(c["scene_id"], {}).get("score", DEFAULT_SCORE)


def parse_order(text: str, valid: set[int]) -> list[int]:
    """rank 응답 → 번호 순서. 실존 scene_id 만, 중복 제거.

    모델이 빠뜨린 번호는 호출부가 뒤에 붙인다 — 순서를 안 준 건 "덜 중요하다"는
    뜻이지 "빼라"는 뜻이 아니다.
    """
    m = re.search(r"순서\s*[:：]\s*(.+)", text)
    body = m.group(1) if m else text
    out: list[int] = []
    for tok in re.findall(r"\d+", body):
        n = int(tok)
        if n in valid and n not in out:
            out.append(n)
    return out


def choose(clips: list[dict], spec: dict, scores: dict[int, dict],
           order: list[int] | None = None) -> tuple[list, list, int]:
    """
    Summary:
        층 순서로 예산을 채운다 — 필수(코드) → 나머지(LLM 순위).
    Args:
        clips: 경계가 확정된 클립 전부 (cut 포함). spec: plan 명세.
        scores: verify 채점 {scene_id: {score, complete, reason}} — 없으면 빈 dict.
        order: order_clips 가 준 scene_id 순서. None 이면 점수순 폴백.
    Returns:
        tuple: (채택, 탈락[(clip, 사유)], 총 길이 초)
    """
    budget = spec["budget"]

    picked: list[dict] = []
    total = 0.0
    # 처리 완료 = 채택됐거나 탈락했거나. 뒷 층은 여기 없는 것만 본다 — 탈락을 표시하지
    # 않으면 ②에서 떨어진 클립을 ④가 또 보고 같은 사유로 다시 떨어뜨린다(실측 v201
    # comp9: 탈락 24건이 실은 12건 × 2). 누적은 늘기만 하므로 앞 층에서 예산에 못 든
    # 클립이 뒷 층에서 들어갈 일도 없다.
    taken: set[int] = set()
    dropped: list[tuple[dict, str]] = []

    def take(c: dict, why: str, force: bool = False) -> bool:
        nonlocal total
        d = _dur(c)
        if not force and total + d > budget:
            return False
        picked.append(c)
        taken.add(c["scene_id"])
        total += d
        log.debug("select 채택[%s] 장면%d %.0fs (누적 %.0f/%d)", why, c["scene_id"], d, total, budget)
        return True

    def drop(c: dict, why: str) -> None:
        taken.add(c["scene_id"])
        dropped.append((c, why))

    # ① 필수 — 사실이 보증하는 장면. 점수 무관, 득점 큰 순.
    # **예산을 넘겨서라도 담되 +30%까지** (방침 확정 2026-08-20): 득점 장면 누락은
    # 취향이 아니라 결함이라 예산 준수보다 우선한다. 다만 무한정 넘기면 요청한
    # 물건이 아니게 되므로 오차범위를 상한으로 둔다. 상한에 걸려 떨어지는 건 득점이
    # 작고 순위가 낮은 뒤쪽부터다.
    must_cap = budget * (1 + MUST_BUDGET_SLACK)
    must = sorted((c for c in clips if is_must(c)),
                  key=lambda c: (-c["score_delta"], -rank.score(c)))
    for c in must:
        if total + _dur(c) > must_cap:
            drop(c, f"필수지만 허용 상한 초과(+{MUST_BUDGET_SLACK:.0%})")
            continue
        take(c, "필수", force=True)
    if total > budget:
        log.info("필수층이 예산 초과 %.0fs/%ds (상한 %.0fs · %d건 중 %d건 채택) — "
                 "품질 우선이라 그대로 담는다",
                 total, budget, must_cap, len(must), len(picked))

    # ② 나머지 — order_clips 가 준 순서대로. 순서가 없으면(콜 실패·미실행) 점수순 폴백.
    rest = [c for c in clips if c["scene_id"] not in taken]
    if order:
        pos = {sid: i for i, sid in enumerate(order)}
        # 순위에 없는 클립은 뒤로 — 모델이 빠뜨린 것이지 버리라는 뜻이 아니다.
        rest.sort(key=lambda c: (pos.get(c["scene_id"], len(pos)), -_score(c, scores)))
    else:
        rest.sort(key=lambda c: (-_score(c, scores), -rank.score(c)))
    for c in rest:
        if not take(c, "순위" if order else "점수"):
            drop(c, "예산 초과")

    picked.sort(key=lambda c: c["cut"]["cs"])        # 서사 = 시간순
    log.info("select: 채택 %d건 %.0fs/%ds (필수 %d · 순위 %s · 탈락 %d)",
             len(picked), total, budget, len(must),
             "LLM" if order else "점수순 폴백", len(dropped))
    return picked, dropped, int(total)


def rescue_longest(clips: list[dict], budget: int) -> list[dict]:
    """전부 예산보다 길어 0건이 된 경우 — 가장 짧은 한 건은 넘겨서라도 담는다.

    실측(comp 1): v202 "홈런 모음" 예산 60초인데 그 경기 홈런은 74초짜리 한 건뿐이라
    0클립 empty 가 나갔다. 사용자에겐 "조건에 맞는 장면 없음"으로 보였지만 홈런은 있었다.
    """
    if not clips:
        return []
    best = min(clips, key=_dur)
    log.warning("전 클립이 예산 초과 — 최단 1건만 담는다: 장면%d %.0fs > %ds",
                best["scene_id"], _dur(best), budget)
    return [best]


def backfill_note(spec: dict) -> str:
    """충원이 왜 일어나지 않았는지 — 무동작이 로그에 안 남던 문제(audit 3-3) 대응."""
    if spec.get("view") != "전체":
        return f"관점={spec['view']} 이라 충원 생략 (홈/원정 기계 판별 불가)"
    if not spec.get("targets"):
        return "대상이 비어 충원 생략"
    return ""


__all__ = ["MUST_LABELS", "backfill_note", "choose", "is_must", "parse_order",
           "rescue_longest"]
