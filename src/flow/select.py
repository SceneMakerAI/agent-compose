"""select — 예산 확정 (순수 계산, LLM 무관). 경계가 다 정해진 **뒤에** 자른다.

왜 마지막인가: 예전에는 cutrank 가 경계 확정 전에 예산을 잘랐고, 그 뒤 endfix 가 클립
끝을 최대 12초씩 늘려 예산 보장이 그 자리에서 무효가 됐다 (실측: 900초 요청에
947·949·964·977·1005·1018초).

왜 층인가: 이닝 균형과 득점 보존은 하나의 점수로 동시에 표현되지 않는다. 예산 배선을
고치자 이닝 커버리지가 100%가 되면서 득점 포함이 53%로 떨어진 것이 그 증거다
(v201 comp 3). 어느 쪽으로 프롬프트를 밀어도 반대쪽이 무너지므로, 균형점을 코드가
층으로 강제한다.

층 순서:
  ① 필수   득점 장면 + 결정 라벨 — verify 가 0점을 줘도 유지한다(사실이 소견을 이긴다)
  ② 질의   spec.targets 에 걸리는 클립
  ③ 커버   이닝별 대표 1건씩 라운드로빈 (mode=collection 일 때만)
  ④ 잔여   점수순
자를 때는 verify 점수가 낮은 것부터, 같으면 rank 점수가 낮은 것부터 버린다.
"""

from flow import rank
from log import get_logger

log = get_logger(__name__)

# 사실이 보증하는 장면 — 이 라벨이 붙으면 필수층이다.
MUST_LABELS = frozenset({"역전", "동점", "끝내기", "경기 종료"})
# verify 가 이 점수를 주면 필수층이 아닌 한 넣지 않는다 — 무관한 클립으로 예산을
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


def _inning_key(c: dict) -> str:
    return c.get("inning") or "?"


def choose(clips: list[dict], spec: dict, scores: dict[int, dict]) -> tuple[list, list, int]:
    """
    Summary:
        층 순서로 예산을 채운다.
    Args:
        clips: 경계가 확정된 클립 전부 (cut 포함). spec: plan 명세.
        scores: verify 채점 {scene_id: {score, complete, reason}} — 없으면 빈 dict.
    Returns:
        tuple: (채택, 탈락[(clip, 사유)], 총 길이 초)
    """
    budget = spec["budget"]
    targets = set(spec.get("targets") or [])
    collection = spec.get("mode") == "collection"

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

    # ② 질의 — targets 에 걸리는 것. 0점은 제외한다.
    if targets:
        rest = [c for c in clips if c["scene_id"] not in taken
                and (targets & set(c["tags"]) or targets & set(c["label_list"]))]
        for c in sorted(rest, key=lambda c: (-_score(c, scores), -rank.score(c))):
            if _score(c, scores) <= DROP_SCORE:
                drop(c, "질의와 무관(verify 0점)")
            elif not take(c, "질의"):
                drop(c, "예산 초과")

    # ③ 커버 — 이닝별 대표 1건씩. 빈 이닝을 억지로 채우지는 않는다(후보가 있을 때만).
    if collection:
        by_inn: dict[str, list[dict]] = {}
        for c in clips:
            if c["scene_id"] not in taken and _score(c, scores) > DROP_SCORE:
                by_inn.setdefault(_inning_key(c), []).append(c)
        for inn in sorted(by_inn):
            best = max(by_inn[inn], key=lambda c: (_score(c, scores), rank.score(c)))
            if not take(best, f"커버{inn}"):
                drop(best, "예산 초과")

    # ④ 잔여 — 점수순.
    rest = [c for c in clips if c["scene_id"] not in taken]
    for c in sorted(rest, key=lambda c: (-_score(c, scores), -rank.score(c))):
        if _score(c, scores) <= DROP_SCORE:
            drop(c, "질의와 무관(verify 0점)")
        elif not take(c, "잔여"):
            drop(c, "예산 초과")

    picked.sort(key=lambda c: c["cut"]["cs"])        # 서사 = 시간순
    log.info("select: 채택 %d건 %.0fs/%ds (필수 %d · 탈락 %d)",
             len(picked), total, budget, len(must), len(dropped))
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


__all__ = ["MUST_LABELS", "backfill_note", "choose", "is_must", "rescue_longest"]
