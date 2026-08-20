"""fill_budget 노드의 본체 — select — 예산 확정 (순수 계산, LLM 무관). 경계가 다 정해진 **뒤에** 자른다.

왜 마지막인가: 예전에는 cutrank 가 경계 확정 전에 예산을 잘랐고, 그 뒤 끝 보정이 클립
끝을 최대 12초씩 늘려 예산 보장이 그 자리에서 무효가 됐다 (실측: 900초 요청에
947·949·964·977·1005·1018초).

왜 층인가: 이닝 균형과 득점 보존은 하나의 점수로 동시에 표현되지 않는다. 예산 배선을
고치자 이닝 커버리지가 100%가 되면서 득점 포함이 53%로 떨어진 것이 그 증거다
(v201 comp 3). 어느 쪽으로 프롬프트를 밀어도 반대쪽이 무너지므로, 균형점을 코드가
층으로 강제한다.

층 순서:
  ① 필수   결정 라벨 — score_match 가 0점을 줘도 유지한다(사실이 소견을 이긴다)
  ② 나머지 일치도 → 이닝 분산 → 중요도 (order_rest — 전부 결정적)

**순서도 합산도 코드가 한다** (2026-08-20 — 구 order_clips 폐기). 순위만 LLM 에
맡겼던 판이 있었으나 트레이스 전수(22건)에서 값을 만든 적이 없다:
- 8회 실행 중 4회는 **후보가 1건**이라 줄 세울 것이 없었다 (17~26초씩 소모).
- comp21(후보 13건)·comp34(20건)는 후보가 전부 예산에 들어가 순서가 결과를 가르지
  않았고, comp34 가 62초를 들여 낸 순서는 일치도 내림차순과 **완전히 같았다**(역전 0/19).
- 정작 절단이 일어난 유일한 판(comp33, 후보 60건)에서는 thinking 이 240초를 넘겨
  취소됐다 — 규모가 클수록 실패하니 필요할 때 쓸 수 없는 단계였다.
LLM 이 코드보다 나을 여지는 "이닝 배분"뿐이었는데, 그건 결정적으로 도는 편이 정확하고
단위 테스트로 고정된다.
"""

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
    """사실이 보증하는 장면인가 — 결정 라벨이 붙었다.

    **득점(score_delta > 0)은 조건이 아니다** (방침 2026-08-20). 득점 장면을 전부
    필수로 두니 질의가 무엇이든 그 경기의 모든 득점이 강제 편입돼 선곡을 덮어썼다 —
    v203 comp30 실측: "역전 장면만" 질의에 select_clips 는 역전 1건(장면5)만 골랐는데
    회수가 득점 7건을 끌어와 8클립이 됐다. 득점 여부는 rank.score 의 가중치로 이미
    반영되므로, 무엇을 담을지는 질의를 읽은 LLM 이 정한다.
    """
    return bool(set(c["label_list"]) & MUST_LABELS)


def recover_must(scenes: list[dict], picked: list[int], spec: dict) -> list[int]:
    """
    Summary:
        select_clips 가 놓친 필수 장면 중 **회수할 것**의 scene_id 목록.
    Args:
        scenes (list[dict]): 인벤토리 전체 장면.
        picked (list[int]): select_clips 가 고른 scene_id.
        spec (dict): 질의 해석 결과 — mode·targets 를 읽는다.
    Returns:
        list[int]: 회수 대상 scene_id (없으면 빈 목록).
    Description:
        회수는 "빠짐없이"가 미덕인 질의에서만 한다 (방침 2026-08-20):
        - **pinpoint 는 회수하지 않는다.** 콕 집어 달라는 요청에 다른 장면을 끼워 넣으면
          그건 요청한 물건이 아니다.
        - collection 등에서도 **질의 대상(targets)에 걸린 라벨만** 회수한다. "역전 장면만"
          에 경기 종료가 딸려 들어오던 문제(v203 comp30)가 여기서 갈린다.
        - targets 가 비면(파싱 실패 등) 회수하지 않는다 — 근거 없는 편입을 만들지 않는다.
    """
    targets = set(spec.get("targets") or ())
    if spec.get("mode") == "pinpoint" or not targets:
        return []
    return [r["scene_id"] for r in scenes
            if r["scene_id"] not in picked and is_must(r)
            and set(r["label_list"]) & targets]


def _score(c: dict, scores: dict[int, dict]) -> int:
    return scores.get(c["scene_id"], {}).get("score", DEFAULT_SCORE)


def order_rest(rest: list[dict], must: list[dict], scores: dict[int, dict]) -> list[dict]:
    """
    Summary:
        필수층 외 후보를 담을 순서로 줄 세운다 — 일치도 → 이닝 분산 → 중요도.
    Args:
        rest (list[dict]): 필수가 아닌 후보 클립.
        must (list[dict]): 이미 확정된 필수 클립 (이닝 카운트의 출발점).
        scores (dict): score_match 채점 {scene_id: {score …}}.
    Returns:
        list[dict]: 담을 순서대로 정렬된 rest 사본.
    Description:
        - **일치도가 1순위**다. 질의에 맞는 정도를 이닝 균형이 뒤집으면 안 된다.
        - 같은 일치도 안에서는 **이닝 라운드로빈** — 지금까지 담긴 이닝 수가 가장 적은
          이닝부터 하나씩 가져온다. 필수층이 이미 먹은 이닝은 자연히 뒤로 밀리므로
          "확정에 없는 이닝을 앞으로" 라는 규칙이 카운터 하나로 표현된다.
        - 같은 이닝 안에서는 rank.score(득점·라벨·태그·이닝 가중) 내림차순.
        - 예산이 남아 전부 담기는 판에서는 순서가 결과를 바꾸지 않는다. 이 정렬이
          값을 하는 건 **절단이 실제로 일어날 때**다.
    """
    counts: dict[str, int] = {}
    for c in must:
        key = c.get("inning") or ""
        counts[key] = counts.get(key, 0) + 1

    out: list[dict] = []
    for s in sorted({_score(c, scores) for c in rest}, reverse=True):
        pool = sorted((c for c in rest if _score(c, scores) == s),
                      key=lambda c: -rank.score(c))
        while pool:
            # 담긴 이닝이 가장 적은 쪽을 먼저. 동률이면 pool 순서(=중요도)가 가른다.
            i = min(range(len(pool)), key=lambda i: (counts.get(pool[i].get("inning") or "", 0), i))
            c = pool.pop(i)
            counts[c.get("inning") or ""] = counts.get(c.get("inning") or "", 0) + 1
            out.append(c)
    return out


def choose(clips: list[dict], spec: dict, scores: dict[int, dict]) -> tuple[list, list, int]:
    """
    Summary:
        층 순서로 예산을 채운다 — 필수 → 나머지(일치도·이닝 분산·중요도).
    Args:
        clips: 경계가 확정된 클립 전부 (cut 포함). spec: plan 명세.
        scores: score_match 채점 {scene_id: {score, complete, reason}} — 없으면 빈 dict.
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

    # ② 나머지 — 일치도 → 이닝 분산 → 중요도 (결정적).
    rest = order_rest([c for c in clips if c["scene_id"] not in taken], must, scores)
    for c in rest:
        if not take(c, "순위"):
            drop(c, "예산 초과")

    picked.sort(key=lambda c: c["cut"]["cs"])        # 서사 = 시간순
    log.info("fill_budget: 채택 %d건 %.0fs/%ds (필수 %d · 나머지 %d · 탈락 %d)",
             len(picked), total, budget, len(must), len(rest), len(dropped))
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


__all__ = ["MUST_LABELS", "choose", "is_must", "order_rest", "recover_must", "rescue_longest"]
