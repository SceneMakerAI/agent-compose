"""장면 중요도 점수 + 인벤토리 복사 — rank — (순수 계산, LLM 무관 — bench4 compose/rank.py 이식).

score 는 fill_budget 의 정렬(select.order_rest)이 같은 이닝 안에서 우열을 가릴 때 쓴다.

예산 절단은 여기서 하지 않는다 — set_bounds 가 클립을 줄인 뒤 assemble 이 실제 EDL
길이로 채택/충원한다 (미리 자르면 예산 미달). 여기서는 순서와 예비 풀만 만든다.

bench4 와 다른 점 (B4 불변화): order 가 **행 복사본**을 돌려준다 — 원본 인벤토리
dict 를 하류 노드가 절대 건드리지 않게 하는 서비스 동시성 전제.
"""

from flow import vocab


def score(r: dict) -> int:
    """장면 중요도 — 보드 사실(델타·라벨·태그·판세·이닝)만으로 결정적 계산.

    태그 가산은 **최대 하나**다 (합이 아니다 — 복수 태그 장면이 태그 수로 이기지
    않게). 태그가 없는 장면이 있다: 상류 해석(labels)이 비면 `tags` 가 빈 목록이라
    default=0 이 필요하다 — 구 스키마에선 `(scene_type or "").split(",")` 이 `[""]`
    를 돌려줘 우연히 비지 않았을 뿐이고, 실측 v200~203 에 그런 행이 12개 있다.

    판세 가산(game_context)은 **단일값 조회**다 — 상류가 우선순위대로 하나만 붙인다.
    """
    s = r["score_delta"] * vocab.RANK_SCORE_DELTA_W
    s += sum(vocab.RANK_LABEL_BONUS.get(lab, 0) for lab in r["label_list"])
    s += max((vocab.RANK_TAG_BONUS.get(t, 0) for t in r["tags"]), default=0)
    s += vocab.RANK_CONTEXT_BONUS.get(r.get("game_context") or "", 0)
    inning_no = int(r["inning"].split("회")[0]) if r["inning"] else 1
    s += inning_no // 3                      # 이닝 가중 — 후반일수록 +1~+4
    return s


def order(scenes: list[dict], picked: list[int]) -> list[dict]:
    """선곡분을 중요도 내림차순으로 (동점은 시간순) — 행은 복사본."""
    by_id = {r["scene_id"]: r for r in scenes}
    rows = [dict(by_id[i]) for i in picked if i in by_id]
    return sorted(rows, key=lambda r: (-score(r), r["s"]))


def fit_budget(clips: list[dict], budget: int | None) -> tuple[list[dict], list[str]]:
    """
    Summary:
        예산(초)에 맞춰 클립을 덜어낸다 — (남길 것 시간순, 버린 기록).
    Args:
        clips (list[dict]): cut 좌표가 붙은 클립들.
        budget (int | None): 목표 분량(초). None·0 이면 절단하지 않는다.
    Returns:
        tuple: (시간순 클립, "장면N(길이,점수)" 기록 목록).
    Description:
        - **덜어내기만 한다.** 모자라도 채우지 않는다 — 예산을 채우려고 선곡에 없던
          장면을 끌어오는 건 폐기된 fill_budget 이 하던 일이고, 그게 "질의를 규칙이
          덮어쓰는 통로"였다(94b58dc 폐기 사유).
        - 버리는 순서는 score 오름차순(= 담는 순서가 내림차순)이다. 질의 의도는 이미
          select_clips 가 걸렀으니 여기 남은 판단은 "그중 무엇이 더 큰 플레이인가"뿐.
        - **최소 1건은 남긴다** — 첫 클립이 예산보다 길어도 빈 편성을 내지 않는다.
        - 절단은 중요도 순으로 하되 **반환은 시간순**이다: 편성은 경기 흐름대로 돈다.
    """
    dropped: list[str] = []
    if budget and clips:
        kept: list[dict] = []
        used = 0.0
        for c in sorted(clips, key=lambda x: (-score(x), x["cut"]["cs"])):
            length = c["cut"]["ce"] - c["cut"]["cs"]
            if used + length <= budget or not kept:
                kept.append(c)
                used += length
            else:
                dropped.append(f"장면{c['scene_id']}({length:.0f}s,점수{score(c)})")
        clips = kept
    return sorted(clips, key=lambda c: c["cut"]["cs"]), dropped
