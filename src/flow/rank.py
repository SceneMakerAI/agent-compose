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
