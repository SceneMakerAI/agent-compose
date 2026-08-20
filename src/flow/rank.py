"""order_clips 의 폴백 정렬 + 인벤토리 복사 — rank — 중요도 계산·정렬 (순수 계산, LLM 무관 — bench4 compose/rank.py 이식).

예산 절단은 여기서 하지 않는다 — set_bounds 가 클립을 줄인 뒤 assemble 이 실제 EDL
길이로 채택/충원한다 (미리 자르면 예산 미달). 여기서는 순서와 예비 풀만 만든다.

bench4 와 다른 점 (B4 불변화): order 가 **행 복사본**을 돌려준다 — 원본 인벤토리
dict 를 하류 노드가 절대 건드리지 않게 하는 서비스 동시성 전제.
"""

from flow import vocab


def score(r: dict) -> int:
    """장면 중요도 — 보드 사실(델타·라벨·태그·이닝)만으로 결정적 계산."""
    s = r["score_delta"] * vocab.RANK_SCORE_DELTA_W
    s += sum(vocab.RANK_LABEL_BONUS.get(lab, 0) for lab in r["label_list"])
    s += max(vocab.RANK_TAG_BONUS.get(t, 0) for t in r["tags"])
    inning_no = int(r["inning"].split("회")[0]) if r["inning"] else 1
    s += inning_no // 3                      # 이닝 가중 — 후반일수록 +1~+4
    return s


def order(scenes: list[dict], picked: list[int]) -> list[dict]:
    """선곡분을 중요도 내림차순으로 (동점은 시간순) — 행은 복사본."""
    by_id = {r["scene_id"]: r for r in scenes}
    rows = [dict(by_id[i]) for i in picked if i in by_id]
    return sorted(rows, key=lambda r: (-score(r), r["s"]))
