"""ComposeState + Inventory — 그래프 상태 스키마와 불변 규약 (B4 수정의 핵심).

bench4 는 인벤토리(scenes·segs·utts)를 노드 클로저로 들고 cutrank·endfix 가
같은 dict 를 in-place 수정했다 — 요청마다 새로 fetch 해 우연히 안전했을 뿐,
캐시·동시 요청 도입 즉시 깨진다. 신규 규약:

- **Inventory 는 요청 시작에 1회 만들어 불변으로 취급** — 노드는 절대 수정하지 않는다.
- 노드가 상태에 넣는 행은 전부 **복사본** (rank.order 가 복사, backfill 도 복사).
- 상태 갱신은 전부 교체 의미(reducer 없음) — run 이 stream 델타를 누적해 최종 상태.
- LLM·Milvus·DB 클라이언트는 상태에 넣지 않는다 (직렬화 불가물 배제 —
  향후 checkpointer 도입 여지). 자원은 graph 빌드 시 주입.
"""

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class Inventory:
    """요청 시점 DB 스냅샷 — 필드는 읽기 전용으로만 쓴다 (행 수정 금지)."""

    v_id: int
    scenes: tuple[dict, ...]                  # t_scene_baseball 발행 행 (s·e·tags·label_list …)
    segs: tuple[dict, ...]                    # scene-cut 샷 (s·e·shot_type)
    utts: tuple[tuple[float, float, str], ...]  # STT (s, e, text) 시간순
    game_line: str                            # "v_id=201  삼성(원정) vs 롯데(홈)"
    inventory_text: str                       # plan 이 보는 목록 렌더 (요청당 1회)


class ComposeState(TypedDict, total=False):
    query: str
    budget: int | None       # 명시 입력 (질의 해석보다 우선)
    inv: Inventory           # 불변 스냅샷 (위 규약)
    evidence: list[dict]     # retrieve 벡터 후보 [{scene_id,hits,sim,snippets}]
    evidence_orphan: list[dict]  # 장면밖 증거 (발행 누락 의심 — 리포트 표기 전용)
    feedback: str            # 재선곡 사유 (feedback 노드가 작성)
    attempt: int             # plan 호출 횟수
    spec: dict               # plan 명세 (mode/targets/view/budget/picked)
    picked: list[dict]       # 채택 클립 (cut 결과 포함 — 전부 복사본 행)
    spare: list[dict]        # 예산 초과 예비 풀
    total: int               # 채택분 컷 후 총 길이(초)
    endfix_moved: list[str]  # 끝 이동 기록 (리포트용)
    suspicions: list[tuple]  # verify 의심 [(scene_id, 사유)]
    status: str              # ok | empty
