"""ComposeState + Inventory — 그래프 상태 스키마와 불변 규약 (B4 수정의 핵심).

bench4 는 인벤토리(scenes·segs·utts)를 노드 클로저로 들고 구 cutrank·endfix 가
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
    # 검출 투구는 장면 행(`scenes[i]["pitches"]`)이 자기 것만 들고 있다 —
    # 경기 전량을 인벤토리에 따로 싣던 `pitches` 필드는 폐기 (2026-08-24,
    # bounds 모듈 docstring: 다른 타석 투구가 시작 후보로 새던 자리).
    # `trans`(전광판 전이)는 없앴다 — score_match 폐기로 읽는 코드가 사라졌고,
    # 그 사이 상류에서 t_transition_baseball 테이블 자체가 삭제됐다 (2026-08-23).


class ComposeState(TypedDict, total=False):
    query: str
    inv: Inventory           # 불변 스냅샷 (위 규약)
    evidence: list[dict]     # retrieve_evidence 벡터 후보 [{scene_id,hits,sim,snippets}]
    evidence_orphan: list[dict]  # 장면밖 증거 (발행 누락 의심 — 리포트 표기 전용)
    feedback: str            # 재선곡 사유 (retry_select 가 작성)
    attempt: int             # select_clips 호출 횟수
    spec: dict               # select_clips 명세 (mode/targets/view/budget/picked)
    picked: list[dict]       # 채택 클립 (cut 결과 포함 — 전부 복사본 행)
    total: float             # 확정분 총 길이(초)
    end_moved: list[str]     # 끝 이동 기록 (refine_end_bound)
    start_moved: list[str]   # 시작 이동 기록 (refine_start_bound)
    status: str              # ok | empty
    budget: int | None       # 목표 분량(초) — finish 가 rank 순으로 절단할 때만 쓴다.
                             # 선곡에는 관여하지 않는다: 예산으로 장면을 끌어오는 통로는
                             # 폐기된 fill_budget 이 그랬고, 그게 질의를 규칙이 덮어쓰는
                             # 자리였다(94b58dc). 여기 예산은 **덜어내기 전용**이다.
    dropped: list[str]       # 예산 절단으로 버린 클립 기록 (리포트·트레이스용)
    # --- 재배선(2026-08-20) 추가 ---
    phrases: list[str]       # rephrase_query 검색어 (없으면 원 질의 하나)
    filters: list[str]       # rephrase_query 메타 필터 힌트 (태그·라벨)
    clips: list[dict]        # 컷·보정 중인 클립 (finish 가 시간순으로 확정)
    trace: object            # Trace | None — 노드·LLM 입출력 수집기 (직렬화 대상 아님)
