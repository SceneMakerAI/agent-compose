"""ComposeState — 야구 편성 그래프의 상태 스키마와 불변 규약.

- 인벤토리(scenes)는 load_inventory 가 1회 채우는 불변 스냅샷 — Scene 이 frozen
  dataclass 라 노드가 행을 수정할 수 없다. 중간 산출은 별도 키(candidates 등)에
  새 객체로 담는다.
- 상태 갱신은 전부 교체 의미(reducer 없음) — run 이 stream 델타를 누적해 최종 상태.
- DB·LLM 클라이언트는 상태에 넣지 않는다 (직렬화 불가물 배제) — 자원은 그래프
  빌드 시 클로저로 주입한다.
"""

from typing import TypedDict

from domains.baseball.repo.scenes import Scene


class ComposeState(TypedDict, total=False):
    # --- 입력 ---
    v_id: int
    query: str               # 사용자 질의 원문
    budget_sec: int | None   # 목표 분량(초) — 마감 단계 덜어내기 전용
    trace: object            # InferTraceLog | None — LLM 콜 기록기 (직렬화 대상 아님)

    # --- 노드 산출 (그래프 확장 시 키 추가) ---
    scenes: list[Scene]      # load_inventory — 인벤토리 불변 스냅샷 (scene_no 순)
    spec: dict               # parse_query — 필터 스펙. innings·teams(+view 관점) 는
                             # 좁히는 축(AND), labels·board_tags 는 넓게 모으는 축(OR 합침).
                             # phrases 는 벡터 검색어. 빈 목록 = 그 축 필터 안 함
    evidence: list[dict]     # retrieve_evidence — 구간 귀속 검색 결과
                             # [{scene_no, hits, sim, by_kind, snippets}] (히트수→유사도 순)
    evidence_orphan: int     # 어느 구간에도 안 겹친 히트 수 (색인 누락 의심 신호)
    evidence_hits: list[dict]  # retrieve_evidence — dedup 된 히트 원본 (kind·text·distance…)
                               # select_clips 의 [질의 유사] 표기 근거 (응답에는 미노출)
    candidates: list[int]    # select_clips — 필터 통과 후보 scene_no (관측용)
    picked: list[int]        # select_clips — 선곡 확정 scene_no (**중요도 내림차순** —
                             # trim_budget 의 예산 덜어내기 근거. 시간순 아님)
    clips: list[dict]        # select_end_point 좌표 확정 → trim_budget 이 예산 절단한 최종.
                             # [{scene_no, start, end, sec, rank, start_from, end_from}] 시간순
                             # rank = 선곡 중요도 순위 (1이 가장 중요)
    dropped: list[str]       # trim_budget — 예산 절단으로 버린 클립 기록 (관측·응답용)
    status: str              # ok | empty
