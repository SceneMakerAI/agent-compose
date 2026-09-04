"""trim_budget 노드 — 목표 분량(budget_sec)에 맞춰 클립을 덜어낸다 (순수 계산, LLM 무관).

목표 분량의 출처는 둘이다: API 파라미터(요청)와 질의 문구("5분짜리로" — parse_query 가
읽어 spec 에 담는다). **API 파라미터가 우선**이다 — 호출자가 명시한 값이 질의 문구에
덮이면, UI 가 지정한 분량이 사용자가 흘려 쓴 표현에 밀린다. 둘 다 없으면 절단하지 않는다.

**덜어내기만 한다.** 모자라도 채우지 않는다 — 예산을 채우려고 선곡에 없던 구간을
끌어오는 통로는 열지 않는다 (질의를 규칙이 덮어쓰게 되는 자리 — 설계 결정).

버리는 순서는 선곡 중요도(rank)의 꼬리부터다 — select_clips 가 매긴 중요도
내림차순으로 담다가 허용 상한(예산 × (1+여유율), BUDGET_MARGIN 설정)이 넘치면
나머지를 버린다. 산수는 코드가 한다 (LLM 에게 분량 계산을 시키지 않는다).
확정 반환은 시간순 — 재생은 경기 흐름대로.
"""

from domains.baseball.graph.state import ComposeState
from log import get_logger

log = get_logger(__name__)


def make_node(budget_margin: float):
    """팩토리 — 순수 계산이라 자원이 없다. budget_margin 은 예산 여유율 (설정)."""

    async def trim_budget(st: ComposeState) -> dict:
        """rank 순으로 담다가 허용 상한 초과분을 버린다 — 목표 분량 없으면 통과."""
        clips = st.get("clips") or []
        # API 파라미터 우선, 없을 때만 질의에서 읽은 분량으로 폴백
        budget = st.get("budget_sec")
        source = "요청"
        if not budget:
            budget = (st.get("spec") or {}).get("budget_sec")
            source = "질의"
        trace = st.get("trace")

        if not budget or not clips:
            return {"clips": clips, "dropped": []}

        # 허용 상한 — 예산에 여유율을 얹는다 (예산 이하로 끝나기보다 목표 분량을 채운다)
        cap = round(budget * (1 + budget_margin))

        # 중요도(rank) 순으로 담는다 — 최소 1건은 남긴다 (첫 클립이 상한보다 길어도
        # 빈 편성을 내지 않는다)
        kept = []
        dropped = []
        used = 0
        for clip in sorted(clips, key=lambda c: c["rank"]):
            if not kept or used + clip["sec"] <= cap:
                kept.append(clip)
                used += clip["sec"]
            else:
                dropped.append(f"scene{clip['scene_no']}(rank{clip['rank']},{clip['sec']}s)")

        # 확정은 시간순 — 재생은 경기 흐름대로
        kept.sort(key=lambda c: c["start"])

        log.info("trim_budget: 예산 %ds[%s](상한 %ds) — %d건 %ds 유지, %d건 버림 %s",
                 budget, source, cap, len(kept), used, len(dropped), dropped or "")
        if trace is not None:
            head = (f"- 예산 {budget}s ({source} 지정 · 허용 상한 {cap}s) "
                    f"→ 유지 {len(kept)}건 {used}s")
            lines = [head]
            for clip in kept:
                lines.append(f"  * scene {clip['scene_no']} rank={clip['rank']} {clip['sec']}s")
            if dropped:
                lines.append(f"- 버림 {len(dropped)}건: {', '.join(dropped)}")
            trace.note("trim_budget", "예산 덜어내기", "\n".join(lines))

        return {"clips": kept, "dropped": dropped}

    return trim_budget
