"""trim_budget 노드 — 목표 분량(budget_sec)에 맞춰 클립을 덜어낸다 (순수 계산, LLM 무관).

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
        """rank 순으로 담다가 허용 상한 초과분을 버린다 — budget_sec 없으면 통과."""
        clips = st.get("clips") or []
        budget = st.get("budget_sec")
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

        log.info("trim_budget: 예산 %ds(상한 %ds) — %d건 %ds 유지, %d건 버림 %s",
                 budget, cap, len(kept), used, len(dropped), dropped or "")
        if trace is not None:
            lines = [f"- 예산 {budget}s (허용 상한 {cap}s) → 유지 {len(kept)}건 {used}s"]
            for clip in kept:
                lines.append(f"  * scene {clip['scene_no']} rank={clip['rank']} {clip['sec']}s")
            if dropped:
                lines.append(f"- 버림 {len(dropped)}건: {', '.join(dropped)}")
            trace.note("trim_budget", "예산 덜어내기", "\n".join(lines))

        return {"clips": kept, "dropped": dropped}

    return trim_budget
