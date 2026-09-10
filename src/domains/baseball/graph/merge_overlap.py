"""merge_overlap 노드 — 한 플레이가 두 구간으로 쪼개져 겹치는 클립을 하나로 접는다.

겹치는 두 구간의 [min(start), max(end)] 는 합집합이라 병합해도 새 영상이 들어오지
않는다 — 중복 재생만 사라진다. 접을지 말지는 mergeable() 한 곳이 판정하고,
사례는 _CASES 에 규칙 함수를 더해 늘린다.
"""

from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.scenes import Scene
from log import get_logger

log = get_logger(__name__)


def out_split(a: Scene, b: Scene) -> bool:
    """병살·주자 아웃 — 한 플레이의 아웃이 전광판에서 두 번 세어진 경우."""
    return a.diff_out > 0 and b.diff_out > 0


# 병합 사례 — 새 사례는 (앞 구간, 뒤 구간) -> bool 함수를 여기 추가한다
_CASES = (out_split,)


def mergeable(a: Scene, b: Scene, a_end: int, b_start: int) -> bool:
    """앞 구간 a(끝 a_end)와 뒤 구간 b(시작 b_start)의 클립을 한 클립으로 볼지 판정한다.

    이닝이 다르면 접지 않는다 — 렌더가 이닝별로 묶는다. 딱 붙은 경계는 중복이 없어 둔다.
    """
    if a.stream_id != b.stream_id or a.inning != b.inning:
        return False
    if b_start >= a_end:
        return False
    return any(case(a, b) for case in _CASES)


def make_node():
    """팩토리 — 순수 계산이라 주입할 자원이 없다."""

    async def merge_overlap(st: ComposeState) -> dict:
        """시간순 클립을 훑으며 겹치는 이웃을 앞 클립에 흡수시킨다."""
        clips = st.get("clips") or []
        trace = st.get("trace")
        if not clips:
            return {"clips": clips}

        by_no = {}
        for scene in st["scenes"]:
            by_no[scene.scene_seq] = scene

        merged: list[dict] = []
        prev_scene: Scene | None = None
        for src in clips:
            scene = by_no.get(src["scene_seq"])
            clip = {**src, "scene_seqs": [src["scene_seq"]]}
            if (merged and scene is not None and prev_scene is not None
                    and mergeable(prev_scene, scene, merged[-1]["end"], clip["start"])):
                head = merged[-1]
                if clip["end"] > head["end"]:
                    head["end"] = clip["end"]
                    head["end_whole"] = clip["end_whole"]
                    head["end_from"] = clip["end_from"]
                head["sec"] = head["end"] - head["start"]
                # 강한 쪽(작은 값)을 승계한다 — 약한 rank 를 물려받아 예산 절단에
                # 먼저 잘리면 안 된다
                head["rank"] = min(head["rank"], clip["rank"])
                head["scene_seqs"].append(clip["scene_seq"])
            else:
                merged.append(clip)
            prev_scene = scene

        folded = len(clips) - len(merged)
        total = 0
        for clip in merged:
            total += clip["sec"]
        log.info("merge_overlap: 클립 %d건 → %d건 (병합 %d건 · 총 %ds)",
                 len(clips), len(merged), folded, total)

        if trace is not None:
            lines = []
            for clip in merged:
                if len(clip["scene_seqs"]) > 1:
                    seqs = "+".join(str(s) for s in clip["scene_seqs"])
                    lines.append(f"- scene {seqs} → {clip['start']}~{clip['end']}s "
                                 f"({clip['sec']}s) rank={clip['rank']}")
            if not lines:
                lines.append(f"- 겹치는 클립 없음 ({len(clips)}건 그대로)")
            trace.note("merge_overlap", "겹침 병합", "\n".join(lines))

        return {"clips": merged}

    return merge_overlap
