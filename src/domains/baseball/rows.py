"""편성 결과 → 저장 행 — 최종 클립에 인벤토리(Scene)의 태그·라벨·이닝을 되붙인다."""


def _union(lists) -> str:
    out = []
    for items in lists:
        for item in items:
            if item not in out:
                out.append(item)
    return ",".join(out)


def clip_rows(state: dict) -> list[dict]:
    """병합 클립은 담은 구간 전부의 태그·라벨을 합치고, 흡수한 구간 번호를 merge_seqs 에 남긴다."""
    by_no = {}
    for scene in state.get("scenes") or []:
        by_no[scene.scene_seq] = scene

    rows = []
    for clip in state.get("clips") or []:
        seqs = clip.get("scene_seqs") or [clip["scene_seq"]]
        scenes = []
        for seq in seqs:
            scene = by_no.get(seq)
            if scene is not None:
                scenes.append(scene)
        absorbed = []
        for seq in seqs[1:]:
            scene = by_no.get(seq)
            if scene is not None:
                absorbed.append(str(scene.scene_stream_seq))
        rows.append({
            "stream_id": clip["stream_id"],
            "scene_stream_seq": clip["scene_stream_seq"],
            "scene_seq": clip["scene_seq"],
            "merge_seqs": ",".join(absorbed) or None,
            "start": clip["start"],
            "end": clip["end"],
            "start_whole": clip["start_whole"],
            "end_whole": clip["end_whole"],
            "tags": _union(s.tags for s in scenes),
            "labels": _union(s.labels for s in scenes),
            "inning": scenes[0].inning if scenes else "",
        })
    return rows
