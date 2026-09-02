"""select_end_point 노드 — 선곡된 구간의 클립 좌표(시작·끝점)를 확정한다.

좌표 재료는 상류(agent-vision)가 이미 계산해 뒀다 — 여기는 후보 중 **선택**만 한다:
- 시작: pitch_idx(투구 앵커). 없으면 구간 시작 (결정적).
- 끝: end_idxs(종료 후보, 시간순 최대 2개) 중 선택.
  · 후보 2개 이상 → **LLM 이 고른다** (클립당 1콜 병렬 — 후보 사이에 무엇이
    담기는지를 보여주고 질의 의도에 맞는 쪽을 고르게 한다).
  · 후보 1개 → 그 후보 (결정적). 없음 → 구간 끝 (결정적).
  · LLM 실패·후보 밖 응답 → 마지막 후보 폴백 (건별 격리 — 배치를 죽이지 않는다).

선곡 중요도(rank — picked 순서)는 클립에 그대로 보존한다 — 뒷단(꼬리 자르기)의
예산 덜어내기 근거다. 재생 순서는 시간순으로 확정한다.

총 분량(budget_sec) 맞춤은 이 노드가 아니라 꼬리 자르기 단계의 몫이다 —
LLM 에게 산수를 시키지 않는다 (분량 규칙은 thinking 폭주를 부른다 — 실측).
"""

import asyncio

from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.evidences import EvidenceRepo
from domains.baseball.repo.scenes import Scene
from infer.chat import ChatLLM
from log import get_logger

log = get_logger(__name__)

KIND_LABEL = {"stt": "해설", "shot": "화면", "etc": "자막"}

SYSTEM = """\
[역할]
당신은 야구 하이라이트 편집자입니다. 클립 하나의 **끝점**을 후보 중에서 고릅니다.

[규칙]
1. 반드시 제시된 후보 번호 중 하나만 답합니다.
2. 클립은 플레이의 결과와 그 결과를 설명하는 해설이 끝나는 지점에서 닫혀야 합니다.
3. 뒤 후보를 고르면 추가되는 내용이 이 플레이의 여운(리플레이·세리머니·마무리 해설)이면
   뒤 후보를, 다음 타석 준비나 무관한 내용이면 앞 후보를 고릅니다.
4. 질의가 분량 취향을 드러내면 따릅니다 — "짧게"·"핵심만" 은 앞 후보,
   "풀버전"·"여운까지" 는 뒤 후보 쪽입니다.

[출력 형식 — 후보 번호만 한 줄]
2\
"""


def clip_of(scene: Scene) -> dict:
    """
    Summary:
        구간 1건 → 기본 클립 좌표 (순수 계산 — LLM 선택 전의 기본값).
    Returns:
        dict: {scene_no, start, end, sec, start_from, end_from}.
    Description:
        - 시작: pitch_idx 가 있으면 그 시각(투구부터), 없으면 구간 시작.
        - 끝: end_idxs 마지막 후보(여운 포함 완결 지점), 없으면 구간 끝.
        - 좌표가 뒤집히면(끝 ≤ 시작 — 상류 데이터 이상) 구간 통째로 폴백한다.
    """
    if scene.pitch_idx is not None:
        start = scene.pitch_idx
        start_from = "pitch_idx"
    else:
        start = scene.start
        start_from = "구간 시작"

    if scene.end_idxs:
        end = scene.end_idxs[-1]
        end_from = f"end_idxs 마지막({len(scene.end_idxs)}중)"
    else:
        end = scene.end
        end_from = "구간 끝"

    # 좌표 역전 — 상류 데이터 이상 신호. 자르지 말고 구간 통째로 폴백한다.
    if end <= start:
        log.warning("select_end_point: scene %d 좌표 역전(%s~%s) — 구간 통째 폴백",
                    scene.scene_no, start, end)
        start, end = scene.start, scene.end
        start_from, end_from = "구간 시작(역전 폴백)", "구간 끝(역전 폴백)"

    return {
        "scene_no": scene.scene_no,
        "start": start,
        "end": end,
        "sec": end - start,
        "start_from": start_from,
        "end_from": end_from,
    }


def content_lines(texts: list[dict], lo: float, hi: float) -> list[str]:
    """[lo, hi] 와 겹치는 증거 원문 → 프롬프트 줄 (시간순, 같은 종류·내용은 1줄)."""
    lines = []
    seen = set()
    for row in texts:
        if row["end_sec"] < lo or row["start_sec"] > hi:
            continue
        key = (row["kind"], row["text"])
        if key in seen:
            continue
        seen.add(key)
        label = KIND_LABEL.get(row["kind"], row["kind"])
        lines.append(f"  * [{label}] {row['text']}")
    return lines


def render_end_user(query: str, scene: Scene, clip: dict, texts: list[dict]) -> str:
    """끝점 선택 프롬프트 — 공통 내용(시작~첫 후보)과 후보별 추가분을 나눠 보여준다.

    시작(투구)부터 첫 후보까지는 어느 후보를 골라도 담기는 **공통 내용**이다 —
    후보 섹션에 섞으면 "그 후보를 골라야 보이는 내용"처럼 읽혀 판단을 흐린다.
    """
    parts = [f"[질의]\n{query}\n",
             f"[클립 — 구간 {scene.scene_no}]",
             f"- 이닝: {scene.inning}",
             f"- 라벨: {','.join(scene.labels) or '-'} / 전광판: {','.join(scene.tags) or '-'}",
             f"- 시작: {clip['start']}s\n"]

    # 공통 내용 — 시작(투구)부터 첫 후보까지 (어느 후보를 골라도 담긴다)
    first = scene.end_idxs[0]
    base = content_lines(texts, clip["start"], first)
    parts.append(f"[공통 내용 — 시작 {clip['start']}s ~ 첫 후보 {first}s]\n"
                 + ("\n".join(base) or "  (내용 없음)"))

    prev = clip["start"]
    for i, cand in enumerate(scene.end_idxs, 1):
        title = f"[후보 {i} — 끝 {cand}s (클립 {cand - clip['start']}s)]"
        if i == 1:
            parts.append(f"{title} — 공통 내용까지 담고 끝납니다.")
        else:
            body = content_lines(texts, prev, cand)
            parts.append(f"{title} — 후보 {i - 1} 에서 추가되는 내용:\n"
                         + ("\n".join(body) or "  (추가 내용 없음)"))
        prev = cand
    return "\n\n".join(parts)


def parse_choice(text: str, n_candidates: int) -> int | None:
    """응답 → 후보 번호(1-base). 범위 밖·판독 불가면 None (호출부가 기본값 폴백)."""
    for token in text.split():
        token = token.strip().rstrip(".")
        if token.isdigit():
            choice = int(token)
            if 1 <= choice <= n_candidates:
                return choice
            return None
    return None


def make_node(llm: ChatLLM, evidence_repo: EvidenceRepo):
    """자원 주입 팩토리 — build.py 가 호출한다."""

    async def select_end_point(st: ComposeState) -> dict:
        """선곡(picked) 구간마다 클립 좌표를 확정한다 — rank 보존, 시간순 반환."""
        picked = st.get("picked") or []
        if not picked:
            return {"clips": []}

        by_no = {}
        for scene in st["scenes"]:
            by_no[scene.scene_no] = scene

        # 기본 좌표 + rank (picked 순서 = 선곡 중요도)
        clips = []
        targets = []        # LLM 에게 물을 클립 (끝 후보 2개 이상)
        for rank, scene_no in enumerate(picked, 1):
            scene = by_no.get(scene_no)
            if scene is None:       # 검산 통과분이라 없을 수 없지만, 침묵 통과는 금지
                log.warning("select_end_point: picked %d 가 인벤토리에 없음 — 제외", scene_no)
                continue
            clip = clip_of(scene)
            clip["rank"] = rank     # 선곡 중요도 순위 — 꼬리 자르기의 예산 덜어내기 근거
            clips.append(clip)
            if len(scene.end_idxs) >= 2:
                targets.append((scene, clip))

        trace = st.get("trace")

        # 끝 후보가 갈리는 클립만 LLM 에게 — 내용 재료는 Milvus 1콜
        if targets:
            try:
                texts = await evidence_repo.fetch_texts(st["v_id"])
            except Exception as e:           # noqa: BLE001 — 재료 실패면 기본값으로 진행
                log.warning("select_end_point 내용 조회 실패(기본 끝점 유지): %s", e)
                texts = []

        async def choose_one(scene: Scene, clip: dict) -> None:
            """클립 1건 끝점 질의 — 실패는 그 클립만 기본값 유지 (건별 격리)."""
            user = render_end_user(st["query"], scene, clip, texts)
            try:
                text = await llm.chat(SYSTEM, user, trace=trace,
                                      name=f"select_end_point[{scene.scene_no}]")
            except Exception as e:           # noqa: BLE001 — 건별 격리
                log.warning("select_end_point: scene %d 콜 실패(기본 끝점 유지): %s",
                            scene.scene_no, e)
                return
            choice = parse_choice(text, len(scene.end_idxs))
            if choice is None:
                log.warning("select_end_point: scene %d 응답 판독 불가 %r — 기본 끝점 유지",
                            scene.scene_no, text)
                return
            clip["end"] = scene.end_idxs[choice - 1]
            clip["sec"] = clip["end"] - clip["start"]
            clip["end_from"] = f"end_idxs {choice}/{len(scene.end_idxs)} (LLM)"

        if targets:
            tasks = []
            for scene, clip in targets:
                tasks.append(choose_one(scene, clip))
            await asyncio.gather(*tasks)

        # 재생은 경기 흐름대로 — 시간순으로 확정한다 (중요도는 rank 가 든다)
        clips.sort(key=lambda c: c["start"])

        total = 0
        for clip in clips:
            total += clip["sec"]
        log.info("select_end_point: 클립 %d건 · 총 %ds (LLM 선택 %d건)",
                 len(clips), total, len(targets))

        if trace is not None:
            lines = []
            for clip in clips:
                lines.append(f"- scene {clip['scene_no']:>3} rank={clip['rank']}: "
                             f"{clip['start']}~{clip['end']}s ({clip['sec']}s) — "
                             f"시작:{clip['start_from']} · 끝:{clip['end_from']}")
            lines.append(f"- 총 {len(clips)}건 · {total}s · LLM 선택 {len(targets)}건")
            trace.note("select_end_point", "클립 좌표 확정", "\n".join(lines))

        return {"clips": clips}

    return select_end_point
