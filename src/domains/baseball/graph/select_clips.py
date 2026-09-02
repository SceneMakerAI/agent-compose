"""select_clips 노드 — 후보 좁히기(스펙 적용) + LLM 선곡.

후보 좁히기는 별도 노드가 아니라 이 노드의 입력 조립이다 (결정적 규칙):
- 매칭식: **이닝 AND 팀명(관점 해석) AND (라벨 OR 전광판)** — 각 축 안의 값끼리는 OR.
  (예: 이닝(4회초 or 4회말) AND 팀명(SSG,공격) AND (라벨(적시타 or 홈런) or 전광판(1점)))
- 팀명은 관점(view)과 조합해 해석한다: 공격 = 그 팀이 타석인 구간 /
  수비 = 상대가 타석인 구간(그 팀이 수비) / 무지정 = 팀으로 좁히지 않는다
  ("OO 하이라이트"는 공격·수비 활약 모두 후보 — select LLM 이 판단).
- 지정 안 된 축은 조건 없음 = 통과. 축이 전부 비면 전량이 후보다.
- 0건 폴백은 **내용 축(라벨·전광판)만** 완화한다 — 이닝·팀명은 사용자가 명시한
  하드 제약이라 절대 완화하지 않는다 (0건이면 그게 사실 — 빈 편성).

선곡은 "LLM 이 제안, 코드가 집행": 후보에 실존하는 구간 번호만 검산 통과.
전송 전에 프롬프트 토큰 수를 서버 /tokenize 로 재서 남긴다 — 후보가 많아
프롬프트가 길어지면 맵-리듀스(후보 분할 선곡 → 병합)로 갈 분기 근거다.
"""

import asyncio
import math

from domains.baseball.graph.select_end_point import clip_of
from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.evidences import EvidenceRepo
from domains.baseball.repo.scenes import Scene
from infer.chat import ChatLLM
from log import get_logger

log = get_logger(__name__)

SYSTEM = """\
[역할]
당신은 야구 하이라이트 편성자입니다. 후보 구간에서 질의에 맞지 않는 것을 걸러내고,
남긴 구간을 중요도 순으로 나열합니다.

[규칙]
1. 후보는 이미 필터를 통과해 대체로 질의에 부합합니다 — **질의에 명백히 맞지 않는
   구간만 제외**하고 나머지는 남깁니다.
2. 반드시 후보에 존재하는 구간 번호만 씁니다. 목록에 없는 번호는 절대 지어내지 않습니다.
3. 질의에 맞는 구간이 하나도 없으면 "없음"만 출력합니다.
4. 클립 내용은 그 클립을 재생하면 실제로 나오는 해설·화면·자막입니다.
   [★해설]·[★화면]·[★자막]처럼 ★ 이 붙은 줄은 사용자 질의와의 벡터 유사도 검색에
   걸린 내용입니다 — 판단의 우선 근거로 삼되, ★ 이 없어도 내용이 질의에 맞으면
   남길 수 있습니다.
5. 보드 사실(라벨·전광판·점수)과 클립 내용이 충돌하면 보드 사실을 우선합니다.
6. 출력은 남긴 구간을 **중요도 내림차순**으로 나열합니다 — 질의 부합도가 높고
   하이라이트 가치(득점·승부를 가른 플레이·결정적 수비)가 큰 순서입니다.
   분량은 신경 쓰지 않습니다 — 최종 분량은 이후 단계가 이 순서를 근거로 맞춥니다.

[출력 형식 — 중요도 내림차순 구간 번호만 콤마로 한 줄]
46,5,26\
"""


def batting_team_of(scene: Scene) -> str | None:
    """구간의 공격팀 — 이닝 초=원정, 말=홈. 이닝 미인식('-1' 등)이면 None."""
    if scene.inning.endswith("초"):
        return scene.away_team
    if scene.inning.endswith("말"):
        return scene.home_team
    return None


def apply_spec(scenes: list[Scene], spec: dict) -> list[Scene]:
    """
    Summary:
        필터 스펙을 인벤토리에 결정적으로 적용한다 — 선곡 후보를 모은다.
    Args:
        scenes (list[Scene]): 인벤토리 전량 (scene_no 순).
        spec (dict): parse_query 스펙 (innings·teams·view·labels·board_tags).
    Returns:
        list[Scene]: 후보 구간 (scene_no 순 유지). 축이 전부 비면 전량 그대로.
    Description:
        - 매칭식: 이닝 AND 팀명(관점 해석) AND (라벨 OR 전광판). 축 안의 값끼리는 OR.
        - 팀명은 view 와 조합: 공격 = 그 팀 타석 / 수비 = 상대 타석 / 무지정 = 안 좁힘.
        - 지정 안 된 축은 조건 없음 = 통과. 축이 전부 비면 전량 그대로.
    """
    candidates = []
    for scene in scenes:
        # 이닝 — AND 축 (지정 시 필수, 값들끼리 OR)
        if spec["innings"] and scene.inning not in spec["innings"]:
            continue

        # 팀명 — 관점(view)과 조합해 해석. 무지정 관점이면 팀으로 좁히지 않는다.
        if spec["teams"] and spec["view"] in ("공격", "수비"):
            batting = batting_team_of(scene)
            if batting is None:
                continue    # 이닝 미인식 구간 — 누가 타석인지 몰라 관점 판정 불가
            if spec["view"] == "공격" and batting not in spec["teams"]:
                continue
            if spec["view"] == "수비" and batting in spec["teams"]:
                continue

        # 내용 — 라벨 OR 전광판 (둘 다 비면 조건 없음 = 통과)
        if spec["labels"] or spec["board_tags"]:
            label_hit = any(label in spec["labels"] for label in scene.labels)
            tag_hit = any(tag in spec["board_tags"] for tag in scene.tags)
            if not label_hit and not tag_hit:
                continue

        candidates.append(scene)
    return candidates


def select_candidates(scenes: list[Scene], spec: dict) -> tuple[list[Scene], bool]:
    """
    Summary:
        선곡 후보 확정 — 스펙 적용, 0건이면 내용 축만 완화해 한 번 더.
    Returns:
        tuple: (후보 구간, 완화 여부). 완화해도 0건이면 빈 목록 그대로 —
            이닝·팀명이 걸러낸 결과는 사실이므로 지어내지 않는다.
    """
    candidates = apply_spec(scenes, spec)
    if candidates:
        return candidates, False

    # 내용 축이 없었다면 완화할 것도 없다 — 0건 그대로 (이닝·팀명의 사실)
    if not spec["labels"] and not spec["board_tags"]:
        return candidates, False

    relaxed = {**spec, "labels": [], "board_tags": []}
    return apply_spec(scenes, relaxed), True


KIND_LABEL = {"stt": "해설", "shot": "화면", "etc": "자막"}


def build_clip_contents(candidates: list[Scene], texts: list[dict],
                        evidence_hits: list[dict]) -> dict[int, dict]:
    """
    Summary:
        후보마다 클립 범위(pitch~끝 후보)와 그 범위에 실제로 담긴 내용을 조립한다.
    Args:
        candidates (list[Scene]): 후보 구간.
        texts (list[dict]): 증거 원문 전량 (fetch_texts — 시간순).
        evidence_hits (list[dict]): 벡터 검색 히트 원본 — [질의 유사] 표기 근거.
    Returns:
        dict[int, dict]: scene_no → {clip: clip_of 결과, lines: 내용 줄 목록}.
    Description:
        - 클립 범위는 select_end_point.clip_of 와 같은 규칙 — 프롬프트에 보여주는
          범위와 실제로 잘리는 범위가 어긋나지 않는다.
        - 시간 겹침으로 귀속한다 (끝 후보가 구간 밖까지 나가는 사례 대응 — 실측).
        - 같은 (종류, 내용) 반복은 1줄만 (etc 프레임 반복 자막 대응).
        - 벡터 검색에 걸린 내용은 끝에 [질의 유사] 를 붙인다.
    """
    hit_keys = set()
    for hit in evidence_hits or []:
        hit_keys.add((hit.get("kind"), hit.get("text")))

    contents: dict[int, dict] = {}
    for scene in candidates:
        clip = clip_of(scene)
        lines = []
        seen = set()
        for row in texts:
            # 시간 겹침 — 점 증거(shot·etc 는 start==end)도 범위 안이면 포함
            if row["end_sec"] < clip["start"] or row["start_sec"] > clip["end"]:
                continue

            key = (row["kind"], row["text"])
            if key in seen:
                continue
            seen.add(key)

            # ★ = 벡터 유사도 검색에 걸린 내용 (SYSTEM 프롬프트가 뜻을 설명한다)
            star = "★" if key in hit_keys else ""
            label = KIND_LABEL.get(row["kind"], row["kind"])
            lines.append(f"  * [{star}{label}] {row['text']}")
        contents[scene.scene_no] = {"clip": clip, "lines": lines}
    return contents


def render_inventory(candidates: list[Scene], contents: dict[int, dict]) -> str:
    """후보 구간 → 프롬프트 블록. 클립 범위와 그 안의 실제 내용을 함께 싣는다."""
    blocks = []
    for scene in candidates:
        team = batting_team_of(scene)
        lines = [f"[구간 {scene.scene_no}]",
                 f"- 이닝: {scene.inning}" + (f" (공격: {team})" if team else "")]
        if scene.labels:
            lines.append(f"- 라벨: {','.join(scene.labels)}")
        if scene.tags:
            lines.append(f"- 전광판: {','.join(scene.tags)}")
        # 점수는 구간 종료 시점 (원정:홈). 이 구간에서 득점이 났으면 +N 을 붙인다.
        score = f"- 점수: {scene.score_away}:{scene.score_home}"
        if scene.diff_score > 0:
            score += f" (이 구간 +{scene.diff_score})"
        lines.append(score)

        entry = contents.get(scene.scene_no)
        if entry:
            clip = entry["clip"]
            lines.append(f"- 클립: {clip['start']}~{clip['end']}s ({clip['sec']}s)")
            if entry["lines"]:
                lines.append("- 클립 내용:")
                lines += entry["lines"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_user(query: str, inventory: str) -> str:
    """질의 + 후보 블록 → 유저 프롬프트."""
    return f"[질의]\n{query}\n\n[후보 구간]\n{inventory}"


def parse_picked(text: str, candidates: list[Scene]) -> list[int]:
    """
    Summary:
        선곡 응답(번호 나열) → 실존 scene_no 목록 — **응답 순서 = 중요도 내림차순 유지**.
    Description:
        - 순서가 정보다: 이후 단계(trim_budget)가 예산을 맞출 때 꼬리(덜 중요한 것)부터
          버리는 근거가 이 순서다. 시간순 정렬은 클립 확정 단계(select_end_point)가 한다.
        - 후보에 없는 번호는 버리고 로그로 드러낸다 (사실이 모델을 이긴다).
        - 중복은 제거한다. "없음"·번호 없음이면 빈 목록.
    """
    known = set()
    for scene in candidates:
        known.add(scene.scene_no)

    picked = []
    ghosts = []
    for token in text.replace("없음", "").split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        scene_no = int(token)
        if scene_no in known:
            if scene_no not in picked:
                picked.append(scene_no)
        else:
            ghosts.append(scene_no)
    if ghosts:
        log.warning("select_clips 검산: 후보 밖 번호 제거 %s", ghosts)
    return picked


def make_node(llm: ChatLLM, evidence_repo: EvidenceRepo, tokens_max: int):
    """자원 주입 팩토리 — build.py 가 호출한다.

    tokens_max: 선곡 콜 1건의 입력 토큰 상한 (Settings.select_tokens_max — .env).
    초과하면 맵-리듀스로 분기한다. 실측: 구간 블록당 ~51토큰(내용 미포함).
    """

    async def select_clips(st: ComposeState) -> dict:
        """스펙으로 후보를 좁히고, LLM 이 질의에 맞는 구간을 고른다."""
        scenes = st["scenes"]
        spec = st.get("spec") or {}
        trace = st.get("trace")

        candidates, relaxed = select_candidates(scenes, spec)
        note = " (내용 축 완화)" if relaxed else ""
        log.info("select_clips 후보: %d/%d구간%s", len(candidates), len(scenes), note)
        if not candidates:
            # 좁히는 축(이닝·공격팀)이 걸러낸 0건은 사실 — 빈 편성으로 종결
            if trace is not None:
                trace.note("select_clips", "후보 0건", "좁히는 축 결과 0건 — 빈 편성 종결")
            return {"candidates": [], "picked": [], "status": "empty"}

        # 클립 내용 재료 — 증거 원문 전량 1콜. 실패해도 선곡은 계속한다 (보드 사실만으로).
        try:
            texts = await evidence_repo.fetch_texts(st["v_id"])
        except Exception as e:               # noqa: BLE001 — 보조 재료, 죽이지 않는다
            log.warning("select_clips 클립 내용 조회 실패(보드 사실만으로 진행): %s", e)
            texts = []
        contents = build_clip_contents(candidates, texts, st.get("evidence_hits") or [])
        user = render_user(st["query"], render_inventory(candidates, contents))

        # 프롬프트 토큰 실측 — tokens_max 초과면 맵-리듀스로 분기.
        # 계수 실패는 단일 콜로 계속한다 (관측 실패가 편성을 막지 않는다).
        try:
            tokens = await llm.count_tokens(SYSTEM, user)
        except Exception as e:               # noqa: BLE001 — 관측 실패는 본 작업 비차단
            log.warning("select_clips 토큰 계수 실패(단일 콜로 진행): %s", e)
            tokens = -1
        log.info("select_clips 프롬프트: 후보 %d구간 · %s토큰", len(candidates), tokens)
        if trace is not None:
            trace.note("select_clips",
                       f"후보 {len(candidates)}/{len(scenes)}구간{note} · 프롬프트 {tokens}토큰",
                       ", ".join(str(scene.scene_no) for scene in candidates))

        if tokens <= tokens_max:
            # 단일 콜 — 후보 전체를 한 프롬프트로. thinking 켬 — 후보 전체에서
            # "질의에 맞는가 + 무엇이 더 중요한가"를 가르는 판단이라 추론 사슬이 필요하다
            # (.env LLM_THINKING=0 으로 일괄 비활성 가능, 교착·빈 본문은 chat 이 폴백).
            text = await llm.chat(SYSTEM, user, thinking=True,
                                  trace=trace, name="select_clips")
            log.info("select_clips 응답: %r", text)
            picked = parse_picked(text, candidates)
        else:
            # 맵-리듀스 — 청크별 병렬 선곡 후 합집합
            picked = await _map_reduce(llm, st["query"], candidates, contents,
                                       tokens, tokens_max, trace)
        log.info("select_clips 선곡: %s", picked or "없음")

        candidate_nos = []
        for scene in candidates:
            candidate_nos.append(scene.scene_no)
        return {"candidates": candidate_nos,
                "picked": picked,
                "status": "ok" if picked else "empty"}

    return select_clips


def split_chunks(candidates: list[Scene], n: int) -> list[list[Scene]]:
    """후보를 시간순 연속 블록 n 개로 등분한다 (마지막 청크가 나머지를 흡수)."""
    size = math.ceil(len(candidates) / n)
    chunks = []
    for start in range(0, len(candidates), size):
        chunks.append(candidates[start:start + size])
    return chunks


async def _map_reduce(llm: ChatLLM, query: str, candidates: list[Scene],
                      contents: dict[int, dict], total_tokens: int,
                      tokens_max: int, trace) -> list[int]:
    """
    Summary:
        맵-리듀스 선곡 — 후보를 청크로 나눠 병렬 선곡(map)하고 합집합(reduce)한다.
    Description:
        - 청크 수 = ceil(총토큰 / tokens_max), 분할은 시간순 등분.
        - map: 청크마다 같은 질의·같은 SYSTEM 으로 동시 콜 (동시성은 ChatLLM
          세마포어가 관리). 청크 1건 실패는 그 청크만 포기한다 (건별 격리).
        - reduce: 청크별 중요도 순위를 **라운드로빈 병합** — 각 청크의 1위들, 2위들…
          순으로 잇는다. 청크끼리 우열은 모르므로 순위 자리로 근사한다 (전역 재순위가
          필요해지면 LLM reduce 콜을 그때 추가한다).
    """
    n = math.ceil(total_tokens / tokens_max)
    chunks = split_chunks(candidates, n)
    log.info("select_clips 맵-리듀스: %d토큰 → %d청크 %s",
             total_tokens, len(chunks), [len(c) for c in chunks])
    if trace is not None:
        lines = []
        for i, chunk in enumerate(chunks, 1):
            scene_nos = ", ".join(str(scene.scene_no) for scene in chunk)
            lines.append(f"- 청크 {i}: {len(chunk)}구간 — {scene_nos}")
        trace.note("select_clips", f"맵-리듀스 분기 — {len(chunks)}청크", "\n".join(lines))

    async def select_one(index: int, chunk: list[Scene]) -> list[int]:
        """청크 1건 선곡 — 실패는 그 청크만 빈 결과 (배치 전체를 죽이지 않는다)."""
        user = render_user(query, render_inventory(chunk, contents))
        try:
            text = await llm.chat(SYSTEM, user, thinking=True, trace=trace,
                                  name=f"select_clips[{index}]")
        except Exception as e:               # noqa: BLE001 — 건별 격리
            log.warning("select_clips 청크 %d 실패(그 청크 선곡 포기): %s", index, e)
            return []
        return parse_picked(text, chunk)

    tasks = []
    for i, chunk in enumerate(chunks, 1):
        tasks.append(select_one(i, chunk))
    results = await asyncio.gather(*tasks)

    # reduce — 중요도 순위 자리 기준 라운드로빈 병합 (각 청크의 1위들 → 2위들 → …)
    merged = []
    seen = set()
    max_len = max((len(picked) for picked in results), default=0)
    for rank in range(max_len):
        for picked in results:
            if rank < len(picked) and picked[rank] not in seen:
                seen.add(picked[rank])
                merged.append(picked[rank])
    return merged
