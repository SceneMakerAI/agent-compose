"""select_clips 노드의 재료 — 인벤토리 렌더 + 응답(줄 형식) 파싱 + 선곡 검산.

"LLM 이 제안, 사실이 처분": 선곡은 인벤토리에 실존하는 scene_id 로만 검산 통과.
LLM 호출 자체는 graph 가 한다 — 여기는 순수 렌더·파싱만 (테스트 단독 실행 가능).
rephrase_query·score_match 응답 파서(parse_expand·parse_verify)도 여기 있다.
"""

import re

from log import get_logger

log = get_logger(__name__)

def render_situation(outs: int | None, bases: str | None) -> str:
    """전광판 아웃·주자 → '1사 1·2루' 표기. 값이 없으면 '?'.

    라벨로 표현되지 않는 유일한 정보라 인벤토리의 핵심 열이다 — 같은 범타라도
    '2사 만루'(위기 탈출)와 '무사 주자없음'(평범한 아웃)은 하이라이트 가치가 다른데
    둘 다 score_delta=0 이라 점수로도 갈리지 않는다.

    (2026-08-20 제거를 검토했다가 유지 — 70장면 기준 900자라 프롬프트 절감이
    2.6KB→2.3KB 에 그쳐 속도 이득이 없었다. 변별을 내주고 얻을 게 없다.)
    """
    if outs is None or bases is None:
        return "?"
    on = [n for n, b in zip(("1루", "2루", "3루"), str(bases)) if b == "1"]
    if len(on) == 3:
        runners = "만루"
    elif on:
        runners = "·".join(on)
    else:
        runners = "주자없음"
    return f"{outs}사 {runners}"


def render_bases(bases: str | None) -> str:
    """주자 비트('110') → '1루·2루' / '만루' / '주자없음'. 값이 없으면 '?'."""
    if bases is None:
        return "?"
    on = [n for n, b in zip(("1루", "2루", "3루"), str(bases)) if b == "1"]
    if len(on) == 3:
        return "만루"
    return "·".join(on) if on else "주자없음"


def score_after(r: dict) -> str:
    """플레이 **후** 점수 '1-0' (원정-홈). 계산할 수 없으면 앞 점수 그대로, 없으면 '?'.

    구 스키마의 `score` 컬럼('KIA 0-0 삼성')이 사라져 **계산으로 바뀐 자리**다
    (vision3 migration_20260823i — 전이 원장 폐기로 팀명 출처가 없어졌다).
    score_before(원정-홈) + score_delta 에 이닝의 초/말로 어느 쪽이 올랐는지를
    더한다: 초는 원정 공격, 말은 홈 공격이다. 공격팀을 모르면 올리지 않는다 —
    어느 쪽인지 모르는 채로 찍으면 점수가 통째로 틀린 값이 된다.
    """
    before = r.get("score_before") or "?"
    delta = r.get("score_delta") or 0
    inning = r.get("inning") or ""
    if before == "?" or delta <= 0:
        return before
    away, _, home = before.partition("-")
    try:
        away, home = int(away), int(home)
    except ValueError:
        return before
    if inning.endswith("초"):
        away += delta
    elif inning.endswith("말"):
        home += delta
    else:
        return before
    return f"{away}-{home}"


def render_score(r: dict) -> str:
    """인벤토리 점수 줄 — '0-0 -> 1-0'. 득점이 없으면 앞 점수 하나만."""
    before = r.get("score_before") or "?"
    after = score_after(r)
    return before if after == before else f"{before} -> {after}"


def render_inventory(scenes: list[dict], evidence: list[dict] | None = None) -> str:
    """t_scene_baseball 행 → select_clips 가 보는 장면 블록들 (경기 단위라 전 행).

    한 줄 표에서 **장면당 블록**으로 바꿨다 (2026-08-20). 열 정렬에 기대면 값이 빈
    항목도 자리를 차지하고, 어느 열이 무엇인지 모델이 헤더를 기억해야 한다. 항목마다
    이름을 달면 없는 값은 줄째로 빠진다.

    **검색 증거를 그 장면 블록 안에 병합**한다. 예전에는 별도 [벡터 후보] 블록이라
    인벤토리와 번호로만 이어져 조인이 모델 몫이었다.
    """
    by_scene = {g["scene_id"]: g for g in (evidence or [])}
    out: list[str] = []
    for r in scenes:
        outs = r.get("outs")
        lines = [f"[장면 {r['scene_id']}]",
                 f"- 이닝: {r['inning'] or '?'}",
                 f"- 아웃: {outs if outs is not None else '?'}",
                 f"- 주자: {render_bases(r.get('bases'))}",
                 f"- 태그: {','.join(r['tags']) or '판별불가'}"]
        if r.get("label_list"):
            lines.append(f"- 라벨: {','.join(r['label_list'])}")
        if r.get("game_context"):
            lines.append(f"- 판세: {r['game_context']}")
        # 전광판 사실 — 해석(태그·라벨)이 비어도 남는 유일한 재료라 항상 싣는다.
        if r.get("board_tags"):
            lines.append(f"- 전광판: {','.join(r['board_tags'])}")
        lines.append(f"- 점수상황: {render_score(r)}")
        lines.append(f"- 영상길이: {r['e'] - r['s']:.0f}s")
        if r.get("etc"):
            lines.append(f"- 기타정보: {r['etc']}")
        if g := by_scene.get(r["scene_id"]):
            # 기타정보로 이미 실은 줄은 증거에서 뺀다 — 그 장면에 걸린 etc 히트가
            # 같은 자막이면 한 블록에 같은 문장이 두 번 실리고, 종류당 2건뿐인
            # 스니펫 자리를 중복이 먹는다.
            snips = [f"  * [{KIND_LABEL.get(k, k)}] {t}"
                     for k, texts in sorted(g.get("by_kind", {}).items())
                     for t in [x for x in texts if x != r.get("etc")][:EVIDENCE_SNIPPETS_MAX]]
            if snips:
                lines.append("- 검색증거:")
                lines += snips
        out.append("\n".join(lines))
    return "\n\n".join(out)


KIND_LABEL = {"stt": "해설", "shot": "화면", "etc": "자막"}
EVIDENCE_SNIPPETS_MAX = 2       # 장면·종류당 스니펫 수


def parse_picked(text: str, scenes: list[dict]) -> list[int]:
    """
    Summary:
        select_clips 응답(번호 나열) → 실존 scene_id 목록.
    Description:
        - 응답 형식이 "5,26,46,59" 로 좁혀졌다 (2026-08-20). 명세(모드·대상·관점·예산)를
          함께 받던 줄 형식은 폐기 — 그 값들은 코드가 정한다.
        - **실존 검산은 여기 남는다**: 목록에 없는 번호를 지어내면 조용히 버리고 로그로
          드러낸다(사실이 모델을 이긴다).
        - 중복은 제거하고 등장 순서를 유지한다.
    """
    known = {r["scene_id"] for r in scenes}
    ids = [int(x) for x in re.findall(r"\d+", text)]
    picked = list(dict.fromkeys(i for i in ids if i in known))
    if ghost := sorted({i for i in ids if i not in known}):
        log.warning("선곡 검산: 실존하지 않는 장면 제거 %s", ghost)
    return picked


def spec_line(spec: dict) -> str:
    """검수·리포트용 명세 한 줄."""
    return f"모드={spec['mode']} 관점={spec['view']} 대상={','.join(spec['targets']) or '-'}"



# ── rephrase_query / score_match 응답 파서 ────────────────

def parse_expand(text: str) -> tuple[list[str], list[str]]:
    """rephrase_query 응답 → (검색어, 필터). 깨지면 빈 값 — 호출부가 원 질의로 폴백."""
    phrases: list[str] = []
    filters: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("검색어:"):
            phrases = [p.strip() for p in line[4:].split(",") if p.strip()]
        elif line.startswith("필터:"):
            body = line[3:].strip()
            if body and body != "없음":
                filters = [f.strip() for f in body.split(",") if f.strip()]
    return phrases[:4], filters


_SCORE_LINE = re.compile(r"장면\s*(\d+)\s*[:：]\s*([0-3])\s*(정상|문제)?\s*(.*)")


def parse_verify(text: str) -> dict[int, dict]:
    """score_match 응답 → {scene_id: {score, complete, reason}}.

    (2026-08-20 채점 노드 폐기 후 미사용 — 되살릴 때를 위해 파서만 남긴다.)
    """
    out: dict[int, dict] = {}
    for line in text.splitlines():
        m = _SCORE_LINE.search(line.strip())
        if not m:
            continue
        out[int(m.group(1))] = {
            "score": int(m.group(2)),
            "complete": (m.group(3) or "정상") == "정상",
            "reason": m.group(4).strip(),
        }
    return out
