"""select_clips 노드의 재료 — 인벤토리 렌더 + 응답(줄 형식) 파싱 + 선곡 검산.

"LLM 이 제안, 사실이 처분": 선곡은 인벤토리에 실존하는 scene_id 로만 검산 통과.
LLM 호출 자체는 graph 가 한다 — 여기는 순수 렌더·파싱만 (테스트 단독 실행 가능).
rephrase_query·score_match 응답 파서(parse_expand·parse_verify)도 여기 있다.
"""

import re

from log import get_logger

log = get_logger(__name__)

DEFAULT_BUDGET_SEC = 180    # 질의·인자에 예산이 없을 때 (bench4 운영값)


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
        before = r["score_before"] or "?"
        cur = r["score"].split()[1] if r["score"] else "?"
        outs = r.get("outs")
        lines = [f"[장면 {r['scene_id']}]",
                 f"- 이닝: {r['inning'] or '?'}",
                 f"- 아웃: {outs if outs is not None else '?'}",
                 f"- 주자: {render_bases(r.get('bases'))}",
                 f"- 태그: {r['scene_type']}"]
        if r.get("labels"):
            lines.append(f"- 라벨: {r['labels']}")
        lines.append(f"- 점수상황: {before} -> {cur}")
        lines.append(f"- 영상길이: {r['e'] - r['s']:.0f}s")
        if r.get("batter"):
            lines.append(f"- 기타정보: {r['batter']}")
        if g := by_scene.get(r["scene_id"]):
            snips = [f"  * [{KIND_LABEL.get(k, k)}] {t[:EVIDENCE_TEXT_MAX]}"
                     for k, texts in sorted(g.get("by_kind", {}).items())
                     for t in texts[:EVIDENCE_SNIPPETS_MAX]]
            if snips:
                lines.append("- 검색증거:")
                lines += snips
        out.append("\n".join(lines))
    return "\n\n".join(out)


KIND_LABEL = {"stt": "해설", "shot": "화면", "etc": "자막"}
EVIDENCE_TEXT_MAX = 70          # 스니펫 표기 길이 (문장 중간 절단은 감수 — 전문은 색인에)
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
    return (f"모드={spec['mode']} 관점={spec['view']} "
            f"대상={','.join(spec['targets']) or '-'} 예산={spec['budget']}s")



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

    기각권이 없으므로 '무엇을 뺄까'가 아니라 '얼마나 맞나'만 읽는다. 파싱 실패한 줄은
    버린다 — 점수가 없는 클립은 fill_budget 이 기본값으로 다룬다(빠지지 않는다).
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
