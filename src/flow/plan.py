"""plan — 인벤토리 렌더 + 응답(줄 형식) 파싱 + select 검산 (bench4 compose/plan.py 이식).

"LLM 이 제안, 사실이 처분": 선곡은 인벤토리에 실존하는 scene_id 로만 검산 통과.
LLM 호출 자체는 nodes 가 한다 — 여기는 순수 렌더·파싱만 (테스트 단독 실행 가능).
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


def render_inventory(scenes: list[dict]) -> str:
    """t_scene_baseball 행 → plan 이 보는 목록 한 줄씩 (경기 단위라 전 행 — 2~3KB).

    점수에 원정팀명을 붙인다 — 가운데 토막만 떼면(`9-7`) 모델이 매 행마다
    초/말 → 원정/홈 → 팀명 → 어느 쪽이 올랐나를 3단 추론해야 한다.
    """
    lines = []
    for r in scenes:
        before = r["score_before"] or "?"
        cur = r["score"].split()[1] if r["score"] else "?"
        away = r.get("away_team") or ""
        score = f"{away} {before}→{cur}" if away else f"{before}→{cur}"
        sit = render_situation(r.get("outs"), r.get("bases"))
        lines.append(
            f"{r['scene_id']:3d}  {r['scene_type']:14s} {r['labels'] or '-':10s} "
            f"{r['inning'] or '?':7s} {sit:12s} {score:18s} {r.get('batter') or '-':6s}"
            f"  {r['e'] - r['s']:.0f}s")
    return "\n".join(lines)


KIND_LABEL = {"stt": "해설", "shot": "화면", "etc": "자막"}
EVIDENCE_TEXT_MAX = 70          # 스니펫 표기 길이 (문장 중간 절단은 감수 — 전문은 색인에)


def render_evidence(evidence: list[dict], orphan: list[dict],
                    scenes: list[dict] | None = None) -> str:
    """retrieve 결과 → plan 이 보는 벡터 후보 섹션. 비어 있으면 빈 문자열 (섹션 생략).

    증거마다 **종류(해설·화면·자막)를 붙이고 종류별로 한 줄씩** 낸다. 종류를 지우면
    모델이 "이게 사람이 한 말인지 화면 설명인지" 를 모르는데, system 프롬프트는
    증거가 태그를 넘어설 권한을 준다 — 무엇을 믿을지 정하려면 출처를 알아야 한다.

    scenes 를 주면 태그·이닝·점수를 같은 줄에 붙인다. 인벤토리와 이 블록이 번호로만
    이어져 있어 조인이 모델 몫이던 문제(audit) 대응.
    """
    if not evidence and not orphan:
        return ""
    meta = {r["scene_id"]: r for r in (scenes or [])}
    lines = ["[벡터 후보 — 질의와 의미가 가까운 검색 증거 (참고)]"]
    for g in evidence:
        m = meta.get(g["scene_id"])
        head = f"장면 {g['scene_id']}  유사도 {g['sim']:.2f}"
        if m:
            away = (m.get("score") or " ").split()
            head += (f"  {m['scene_type']}"
                     + (f"·{m['labels']}" if m.get("labels") else "")
                     + f"  {m['inning']}"
                     + (f"  {away[0]} {m['score_before']}→{away[1]}" if len(away) > 1 else ""))
        lines.append(head)
        for kind, texts in sorted(g.get("by_kind", {}).items()):
            lines.append(f"   {KIND_LABEL.get(kind, kind)} "
                         + " / ".join(f"\"{t[:EVIDENCE_TEXT_MAX]}\"" for t in texts[:2]))
    for o in orphan:
        lines.append(f"※ 장면 밖 {int(o['s'])}s: \"{o['text'][:60]}\" — 발행 장면 없음 (선곡 불가)")
    return "\n".join(lines)


def parse(text: str, scenes: list[dict]) -> dict:
    """LLM 응답(줄 형식) → 명세 dict. 선곡은 실존 scene_id 로 검산(select)."""
    spec = {"mode": "compose", "targets": [], "view": "전체",
            "budget": DEFAULT_BUDGET_SEC, "picked": [], "reason": "", "raw": text}
    known = {r["scene_id"] for r in scenes}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("모드:"):
            m = re.search(r"pinpoint|collection|compose", line)
            spec["mode"] = m.group(0) if m else "compose"
        elif line.startswith("대상:"):
            spec["targets"] = [t.strip() for t in line[3:].split(",") if t.strip()]
        elif line.startswith("관점:"):
            spec["view"] = ("홈" if "홈" in line else "원정" if "원정" in line else "전체")
        elif line.startswith("예산:"):
            m = re.search(r"\d+", line)
            if m:
                spec["budget"] = int(m.group(0))
        elif line.startswith("선곡:"):
            ids = [int(x) for x in re.findall(r"\d+", line)]
            spec["picked"] = [i for i in ids if i in known]        # select: 사실 검산
            if ghost := [i for i in ids if i not in known]:
                log.warning("선곡 검산: 실존하지 않는 장면 제거 %s", ghost)
        elif line.startswith("사유:"):
            spec["reason"] = line[3:].strip()
    return spec


def spec_line(spec: dict) -> str:
    """검수·리포트용 명세 한 줄."""
    return (f"모드={spec['mode']} 관점={spec['view']} "
            f"대상={','.join(spec['targets']) or '-'} 예산={spec['budget']}s")



# ── expand / verify 파서 (신설) ───────────────────────────

def parse_expand(text: str) -> tuple[list[str], list[str]]:
    """expand 응답 → (검색어, 필터). 형식이 깨지면 빈 값 — 호출부가 원 질의로 폴백한다."""
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
    """verify 응답 → {scene_id: {score, complete, reason}}.

    기각권이 없으므로 '무엇을 뺄까'가 아니라 '얼마나 맞나'만 읽는다. 파싱 실패한 줄은
    버린다 — 점수가 없는 클립은 select 가 기본값으로 다룬다(빠지지 않는다).
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
