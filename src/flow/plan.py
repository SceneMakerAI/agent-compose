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


def render_evidence(evidence: list[dict], orphan: list[dict]) -> str:
    """retrieve 결과 → plan 이 보는 벡터 후보 섹션. 비어 있으면 빈 문자열 (섹션 생략)."""
    if not evidence and not orphan:
        return ""
    lines = ["[벡터 후보 — 질의와 의미가 가까운 검색 증거 (참고)]"]
    for g in evidence:
        sn = " / ".join(f"\"{t[:60]}\"" for t in g["snippets"])
        lines.append(f"장면 {g['scene_id']} (증거 {g['hits']}건, 유사도 {g['sim']:.2f}): {sn}")
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


def parse_verify(text: str) -> tuple[list[int], dict[int, str], str]:
    """verify 응답 → (의심 scene_id, 장면별 사유, 공통 사유). 파싱 실패 = 의심 없음.

    A2 수정: 사유를 "장면 N: 한 줄" 로 클립별 매핑 — bench4 는 마지막 사유 한 줄을
    전 의심 클립에 복사했다. 매핑 안 되는 사유 줄은 공통 소견으로 강등.
    """
    if not text.strip().startswith("판정"):
        log.warning("verify 형식 밖 응답 — 기각 없음 처리")
        return [], {}, ""
    rejected: list[int] = []
    per: dict[int, str] = {}
    common: list[str] = []
    in_reason = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("기각:"):
            in_reason = False
            if "없음" not in line:
                rejected = [int(x) for x in re.findall(r"\d+", line)]
        elif line.startswith("사유:"):
            in_reason = True
            line = line[3:].strip()
            if not line or line == "없음":
                continue
        elif not in_reason:
            continue
        m = re.match(r"장면\s*(\d+)\s*[:：]\s*(.+)", line)
        if m:
            per[int(m.group(1))] = m.group(2).strip()
        elif line and line != "없음":
            common.append(line)
    return rejected, per, " ".join(common)
