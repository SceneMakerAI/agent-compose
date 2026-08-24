"""실행 트레이스 — 편성 1건 = JSON 1개 (노드 입출력·LLM 프롬프트/응답 전문).

왜 로그가 아니라 파일인가:
- plan user 프롬프트만 v201 기준 5,883자다. 전역 LOG_LEVEL=DEBUG 로 켜면 운영 로그를
  덮어버린다 (실측).
- A/B 를 판정하려면 "같은 입력에 왜 다른 선곡이 나왔나"를 **파일 두 개 diff** 로 봐야
  한다. 줄 단위 로그로는 프롬프트가 흩어져 대조가 안 된다.

끄면(TRACE_DIR 빈 값) 아무 일도 하지 않는다 — 수집 자체가 no-op 이라 운영 부담이 없다.
트레이스 기록 실패는 편성을 죽이지 않는다 (관측용이지 산출물이 아니다).
"""

import json
import re
import time
from pathlib import Path
from typing import Any

from log import get_logger

log = get_logger(__name__)

# src/ 의 부모 = 레포 루트. 상대 TRACE_DIR 은 여기 기준으로 푼다.
_REPO_ROOT = Path(__file__).resolve().parents[1]

_BACKTICKS = re.compile(r"`+")


class Trace:
    """편성 1회 분량의 수집기. 비활성이면 모든 메서드가 즉시 반환한다."""

    def __init__(self, root: str | None, v_id: int, query: str) -> None:
        # 상대 경로는 **레포 루트 기준**으로 푼다 — 서비스의 CWD 에 좌우되면 트레이스가
        # 어디에 떨어졌는지 찾을 수 없다. 기본값 logs/trace 도 레포 안에 남는다.
        self._dir = None
        if root:
            p = Path(root)
            self._dir = p if p.is_absolute() else _REPO_ROOT / p
        self._t0 = time.monotonic()
        self._data: dict[str, Any] = {
            "v_id": v_id, "query": query, "nodes": [], "llm": [],
        }

    @property
    def on(self) -> bool:
        return self._dir is not None

    def node(self, name: str, **fields: Any) -> None:
        """노드 1개의 입출력 요약 — 값까지 남긴다 (키 이름만으로는 대조가 안 된다)."""
        if not self.on:
            return
        self._data["nodes"].append(
            {"node": name, "at": round(time.monotonic() - self._t0, 1), **fields})

    def llm(self, name: str, system: str, user: str, response: str,
            thinking: str | None = None, elapsed: float | None = None) -> None:
        """LLM 콜 1회 — 프롬프트·응답 **전문**. 잘라 남기면 재현이 안 된다."""
        if not self.on:
            return
        self._data["llm"].append({
            "call": name, "elapsed": round(elapsed, 1) if elapsed else None,
            "system": system, "user": user, "response": response, "thinking": thinking,
        })

    def finish(self, comp_id: int | None, status: str, **fields: Any) -> None:
        """파일로 떨군다. 이름은 {v_id}-{comp_id} — 영상별로 묶여 보이고 편성과 1:1 이다."""
        if not self.on:
            return
        self._data.update(
            comp_id=comp_id, status=status,
            elapsed=round(time.monotonic() - self._t0, 1), **fields)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            v_id = self._data["v_id"]
            stem = f"{v_id}-{comp_id}" if comp_id else f"{v_id}-failed"
            path = self._dir / f"{stem}.json"
            path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            # 사람이 읽는 짝 — JSON 은 thinking 이 한 줄에 \n 이스케이프로 뭉쳐(실측
            # comp8: 33,863·90,111·67,694자) 눈으로 못 읽는다. 둘 다 남긴다:
            # .json 은 diff·도구, .md 는 사람.
            (self._dir / f"{stem}.md").write_text(render_md(self._data), encoding="utf-8")
            log.info("트레이스 기록: %s (+.md) (%d노드 · LLM %d콜)",
                     path, len(self._data["nodes"]), len(self._data["llm"]))
        except OSError as e:                 # 관측 실패가 편성을 죽이면 안 된다
            log.warning("트레이스 기록 실패(무시): %s", e)


# ──────────────────────────────────────────────────────
# 사람이 읽는 렌더 (JSON → Markdown)
# ──────────────────────────────────────────────────────

def _fence(text: str, lang: str = "text") -> str:
    """코드펜스 — 본문에 백틱 3개가 있으면 울타리를 늘려 깨지지 않게 한다."""
    body = (text or "").rstrip()
    bar = "`" * max(3, max((len(m) for m in _BACKTICKS.findall(body)), default=0) + 1)
    return f"{bar}{lang}\n{body}\n{bar}"


def _dur(sec: float | None) -> str:
    """초 → '3분 14초' (사람이 읽는 단위). 1분 미만은 소수 1자리."""
    if sec is None:
        return "-"
    return f"{int(sec) // 60}분 {int(sec) % 60}초" if sec >= 60 else f"{sec:.1f}초"


def render_md(d: dict[str, Any]) -> str:
    """
    Summary:
        트레이스 dict → 읽기용 Markdown. 전문은 그대로 두되 줄바꿈을 살린다.
    Description:
        요약(무슨 일이 일어났나) → 노드 타임라인 → LLM 콜 전문 순. 노드가 먼저인 이유:
        "어디서 시간을 썼나"를 먼저 보고 그 콜만 펼쳐 읽게 된다 (전문을 위에 두면
        스크롤로 타임라인을 못 찾는다).
    """
    v_id, comp_id = d.get("v_id"), d.get("comp_id")
    out = [f"# v{v_id} · comp{comp_id} — {d.get('query', '')}", ""]
    out += [(f"- 결과: **{d.get('status')}** · 클립 {d.get('clips', '?')}건 "
             f"· {d.get('total', '?')}초 · 총 {_dur(d.get('elapsed'))}"),
            f"- LLM {len(d.get('llm', []))}콜 / 노드 {len(d.get('nodes', []))}개", ""]

    out += ["## 노드 타임라인", ""]
    nodes = d.get("nodes", [])
    prev = 0.0
    for n in nodes:
        # at 은 노드 **완료** 시각이다(수집이 노드 끝에서 일어난다). 소요는 직전 완료와의
        # 차이 — 다음 노드의 at 을 쓰면 한 칸씩 밀려 엉뚱한 노드가 범인이 된다.
        at = n.get("at", 0)
        out.append(f"### {n['node']} — {_dur(at - prev)} 소요 ({_dur(at)} 지점 완료)")
        prev = at
        for k, v in n.items():
            if k in ("node", "at"):
                continue
            out.append(f"- **{k}**: {_compact(v)}")
        out.append("")

    if dropped := d.get("dropped"):
        out += ["## 탈락", ""]
        out += [f"- 장면 {sid} — {why}" for sid, why in dropped] + [""]

    out += ["## LLM 콜 전문", ""]
    for i, c in enumerate(d.get("llm", []), 1):
        out.append(f"### [{i}] {c.get('call')} — {_dur(c.get('elapsed'))}")
        out += ["", "**system**", "", _fence(c.get("system", "")), ""]
        out += ["**user**", "", _fence(c.get("user", "")), ""]
        if think := c.get("thinking"):
            out += [f"<details><summary>thinking ({len(think):,}자)</summary>", "",
                    _fence(think), "", "</details>", ""]
        out += ["**응답**", "", _fence(c.get("response", "")), ""]
    return "\n".join(out)


def _compact(v: Any) -> str:
    """노드 필드 1개 → 한 줄. 긴 목록은 앞부분만 (전문은 .json 에 있다)."""
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= 600 else s[:600] + f"… (총 {len(s):,}자, 전문은 .json)"


__all__ = ["Trace", "render_md"]
