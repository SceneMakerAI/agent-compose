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
import time
from pathlib import Path
from typing import Any

from log import get_logger

log = get_logger(__name__)


class Trace:
    """편성 1회 분량의 수집기. 비활성이면 모든 메서드가 즉시 반환한다."""

    def __init__(self, root: str | None, v_id: int, query: str) -> None:
        self._dir = Path(root) if root else None
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
        """파일로 떨군다. 이름은 comp_id 기준 — 편성 결과와 1:1 로 짝지어야 대조가 된다."""
        if not self.on:
            return
        self._data.update(
            comp_id=comp_id, status=status,
            elapsed=round(time.monotonic() - self._t0, 1), **fields)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = f"comp{comp_id}" if comp_id else f"v{self._data['v_id']}-failed"
            path = self._dir / f"{stem}.json"
            path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("트레이스 기록: %s (%d노드 · LLM %d콜)",
                     path, len(self._data["nodes"]), len(self._data["llm"]))
        except OSError as e:                 # 관측 실패가 편성을 죽이면 안 된다
            log.warning("트레이스 기록 실패(무시): %s", e)
