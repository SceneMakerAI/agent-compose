"""편성 백그라운드 본체 — flow 실행 → 저장 → 잡 갱신 → 통보. HTTP 를 모른다."""

import asyncio
import time
from dataclasses import dataclass

from domains.baseball import callback
from domains.baseball.rows import clip_rows
from jobs import JobStore
from log import bind_v_id, get_logger
from rdb.composes import ComposeRepo, ComposeStatus

log = get_logger(__name__)


@dataclass(frozen=True)
class Order:
    """접수된 편성 요청."""

    v_id: int
    query: str
    stream_id: str | None
    budget_sec: int | None
    callback_url: str | None


# 노드 완료 → 다음 국면 (PLAN 은 create, OK/EMPTY 는 finish 가 찍는다)
_PHASE_AFTER = {
    "select_clips": ComposeStatus.CUT,
    "select_end_point": ComposeStatus.VERIFY,
}


async def execute(st, jobs: JobStore, comp_id: int, search_id: str, flow,
                  order: Order) -> None:
    """편성 1건 실행 — 잡 (v_id, comp_id) 는 호출부가 미리 만들어 둔다."""
    repo = ComposeRepo(st.db)
    job_key = (order.v_id, comp_id)
    progress: list[dict] = jobs.get(job_key)["progress"]
    started = time.monotonic()

    async def on_node(node: str, elapsed: float) -> None:
        progress.append({"node": node, "sec": elapsed})
        phase = _PHASE_AFTER.get(node)
        if phase is not None:
            await repo.set_status(order.v_id, comp_id, phase)

    try:
        with bind_v_id(order.v_id):
            state = await flow(
                order.v_id, comp_id, order.query, order.budget_sec, order.stream_id,
                st.db, st.llm, st.embedder, st.vector, st.settings, on_node=on_node)

            final = (
                ComposeStatus.EMPTY if state.get("status") == "empty" else ComposeStatus.OK)
            rows = clip_rows(state)
            await repo.finish(order.v_id, comp_id, final, rows)

        jobs.replace(job_key, {
            "comp_id": comp_id,
            "status": state.get("status", "ok"),
            "v_id": order.v_id,
            "query": order.query,
            "progress": progress,
            "budget_sec": order.budget_sec,
            "elapsed_sec": round(time.monotonic() - started, 1),
            "scene_count": len(state.get("scenes", [])),
            "spec": state.get("spec"),
            "evidence": state.get("evidence"),
            "evidence_orphan": state.get("evidence_orphan"),
            "candidates": state.get("candidates"),
            "picked": state.get("picked"),
            "clips": state.get("clips"),
            "dropped": state.get("dropped"),
            "duration_sec": sum(c["sec"] for c in (state.get("clips") or [])),
        })
        await _notify(order, search_id, 0, "OK", rows, st.settings)

    except asyncio.CancelledError:
        raise

    except Exception as e:
        log.exception("compose 실패: v_id=%s %r", order.v_id, order.query)
        await repo.set_status(order.v_id, comp_id, ComposeStatus.ERROR)
        jobs.replace(job_key, {
            "comp_id": comp_id,
            "status": "error", "v_id": order.v_id, "query": order.query,
            "progress": progress, "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": round(time.monotonic() - started, 1),
        })
        await _notify(order, search_id, -1, "fail", [], st.settings)


async def _notify(order: Order, search_id: str, code: int, result: str,
                  rows: list[dict], settings) -> None:
    if not order.callback_url:
        return
    payload = callback.build(order.v_id, search_id, order.query, code, result, rows)
    await callback.send(order.callback_url, payload, settings.callback_timeout)
