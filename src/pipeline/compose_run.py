"""편성 백그라운드 본체 — flow 실행 → 저장 → 통보. HTTP 를 모른다."""

import asyncio
from dataclasses import dataclass

from domains.baseball import callback
from domains.baseball.rows import clip_rows
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


async def execute(st, comp_id: int, search_id: str, flow, order: Order) -> None:
    """편성 1건 실행 — t_compose 행(PLAN)은 호출부가 미리 만들어 둔다."""
    repo = ComposeRepo(st.db)

    async def on_node(node: str, elapsed: float) -> None:
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

        await _notify(order, search_id, 0, "OK", rows, st.settings)

    except asyncio.CancelledError:
        log.warning("compose 취소: v_id=%s comp_id=%s (종료 중)", order.v_id, comp_id)
        # 뒷정리 중 재취소돼도 ERROR 기록·통보는 끝까지 간다
        await asyncio.shield(_fail(repo, comp_id, order, search_id, st.settings))
        raise

    except Exception:
        log.exception("compose 실패: v_id=%s %r", order.v_id, order.query)
        await _fail(repo, comp_id, order, search_id, st.settings)


async def _fail(repo: ComposeRepo, comp_id: int, order: Order, search_id: str,
                settings) -> None:
    """실패·취소 공통 뒷정리 — ERROR 기록 → fail 통보."""
    await repo.set_status(order.v_id, comp_id, ComposeStatus.ERROR)
    await _notify(order, search_id, -1, "fail", [], settings)


async def _notify(order: Order, search_id: str, code: int, result: str,
                  rows: list[dict], settings) -> None:
    if not order.callback_url:
        return
    payload = callback.build(order.v_id, search_id, order.query, code, result, rows)
    await callback.send(order.callback_url, payload, settings.callback_timeout)
