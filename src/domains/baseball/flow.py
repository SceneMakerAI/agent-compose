"""야구 편성 플로우 진입점 — pipeline.dispatch 가 cate_id(5100)로 run 을 찾는다.

그래프의 상태·노드·배선은 graph/ 패키지 소유 (노드 1개 = 모듈 1개).
여기는 실행만 담당한다: 그래프 빌드 → 트레이스 준비 → astream 으로 델타 누적.
"""

import time

from config import Settings
from domains.baseball.graph.build import build_graph
from domains.baseball.graph.state import ComposeState
from infer.chat import ChatLLM
from infer.embedder import Embedder
from infer.tracelog import InferTraceLog
from log import get_logger
from rdb.pool import Database
from vector.client import VectorClient

log = get_logger(__name__)


async def run(v_id: int, comp_id: int, query: str, budget_sec: int | None,
              db: Database, llm: ChatLLM, embedder: Embedder,
              vector: VectorClient, settings: Settings,
              on_node=None) -> ComposeState:
    """
    Summary:
        야구 편성 그래프 1회 실행 — pipeline.dispatch 가 호출.
    Args:
        v_id (int): 대상 영상 id.
        comp_id (int): 편성 id — 트레이스 디렉터리({v_id}_{comp_id}/)를 특정한다.
        query (str): 편성 질의.
        budget_sec (int | None): 목표 분량(초) — 마감 단계의 덜어내기 전용.
        db/llm/embedder/vector/settings: lifespan 공유 자원 (app.state).
        on_node: 노드 완료 **async** 콜백(node_name, elapsed_sec) — API 가 job 진행
            표시·국면 기록(DB)에 쓴다. elapsed_sec 은 그 노드의 소요 초.
            await 로 부르므로 국면 UPDATE 가 다음 노드 시작 전에 순서 보장된다.
    Returns:
        ComposeState: 최종 상태 — stream 델타 누적.
    Description:
        - astream 으로 노드 순서를 로그에 남기고 델타를 누적한다 — 진행 노출은
          이 스트림으로 충분 (checkpointer 미도입: 짧은 잡은 재실행이 더 싸다).
        - LLM 을 거치는 모든 노드는 트레이스를 남긴다 — ChatLLM.chat 에
          trace 를 물려주는 것으로 강제한다 ({v_id}_{comp_id}/{v_id}_{comp_id}_{node}.md).
    """
    graph = build_graph(db, llm, embedder, vector, settings)
    trace = InferTraceLog(settings.trace_dir, v_id, comp_id)
    state: ComposeState = {}
    prev = time.monotonic()
    async for step in graph.astream(
            {"v_id": v_id, "query": query, "budget_sec": budget_sec, "trace": trace},
            stream_mode="updates"):

        for node, upd in step.items():
            now = time.monotonic()
            elapsed = round(now - prev, 1)
            prev = now
            log.info("── 노드 %s 완료 (%.1fs, 갱신: %s)", node, elapsed,
                     ", ".join(k for k in (upd or ()) if k != "scenes") or "-")
            state.update(upd or {})
            if on_node:
                await on_node(node, elapsed)

    return state
