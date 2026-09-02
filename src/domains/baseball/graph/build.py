"""그래프 배선 — 노드 등록·엣지·분기는 여기 한 곳에만 있다.

그래프:

  load_inventory ─► parse_query ─► retrieve_evidence
    ─► select_clips ─► select_end_point ─► trim_budget ─► END
"""

from langgraph.graph import END, START, StateGraph

from config import Settings
from domains.baseball.graph import (
    load_inventory,
    parse_query,
    retrieve_evidence,
    select_clips,
    select_end_point,
    trim_budget,
)
from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.evidences import EvidenceRepo
from domains.baseball.repo.scenes import SceneRepo
from domains.baseball.repo.teams import TeamRepo
from infer.chat import ChatLLM
from infer.embedder import Embedder
from rdb.pool import Database
from vector.client import VectorClient


def build_graph(db: Database, llm: ChatLLM, embedder: Embedder,
                vector: VectorClient, settings: Settings):
    """자원 주입 배선 — 그래프를 컴파일해 돌려준다.

    지금은 요청마다 빌드한다 — 자원이 다 배선돼 구성이 굳으면 lifespan 1회
    컴파일(app.state 공유)로 옮긴다 (요청마다 재컴파일 금지 원칙).
    """
    scene_repo = SceneRepo(db)
    evidence_repo = EvidenceRepo(vector, settings)
    team_repo = TeamRepo(db)

    g = StateGraph(ComposeState)

    g.add_node("load_inventory", load_inventory.make_node(scene_repo))
    g.add_node("parse_query", parse_query.make_node(llm, evidence_repo, team_repo))
    g.add_node("retrieve_evidence", retrieve_evidence.make_node(embedder, evidence_repo))
    g.add_node("select_clips", select_clips.make_node(llm, evidence_repo,
                                                      settings.select_tokens_max))
    g.add_node("select_end_point", select_end_point.make_node(llm, evidence_repo))
    g.add_node("trim_budget", trim_budget.make_node(settings.budget_margin))

    g.add_edge(START, "load_inventory")
    g.add_edge("load_inventory", "parse_query")
    g.add_edge("parse_query", "retrieve_evidence")
    g.add_edge("retrieve_evidence", "select_clips")
    g.add_edge("select_clips", "select_end_point")
    g.add_edge("select_end_point", "trim_budget")
    g.add_edge("trim_budget", END)

    return g.compile()
