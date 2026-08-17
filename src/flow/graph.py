"""compose 그래프 — 질의 1건 → 편성 (bench4 compose/__init__ 이식 + 재배선).

그래프 (bench4 대비 endfix 를 route/backfill 뒤로 이동 — A1 수정):
  retrieve ──► plan ──► cutrank ──► (route) ──► endfix ──► verify ──► END
              ▲                       │            ▲
              └── feedback ◄──────────┤ (0건·여유)  │
                                      ├──► backfill ┘ (미달 — 충원분도 끝보정 받는다)
                                      └──► empty ──► END (0건·소진)

완화 사다리 (bench4 계승):
  L1   투구 앵커 없는 클립 → 통째 폴백 (cut 내장)
  L2   선곡 검산 후 0건 → feedback → plan 재선곡 (MAX_REPLAN 회, 어휘 오역 회수 전용)
  L2.5 예산 미달 → backfill 결정적 충원 (관점=전체 한정 — 미달 재선곡은 폐기)
  L3   그래도 0건 → status=empty

verify 는 소견 전용 — 기각권 없음 (태그 사실을 LLM 재심으로 뒤집는 오탐 실측).
retrieve 실패는 전파 — bench4 의 fail-open 폐기 (서비스 결정, design.md §2).
그래프는 무상태 배선이라 **프로세스당 1회 컴파일** (bench4 는 요청마다).
"""

import math
import re

from langgraph.graph import END, START, StateGraph

from flow import cut, plan, prompts, rank, vocab
from flow.llm import ChatLLM
from flow.state import ComposeState, Inventory
from log import get_logger
from vector.embedder import Embedder
from vector.store import QUERY_INSTRUCT, VectorStore

log = get_logger(__name__)

EVIDENCE_SCENES_MAX = 8      # plan 에 주입할 후보 장면 상한 (bench4 vector.py 운영값)
EVIDENCE_SNIPPETS_MAX = 2    # 장면당 증거 스니펫 상한
ORPHAN_MAX = 3               # 장면밖 증거 표기 상한 (발행 누락 의심 신호용)


def build_graph(llm: ChatLLM, embedder: Embedder, store: VectorStore):
    """자원 주입 배선 — lifespan 에서 1회 호출해 컴파일 결과를 공유한다."""

    async def retrieve_node(st: ComposeState) -> dict:
        qv = await embedder.embed_query(QUERY_INSTRUCT, st["query"])
        hits = await store.search(qv, st["inv"].v_id)
        ev, orphan = _group_hits(hits)
        if ev or orphan:
            log.info("retrieve: 후보 장면 %s, 장면밖 %d건",
                     [g["scene_id"] for g in ev], len(orphan))
        return {"evidence": ev, "evidence_orphan": orphan}

    async def plan_node(st: ComposeState) -> dict:
        inv = st["inv"]
        text = await llm.chat(
            prompts.PLAN_SYSTEM.replace("{budget}", str(plan.DEFAULT_BUDGET_SEC)),
            prompts.plan_user(st["query"], inv.game_line, inv.inventory_text,
                              plan.DEFAULT_BUDGET_SEC, st.get("feedback", ""),
                              plan.render_evidence(st.get("evidence", []),
                                                   st.get("evidence_orphan", []))),
            thinking=True,      # 선곡 추론에만 심층 사고 (verify 는 소견 전용이라 제외)
        )
        log.info("plan 응답: %r", text)
        spec = plan.parse(text, list(inv.scenes))
        if st.get("budget"):
            spec["budget"] = st["budget"]      # 명시 입력이 질의 해석보다 우선
        log.info("plan: %s 선곡 %s", plan.spec_line(spec), spec["picked"])
        return {"spec": spec, "attempt": st.get("attempt", 0) + 1}

    async def cutrank_node(st: ComposeState) -> dict:
        inv = st["inv"]
        pool = rank.order(list(inv.scenes), st["spec"]["picked"])   # 복사본 행
        for r in pool:
            r["cut"] = cut.clip(r, list(inv.segs), inv.utts)
        picked, spare = _assemble(pool, st["spec"]["budget"])
        total = sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked)
        return {"picked": picked, "spare": spare, "total": total}

    def route(st: ComposeState) -> str:
        """cutrank 뒤 분기 — 예비가 남아 있으면 예산이 한계였던 것이므로 정상."""
        can_retry = st["attempt"] <= vocab.MAX_REPLAN
        if not st["picked"]:
            return "feedback" if can_retry else "empty"
        underfill = (not st["spare"]
                     and st["total"] < st["spec"]["budget"] * vocab.UNDERFILL_MIN_FRAC)
        if underfill:
            log.info("예산 미달 %d/%ds — 대상 안 backfill 만 (순수성 유지)",
                     st["total"], st["spec"]["budget"])
            return "backfill"
        return "endfix"

    def feedback_node(st: ComposeState) -> dict:
        # 0건 전용 — 어휘 오역(질의를 태그로 잘못 번역) 회수용 재선곡
        return {"feedback": (f"선곡 {st['spec']['picked']}이 검산에서 비었다. "
                             f"질의를 어휘로 다시 번역해 골라라.")}

    def backfill_node(st: ComposeState) -> dict:
        inv = st["inv"]
        picked, total = _backfill(list(inv.scenes), list(inv.segs), inv.utts,
                                  st["spec"], list(st["picked"]), st["total"])
        return {"picked": picked, "total": total}       # A3: total 도 갱신

    async def endfix_node(st: ComposeState) -> dict:
        # A1: route/backfill 뒤에 실행 — 충원 클립도 같은 끝 보정을 받는다
        picked = [dict(r, cut=dict(r["cut"])) for r in st["picked"]]   # 복사 후 수정
        rows = []
        for r in picked:
            ce = r["cut"]["ce"]
            # 후보 = 검증기가 수용할 수 있는 발화만 (상한 밖 제시 → 제안 전멸 실측 comp 28)
            near = [(us, ue, t) for us, ue, t in st["inv"].utts
                    if ce < ue <= ce + vocab.ENDFIX_MAX_EXT_SEC][:vocab.ENDFIX_UTT_MAX]
            if near:
                rows.append((r["scene_id"], ce, near))
        if not rows:
            return {}
        text = await llm.chat(prompts.ENDFIX_SYSTEM, prompts.endfix_user(rows))
        moved = _apply_endfix(picked, rows, text)
        if moved:
            log.info("endfix: 끝 이동 %s", moved)
        return {"picked": picked, "endfix_moved": moved,
                "total": sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked)}

    async def verify_node(st: ComposeState) -> dict:
        # verify = 소견 전용 — 의심은 리포트로, 클립은 유지
        text = await llm.chat(
            prompts.VERIFY_SYSTEM,
            prompts.verify_user(plan.spec_line(st["spec"]),
                                _packets(st["inv"], st["picked"])))
        log.info("verify 응답: %r", text)
        suspected, per, common = plan.parse_verify(text)
        if suspected:
            log.warning("verify 의심 %d건 (클립 유지): %s", len(suspected), suspected)
        # A2: 클립별 사유 매핑 — 없는 클립은 공통 소견으로
        return {"suspicions": [(i, per.get(i, common)) for i in suspected], "status": "ok"}

    def empty_node(st: ComposeState) -> dict:
        return {"picked": [], "status": "empty"}

    g = StateGraph(ComposeState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("plan", plan_node)
    g.add_node("cutrank", cutrank_node)
    g.add_node("feedback", feedback_node)
    g.add_node("backfill", backfill_node)
    g.add_node("endfix", endfix_node)
    g.add_node("verify", verify_node)
    g.add_node("empty", empty_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "cutrank")
    g.add_conditional_edges("cutrank", route,
                            {"feedback": "feedback", "backfill": "backfill",
                             "endfix": "endfix", "empty": "empty"})
    g.add_edge("feedback", "plan")
    g.add_edge("backfill", "endfix")
    g.add_edge("endfix", "verify")
    g.add_edge("verify", END)
    g.add_edge("empty", END)
    return g.compile()


async def run_compose(graph, inv: Inventory, query: str, budget: int | None,
                      on_node=None) -> ComposeState:
    """그래프 1회 실행 — stream 으로 노드 순서를 로그에 남기고 델타를 누적한다.

    on_node(node_name): 노드 완료 콜백 — API 가 job 진행 표시에 쓴다 (폴링 응답의
    progress 필드). checkpointer 는 도입하지 않는다 — 1~3분 잡은 재실행이 더 싸고,
    진행 노출은 이 스트림으로 충분 (Phase 4 검토 2026-08-18).
    """
    state: ComposeState = {}
    async for step in graph.astream({"query": query, "budget": budget, "inv": inv},
                                    stream_mode="updates"):
        for node, upd in step.items():
            log.info("── 노드 %s 완료 (갱신: %s)", node,
                     ", ".join(k for k in (upd or ()) if k != "inv") or "-")
            state.update(upd or {})
            if on_node:
                on_node(node)
    return state


# ──────────────────────────────────────────────────────
# 순수 보조 (bench4 원문 이식)
# ──────────────────────────────────────────────────────

def _group_hits(hits: list[dict]) -> tuple[list[dict], list[dict]]:
    """검색 히트 → 장면 그룹(히트수→유사도 정렬 상위) + 장면밖 (bench4 vector.retrieve 후반)."""
    groups: dict[int, dict] = {}
    orphan = []
    for h in hits:
        sid = h["scene_id"]
        if sid < 0:
            orphan.append(h)
            continue
        g = groups.setdefault(sid, {"scene_id": sid, "hits": 0, "sim": 0.0, "snippets": []})
        g["hits"] += 1
        g["sim"] = max(g["sim"], h["distance"])
        if len(g["snippets"]) < EVIDENCE_SNIPPETS_MAX:
            g["snippets"].append(h["text"])
    ev = sorted(groups.values(), key=lambda g: (-g["hits"], -g["sim"]))[:EVIDENCE_SCENES_MAX]
    return ev, orphan[:ORPHAN_MAX]


def _assemble(pool: list[dict], budget: int) -> tuple[list[dict], list[dict]]:
    """cut 후 실제 길이 기준으로 예산에 채운다 — 초과분은 예비 풀.
    채택분은 시간순 배치 (서사 = 시간순 v1)."""
    picked, spare, total = [], [], 0
    for r in pool:
        d = r["cut"]["ce"] - r["cut"]["cs"]
        if total + d <= budget:
            picked.append(r)
            total += d
        else:
            spare.append(r)
    return sorted(picked, key=lambda r: r["cut"]["cs"]), spare


def _backfill(scenes, segs, utts, spec, picked: list[dict],
              total: int) -> tuple[list[dict], int]:
    """L2.5 결정적 충원 — plan 이 정한 대상 태그·라벨 부합 미선곡 장면을 rank 순으로.

    관점=전체일 때만 (홈/원정 플레이 기계 판별 불가). 재선곡은 비결정적이라 폐기 —
    마지막 채움은 사실 기반 계산 (bench4 실측). 반환에 total 포함 (A3 수정).
    """
    if spec["view"] != "전체" or not spec["targets"]:
        return picked, total
    have = {r["scene_id"] for r in picked}
    tg = set(spec["targets"])
    cands = [r for r in scenes if r["scene_id"] not in have
             and (tg & set(r["tags"]) or tg & set(r["label_list"]))]
    added = 0
    for r in sorted(cands, key=lambda r: (-rank.score(r), r["s"])):
        r = dict(r)                                    # 복사 후 수정 (B4)
        r["cut"] = cut.clip(r, segs, utts)
        d = r["cut"]["ce"] - r["cut"]["cs"]
        if total + d <= spec["budget"]:
            picked.append(r)
            total += d
            added += 1
    if added:
        log.info("예산 미달 충원: 대상 부합 %d장면 추가 → %ds", added, total)
    return sorted(picked, key=lambda r: r["cut"]["cs"]), total


def _apply_endfix(picked: list[dict], rows, text: str) -> list[str]:
    """endfix 제안 처분 — "장면 N: 유지|<초>" 파싱 후 검증 통과분만 적용.

    수용 조건: ① 제안 초가 그 클립에 제시한 발화의 실제 끝(올림)과 일치(±1s 허용)
    ② 현재 끝보다 뒤 ③ 연장 폭 ENDFIX_MAX_EXT_SEC 이내. 그 외는 무시 (임의 초 기각).
    """
    shown = {sid: (ce, near) for sid, ce, near in rows}
    by_id = {r["scene_id"]: r for r in picked}
    moved = []
    for line in text.splitlines():
        m = re.match(r"장면\s*(\d+)\s*:\s*(\d+)", line.strip())
        if not m:
            continue
        sid, new_end = int(m.group(1)), int(m.group(2))
        if sid not in shown or sid not in by_id:
            continue
        ce, near = shown[sid]
        ok = any(abs(new_end - math.ceil(ue)) <= 1 for _, ue, _ in near)
        if ok and ce < new_end <= ce + vocab.ENDFIX_MAX_EXT_SEC:
            cut_ = by_id[sid]["cut"]
            cut_["ce"] = new_end
            cut_["mode"] += "+끝보정(agent)"
            moved.append(f"장면{sid} {ce}→{new_end}")
        elif new_end != ce:
            log.info("endfix 기각: 장면%d %d→%d (발화 끝 불일치 또는 상한 초과)",
                     sid, ce, new_end)
    return moved


def _packets(inv: Inventory, clips: list[dict]) -> str:
    """채택 클립들 → 검수 패킷 텍스트 (bench4 verify.packets — DB 재조회 대신 인벤토리).

    대사는 장면 전체 구간에서 — 잘린 클립(투구 샷 3초)으로 좁히면 이웃 플레이 해설이
    섞여 증거가 오염된다 (실측: 삼진 22건 전멸 오기각의 한 원인).
    """
    lines = []
    for c in clips:
        shots = " → ".join(t or "미분류" for _, _, t in c["cut"]["shots"][:6]) or "-"
        overlap = [t for us, ue, t in inv.utts if us < c["e"] and ue > c["s"]]
        stt = " / ".join(overlap[:3])[:200]            # bench4 LIMIT 3 등가
        lines.append(
            f"[{c['scene_id']}] {c['scene_type']} ({c['labels'] or '라벨없음'}) "
            f"{c['inning']} {c['score_before']}→{c['score'].split()[1]}\n"
            f"    화면: {shots}\n"
            f"    대사: {stt or '(없음)'}")
    return "\n".join(lines)
