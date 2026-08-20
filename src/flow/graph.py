"""compose 그래프 — 질의 1건 → 편성.

  rephrase_query ─► retrieve_evidence ─► select_clips ─► set_bounds
     ─► refine_bounds ─► score_match ─► drop_unmatched ─► order_clips
     ─► fill_budget ─► END
                          ▲
     select_clips ◄── retry_select ◄──┤ (0건·재시도 여유)
                       end_empty ◄────┘ (0건·소진) ──► END

노드 이름은 전부 **동사_목적어**이고, 그 노드가 무엇을 만들어 내는지를 가리킨다
(2026-08-20 개편 — 구 expand·retrieve·plan·cut·bounds·verify·drop0·rank·select).
구현을 이름에 넣지 않는다: 구 `drop0` 은 임계값이 바뀌면 거짓말이 되는 이름이었고,
LLM 사용 여부도 이름에 박지 않는다 — 이 파이프라인에선 단계가 LLM↔규칙 사이를
오간 전례가 있다(구 bounds 의 시작 판정이 set_bounds 규칙으로 내려온 것처럼).

2026-08-20 재배선 — cutrank 를 해체했다:
- **절단이 마지막**(fill_budget). 예전에는 경계 확정 전에 잘랐고 그 뒤 끝 보정이
  끝을 늘려 예산 보장이 무효가 됐다 (실측: 900초 요청에 947~1018초).
- **set_bounds 가 정하고 refine_bounds 가 다듬는다.** 시작·끝을 함께 보는 이유는
  따로 물으면 서로를 모르기 때문이다 — 시작을 25초 옮기면 끝의 여유도 달라진다.
- **score_match 가 값을 한다**. 예전에는 소견 전용이라 출력물을 바꾸는 경로가 아예
  없었다. 이제 클립별 일치도(0~3)를 매기고 fill_budget 이 그 점수를 쓴다. 기각권은
  여전히 없다 — 필수층(득점·역전·동점·끝내기·경기 종료)은 0점이어도 유지한다.
- **rephrase_query** 가 질의를 중계의 언어로 옮긴다 (실측: 추상 질의 최고 유사도
  0.58 vs 구체 질의 0.66~0.78).

완화 사다리:
  L1   투구 앵커 없는 클립 → 통째 폴백 (set_bounds 내장)
  L2   선곡 검산 후 0건 → retry_select → select_clips 재선곡 (MAX_REPLAN 회)
  L2.5 전 클립이 예산 초과 → 최단 1건 구제 (select.rescue_longest)
  L3   그래도 0건 → status=empty

retrieve_evidence 실패는 전파 — bench4 의 fail-open 폐기 (design.md §2).
그래프는 무상태 배선이라 프로세스당 1회 컴파일.
"""

import asyncio
import inspect

from langgraph.graph import END, START, StateGraph

from flow import bounds as bounds_mod
from flow import cut, plan, prompts, rank, vocab
from flow import select as select_mod
from flow.llm import ChatLLM
from flow.state import ComposeState, Inventory
from log import get_logger
from vector.embedder import Embedder
from vector.store import QUERY_INSTRUCT, VectorStore

log = get_logger(__name__)

EVIDENCE_SCENES_MAX = 8      # select_clips 에 주입할 후보 장면 상한 (bench4 vector.py 운영값)
EVIDENCE_SNIPPETS_MAX = 2    # 장면당 증거 스니펫 상한
ORPHAN_MAX = 3               # 장면밖 증거 표기 상한 (발행 누락 의심 신호용)


def build_graph(llm: ChatLLM, embedder: Embedder, store: VectorStore, settings=None):
    """자원 주입 배선 — lifespan 에서 1회 호출해 컴파일 결과를 공유한다.

    settings 의 use_* 플래그로 새 단계를 끌 수 있다 — 여러 변경을 한꺼번에 넣으면
    "무엇이 달라졌나"는 트레이스로 봐도 "어느 변경 때문인가"는 갈리지 않는다.
    """
    # Settings 필드명(use_expand 등)은 .env 계약이라 유지 — 지역 변수만 노드명에 맞춘다.
    use_rephrase = getattr(settings, "use_expand", True)
    use_refine = getattr(settings, "use_bounds", True)
    use_fill = getattr(settings, "use_select", True)

    async def rephrase_query_node(st: ComposeState) -> dict:
        """질의 → 중계 문장형 검색어 + 메타 필터 힌트. 실패하면 원 질의로 폴백."""
        if not use_rephrase:
            return {"phrases": [st["query"]], "filters": []}
        try:
            text = await llm.chat(prompts.EXPAND_SYSTEM, prompts.expand_user(st["query"]),
                                  thinking=False, trace=st.get("trace"),
                                  name="rephrase_query")
            phrases, filters = plan.parse_expand(text)
        except Exception as e:                       # noqa: BLE001 — 보조 단계, 죽이지 않는다
            log.warning("rephrase_query 실패(원 질의로 진행): %s", e)
            phrases, filters = [], []
        phrases = phrases or [st["query"]]
        log.info("rephrase_query: 검색어 %s / 필터 %s", phrases, filters or "-")
        if tr := st.get("trace"):
            tr.node("rephrase_query", phrases=phrases, filters=filters)
        return {"phrases": phrases, "filters": filters}

    async def retrieve_evidence_node(st: ComposeState) -> dict:
        """검색어마다 임베딩해 병합 — 필터 건 검색과 안 건 검색을 함께 돌린다.

        하드 필터는 쓰지 않는다: 장면 65 는 기록이 '범타'인데 화면은 호수비였다.
        필터로만 좁혔다면 영영 못 찾는다 — 병합 + 우선권이지 배타적 선택이 아니다.
        """
        v_id = st["inv"].v_id
        hits: list[dict] = []
        for phrase in st.get("phrases") or [st["query"]]:
            qv = await embedder.embed_query(QUERY_INSTRUCT, phrase)
            hits += await store.search(qv, v_id)
            for f in st.get("filters") or []:
                hits += await store.search(qv, v_id, extra=_label_filter(f))
        ev, orphan = _group_hits(_dedup_hits(hits))
        if ev or orphan:
            log.info("retrieve_evidence: 후보 장면 %s, 장면밖 %d건",
                     [g["scene_id"] for g in ev], len(orphan))
        if tr := st.get("trace"):
            tr.node("retrieve_evidence", candidates=[g["scene_id"] for g in ev],
                    orphan=len(orphan), hits=len(hits))
        return {"evidence": ev, "evidence_orphan": orphan}

    async def select_clips_node(st: ComposeState) -> dict:
        inv = st["inv"]
        # 호출자 예산을 LLM 에게도 알린다 — 기본값만 보내던 탓에 900s 요청에도 모델이
        # "예산: 180" 으로 답하며 그 전제로 선곡했다 (2026-08-19 실측 로그).
        budget = st.get("budget") or plan.DEFAULT_BUDGET_SEC
        text = await llm.chat(
            prompts.PLAN_SYSTEM.replace("{budget}", str(budget)),
            prompts.plan_user(
                st["query"], inv.game_line, inv.inventory_text, budget,
                st.get("feedback", ""),
                # orphan 은 프롬프트에서 뺀다 (선곡 불가라 자리만 차지)
                plan.render_evidence(st.get("evidence", []), [], list(inv.scenes)),
            ),
            thinking=True, trace=st.get("trace"), name="select_clips",
        )
        log.info("select_clips 응답: %r", text)
        spec = plan.parse(text, list(inv.scenes))
        if st.get("budget"):
            spec["budget"] = st["budget"]      # 명시 입력이 질의 해석보다 우선
        spec["picked"] = list(dict.fromkeys(spec["picked"]))   # 중복 선곡 제거
        log.info("select_clips: %s 선곡 %s", plan.spec_line(spec), spec["picked"])
        if tr := st.get("trace"):
            tr.node("select_clips", spec={k: v for k, v in spec.items() if k != "raw"})
        return {"spec": spec, "attempt": st.get("attempt", 0) + 1}

    async def set_bounds_node(st: ComposeState) -> dict:
        """구간 확정 — 순위도 절단도 하지 않는다 (구 cutrank 의 1/3).

        선곡분에 **회수 대상 필수 장면을 합쳐서** 컷한다. fill_budget 의 필수층이
        일하려면 그 장면이 후보에 있어야 하는데, select_clips 가 놓치면 영영 들어올 수
        없기 때문이다 (구 backfill 이 하던 회수를 여기서 보장한다). 무엇을 회수하는지는
        select.recover_must 가 정한다 — pinpoint 는 회수하지 않고, 그 외에도 질의
        대상과 겹치는 라벨만 끌어온다(질의를 회수가 덮어쓰지 않게).
        구간 계산은 순수 함수라 몇 건 늘어도 비용이 없고, LLM 을 타는
        refine_bounds·score_match 의 입력만 그만큼 커진다.
        """
        inv = st["inv"]
        picked = list(st["spec"]["picked"])
        must = select_mod.recover_must(list(inv.scenes), picked, st["spec"])
        if must:
            log.info("set_bounds: 필수 장면 회수 %s (select_clips 미선곡)", must)
        clips = rank.order(list(inv.scenes), picked + must)
        for r in clips:
            r["cut"] = cut.clip(r, list(inv.segs), inv.utts)
        if tr := st.get("trace"):
            tr.node("set_bounds",
                    clips=[(r["scene_id"], round(r["cut"]["cs"]), round(r["cut"]["ce"]))
                           for r in clips])
        return {"clips": clips}

    def route(st: ComposeState) -> str:
        if st.get("clips"):
            return "refine_bounds"
        return "retry_select" if st["attempt"] <= vocab.MAX_REPLAN else "end_empty"

    def retry_select_node(st: ComposeState) -> dict:
        return {"feedback": (f"선곡 {st['spec']['picked']}이 검산에서 비었다. "
                             f"질의를 어휘로 다시 번역해 골라라.")}

    async def refine_bounds_node(st: ComposeState) -> dict:
        """시작·끝을 함께 정한다. 후보는 결정적으로 만들고 LLM 은 고르기만 한다.

        **클립 1건 = 콜 1건으로 펼쳐 동시에 보낸다.** 한 콜에 몰면 전송이 직렬이라
        GPU 가 놀았다 (실측 v201: 24클립 10분 26초 동안 서버는 내내 Running 1 ·
        KV 3%). 경계 판정은 클립끼리 독립이라 나눠도 근거가 줄지 않고, 응답을
        **자기 행에만** 대조하므로 한 콜의 헛번호가 남의 클립을 못 건드린다.
        """
        inv = st["inv"]
        clips = [dict(r, cut=dict(r["cut"])) for r in st["clips"]]     # 복사 후 수정
        if not use_refine:
            return {"clips": clips}
        rows = bounds_mod.build_rows(clips, list(inv.segs), inv.utts, list(inv.pitches))
        log.info("refine_bounds 대상 %d/%d클립 — 나머지는 투구 앵커가 있어 set_bounds 경계 그대로",
                 len(rows), len(clips))
        if not rows:
            return {"clips": clips}

        async def one(row: dict) -> str:
            """행 1건 질의 — 실패는 그 클립만 원 경계 유지 (배치 전체를 죽이지 않는다)."""
            try:
                return await llm.chat(
                    prompts.BOUNDS_SYSTEM, prompts.bounds_user([row]), thinking=True,
                    trace=st.get("trace"), name=f"refine_bounds[{row['scene_id']}]")
            except Exception as e:                   # noqa: BLE001 — 건별 격리
                log.warning("refine_bounds 실패(장면%d 경계 유지): %s", row["scene_id"], e)
                return ""

        texts = await asyncio.gather(*(one(r) for r in rows))
        moved: list[str] = []
        for row, text in zip(rows, texts):
            moved += bounds_mod.apply(clips, [row], text)
        if moved:
            log.info("refine_bounds: %d콜 동시 · 경계 이동 %s", len(rows), moved)
        if tr := st.get("trace"):
            tr.node("refine_bounds", asked=len(rows), moved=moved)
        return {"clips": clips, "endfix_moved": moved}

    async def score_match_node(st: ComposeState) -> dict:
        """클립별 채점 — 기각권은 없다. fill_budget 이 자를 때 이 점수를 쓴다.

        refine_bounds 와 같은 이유로 **클립당 1콜을 동시에** 보낸다. 채점 기준은 클립 자신과
        명세뿐이라 다른 클립을 볼 이유가 없었다.

        실패를 삼킨다: 이 콜 하나가 이미 확정된 클립을 통째로 날리던 문제(audit 5-1).
        이제 격리 단위가 클립이라 한 건이 실패해도 나머지 채점은 남는다 (점수 없는
        클립은 fill_budget 이 DEFAULT_SCORE 로 중립 처리).
        """
        clips = st["clips"]
        spec_line = plan.spec_line(st["spec"])

        async def one(c: dict) -> dict:
            try:
                text = await llm.chat(
                    prompts.VERIFY_SYSTEM,
                    prompts.verify_user(st["query"], spec_line, _packet(st["inv"], c)),
                    thinking=True, trace=st.get("trace"),
                    name=f"score_match[{c['scene_id']}]",
                )
                got = plan.parse_verify(text)
                sid = c["scene_id"]
                # 자기 장면 번호만 취한다 — 한 콜이 헛번호를 뱉어 남의 클립 점수를
                # 덮는 경로를 막는다. 번호를 틀렸어도 답이 한 줄이면 이 클립 것이다.
                if sid in got:
                    return {sid: got[sid]}
                return {sid: next(iter(got.values()))} if len(got) == 1 else {}
            except Exception as e:                   # noqa: BLE001 — 소견이 편성을 죽이면 안 된다
                log.warning("score_match 실패(장면%d 채점 없이 진행): %s", c["scene_id"], e)
                return {}

        scores: dict[int, dict] = {}
        for part in await asyncio.gather(*(one(c) for c in clips)):
            scores.update(part)
        log.info("score_match: %d콜 동시 · 채점 %d건", len(clips), len(scores))
        broken = [i for i, v in scores.items() if not v["complete"]]
        if broken:
            log.warning("score_match 완결성 문제 %s — 시작 보정 신호(클립은 유지)", broken)
        if tr := st.get("trace"):
            tr.node("score_match", scores=scores, incomplete=broken)
        return {"scores": scores}

    def drop_unmatched_node(st: ComposeState) -> dict:
        """score_match 0점 제외 — 순수 필터. 필수 장면은 예외(사실이 소견을 이긴다).

        order_clips 에 넘어가는 후보를 줄여 프롬프트를 짧게 하고, "무관한 클립으로
        예산을 채우는" 경로를 fill_budget 이전에 끊는다.
        """
        scores = st.get("scores", {})
        keep, cut_ = [], []
        for c in st["clips"]:
            zero = scores.get(c["scene_id"], {}).get("score", select_mod.DEFAULT_SCORE) <= \
                select_mod.DROP_SCORE
            (cut_ if zero and not select_mod.is_must(c) else keep).append(c)
        if cut_:
            log.info("0점 제외 %s", [c["scene_id"] for c in cut_])
        if tr := st.get("trace"):
            tr.node("drop_unmatched", kept=len(keep), zero=[c["scene_id"] for c in cut_])
        return {"clips": keep, "zero_dropped": [(c["scene_id"], "질의와 무관(score_match 0점)")
                                                for c in cut_]}

    async def order_clips_node(st: ComposeState) -> dict:
        """남은 후보를 질의에 맞는 **순서**로 줄 세운다 (1콜 — 비교라 나눌 수 없다).

        담을지 말지는 정하지 않는다. 순서만 받고 예산 합산·절단은 fill_budget 이 한다 —
        예산 절단은 산술이고 LLM 은 산술을 못 한다(실측: 900초 요청에 947~1018초).
        """
        clips, spec = st["clips"], st["spec"]
        scores = st.get("scores", {})
        must = [c for c in clips if select_mod.is_must(c)]
        rest = [c for c in clips if not select_mod.is_must(c)]
        if not rest:
            return {"order": []}
        used = sum(r["cut"]["ce"] - r["cut"]["cs"] for r in must)
        left = max(0, int(spec["budget"] - used))
        # 필수층이 예산을 다 쓰면 순위를 매겨도 담길 자리가 없다 — 1~2분짜리 콜을
        # 낭비하지 않는다 (v201 실측: 필수 16건 1103s > 예산 900s → 남은 0s).
        if left < min(c["cut"]["ce"] - c["cut"]["cs"] for c in rest):
            log.info("order_clips 생략: 남은 예산 %ds 로는 최단 후보도 못 담는다", left)
            return {"order": []}
        try:
            text = await llm.chat(
                prompts.RANK_SYSTEM,
                prompts.rank_user(
                    st["query"], st["inv"].game_line, left,
                    sorted(must, key=lambda c: c["cut"]["cs"]),
                    [(c, scores.get(c["scene_id"], {}).get("score",
                                                           select_mod.DEFAULT_SCORE))
                     for c in sorted(rest, key=lambda c: c["cut"]["cs"])]),
                thinking=True, trace=st.get("trace"), name="order_clips")
            order = select_mod.parse_order(text, {c["scene_id"] for c in rest})
        except Exception as e:                       # noqa: BLE001 — 폴백이 있다
            log.warning("order_clips 실패(점수순 폴백): %s", e)
            order = []
        log.info("order_clips: 후보 %d건 순서 %s", len(rest), order or "(폴백)")
        if tr := st.get("trace"):
            tr.node("order_clips", asked=len(rest), order=order, budget_left=left)
        return {"order": order}

    def fill_budget_node(st: ComposeState) -> dict:
        """예산 확정 — 층 순서로 채우고, 마지막이라 예산이 정확하다."""
        spec = st["spec"]
        clips, scores = st["clips"], st.get("scores", {})
        if not use_fill:
            picked, spare = _assemble(clips, spec["budget"])
            total = sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked)
            return {"picked": picked, "spare": spare, "total": int(total), "status": "ok"}
        picked, dropped, total = select_mod.choose(clips, spec, scores, st.get("order"))
        if not picked:
            picked = select_mod.rescue_longest(clips, spec["budget"])
            total = int(sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked))
        if tr := st.get("trace"):
            tr.node("fill_budget", picked=[c["scene_id"] for c in picked], total=total,
                    dropped=[(c["scene_id"], why) for c, why in dropped])
        return {
            "picked": picked, "total": total, "status": "ok" if picked else "empty",
            "dropped": (st.get("zero_dropped") or [])
                       + [(c["scene_id"], why) for c, why in dropped],
            "suspicions": [(i, v["reason"]) for i, v in scores.items() if v["score"] <= 1],
        }

    def end_empty_node(st: ComposeState) -> dict:
        return {"picked": [], "status": "empty"}

    g = StateGraph(ComposeState)
    g.add_node("rephrase_query", rephrase_query_node)
    g.add_node("retrieve_evidence", retrieve_evidence_node)
    g.add_node("select_clips", select_clips_node)
    g.add_node("set_bounds", set_bounds_node)
    g.add_node("retry_select", retry_select_node)
    g.add_node("refine_bounds", refine_bounds_node)
    g.add_node("score_match", score_match_node)
    g.add_node("drop_unmatched", drop_unmatched_node)
    g.add_node("order_clips", order_clips_node)
    g.add_node("fill_budget", fill_budget_node)
    g.add_node("end_empty", end_empty_node)
    g.add_edge(START, "rephrase_query")
    g.add_edge("rephrase_query", "retrieve_evidence")
    g.add_edge("retrieve_evidence", "select_clips")
    g.add_edge("select_clips", "set_bounds")
    g.add_conditional_edges(
        "set_bounds", route, {"retry_select": "retry_select",
                              "refine_bounds": "refine_bounds", "end_empty": "end_empty"})
    g.add_edge("retry_select", "select_clips")
    g.add_edge("refine_bounds", "score_match")
    g.add_edge("score_match", "drop_unmatched")
    g.add_edge("drop_unmatched", "order_clips")
    g.add_edge("order_clips", "fill_budget")
    g.add_edge("fill_budget", END)
    g.add_edge("end_empty", END)
    return g.compile()


def _label_filter(term: str) -> str:
    """메타 필터 한 항목 → Milvus 표현식. 태그·라벨 어느 쪽에 있어도 걸리게."""
    safe = term.replace('"', "")
    return f'(tags like "%{safe}%" or labels like "%{safe}%")'


def _dedup_hits(hits: list[dict]) -> list[dict]:
    """여러 검색어·필터 결과 병합 — 같은 (kind, s, text) 는 유사도 높은 쪽만 남긴다."""
    best: dict[tuple, dict] = {}
    for h in hits:
        key = (h.get("kind"), round(float(h.get("s", 0))), h.get("text", "")[:40])
        if key not in best or h["distance"] > best[key]["distance"]:
            best[key] = h
    return sorted(best.values(), key=lambda h: -h["distance"])


async def run_compose(
    graph, inv: Inventory, query: str, budget: int | None, on_node=None,
    trace=None) -> ComposeState:
    """그래프 1회 실행 — stream 으로 노드 순서를 로그에 남기고 델타를 누적한다.

    on_node(node_name): 노드 완료 콜백 — API 가 job 진행 표시에 쓴다 (폴링 응답의
    progress 필드). checkpointer 는 도입하지 않는다 — 1~3분 잡은 재실행이 더 싸고,
    진행 노출은 이 스트림으로 충분 (Phase 4 검토 2026-08-18).
    """
    state: ComposeState = {}
    async for step in graph.astream(
            {"query": query, "budget": budget, "inv": inv, "trace": trace},
            stream_mode="updates"):
        for node, upd in step.items():
            log.info(
                "── 노드 %s 완료 (갱신: %s)", 
                node, ", ".join(k for k in (upd or ()) if k != "inv") or "-"
            )
            state.update(upd or {})
            
            if on_node:
                r = on_node(node)
                if inspect.isawaitable(r):      # async 콜백 지원 (상태 코드 DB 기록 등)
                    await r
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
        g = groups.setdefault(sid, {"scene_id": sid, "hits": 0, "sim": 0.0,
                                    "snippets": [], "by_kind": {}})
        g["hits"] += 1
        g["sim"] = max(g["sim"], h["distance"])
        # 종류별로 최소 1건씩 확보한다 — 유사도 순으로만 자르면 한 종류가 자리를
        # 다 먹는다 (v201 comp9 실측: 후보 8장면 전부 화면 캡션, 해설이 전멸).
        # 캡션은 서로 구별이 안 되는데("야수들이 타구를 향해 움직이는 장면"류가 다섯
        # 장면에 동일) 해설은 "펜스를 직격하는 적시타"처럼 장면을 특정한다.
        g["by_kind"].setdefault(h.get("kind") or "?", []).append(h["text"])
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


# 패킷에 실을 샷 — 앞뒤를 나눠 잡는다. 앞에서 8개만 자르면 **마지막 샷이 사라지는데**
# VERIFY_SYSTEM 이 "마지막 샷에 결과가 없으면 문제" 로 판정하라고 지시한다. 8샷 초과
# 클립이 경기당 8~15건(최대 58샷)이라 전부 오판정 후보였다 (2026-08-20 사전 점검).
SHOT_HEAD = 5         # 시작 판정용 — 첫 샷이 그 플레이인지 앞 플레이 잔상인지
SHOT_TAIL = 3         # 끝 판정용 — 결과 화면이 있는지
UTT_CLIP = 90         # 해설 1건 표기 길이


def _packet(inv: Inventory, c: dict) -> str:
    """클립 1건 → 검수 패킷. **시각 한 축**으로 전광판·화면·해설을 정렬한다.

    예전에는 화면 유형을 화살표로 나열하고(`투구 → 타구·수비 → …`) 대사를 따로 이어
    붙였다. 어느 해설이 어느 화면의 것인지 알 수 없어 "무슨 일이 일어났나"가 안 읽혔고,
    유형명만으로는 첫 샷이 그 플레이인지 앞 플레이 잔상인지 갈리지 않았다
    (v201 장면9: 첫 샷이 앞 타구의 "야수가 공을 쫓아 걷고 있다"인데 완결성 '정상').

    구간은 **컷 좌표**로 잡는다 — 장면 전체로 넓히면 이웃 플레이가 섞인다.
    """
    cs, ce = c["cut"]["cs"], c["cut"]["ce"]
    away = (c.get("score") or " ").split()[0] if c.get("score") else ""
    head = (f"[클립] {c['scene_id']} · {c['scene_type']}"
            f"({c['labels'] or '라벨없음'}) · {c['inning']} · {cs:.0f}~{ce:.0f}s "
            f"({ce - cs:.0f}s)\n"
            f"       {away} {c['score_before']} → {away} {c['score'].split()[1]}"
            if c.get("score") else f"[클립] {c['scene_id']}")

    lines = [head]
    tr = [r for r in inv.trans if cs - 20 <= r["sec"] <= ce]
    if tr:
        lines.append("")
        lines.append("  전광판")
        for r in tr:
            lines.append(
                f"    {r['sec']}s  {r['outs']}사 {r['balls']}볼{r['strikes']}스 "
                f"주자 {r['bases']}  {r['away_score']}-{r['home_score']}"
                + (f"  ({r['hint']})" if r.get("hint") else ""))

    shots = [s for s in inv.segs if s["s"] < ce and s["e"] > cs]
    if shots:
        lines.append("")
        head, tail = shots[:SHOT_HEAD], shots[-SHOT_TAIL:]
        skipped = len(shots) - len(head) - len(tail)
        shown = head + tail if skipped > 0 else shots
        for i, s in enumerate(shown):
            if skipped > 0 and i == len(head):
                lines.append(f"     … (중략 {skipped}샷)")
            lines.append(f"  {s['s']:.0f}s [{s.get('shot_type') or '미분류'}]")
            if s.get("summary"):
                lines.append(f"     화면 {s['summary']}")
            utt = next((x for us, ue, x in inv.utts if us < s["e"] and ue > s["s"]), "")
            if utt:
                lines.append(f'     해설 "{utt[:UTT_CLIP]}"')
    return "\n".join(lines)
