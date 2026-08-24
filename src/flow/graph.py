"""compose 그래프 — 질의 1건 → 편성.

  rephrase_query ─► retrieve_evidence ─► select_clips
     ─► refine_end_bound ─► refine_start_bound ─► finish ─► END
                          ▲
     select_clips ◄── retry_select ◄──┤ (0건·재시도 여유)
                       end_empty ◄────┘ (0건·소진) ──► END

노드 이름은 전부 **동사_목적어**이고, 그 노드가 무엇을 만들어 내는지를 가리킨다.

2026-08-20 재설계 — 경계만 남기고 걷어냈다:
- **선곡이 곧 편성이다.** 채점(score_match)·0점 제외(drop_unmatched)·예산 절단
  (fill_budget)·필수 회수를 폐기했다. 채점은 후보 90% 이상에 일치도 3 을 주어 변별이
  없었고, 예산 절단과 회수는 질의를 규칙이 덮어쓰는 통로였다.
- **끝과 시작을 다른 노드로 나눈다.** 끝은 모든 클립이 묻고(코드가 답할 수 없다),
  시작은 투구 앵커를 못 찾은 클립만 묻는다(앵커가 있으면 코드의 답이 정답이다).
  둘 다 클립당 1콜을 동시에 보낸다.
- 컷 계산(앵커·레시피)은 노드가 아니라 refine_end_bound 안의 순수 계산으로 남는다.
- **rephrase_query** 가 질의를 중계의 언어로 옮긴다 (실측: 추상 질의 최고 유사도
  0.58 vs 구체 질의 0.66~0.78).

완화 사다리:
  L1 투구 앵커 없는 클립 → 통째 폴백 (cut 내장) + refine_start_bound 가 시작을 되찾는다
  L2 선곡 검산 후 0건 → retry_select → select_clips 재선곡 (MAX_REPLAN 회)
  L3 그래도 0건 → status=empty

retrieve_evidence 실패는 전파 — bench4 의 fail-open 폐기 (design.md §2).
그래프는 무상태 배선이라 프로세스당 1회 컴파일.
"""

import asyncio
import inspect

from langgraph.graph import END, START, StateGraph

from flow import bounds as bounds_mod
from flow import cut, plan, prompts, rank, vocab
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
        """선곡 — 모델은 **번호만** 답한다. 그 선곡이 곧 편성이다.

        예산·채점·절단을 전부 걷어냈으므로(2026-08-20) 여기서 고른 장면이 그대로
        결과가 된다. 대상 어휘(targets)는 rephrase_query 가 질의에서 뽑아 둔 필터를
        옮겨 담을 뿐이고, 모델에게 남긴 판단은 "질의에 맞는 장면이 어느 것인가" 하나다.
        """
        inv = st["inv"]
        # 인벤토리는 요청당 1회 렌더가 원칙이나, 증거를 장면 블록에 병합하려면
        # retrieve_evidence 결과가 필요해 여기서 다시 만든다 (순수 문자열 조립).
        inventory = plan.render_inventory(list(inv.scenes), st.get("evidence", []))
        # 이 노드는 thinking 을 **켠다** — 85장면 인벤토리에서 "질의에 맞는 것"을 고르는
        # 일이라 경계 고르기(refine_*)와 성격이 다르다. 대신 시간 가드를 240→480 으로
        # 올렸다(llm.THINK_TIMEOUT_SEC): 240 에서는 comp39·40 이 매번 타임아웃해 편성
        # 소요의 96%가 버려진 대기였고, 그때 나온 선곡은 사실 thinking 없는 폴백이었다.
        text = await llm.chat(
            prompts.PLAN_SYSTEM,
            prompts.plan_user(st["query"], inv.game_line, inventory, st.get("feedback", "")),
            thinking=True, trace=st.get("trace"), name="select_clips",
        )
        log.info("select_clips 응답: %r", text)
        spec = {
            "mode": "compose",
            "budget": st.get("budget") or 0,        # t_compose.budget_sec 에 남는다
            # 대상 어휘는 rephrase_query 의 메타 필터 — 필수 장면 회수 범위를 가른다.
            "targets": list(st.get("filters") or []),
            "view": "전체",
            "picked": plan.parse_picked(text, list(inv.scenes)),
            "raw": text,
        }
        log.info("select_clips: %s 선곡 %s", plan.spec_line(spec), spec["picked"])
        if tr := st.get("trace"):
            tr.node("select_clips", spec={k: v for k, v in spec.items() if k != "raw"})
        return {"spec": spec, "attempt": st.get("attempt", 0) + 1}

    def _cut_clips(st: ComposeState) -> list[dict]:
        """선곡 → 컷 좌표가 붙은 클립 (순수 계산 — 노드로 두지 않는다).

        시작은 투구 앵커, 끝은 태그 레시피가 1차로 정한다. 이 계산이 없으면 모든
        클립이 장면 통째가 되는데(v1003 장면10 은 그렇게 116초), 두 보정 노드는 끝을
        늘리기만 하므로 되돌릴 방법이 없다.
        """
        inv = st["inv"]
        clips = rank.order(list(inv.scenes), list(st["spec"]["picked"]))
        for r in clips:
            r["cut"] = cut.clip(r, list(inv.segs), inv.utts)
        return clips

    async def refine_end_bound_node(st: ComposeState) -> dict:
        """끝 확정 — **모든 클립**이 대상. 클립 1건 = 콜 1건으로 펼쳐 동시에 보낸다.

        "결과를 설명하는 해설이 어디서 끝나는가"는 코드가 답할 수 없다. 레시피 체인은
        샷 오분류 하나로 결과 전에 끊기고(v203 도루: '주루' 샷이 없어 실물이 잘림),
        규칙으로 남은 건 발화 꼬리 스냅(9초 상한)뿐이었다.

        한 콜에 몰면 전송이 직렬이라 GPU 가 논다 (실측 v201: 24클립 10분 26초 동안
        서버는 내내 Running 1 · KV 3%). 판정은 클립끼리 독립이고, 응답을 **자기 행에만**
        대조하므로 한 콜의 헛번호가 남의 클립을 못 건드린다.
        """
        inv = st["inv"]
        clips = [dict(r, cut=dict(r["cut"])) for r in _cut_clips(st)]   # 복사 후 수정
        if not use_refine or not clips:
            return _skipped(st, "refine_end_bound", clips,
                            "보정 꺼짐" if not use_refine else "클립 없음")
        rows = bounds_mod.end_rows(clips, list(inv.segs), inv.utts, list(inv.scenes))
        log.info("refine_end_bound 대상 %d/%d클립", len(rows), len(clips))
        if not rows:
            return _skipped(st, "refine_end_bound", clips, "후보 없음")

        async def one(row: dict) -> str:
            """행 1건 질의 — 실패는 그 클립만 원 경계 유지 (배치 전체를 죽이지 않는다)."""
            try:
                # thinking 끔 (2026-08-24 결정). 끝 고르기는 후보를 나란히 놓고 해설이
                # 어디서 끝나는지 보는 일이라 추론 사슬이 필요 없고, 켜 두면 후보끼리
                # 우열 근거가 약할 때 판별자를 찾다가 폭주한다 — comp38 장면5 는
                # 41,317자를 태우고 본문 없이 떨어져 재시도로 겨우 건졌다.
                return await llm.chat(
                    prompts.END_SYSTEM, prompts.end_user([row]), thinking=False,
                    trace=st.get("trace"), name=f"refine_end_bound[{row['scene_id']}]")
            except Exception as e:                   # noqa: BLE001 — 건별 격리
                log.warning("refine_end_bound 실패(장면%d 경계 유지): %s", row["scene_id"], e)
                return ""

        texts = await asyncio.gather(*(one(r) for r in rows))
        moved: list[str] = []
        for row, text in zip(rows, texts):
            moved += bounds_mod.apply_end(clips, [row], text)
        log.info("refine_end_bound: %d콜 동시 · 끝 이동 %s", len(rows), moved or "없음")
        if tr := st.get("trace"):
            tr.node("refine_end_bound", asked=len(rows), moved=moved)
        return {"clips": clips, "end_moved": moved}

    async def refine_start_bound_node(st: ComposeState) -> dict:
        """시작 확정 — **투구 앵커를 못 찾은 클립만**. 역시 클립당 1콜 동시.

        앵커가 잡히면 cut 이 정한 시작이 이미 그 플레이의 투구 샷이라 물을 이유가 없다.
        앵커가 없는 클립은 장면 시작을 그대로 쓰는데, 전이 원장의 시각이 투구가 아니라
        관측 시점에 찍히면 그 시작이 이미 플레이 도중이다 — v203 장면5(홈런)는
        "담장 넘어갑니다"부터 시작해 스윙이 없었다.
        """
        inv = st["inv"]
        clips = [dict(r, cut=dict(r["cut"])) for r in st["clips"]]
        if not use_refine or not clips:
            return _skipped(st, "refine_start_bound", clips,
                            "보정 꺼짐" if not use_refine else "클립 없음")
        rows = bounds_mod.start_rows(clips, list(inv.segs), inv.utts)
        log.info("refine_start_bound 대상 %d/%d클립 — 나머지는 앵커 샷이 %s 라 시작 그대로",
                 len(rows), len(clips), "·".join(sorted(bounds_mod.TRUSTED_ANCHOR_SHOTS)))
        if not rows:
            return _skipped(st, "refine_start_bound", clips, "믿을 수 있는 앵커뿐")

        async def one(row: dict) -> str:
            try:
                # thinking 끔 (2026-08-24) — 끝과 같은 이유다. 켜 두는 근거였던 실측
                # ("끄자 13콜 중 11건이 시작 유지→이동으로 뒤집혔다")은 **구 게이트
                # 시절 수치**라 지금은 성립하지 않는다: 그때는 앵커 없는 클립만 물어
                # 대상이 13건이었고, 지금은 앵커 샷 유형으로 걸러 대상이 한 자리다.
                # 다시 켤 근거가 생기면 그때 켜고/끈 결과를 나란히 놓고 정한다.
                return await llm.chat(
                    prompts.START_SYSTEM, prompts.start_user([row]), thinking=False,
                    trace=st.get("trace"), name=f"refine_start_bound[{row['scene_id']}]")
            except Exception as e:                   # noqa: BLE001 — 건별 격리
                log.warning("refine_start_bound 실패(장면%d 경계 유지): %s", row["scene_id"], e)
                return ""

        texts = await asyncio.gather(*(one(r) for r in rows))
        moved: list[str] = []
        for row, text in zip(rows, texts):
            moved += bounds_mod.apply_start(clips, [row], text)
        log.info("refine_start_bound: %d콜 동시 · 시작 이동 %s", len(rows), moved or "없음")
        if tr := st.get("trace"):
            tr.node("refine_start_bound", asked=len(rows), moved=moved)
        return {"clips": clips, "start_moved": moved}

    def finish_node(st: ComposeState) -> dict:
        """마감 — 예산 절단 후 시간순 확정. 채점은 없다(선곡이 곧 편성이다).

        예산이 있으면 **rank.score 내림차순으로 담다가 넘치면 버린다** (2026-08-24).
        구 fill_budget 과 다른 점이 결정적이다: 그건 예산을 채우려고 선곡에 없던
        장면을 끌어와 "질의를 규칙이 덮어쓰는 통로"였다(94b58dc 폐기 사유). 여기는
        **덜어내기만** 한다 — 모자라면 모자란 대로 둔다. 선곡이 관련 있는 것만
        골랐다는 전제가 유지되고, 코드는 길이만 책임진다.

        순위는 rank.score(득점·라벨·태그·판세·이닝)다. 질의 의도는 안 들어가지만,
        여기 오는 건 이미 select_clips 가 질의로 걸러 낸 것들이라 남은 판단은
        "그중 무엇이 더 큰 플레이인가"뿐이다. LLM 콜 0개로 재현 가능하다.

        절단 후 **시간순으로 되돌린다** — 편성은 경기 흐름대로 재생돼야 한다.
        """
        budget = st.get("budget")
        picked, dropped = rank.fit_budget(list(st.get("clips") or []), budget)
        total = sum(c["cut"]["ce"] - c["cut"]["cs"] for c in picked)
        if dropped:
            log.info("finish: 예산 %ds 절단 — %d건 버림 %s", budget, len(dropped), dropped)
        log.info("finish: 클립 %d건 %.0fs", len(picked), total)
        if tr := st.get("trace"):
            tr.node("finish", picked=[c["scene_id"] for c in picked],
                    total=round(total, 1), budget=budget, dropped=dropped)
        return {"picked": picked, "total": round(total, 1), "dropped": dropped,
                "status": "ok" if picked else "empty"}

    def route(st: ComposeState) -> str:
        if st["spec"]["picked"]:
            return "refine_end_bound"
        return "retry_select" if st["attempt"] <= vocab.MAX_REPLAN else "end_empty"

    def retry_select_node(st: ComposeState) -> dict:
        return {"feedback": (f"선곡 {st['spec']['picked']}이 검산에서 비었다. "
                             f"질의를 어휘로 다시 번역해 골라라.")}

    def end_empty_node(st: ComposeState) -> dict:
        return {"picked": [], "status": "empty"}

    g = StateGraph(ComposeState)
    g.add_node("rephrase_query", rephrase_query_node)
    g.add_node("retrieve_evidence", retrieve_evidence_node)
    g.add_node("select_clips", select_clips_node)
    g.add_node("retry_select", retry_select_node)
    g.add_node("refine_end_bound", refine_end_bound_node)
    g.add_node("refine_start_bound", refine_start_bound_node)
    g.add_node("finish", finish_node)
    g.add_node("end_empty", end_empty_node)
    g.add_edge(START, "rephrase_query")
    g.add_edge("rephrase_query", "retrieve_evidence")
    g.add_edge("retrieve_evidence", "select_clips")
    g.add_conditional_edges(
        "select_clips", route, {"retry_select": "retry_select",
                                "refine_end_bound": "refine_end_bound",
                                "end_empty": "end_empty"})
    g.add_edge("retry_select", "select_clips")
    g.add_edge("refine_end_bound", "refine_start_bound")
    g.add_edge("refine_start_bound", "finish")
    g.add_edge("finish", END)
    g.add_edge("end_empty", END)
    return g.compile()


def _skipped(st: ComposeState, node: str, clips: list[dict], why: str) -> dict:
    """할 일이 없어 건너뛴 노드도 트레이스에 남긴다 (2026-08-24).

    구 배선은 조기 반환이 `tr.node()` 앞에 있어 이 노드가 트레이스에서 통째로
    사라졌다. 그래서 **"돌았는데 할 게 없었다"와 "아예 안 돌았다"가 구분되지 않았다** —
    refine_start_bound 가 게이트 조건 때문에 대상 0건으로 놀고 있던 걸 comp37~39
    어디에서도 알 수 없었던 이유다. 침묵이 성공처럼 보이는 자리를 없앤다.
    """
    log.info("%s 건너뜀: %s", node, why)
    if tr := st.get("trace"):
        tr.node(node, asked=0, skipped=why)
    out: dict = {"clips": clips}
    if node == "refine_start_bound":
        out["status"] = "ok"
    return out


def _label_filter(term: str) -> str:
    """메타 필터 한 항목 → Milvus 표현식. 네 축 어디에 있어도 걸리게.

    축이 넷으로 갈렸다 (2026-08-23 상류 개편): 행위 태그(tags) · 파생 라벨(labels) ·
    판세(game_context) · 전광판 사실(board_tags). 앞 둘만 보면 '역전'·'동점' 같은
    질의가 조용히 무필터가 된다 — 그 말들이 labels 에서 game_context 로 이사했다.
    """
    safe = term.replace('"', "")
    return (f'(tags like "%{safe}%" or labels like "%{safe}%" '
            f'or game_context like "%{safe}%" or board_tags like "%{safe}%")')


def _dedup_hits(hits: list[dict]) -> list[dict]:
    """여러 검색어·필터 결과 병합 — 같은 (kind, s, text) 는 유사도 높은 쪽만 남긴다."""
    best: dict[tuple, dict] = {}
    for h in hits:
        key = (h.get("kind"), round(float(h.get("s", 0))), h.get("text", "")[:40])
        if key not in best or h["distance"] > best[key]["distance"]:
            best[key] = h
    return sorted(best.values(), key=lambda h: -h["distance"])


async def run_compose(graph, inv: Inventory, query: str, budget: int | None = None,
                      on_node=None, trace=None) -> ComposeState:
    """그래프 1회 실행 — stream 으로 노드 순서를 로그에 남기고 델타를 누적한다.

    on_node(node_name): 노드 완료 콜백 — API 가 job 진행 표시에 쓴다 (폴링 응답의
    progress 필드). checkpointer 는 도입하지 않는다 — 1~3분 잡은 재실행이 더 싸고,
    진행 노출은 이 스트림으로 충분 (Phase 4 검토 2026-08-18).
    """
    state: ComposeState = {}
    async for step in graph.astream(
            {"query": query, "inv": inv, "trace": trace, "budget": budget},
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
