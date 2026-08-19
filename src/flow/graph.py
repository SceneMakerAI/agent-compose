"""compose 그래프 — 질의 1건 → 편성.

  expand ─► retrieve ─► plan ─► cut ─► bounds ─► verify ─► select ─► END
                         ▲               (경계)   (채점)   (예산확정)
                         └── feedback ◄── (0건·재시도 여유)
                                       └─► empty ──► END (0건·소진)

2026-08-20 재배선 — cutrank 를 해체했다:
- **절단이 마지막**(select). 예전에는 경계 확정 전에 잘랐고 그 뒤 endfix 가 끝을 늘려
  예산 보장이 무효가 됐다 (실측: 900초 요청에 947~1018초).
- **bounds** 가 시작·끝을 함께 정한다. 따로 물으면 서로를 모른다 — 시작을 25초 당기면
  끝의 여유도 달라진다. 홈런이 FULL_CLIP_TAGS 라 앵커 로직을 건너뛰어 '주자가 뛰는
  장면'부터 시작하던 문제(v202 장면 11)도 여기서 잡는다.
- **verify 가 값을 한다**. 예전에는 소견 전용이라 출력물을 바꾸는 경로가 아예 없었다.
  이제 클립별 일치도(0~3)를 매기고 select 가 자를 때 그 점수를 쓴다. 기각권은 여전히
  없다 — 필수층(득점·역전·동점·끝내기·경기 종료)은 0점이어도 유지한다.
- **expand** 가 질의를 중계의 언어로 옮긴다 (실측: 추상 질의 최고 유사도 0.58 vs
  구체 질의 0.66~0.78).

완화 사다리:
  L1   투구 앵커 없는 클립 → 통째 폴백 (cut 내장)
  L2   선곡 검산 후 0건 → feedback → plan 재선곡 (MAX_REPLAN 회)
  L2.5 전 클립이 예산 초과 → 최단 1건 구제 (select.rescue_longest)
  L3   그래도 0건 → status=empty

retrieve 실패는 전파 — bench4 의 fail-open 폐기 (design.md §2).
그래프는 무상태 배선이라 프로세스당 1회 컴파일.
"""

import asyncio
import inspect
import math
import re

from langgraph.graph import END, START, StateGraph

from flow import bounds as bounds_mod
from flow import cut, plan, prompts, rank, vocab
from flow import select as select_mod
from flow.llm import MAX_TOKENS_PICK, ChatLLM
from flow.state import ComposeState, Inventory
from log import get_logger
from vector.embedder import Embedder
from vector.store import QUERY_INSTRUCT, VectorStore

log = get_logger(__name__)

EVIDENCE_SCENES_MAX = 8      # plan 에 주입할 후보 장면 상한 (bench4 vector.py 운영값)
EVIDENCE_SNIPPETS_MAX = 2    # 장면당 증거 스니펫 상한
ORPHAN_MAX = 3               # 장면밖 증거 표기 상한 (발행 누락 의심 신호용)


def build_graph(llm: ChatLLM, embedder: Embedder, store: VectorStore, settings=None):
    """자원 주입 배선 — lifespan 에서 1회 호출해 컴파일 결과를 공유한다.

    settings 의 use_* 플래그로 새 단계를 끌 수 있다 — 여러 변경을 한꺼번에 넣으면
    "무엇이 달라졌나"는 트레이스로 봐도 "어느 변경 때문인가"는 갈리지 않는다.
    """
    use_expand = getattr(settings, "use_expand", True)
    use_bounds = getattr(settings, "use_bounds", True)
    use_select = getattr(settings, "use_select", True)

    async def expand_node(st: ComposeState) -> dict:
        """질의 → 중계 문장형 검색어 + 메타 필터 힌트. 실패하면 원 질의로 폴백."""
        if not use_expand:
            return {"phrases": [st["query"]], "filters": []}
        try:
            text = await llm.chat(prompts.EXPAND_SYSTEM, prompts.expand_user(st["query"]),
                                  thinking=False, trace=st.get("trace"), name="expand")
            phrases, filters = plan.parse_expand(text)
        except Exception as e:                       # noqa: BLE001 — 보조 단계, 죽이지 않는다
            log.warning("expand 실패(원 질의로 진행): %s", e)
            phrases, filters = [], []
        phrases = phrases or [st["query"]]
        log.info("expand: 검색어 %s / 필터 %s", phrases, filters or "-")
        if tr := st.get("trace"):
            tr.node("expand", phrases=phrases, filters=filters)
        return {"phrases": phrases, "filters": filters}

    async def retrieve_node(st: ComposeState) -> dict:
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
            log.info("retrieve: 후보 장면 %s, 장면밖 %d건",
                     [g["scene_id"] for g in ev], len(orphan))
        if tr := st.get("trace"):
            tr.node("retrieve", candidates=[g["scene_id"] for g in ev],
                    orphan=len(orphan), hits=len(hits))
        return {"evidence": ev, "evidence_orphan": orphan}

    async def plan_node(st: ComposeState) -> dict:
        inv = st["inv"]
        # 호출자 예산을 LLM 에게도 알린다 — 기본값만 보내던 탓에 900s 요청에도 모델이
        # "예산: 180" 으로 답하며 그 전제로 선곡했다 (2026-08-19 실측 로그).
        budget = st.get("budget") or plan.DEFAULT_BUDGET_SEC
        text = await llm.chat(
            prompts.PLAN_SYSTEM.replace("{budget}", str(budget)),
            prompts.plan_user(
                st["query"], inv.game_line, inv.inventory_text, budget,
                st.get("feedback", ""),
                plan.render_evidence(st.get("evidence", []), []),   # orphan 은 프롬프트에서 뺀다
            ),
            thinking=True, trace=st.get("trace"), name="plan",
        )
        log.info("plan 응답: %r", text)
        spec = plan.parse(text, list(inv.scenes))
        if st.get("budget"):
            spec["budget"] = st["budget"]      # 명시 입력이 질의 해석보다 우선
        spec["picked"] = list(dict.fromkeys(spec["picked"]))   # 중복 선곡 제거
        log.info("plan: %s 선곡 %s", plan.spec_line(spec), spec["picked"])
        if tr := st.get("trace"):
            tr.node("plan", spec={k: v for k, v in spec.items() if k != "raw"})
        return {"spec": spec, "attempt": st.get("attempt", 0) + 1}

    async def cut_node(st: ComposeState) -> dict:
        """경계 계산만 — 순위도 절단도 하지 않는다 (구 cutrank 의 1/3).

        선곡분에 **필수 장면(득점·역전·동점·끝내기·경기 종료)을 합쳐서** 컷한다.
        select 의 필수층이 일하려면 그 장면이 후보에 있어야 하는데, plan 이 놓치면
        영영 들어올 수 없기 때문이다 (구 backfill 이 하던 회수를 여기서 보장한다).
        컷 계산은 순수 함수라 몇 건 늘어도 비용이 없고, LLM 을 타는 bounds·verify 의
        입력만 그만큼 커진다.
        """
        inv = st["inv"]
        picked = list(st["spec"]["picked"])
        must = [r["scene_id"] for r in inv.scenes
                if r["scene_id"] not in picked and select_mod.is_must(r)]
        if must:
            log.info("cut: 필수 장면 회수 %s (plan 미선곡)", must)
        clips = rank.order(list(inv.scenes), picked + must)
        for r in clips:
            r["cut"] = cut.clip(r, list(inv.segs), inv.utts)
        if tr := st.get("trace"):
            tr.node("cut", clips=[(r["scene_id"], round(r["cut"]["cs"]), round(r["cut"]["ce"]))
                                  for r in clips])
        return {"clips": clips}

    def route(st: ComposeState) -> str:
        if st.get("clips"):
            return "bounds"
        return "feedback" if st["attempt"] <= vocab.MAX_REPLAN else "empty"

    def feedback_node(st: ComposeState) -> dict:
        return {"feedback": (f"선곡 {st['spec']['picked']}이 검산에서 비었다. "
                             f"질의를 어휘로 다시 번역해 골라라.")}

    async def bounds_node(st: ComposeState) -> dict:
        """시작·끝을 함께 정한다. 후보는 결정적으로 만들고 LLM 은 고르기만 한다.

        **클립 1건 = 콜 1건으로 펼쳐 동시에 보낸다.** 한 콜에 몰면 전송이 직렬이라
        GPU 가 놀았다 (실측 v201: 24클립 10분 26초 동안 서버는 내내 Running 1 ·
        KV 3%). 경계 판정은 클립끼리 독립이라 나눠도 근거가 줄지 않고, 응답을
        **자기 행에만** 대조하므로 한 콜의 헛번호가 남의 클립을 못 건드린다.
        """
        inv = st["inv"]
        clips = [dict(r, cut=dict(r["cut"])) for r in st["clips"]]     # 복사 후 수정
        if not use_bounds:
            return {"clips": clips}
        rows = bounds_mod.build_rows(clips, list(inv.segs), inv.utts, inv.pitches)
        if not rows:
            return {"clips": clips}

        async def one(row: dict) -> str:
            """행 1건 질의 — 실패는 그 클립만 원 경계 유지 (배치 전체를 죽이지 않는다)."""
            try:
                return await llm.chat(
                    prompts.BOUNDS_SYSTEM, prompts.bounds_user([row]), thinking=True,
                    trace=st.get("trace"), name=f"bounds[{row['scene_id']}]",
                    think_max=MAX_TOKENS_PICK)
            except Exception as e:                   # noqa: BLE001 — 건별 격리
                log.warning("bounds 실패(장면%d 경계 유지): %s", row["scene_id"], e)
                return ""

        texts = await asyncio.gather(*(one(r) for r in rows))
        moved: list[str] = []
        for row, text in zip(rows, texts):
            moved += bounds_mod.apply(clips, [row], text)
        if moved:
            log.info("bounds: %d콜 동시 · 경계 이동 %s", len(rows), moved)
        if tr := st.get("trace"):
            tr.node("bounds", asked=len(rows), moved=moved)
        return {"clips": clips, "endfix_moved": moved}

    async def verify_node(st: ComposeState) -> dict:
        """클립별 채점 — 기각권은 없다. select 가 자를 때 이 점수를 쓴다.

        bounds 와 같은 이유로 **클립당 1콜을 동시에** 보낸다. 채점 기준은 클립 자신과
        명세뿐이라 다른 클립을 볼 이유가 없었다.

        실패를 삼킨다: 이 콜 하나가 이미 확정된 클립을 통째로 날리던 문제(audit 5-1).
        이제 격리 단위가 클립이라 한 건이 실패해도 나머지 채점은 남는다 (점수 없는
        클립은 select 가 DEFAULT_SCORE 로 중립 처리).
        """
        clips = st["clips"]
        spec_line = plan.spec_line(st["spec"])

        async def one(c: dict) -> dict:
            try:
                text = await llm.chat(
                    prompts.VERIFY_SYSTEM,
                    prompts.verify_user(spec_line, _packets(st["inv"], [c])),
                    thinking=True, trace=st.get("trace"),
                    name=f"verify[{c['scene_id']}]", think_max=MAX_TOKENS_PICK,
                )
                got = plan.parse_verify(text)
                sid = c["scene_id"]
                # 자기 장면 번호만 취한다 — 한 콜이 헛번호를 뱉어 남의 클립 점수를
                # 덮는 경로를 막는다. 번호를 틀렸어도 답이 한 줄이면 이 클립 것이다.
                if sid in got:
                    return {sid: got[sid]}
                return {sid: next(iter(got.values()))} if len(got) == 1 else {}
            except Exception as e:                   # noqa: BLE001 — 소견이 편성을 죽이면 안 된다
                log.warning("verify 실패(장면%d 채점 없이 진행): %s", c["scene_id"], e)
                return {}

        scores: dict[int, dict] = {}
        for part in await asyncio.gather(*(one(c) for c in clips)):
            scores.update(part)
        log.info("verify: %d콜 동시 · 채점 %d건", len(clips), len(scores))
        broken = [i for i, v in scores.items() if not v["complete"]]
        if broken:
            log.warning("verify 완결성 문제 %s — startfix 신호(클립은 유지)", broken)
        if tr := st.get("trace"):
            tr.node("verify", scores=scores, incomplete=broken)
        return {"scores": scores}

    def select_node(st: ComposeState) -> dict:
        """예산 확정 — 층 순서로 채우고, 마지막이라 예산이 정확하다."""
        spec = st["spec"]
        clips, scores = st["clips"], st.get("scores", {})
        if not use_select:
            picked, spare = _assemble(clips, spec["budget"])
            total = sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked)
            return {"picked": picked, "spare": spare, "total": int(total), "status": "ok"}
        picked, dropped, total = select_mod.choose(clips, spec, scores)
        if not picked:
            picked = select_mod.rescue_longest(clips, spec["budget"])
            total = int(sum(r["cut"]["ce"] - r["cut"]["cs"] for r in picked))
        if note := select_mod.backfill_note(spec):
            log.info("select: %s", note)
        if tr := st.get("trace"):
            tr.node("select", picked=[c["scene_id"] for c in picked], total=total,
                    dropped=[(c["scene_id"], why) for c, why in dropped])
        return {
            "picked": picked, "total": total, "status": "ok" if picked else "empty",
            "dropped": [(c["scene_id"], why) for c, why in dropped],
            "suspicions": [(i, v["reason"]) for i, v in scores.items() if v["score"] <= 1],
        }

    def empty_node(st: ComposeState) -> dict:
        return {"picked": [], "status": "empty"}

    g = StateGraph(ComposeState)
    g.add_node("expand", expand_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("plan", plan_node)
    g.add_node("cut", cut_node)
    g.add_node("feedback", feedback_node)
    g.add_node("bounds", bounds_node)
    g.add_node("verify", verify_node)
    g.add_node("select", select_node)
    g.add_node("empty", empty_node)
    g.add_edge(START, "expand")
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "cut")
    g.add_conditional_edges(
        "cut", route, {"feedback": "feedback", "bounds": "bounds", "empty": "empty"})
    g.add_edge("feedback", "plan")
    g.add_edge("bounds", "verify")
    g.add_edge("verify", "select")
    g.add_edge("select", END)
    g.add_edge("empty", END)
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
            f"    대사: {stt or '(없음)'}"
        )
    return "\n".join(lines)
