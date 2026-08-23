"""그래프 경로 검증 — 스텁 LLM/벡터로 3경로 + A1·A2·A3·B4 고정 (네트워크 없음)."""
import re

from flow.graph import build_graph, run_compose
from flow.state import Inventory


class StubLLM:
    """콜 이름별 응답 스텁 — 노드가 늘어도 순서에 의존하지 않는다.

    refine_bounds·score_match 는 클립당 1콜이라 이름이 "score_match[2]" 처럼 붙는다 —
    대괄호를 떼고 찾는다 (안 떼면 스텁이 조용히 빗나가 기본값으로 흘러 테스트가 거짓 통과한다).
    """

    def __init__(self, by_name: dict | None = None, **kw):
        self.by_name = dict(by_name or {}, **kw)

    async def chat(self, system, user, thinking=False, trace=None, name=""):
        base = name.split("[")[0]
        return self.by_name.get(base, DEFAULTS.get(base, ""))


class StubEmb:
    async def embed_query(self, i, q):
        return [0.0]


class StubStore:
    async def search(self, qv, v_id, extra=None):
        return []


def scene(i, s, e, *, tags="안타", labels="", delta=0, inning="1회 초", pitch=None):
    return {"scene_id": i, "h_id": i, "s": float(s), "e": float(e),
            "scene_type": tags, "tags": tags.split(","),
            "labels": labels, "label_list": labels.split(",") if labels else [],
            "score_delta": delta, "inning": inning,
            "score": "삼성 1-0 롯데", "score_before": "0-0", "pitch_sec": pitch}


SCENES = (scene(1, 100, 130, pitch=102), scene(2, 200, 240, tags="범타", pitch=202))
SEGS = ({"seg_id": 1, "s": 100.0, "e": 106.0, "shot_type": "투구"},
        {"seg_id": 2, "s": 106.0, "e": 120.0, "shot_type": "타구·수비"},
        {"seg_id": 3, "s": 200.0, "e": 206.0, "shot_type": "투구"},
        {"seg_id": 4, "s": 206.0, "e": 214.0, "shot_type": "타구·수비"})
INV = Inventory(v_id=999, scenes=SCENES, segs=SEGS, utts=(),
                game_line="v_id=999 삼성(원정) vs 롯데(홈)")

PLAN_OK = "모드: collection\n대상: 안타\n관점: 전체\n예산: 60\n선곡: 1, 2\n사유: 테스트"
PLAN_EMPTY = "모드: collection\n대상: 홈런\n관점: 전체\n예산: 60\n선곡: 없음\n사유: 없음"
VERIFY_OK = "장면 1: 3 정상 근거\n장면 2: 3 정상 근거"
EXPAND_OK = "검색어: 안타, 적시타\n필터: 안타"
DEFAULTS = {"rephrase_query": EXPAND_OK, "select_clips": PLAN_OK,
            "refine_end_bound": "", "refine_start_bound": ""}


async def test_happy_path():
    g = build_graph(StubLLM(), StubEmb(), StubStore())
    st = await run_compose(g, INV, "안타 모음")
    assert st["status"] == "ok" and [r["scene_id"] for r in st["picked"]] == [1, 2]


async def test_empty_after_replan():
    g = build_graph(StubLLM(select_clips=PLAN_EMPTY), StubEmb(), StubStore())
    st = await run_compose(g, INV, "홈런 모음")
    assert st["status"] == "empty" and st["picked"] == []


class CapturingLLM(StubLLM):
    """system/user 프롬프트를 보존하는 스텁 — 배선 검증용."""

    def __init__(self, by_name: dict | None = None, **kw):
        super().__init__(by_name, **kw)
        self.calls: list[tuple[str, str, str]] = []

    async def chat(self, system, user, thinking=False, trace=None, name=""):
        self.calls.append((system, user, name))
        return await super().chat(system, user, thinking, trace, name)

    def prompt_of(self, call: str) -> tuple[str, str]:
        """해당 콜의 (system, user) — 노드 순서가 바뀌어도 이름으로 찾는다."""
        return next((s, u) for s, u, n in self.calls if n == call)

    def names_of(self, call: str) -> list[str]:
        """해당 콜 이름으로 시작하는 콜 전부 — 팬아웃 건수 확인용."""
        return [n for _s, _u, n in self.calls if n.split("[")[0] == call]


async def test_bound_nodes_fan_out_per_clip():
    """클립마다 1콜씩 나가야 한다 — 한 콜에 몰면 전송이 직렬이라 GPU 가 논다.

    묶어 보내던 시절 v201 은 refine_bounds 10분 26초 · score_match 9분 8초를 쓰면서 서버는 내내
    Running 1 · KV 3% 였다 (실측 2026-08-20). 콜 수는 눈으로 안 보이니 여기서 고정한다.
    """
    # 시작은 **투구 앵커가 없는** 클립만 묻는다. 앵커가 잡히면 cut 의 시작이 정답이라
    # 아예 묻지 않으므로 pitch_sec 과 '투구' 샷을 뺀다(앵커 있는 경우는
    # test_start_rows_skips_clips_with_pitch_anchor). 끝은 모든 클립이 묻되 후보가
    # 있어야 하므로 클립 끝 뒤에 발화 끝을 둔다.
    scenes = (scene(1, 100, 130), scene(2, 200, 240, tags="범타"))
    segs = tuple(dict(s, shot_type="타구·수비", summary=f"샷{s['seg_id']}") for s in SEGS)
    inv = Inventory(v_id=999, scenes=scenes, segs=segs,
                    utts=((108.0, 112.0, "던집니다"), (118.0, 124.0, "넘어갑니다"),
                          (131.0, 136.0, "정리됩니다"),
                          (208.0, 212.0, "던집니다"), (212.0, 218.0, "잡아냅니다"),
                          (241.0, 246.0, "마무리됩니다")),
                    game_line="g",
                    pitches=((110, 111), (210, 211)))
    llm = CapturingLLM()
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, inv, "안타 모음")
    # 끝은 모든 클립이, 시작은 앵커 없는 클립만 묻는다
    assert sorted(llm.names_of("refine_end_bound")) == [
        "refine_end_bound[1]", "refine_end_bound[2]"]
    assert sorted(llm.names_of("refine_start_bound")) == [
        "refine_start_bound[1]", "refine_start_bound[2]"]
    # 프롬프트에는 자기 클립만 실린다 (남의 장면이 섞이면 나눈 의미가 없다)
    _sys, user = next((s, u) for s, u, n in llm.calls if n == "refine_end_bound[1]")
    assert "■ 장면 1 " in user and "■ 장면 2 " not in user


async def test_evidence_is_merged_into_scene_block():
    """검색 증거가 그 장면 블록 안에 실린다 — 번호로만 잇던 조인을 없앴다."""
    class EvStore:
        async def search(self, qv, v_id, extra=None):
            return [{"scene_id": 1, "kind": "shot", "text": "외야수가 담장 앞에서 잡는다",
                     "distance": 0.7, "s": 100.0}]

    llm = CapturingLLM()
    g = build_graph(llm, StubEmb(), EvStore())
    await run_compose(g, INV, "호수비 모음")
    _sys, user = llm.prompt_of("select_clips")
    block = user.split("[장면 1]")[1].split("[장면 2]")[0]
    assert "- 검색증거:" in block and "[화면] 외야수가 담장 앞에서 잡는다" in block


# ── 인벤토리 상황 렌더 ─────────────────────────────────

def test_render_situation():
    """전광판 아웃·주자 → 사람이 읽는 상황 문구. 값 부재는 '?'."""
    from flow.plan import render_situation

    assert render_situation(2, "111") == "2사 만루"
    assert render_situation(1, "110") == "1사 1루·2루"
    assert render_situation(0, "000") == "0사 주자없음"
    assert render_situation(2, "001") == "2사 3루"
    assert render_situation(None, "100") == "?"      # 전이 조인 실패
    assert render_situation(1, None) == "?"


def _inv_row(**kw):
    """인벤토리 렌더 입력 — 키는 SourceRepo.fetch_scenes 산출과 같아야 한다."""
    row = {"scene_id": 65, "tags": ["범타"], "label_list": [], "board_tags": ["아웃"],
           "game_context": None, "inning": "8회말", "score_before": "9-7",
           "score_delta": 0, "s": 100.0, "e": 114.0, "outs": 0, "bases": "100"}
    return {**row, **kw}


def test_render_inventory_carries_situation():
    """상황(아웃·주자)이 실제로 한 줄에 실린다 — 라벨로는 표현되지 않는 정보."""
    from flow.plan import render_inventory

    block = render_inventory([_inv_row()])
    assert "- 아웃: 0" in block and "- 주자: 1루" in block
    assert "- 점수상황: 9-7" in block and "->" not in block   # 무득점은 전개 없음
    assert "[장면 65]" in block


def test_render_inventory_shows_scoring_and_context():
    """득점은 전개로, 판세는 제 줄로 — 판세는 labels 에서 game_context 로 이사했다."""
    from flow.plan import render_inventory

    block = render_inventory([_inv_row(score_delta=2, label_list=["적시타"],
                                       game_context="역전")])
    assert "- 점수상황: 9-7 -> 9-9" in block          # 8회'말' = 홈 공격
    assert "- 라벨: 적시타" in block and "- 판세: 역전" in block


def test_render_inventory_keeps_board_facts_when_unjudged():
    """해석이 비어도 전광판 사실은 남는다 — 그게 tags 컬럼의 존재 이유다.

    실측 v200~203 에 labels 가 NULL 인 행이 12개 있다. 사실까지 빠지면 그 장면은
    인벤토리에서 태그 없는 빈칸이 돼 선곡 후보에조차 못 오른다.
    """
    from flow.plan import render_inventory

    block = render_inventory([_inv_row(tags=[], label_list=[],
                                       board_tags=["아웃", "주자아웃"])])
    assert "- 태그: 판별불가" in block
    assert "- 전광판: 아웃,주자아웃" in block


def test_render_inventory_survives_missing_board():
    """전광판 조인이 비어도(판독 공백) 렌더가 죽지 않는다."""
    from flow.plan import render_inventory

    rows = [_inv_row(scene_id=1, score_before=None, inning=None,
                     outs=None, bases=None, board_tags=[])]
    assert "?" in render_inventory(rows)


# ── ETC 자막 → 타자 이름 ──────────────────────────────

def test_batter_from_etc_prefers_recent_not_frequent():
    """직전 타석 자막이 더 많아도 **가장 최근** 이름을 고른다.

    90초 창은 이전 타석까지 물 수 있다 — 최빈으로 고르면 앞 타자가 이긴다
    (실측 v201 장면 65: 직전 자막은 김동혁인데 최빈은 전민재).
    """
    from flow.players import batter_of

    rows = [(100, "2026 전민재 .251 / 3홈런 20타점"),
            (105, "2026 전민재 .251 / 3홈런 20타점"),
            (110, "2026 전민재 .251 / 3홈런 20타점"),
            (150, "2026 김동혁 .200 ▶ 3안타 9득점"),
            (155, "2026 김동혁 .200 ▶ 3안타 9득점")]
    assert batter_of(rows, 160.0) == "김동혁"


def test_batter_skips_pitcher_lines():
    """투수 성적 줄의 주어는 타자가 아니다 — 경기 초반 선발 소개 자막 방어."""
    from flow.players import batter_of

    rows = [(10, "▶ 김진욱 6일 만의 등판(7.25 vs 두산 승)"),
            (20, "▶ 김진욱 통산 사직 14G 5승 2패 2.59"),
            (40, "2026 김현준 .267 / 7홈런 40타점"),
            (45, "2026 김현준 .267 / 7홈런 40타점")]
    assert batter_of(rows, 50.0) == "김현준"


def test_batter_excludes_crew_and_teams():
    """중계진·팀명은 이름이 아니다. 근거가 없으면 None (억지로 채우지 않는다)."""
    from flow.players import batter_of

    rows = [(10, "캐스터 김수환 해설 민병헌"), (20, "캐스터 김수환 해설 민병헌"),
            (30, "시즌 전적 6승 2패 NC 우세")]
    assert batter_of(rows, 40.0) is None


def test_batter_pair_line_takes_batter():
    """'P 투수 / 번호 타자' 형태는 뒤쪽(타자)을 고른다."""
    from flow.players import batter_of

    rows = [(10, "P 정철원 / 7 전병우 .229 2/3"), (15, "P 정철원 / 7 전병우 .229 2/3")]
    assert batter_of(rows, 20.0) == "전병우"


# ── 예산 절단 (finish) — 2026-08-24 ─────────────────────

def _bclip(sid, cs, ce, **kw):
    """finish 절단이 보는 최소 형태 — rank.score 재료 + cut 좌표."""
    return {"scene_id": sid, "tags": kw.get("tags", ["범타"]),
            "label_list": kw.get("labels", []), "game_context": kw.get("ctx"),
            "score_delta": kw.get("delta", 0), "inning": kw.get("inning", "1회초"),
            "cut": {"cs": float(cs), "ce": float(ce), "mode": "레시피"}}


def _finish(clips, budget=None):
    """실제 구현(rank.fit_budget)을 그대로 부른다 — 로직을 복사하면 코드가 깨져도
    테스트가 통과한다. finish_node 도 같은 함수를 쓴다."""
    from flow import rank

    picked, dropped = rank.fit_budget(clips, budget)
    return [c["scene_id"] for c in picked], [int(re.search(r"장면(\d+)", d).group(1))
                                             for d in dropped]


def test_budget_drops_lowest_rank_first():
    """예산을 넘치면 rank.score 낮은 것부터 버린다 — 큰 플레이가 남는다."""
    clips = [_bclip(1, 0, 60, tags=["범타"]),                    # 점수 0
             _bclip(2, 100, 160, tags=["홈런"], delta=1, ctx="역전"),  # 높음
             _bclip(3, 200, 260, tags=["안타"], labels=["적시타"], delta=1)]
    picked, dropped = _finish(clips, budget=120)
    assert picked == [2, 3] and dropped == [1]


def test_budget_output_stays_in_time_order():
    """절단은 중요도 순으로 하되 **출력은 시간순**이다 — 편성은 경기 흐름대로 돈다."""
    clips = [_bclip(9, 900, 960, tags=["범타"]),
             _bclip(1, 100, 160, tags=["홈런"], delta=1)]
    picked, _ = _finish(clips, budget=1000)
    assert picked == [1, 9]


def test_budget_never_fills_only_removes():
    """예산에 미달해도 **채우지 않는다** — 폐기된 fill_budget 이 그 통로였다(94b58dc).

    선곡에 없던 장면을 예산 채우려고 끌어오면 질의를 규칙이 덮어쓴다.
    """
    clips = [_bclip(1, 0, 30, tags=["범타"])]
    picked, dropped = _finish(clips, budget=600)
    assert picked == [1] and dropped == []          # 30s 뿐이어도 그대로


def test_budget_keeps_at_least_one_clip():
    """첫 클립이 예산보다 길어도 빈 편성을 내지 않는다."""
    clips = [_bclip(1, 0, 300, tags=["홈런"], delta=1), _bclip(2, 400, 430)]
    picked, dropped = _finish(clips, budget=60)
    assert picked == [1] and dropped == [2]


def test_no_budget_means_no_truncation():
    """예산이 없으면 절단하지 않는다 — 선곡이 곧 편성이다."""
    clips = [_bclip(i, i * 100, i * 100 + 90) for i in range(1, 6)]
    picked, dropped = _finish(clips, budget=None)
    assert len(picked) == 5 and dropped == []
