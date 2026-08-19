"""그래프 경로 검증 — 스텁 LLM/벡터로 3경로 + A1·A2·A3·B4 고정 (네트워크 없음)."""

from flow.graph import build_graph, run_compose
from flow.state import Inventory


class StubLLM:
    """콜 이름별 응답 스텁 — 노드가 늘어도 순서에 의존하지 않는다.

    bounds·verify 는 클립당 1콜이라 이름이 "verify[2]" 처럼 붙는다 — 대괄호를 떼고
    찾는다 (안 떼면 스텁이 조용히 빗나가 기본값으로 흘러 테스트가 거짓 통과한다).
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
                game_line="v_id=999 삼성(원정) vs 롯데(홈)", inventory_text="(목록)")

PLAN_OK = "모드: collection\n대상: 안타\n관점: 전체\n예산: 60\n선곡: 1, 2\n사유: 테스트"
PLAN_EMPTY = "모드: collection\n대상: 홈런\n관점: 전체\n예산: 60\n선곡: 없음\n사유: 없음"
VERIFY_OK = "장면 1: 3 정상 근거\n장면 2: 3 정상 근거"
EXPAND_OK = "검색어: 안타, 적시타\n필터: 안타"
DEFAULTS = {"expand": EXPAND_OK, "plan": PLAN_OK, "bounds": "", "verify": VERIFY_OK}


async def test_happy_path():
    g = build_graph(StubLLM(), StubEmb(), StubStore())
    st = await run_compose(g, INV, "안타 모음", None)
    assert st["status"] == "ok" and [r["scene_id"] for r in st["picked"]] == [1, 2]


async def test_empty_after_replan():
    g = build_graph(StubLLM(plan=PLAN_EMPTY), StubEmb(), StubStore())
    st = await run_compose(g, INV, "홈런 모음", None)
    assert st["status"] == "empty" and st["picked"] == []


async def test_must_scene_recovered_when_plan_misses_it():
    """plan 이 놓친 득점 장면을 필수층이 회수한다.

    선곡분만 컷하면 plan 이 빠뜨린 장면은 영영 들어올 수 없어 select 의 필수층이
    무의미해진다 (구 backfill 이 하던 회수).
    """
    scenes = (scene(1, 100, 130, pitch=102),
              scene(2, 200, 240, tags="안타", labels="역전,적시타", delta=1, pitch=202))
    inv = Inventory(v_id=999, scenes=scenes, segs=SEGS, utts=(),
                    game_line="g", inventory_text="(목록)")
    plan_one = "모드: collection\n대상: 안타\n관점: 전체\n예산: 300\n선곡: 1\n사유: 2번 누락"
    g = build_graph(StubLLM(plan=plan_one), StubEmb(), StubStore())
    st = await run_compose(g, inv, "안타 모음", 300)
    assert [r["scene_id"] for r in st["picked"]] == [1, 2]
    assert all("cut" not in s for s in scenes)              # B4 인벤토리 불변


async def test_verify_score_drives_drop_not_removal():
    """0점은 빼되 필수 장면은 유지한다 — 사실이 소견을 이긴다."""
    scenes = (scene(1, 100, 130, tags="범타"),
              scene(2, 200, 240, tags="안타", labels="역전", delta=1, pitch=202))
    inv = Inventory(v_id=999, scenes=scenes, segs=SEGS, utts=(),
                    game_line="g", inventory_text="(목록)")
    plan_two = "모드: collection\n대상: 안타\n관점: 전체\n예산: 300\n선곡: 1, 2\n사유: t"
    zero_both = "장면 1: 0 정상 무관\n장면 2: 0 정상 무관"
    g = build_graph(StubLLM(plan=plan_two, verify=zero_both), StubEmb(), StubStore())
    st = await run_compose(g, inv, "안타 모음", 300)
    ids = [r["scene_id"] for r in st["picked"]]
    assert 2 in ids                                        # 필수(역전·득점) 는 0점이어도 유지
    assert 1 not in ids                                    # 0점 일반 클립은 제외
    assert any(sid == 1 for sid, _ in st["dropped"])


async def test_budget_is_exact_after_bounds():
    """경계 보정이 끝난 뒤 자르므로 총 길이가 예산을 넘지 않는다."""
    g = build_graph(StubLLM(), StubEmb(), StubStore())
    st = await run_compose(g, INV, "안타 모음", 40)
    assert st["total"] <= 40


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


async def test_bounds_and_verify_fan_out_per_clip():
    """클립마다 1콜씩 나가야 한다 — 한 콜에 몰면 전송이 직렬이라 GPU 가 논다.

    묶어 보내던 시절 v201 은 bounds 10분 26초 · verify 9분 8초를 쓰면서 서버는 내내
    Running 1 · KV 3% 였다 (실측 2026-08-20). 콜 수는 눈으로 안 보이니 여기서 고정한다.
    """
    # bounds 는 후보가 있는 클립만 묻는다 — 보드 투구와 꼬리 발화를 깔아 둘 다 물게 한다
    inv = Inventory(v_id=999, scenes=SCENES, segs=SEGS,
                    utts=((118.0, 124.0, "넘어갑니다"), (212.0, 218.0, "잡아냅니다")),
                    game_line="g", inventory_text="(목록)",
                    pitches=((96, 97), (196, 197)))
    llm = CapturingLLM()
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, inv, "안타 모음", 300)
    assert llm.names_of("bounds") == ["bounds[1]", "bounds[2]"]
    assert sorted(llm.names_of("verify")) == ["verify[1]", "verify[2]"]
    # 프롬프트에는 자기 클립만 실린다 (남의 장면이 섞이면 나눈 의미가 없다)
    _sys, user = next((s, u) for s, u, n in llm.calls if n == "verify[1]")
    assert "[클립] 1 " in user and "[클립] 2 " not in user
    assert "[질의] 안타 모음" in user            # 질의가 실려야 채점 기준이 선다


async def test_caller_budget_reaches_plan_prompt():
    """호출자 예산이 plan 프롬프트에 실제로 실린다.

    기본값(180)만 보내던 배선 탓에 900s 요청에도 모델이 "예산: 180"으로 답하며 그
    전제로 선곡했다 (2026-08-19 실측). 프롬프트 문자열까지 확인해야 잡히는 종류라
    응답 파싱이 아니라 송신 내용을 본다.
    """
    llm = CapturingLLM()
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, INV, "안타 모음", 900)
    plan_system, _ = llm.prompt_of("plan")
    assert "900" in plan_system
    assert "{budget}" not in plan_system            # 치환 누락 방지
    assert str(180) not in plan_system.split("예산(초)이")[1][:80]   # 기본값 잔존 금지


async def test_budget_omitted_falls_back_to_default():
    """예산 미지정이면 기본값이 그대로 프롬프트에 실린다."""
    from flow import plan as plan_mod

    llm = CapturingLLM()
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, INV, "안타 모음", None)
    assert str(plan_mod.DEFAULT_BUDGET_SEC) in llm.prompt_of("plan")[0]


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


def test_render_inventory_carries_situation_and_team():
    """상황·팀명이 실제로 한 줄에 실린다 — 둘 다 라벨로는 표현되지 않는 정보."""
    from flow.plan import render_inventory

    rows = [{"scene_id": 65, "scene_type": "범타", "labels": "", "inning": "8회 말",
             "score": "삼성 9-7 롯데", "score_before": "9-7", "s": 100.0, "e": 114.0,
             "outs": 0, "bases": "100", "away_team": "삼성", "home_team": "롯데"}]
    line = render_inventory(rows)
    assert "0사 1루" in line
    assert "삼성 9-7→9-7" in line


def test_render_inventory_survives_missing_transition():
    """전이 조인이 비어도(병합·판독 공백) 렌더가 죽지 않는다."""
    from flow.plan import render_inventory

    rows = [{"scene_id": 1, "scene_type": "범타", "labels": "", "inning": "1회 초",
             "score": "", "score_before": None, "s": 0.0, "e": 10.0,
             "outs": None, "bases": None, "away_team": None, "home_team": None}]
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
