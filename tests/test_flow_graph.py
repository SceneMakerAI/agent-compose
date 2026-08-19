"""그래프 경로 검증 — 스텁 LLM/벡터로 3경로 + A1·A2·A3·B4 고정 (네트워크 없음)."""

from flow.graph import build_graph, run_compose
from flow.state import Inventory


class StubLLM:
    def __init__(self, answers):
        self.answers = list(answers)

    async def chat(self, system, user, thinking=False):
        return self.answers.pop(0)


class StubEmb:
    async def embed_query(self, i, q):
        return [0.0]


class StubStore:
    async def search(self, qv, v_id):
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
VERIFY_OK = "판정: ok\n기각: 없음\n사유: 없음"


async def test_happy_path():
    g = build_graph(StubLLM([PLAN_OK, VERIFY_OK]), StubEmb(), StubStore())
    st = await run_compose(g, INV, "안타 모음", None)
    assert st["status"] == "ok" and [r["scene_id"] for r in st["picked"]] == [1, 2]


async def test_empty_after_replan():
    g = build_graph(StubLLM([PLAN_EMPTY, PLAN_EMPTY]), StubEmb(), StubStore())
    st = await run_compose(g, INV, "홈런 모음", None)
    assert st["status"] == "empty" and st["picked"] == []


async def test_backfill_then_endfix_and_perclip_suspicion():
    """A1: backfill 충원분도 endfix 노드를 지난다 / A2: 클립별 사유 / A3: total 갱신."""
    plan_underfill = ("모드: collection\n대상: 안타,범타\n관점: 전체\n예산: 300\n"
                      "선곡: 1\n사유: 미달 유도")
    verify_perclip = "판정: ok\n기각: 2\n사유: 장면 2: 관점 위반 테스트"
    g = build_graph(StubLLM([plan_underfill, verify_perclip]), StubEmb(), StubStore())
    st = await run_compose(g, INV, "길게", 300)
    assert [r["scene_id"] for r in st["picked"]] == [1, 2]          # backfill 충원
    assert st["total"] == sum(r["cut"]["ce"] - r["cut"]["cs"] for r in st["picked"])  # A3
    assert st["suspicions"] == [(2, "관점 위반 테스트")]              # A2
    assert all("cut" not in s for s in SCENES)                       # B4 인벤토리 불변


class CapturingLLM(StubLLM):
    """system/user 프롬프트를 보존하는 스텁 — 배선 검증용."""

    def __init__(self, answers):
        super().__init__(answers)
        self.calls: list[tuple[str, str]] = []

    async def chat(self, system, user, thinking=False):
        self.calls.append((system, user))
        return await super().chat(system, user, thinking)


async def test_caller_budget_reaches_plan_prompt():
    """호출자 예산이 plan 프롬프트에 실제로 실린다.

    기본값(180)만 보내던 배선 탓에 900s 요청에도 모델이 "예산: 180"으로 답하며 그
    전제로 선곡했다 (2026-08-19 실측). 프롬프트 문자열까지 확인해야 잡히는 종류라
    응답 파싱이 아니라 송신 내용을 본다.
    """
    llm = CapturingLLM([PLAN_OK, VERIFY_OK])
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, INV, "안타 모음", 900)
    plan_system = llm.calls[0][0]
    assert "900" in plan_system
    assert "{budget}" not in plan_system            # 치환 누락 방지
    assert str(180) not in plan_system.split("예산(초)이")[1][:80]   # 기본값 잔존 금지


async def test_budget_omitted_falls_back_to_default():
    """예산 미지정이면 기본값이 그대로 프롬프트에 실린다."""
    from flow import plan as plan_mod

    llm = CapturingLLM([PLAN_OK, VERIFY_OK])
    g = build_graph(llm, StubEmb(), StubStore())
    await run_compose(g, INV, "안타 모음", None)
    assert str(plan_mod.DEFAULT_BUDGET_SEC) in llm.calls[0][0]


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
