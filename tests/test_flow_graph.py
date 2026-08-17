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
