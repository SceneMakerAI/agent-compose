"""결정 계층 등가 검증 — rank·cut 을 bench4 원본 모듈과 직접 대조 (모노레포 전제).

이식 게이트: LLM 무관 순수 계산은 bench4 와 **전행 일치**해야 한다 (compose-flow.md §5-2).
bench4 모듈은 sys.path 주입으로 로드 — 값 복사가 아니라 원본 실행 결과와 비교하므로
bench4 쪽 상수가 바뀌면 이 테스트가 드리프트를 즉시 드러낸다.
"""

import importlib
import sys
from pathlib import Path

import pytest

from flow import cut, rank, vocab

_BENCH4 = Path(__file__).resolve().parents[3] / "poc" / "poc-search-bench4" / "src"


@pytest.fixture(scope="module")
def bench4():
    """bench4 rank·cut·config 원본 로드 (없으면 스킵 — 모노레포 밖 실행 허용).

    compose 패키지 __init__ 는 pymysql·langgraph 를 끌고 오므로 파일 단위로 직접
    로드한다. 우리 쪽 config/log 와 모듈명이 겹치므로 sys.modules 를 잠시 바꿔치기.
    """
    if not _BENCH4.exists():
        pytest.skip("poc-search-bench4 미존재 — 등가 대조 생략")
    saved = {m: sys.modules.pop(m) for m in ("config", "log") if m in sys.modules}
    sys.path.insert(0, str(_BENCH4))
    try:
        b_config = importlib.import_module("config")   # bench4/src/config.py
        importlib.import_module("log")                 # rank·cut 의 `from log import ...`

        def _load(name: str, rel: str):
            spec = importlib.util.spec_from_file_location(name, _BENCH4 / rel)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        b_rank = _load("b4_rank", "compose/rank.py")
        b_cut = _load("b4_cut", "compose/cut.py")
        yield {"config": b_config, "rank": b_rank, "cut": b_cut}
    finally:
        sys.path.remove(str(_BENCH4))
        for m in ("config", "log", "b4_rank", "b4_cut"):
            sys.modules.pop(m, None)
        sys.modules.update(saved)


def scene(scene_id, s, e, *, tags="안타", labels="", delta=0, inning="7회 초", pitch=None):
    return {"scene_id": scene_id, "s": float(s), "e": float(e),
            "tags": tags.split(","), "label_list": labels.split(",") if labels else [],
            "score_delta": delta, "inning": inning, "pitch_sec": pitch}


SCENES = [
    scene(1, 100, 130, tags="홈런", delta=1, inning="1회 초", pitch=102),
    scene(2, 200, 240, tags="안타", labels="적시타,역전", delta=2, inning="8회 말", pitch=205),
    scene(3, 300, 320, tags="삼진", inning="9회 초", pitch=301),
    scene(4, 400, 460, tags="범타", labels="병살", inning="5회 말", pitch=None),
]
SEGS = [
    {"seg_id": 1, "s": 100.0, "e": 106.0, "shot_type": "투구"},
    {"seg_id": 2, "s": 106.0, "e": 118.0, "shot_type": "기타"},
    {"seg_id": 3, "s": 200.0, "e": 207.0, "shot_type": "투구"},
    {"seg_id": 4, "s": 207.0, "e": 215.0, "shot_type": "타구·수비"},
    {"seg_id": 5, "s": 215.0, "e": 222.0, "shot_type": "득점·홈인"},
    {"seg_id": 6, "s": 222.0, "e": 231.0, "shot_type": "리플레이"},
    {"seg_id": 7, "s": 300.0, "e": 304.0, "shot_type": "투구"},
    {"seg_id": 8, "s": 304.0, "e": 315.0, "shot_type": "리액션"},
    {"seg_id": 9, "s": 400.0, "e": 405.0, "shot_type": "투구"},
    {"seg_id": 10, "s": 405.0, "e": 412.0, "shot_type": "타구·수비"},
    {"seg_id": 11, "s": 412.0, "e": 420.0, "shot_type": "주루"},
    {"seg_id": 12, "s": 420.0, "e": 434.0, "shot_type": "리액션"},
]
UTTS = [(213.0, 219.5, "적시타! 주자 들어옵니다"), (302.0, 316.0, "삼진 아웃 긴 발화")]


def test_rank_equiv(bench4):
    ours = [(r["scene_id"], rank.score(r)) for r in rank.order(SCENES, [1, 2, 3, 4])]
    theirs = [(r["scene_id"], bench4["rank"].score(r))
              for r in bench4["rank"].order(SCENES, [1, 2, 3, 4])]
    assert ours == theirs


def test_cut_equiv(bench4):
    for sc in SCENES:
        ours = cut.clip(dict(sc), SEGS, UTTS)
        theirs = bench4["cut"].clip(dict(sc), SEGS, UTTS)
        assert ours == theirs, f"scene {sc['scene_id']} 불일치: {ours} != {theirs}"


def test_cut_immutability():
    """B4 — cut 은 입력 행·세그를 수정하지 않는다."""
    sc = scene(2, 200, 240, tags="안타", labels="적시타", delta=2, pitch=205)
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in sc.items()}
    segs_before = [dict(s) for s in SEGS]
    cut.clip(sc, SEGS, UTTS)
    assert sc == before and SEGS == segs_before


def test_vocab_constants_equal_bench4(bench4):
    """상수 드리프트 감시 — bench4 config 와 값 일치."""
    c = bench4["config"]
    assert vocab.CUT_RECIPE == c.CUT_RECIPE
    assert vocab.FULL_CLIP_TAGS == c.FULL_CLIP_TAGS
    assert vocab.LABEL_EXTRA_SHOTS == c.LABEL_EXTRA_SHOTS
    assert vocab.RANK_LABEL_BONUS == c.RANK_LABEL_BONUS
    assert vocab.RANK_TAG_BONUS == c.RANK_TAG_BONUS
    assert (vocab.LABEL_EXTRA_MAX_SEC, vocab.DIALOGUE_TAIL_MAX_SEC,
            vocab.ENDFIX_MAX_EXT_SEC, vocab.ENDFIX_UTT_MAX,
            vocab.MAX_REPLAN, vocab.UNDERFILL_MIN_FRAC) == (
        c.LABEL_EXTRA_MAX_SEC, c.DIALOGUE_TAIL_MAX_SEC,
        c.ENDFIX_MAX_EXT_SEC, c.ENDFIX_UTT_MAX, c.MAX_REPLAN, c.UNDERFILL_MIN_FRAC)


def test_vocab_tags_synced_with_vision3():
    """어휘 SSOT(vision3 vocab.PLAY_TAGS)와 태그 이름 등가 — 복제 드리프트 감시."""
    v3 = Path(__file__).resolve().parents[2] / "agent-vision3" / "src" / "sports" / "baseball" / "vocab.py"
    if not v3.exists():
        pytest.skip("agent-vision3 미존재")
    spec = importlib.util.spec_from_file_location("v3vocab", v3)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(vocab.PLAY_TAGS) == {t.name for t in mod.TAGS}


def test_vocab_validate():
    vocab.validate()   # 드리프트 없으면 무예외


# ── 프롬프트 byte 등가 (compose-flow.md §5-1 게이트) ───

def test_prompts_byte_equal_bench4(bench4):
    """PLAN·ENDFIX 는 byte 등가, VERIFY 는 A2 사유 줄 하나만 다름을 고정."""
    spec = importlib.util.spec_from_file_location("b4_prompt", _BENCH4 / "compose" / "prompt.py")
    b4p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b4p)
    from flow import prompts
    assert prompts.PLAN_SYSTEM == b4p.PLAN_SYSTEM
    assert prompts.ENDFIX_SYSTEM == b4p.ENDFIX_SYSTEM
    ours = prompts.VERIFY_SYSTEM.splitlines()
    theirs = b4p.VERIFY_SYSTEM.splitlines()
    diff = [(a, b) for a, b in zip(theirs, ours) if a != b]
    assert len(ours) == len(theirs) and len(diff) == 1          # 정확히 한 줄만
    assert diff[0][0].startswith("사유:") and "장면 <번호>" in diff[0][1]   # 그 줄 = A2 수정
    # user 프롬프트 렌더도 동일 입력에서 byte 등가
    assert prompts.plan_user("q", "g", "inv", 180, "fb", "ev") == \
        b4p.plan_user("q", "g", "inv", 180, "fb", "ev")
    rows = [(7, 100, [(95.0, 104.2, "발화")])]
    assert prompts.endfix_user(rows) == b4p.endfix_user(rows)
    assert prompts.verify_user("스펙", "패킷") == b4p.verify_user("스펙", "패킷")
