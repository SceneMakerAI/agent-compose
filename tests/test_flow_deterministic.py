"""결정 계층 등가 검증 — rank·cut 을 bench4 원본 모듈과 직접 대조 (모노레포 전제).

이식 게이트: LLM 무관 순수 계산은 bench4 와 **전행 일치**해야 한다 (compose-flow.md §5-2).
bench4 모듈은 sys.path 주입으로 로드 — 값 복사가 아니라 원본 실행 결과와 비교하므로
bench4 쪽 상수가 바뀌면 이 테스트가 드리프트를 즉시 드러낸다.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

from flow import cut, rank, vocab

_BENCH4 = Path(__file__).resolve().parents[3] / "poc" / "poc-search-bench4" / "src"

# bench4 는 동결된 POC 라, 어휘는 vision3 를 따라가며 의도적으로 갈라진다.
# 그 갈라진 지점만 여기 명시하고 나머지는 계속 전행 등가로 묶는다 —
# 목록을 늘리지 않으면 새 드리프트는 그대로 실패로 드러난다.
_NEW_TAGS = {"보크"}                          # bench4 이후 vision3 에 추가된 태그
_RENAMED_LABELS = {"희생플라이": "진루타"}       # 전광판으론 뜬공/땅볼 구분 불가 → 포괄 명칭


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
    """FULL_CLIP_TAGS 장면은 등가 대상에서 뺀다 — 2026-08-20 정책 분리.

    bench4 는 그 태그에서 시작·끝을 **둘 다** 포기해 통째로 냈다. 지금은 끝만 통째로
    두고 시작은 앵커를 쓴다 (cut 모듈 docstring — v202 장면11 이 그 규칙으로 풀린다).
    나머지 경로는 여전히 byte 등가여야 한다.
    """
    from flow import vocab

    for sc in SCENES:
        if set(sc["tags"]) & vocab.FULL_CLIP_TAGS:
            continue
        ours = cut.clip(dict(sc), SEGS, UTTS)
        theirs = bench4["cut"].clip(dict(sc), SEGS, UTTS)
        assert ours == theirs, f"scene {sc['scene_id']} 불일치: {ours} != {theirs}"


def test_cut_full_tag_keeps_start_anchor():
    """홈런은 **끝만** 통째다 — 시작은 앵커(투구 샷)를 쓴다.

    구 동작은 태그 하나로 앵커를 버려 클립이 장면 시작(앞 타석 꼬리)부터 시작했고,
    그 클립이 bounds 로 넘어가 LLM 이 시작을 다시 찾았다. 실측 v202 장면11:
    장면 2558s · 첫 투구 샷 2583s.
    """
    sc = scene(9, 400, 440, tags="홈런", labels="", delta=1, pitch=None)
    got = cut.clip(dict(sc), SEGS, UTTS)
    assert got["anchor"] == (400.0, 405.0)            # 첫 '투구' 샷
    assert got["cs"] == 400.0
    assert got["ce"] == 440.0                          # 끝은 장면 그대로 (레시피로 안 좁힌다)
    assert got["mode"].startswith("끝 통째")


def test_cut_full_tag_without_pitch_shot_stays_whole():
    """'투구' 샷이 아예 없으면 홈런도 예전처럼 통째 — 앵커를 지어내지 않는다.

    5경기 실측에서 이 경로가 6건 남는다(진짜 미해결). 이것만 bounds 가 맡는다.
    """
    segs = [{"seg_id": 1, "s": 500.0, "e": 512.0, "shot_type": "주루"},
            {"seg_id": 2, "s": 512.0, "e": 530.0, "shot_type": "리액션"}]
    sc = scene(9, 500, 530, tags="홈런", labels="", delta=1, pitch=None)
    got = cut.clip(dict(sc), segs, [])
    assert got["anchor"] is None
    assert (got["cs"], got["ce"]) == (500.0, 530.0)
    assert got["mode"] == "통째(레시피 제외 태그)"


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
    assert {k: v for k, v in vocab.CUT_RECIPE.items() if k not in _NEW_TAGS} == c.CUT_RECIPE
    assert vocab.FULL_CLIP_TAGS == c.FULL_CLIP_TAGS
    assert vocab.LABEL_EXTRA_SHOTS == {
        _RENAMED_LABELS.get(k, k): v for k, v in c.LABEL_EXTRA_SHOTS.items()
    }
    assert vocab.RANK_LABEL_BONUS == c.RANK_LABEL_BONUS
    assert vocab.RANK_TAG_BONUS == c.RANK_TAG_BONUS
    # endfix·underfill 상수는 그 단계가 사라져 대조 대상에서 뺐다 (2026-08-20).
    # 등가 테스트가 죽은 상수를 붙잡아 두는 역전을 막는다 — 감시할 건 **쓰는 값**이다.
    assert (vocab.LABEL_EXTRA_MAX_SEC, vocab.DIALOGUE_TAIL_MAX_SEC, vocab.MAX_REPLAN) == (
        c.LABEL_EXTRA_MAX_SEC, c.DIALOGUE_TAIL_MAX_SEC, c.MAX_REPLAN)


def test_vocab_tags_synced_with_vision3():
    """어휘 SSOT(vision3 vocab.PLAY_TAGS)와 태그 이름 등가 — 복제 드리프트 감시."""
    v3 = Path(__file__).resolve().parents[2] / "agent-vision3" / "src" / "sports" / "baseball" / "vocab.py"
    if not v3.exists():
        pytest.skip("agent-vision3 미존재")
    spec = importlib.util.spec_from_file_location("v3vocab", v3)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(vocab.PLAY_TAGS) == {t.name for t in mod.TAGS}
    assert vocab.PLAY_TAGS == tuple(t.name for t in mod.TAGS)   # 프롬프트 어휘 줄 순서


def _vision3_labels(pub: Path) -> set[str]:
    """vision3 publish 가 t_scene_baseball.labels 에 넣을 수 있는 문자열 전부.

    labels() 는 `[lab for cond, lab in [(조건, "라벨"), ...] if cond]`,
    end_labels() 는 리스트 리터럴 + out.append(...) 형태다. 두 함수 안의 문자열을
    구조로 집어낸다 — 조건절에 섞인 태그명("삼진" 등)까지 긁지 않으려면 위치가
    필요하다. vision3 가 이 형태를 바꾸면 여기서 빈 집합이 되어 테스트가 깨진다
    (조용히 통과하는 것보다 낫다).
    """
    tree = ast.parse(pub.read_text(encoding="utf-8"))
    found: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ("labels", "end_labels"):
            continue
        for node in ast.walk(fn):
            # (조건, "라벨") 튜플의 두 번째 원소
            if isinstance(node, ast.Tuple) and len(node.elts) == 2 \
                    and isinstance(node.elts[1], ast.Constant) \
                    and isinstance(node.elts[1].value, str):
                found.add(node.elts[1].value)
            # out = ["경기 종료"] / out.append("끝내기")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "append" and node.args \
                    and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                found |= {e.value for e in node.value.elts
                          if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return found


def test_vocab_labels_synced_with_vision3():
    """파생 라벨은 vision3 publish 가 실제로 붙이는 문자열과 **양방향** 등가.

    한쪽(우리 것이 저쪽에 있나)만 보면 저쪽에 새로 생긴 라벨을 영원히 못 잡는다:
    '희생플라이'→'진루타' 개명이 그렇게 새어 LABEL_EXTRA_SHOTS 키가 죽은 이름으로
    남았고(실측 comp 15·16), 뒤이어 '끝내기'가 같은 구멍으로 빠져 경기 최고 장면이
    가산점 없이 일반 안타와 동급이 됐다.
    """
    pub = (Path(__file__).resolve().parents[2] / "agent-vision3" / "src" / "sports"
           / "baseball" / "publish" / "__init__.py")
    if not pub.exists():
        pytest.skip("agent-vision3 미존재")
    assert _vision3_labels(pub) == set(vocab.LABELS)


def test_vocab_validate():
    vocab.validate()   # 드리프트 없으면 무예외


# ── 프롬프트 byte 등가 (compose-flow.md §5-1 게이트) ───


def test_plan_user_carries_query_and_inventory():
    """select_clips 유저 프롬프트 조립 — 경기·인벤토리·질의가 제자리에 실린다.

    bench4 byte 등가 게이트였던 자리다 (2026-08-20 폐기): 프롬프트가 줄 형식 명세에서
    **번호만 답하는** 형식으로 바뀌면서 원본과 구조가 갈렸다. 이식 검증은 끝났고
    지금 지켜야 하는 건 "재료가 빠지지 않는가" 다.
    """
    from flow import prompts

    u = prompts.plan_user("홈런 모음", "v_id=203 KT(원정) vs NC(홈)", "[장면 1]\n- 태그: 홈런")
    assert "[경기 정보]\nv_id=203 KT(원정) vs NC(홈)" in u
    assert "[인벤토리]" in u and "[장면 1]" in u
    assert u.rstrip().endswith("[질의]\n홈런 모음")
    assert "[이전 시도 피드백]" not in u              # 피드백 없으면 블록째 빠진다
    assert "[이전 시도 피드백]\n다시" in prompts.plan_user("q", "g", "inv", "다시")


def test_plan_system_keeps_vocab():
    """문구는 자유롭게 고치되 어휘 렌더는 남아야 한다.

    하드코딩하면 vision3 에 태그가 늘어도 프롬프트만 옛 어휘로 남는다(보크 실측).
    """
    from flow import prompts

    assert f"- 태그: {prompts.TAG_VOCAB}" in prompts.PLAN_SYSTEM
    assert f"- 라벨: {prompts.LABEL_VOCAB}" in prompts.PLAN_SYSTEM
    assert "{budget}" not in prompts.PLAN_SYSTEM     # 예산은 프롬프트로 가지 않는다
