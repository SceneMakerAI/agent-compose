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
_RENAMED_TAGS = {"실책": "수비실책"}            # vision3 어휘 SSOT 이름 (드리프트 실측 2026-08-23)
_RENAMED_LABELS = {"희생플라이": "진루타"}       # 전광판으론 뜬공/땅볼 구분 불가 → 포괄 명칭
# 판세 축(game_context)으로 이사한 라벨 — bench4 RANK_LABEL_BONUS 에는 남아 있다.
_MOVED_TO_CONTEXT = {"역전", "동점"}


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


def scene(scene_id, s, e, *, tags="안타", labels="", delta=0, inning="7회초",
          pitch=None, context=None):
    """발행본 행 흉내 — 키는 SourceRepo.fetch_scenes 산출과 같아야 한다.

    이닝 표기는 '7회초'(공백 없음) — 상류 t_play_baseball.inning 원문 그대로다.
    """
    return {"scene_id": scene_id, "s": float(s), "e": float(e),
            "tags": tags.split(",") if tags else [],
            "label_list": labels.split(",") if labels else [],
            "game_context": context,
            "score_delta": delta, "inning": inning, "pitch_sec": pitch}


SCENES = [
    scene(1, 100, 130, tags="홈런", delta=1, inning="1회초", pitch=102),
    scene(2, 200, 240, tags="안타", labels="적시타", context="역전", delta=2,
          inning="8회말", pitch=205),
    scene(3, 300, 320, tags="삼진", inning="9회초", pitch=301),
    scene(4, 400, 460, tags="범타", labels="병살", inning="5회말", pitch=None),
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


def test_rank_order_is_deterministic():
    """rank 는 중요도 내림차순·동점은 시간순. bench4 전행 등가 대조는 **은퇴**했다.

    상류가 판세(역전·동점)를 labels 에서 game_context 로 옮기면서(2026-08-23)
    가중이 RANK_LABEL_BONUS 에서 RANK_CONTEXT_BONUS 로 갈라졌다. bench4 는 동결된
    POC 라 그 축이 아예 없어, 계속 대조하면 **낡은 계약을 붙잡는 쪽**이 된다.
    이식 검증은 끝났고 지금 지켜야 하는 건 정렬 규칙 자체다 (cut 은 등가 유지).
    """
    rows = rank.order(SCENES, [1, 2, 3, 4])
    scores = [rank.score(r) for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert [r["scene_id"] for r in rows] == [2, 1, 3, 4]
    # 장면2: 델타2×3 + 적시타1 + 역전4(판세) + 8회//3=2 = 13 — 판세가 실제로 걸린다
    assert rank.score(rows[0]) == 13


def test_rank_survives_unjudged_scene():
    """해석이 빈 장면(labels NULL)도 점수가 난다 — 실측 v200~203 에 12행.

    구 스키마에선 `(scene_type or "").split(",")` 이 `[""]` 를 돌려줘 태그 목록이
    우연히 안 비었다. 이제 빈 목록이라 max() 에 default 가 없으면 ValueError 로
    편성 전체가 죽는다.
    """
    r = scene(9, 100, 130, tags="", labels="", delta=0, inning="3회초")
    assert r["tags"] == [] and rank.score(r) == 1        # 이닝 가중만


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
        # anchor_type 은 bench4 이후 신설 — 좌표가 아니라 하류(start_rows 게이트)가
        # "이 시작을 믿을 수 있나"를 판단할 재료다. 등가 대상은 좌표·모드다.
        ours.pop("anchor_type", None)
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
    assert {k: v for k, v in vocab.CUT_RECIPE.items() if k not in _NEW_TAGS} == {
        _RENAMED_TAGS.get(k, k): v for k, v in c.CUT_RECIPE.items()
    }
    assert vocab.FULL_CLIP_TAGS == c.FULL_CLIP_TAGS
    assert vocab.LABEL_EXTRA_SHOTS == {
        _RENAMED_LABELS.get(k, k): v for k, v in c.LABEL_EXTRA_SHOTS.items()
    }
    # 판세로 이사한 라벨은 **값을 그대로** 들고 갔다 — 구조만 갈라졌지 숫자는 안 건드렸다.
    merged = vocab.RANK_LABEL_BONUS | vocab.RANK_CONTEXT_BONUS
    assert all(merged[k] == v for k, v in c.RANK_LABEL_BONUS.items())
    assert set(c.RANK_LABEL_BONUS) - set(vocab.RANK_LABEL_BONUS) == _MOVED_TO_CONTEXT
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


_V3 = Path(__file__).resolve().parents[2] / "agent-vision3" / "src" / "sports" / "baseball"


def _load_v3(rel: str):
    """vision3 모듈 소스를 AST 로 연다 — 없으면 스킵 (모노레포 밖 실행 허용)."""
    path = _V3 / rel
    if not path.exists():
        pytest.skip(f"agent-vision3/{rel} 미존재")
    return ast.parse(path.read_text(encoding="utf-8"))


def _derived_labels(tree: ast.Module) -> set[str]:
    """vision3 `scene/judge.derive` 가 붙일 수 있는 파생 라벨 전부.

    derive() 는 `[lab for cond, lab in [(조건, "라벨"), ...] if cond]` 형태다 —
    그 함수 안 2원소 튜플의 **두 번째** 원소만 집는다. 조건절에 섞인 태그명
    ("삼진" 등)까지 긁지 않으려면 위치가 필요하다. vision3 가 이 형태를 바꾸면
    여기서 빈 집합이 되어 테스트가 깨진다 (조용히 통과하는 것보다 낫다).

    2026-08-23 이전에는 `publish/__init__.py` 의 labels()·end_labels() 가 출처였다.
    scene 재설계로 그 함수들이 사라지면서 판정·파생이 scene/judge 로 옮겨왔다.
    """
    found: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name != "derive":
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Tuple) and len(node.elts) == 2
                    and isinstance(node.elts[1], ast.Constant)
                    and isinstance(node.elts[1].value, str)):
                found.add(node.elts[1].value)
    return found


def _context_names(tree: ast.Module) -> set[str]:
    """vision3 `scene/context` 가 붙일 수 있는 판세 이름 전부.

    `WALKOFF, FIRST, LEAD_CHANGE, TIE = "끝내기", "선제", "역전", "동점"` 한 줄이
    출처다 — 모듈 최상단의 다중 대입에서 문자열만 집는다.
    """
    found: set[str] = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in node.value.elts)):
            found |= {e.value for e in node.value.elts}
    return found


def test_vocab_labels_synced_with_vision3():
    """파생 라벨은 vision3 scene/judge.derive 가 실제로 붙이는 문자열과 **양방향** 등가.

    한쪽(우리 것이 저쪽에 있나)만 보면 저쪽에 새로 생긴 라벨을 영원히 못 잡는다:
    '희생플라이'→'진루타' 개명이 그렇게 새어 LABEL_EXTRA_SHOTS 키가 죽은 이름으로
    남았고(실측 comp 15·16), 뒤이어 '끝내기'가 같은 구멍으로 빠져 경기 최고 장면이
    가산점 없이 일반 안타와 동급이 됐다.
    """
    got = _derived_labels(_load_v3("scene/judge.py"))
    assert got, "derive() 에서 라벨을 못 읽었다 — vision3 가 형태를 바꿨다"
    assert got == set(vocab.LABELS)


def test_vocab_contexts_synced_with_vision3():
    """판세 어휘는 vision3 scene/context 와 등가 — labels 에서 갈라져 나온 축.

    이 축이 어긋나면 RANK_CONTEXT_BONUS 가 조용히 안 걸린다. 역전·동점이 정확히
    그렇게 죽어 있었다: 상류가 두 값을 game_context 로 옮겼는데 이쪽 LABELS 에
    이름만 남아 `validate()` 도 통과했다 (2026-08-23 실측).
    """
    got = _context_names(_load_v3("scene/context.py"))
    assert got, "판세 이름을 못 읽었다 — vision3 가 형태를 바꿨다"
    assert got == set(vocab.GAME_CONTEXTS)


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
    assert f"- 판세: {prompts.CONTEXT_VOCAB}" in prompts.PLAN_SYSTEM
    assert "{budget}" not in prompts.PLAN_SYSTEM     # 예산은 값으로 안 간다 — 질의 문구뿐


def test_bound_systems_share_one_shape():
    """끝·시작 시스템 프롬프트가 같은 뼈대다 — 역할 한 줄 + 세 블록 + 같은 출력 규약.

    두 노드가 하는 일이 "제시된 지점 중 고르기"로 같은데 시작 쪽만 [시스템 역할 및
    규칙] 머리·번호 매긴 절차로 남아 있었다 (2026-08-24 통일).
    """
    from flow import prompts

    for sysmsg in (prompts.END_SYSTEM, prompts.START_SYSTEM):
        assert sysmsg.startswith("당신은 야구 하이라이트 클립의 ")
        assert "[시스템 역할 및 규칙]" not in sysmsg
        for block in ("[후보 읽는 법]", "[고르는 법]", "[RULES]",
                      "[OUTPUT — 번호 하나 또는 \"유지\", 다른 말 금지]"):
            assert block in sysmsg, (block, sysmsg[:40])
        assert sysmsg.rstrip().endswith("2")          # 응답 예시는 번호 하나
        assert "번호 중 하나" in sysmsg and "시각이나" in sysmsg


def test_start_system_names_no_absent_block():
    """시작 시스템 프롬프트는 **없는 블록**을 가리키지 않는다 (2026-08-24).

    647a8d1 이 고친 것과 같은 결함이다 — 프롬프트가 "아래 [구간 화면]에서 찾아라"고
    말하는데 유저 메시지에 그 블록이 없으면 모델은 없는 재료를 찾다 사고를 태운다.
    구간 블록을 뺐으니 그 지시도 함께 사라져야 한다(근거: bounds.start_rows).
    """
    from flow import prompts

    assert "[구간 화면]" not in prompts.START_SYSTEM
    assert "[구간 대사]" not in prompts.START_SYSTEM
    # 대신 끝과 같은 안내: 후보 줄이 자기완결적이다.
    assert "후보마다 그 시각의 화면과 해설이 함께 적혀 있다" in prompts.START_SYSTEM


def test_plan_system_leaves_length_to_code():
    """분량 산술은 모델 몫이 아니다 — 예산은 인자로 받아 finish 가 덜어낸다.

    배수 보정을 걸었다가 되돌린 자리다: 85장면 길이를 합산해 목표를 맞추는 산술이
    select_clips 를 480초 가드 밖으로 밀었다(2배 comp43 · 1.5배 comp44 492초;
    규칙 없는 comp42 는 204.8초 완주). 이 줄이 되살아나면 그 실패가 재현된다.
    """
    from flow import prompts

    assert "분량은 신경 쓰지 않습니다" in prompts.PLAN_SYSTEM
    assert "배" not in prompts.PLAN_SYSTEM.split("4. 분량 규칙")[1].split("[출력")[0]


def test_plan_system_closes_with_game_ending_out():
    """일반 하이라이트 요청이면 경기 종료(마지막 아웃) 장면으로 닫으라는 규칙.

    찾는 근거는 인벤토리의 '전광판'(t_scene_baseball.tags) 줄에 있는 '이닝종료' 다.
    그 표식은 **매 반이닝 끝마다** 붙으므로 탐색을 가장 늦은 이닝으로 묶어야 한다 —
    안 묶으면 8회말 종료 아웃을 경기 종료로 집는다.

    없을 수도 있다: v1003 은 9회말 마지막 장면이 '2루'(안타)이고 그 이닝에 '이닝종료'가
    없다(중계가 경기 종료 전에 끝났다). 그때 억지로 채우지 말라는 줄이 함께 있어야 한다.
    """
    from flow import prompts

    assert "이닝종료" in prompts.PLAN_SYSTEM
    assert "가장 늦은 이닝" in prompts.PLAN_SYSTEM
    assert "넣지 않습니다" in prompts.PLAN_SYSTEM        # 없을 때 폴백
