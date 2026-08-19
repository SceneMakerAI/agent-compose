"""bounds 후보 생성·처분 / select 층 구조 — 순수 계산이라 스텁 없이 검증."""

from flow import bounds, select

SEGS = [
    {"s": 100.0, "e": 106.0, "shot_type": "리액션"},
    {"s": 106.0, "e": 112.0, "shot_type": "투구"},
    {"s": 112.0, "e": 120.0, "shot_type": "타구·수비"},
    {"s": 120.0, "e": 128.0, "shot_type": "리액션"},
]
UTTS = [(118.0, 127.5, "넘어갑니다"), (130.0, 134.0, "다음 타석")]


def clip(scene_id=1, cs=115.0, ce=120.0, tags="홈런", labels="", delta=1, inning="1회 초"):
    return {"scene_id": scene_id, "tags": tags.split(","),
            "label_list": labels.split(",") if labels else [],
            "score_delta": delta, "inning": inning, "s": cs, "e": ce,
            "pitch_sec": None, "cut": {"cs": cs, "ce": ce, "mode": "full", "shots": []}}


# ── bounds ────────────────────────────────────────────

def test_start_candidates_use_board_pitch_when_shot_missing():
    """샷 분류가 없어도 보드 검출 투구가 시작 후보가 된다.

    장면 이전은 shot_type 이 NULL 인 경우가 많아(하이라이트 구간만 분류) 보드 검출이
    유일한 단서일 때가 있다 — 역전 홈런이 '주자가 뛰는 장면'부터 시작하던 원인.
    """
    c = clip(cs=115.0)
    got = bounds.start_candidates(c, [{"s": 100.0, "e": 115.0, "shot_type": None}],
                                  [(104, 106)])
    assert (104.0, "보드 검출 투구") in got


def test_start_candidates_prefer_pitch_over_batted():
    """투구 후보가 있으면 타구·수비는 넣지 않는다 (원칙은 투구부터)."""
    got = bounds.start_candidates(clip(cs=115.0), SEGS, [])
    assert any("투구" in why for _sec, why in got)
    assert not any("타구" in why for _sec, why in got)


def test_start_candidates_never_move_forward():
    """시작은 앞으로만 되돌린다 — 뒤로 미루면 플레이를 잘라 먹는다."""
    got = bounds.start_candidates(clip(cs=107.0), SEGS, [(200, 201)])
    assert all(sec < 107.0 for sec, _ in got)


def test_end_candidates_mark_coincidence():
    """발화 끝과 화면 전환이 겹치면 표시한다 — 가장 깔끔한 끝점을 모델이 알게."""
    got = bounds.end_candidates(clip(ce=120.0), SEGS, UTTS)
    assert any("발화 끝 + 화면 전환" in why for _sec, why in got)


def test_apply_rejects_values_outside_candidates():
    """제시하지 않은 초는 기각 — 모델이 시각을 지어낼 수 없다."""
    c = clip(cs=115.0, ce=120.0)
    rows = bounds.build_rows([c], SEGS, UTTS, {})
    bounds.apply([c], rows, "장면 1: 시작 999 끝 888")
    assert (c["cut"]["cs"], c["cut"]["ce"]) == (115.0, 120.0)


def test_apply_moves_when_candidate_matches():
    c = clip(cs=115.0, ce=120.0)
    rows = bounds.build_rows([c], SEGS, UTTS, {})
    start = int(rows[0]["starts"][0][0])
    moved = bounds.apply([c], rows, f"장면 1: 시작 {start} 끝 유지")
    assert c["cut"]["cs"] == float(start) and moved


# ── select ────────────────────────────────────────────

SPEC = {"mode": "collection", "targets": ["안타"], "view": "전체", "budget": 100}


def test_must_layer_survives_zero_score():
    """필수 장면은 verify 0점이어도 유지 — 사실이 소견을 이긴다."""
    must = clip(1, 0, 30, tags="안타", labels="역전", delta=1)
    plain = clip(2, 40, 60, tags="범타", delta=0)
    picked, _dropped, total = select.choose(
        [must, plain], SPEC, {1: {"score": 0, "complete": True, "reason": ""},
                              2: {"score": 0, "complete": True, "reason": ""}})
    ids = [c["scene_id"] for c in picked]
    assert 1 in ids and 2 not in ids
    assert total == 30


def test_budget_is_never_exceeded():
    """예산 초과는 담지 않는다 — 절단이 마지막이라 총합이 정확하다."""
    clips = [clip(i, i * 100, i * 100 + 40, tags="안타", delta=0) for i in range(1, 6)]
    picked, _dropped, total = select.choose(clips, SPEC, {})
    assert total <= SPEC["budget"]
    assert sum(c["cut"]["ce"] - c["cut"]["cs"] for c in picked) == total


def test_coverage_layer_spreads_innings():
    """collection 이면 이닝별 대표를 먼저 채운다."""
    clips = [clip(1, 0, 10, tags="범타", delta=0, inning="1회 초"),
             clip(2, 20, 30, tags="범타", delta=0, inning="1회 초"),
             clip(3, 40, 50, tags="범타", delta=0, inning="2회 초")]
    picked, _d, _t = select.choose(clips, {**SPEC, "targets": [], "budget": 20}, {})
    assert {c["inning"] for c in picked} == {"1회 초", "2회 초"}


def test_must_scenes_exceed_budget_on_purpose():
    """필수 장면은 예산을 넘겨서라도 담는다 — 예산은 목표지 상한이 아니다.

    실측(comp 1): v202 '홈런 모음' 예산 60초인데 홈런이 74초 한 건이라 0클립이 나갔다.
    득점 장면이 빠지는 건 취향이 아니라 결함이라, 이제 넘겨서 담는다.
    """
    clips = [clip(1, 0, 74, tags="홈런", delta=1)]
    picked, _d, total = select.choose(clips, {**SPEC, "budget": 60}, {})
    assert [c["scene_id"] for c in picked] == [1]
    assert total > 60


def test_must_layer_stops_at_slack_ceiling():
    """초과는 오차범위(+30%)까지 — 그 위로는 요청한 물건이 아니게 된다.

    실측(v201 comp 8): 필수 16건이 900초 예산에 다 안 들어가 12건만 담겼다. 무제한
    허용이 답이 아니라 상한을 두는 게 답 — 상한에 걸리면 득점 작은 뒤쪽부터 떨어진다.
    """
    # 예산 100 → 상한 130. 100s + 25s = 125 까지는 담고, 그 뒤 60s 는 185 라 탈락.
    clips = [clip(1, 0, 100, tags="홈런", delta=3), clip(2, 200, 225, tags="홈런", delta=2),
             clip(3, 400, 460, tags="홈런", delta=1)]
    picked, dropped, total = select.choose(clips, {**SPEC, "budget": 100}, {})
    assert [c["scene_id"] for c in picked] == [1, 2]
    assert total == 125                                      # 예산 초과 · 상한 이내
    assert [c["scene_id"] for c, _why in dropped] == [3]      # 중복 없이 한 번만


def test_rescue_longest_when_nothing_qualifies():
    """필수도 없고 전부 예산 초과면 최단 1건은 담는다 — '장면 없음'으로 끝내지 않는다."""
    clips = [clip(1, 0, 74, tags="범타", delta=0), clip(2, 100, 190, tags="범타", delta=0)]
    picked, _d, _t = select.choose(clips, {**SPEC, "budget": 60}, {})
    assert not picked
    assert [c["scene_id"] for c in select.rescue_longest(clips, 60)] == [1]


def test_backfill_note_explains_no_op():
    """충원 무동작에 사유를 남긴다 — 로그만 보면 시도한 것처럼 읽히던 문제."""
    assert "관점" in select.backfill_note({**SPEC, "view": "홈"})
    assert "대상" in select.backfill_note({**SPEC, "targets": []})
    assert select.backfill_note(SPEC) == ""
