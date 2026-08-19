"""bounds 후보 생성·처분 / select 층 구조 — 순수 계산이라 스텁 없이 검증."""

from flow import bounds, select

SEGS = [
    {"s": 100.0, "e": 106.0, "shot_type": "리액션", "summary": "타자가 타석에 들어선다"},
    {"s": 106.0, "e": 112.0, "shot_type": "투구", "summary": "투수가 공을 던진다"},
    {"s": 112.0, "e": 120.0, "shot_type": "타구·수비", "summary": "외야수가 타구를 쫓는다"},
    {"s": 120.0, "e": 128.0, "shot_type": "리액션", "summary": "관중이 환호한다"},
    {"s": 128.0, "e": 132.0, "shot_type": "광고", "summary": None},
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


def test_start_candidates_can_move_forward():
    """시작을 뒤로도 옮길 수 있다 — 장면 경계가 앞 타석 꼬리를 물 때가 있다.

    실측(v202 장면11 역전 홈런): 장면은 2558s 에 시작하는데 그 홈런의 투구는 2583s.
    앞 25초는 심우준 타석의 견제 장면이라 무슨 일인지 알 수 없는 시작이었다.
    앞쪽만 보던 탓에 정답이 구조적으로 후보에 못 들어왔다.
    """
    c = clip(cs=100.0, ce=160.0)
    got = bounds.start_candidates(c, [{"s": 120.0, "e": 126.0, "shot_type": "투구",
                                       "summary": "투수가 공을 던진다"}], [])
    assert 120 in [int(sec) for sec, _ in got]


def test_start_candidates_keep_clip_long_enough():
    """뒤로 미뤄도 클립이 MIN_CLIP_SEC 밑으로 짧아지면 후보가 아니다."""
    c = clip(cs=100.0, ce=120.0)                  # 20s — 116s 로 미루면 4s 만 남는다
    got = bounds.start_candidates(c, [{"s": 116.0, "e": 118.0, "shot_type": "투구",
                                       "summary": ""}], [])
    assert 116 not in [int(sec) for sec, _ in got]


def test_start_candidates_use_all_board_pitches():
    """보드 검출 투구는 장면 소유분이 아니라 창 안 전량을 본다.

    실측(v202 장면11): 창(2518~2558) 안에 2532·2550 이 있었는데 장면 소유 행의
    2583 만 실려 후보가 0건이었다. 장면 기준 조인이 원인이었다.
    """
    c = clip(cs=2558.0, ce=2633.0)
    got = bounds.start_candidates(c, [], [(2532, 2533), (2550, 2552), (2583, 2584)])
    secs = {int(s) for s, _ in got}
    assert {2532, 2550, 2583} <= secs


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
    start = int(rows[0]["starts"][0]["sec"])
    moved = bounds.apply([c], rows, f"장면 1: 시작 {start} 끝 유지")
    assert c["cut"]["cs"] == float(start) and moved


# ── select ────────────────────────────────────────────

SPEC = {"mode": "collection", "targets": ["안타"], "view": "전체", "budget": 100}


def test_must_layer_survives_zero_score():
    """필수 장면은 verify 0점이어도 필수층에 담긴다 — 사실이 소견을 이긴다.

    0점 일반 클립을 빼는 건 이제 drop0 노드 몫이라 여기서는 검사하지 않는다
    (select 는 넘어온 것만 순서대로 담는다).
    """
    must = clip(1, 0, 30, tags="안타", labels="역전", delta=1)
    picked, _dropped, total = select.choose(
        [must], SPEC, {1: {"score": 0, "complete": True, "reason": ""}})
    assert [c["scene_id"] for c in picked] == [1]
    assert total == 30


def test_llm_order_drives_fill():
    """rank 가 준 순서대로 담는다 — 점수가 낮아도 순위가 앞서면 먼저다."""
    clips = [clip(1, 0, 40, tags="범타", delta=0, inning="1회 초"),
             clip(2, 100, 140, tags="범타", delta=0, inning="2회 초"),
             clip(3, 200, 240, tags="범타", delta=0, inning="3회 초")]
    scores = {1: {"score": 3, "complete": True, "reason": ""},
              2: {"score": 1, "complete": True, "reason": ""},
              3: {"score": 3, "complete": True, "reason": ""}}
    picked, _d, _t = select.choose(clips, {**SPEC, "budget": 80}, scores, order=[2, 3, 1])
    assert [c["scene_id"] for c in picked] == [2, 3]      # 1점짜리 2번이 먼저


def test_order_missing_ids_go_last():
    """순위에 없는 클립은 버리지 않고 뒤로 — 모델이 빠뜨린 것이지 빼라는 뜻이 아니다."""
    clips = [clip(i, i * 100, i * 100 + 40, tags="범타", delta=0) for i in (1, 2, 3)]
    # 40s × 3, 예산 80 → 두 건만 들어간다. 순위에 3만 있으면 3이 먼저 담긴다.
    picked, dropped, _t = select.choose(clips, {**SPEC, "budget": 80}, {}, order=[3])
    assert 3 in [c["scene_id"] for c in picked]
    assert len(picked) == 2 and len(dropped) == 1


def test_parse_order_keeps_valid_only():
    """실존 번호만, 중복 없이, 나온 순서대로."""
    got = select.parse_order("순서: 15, 3, 99, 15, 31", {3, 15, 31})
    assert got == [15, 3, 31]


def test_budget_is_never_exceeded():
    """예산 초과는 담지 않는다 — 절단이 마지막이라 총합이 정확하다."""
    clips = [clip(i, i * 100, i * 100 + 40, tags="안타", delta=0) for i in range(1, 6)]
    picked, _dropped, total = select.choose(clips, SPEC, {})
    assert total <= SPEC["budget"]
    assert sum(c["cut"]["ce"] - c["cut"]["cs"] for c in picked) == total


def test_score_order_is_fallback_without_llm_order():
    """rank 가 실패하면 점수순으로 채운다 — 콜 하나가 편성을 죽이지 않는다."""
    clips = [clip(1, 0, 10, tags="범타", delta=0), clip(2, 20, 30, tags="범타", delta=0)]
    scores = {1: {"score": 1, "complete": True, "reason": ""},
              2: {"score": 3, "complete": True, "reason": ""}}
    picked, _d, _t = select.choose(clips, {**SPEC, "budget": 10}, scores, order=None)
    assert [c["scene_id"] for c in picked] == [2]


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


def test_dropped_has_no_duplicates():
    """한 클립은 탈락 목록에 한 번만 — 층을 지날 때마다 다시 떨어뜨리지 않는다.

    실측(v201 comp9): 탈락 24건이 실은 12건 × 2였다. ②질의층에서 예산에 못 든 클립을
    ④잔여층이 또 보고 같은 사유로 다시 실었다. 리포트가 실제보다 두 배로 부풀었다.
    """
    clips = [clip(1, 0, 60, tags="안타", delta=1),
             *(clip(i, i * 200, i * 200 + 50, tags="안타", delta=0) for i in range(2, 7))]
    _picked, dropped, _t = select.choose(clips, {**SPEC, "budget": 100}, {})
    ids = [c["scene_id"] for c, _why in dropped]
    assert len(ids) == len(set(ids)), f"중복 탈락: {ids}"


def test_end_candidates_keep_speech_over_screen_when_truncating():
    """상한에 걸리면 화면 전환을 버리고 발화 끝을 남긴다.

    실측(v201 장면5): 시간순으로 자르니 앞쪽 화면 전환 넷이 자리를 다 먹고 유일한
    정답인 발화 끝 1365s 가 버려졌다. 규칙은 "해설이 끝나는 지점"인데 고를 수가 없었다.
    """
    segs = [{"s": 120.0 + i, "e": 121.0 + i, "shot_type": "리액션", "summary": ""}
            for i in range(6)]                       # 끝 121·122·123·124·125·126
    utts = [(118.0, 130.0, "결국에 이겨내네요")]      # 발화 끝 130 — 시간순이면 꼴찌
    got = bounds.end_candidates(clip(ce=120.0), segs, utts)
    assert 130 in [int(sec) for sec, _ in got], got
    assert len(got) <= bounds.CAND_MAX


def test_end_candidates_skip_ad_boundary():
    """광고 경계는 끝 후보가 아니다 — 중계가 아닌 데서 끊긴다."""
    segs = [{"s": 120.0, "e": 122.0, "shot_type": "광고", "summary": None},
            {"s": 122.0, "e": 124.0, "shot_type": "리액션", "summary": ""}]
    got = bounds.end_candidates(clip(ce=119.0), segs, [])
    assert 122 not in [int(sec) for sec, _ in got]
    assert 124 in [int(sec) for sec, _ in got]


def test_build_rows_marks_current_start_is_pitch():
    """현재 시작이 이미 투구 샷이면 그렇게 표시한다 — 규칙에 있는데 재료가 없었다."""
    c = clip(cs=106.0, ce=120.0)
    rows = bounds.build_rows([c], SEGS, UTTS, {})
    assert rows[0]["cur"]["at_shot_start"] is True
    assert rows[0]["cur"]["shot_type"] == "투구"


def test_bounds_user_attaches_narrative_per_candidate():
    """후보 밑에 그 시각의 화면·해설이 붙는다 — 따로 주면 모델이 조인해야 한다."""
    from flow.prompts import bounds_user

    rows = bounds.build_rows([clip(cs=115.0, ce=120.0)], SEGS, UTTS, {})
    text = bounds_user(rows)
    assert "화면 [투구] 투수가 공을 던진다" in text
    assert "해설" in text


def test_apply_accepts_unit_suffix():
    """숫자 뒤 단위 표기를 허용한다 — 모델마다 다르게 붙인다.

    Qwen3.8 은 "끝 9303s", Qwen3.6 은 "시작 1342초 끝 1355초" 로 답한다(전환 실측).
    안 받으면 줄 전체가 매칭 실패해 조용히 무시되고 "경계 이동 0건"으로만 남는다.
    """
    for text, want in (("장면 1: 시작 유지 끝 128초", 128.0),
                       ("장면 1: 시작 유지 끝 128s", 128.0),
                       ("장면 1: 시작 유지 끝 128", 128.0)):
        c = clip(cs=115.0, ce=120.0)
        rows = bounds.build_rows([c], SEGS, UTTS, [])
        bounds.apply([c], rows, text)
        assert c["cut"]["ce"] == want, text
