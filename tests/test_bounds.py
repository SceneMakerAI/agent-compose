"""bounds 후보 생성·처분 — 순수 계산이라 스텁 없이 검증."""

from flow import bounds

SEGS = [
    {"s": 100.0, "e": 106.0, "shot_type": "리액션", "summary": "타자가 타석에 들어선다"},
    {"s": 106.0, "e": 112.0, "shot_type": "투구", "summary": "투수가 공을 던진다"},
    {"s": 112.0, "e": 120.0, "shot_type": "타구·수비", "summary": "외야수가 타구를 쫓는다"},
    {"s": 120.0, "e": 128.0, "shot_type": "리액션", "summary": "관중이 환호한다"},
    {"s": 128.0, "e": 132.0, "shot_type": "광고", "summary": None},
]
UTTS = [(118.0, 127.5, "넘어갑니다"), (130.0, 134.0, "다음 타석")]


def clip(scene_id=1, cs=115.0, ce=120.0, tags="홈런", labels="", delta=1, inning="1회 초",
         pitches=(), s=None, e=None):
    """cs·ce 는 컷 좌표, s·e 는 **장면 구간**(t_scene_baseball.start_time~end_time).

    후보는 장면 구간 안에서만 나므로 둘을 구분해야 한다 — 기본값은 컷을 넉넉히
    감싸는 창이라 구간 제한이 걸리지 않는다. 제한 자체를 보는 테스트만 s·e 를 준다.
    pitches = 그 장면 자신의 검출 투구 (t_scene_baseball.pitch_idxs 파싱 결과).
    """
    return {"scene_id": scene_id, "tags": tags.split(","),
            "label_list": labels.split(",") if labels else [],
            "score_delta": delta, "inning": inning,
            "s": cs - 60 if s is None else s, "e": ce + 60 if e is None else e,
            "pitch_sec": None, "pitches": tuple(pitches),
            "cut": {"cs": cs, "ce": ce, "mode": "full", "shots": []}}


def test_start_candidates_come_from_scene_pitch_idxs_only():
    """시작 후보는 **그 장면 자신의 검출 투구**뿐 — 샷 분류는 보지 않는다 (2026-08-24).

    구 코드는 분류된 '투구'·'타구·수비' 샷도 후보로 냈는데, 6경기 실측에서 프롬프트에
    실린 적이 0건이다(같은 정수 초의 보드 후보에 매번 흡수). 재료를 장면 자신의
    pitch_idxs 하나로 좁혀 다른 타석 투구가 새는 통로를 막는다.
    """
    c = clip(cs=100.0, ce=160.0, pitches=[(126, 128)])
    assert bounds.start_candidates(c) == [(126.0, "보드 검출 투구")]
    # 샷 목록에 '투구'·'타구·수비' 가 있어도 후보가 되지 않는다
    assert bounds.start_candidates(clip(cs=100.0, ce=160.0)) == []


def test_start_candidates_go_backward_for_board_pitch():
    """현재 시작이 이미 플레이 도중이면 **앞쪽** 투구도 후보로 낸다.

    구 방침("앞으로만")은 "장면 시작은 그 플레이의 투구보다 이르거나 같다"를 전제했는데
    v203 이 반증했다: 장면5(홈런)는 장면 657s / 투구 646~650s 로, 클립이 "담장
    넘어갑니다"부터 시작해 스윙이 없었다.
    """
    c = clip(cs=657.0, ce=683.0, pitches=[(646, 650), (626, 628)])   # v203 장면5 (홈런)
    assert {int(x) for x, _ in bounds.start_candidates(c)} == {646, 626}


def test_start_candidates_bounded_by_scene_window_only():
    """거리 상한은 없고, 경계는 **장면 구간**뿐이다 (2026-08-24).

    구 상한(뒤 15s / 앞 120s)의 근거는 "넓히면 앞 타석 투구가 들어온다"였고, 후보가
    그 장면 자신의 것뿐인 지금은 성립하지 않는다 — 구간 안에서는 아무리 멀어도 후보다.
    대신 장면(start_time~end_time) 밖은 내지 않는다.

    실측 주의: 발행본은 `start_time == pitch_time`(489행 중 486행 일치)이라 그 장면의
    **앞선 투구는 대부분 장면 밖**이다 — v1004 장면43 은 pitch_idxs 가
    `4811-4819,4827-4828` 인데 장면 시작이 4827 이라 4811 이 구간 밖으로 떨어진다.
    """
    c = clip(cs=4827.0, ce=4900.0, s=4800.0, e=4900.0, pitches=[(4811, 4819)])
    assert [int(x) for x, _ in bounds.start_candidates(c)] == [4811]   # 16초 앞이어도 후보

    c = clip(cs=4827.0, ce=4900.0, s=4827.0, e=4900.0, pitches=[(4811, 4819)])
    assert bounds.start_candidates(c) == []                            # 장면 밖은 안 낸다


def test_start_candidates_keep_clip_long_enough():
    """뒤로 미뤄도 클립이 MIN_CLIP_SEC 밑으로 짧아지면 후보가 아니다."""
    c = clip(cs=100.0, ce=120.0, pitches=[(116, 118)])   # 116s 로 미루면 4s 만 남는다
    assert bounds.start_candidates(c) == []


def test_start_candidates_drop_same_point_and_cap():
    """현재 시작과 MERGE_GAP_SEC 이내면 대안이 아니고, 후보는 가까운 순 CAND_MAX 개."""
    c = clip(cs=100.0, ce=300.0, pitches=[(98, 99), (120, 122), (140, 142),
                                          (160, 162), (180, 182), (200, 202)])
    got = [int(x) for x, _ in bounds.start_candidates(c)]
    assert 98 not in got                                 # 2초 차 = 같은 지점
    assert got == [120, 140, 160, 180][:bounds.CAND_MAX]


def test_end_candidates_stop_at_scene_end():
    """끝 후보는 **장면 끝(end_time)** 을 넘지 않는다 (2026-08-24).

    구 상한은 현재 끝 + 12초라 발행이 정한 장면 밖으로 넘어갈 수 있었다.
    """
    c = clip(cs=100.0, ce=119.0, s=100.0, e=121.0)
    got = [int(sec) for sec, _ in bounds.end_candidates(c, SEGS, UTTS)]
    assert 120 in got                                   # 장면 안 샷 끝
    assert 128 not in got and 132 not in got            # 장면 밖 샷 끝


def test_end_candidates_mark_coincidence():
    """샷 끝에 발화 끝이 겹치면 표시한다 — 가장 깔끔한 끝점을 모델이 알게.

    컷 좌표는 샷 경계이고 발화 끝은 그 자리를 고를 **근거**다 (2026-08-24 재설계).
    """
    got = bounds.end_candidates(clip(ce=120.0), SEGS, UTTS)
    assert any("화면 전환 + 발화 끝" in why for _sec, why in got)


def test_apply_rejects_values_outside_candidates():
    """제시하지 않은 초는 기각 — 모델이 시각을 지어낼 수 없다."""
    c = clip(cs=115.0, ce=160.0, pitches=[(100, 102)])
    rows = bounds.start_rows([c], SEGS, UTTS)
    bounds.apply_start([c], rows, "999s")
    assert (c["cut"]["cs"], c["cut"]["ce"]) == (115.0, 160.0)


def test_apply_moves_when_candidate_matches():
    """시작도 **후보 번호**로 받는다 (2026-08-24 형식 통일 — 끝과 같은 규칙)."""
    c = clip(cs=100.0, ce=160.0, pitches=[(120, 122)])
    rows = bounds.start_rows([c], SEGS, UTTS)
    want = rows[0]["cands"][0]["sec"]
    moved = bounds.apply_start([c], rows, "1")
    assert c["cut"]["cs"] == want and moved


def test_start_rows_gate_by_anchor_shot_type():
    """게이트는 앵커 **유무**가 아니라 앵커 샷의 **유형**이다 (2026-08-24).

    구 조건(앵커만 있으면 건너뜀)은 "앵커가 있으면 시작이 곧 투구"라는 구 상류 실측에
    기댔다. 지금은 상류가 대표 투구를 골라 주고 cut 이 그 초가 든 샷을 쓰는데 그 샷의
    61%가 투구 화면이 아니다(406장면). 조건이 남아 이 노드가 대상 0건으로 놀았다.

    '투구'·'리액션'(타석 준비)은 믿고, 이미 플레이 도중이거나 중계가 아닌 화면만 묻는다.
    """
    def anchored(sid, shot_type):
        c = clip(scene_id=sid, cs=115.0, ce=160.0, pitches=[(104, 106)])
        c["cut"]["anchor"] = (106.0, 112.0)
        c["cut"]["anchor_type"] = shot_type
        return c

    clips = [anchored(1, "투구"), anchored(2, "리액션"), anchored(3, "타구·수비"),
             anchored(4, "광고"), anchored(5, None),
             clip(scene_id=6, cs=115.0, ce=160.0,        # anchor_type 키 없음 = 앵커 없음
                  pitches=[(104, 106)])]
    rows = bounds.start_rows(clips, SEGS, UTTS)
    assert [r["scene_id"] for r in rows] == [3, 4, 5, 6]


def test_apply_keeps_candidate_fraction():
    """수용한 후보는 **원래 소수 초**로 들어간다 — 정수로 내리면 샷이 바뀐다.

    v201 은 t_segment 경계가 소수라 4369.3s('투구' 시작)와 4369s(직전 '리액션'
    끝자락)가 서로 다른 샷이다. 프롬프트는 정수로 보여주되 적용은 실측값으로 한다.
    """
    segs = [{"s": 100.0, "e": 108.3, "shot_type": "리액션", "summary": None},
            {"s": 108.3, "e": 120.0, "shot_type": "투구", "summary": "투수가 던진다"}]
    c = clip(cs=120.0, ce=160.0, pitches=[(108, 109)])
    rows = bounds.start_rows([c], segs, UTTS)
    assert rows[0]["cands"][0]["sec"] == 108.3       # 후보는 소수를 그대로 들고 있다
    bounds.apply_start([c], rows, "1")               # 번호 → 소수 초로 치환
    assert c["cut"]["cs"] == 108.3


def test_end_candidates_give_every_shot_end_no_cap():
    """창 안의 샷 끝은 **전부** 낸다 — 개수 상한도 근접 접기도 없다 (2026-08-24).

    구 설계는 종류 우선으로 CAND_MAX(4)까지 자르고 5초 이내를 접어 클립당 최종 후보가
    평균 1.4개로 줄었다 (comp37 실측: 후보 1개인 클립이 8/10 — 이지선다). 상한이
    있던 이유(후보 우열이 약하면 thinking 폭주)는 이 단계의 thinking 을 끄면서 사라졌다.
    """
    segs = [{"s": 120.0 + i, "e": 121.0 + i, "shot_type": "리액션", "summary": f"샷{i}"}
            for i in range(6)]                       # 끝 121·122·123·124·125·126
    utts = [(118.0, 130.0, "결국에 이겨내네요")]      # 발화 끝 130 — 샷과 안 겹친다
    got = bounds.end_candidates(clip(ce=120.0), segs, utts)
    assert [int(sec) for sec, _ in got] == [121, 122, 123, 124, 125, 126, 130]
    assert len(got) > bounds.CAND_MAX                # 상한에 안 걸린다


def test_end_candidates_keep_speech_end_when_no_shot_boundary():
    """창 안에 샷 경계가 없으면 발화 끝이 후보다 — 커버리지를 잃지 않는다.

    실측(comp38 장면48): 12초 창 안 샷 끝이 0개였다. 샷만 내면 이 클립은 후보가 없어
    끝 보정을 통째로 건너뛴다.
    """
    got = bounds.end_candidates(clip(ce=120.0), [], [(118.0, 128.0, "이겨내네요")])
    assert [(int(x), w) for x, w in got] == [(128, "발화 끝")]


def test_end_candidates_skip_ad_boundary():
    """광고 경계는 끝 후보가 아니다 — 중계가 아닌 데서 끊긴다."""
    segs = [{"s": 120.0, "e": 122.0, "shot_type": "광고", "summary": None},
            {"s": 122.0, "e": 124.0, "shot_type": "리액션", "summary": ""}]
    got = bounds.end_candidates(clip(ce=119.0), segs, [])
    assert 122 not in [int(sec) for sec, _ in got]
    assert 124 in [int(sec) for sec, _ in got]


def test_start_rows_carries_current_shot():
    """현재 시작의 화면·해설이 실린다 — 모델이 "이미 투구 지점인가"를 볼 재료다."""
    c = clip(cs=106.0, ce=120.0, pitches=[(100, 101)])
    rows = bounds.start_rows([c], SEGS, UTTS)
    assert rows[0]["cur"]["shot_type"] == "투구"
    assert rows[0]["cur"]["shot"] == "투수가 공을 던진다"


def test_start_cand_snaps_to_next_captioned_shot():
    """후보 초가 캡션 없는 샷의 마지막 0.x초에 걸리면 **직후 샷**의 화면을 붙인다.

    보드 검출 투구는 정수 초라 그 투구 샷 시작(소수)보다 이르다 — 4경기 실측에서
    화면 없는 후보 4건이 전부 이 모양이었고(간격 0.50~0.70s) 정작 그 자리의 캡션은
    바로 뒤에 있었다. 구간 화면 블록이 가려 주던 결함이라, 블록을 빼면 드러난다.
    """
    segs = [{"s": 100.0, "e": 106.6, "shot_type": None, "summary": None},
            {"s": 106.6, "e": 114.0, "shot_type": "투구", "summary": "투수가 공을 던진다"}]
    c = clip(cs=120.0, ce=160.0, pitches=[(106, 107)])
    rows = bounds.start_rows([c], segs, UTTS)
    cand = rows[0]["cands"][0]
    assert cand["shot"] == "투수가 공을 던진다"        # 화면은 0.6초 뒤 샷에서 가져온다
    assert cand["shot_type"] == "투구"
    # 컷 좌표도 그 샷 시작으로 — 보여준 화면과 실제 자르는 자리가 어긋나면 안 된다.
    assert cand["sec"] == 106.6
    assert bounds.apply_start([c], rows, "1") == ["장면1 시작 120.0→106.6"]


def test_start_cand_shot_snap_does_not_reach_next_scene():
    """스냅은 CAND_SHOT_SNAP_SEC 안쪽만 — 멀리 있는 샷을 끌어오지 않는다."""
    segs = [{"s": 100.0, "e": 110.0, "shot_type": None, "summary": None},
            {"s": 110.0, "e": 118.0, "shot_type": "투구", "summary": "투수가 공을 던진다"}]
    c = clip(cs=118.0, ce=160.0, pitches=[(106, 107)])
    rows = bounds.start_rows([c], segs, UTTS)
    cand = rows[0]["cands"][0]
    assert cand["sec"] == 106.0 and cand["shot"] == ""   # 4초 밖이라 그대로 빈다


def test_start_user_matches_end_prompt_shape():
    """시작 프롬프트 뼈대가 끝과 같다 — 머리줄·현재·번호 붙은 후보·[질문] 한 줄.

    두 노드가 하는 일이 "제시된 지점 중 고르기"로 같은데 형식이 달라 규칙과 파서를
    따로 들고 있었다 (2026-08-24 통일).

    구간의 대사·화면 블록은 **없다**: 후보 줄이 자기완결적이라(각자 화면·해설을 달고
    있다) 시각을 맞춰 조인할 재료를 따로 줄 이유가 없다 — 근거는 bounds.start_rows.
    """
    from flow.prompts import start_user

    utts = [(104.0, 112.0, "던집니다"), *UTTS]
    rows = bounds.start_rows([clip(cs=100.0, ce=160.0, pitches=[(112, 113)])], SEGS, utts)
    text = start_user(rows)
    assert text.startswith("■ 장면 1 [홈런]")
    assert "현재 00:01:40~00:02:40" in text              # hh:mm:ss 표기
    assert "  현재 시작 00:01:40" in text
    assert "  1) 시작후보 " in text
    assert "[구간 대사]" not in text and "[구간 화면]" not in text
    assert text.rstrip().endswith("[질문] 시작을 어디로 할까?")


def test_apply_start_keeps_when_model_says_stay():
    """'유지' 면 아무것도 옮기지 않는다."""
    from flow import bounds as b

    c = clip(cs=115.0, ce=160.0, pitches=[(106, 107)])
    rows = b.start_rows([c], SEGS, UTTS)
    assert b.apply_start([c], rows, "유지") == []
    assert c["cut"]["cs"] == 115.0


def test_apply_end_maps_index_to_exact_candidate_second():
    """끝 응답은 **후보 번호**이고, 적용은 그 후보의 소수 초로 한다 (2026-08-24).

    초를 받아쓰게 하면 소수가 사라진다 — v201 은 t_segment 경계가 소수라 4369.3s
    ('투구' 시작)와 4369s(직전 '리액션' 끝자락)가 서로 다른 샷이다. 구 형식
    (`장면 1: 끝 128초`)은 폐기했다.
    """
    for text in ("1", "1번", "번호 1"):
        c = clip(cs=115.0, ce=120.0)
        rows = bounds.end_rows([c], SEGS, UTTS)
        want = rows[0]["ends"][0]["sec"]
        bounds.apply_end([c], rows, text)
        assert c["cut"]["ce"] == want, text


# ── 중복 후보 병합 ─────────────────────────────────────

# 90~130 은 현재 시작이 놓일 자리(다른 샷·다른 발화), 130~200 은 한 투구 샷 + 한 발화
# → 그 안의 후보들은 서로 구별되지 않는다. 후보는 전부 현재 시작보다 뒤에 둔다.
_SEGS2 = [{"s": 90.0, "e": 130.0, "shot_type": "리액션", "summary": "타자가 걸어간다"},
          {"s": 130.0, "e": 200.0, "shot_type": "투구", "summary": "투수가 공을 던진다"}]
_UTTS2 = [(90.0, 130.0, "다른 문장이다"), (130.0, 200.0, "같은 문장이 계속된다")]


def test_end_rows_drop_speech_candidate_identical_to_current():
    """현재와 화면·해설이 똑같은 **발화 끝** 후보는 뺀다 — 정말 아무것도 아니다.

    샷이 창(+12s) 밖까지 이어져 샷 경계 후보가 없는 상황. 남는 건 발화 끝뿐인데
    그마저 현재와 같으면 물어볼 게 없어 행 자체가 나오지 않는다.
    """
    segs = [{"s": 120.0, "e": 200.0, "shot_type": "리액션", "summary": "같은 화면"}]
    utts = [(120.0, 136.0, "같은 해설")]
    assert bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts) == []


def test_end_rows_drop_shot_boundary_identical_to_current():
    """샷 경계 후보도 설명이 현재와 똑같으면 뺀다 — 종류를 안 가린다 (2026-08-24).

    현재 끝(130s)이 샷 한복판이라 그 샷의 끝(133s)은 같은 샷이라서 화면·해설이 같다.
    한때는 "샷을 끝까지 보고 자른다"가 다른 선택이라 남겼는데, 실측이 그 걱정을
    지웠다: 4회 실행 78콜에서 그런 후보는 4건뿐이고 전부 다른 후보와 함께 나왔다.
    남겨 봐야 고를 근거가 프롬프트 안에 없는 선택지만 하나 느는 것이다.
    """
    segs = [{"s": 120.0, "e": 133.0, "shot_type": "리액션", "summary": "같은 화면"},
            {"s": 133.0, "e": 136.0, "shot_type": "타구·수비", "summary": "다른 화면"}]
    utts = [(120.0, 133.5, "같은 해설"), (133.5, 136.0, "다른 해설")]
    ends = bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts)[0]["ends"]
    assert [int(e["sec"]) for e in ends] == [136]        # 133 은 현재와 구별 안 돼 빠짐


def test_end_rows_skip_clip_when_nothing_left_to_ask():
    """후보가 전부 빠지면 그 클립은 행에서 빠진다 — 물어볼 게 없으면 안 묻는다."""
    segs = [{"s": 120.0, "e": 133.0, "shot_type": "리액션", "summary": "같은 화면"},
            {"s": 133.0, "e": 136.0, "shot_type": "리액션", "summary": "같은 화면"}]
    utts = [(120.0, 136.0, "같은 해설")]
    assert bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts) == []


# ── 처분 파서 — 응답 형식 (끝=번호 / 시작=초) ───────────

def _clip(sid=47, cs=6696.0, ce=6761.0):
    return {"scene_id": sid, "cut": {"cs": cs, "ce": ce, "mode": "레시피"}}


ROW = {"scene_id": 47, "ends": [{"sec": 6769.4}, {"sec": 6781.0}],
       "cands": [{"sec": 6690.3}]}


def test_apply_end_takes_candidate_index():
    """번호 1·2 가 각각 첫째·둘째 후보의 실제 초로 치환된다."""
    for text, want in (("1", 6769.4), ("2", 6781.0)):
        clips = [_clip()]
        assert bounds.apply_end(clips, [ROW], text) == [f"장면47 끝 6761.0→{want}"]
        assert clips[0]["cut"]["ce"] == want
        assert clips[0]["cut"]["mode"].endswith("+끝보정")


def test_apply_end_ignores_leading_prose():
    """형식대로면 답은 맨 끝 숫자다 — 앞에 군말이 붙어도 답을 잃지 않는다."""
    clips = [_clip()]
    bounds.apply_end(clips, [ROW], "2번이 좋겠다.\n2")
    assert clips[0]["cut"]["ce"] == 6781.0


def test_apply_end_rejects_hold_unknown_and_offlist_index():
    """유지·해석불가·목록 밖 번호는 **원 경계 유지**.

    번호로 받으면 지어낸 값이 목록 밖으로 떨어져 시각을 지어내는 것보다 안전하다.
    """
    for text in ("유지", "잘 모르겠다", "3", "0", "9999"):
        clips = [_clip()]
        assert bounds.apply_end(clips, [ROW], text) == []
        assert clips[0]["cut"]["ce"] == 6761.0
        assert clips[0]["cut"]["mode"] == "레시피"


def test_apply_start_takes_candidate_index_like_end():
    """시작도 번호로 받는다 — 끝과 같은 파서·같은 기각 규칙."""
    clips = [_clip()]
    assert bounds.apply_start(clips, [ROW], "1") == ["장면47 시작 6696.0→6690.3"]
    assert clips[0]["cut"]["cs"] == 6690.3
    for text in ("유지", "2", "0", "6690s"):          # 목록 밖 번호·초 표기는 기각
        clips = [_clip()]
        assert bounds.apply_start(clips, [ROW], text) == []
        assert clips[0]["cut"]["cs"] == 6696.0
