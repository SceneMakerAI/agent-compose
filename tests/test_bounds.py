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


def clip(scene_id=1, cs=115.0, ce=120.0, tags="홈런", labels="", delta=1, inning="1회 초"):
    return {"scene_id": scene_id, "tags": tags.split(","),
            "label_list": labels.split(",") if labels else [],
            "score_delta": delta, "inning": inning, "s": cs, "e": ce,
            "pitch_sec": None, "cut": {"cs": cs, "ce": ce, "mode": "full", "shots": []}}


def test_start_candidates_use_board_pitch_when_shot_missing():
    """샷 분류가 없어도 보드 검출 투구가 시작 후보가 된다.

    장면과 겹치는 샷만 분류돼 shot_type 이 NULL 인 구간이 많다 — 보드 검출이 유일한
    단서일 때가 있다. 역전 홈런이 '주자가 뛰는 장면'부터 시작하던 원인.
    """
    c = clip(cs=100.0, ce=160.0)
    got = bounds.start_candidates(c, [{"s": 100.0, "e": 130.0, "shot_type": None}],
                                  [(126, 128)])
    assert (126.0, "보드 검출 투구") in got


def test_start_candidates_prefer_pitch_over_batted():
    """투구 후보가 있으면 타구·수비는 넣지 않는다 (원칙은 투구부터)."""
    got = bounds.start_candidates(clip(cs=100.0, ce=160.0), SEGS, [])
    assert any("투구" in why for _sec, why in got)
    assert not any("타구" in why for _sec, why in got)


def test_start_candidates_go_backward_for_board_pitch():
    """현재 시작이 이미 플레이 도중이면 **앞쪽** 투구도 후보로 낸다.

    구 방침("앞으로만")은 "장면 시작은 그 플레이의 투구보다 이르거나 같다"를 전제했는데
    v203 이 반증했다: 장면5(홈런)는 장면 657s / 투구 646~650s 로, 클립이 "담장
    넘어갑니다"부터 시작해 스윙이 없었다. 앞쪽 후보는 분류가 NULL 인 구간이라
    **보드 검출 투구**만 쓴다.
    """
    c = clip(cs=657.0, ce=683.0)                   # v203 장면5 (홈런)
    got = bounds.start_candidates(c, [], [(646, 650), (626, 628)])
    secs = {int(x) for x, _ in got}
    assert 646 in secs                             # 그 홈런의 투구 — 11초 앞
    assert 626 not in secs                         # 31초 앞 = 앞 타석 (상한 밖)


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


def test_end_candidates_mark_coincidence():
    """발화 끝과 화면 전환이 겹치면 표시한다 — 가장 깔끔한 끝점을 모델이 알게."""
    got = bounds.end_candidates(clip(ce=120.0), SEGS, UTTS)
    assert any("발화 끝 + 화면 전환" in why for _sec, why in got)


def test_apply_rejects_values_outside_candidates():
    """제시하지 않은 초는 기각 — 모델이 시각을 지어낼 수 없다."""
    c = clip(cs=115.0, ce=120.0)
    rows = bounds.start_rows([c], SEGS, UTTS, [])
    bounds.apply_start([c], rows, "999s")
    assert (c["cut"]["cs"], c["cut"]["ce"]) == (115.0, 120.0)


def test_apply_moves_when_candidate_matches():
    c = clip(cs=100.0, ce=160.0)
    rows = bounds.start_rows([c], SEGS, UTTS, [])
    start = int(rows[0]["cands"][0]["sec"])
    moved = bounds.apply_start([c], rows, f"{start}s")
    assert c["cut"]["cs"] == float(start) and moved


def test_start_rows_skips_clips_with_pitch_anchor():
    """cut 이 투구 앵커를 잡은 클립은 아예 묻지 않는다.

    앵커가 있으면 cut 의 시작이 이미 그 플레이의 투구다. LLM 에 물으면 되돌리기만
    했다 — v201 comp20 장면42 는 앵커 7510.3s 를 7473.0s('타구·수비' = 앞 타석)로
    37초 물렀다.
    """
    anchored = clip(scene_id=1, cs=115.0, ce=120.0)
    anchored["cut"]["anchor"] = (106.0, 112.0)
    free = clip(scene_id=2, cs=115.0, ce=120.0)          # anchor 키 없음 = 앵커 없음
    rows = bounds.start_rows([anchored, free], SEGS, UTTS, [(104, 106)])
    assert [r["scene_id"] for r in rows] == [2]


def test_apply_keeps_candidate_fraction():
    """수용한 후보는 **원래 소수 초**로 들어간다 — 정수로 내리면 샷이 바뀐다.

    v201 은 t_segment 경계가 소수라 4369.3s('투구' 시작)와 4369s(직전 '리액션'
    끝자락)가 서로 다른 샷이다. 프롬프트는 정수로 보여주되 적용은 실측값으로 한다.
    """
    segs = [{"s": 100.0, "e": 108.3, "shot_type": "리액션", "summary": "타석"},
            {"s": 108.3, "e": 120.0, "shot_type": "투구", "summary": "투수가 던진다"}]
    c = clip(cs=100.0, ce=140.0)
    rows = bounds.start_rows([c], segs, UTTS, [])
    assert rows[0]["cands"][0]["sec"] == 108.3       # 후보는 소수를 그대로 들고 있다
    bounds.apply_start([c], rows, "108s")
    assert c["cut"]["cs"] == 108.3


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


def test_start_rows_carries_current_shot():
    """현재 시작의 화면·해설이 실린다 — 모델이 "이미 투구 지점인가"를 볼 재료다."""
    c = clip(cs=106.0, ce=120.0)
    rows = bounds.start_rows([c], SEGS, UTTS, [(100, 101)])
    assert rows[0]["cur"]["shot_type"] == "투구"
    assert rows[0]["cur"]["shot"] == "투수가 공을 던진다"


def test_start_user_lists_window_and_choices():
    """구간의 대사·화면을 통째로 주고, 후보는 질문 줄에 초 값으로 나열한다."""
    from flow.prompts import start_user

    utts = [(104.0, 112.0, "던집니다"), *UTTS]
    rows = bounds.start_rows([clip(cs=100.0, ce=160.0)], SEGS, utts, [(112, 113)])
    text = start_user(rows)
    assert "[현재 장면] 100s" in text
    assert "[대사]" in text and "[화면]" in text
    assert "[질문] 현재(100s)의 투구 플레이가 시작되는 올바른 시각을 고르세요:" in text
    assert "112s / 유지" in text


def test_apply_start_keeps_when_model_says_stay():
    """'유지' 면 아무것도 옮기지 않는다."""
    from flow import bounds as b

    c = clip(cs=115.0, ce=160.0)
    rows = b.start_rows([c], SEGS, UTTS, [(106, 107)])
    assert b.apply_start([c], rows, "유지") == []
    assert c["cut"]["cs"] == 115.0


def test_apply_accepts_unit_suffix():
    """숫자 뒤 단위 표기를 허용한다 — 모델마다 다르게 붙인다.

    Qwen3.8 은 "끝 9303s", Qwen3.6 은 "시작 1342초 끝 1355초" 로 답한다(전환 실측).
    안 받으면 줄 전체가 매칭 실패해 조용히 무시되고 "경계 이동 0건"으로만 남는다.
    """
    for text, want in (("장면 1: 끝 128초", 128.0),
                       ("장면 1: 끝 128s", 128.0),
                       ("장면 1: 끝 128", 128.0)):
        c = clip(cs=115.0, ce=120.0)
        rows = bounds.end_rows([c], SEGS, UTTS)
        bounds.apply_end([c], rows, text)
        assert c["cut"]["ce"] == want, text


# ── 중복 후보 병합 ─────────────────────────────────────

# 90~130 은 현재 시작이 놓일 자리(다른 샷·다른 발화), 130~200 은 한 투구 샷 + 한 발화
# → 그 안의 후보들은 서로 구별되지 않는다. 후보는 전부 현재 시작보다 뒤에 둔다.
_SEGS2 = [{"s": 90.0, "e": 130.0, "shot_type": "리액션", "summary": "타자가 걸어간다"},
          {"s": 130.0, "e": 200.0, "shot_type": "투구", "summary": "투수가 공을 던진다"}]
_UTTS2 = [(90.0, 130.0, "다른 문장이다"), (130.0, 200.0, "같은 문장이 계속된다")]


def test_dedup_keeps_speech_end_over_screen():
    """끝 후보가 겹치면 '발화 끝' 쪽을 남긴다 — 규칙이 고르라는 지점이다."""
    segs = [{"s": 120.0, "e": 133.0, "shot_type": "리액션", "summary": "같은 화면"},
            {"s": 133.0, "e": 136.0, "shot_type": "리액션", "summary": "같은 화면"}]
    utts = [(120.0, 136.0, "같은 해설")]
    got = bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts)
    whys = [e["why"] for e in got[0]["ends"]]
    assert any("발화" in w for w in whys), whys
