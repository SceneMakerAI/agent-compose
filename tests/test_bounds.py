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
    """샷 끝에 발화 끝이 겹치면 표시한다 — 가장 깔끔한 끝점을 모델이 알게.

    컷 좌표는 샷 경계이고 발화 끝은 그 자리를 고를 **근거**다 (2026-08-24 재설계).
    """
    got = bounds.end_candidates(clip(ce=120.0), SEGS, UTTS)
    assert any("화면 전환 + 발화 끝" in why for _sec, why in got)


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


def test_start_rows_gate_by_anchor_shot_type():
    """게이트는 앵커 **유무**가 아니라 앵커 샷의 **유형**이다 (2026-08-24).

    구 조건(앵커만 있으면 건너뜀)은 "앵커가 있으면 시작이 곧 투구"라는 구 상류 실측에
    기댔다. 지금은 상류가 대표 투구를 골라 주고 cut 이 그 초가 든 샷을 쓰는데 그 샷의
    61%가 투구 화면이 아니다(406장면). 조건이 남아 이 노드가 대상 0건으로 놀았다.

    '투구'·'리액션'(타석 준비)은 믿고, 이미 플레이 도중이거나 중계가 아닌 화면만 묻는다.
    """
    def anchored(sid, shot_type):
        c = clip(scene_id=sid, cs=115.0, ce=160.0)
        c["cut"]["anchor"] = (106.0, 112.0)
        c["cut"]["anchor_type"] = shot_type
        return c

    clips = [anchored(1, "투구"), anchored(2, "리액션"), anchored(3, "타구·수비"),
             anchored(4, "광고"), anchored(5, None),
             clip(scene_id=6, cs=115.0, ce=160.0)]        # anchor_type 키 없음 = 앵커 없음
    rows = bounds.start_rows(clips, SEGS, UTTS, [(104, 106)])
    assert [r["scene_id"] for r in rows] == [3, 4, 5, 6]


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


def test_end_rows_keep_shot_boundary_even_if_description_matches():
    """샷 경계 후보는 설명이 현재와 같아도 남는다 — 샷 스냅이 곧 그 선택이다.

    현재 끝(130s)이 샷 한복판이라 그 샷의 끝(133s)은 같은 샷이라서 화면·해설이 당연히
    같다. 그런데도 "샷을 끝까지 보고 자른다"는 다른 선택이고, 후보 줄의 '화면 전환'
    표기가 그 차이를 보여 준다. 서명으로 지우면 이 설계의 핵심이 사라진다.
    """
    segs = [{"s": 120.0, "e": 133.0, "shot_type": "리액션", "summary": "같은 화면"},
            {"s": 133.0, "e": 136.0, "shot_type": "타구·수비", "summary": "다른 화면"}]
    utts = [(120.0, 136.0, "같은 해설")]
    ends = bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts)[0]["ends"]
    assert [int(e["sec"]) for e in ends] == [133, 136]
    assert ends[0]["shot"] == "같은 화면" and ends[0]["why"] == "화면 전환"


def test_end_rows_fold_candidates_sharing_a_signature():
    """후보끼리 화면·해설이 같으면 하나로 접는다 — 고를 근거가 프롬프트 안에 없다."""
    segs = [{"s": 120.0, "e": 133.0, "shot_type": "리액션", "summary": "같은 화면"},
            {"s": 133.0, "e": 136.0, "shot_type": "리액션", "summary": "같은 화면"}]
    utts = [(120.0, 136.0, "같은 해설")]
    got = bounds.end_rows([clip(cs=100.0, ce=130.0)], segs, utts)
    assert [int(e["sec"]) for e in got[0]["ends"]] == [133]


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


def test_apply_start_still_takes_seconds():
    """시작은 아직 초 값으로 받는다 — 끝만 번호로 바꿨다 (대상 0건이라 미검증).

    단위 표기는 붙든 안 붙든 받는다 (Qwen3.8 "9303s" / Qwen3.6 "1355초" 실측).
    """
    for text in ("6690s", "6690초", "6690"):
        clips = [_clip()]
        assert bounds.apply_start(clips, [ROW], text) == ["장면47 시작 6696.0→6690.3"]
        assert clips[0]["cut"]["cs"] == 6690.3
