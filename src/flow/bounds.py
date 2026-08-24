"""refine_end_bound · refine_start_bound 의 재료 — 후보 생성과 처분 (순수 계산, LLM 무관).

**끝과 시작을 다른 노드로 나눈다** (2026-08-20 재설계). 대상이 다르기 때문이다:
- 끝은 **모든 클립**이 묻는다. "결과를 설명하는 해설이 어디서 끝나는가"는 코드가 답할
  수 없고, cut 의 레시피 체인은 샷 오분류 하나로 결과 전에 끊긴다.
- 시작은 **투구 앵커가 없는 클립만** 묻는다. 앵커가 잡히면 cut 이 정한 시작이 이미
  그 플레이의 투구라 모델이 손대면 앞 타석으로 되돌아갔다 (v201 comp20: 장면9 -33.0s,
  장면42 -37.3s).

후보는 여기서 **결정적으로** 만들고 LLM 은 고르기만 한다 — 검증기(apply_*)가 제시 목록
밖의 값을 기각하므로 모델이 시각을 지어낼 수 없다.

**시작 후보는 그 장면 자신의 검출 투구뿐이다** (2026-08-24 — t_scene_baseball.pitch_idxs).
출처를 하나로 좁혔다:
- 구 ①은 t_play_baseball 을 **경기 전량** 훑어 다른 타석의 투구까지 후보로 냈다.
  실측 6경기 36클립 중 5건(v200 장면15·v202 장면4·36·42·v203 장면58)이 그렇게 만든
  후보였다 — 그 장면과 무관한 투구다.
- 구 ②③(분류된 '투구'·'타구·수비' 샷)은 뒤쪽 창에서만 냈는데, 6경기에서 프롬프트에
  실린 적이 **0건**이다: 같은 정수 초의 보드 후보에 매번 흡수됐다(uniq).

**cs 앞뒤 모두 낸다.** "장면 시작은 그 플레이의 투구보다 이르거나 같다"는 전제는 v203 이
반증했다: 장면5(홈런)는 장면이 657s 에 시작하는데 그 홈런의 투구는 646~650s 다. 클립이
"담장 넘어갑니다"부터 시작해 스윙이 없다. 장면37 도 같다(장면 8321s / 투구 8314~8318s).
원장의 시각이 투구가 아니라 관측 시점(타구·득점 판정)에 찍히면 이렇게 뒤집힌다.

**앞뒤 폭 제한은 없앴다.** 구 상한(뒤 15s / 앞 120s)의 근거는 "넓히면 앞 타석의 투구까지
들어온다(타석 간격 20~40초)" 였는데, 후보가 그 장면 자신의 것뿐이면 앞 타석이 섞일
자리가 없다. 상한을 두면 같은 타석 투구가 버려진다 — 실측 37건(v1004 장면43 은 16초
앞이라 1초 차로 탈락해 후보 0건, 그래서 물어보지도 못했다). 남은 제약은 클립을 잘라
먹지 않는 것(MIN_CLIP_SEC)과 현재 시작과 같은 지점 병합(MERGE_GAP_SEC)뿐이다.

**끝 창은 [그 플레이의 투구, 장면 끝 + END_OUT_EXT_SEC] 이다** (2026-08-24 재조정).
두 끝을 다 옮겼다.

*위쪽* — 상한이 장면 끝이던 동안 이 노드는 클립 4건 중 1건에서 침묵했다. 6경기
477클립 실측에서 133건이 `ce >= e` 라 창이 비었고(그중 127건은 장면 끝 밖에 후보가
있었다) 남은 343건만 물었다. 컷 끝이 장면 끝에 붙는 건 사고가 아니라 정상 경로다 —
FULL_CLIP_TAGS(홈런·비디오 판독)는 끝을 장면 그대로 쓰고 관측 하한(_obs_floor)도
장면 끝을 상한으로 민다. 그래서 발행 경계 밖으로 조금 나간다. 넘는 것 자체는 이
변경이 처음도 아니다: cut._snap_tail 의 대사 꼬리가 이미 최대 9초를 넘긴다.

*아래쪽* — 하한이 '현재 끝'이라 이 노드는 **늘리기 전용**이었다. 끝이 늦은 클립
(리플레이·다음 타석 준비까지 물고 늘어진 컷)은 되돌릴 방법이 아예 없었다. 하한을
그 플레이의 투구로 내리면 클립 안쪽 샷 경계도 후보가 되어 당기기가 가능해진다.
투구보다 앞은 그 플레이가 시작되지도 않은 자리라 끝 후보일 수 없다.

MIN_CLIP_SEC 은 여기서도 지킨다 — 투구 직후 1초짜리 후보를 고르면 플레이가 통째로
사라진다. 시작 쪽 제약과 같은 값, 같은 이유다.
"""

import math
import re

from flow import cut
from log import get_logger

log = get_logger(__name__)

# 시작 후보에 붙이는 설명 — 출처가 하나뿐이라 상수다 (전광판 pitching_yn 검출).
START_CAND_WHY = "보드 검출 투구"
# 시작을 미뤄도 클립이 이보다 짧아지면 안 된다 — 플레이를 잘라 먹는다.
MIN_CLIP_SEC = 8
# 후보끼리 이보다 가까우면 **같은 지점**으로 본다.
# 근거: 해설 발화가 중앙 4.7~7.0초, 샷이 중앙 3.0~4.0초다(v200·201·202 실측).
# 후보가 그보다 촘촘하면 화면도 해설도 그 차이를 표현하지 못한다 — 실제로 v200
# 장면41 의 7702·7697 은 해설이 글자 그대로 같고 화면은 "포수 뒤쪽 기본 앵글에서…"
# 정형구라 판단에 쓸 수 없었다. 답이 없는 문제를 내면 모델은 판별자를 찾다가 사고를
# 태운다(장면61 94,575자·804초). 정보 손실이 아니라 없는 선택지를 지우는 것이다.
MERGE_GAP_SEC = 5.0
# 후보 개수 상한 — 많으면 모델이 고르지 못하고 프롬프트만 커진다.
CAND_MAX = 4
# 발화 끝과 샷 경계가 이 안에 함께 오면 "둘 다"로 표시한다 (가장 깔끔한 끝점).
COINCIDE_SEC = 2.0
# 끝 후보를 장면 끝(end_time) 밖으로 낼 폭 — 근거는 모듈 docstring.
# 시작에는 없는 값이다(시작 후보는 장면 자신의 pitch_idxs 뿐이라 밖이 존재하지 않는다).
# **이 값을 올릴 거면 다음 장면 침범을 같이 막아야 한다.** 5초는 6경기 실측 장면 간격
# (min 10.0s · p50 157s)보다 좁아 다음 장면에 닿지 않으므로 클램프를 두지 않았다.
END_OUT_EXT_SEC = 5.0
# 후보로 쓰지 않는 샷 유형 — 광고 경계를 클립 끝으로 삼으면 중계가 아닌 데서 끊긴다.
# (v201 장면5 실측: 끝 후보 넷 중 하나가 광고 경계였다.)
SKIP_SHOT_TYPES = frozenset({"광고"})
# 시작 후보의 화면이 비었을 때 **직후 샷**까지 당겨 보는 폭 (_ctx 근거 주석).
# 실측 간격이 최대 0.70초라 1.0 이면 전부 덮고, 샷 중앙 길이(3~4초)보다 한참 짧아
# 다음 장면을 끌어오지 않는다. 1.0~3.0 을 훑어도 결과가 같아 가장 좁은 값을 썼다.
CAND_SHOT_SNAP_SEC = 1.0
# 시작을 그대로 믿는 앵커 샷 유형 — 이 화면이면 시작 보정을 묻지 않는다.
# '투구'는 그 자체로 맞고, '리액션'은 타석 준비 화면이라 투구 직전일 수 있다.
# 나머지(타구·수비·주루·득점·홈인·광고·기타·미분류·앵커 없음)는 시작이 이미
# 플레이 도중이거나 중계가 아닌 화면이라 묻는다 — start_rows docstring 의 실측 근거.
TRUSTED_ANCHOR_SHOTS = frozenset({"투구", "리액션"})


def _shot_at(sec: float, segs: list[dict], at_end: bool = False) -> dict | None:
    """그 시각의 샷. at_end 면 **거기서 끝나는** 샷을 준다.

    끝 후보에 다음 샷을 붙이면 문맥이 뒤집힌다 — v201 장면5 의 끝 후보 1359s 는
    1357~1359 리액션이 끝나는 지점인데, 시작 기준으로 찾으면 그 뒤 광고가 잡혀
    "화면 [광고]" 로 나왔다. 고르라는 건 끝나는 지점이지 시작하는 지점이 아니다.
    """
    if at_end:
        for s in segs:
            if abs(s["e"] - sec) < 1:
                return s
        return None
    for s in segs:
        if s["s"] <= sec < s["e"]:
            return s
    return None


def _utt_at(sec: float, utts: list, window: float = 6.0, at_end: bool = False) -> str:
    """그 시각의 해설. at_end 면 **거기서 끝나는** 발화를 우선한다.

    끝 후보의 관심사는 "여기서 말이 끝나는가"다. 겹치는 발화를 주면 다음 문장이
    잡혀 그 후보를 고를 근거가 사라진다.
    """
    if at_end:
        ends = [t for _us, ue, t in utts if abs(math.ceil(ue) - sec) <= 1]
        if ends:
            return ends[-1]
    over = [t for us, ue, t in utts if us <= sec < ue]
    if over:
        return over[-1]
    prev = [t for _us, ue, t in utts if sec - window <= ue <= sec]
    return prev[-1] if prev else ""


def _snap_shot(sec: float, segs: list[dict]) -> dict | None:
    """캡션 없는 샷 꼬리에 걸린 초 → **직후 캡션 있는 샷**. 해당 없으면 None.

    보드 검출 투구는 정수 초라 그 투구 샷이 시작되기 직전, 캡션 없는 샷의 마지막
    0.x초에 걸린다 — 4경기 실측에서 화면 없는 시작 후보 11건이 이 모양이었고
    (간격 0.50~0.70s) 정작 그 자리의 '투구'·'타구·수비' 캡션은 바로 뒤에 멀쩡히
    있었다 (v200 장면80: 후보 12050 이 12045.93~12050.63 무캡션 샷 꼬리, 12050.63
    부터가 "투수가 마운드에서 공을 던지고…"). 구간 화면 블록이 이걸 가려 주고 있었다.

    화면(_ctx)과 **컷 좌표(start_rows 의 후보 초)에 함께** 쓴다. 보여준 화면과 실제로
    잘리는 자리가 어긋나면, 모델은 투구 샷을 보고 골랐는데 클립은 그 0.6초 앞
    무캡션 샷 조각부터 시작한다 — verify 의 완결성이 "첫 샷이 이전 상황 잔상"으로
    읽는 바로 그 모양이다. cut._anchor 의 교정(미분류 부스러기 샷 → 뒤 '투구' 샷)과
    같은 처방이다.
    """
    cur = _shot_at(sec, segs)
    if cur is not None and cur.get("summary"):
        return None
    return next((x for x in segs
                 if sec < x["s"] <= sec + CAND_SHOT_SNAP_SEC and x.get("summary")), None)


def _ctx(sec: float, segs: list[dict], utts: list, at_end: bool = False) -> dict:
    """후보 1건에 붙일 서사 — 그 시각의 화면과 해설.

    후보·서사·해설을 따로 세 블록으로 주면 모델이 시각을 맞춰 조인해야 하고 그
    조인이 사고를 태운다 (v201 장면5: thinking 59,501자·422초). 후보 밑에 바로
    붙이면 각 줄이 자기완결적이라 비교만 하면 된다.

    시작(at_end=False)은 화면이 비면 **직후 샷**까지 본다 — 근거는 _snap_shot.
    끝(at_end=True)은 손대지 않는다 — 끝의 관심사는 "여기서 끝나는 샷"이라 뒤를
    보면 문맥이 뒤집힌다(_shot_at docstring).
    """
    s = _shot_at(sec, segs, at_end) or (_shot_at(sec, segs) if at_end else None)
    if not at_end:
        s = _snap_shot(sec, segs) or s
    return {"shot_type": (s or {}).get("shot_type") or "",
            "shot": (s or {}).get("summary") or "",
            "utt": _utt_at(sec, utts, at_end=at_end)}


def _sig(ctx: dict) -> tuple:
    """후보의 변별 서명 — 모델에게 보이는 것(화면·해설)만으로 만든다.

    초 값이 달라도 화면·해설이 같으면 모델 입장에선 **같은 선택지**다. 고를 근거가
    프롬프트 안에 없으니 판별자를 찾다가 사고가 폭주한다 — v200 comp16 실측:
    장면61 의 후보 11397·11394·11392 는 해설이 글자 그대로 같았고(화면은 아예 없음)
    thinking 94,575자를 태운 뒤 본문 없이 재시도로 떨어졌다. 반면 후보 5개가 전부
    다른 해설을 단 장면11 은 4,343자로 즉결이었다 — **개수가 아니라 변별력**이다.
    """
    return (ctx.get("shot_type") or "", ctx.get("shot") or "", ctx.get("utt") or "")


def _dedup(cands: list[dict], drop_sig: tuple | None = None,
           near: float | None = None) -> list[dict]:
    """같은 지점으로 볼 후보를 하나로 접는다. 앞선 것(호출자가 정한 우선순위)이 남는다.

    두 기준을 함께 쓴다:
    - **서명**(화면·해설)이 같으면 같은 지점. drop_sig 와 같으면 통째로 뺀다 —
      '현재'와 같은 후보는 대안이 아니고 그 답은 이미 "유지"다.
    - **시간**이 MERGE_GAP_SEC 이내면 같은 지점. 문구 비교는 정형구 때문에 샌다
      (장면41: 해설 동일·화면만 미세하게 달라 둘 다 남았다). 시간은 새지 않는다.
    """
    seen: set[tuple] = set()
    kept: list[float] = []
    out = []
    for c in cands:
        s = _sig(c)
        if s == drop_sig or s in seen:
            continue
        if near is not None and any(abs(c["sec"] - k) <= near for k in kept):
            continue
        seen.add(s)
        kept.append(c["sec"])
        out.append(c)
    return out


# ──────────────────────────────────────────────────────
# 끝 — 모든 클립이 묻는다
# ──────────────────────────────────────────────────────

def end_candidates(clip: dict, segs: list[dict], utts: list) -> list:
    """끝 후보 [(초, 설명)] — **창 안의 샷 끝을 전부** 낸다.

    2026-08-24 재설계. 이전에는 발화 끝과 샷 끝을 섞어 종류 우선으로 CAND_MAX(4)까지
    자르고 5초 이내를 접었다. 그 결과 클립당 최종 후보가 평균 1.4개로 줄어 "고르기"가
    아니라 이지선다가 됐다 (comp37 실측: 후보 1개인 클립이 8/10).

    **샷 끝이 후보의 기준인 이유**: 클립이 실제로 끊기는 자리는 화면이 바뀌는 자리다.
    샷 한복판에서 자르면 다음 장면이 한 조각 붙거나 동작이 잘린다. 발화 끝은 그
    자체로 컷 지점이 아니라 "여기서 말이 끝난다"는 **근거**다 — 그래서 후보의 설명에
    병기하고, 컷 좌표는 샷 경계를 쓴다.

    발화 끝도 후보로 남긴다(샷 끝과 겹치지 않는 것만): 창 안에 샷 경계가 하나도 없는
    클립이 있어서다 (comp38 장면48 실측 — 샷만 내면 후보 0건이라 이 단계를 통째로
    건너뛴다). 다만 순서는 샷이 먼저다.

    상한·근접 접기는 없앴다. 후보가 많아 생기던 비용(우열 근거가 약할 때 thinking 폭주)은
    이 단계의 thinking 을 끄면서 사라졌다 — 근거는 graph.refine_end_bound_node 주석.

    **창은 [그 플레이의 투구, 장면 끝 + END_OUT_EXT_SEC] 이다** (2026-08-24 재조정).
    근거는 모듈 docstring. 두 끝 다 옮겼으므로 후보는 **현재 끝보다 앞일 수도 뒤일
    수도** 있다 — 앞이면 당기는 것이고 뒤면 미루는 것이다.

    하한은 `max(투구, 컷 시작 + MIN_CLIP_SEC, 관측 하한)` 이다. 투구는
    clip["pitch_sec"](t_scene_baseball.pitch_time — 상류가 고른 그 플레이의 대표
    투구)이고, 없으면 컷 시작을 쓴다. MIN_CLIP_SEC 을 더하는 건 투구 직후 후보를
    골라 플레이가 통째로 사라지는 걸 막기 위해서다.

    **관측 하한**(cut.obs_floor)은 전이 원장이 보증하는 결과 시점이다 — 그 앞은
    결과가 아직 안 나온 자리라 끝일 수 없다. 컷이 이미 지키는 바닥(cut._obs_floor)을
    후보 창도 같이 지킨다. 안 걸면 6경기 3,495후보 중 939건(27%)이 결과를 버리는
    선택지가 되고, 그런 후보를 가진 클립이 223건이다 — 그중 14건은 FULL_CLIP_TAGS
    라 컷 쪽 바닥조차 안 걸리는(_full 경로) 클립이었다.

    장면 끝을 넘는 후보에는 why 에 "장면 밖"을 단다. 발행이 "여기까지가 이 사건"이라고
    정한 선을 넘는다는 사실 자체가 판단 재료다 — 숨기고 내면 모델은 그게 같은 장면
    안인 줄 알고 고른다.
    """
    cs, ce = clip["cut"]["cs"], clip["cut"]["ce"]
    e = clip["e"]
    pitch = clip.get("pitch_sec")
    lo = max(cs + MIN_CLIP_SEC, cs if pitch is None else float(pitch),
             cut.obs_floor(clip, segs) or 0.0)
    # 대사 꼬리(cut._snap_tail)로 ce 가 이미 장면 끝을 넘어선 클립이 있다 — 그때도
    # 확장분은 **현재 끝 기준**으로 확보한다(장면 끝 기준이면 창이 그만큼 좁아진다).
    hi = max(e, ce) + END_OUT_EXT_SEC
    utt_ends = [math.ceil(ue) for _us, ue, _t in utts if lo < ue <= hi]
    shot_ends = [s["e"] for s in segs
                 if lo < s["e"] <= hi and s.get("shot_type") not in SKIP_SHOT_TYPES]

    def _why(sec: float, base: str) -> str:
        return f"{base} · 장면 밖" if sec > e else base

    out: list[tuple[float, str]] = []
    for se in sorted(shot_ends):
        near = any(abs(ue - se) <= COINCIDE_SEC for ue in utt_ends)
        out.append((float(se), _why(se, "화면 전환 + 발화 끝" if near else "화면 전환")))
    for ue in sorted(utt_ends):
        if not any(abs(ue - se) <= COINCIDE_SEC for se in shot_ends):
            out.append((float(ue), _why(ue, "발화 끝")))

    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in out:                       # 같은 정수 초는 하나 (샷 우선 — 먼저 넣었다)
        uniq.setdefault(int(sec), (sec, why))
    return sorted(uniq.values())


def end_rows(clips: list[dict], segs: list[dict], utts: list) -> list[dict]:
    """refine_end_bound 에 낼 행 — 후보가 없는 클립은 물어볼 게 없으므로 뺀다.

    **모든 클립이 대상**이다(앵커 유무 무관). 끝은 코드가 답할 수 없는 질문이라
    앵커가 잡힌 클립도 끝은 물어야 한다 — 구 배선은 앵커 클립을 통째로 건너뛰어
    끝조차 묻지 않았고, 그 결과 이 단계가 전체 클립의 2%만 다뤘다.
    """
    rows = []
    for c in clips:
        cs, ce = c["cut"]["cs"], c["cut"]["ce"]
        raw = [{"sec": sec, "why": why, **_ctx(sec, segs, utts, at_end=True)}
               for sec, why in end_candidates(c, segs, utts)]
        # 현재와 화면·해설이 똑같은 후보는 뺀다 — 종류를 가리지 않는다 (2026-08-24).
        #
        # 모델이 보는 것이 전부 같으면 고를 근거가 프롬프트 안에 없다. 답이 없는 문제를
        # 내면 판별자를 찾다가 사고를 태운다(comp37 장면46: 그런 후보 하나뿐인 콜이
        # thinking 11,366자 — 다른 콜의 5~10배).
        #
        # 처음엔 '발화 끝' 후보에만 걸었다. 샷 경계는 설명이 같아도 "샷을 끝까지 보고
        # 자른다"는 다른 선택이라 남겨야 한다고 봤는데, 실측이 그 걱정을 지웠다:
        # 4회 실행 78콜에서 '현재와 동일'한 샷 경계 후보는 4건뿐이고 **전부 다른 후보와
        # 함께** 나왔다. 즉 이걸 빼도 물어볼 콜이 사라지지 않는다 — 고를 수 없는
        # 선택지만 없어진다.
        #
        # 남은 후보가 0이면 end_rows 가 그 클립을 행에서 빼고, graph 가 asked=0 으로
        # 트레이스에 남긴다. "물어볼 게 없어서 안 물었다"가 기록으로 보인다.
        cur = _ctx(ce, segs, utts, at_end=True)
        ends = sorted(_dedup(raw, drop_sig=_sig(cur)), key=lambda e: e["sec"])
        if ends:
            # 장면 시작·투구도 후보와 **같은 모양(초 + 화면 + 해설)** 으로 싣는다
            # (2026-08-24). 후보만 나열하면 모델은 그 플레이가 어디서 시작해 어떻게
            # 흘러왔는지 모르는 채로 끝을 고른다 — 창의 하한이 투구로 내려가 현재보다
            # 앞선 후보가 나오는 지금은 특히 그렇다. 앞쪽 후보를 고르는 건 결과를
            # 버리는 일일 수도 있고 늘어진 꼬리를 자르는 일일 수도 있는데, 그 둘은
            # 투구 이후의 서사를 봐야 갈린다. 프롬프트는 이 재료를 시간순으로 편다.
            pitch = c.get("pitch_sec")
            rows.append({"scene_id": c["scene_id"], "tags": c["tags"],
                         "inning": c.get("inning") or "", "cs": cs, "ce": ce,
                         "scene_s": c["s"], "scene_ctx": _ctx(c["s"], segs, utts),
                         "pitch": None if pitch is None else float(pitch),
                         "pitch_ctx": (None if pitch is None
                                       else _ctx(float(pitch), segs, utts)),
                         "cur": cur, "ends": ends})
    return rows


# ──────────────────────────────────────────────────────
# 시작 — 투구 앵커가 없는 클립만 묻는다
# ──────────────────────────────────────────────────────

def start_candidates(clip: dict) -> list:
    """시작 후보 [(초, 설명)] — 그 장면 자신의 검출 투구 전량 (cs 앞뒤 모두).

    재료는 `clip["pitches"]` = t_scene_baseball.pitch_idxs 를 파싱한 (시작, 끝) 목록이다
    (repos.fetch_scenes). 다른 출처는 쓰지 않는다 — 근거는 모듈 docstring.

    **후보는 장면 구간(t_scene_baseball.start_time~end_time) 안에서만 낸다**
    (2026-08-24). 그 밖을 후보로 내면 발행이 정한 장면 경계를 이 단계가 넘어서게 되고,
    상류가 "여기부터 여기까지가 이 사건"이라고 정한 판단을 하류가 뒤집는 셈이 된다.

    거르는 건 셋이다:
    - 장면 구간 밖 (s 이전)
    - 뒤로 미뤄 클립이 MIN_CLIP_SEC 밑으로 짧아지는 값 (플레이를 잘라 먹는다)
    - 현재 시작과 MERGE_GAP_SEC 이내인 값 (같은 지점이라 대안이 아니다)
    """
    cs, ce = clip["cut"]["cs"], clip["cut"]["ce"]
    lo, hi = clip["s"], ce - MIN_CLIP_SEC      # hi 는 장면 끝(e) 안쪽이다 — ce <= e
    uniq: dict[int, tuple[float, str]] = {}
    for ps, _pe in clip.get("pitches") or ():
        sec = float(ps)
        if not lo <= sec <= hi or abs(sec - cs) < MERGE_GAP_SEC:
            continue
        uniq.setdefault(int(sec), (sec, START_CAND_WHY))
    # 현재 시작에 가까운 것부터 — 같은 타석 안에서도 멀수록 앞선 볼카운트의 투구다.
    return sorted(uniq.values(), key=lambda x: abs(x[0] - cs))[:CAND_MAX]


def start_rows(clips: list[dict], segs: list[dict], utts: list) -> list[dict]:
    """refine_start_bound 에 낼 행 — **시작이 못 미더운 클립만**.

    게이트가 "앵커 유무"에서 "앵커 샷의 유형"으로 바뀌었다 (2026-08-24). 구 조건은
    앵커만 잡히면 건너뛰었고 그 근거는 실측이었다(2026-08-20: v200 66/67 · v201
    66/71 · v202 51/56 이 '투구' 시작). **그 실측이 구 상류 기준이라 더 이상 성립하지
    않는다** — 지금은 상류가 대표 투구(pitch_time)를 골라 주고 cut 이 그 초가 든 샷을
    앵커로 쓰는데, 그 샷의 61%가 투구 화면이 아니다(406장면: 리액션 219·타구/수비
    16·광고 4·기타 4·주루 3·미분류 1). 조건이 그대로 남아 이 노드는 대상 0건으로
    조용히 놀고 있었다.

    그렇다고 전부 묻지는 않는다. 샷 분류가 25%만 채워져 있어(v201 4,013샷 중 1,000)
    '리액션'이 진짜 리액션인지 투구 오분류인지 갈리지 않고, 맞는 시작을 되돌리는
    사고가 실재한다 — v201 comp20 의 장면9(-33s)·42(-37.3s). 그래서 **투구일 수 있는
    화면**(투구·리액션=타석 준비)은 믿고 넘기고, 명백히 아닌 것만 묻는다:
    타구·수비·주루·득점·홈인(이미 플레이 도중) · 광고(중계가 아님) · 미분류 · 앵커 없음.

    행에는 **후보와 그 시각의 화면·해설만** 싣는다 — 구간의 대사·화면을 통째로
    주던 블록은 뺐다 (2026-08-24). 그 블록의 존재 이유는 "후보의 화면 분류가 NULL
    이면 후보끼리 구별되지 않는다"였는데, 그 결손은 분류가 없어서가 아니라 후보 초가
    캡션 있는 샷 시작보다 0.x초 일렀던 탓이라 _snap_shot 이 후보 줄에서 직접 푼다.

    실측(4경기 325장면, 후보 재료가 t_play_baseball 전량이던 시점 → 행 18·후보 34):
    화면 없는 후보 0 · 완전 공백 0 · 현재와 서명이 같은 후보 0 · 후보끼리 겹치는
    서명 0. 스냅을 끄면 화면 없는 후보가 11건으로 돌아가고 그중 1건은 해설조차
    없다 — 블록이 가려 주던 게 이 결손이다. **후보 재료가 바뀌면 이 수치는 다시
    재야 한다**(행·후보 개수는 재료에 딸린 값이고, 변별력 0 이라는 결론만 재료와
    독립이다).

    구간 블록은 그 대가로 프롬프트를 두 배 남짓 불렸다(같은 실측에서 콜당 중앙
    699 → 379자). 모델이 시각을 맞춰 조인해야 하는 형식이라 사고를 태우는 쪽이기도
    했다 — _ctx docstring 의 v201 장면5(thinking 59,501자).
    """
    rows = []
    for c in clips:
        if c["cut"].get("anchor_type") in TRUSTED_ANCHOR_SHOTS:
            continue
        cs, ce = c["cut"]["cs"], c["cut"]["ce"]
        cands = start_candidates(c)
        if not cands:
            continue
        cur = _ctx(cs, segs, utts)
        # 후보에도 그 시각의 화면·해설을 붙인다 (2026-08-24). 구간 블록만 주면
        # 모델이 시각을 맞춰 조인해야 하고, 후보 줄이 초 하나뿐이면 후보끼리
        # 구별할 재료가 없다 — v200 장면15 의 정답 후보가 그랬다.
        # 후보 초도 스냅한다 — 보여준 화면이 시작되는 자리가 곧 컷 자리다
        # (_snap_shot). 보드 검출 정수 초를 그대로 적용하면 클립이 무캡션 샷
        # 조각 0.6초부터 시작한다.
        raw = [{"sec": (nx["s"] if (nx := _snap_shot(sec, segs)) else sec),
                "why": why, **_ctx(sec, segs, utts)}
               for sec, why in cands]          # start_candidates 순 = 현재에 가까운 순
        # 화면·해설이 똑같은 후보는 접는다 — 끝(end_rows)과 같은 기준이다.
        # 모델에게 보이는 것이 같으면 고를 근거가 프롬프트 안에 없고, 답 없는 문제를
        # 내면 판별자를 찾다 사고를 태운다(_sig). 새로 필요해진 건 스냅 때문이다:
        # 긴 '투구' 샷 하나에 검출 투구가 둘이면 둘 다 그 샷의 캡션으로 스냅돼
        # 글자 그대로 같은 선택지가 된다 (v203 장면67: 10993·11003 실측).
        # **가까운 쪽이 남는다** — 순서가 곧 우선순위인데 start_candidates 가 현재에
        # 가까운 순으로 주고, "현재에서 가장 가까운 투구를 고른다"가 START_SYSTEM 의
        # 규칙이다. 먼 쪽을 남기면 앞선 볼카운트의 투구로 시작이 끌려간다.
        kept = sorted(_dedup(raw, drop_sig=_sig(cur)), key=lambda x: x["sec"])
        if not kept:
            # 끝은 후보가 없으면 그만이지만, 시작은 게이트가 이미 "이 클립의 시작은
            # 못 미덥다"고 판정한 뒤다 — 조용히 넘기면 나쁜 시작이 안 물어본 채 남는다.
            log.info("refine_start_bound 물을 게 없음: 장면%d 후보 %d건이 전부 현재와 같다",
                     c["scene_id"], len(raw))
            continue
        rows.append({
            "scene_id": c["scene_id"], "tags": c["tags"],
            "inning": c.get("inning") or "", "cs": cs, "ce": ce,
            "cur": cur, "cands": kept,
        })
    return rows


# ──────────────────────────────────────────────────────
# 처분 — 제시한 후보와 일치할 때만 적용
# ──────────────────────────────────────────────────────

def _match(cands: list[dict], sec: int) -> float | None:
    """모델이 답한 정수 초 → 그 후보의 **실제 초(소수 포함)**. 없으면 None.

    프롬프트는 후보를 정수로 보여주지만(모델에게 0.3초는 의미가 없다) 적용은 원래
    값으로 해야 한다. 정수로 되돌리면 샷 경계 바로 뒤였던 시작이 직전 샷으로 넘어간다
    — v201 은 t_segment 경계가 소수라 4369.3s('투구' 시작)가 4369s('리액션' 끝자락)가
    됐다. 후보는 int(sec) 로 중복 제거돼 있어 정수 1개당 후보 1건이다.
    """
    return next((c["sec"] for c in cands if int(c["sec"]) == sec), None)


def _apply_one(clips: list[dict], rows: list[dict], text: str, *,
               field: str, key: str, label: str, by_index: bool) -> list[str]:
    """제안 처분 공통 — 응답은 **값 하나 또는 "유지"** (장면 번호 없음).

    클립 1건 = 콜 1건이라(refine_*_bound_node 가 행마다 따로 부른다) 번호가 없어도
    어느 클립인지 모호하지 않다. 줄 형식이 어긋나면 응답 전체가 **조용히 무시**돼
    로그엔 "이동 0건"으로만 남으므로, 앞에 군말이 붙어도 마지막 숫자를 답으로 읽는다.

    by_index=True 면 그 숫자는 **후보 번호(1부터)** 다 (끝 — 2026-08-24). 모델에게
    시각을 받아쓰게 하면 소수가 사라진다: 내부 좌표는 t_segment 경계라 소수를 갖고
    있고(4369.3s = '투구' 샷 시작), 정수로 되돌리면 직전 샷 꼬리로 넘어간다. 번호로
    받으면 원래 값을 그대로 되찾고, 지어낸 번호는 목록 밖이라 기각된다.

    by_index=False 면 숫자는 초다 (시작 — 아직 초 형식). 단위 표기(초·s·sec)는 붙든
    안 붙든 받는다: 모델마다 다르게 붙인다 (Qwen3.8 "9303s" / Qwen3.6 "1355초" 실측).
    """
    if not rows:
        return []
    row = rows[0]
    cut_ = next((c["cut"] for c in clips if c["scene_id"] == row["scene_id"]), None)
    if cut_ is None:
        return []
    body = text.strip()
    if "유지" in body:
        return []
    nums = re.findall(r"\d+", body)
    if not nums:
        log.info("bounds 기각: 장면%d %s 응답 해석 불가 %r", row["scene_id"], label, body[:40])
        return []
    n = int(nums[-1])
    cands = row[key]
    if by_index:
        if not 1 <= n <= len(cands):
            log.info("bounds 기각: 장면%d %s 후보 번호 %s (1~%d 밖)",
                     row["scene_id"], label, n, len(cands))
            return []
        hit = cands[n - 1]["sec"]
    elif (hit := _match(cands, n)) is None:
        log.info("bounds 기각: 장면%d %s %s (후보 밖)", row["scene_id"], label, n)
        return []
    moved = [f"장면{row['scene_id']} {label} {cut_[field]:.1f}→{hit:.1f}"]
    cut_[field] = hit
    cut_["mode"] += f"+{label}보정"
    return moved


def apply_end(clips: list[dict], rows: list[dict], text: str) -> list[str]:
    """끝 제안 처분 — 응답은 **후보 번호**. 이동 기록을 돌려준다."""
    return _apply_one(clips, rows, text, field="ce", key="ends", label="끝", by_index=True)


def apply_start(clips: list[dict], rows: list[dict], text: str) -> list[str]:
    """시작 제안 처분 — 끝과 같이 **후보 번호**로 받는다 (2026-08-24 형식 통일)."""
    return _apply_one(clips, rows, text, field="cs", key="cands", label="시작",
                      by_index=True)
