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

**끝은 후보 목록이 아니라 서사다** (2026-08-24 재설계). 장면 시작부터 장면 끝까지
세그먼트를 시간순으로 다 싣고, 그중 **투구 이후의 자를 수 있는 자리**에만 번호를
붙인다. 이유는 두 가지다.

*빠져 있던 것* — 구 형식은 "현재 끝 이후의 샷 끝"만 뽑아 나열했다. 시작도 투구도
결과도 프롬프트에 없어서, 모델은 이 플레이가 어떻게 시작해 어디서 결과가 났는지
모르는 채 끝만 골랐다.

*못 하던 것* — 후보가 현재 끝 이후뿐이라 **늘리기 전용**이었다. 리플레이·관중
리액션·다음 타석까지 물고 늘어진 컷(v1004 장면20 은 69초)을 되돌릴 방법이 없었다.
후보의 하한을 그 플레이의 투구로 내려 당기기를 연다. MIN_CLIP_SEC 은 여기서도
지킨다 — 투구 직후를 고르면 스윙만 남고 결과가 사라진다.

상한은 장면 끝이다. 한때 장면 끝 + 5초까지 열었다가 접었다(발행 경계를 하류가
넘어서지 않는다 — 90287c9 의 규칙). 창의 하한이 투구로 내려간 지금은 컷 끝이 장면
끝에 붙은 클립(6경기 133건)도 안쪽에 고를 자리가 생겨, 밖으로 나갈 이유가 없다.

관측 하한(cut.obs_floor)은 후보에서 잘라 내지 않고 "결과 전광판" 마커로 보여 준다 —
v1004 장면20 은 전광판이 결과보다 한참 뒤(00:42:26)에 찍혀, 잘라 냈으면 손댈 방법이
없었다. 코드가 자르는 대신 무슨 일이 있었는지를 보여 주고 판단을 맡긴다.
"""

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
# 장면 경계에 걸친 세그먼트가 클램프 후 이보다 짧으면 줄로 내지 않는다.
# 0.x초짜리 조각은 화면이라 부를 게 없고, 마커만 그리로 끌려가 엉뚱한 줄에 붙는다.
SEG_MIN_SEC = 1.0
# 대표 투구(pitch_time)가 세그먼트 끝 이 안쪽에 걸리면 **다음** 세그먼트가 투구 화면이다.
# 보드 검출은 정수 초라 투구 샷 직전 조각의 꼬리에 걸린다 — cut._anchor 의 교정과 같다.
PITCH_SNAP_SEC = 1.0
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


def _shot_at(sec: float, segs: list[dict]) -> dict | None:
    """그 시각의 샷 (시작 보정 전용 — 끝은 세그먼트 골격이라 이 조회를 안 쓴다)."""
    for s in segs:
        if s["s"] <= sec < s["e"]:
            return s
    return None


def _utt_at(sec: float, utts: list, window: float = 6.0) -> str:
    """그 시각의 해설 — 겹치는 발화, 없으면 window 안의 직전 발화."""
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


def _ctx(sec: float, segs: list[dict], utts: list) -> dict:
    """시작 후보 1건에 붙일 서사 — 그 시각의 화면과 해설.

    후보·서사·해설을 따로 세 블록으로 주면 모델이 시각을 맞춰 조인해야 하고 그
    조인이 사고를 태운다 (v201 장면5: thinking 59,501자·422초). 후보 밑에 바로
    붙이면 각 줄이 자기완결적이라 비교만 하면 된다.

    화면이 비면 **직후 샷**까지 본다 — 근거는 _snap_shot.
    """
    s = _snap_shot(sec, segs) or _shot_at(sec, segs)
    return {"shot_type": (s or {}).get("shot_type") or "",
            "shot": (s or {}).get("summary") or "",
            "utt": _utt_at(sec, utts)}


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

def _span(clip: dict, segs: list[dict]) -> list[dict]:
    """장면 구간에 걸친 세그먼트 — 표시 시각을 장면 경계로 클램프한 사본.

    클램프하지 않으면 경계에 걸친 세그먼트가 장면 밖 시각을 물고 온다 — "장면 시작
    00:08:17" 인데 첫 줄이 00:08:14 로 찍힌다. 클램프 후 SEG_MIN_SEC 미만으로 남는
    조각은 줄로 내지 않는다(0.x초짜리는 화면이라 부를 게 없다).
    """
    s0, e0 = clip["s"], clip["e"]
    out = []
    for x in segs:
        s1, e1 = max(x["s"], s0), min(x["e"], e0)
        if e1 - s1 >= SEG_MIN_SEC:
            out.append({"s": s1, "e": e1, "shot_type": x.get("shot_type") or "",
                        "shot": x.get("summary") or ""})
    return out


def _pitch_idx(span: list[dict], pitch: float | None) -> int | None:
    """투구 화면인 세그먼트의 인덱스 — 경계 부스러기면 다음으로 스냅.

    대표 투구(pitch_time)는 정수 초라 투구 샷이 시작되기 직전 세그먼트의 마지막
    0.x초에 걸리는 일이 흔하다 (cut._anchor 의 교정과 같은 실측). 그대로 두면
    "◀ 투구" 마커가 앞 리액션 샷에 붙는다 — v1004 장면7 실측.
    """
    if pitch is None or not span:
        return None
    i = next((k for k, x in enumerate(span) if x["s"] <= pitch < x["e"]), None)
    if i is None:
        i = next((k for k, x in enumerate(span) if x["s"] >= pitch), None)
    if i is not None and span[i]["e"] - pitch < PITCH_SNAP_SEC and i + 1 < len(span):
        i += 1
    return i


def _end_lines(span: list[dict], utts: list, *, pi: int | None, lo: float,
               obs: float | None, ce: float) -> tuple[list[dict], list[dict]]:
    """세그먼트 골격 → (줄 목록, 후보 목록). 번호는 줄과 후보가 같은 순서로 공유한다.

    접는 기준은 "모델에게 보여 줄 게 없는 줄"이다 — 화면 설명도 해설도 마커도 없고
    지금 자리도 아닌 세그먼트. 샷 캡션이 25%만 채워져 있어(v201 4,013샷 중 1,000)
    접지 않으면 '(설명 없음)' 줄이 대부분이 된다.
    """
    lines: list[dict] = []
    ends: list[dict] = []
    fold: list[dict] = []
    seen: set[tuple[float, float]] = set()

    def flush() -> None:
        if fold:
            lines.append({"kind": "fold", "at": fold[0]["s"], "n": len(fold)})
            fold.clear()

    for k, x in enumerate(span):
        said = [(us, ue, t) for us, ue, t in utts
                if us < x["e"] and ue > x["s"] and (us, ue) not in seen]
        marks = []
        if k == 0:
            marks.append("장면 시작")
        if pi is not None and k == pi and k != 0:
            marks.append("투구")
        if obs is not None and x["s"] < obs <= x["e"]:
            marks.append("결과 전광판")
        is_cur = abs(x["s"] - ce) < 1
        if not (x["shot"] or said or marks or is_cur):
            fold.append(x)
            continue
        flush()
        line = {"kind": "cur" if is_cur else "seg", "at": x["s"],
                "shot_type": x["shot_type"], "shot": x["shot"], "marks": marks,
                "utts": [t for _us, _ue, t in said]}
        # 자를 수 있는 자리 = 투구 이후 · 광고 아님 · 지금 자리 아님 · 변별 재료 있음
        if (not is_cur and x["s"] > lo and x["shot_type"] not in SKIP_SHOT_TYPES
                and (x["shot"] or said) and abs(x["e"] - ce) >= 1):
            ends.append({"sec": x["e"], "at": x["s"], "shot_type": x["shot_type"],
                         "shot": x["shot"]})
            line["num"] = len(ends)
        lines.append(line)
        seen.update((us, ue) for us, ue, _t in said)
    flush()
    return lines, ends


def end_rows(clips: list[dict], segs: list[dict], utts: list) -> list[dict]:
    """refine_end_bound 에 낼 행 — **장면 전체를 세그먼트 한 줄기로** 편다.

    **모든 클립이 대상**이다(앵커 유무 무관). 끝은 코드가 답할 수 없는 질문이라
    앵커가 잡힌 클립도 끝은 물어야 한다.

    2026-08-24 재설계 — 후보 목록에서 서사로. 이전에는 "현재 끝 이후의 샷 끝"만
    후보로 뽑아 나열했다. 그 형식은 두 가지를 못 했다:
    - **시작~결과 사이가 통째로 빠졌다.** 모델은 이 플레이가 어떻게 시작해서 어디서
      결과가 났는지 모르는 채 끝만 골랐다.
    - **늘리기만 됐다.** 리플레이·관중 리액션까지 물고 늘어진 컷(v1004 장면20 은
      69초)을 되돌릴 후보가 아예 없었다.

    이제 장면 시작부터 장면 끝까지 세그먼트를 시간순으로 다 싣고, 그중 **투구
    이후의 자를 수 있는 자리**에만 번호를 붙인다. 서사와 선택지가 같은 줄이다.

    줄(lines)의 종류:
    - seg  — 세그먼트 1개. 표시 시각은 **시작**(그 화면이 시작하는 자리), 번호가
             붙었으면 고를 수 있다. 고르면 컷 끝은 그 세그먼트의 **끝**이 된다
             (ends[번호-1]["sec"]) — 고른 화면까지 클립에 들어간다.
    - fold — 화면 설명도 해설도 마커도 없는 연속 구간을 한 줄로 접은 것. 번호 없음.
             샷 캡션이 25%만 채워져 있어(v201 4,013샷 중 1,000) 접지 않으면
             '(설명 없음)' 줄이 15개 중 13개가 된다 — 고를 근거가 없는 선택지다.
    - cur  — 지금 컷이 끝나는 자리. 세그먼트 시작과 맞으면 그 줄이 cur 이 되고,
             안 맞으면(장면 끝까지 쓰는 클립이 흔하다) 그 자리의 화면·해설을 붙여
             맨 뒤에 따로 낸다.

    마커(marks): 장면 시작 · 투구 · 결과 전광판(cut.obs_floor — 전이 원장이 보증하는
    결과 관측). 관측 하한을 후보에서 잘라 내는 대신 마커로 보여 주고 판단을 맡긴다 —
    v1004 장면20 은 전광판이 결과보다 한참 뒤(00:42:26)에 찍혀, 잘라 냈으면 이 장면은
    손댈 방법이 없었다.
    """
    rows = []
    for c in clips:
        cs, ce = c["cut"]["cs"], c["cut"]["ce"]
        span = _span(c, segs)
        if not span:
            continue
        pi = _pitch_idx(span, None if c.get("pitch_sec") is None else float(c["pitch_sec"]))
        # 후보는 투구 이후부터. MIN_CLIP_SEC 은 여기서도 지킨다 — 투구 직후를 고르면
        # 스윙만 남고 결과가 통째로 사라진다.
        lo = max(cs + MIN_CLIP_SEC, span[pi]["s"] if pi is not None else cs)
        obs = cut.obs_floor(c, segs)

        lines, ends = _end_lines(span, utts, pi=pi, lo=lo, obs=obs, ce=ce)

        if not ends:
            continue
        if not any(ln["kind"] == "cur" for ln in lines):
            # 컷 끝이 세그먼트 시작과 안 맞는다 — 장면 끝까지 쓰거나(가장 흔하다)
            # 대사 꼬리로 화면 도중에 걸린 경우. 그 자리의 화면·해설을 같이 낸다.
            at = next((x for x in span if x["s"] <= ce <= x["e"]), None)
            lines.append({"kind": "cur", "at": ce, "sec": ce,
                          "shot_type": (at or {}).get("shot_type", ""),
                          "shot": (at or {}).get("shot", ""), "marks": [],
                          "note": "장면 끝" if ce >= c["e"] - 1 else "화면 도중",
                          "utts": [t for us, ue, t in utts if us <= ce < ue][-1:]
                                  or [t for _us, ue, t in utts if ce - 6 <= ue <= ce][-1:]})
        rows.append({"scene_id": c["scene_id"], "tags": c["tags"],
                     "inning": c.get("inning") or "", "cs": cs, "ce": ce,
                     "scene_s": c["s"], "scene_e": c["e"],
                     "lines": lines, "ends": ends})
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
