"""refine_end_bound · refine_start_bound 의 재료 — 후보 생성과 처분 (순수 계산, LLM 무관).

**끝과 시작을 다른 노드로 나눈다** (2026-08-20 재설계). 대상이 다르기 때문이다:
- 끝은 **모든 클립**이 묻는다. "결과를 설명하는 해설이 어디서 끝나는가"는 코드가 답할
  수 없고, cut 의 레시피 체인은 샷 오분류 하나로 결과 전에 끊긴다.
- 시작은 **투구 앵커가 없는 클립만** 묻는다. 앵커가 잡히면 cut 이 정한 시작이 이미
  그 플레이의 투구라 모델이 손대면 앞 타석으로 되돌아갔다 (v201 comp20: 장면9 -33.0s,
  장면42 -37.3s).

후보는 여기서 **결정적으로** 만들고 LLM 은 고르기만 한다 — 검증기(apply_*)가 제시 목록
밖의 값을 기각하므로 모델이 시각을 지어낼 수 없다.

**시작 후보는 cs 앞뒤로 낸다** (2026-08-20 — 구 "앞으로만" 폐기). 그 전제는 "장면 시작은
그 플레이의 투구보다 이르거나 같다" 였는데 v203 이 반증했다: 장면5(홈런)는 장면이 657s 에
시작하는데 그 홈런의 투구는 646~650s 다. 클립이 "담장 넘어갑니다"부터 시작해 스윙이 없다.
장면37 도 같다(장면 8321s / 투구 8314~8318s). 전이 원장의 시각이 투구가 아니라 관측
시점(타구·득점 판정)에 찍히면 이렇게 뒤집힌다.

뒤쪽 후보는 분류된 '투구' 샷도 쓰지만 **앞쪽은 보드 검출 투구만** 낸다. shot_type 은
하이라이트 구간과 겹치는 샷만 채워져 장면 이전은 대부분 NULL 이라(v203 실측: 3,238샷 중
2,613샷 81% 가 NULL) 분류 기반 후보를 만들 수 없다. 보드 검출은 shot_type 과 독립이라
그 구간에서 살아 있는 유일한 단서다.
"""

import math
import re

from log import get_logger

log = get_logger(__name__)

# 시작을 **뒤로 미룰 수 있는** 최대 폭 — 장면 경계가 앞 타석 꼬리를 무는 경우.
# v202 장면11(역전 홈런): 장면 2558s 인데 그 홈런의 투구는 2583s, 앞 25초는 앞 타자의
# 견제 장면이라 "무슨 일인지 알 수 없는" 시작이었다.
START_MAX_FWD_SEC = 30
# 시작을 **앞으로 당길 수 있는** 최대 폭 — 장면 시작이 이미 플레이 도중인 경우.
# v203 장면5 는 11초, 장면37 은 7초 앞에 그 플레이의 투구가 있다. 넓히면 앞 타석의
# 투구까지 들어오므로(타석 간격 20~40초) 그 안쪽으로 끊는다.
START_MAX_BACK_SEC = 15
# 시작을 미뤄도 클립이 이보다 짧아지면 안 된다 — 플레이를 잘라 먹는다.
MIN_CLIP_SEC = 8
# 후보끼리 이보다 가까우면 **같은 지점**으로 본다.
# 근거: 해설 발화가 중앙 4.7~7.0초, 샷이 중앙 3.0~4.0초다(v200·201·202 실측).
# 후보가 그보다 촘촘하면 화면도 해설도 그 차이를 표현하지 못한다 — 실제로 v200
# 장면41 의 7702·7697 은 해설이 글자 그대로 같고 화면은 "포수 뒤쪽 기본 앵글에서…"
# 정형구라 판단에 쓸 수 없었다. 답이 없는 문제를 내면 모델은 판별자를 찾다가 사고를
# 태운다(장면61 94,575자·804초). 정보 손실이 아니라 없는 선택지를 지우는 것이다.
MERGE_GAP_SEC = 5.0
# 끝을 늘릴 수 있는 최대 폭.
END_MAX_EXT_SEC = 12
# 후보 개수 상한 — 많으면 모델이 고르지 못하고 프롬프트만 커진다.
CAND_MAX = 4
# 발화 끝과 샷 경계가 이 안에 함께 오면 "둘 다"로 표시한다 (가장 깔끔한 끝점).
COINCIDE_SEC = 2.0
# 후보로 쓰지 않는 샷 유형 — 광고 경계를 클립 끝으로 삼으면 중계가 아닌 데서 끊긴다.
# (v201 장면5 실측: 끝 후보 넷 중 하나가 광고 경계였다.)
SKIP_SHOT_TYPES = frozenset({"광고"})
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


def _ctx(sec: float, segs: list[dict], utts: list, at_end: bool = False) -> dict:
    """후보 1건에 붙일 서사 — 그 시각의 화면과 해설.

    후보·서사·해설을 따로 세 블록으로 주면 모델이 시각을 맞춰 조인해야 하고 그
    조인이 사고를 태운다 (v201 장면5: thinking 59,501자·422초). 후보 밑에 바로
    붙이면 각 줄이 자기완결적이라 비교만 하면 된다.
    """
    s = _shot_at(sec, segs, at_end) or (_shot_at(sec, segs) if at_end else None)
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
    """끝 후보 [(초, 설명)] — **창 안의 샷 끝을 전부** 낸다. 현재 끝 이후만.

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
    """
    ce = clip["cut"]["ce"]
    hi = ce + END_MAX_EXT_SEC
    utt_ends = [math.ceil(ue) for _us, ue, _t in utts if ce < ue <= hi]
    shot_ends = [s["e"] for s in segs
                 if ce < s["e"] <= hi and s.get("shot_type") not in SKIP_SHOT_TYPES]

    out: list[tuple[float, str]] = []
    for se in sorted(shot_ends):
        near = any(abs(ue - se) <= COINCIDE_SEC for ue in utt_ends)
        out.append((float(se), "화면 전환 + 발화 끝" if near else "화면 전환"))
    for ue in sorted(utt_ends):
        if not any(abs(ue - se) <= COINCIDE_SEC for se in shot_ends):
            out.append((float(ue), "발화 끝"))

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
            rows.append({"scene_id": c["scene_id"], "tags": c["tags"],
                         "inning": c.get("inning") or "", "cs": cs, "ce": ce,
                         "cur": cur, "ends": ends})
    return rows


# ──────────────────────────────────────────────────────
# 시작 — 투구 앵커가 없는 클립만 묻는다
# ──────────────────────────────────────────────────────

def start_candidates(clip: dict, segs: list[dict], pitches: list[tuple[int, int]]) -> list:
    """시작 후보 [(초, 설명)] — cs 앞뒤 모두. 클립이 너무 짧아지는 값은 뺀다.

    앞쪽(cs 이전)은 **보드 검출 투구만** 낸다: 그 구간의 shot_type 은 대부분 NULL 이라
    분류 기반 후보를 만들 수 없고(모듈 docstring), 보드 검출은 분류와 독립이다.
    뒤쪽(cs 이후)은 보드 검출 + 분류된 '투구' 샷, 둘 다 없으면 '타구·수비' 시작.
    """
    cs, ce = clip["cut"]["cs"], clip["cut"]["ce"]
    fwd_hi = min(cs + START_MAX_FWD_SEC, ce - MIN_CLIP_SEC)
    back_lo = cs - START_MAX_BACK_SEC

    def near_fwd(sec: float) -> bool:
        # 현재 시작과 MERGE_GAP_SEC 이내면 같은 지점이라 대안이 아니다.
        return cs <= sec <= fwd_hi and abs(sec - cs) >= MERGE_GAP_SEC

    def near_back(sec: float) -> bool:
        return back_lo <= sec < cs and abs(sec - cs) >= MERGE_GAP_SEC

    out: list[tuple[float, str]] = []
    for ps, _pe in pitches:                        # ① 보드 검출 투구 (분류 없이도 있다)
        if near_back(ps) or near_fwd(ps):
            out.append((float(ps), "보드 검출 투구"))
    for s in segs:                                 # ② 분류된 투구 샷 시작 (뒤쪽만)
        if near_fwd(s["s"]) and s.get("shot_type") == "투구":
            out.append((s["s"], "'투구' 샷 시작"))
    if not out:                                    # ③ 투구를 못 찾으면 타구·수비 시작
        for s in segs:
            if near_fwd(s["s"]) and s.get("shot_type") == "타구·수비":
                out.append((s["s"], "'타구·수비' 샷 시작"))
    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in out:
        uniq.setdefault(int(sec), (sec, why))
    # 현재 시작에 가까운 것부터 — 멀수록 **다른 타석**의 투구일 위험이 크다.
    return sorted(uniq.values(), key=lambda x: abs(x[0] - cs))[:CAND_MAX]


# 구간 목록 상한 — 후보~현재 사이가 길어도 프롬프트가 불어나지 않게.
CONTEXT_MAX = 12


def _window(items: list, lo: float, hi: float, key_s, key_e) -> list:
    """[lo, hi] 와 겹치는 항목만 시간순 (뒤쪽 우선으로 상한 적용 — 가까울수록 중요)."""
    hit = [x for x in items if key_e(x) > lo and key_s(x) < hi]
    return hit[-CONTEXT_MAX:]


def start_rows(clips: list[dict], segs: list[dict], utts: list,
               pitches: list[tuple[int, int]]) -> list[dict]:
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

    행에는 후보 초 목록과 함께 **후보~현재 구간의 대사·화면을 통째로** 싣는다
    (2026-08-20 형식 변경). 후보마다 그 시각의 한 줄만 붙이던 방식은 후보의 화면
    분류가 NULL 이면(v203 실측 81%) 보여줄 게 없어 후보끼리 구별되지 않았다.
    구간을 통으로 주면 모델이 "그 사이에 무슨 일이 있었나"로 판별할 수 있다.
    """
    rows = []
    for c in clips:
        if c["cut"].get("anchor_type") in TRUSTED_ANCHOR_SHOTS:
            continue
        cs, ce = c["cut"]["cs"], c["cut"]["ce"]
        cands = start_candidates(c, segs, pitches)
        if not cands:
            continue
        lo = min(cs, min(sec for sec, _ in cands))
        hi = max(cs, max(sec for sec, _ in cands))
        rows.append({
            "scene_id": c["scene_id"], "tags": c["tags"],
            "inning": c.get("inning") or "", "cs": cs, "ce": ce,
            "cur": _ctx(cs, segs, utts),
            "cands": [{"sec": sec, "why": why} for sec, why in sorted(cands)],
            "utts": _window(utts, lo, hi, lambda u: u[0], lambda u: u[1]),
            "shots": _window(segs, lo, hi, lambda x: x["s"], lambda x: x["e"]),
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
