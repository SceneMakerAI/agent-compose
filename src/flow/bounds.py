"""bounds — 클립 시작·끝 후보 생성과 처분 (순수 계산, LLM 무관).

시작과 끝을 한 곳에서 다루는 이유: 따로 물으면 서로를 모른다. 시작을 25초 당기면
끝의 여유도 달라진다.

후보는 여기서 **결정적으로** 만들고 LLM 은 고르기만 한다 — 검증기(apply)가 제시 목록
밖의 값을 기각하므로 모델이 시각을 지어낼 수 없다.

**대상은 투구 앵커가 없는 클립뿐이다** (2026-08-20 축소). 앵커가 잡힌 클립에서
cut 의 시작은 이미 그 플레이의 투구 샷이고, LLM 이 손대면 앞 타석으로 되돌아갔다
— v201 comp20 실측: 15클립 중 12건이 cut 단계에서 '투구' 시작이었는데 bounds 가
장면9 를 -33.0s, 장면42 를 -37.3s 옮겨 각각 미분류·'타구·수비' 시작이 됐다.
(그 이전 판들이 "투구 시작 0%"로 보였던 건 별개 원인 — t_compose_clip.start_time 이
초 정밀도라 4369.3s→4369.0s 로 내려가며 직전 샷에 걸친 착시였다. 컬럼을 time(3)
으로 넓혀 해소.)

**시작 후보는 앞으로만 낸다** (2026-08-20 축소 — 구 START_MAX_BACK_SEC 폐지).
5경기 전수 실측: 게이트를 통과한 클립의 뒤쪽 시작 후보 29건 중 **그 플레이 자신의
투구는 0건**이었다. 전부 클립 시작보다 이른, 앞 플레이의 투구였다. 우연이 아니라
구조가 그렇다 —

- 여기까지 오는 클립은 앵커가 없어 cut 이 `_full` 로 처리한 것이라 cs = 장면 시작이다.
- 장면 시작은 vision3 가 **투구 샷 머리**(refined) 또는 전이 원장 start_sec
  (unresolved)로 잡는다. 앞은 정의상 투구 샷의 시작이고, 뒤는 앞 전이의 관측이라
  더 이르다 — 어느 쪽이든 그 플레이의 투구보다 **이르거나 같다**.
- 그러니 고칠 여지는 언제나 cs 이후에 있다. v202 장면11(역전 홈런)이 전형이다:
  장면 2558s, 그 홈런의 투구 2583s, 앞 25초는 앞 타자의 견제 장면.

shot_type 은 하이라이트 구간과 겹치는 샷만 채워져 장면 이전은 NULL 이 많다.
그래서 보드 검출(t_transition.pitches)을 독립 재료로 함께 쓴다 — v201 전이 293건 중
246건(84%)에 값이 있다.
"""

import math
import re

from log import get_logger

log = get_logger(__name__)

# 시작을 **뒤로 미룰 수 있는** 최대 폭 — 유일한 시작 이동 방향이다 (모듈 docstring).
# 장면 경계가 전이 기준이라 앞 타석 꼬리를 무는 경우가 있다: v202 장면11(역전 홈런)은
# 장면이 2558s 에 시작하는데 그 홈런을 만든 투구는 2583s 다. 앞 25초는 심우준 타석의
# 견제 장면이라 "무슨 일인지 알 수 없는" 시작이었다.
START_MAX_FWD_SEC = 30
# 시작을 미뤄도 클립이 이보다 짧아지면 안 된다 — 플레이를 잘라 먹는다.
MIN_CLIP_SEC = 8
# 후보끼리 이보다 가까우면 **같은 지점**으로 본다.
# 근거: 해설 발화가 중앙 4.7~7.0초, 샷이 중앙 3.0~4.0초다(v200·201·202 실측).
# 후보가 그보다 촘촘하면 화면도 해설도 그 차이를 표현하지 못한다 — 실제로 v200
# 장면41 의 7702·7697 은 해설이 글자 그대로 같고 화면은 "포수 뒤쪽 기본 앵글에서…"
# 정형구라 판단에 쓸 수 없었다. 답이 없는 문제를 내면 모델은 판별자를 찾다가 사고를
# 태운다(장면61 94,575자·804초). 78초 클립의 시작이 7702냐 7697이냐는 보는 사람에게
# 같으므로, 정보 손실이 아니라 없는 선택지를 지우는 것이다.
MERGE_GAP_SEC = 5.0
# 끝을 늘릴 수 있는 최대 폭 (구 ENDFIX_MAX_EXT_SEC 계승).
END_MAX_EXT_SEC = 12
# 후보 개수 상한 — 많으면 모델이 고르지 못하고 프롬프트만 커진다.
CAND_MAX = 4
# 발화 끝과 샷 경계가 이 안에 함께 오면 "둘 다"로 표시한다 (가장 깔끔한 끝점).
COINCIDE_SEC = 2.0
# 후보로 쓰지 않는 샷 유형 — 광고 경계를 클립 끝으로 삼으면 중계가 아닌 데서 끊긴다.
# (v201 장면5 실측: 끝 후보 넷 중 하나가 광고 경계였다.)
SKIP_SHOT_TYPES = frozenset({"광고"})


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


def start_candidates(clip: dict, segs: list[dict], pitches: list[tuple[int, int]]) -> list:
    """시작 후보 [(초, 설명)] — **현재 시작 이후만**. 클립이 너무 짧아지는 값은 뺀다.

    되돌리는 방향은 내지 않는다: cs 는 장면 시작이고 장면 시작은 그 플레이의 투구보다
    이르거나 같으므로, cs 이전 후보는 전부 앞 플레이의 투구다 (모듈 docstring — 5경기
    29건 전수). 후보에 화면·해설이 붙으므로 모델이 "이건 앞 타석 견제다"를 읽고 가린다.
    """
    cs, ce = clip["cut"]["cs"], clip["cut"]["ce"]
    lo, hi = cs, min(cs + START_MAX_FWD_SEC, ce - MIN_CLIP_SEC)

    def near(sec: float) -> bool:
        # 현재 시작과 MERGE_GAP_SEC 이내면 같은 지점이라 대안이 아니다.
        return lo <= sec <= hi and abs(sec - cs) >= MERGE_GAP_SEC

    out: list[tuple[float, str]] = []
    for ps, _pe in pitches:                       # ① 보드 검출 투구 (분류 없이도 있다)
        if near(ps):
            out.append((float(ps), "보드 검출 투구"))
    for s in segs:                                 # ② 분류된 투구 샷 시작
        if near(s["s"]) and s.get("shot_type") == "투구":
            out.append((s["s"], "'투구' 샷 시작"))
    if not out:                                    # ③ 투구를 못 찾으면 타구·수비 시작
        for s in segs:
            if near(s["s"]) and s.get("shot_type") == "타구·수비":
                out.append((s["s"], "'타구·수비' 샷 시작"))
    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in out:
        uniq.setdefault(int(sec), (sec, why))
    # 현재 시작에 가까운 것부터 — 멀수록 **다음** 타석의 투구일 위험이 크다.
    return sorted(uniq.values(), key=lambda x: abs(x[0] - cs))[:CAND_MAX]


def end_candidates(clip: dict, segs: list[dict], utts: list) -> list:
    """끝 후보 [(초, 설명)] — 발화 끝과 샷 경계를 함께 준다.

    둘이 COINCIDE_SEC 안에 겹치면 그렇게 표시한다: 말도 끝나고 화면도 바뀌는 지점이
    가장 깔끔한 끝점인데, 지금까지는 그 정보가 프롬프트에 없어 모델이 알 수 없었다.

    **상한 절단은 시간순이 아니라 종류 우선**이다. 시간순으로 자르면 앞쪽에 몰린 화면
    전환이 자리를 다 먹고 뒤쪽의 발화 끝이 잘려나간다 — v201 장면5 실측: 후보가
    1355·1357·1359·1361(전부 화면 전환, 하나는 광고)만 남고 유일한 정답인 발화 끝
    1365 가 버려졌다. 규칙은 "해설이 끝나는 지점"인데 그걸 고를 수가 없었다.
    """
    ce = clip["cut"]["ce"]
    hi = ce + END_MAX_EXT_SEC
    utt_ends = [math.ceil(ue) for _us, ue, _t in utts if ce < ue <= hi]
    shot_ends = [s["e"] for s in segs
                 if ce < s["e"] <= hi and s.get("shot_type") not in SKIP_SHOT_TYPES]
    speech: list[tuple[float, str]] = []
    screen: list[tuple[float, str]] = []
    for ue in utt_ends:
        near = any(abs(ue - se) <= COINCIDE_SEC for se in shot_ends)
        speech.append((float(ue), "발화 끝 + 화면 전환" if near else "발화 끝"))
    for se in shot_ends:
        if not any(abs(ue - se) <= COINCIDE_SEC for ue in utt_ends):
            screen.append((float(se), "화면 전환"))
    # 발화 끝을 먼저 채우고 남는 자리만 화면 전환으로 (개수는 CAND_MAX 그대로).
    out = sorted(speech)[:CAND_MAX] + sorted(screen)[:max(0, CAND_MAX - len(speech))]
    uniq: dict[int, tuple[float, str]] = {}
    for sec, why in sorted(out):
        uniq.setdefault(int(sec), (sec, why))
    return sorted(uniq.values())[:CAND_MAX]


def build_rows(clips: list[dict], segs: list[dict], utts: list,
               pitches: list[tuple[int, int]]) -> list[dict]:
    """LLM 에 낼 행 — 후보가 하나도 없는 클립은 물어볼 게 없으므로 뺀다.

    **투구 앵커가 있는 클립은 아예 묻지 않는다** — cut 이 정한 시작이 이미 정답이고,
    LLM 은 그걸 되돌리기만 했다(모듈 docstring 참조). 그 대가로 그런 클립은 끝도
    cut 결과(대사 꼬리 스냅·관측 하한) 그대로 간다.

    후보마다 그 시각의 화면·해설을 붙이고, **현재 시작이 무엇인지도** 함께 싣는다.
    규칙에 "현재 시작이 이미 투구 직전이면 유지"가 있는데 후보는 cs 이전만 나열해서
    모델이 확인할 방법이 없었다 (v201 장면5: cs=1342 가 이미 '투구' 샷 시작인데
    28초 앞 앞선 타석 투구만 후보로 받았다).
    """
    rows = []
    for c in clips:
        if c["cut"].get("anchor") is not None:
            # 투구 앵커가 잡힌 클립은 물어보지 않는다. cut 이 이미 그 플레이의 투구
            # 샷에서 시작시켰고(실측 2026-08-20: v200 66/67 · v201 66/71 · v202 51/56
            # 이 '투구' 시작), 그걸 LLM 이 되돌리는 쪽이 손해였다 — v201 comp20 의
            # 장면9(-33s)·42(-37.3s)는 앵커를 무르고 앞 타석 꼬리로 돌아갔다.
            # bounds 가 필요한 건 앵커를 못 찾은 클립(홈런·판독 통째, pitch_sec NULL)이다.
            continue
        cs, ce = c["cut"]["cs"], c["cut"]["ce"]
        cur = _ctx(cs, segs, utts)
        at_start = _shot_at(cs, segs)
        cur["at_shot_start"] = bool(at_start and abs(at_start["s"] - cs) <= 1)
        # 시작: 현재와 구별 불가한 것은 빼고(답이 이미 "유지"다), 나머지를 접는다.
        # 접힐 때 남길 대표는 **현재와 해설이 다른 쪽**을 먼저 본다 — 시간만으로 고르면
        # 현재의 발화가 아직 이어지는 후보가 이기는데(v200 장면21: 3636 이 3640 을 밀어냄)
        # 정작 그 플레이를 말하는 건 뒤쪽이다("김성윤의 적시타가 나옵니다").
        cur_utt = cur.get("utt") or ""
        cands = [{"sec": sec, "why": why, "gap": round(sec - cs), **_ctx(sec, segs, utts)}
                 for sec, why in start_candidates(c, segs, pitches)]
        cands.sort(key=lambda x: ((x.get("utt") or "") == cur_utt, abs(x["sec"] - cs)))
        starts = sorted(_dedup(cands, drop_sig=_sig(cur), near=MERGE_GAP_SEC),
                        key=lambda x: abs(x["sec"] - cs))
        # 끝: 서명이 같으면 '발화 끝' 쪽을 남긴다 — 그게 규칙이 고르라는 지점이다.
        raw_ends = [{"sec": sec, "why": why, **_ctx(sec, segs, utts, at_end=True)}
                    for sec, why in end_candidates(c, segs, utts)]
        ends = sorted(_dedup(sorted(raw_ends,
                                    key=lambda e: (0 if "발화" in e["why"] else 1, e["sec"])),
                             near=MERGE_GAP_SEC),
                      key=lambda e: e["sec"])
        if starts or ends:
            rows.append({"scene_id": c["scene_id"], "tags": c["tags"],
                         "inning": c.get("inning") or "", "cs": cs, "ce": ce,
                         "cur": cur, "starts": starts, "ends": ends})
    return rows


# 숫자 뒤 단위 표기를 허용한다 — 모델마다 다르게 붙인다. Qwen3.8 은 "끝 9303s",
# Qwen3.6 은 "시작 1342초 끝 1355초" 로 답한다(2026-08-20 전환 실측). 단위를 안 받으면
# 줄 전체가 매칭에 실패해 **조용히 무시**되고, 로그엔 "경계 이동 0건"으로만 남아
# 모델 탓인지 파싱 탓인지 구분되지 않는다.
_UNIT = r"(?:초|s|sec)?"
_LINE = re.compile(
    rf"장면\s*(\d+)\s*[:：]\s*시작\s*(유지|\d+){_UNIT}\s*끝\s*(유지|\d+){_UNIT}")


def _match(cands: list[dict], sec: int) -> float | None:
    """모델이 답한 정수 초 → 그 후보의 **실제 초(소수 포함)**. 없으면 None.

    프롬프트는 후보를 정수로 보여주지만(모델에게 0.3초는 의미가 없다) 적용은 원래
    값으로 해야 한다. 정수로 되돌리면 샷 경계 바로 뒤였던 시작이 직전 샷으로 넘어간다
    — v201 은 t_segment 경계가 소수라 4369.3s('투구' 시작)가 4369s('리액션' 끝자락)가
    됐다. 후보는 int(sec) 로 중복 제거돼 있어 정수 1개당 후보 1건이다.
    """
    return next((c["sec"] for c in cands if int(c["sec"]) == sec), None)


def apply(clips: list[dict], rows: list[dict], text: str) -> list[str]:
    """제안 처분 — 제시한 후보의 초 값과 일치할 때만 적용. 그 외는 무시(임의 초 기각)."""
    shown = {r["scene_id"]: r for r in rows}
    by_id = {c["scene_id"]: c for c in clips}
    moved: list[str] = []
    for line in text.splitlines():
        m = _LINE.search(line.strip())
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in shown or sid not in by_id:
            continue
        row, cut_ = shown[sid], by_id[sid]["cut"]
        note = []
        if m.group(2) != "유지":
            new = int(m.group(2))
            if (hit := _match(row["starts"], new)) is not None:
                note.append(f"시작 {cut_['cs']:.1f}→{hit:.1f}")
                cut_["cs"] = hit
            else:
                log.info("bounds 기각: 장면%d 시작 %s (후보 밖)", sid, new)
        if m.group(3) != "유지":
            new = int(m.group(3))
            if (hit := _match(row["ends"], new)) is not None:
                note.append(f"끝 {cut_['ce']:.1f}→{hit:.1f}")
                cut_["ce"] = hit
            else:
                log.info("bounds 기각: 장면%d 끝 %s (후보 밖)", sid, new)
        if note:
            cut_["mode"] += "+경계보정"
            moved.append(f"장면{sid} " + " ".join(note))
    return moved
