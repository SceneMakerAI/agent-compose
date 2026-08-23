"""편성 어휘·랭크 가중·컷 레시피 — 도메인 상수의 단일 원천 (bench4 config 이식).

bench4 는 태그 어휘가 프롬프트 문자열(prompt.py)과 CUT_RECIPE 키(config.py)에
따로 하드코딩돼 드리프트가 확정 상태였다. 여기서는:
- 태그·라벨 어휘를 이 모듈이 소유하고, 프롬프트는 여기서 렌더한다.
- 원본은 vision3 `sports/baseball/vocab.py`(판정 어휘 SSOT) — 복제본이므로
  tests/test_flow_deterministic.py 의 `*_synced_with_vision3` 가 모노레포 상대
  경로로 원본과 등가를 회귀 고정한다.
- CUT_RECIPE 등 태그를 키로 쓰는 상수는 부팅 시 `validate()` 로 어휘 존재를 검증
  (없는 태그 키 = 드리프트 → 즉시 실패).

각 상수의 실측 근거 주석은 bench4 원문을 그대로 옮겼다 — 근거 없이 값을 바꾸지 말 것.
"""

# ── 태그 어휘 (vision3 vocab.PLAY_TAGS 와 동기 — 판정이 낼 수 있는 행위 태그 전체) ──
# 나열 순서 = vision3 vocab.TAGS 정의 순 (판정 프롬프트 선택지 순서) — 이 순서가
# 그대로 plan 프롬프트의 태그 어휘 줄이 된다.
PLAY_TAGS: tuple[str, ...] = (
    "홈런", "안타", "번트", "볼넷", "사구", "삼진", "범타",
    "도루", "폭투·포일", "견제", "보크",
    "수비실책", "호수비",
    "비디오 판독·리플레이", "판별불가",
)

# 파생 라벨 — vision3 `scene/judge.derive` 가 계산해 labels 컬럼에 행위 태그와
# 함께 담는 어휘 (2026-08-23 개편). '역전·동점·끝내기'는 여기서 빠졌다: 판세 변화는
# game_context 컬럼으로 이사했다 (GAME_CONTEXTS). '경기 종료'는 폐기된 publish 소유라
# 함께 사라졌다 — 지금은 어느 단계도 붙이지 않는다.
# '희생플라이'→'진루타' 개명: 전광판만으로는 뜬공/땅볼을 못 가르는데 희생플라이로
# 단정하면 땅볼 진루타가 오답이 된다 (v1003 실측).
LABELS: tuple[str, ...] = (
    "적시타", "진루타", "밀어내기", "병살", "삼중살", "도루사", "견제사",
)

# 점수 상황 변화 (t_scene_baseball.game_context — vision3 `scene/context.of` 산출).
# **단일값**이다: 우선순위대로 하나만 붙고, 판세가 안 바뀐 득점은 NULL 이다.
# 나열 순서 = context.of 의 판정 우선순위 (끝내기가 역전을 이긴다).
GAME_CONTEXTS: tuple[str, ...] = ("끝내기", "선제", "역전", "동점")

# ── rank 가중 (순수 계산 — 유사도·LLM 무관) ─────────────────────────────
RANK_SCORE_DELTA_W = 3
RANK_LABEL_BONUS = {"적시타": 1, "병살": 2}
RANK_TAG_BONUS = {"홈런": 4, "비디오 판독·리플레이": 2, "폭투·포일": 1, "호수비": 2}
# 판세 가중 — 구 RANK_LABEL_BONUS 의 역전·동점이 game_context 로 이사하며 갈라져 나왔다.
# 값은 그대로 옮긴다 (역전 4·동점 2): 구조를 바꾸면서 실측 숫자까지 흔들면 나중에
# 품질 변화의 원인을 못 가른다. '끝내기'는 context.of 에서 역전과 배타라 같은 값을
# 준다 — 역전의 특수형(경기가 그 자리에서 끝나는)이라 아래로 내려갈 이유가 없다.
# '선제'는 **미실측 초기값**이다. 판세를 여는 득점이라 0 은 아니되, 역전·동점만큼
# 결정적이지 않다는 판단 하나뿐 — 실측이 생기면 그 근거로 고칠 것.
RANK_CONTEXT_BONUS = {"끝내기": 4, "역전": 4, "동점": 2, "선제": 1}

# ── cut 샷 레시피 — 태그 → 앵커(투구) 이후 이어붙일 샷 유형 집합 ──────────
# 앵커 = ★pitch_sec 이 속한 샷 (유형 무관 — 긴 혼합 샷 대비), 없으면 구간 첫 '투구' 샷,
# 그것도 없으면 클립 통째(FULL 폴백 = 완화 L2 내장).
CUT_RECIPE: dict[str, set[str]] = {
    "안타": {"타구·수비", "주루"},
    "번트": {"타구·수비", "주루"},
    "범타": {"타구·수비"},
    "삼진": set(),                 # ★투구 샷 하나 — 스윙까지 그 안에 있다
    "볼넷": set(),
    "사구": set(),
    "도루": {"주루"},
    "폭투·포일": {"주루"},
    # 보크는 투구가 무효라 앵커 투구 샷만으로는 내용이 없다 — 진루가 곧 결과
    "보크": {"주루"},
    "수비실책": {"타구·수비", "주루"},
    "호수비": {"타구·수비", "리액션"},
}
# 과정이 내용 — 통째 사용. 홈런: 투구 뒤가 타구 비행(기타)·세리머니·리액션으로 이어져
# 레시피 체인이 첫 샷에서 끊긴다 (실측 v201 장면 9: 6초 홈런 사고).
FULL_CLIP_TAGS = {"비디오 판독·리플레이", "홈런"}
# 라벨 기반 추가 채택 — 큰 플레이(병살·적시타 등)는 여운까지. 득점 라벨에 '득점·홈인'을
# 넣는 근거(실측 장면 38): 득점 순간 방송이 스코어 그래픽을 숨겨 홈인 라이브가 과거
# '무전광판'으로 뭉개졌다. 추가 채택은 LABEL_EXTRA_MAX_SEC 상한까지만 (리플레이 오분류 방어).
LABEL_EXTRA_SHOTS: dict[str, set[str]] = {
    "병살": {"주루", "리액션"},
    "적시타": {"득점·홈인", "리액션"},
    "진루타": {"득점·홈인", "리액션"},
    "밀어내기": {"득점·홈인"},
}
LABEL_EXTRA_MAX_SEC = 12
# 대사 꼬리 스냅 상한 — 실측(comp 18·19): 삼진 7클립 중 5개가 발화 중간 절단,
# 필요 연장 +4~8.2s → 9s 면 전부 회수, 별개 문장(+11s)은 자연 차단.
DIALOGUE_TAIL_MAX_SEC = 9

# 완화 사다리 상한 — 미달 재선곡 폐기 근거: "다이빙 캐치"가 범타 13클립으로 희석(comp 28)
# + thinking 재선곡 콜 1회 절감(~2.5분).
MAX_REPLAN = 1


def validate() -> None:
    """태그를 키로 쓰는 상수의 어휘 존재 검증 — 드리프트는 부팅 실패로 드러낸다."""
    tags = set(PLAY_TAGS)
    labels = set(LABELS)
    for name, keys, base in [
        ("CUT_RECIPE", CUT_RECIPE.keys(), tags),
        ("RANK_TAG_BONUS", RANK_TAG_BONUS.keys(), tags),
        ("FULL_CLIP_TAGS", FULL_CLIP_TAGS, tags),
        ("RANK_LABEL_BONUS", RANK_LABEL_BONUS.keys(), labels),
        ("LABEL_EXTRA_SHOTS", LABEL_EXTRA_SHOTS.keys(), labels),
    ]:
        unknown = set(keys) - base
        if unknown:
            raise ValueError(f"vocab 드리프트: {name} 에 어휘 밖 키 {sorted(unknown)}")
    unknown = set(RANK_CONTEXT_BONUS) - set(GAME_CONTEXTS)
    if unknown:
        raise ValueError(f"vocab 드리프트: RANK_CONTEXT_BONUS 에 어휘 밖 키 {sorted(unknown)}")
    if overlap := set(PLAY_TAGS) & set(LABELS):
        # 두 어휘가 겹치면 split_labels 의 분해가 모호해진다 — 상류가 한 컬럼에
        # 섞어 담으므로 이름이 겹치는 순간 어느 축인지 알 수 없다.
        raise ValueError(f"vocab 드리프트: 태그·라벨 이름 충돌 {sorted(overlap)}")


def split_labels(text: str | None) -> tuple[list[str], list[str]]:
    """
    Summary:
        t_scene_baseball.labels 한 칸 → (행위 태그, 파생 라벨) 두 축으로 분해.
    Args:
        text (str | None): '안타,적시타' 같은 쉼표 나열. NULL·빈 칸이면 둘 다 빈 목록.
    Returns:
        tuple[list[str], list[str]]: (PLAY_TAGS 소속, LABELS 소속). 등장 순서 유지.
    Description:
        상류(vision3)는 2026-08-23 개편으로 두 축을 labels 한 칸에 함께 담는다
        (구 scene_type 폐기 — LLM 칸과 알고리즘 칸을 나눌 실익이 사라졌다는 결정).
        compose 는 두 축을 계속 나눠 쓴다 — rank 가중(RANK_LABEL_BONUS vs
        RANK_TAG_BONUS)도 cut 레시피(CUT_RECIPE vs LABEL_EXTRA_SHOTS)도 축마다
        다른 표를 보기 때문이다. 그래서 분해는 여기 어휘 소유 모듈이 한다.

        **어느 쪽에도 없는 값은 버리지 않고 파생 라벨로 넘긴다.** 상류에 라벨이
        새로 생겼는데 이 복제본이 안 따라간 경우가 그 자리인데, 조용히 지우면
        인벤토리 렌더에서 통째로 사라진다. 동기 테스트가 드리프트를 따로 잡는다.
    """
    tags, labels, known = [], [], set(PLAY_TAGS)
    for x in (t.strip() for t in (text or "").split(",")):
        if not x:
            continue
        (tags if x in known else labels).append(x)
    return tags, labels
