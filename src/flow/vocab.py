"""편성 어휘·랭크 가중·컷 레시피 — 도메인 상수의 단일 원천 (bench4 config 이식).

bench4 는 태그 어휘가 프롬프트 문자열(prompt.py)과 CUT_RECIPE 키(config.py)에
따로 하드코딩돼 드리프트가 확정 상태였다. 여기서는:
- 태그·라벨 어휘를 이 모듈이 소유하고, 프롬프트는 여기서 렌더한다.
- 원본은 vision3 `sports/baseball/vocab.py`(판정 어휘 SSOT) — 복제본이므로
  tests/test_vocab_sync.py 가 모노레포 상대 경로로 원본과 등가를 회귀 고정한다.
- CUT_RECIPE 등 태그를 키로 쓰는 상수는 부팅 시 `validate()` 로 어휘 존재를 검증
  (없는 태그 키 = 드리프트 → 즉시 실패).

각 상수의 실측 근거 주석은 bench4 원문을 그대로 옮겼다 — 근거 없이 값을 바꾸지 말 것.
"""

# ── 태그 어휘 (vision3 vocab.PLAY_TAGS 와 동기 — 판정이 낼 수 있는 행위 태그 전체) ──
PLAY_TAGS: tuple[str, ...] = (
    "안타", "번트", "홈런", "범타", "삼진", "볼넷", "사구",
    "도루", "폭투·포일", "견제",
    "실책", "호수비",
    "비디오 판독·리플레이", "판별불가",
)

# 파생 라벨 (publish 가 발행 시 계산 — t_scene.labels 에 등장 가능한 어휘)
# 나열 순서는 bench4 LABEL_VOCAB 원문 유지 — 프롬프트 byte 등가 게이트의 전제
LABELS: tuple[str, ...] = ("역전", "동점", "적시타", "희생플라이", "밀어내기", "병살", "도루사")

# ── rank 가중 (순수 계산 — 유사도·LLM 무관) ─────────────────────────────
RANK_SCORE_DELTA_W = 3
RANK_LABEL_BONUS = {"역전": 4, "동점": 2, "적시타": 1, "병살": 2}
RANK_TAG_BONUS = {"홈런": 4, "비디오 판독·리플레이": 2, "폭투·포일": 1, "호수비": 2}

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
    "실책": {"타구·수비", "주루"},
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
    "희생플라이": {"득점·홈인", "리액션"},
    "밀어내기": {"득점·홈인"},
}
LABEL_EXTRA_MAX_SEC = 12
# 대사 꼬리 스냅 상한 — 실측(comp 18·19): 삼진 7클립 중 5개가 발화 중간 절단,
# 필요 연장 +4~8.2s → 9s 면 전부 회수, 별개 문장(+11s)은 자연 차단.
DIALOGUE_TAIL_MAX_SEC = 9
# endfix 수용 상한 — 상한 밖 발화를 제시했더니 LLM 이 그걸 골라 제안 전멸(실측 comp 28).
ENDFIX_MAX_EXT_SEC = 12
ENDFIX_UTT_MAX = 4

# 완화 사다리 상한 — 미달 재선곡 폐기 근거: "다이빙 캐치"가 범타 13클립으로 희석(comp 28)
# + thinking 재선곡 콜 1회 절감(~2.5분).
MAX_REPLAN = 1
UNDERFILL_MIN_FRAC = 0.7


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
