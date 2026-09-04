"""parse_query 노드 — 질의 해석: 프롬프트·파서·노드 구현이 이 한 파일에 있다.

"LLM 이 제안, 코드가 집행": LLM 은 이 경기에 실존하는 메타 어휘(EvidenceRepo.meta_vocab)
중에서만 필터를 고르고, 파서는 어휘 밖 값을 버린다 (사실이 모델을 이긴다).
render_user·parse 는 순수 함수라 단독 테스트 가능.

출력은 JSON 이 아니라 줄 형식(키: 값)이다 — JSON 보다 Qwen 파싱이 안정적이다(실측).
"""

from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.evidences import EvidenceRepo
from domains.baseball.repo.teams import TeamRepo
from infer.chat import ChatLLM
from log import get_logger

log = get_logger(__name__)


def make_node(llm: ChatLLM, evidence_repo: EvidenceRepo, team_repo: TeamRepo):
    """자원 주입 팩토리 — build.py 가 호출한다."""

    async def parse_query(st: ComposeState) -> dict:
        """질의 해석 — 이 경기의 메타 어휘 중에서 LLM 이 필터를 고른다.

        실패하면 필터 없음(전체 인벤토리)으로 진행한다 — 보조 단계가 편성을 죽이지
        않는다. 어휘 밖 값은 파서가 버린다.
        """
        vocab = await evidence_repo.meta_vocab(st["v_id"])
        team_aliases = await team_repo.fetch()
        try:
            text = await llm.chat(SYSTEM, render_user(st["query"], vocab, team_aliases),
                                  trace=st.get("trace"), name="parse_query")
            log.info("parse_query 응답: %r", text)
            spec = parse(text, vocab)
            # 검색어 속 팀 표기 → 별칭 묶음 확장 (표기 변형을 고르게 회수)
            spec["phrases"] = expand_team_phrases(spec["phrases"], team_aliases)
        except Exception as e:               # noqa: BLE001 — 보조 단계, 죽이지 않는다
            log.warning("parse_query 실패(필터 없이 진행): %s", e)
            spec = empty_spec()
        log.info("parse_query: %s", spec)
        return {"spec": spec}

    return parse_query

SYSTEM = """\
[역할]
당신은 야구 하이라이트 편성기의 질의 해석기입니다. 사용자 질의를 아래 필터 축으로 번역합니다.

[필터 축 — 뜻과 고르는 기준]
- 이닝: 경기 구간(회차). 질의가 특정 회차를 지정할 때만 고릅니다.
- 팀명: 질의가 언급한 팀입니다. 선택지는 "대표(별칭들)" 형식입니다 — 질의의 표기가
  별칭에 해당하면 반드시 **대표 표기**로 답합니다 (예: 선택지 "SSG(랜더스, 신세계)"
  에서 "랜더스" 질의 → SSG).
- 관점: 팀명을 골랐을 때 그 팀의 어느 쪽 장면을 묻는지입니다.
  "OO 타선"·"OO 득점"·"OO 공격" → 공격 / "OO 수비"·"OO 호수비" → 수비 /
  구분이 없으면("OO 하이라이트") → 없음. 팀명이 없으면 관점도 없음입니다.
- 라벨: 그 구간에서 일어난 플레이의 판정 결과 용어입니다 (안타·삼진·병살 등).
  질의가 가리키는 플레이 종류를 고릅니다 (예: "삼진 모음" → 삼진).
- 전광판: 전광판 수치 변화로 검출된 경기 사실입니다 (점수·주자 진루·아웃·이닝종료 등).
  질의가 가리키는 경기 사실을 고릅니다 (예: "선취점 장면" → 선취점).
- 검색어: STT(중계 멘트) 및 Vision(장면 묘사) Vector DB 검색용 **구체적 행위·상황 단어
  조합**입니다.
  - 실제 중계진이 말하거나 화면에 보이는 구체적 플레이/상황(예: 홈런 | 담장을 넘어갑니다 |
    다이빙 캐치 | 역전 적시타 | 김도영)으로 1~3개 만듭니다.
  - 질의의 핵심 대상(플레이 명사·선수명·팀명 등 고유명사)은 라벨·전광판에 이미
    담았더라도 검색어에 **중복 포함**합니다 — 검색어는 필터와 별개의 검색 채널입니다.
    (예: "홈런 장면" → 검색어: 홈런)
  - 팀명·선수명은 [선택지]에 없는 표기라도 검색어에 포함합니다 — 중계·자막은 같은
    팀을 다르게 부릅니다 (예: SSG = 신세계 랜더스 = 랜더스).
  - **[절대 금지]**: '하이라이트', '명장면', '모음', '장면', '영상' 등 임베딩 매칭률을
    떨어뜨리는 추상적 메타 단어는 절대 포함하지 않습니다.
  - '없음'은 위를 다 적용해도 남는 구체 단어가 없을 때만 씁니다 (예: "하이라이트 모음").
  - 선택지 제약이 없으며, 여러 개는 | 로 구분합니다.
- 분량: 질의가 완성본 **전체 길이**를 숫자로 지정했을 때만 그 값을 **초 단위 정수**로
  적습니다 (예: "5분짜리로" → 300, "3분 이내" → 180, "90초" → 90).
  - '짧게'·'핵심만'·'풀버전'처럼 숫자 없는 표현은 분량이 아닙니다 → '없음'.
    (그 취향은 다른 단계가 클립 하나하나의 길이로 반영합니다)
  - 한 장면의 길이가 아니라 편성 전체의 목표 길이입니다.
  - 지정이 없으면 '없음'.

[선택 원칙]
1. 각 축은 [선택지]에 제시된 값 중에서만 고릅니다. 목록에 없는 값은 절대 지어내지
   않습니다. (단, '검색어' 축은 제외)
2. 이닝·팀명은 범위를 **좁히는** 축입니다 — 질의가 지정할 때만 고르고, 아니면 "없음".
3. 라벨·전광판은 내용을 **넓게 모으는** 축입니다 — 질의 의도에 해당하는 값을 양쪽 모두
   빠짐없이 담습니다. 득점 장면을 묻는 질의면 득점성 라벨(예: 적시타, 홈런)과
   점수 전광판 태그(예: 1점, 2점)를 전부 고릅니다.
4. 용어 번역: 질의의 표현을 선택지 용어로 바꿉니다 (예: 더블플레이 → 병살, 포볼 → 볼넷).
5. 질의가 "하이라이트"·"명장면"처럼 사건 전반을 물으면:
   - 라벨/전광판: 볼·스트라이크 같은 단순 카운트를 제외한 사건 라벨들을 고릅니다.
   - 검색어: 추상적 단어('하이라이트', '명장면' 등)를 절대 추출하지 말고, 구체적
     묘사가 없다면 '없음'으로 출력합니다.
6. 수비 장면을 묻는 질의: 라벨은 아웃을 만드는 플레이(땅볼 아웃·플라이 아웃·태그 아웃·
   병살·삼진 등)를 고르고, 안타는 수비 실패이므로 고르지 않습니다.

[출력 형식 — 아래 7줄만 출력]
이닝: ...
팀명: ...
관점: 공격|수비|없음
라벨: ...
전광판: ...
검색어: ...
분량: 초 단위 정수|없음\
"""


def render_user(query: str, vocab: dict, team_aliases: dict[str, list[str]]) -> str:
    """질의 + 이 경기의 메타 어휘(선택지) → 유저 프롬프트.

    팀명 선택지는 **메타에 실제로 나온 값(대표)**만 나열한다 — 별칭은 괄호로 병기해
    LLM 이 질의의 다른 표기(랜더스 등)를 대표로 옮겨 답할 근거를 준다.
    """
    team_choices = []
    for team_id in vocab["teams"]:
        aliases = []
        for alias in team_aliases.get(team_id, []):
            if alias != team_id:            # 대표 자신은 괄호에서 뺀다
                aliases.append(alias)
        if aliases:
            team_choices.append(f"{team_id}({', '.join(aliases)})")
        else:
            team_choices.append(team_id)

    return (
        f"[질의]\n{query}\n\n"
        f"[선택지]\n"
        f"이닝: {', '.join(vocab['innings'])}\n"
        f"팀명: {', '.join(team_choices)}\n"
        f"라벨: {', '.join(vocab['labels'])}\n"
        f"전광판: {', '.join(vocab['board_tags'])}"
    )


PHRASES_MAX = 3     # 검색어 상한 — 검색 1회가 임베딩 1콜 + Milvus 1콜이다


def expand_team_phrases(phrases: list[str],
                        team_aliases: dict[str, list[str]]) -> list[str]:
    """
    Summary:
        검색어 속 팀 표기를 그 팀의 **별칭 묶음**으로 확장한다 (결정적 — 사전 기반).
    Description:
        - 실측: "SSG 랜더스 신세계"처럼 별칭을 한 검색어에 묶으면 표기 변형이
          고르게 회수된다 (stt 팀 언급 25/50). 대표 단독은 그 표기만 잡고(19/50),
          별칭을 검색어 여러 개로 나누면 콜 수만 늘고 잡음이 는다(16/103).
        - 검색어를 공백 단위로 훑어 사전의 표기와 일치하는 토큰만 치환한다
          (예: "랜더스 홈런" → "SSG 쓱 랜더스 신세계 … 홈런").
        - 영문 대소문자는 정규화해 비교한다 (kt → KT).
    """
    # 표기(정규화) → 그 팀의 별칭 묶음 문자열
    lookup = {}
    for team_id, aliases in team_aliases.items():
        joined = " ".join(aliases)
        lookup[team_id.upper()] = joined
        for alias in aliases:
            lookup[alias.upper()] = joined

    expanded = []
    for phrase in phrases:
        tokens = []
        for token in phrase.split():
            tokens.append(lookup.get(token.upper(), token))
        expanded.append(" ".join(tokens))
    return expanded


def empty_spec() -> dict:
    """필터 없음 스펙 — 해석 실패 시 폴백 (전체 인벤토리로 진행)."""
    return {
        "innings": [],
        "teams": [],        # 질의가 언급한 팀 (대표 표기) — 관점(view)과 조합해 매칭
        "view": "",         # 공격 | 수비 | ""(무지정 — 팀으로 좁히지 않음)
        "labels": [],
        "board_tags": [],
        "phrases": [],      # 키워드 검색어 — 비면 벡터 검색 생략
        "budget_sec": None,  # 질의가 지정한 목표 분량(초) — None 이면 분량 언급 없음.
                             # 실제 채택은 trim_budget 이 정한다 (API 파라미터가 우선)
    }


# 분량 판독 가드 — 이 범위 밖은 오독으로 보고 버린다. 잘못 읽은 값으로 편성을
# 잘라내는 쪽이, 분량 지정을 무시하는 쪽보다 나쁘다.
BUDGET_MIN_SEC = 10
BUDGET_MAX_SEC = 3 * 60 * 60


def parse_budget(body: str) -> int | None:
    """
    Summary:
        분량 줄 → 목표 분량(초). 판독 불가·범위 밖이면 None (절단 안 함).
    Args:
        body (str): 분량 줄의 값 부분 (예: "300", "5분", "없음").
    Returns:
        int | None: 초 단위 정수. 해석 불가면 None.
    Description:
        - 프롬프트는 초 단위 정수를 요구하지만, 모델이 단위를 붙여 답하는 경우가
          흔해 "분"·"시간" 표기를 환산한다 ("초"가 함께 있으면 그대로 초로 읽는다).
    """
    digits = "".join(ch for ch in body if ch.isdigit())
    if not digits:
        return None
    sec = int(digits)
    if "초" not in body:
        if "시간" in body:
            sec *= 3600
        elif "분" in body:
            sec *= 60
    if BUDGET_MIN_SEC <= sec <= BUDGET_MAX_SEC:
        return sec
    log.warning("parse_query 분량 범위 밖 — 무시: %r", body)
    return None


def parse(text: str, vocab: dict) -> dict:
    """
    Summary:
        응답(줄 형식) → 필터 스펙. 어휘 밖 값은 버리고 로그로 드러낸다.
    Args:
        text (str): LLM 응답 본문.
        vocab (dict): meta_vocab 결과 — 검산 기준 (이 안의 값만 통과).
    Returns:
        dict: {innings, teams, labels, board_tags, phrases: list[str], view: str,
            budget_sec: int | None}. 빈 목록 = 그 축 필터 안 함. teams 는 질의가
            언급한 팀(사실)이고 view(공격|수비|무지정)와 조합해 apply_spec 이
            공격/수비 구간을 계산한다.
            phrases 는 벡터 검색용 키워드 (자유 — 어휘 검산 없음).
            budget_sec 은 질의가 지정한 목표 분량 — 없으면 None.
    """
    spec = empty_spec()

    # 축 이름 → (스펙 키, 검산 어휘). 줄 접두어로 매칭한다.
    axes = {
        "이닝": ("innings", vocab["innings"]),
        "팀명": ("teams", vocab["teams"]),
        "라벨": ("labels", vocab["labels"]),
        "전광판": ("board_tags", vocab["board_tags"]),
    }

    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        axis, _, body = line.partition(":")
        axis = axis.strip()
        body = body.strip()

        # 관점 — 단일값 (공격/수비만 유효, 그 외는 무지정)
        if axis == "관점":
            if body.startswith("공격"):
                spec["view"] = "공격"
            elif body.startswith("수비"):
                spec["view"] = "수비"
            continue

        # 검색어 — 자유 문장이라 어휘 검산 없이 | 로 나눠 담는다 (상한 PHRASES_MAX)
        if axis == "검색어":
            phrases = []
            for phrase in body.split("|"):
                phrase = phrase.strip()
                if phrase and phrase != "없음":
                    phrases.append(phrase)
            spec["phrases"] = phrases[:PHRASES_MAX]
            continue

        # 분량 — 자유 숫자라 어휘 검산 없음. 판독 실패는 None(절단 안 함)으로 흘린다
        if axis == "분량":
            spec["budget_sec"] = parse_budget(body)
            continue

        if axis not in axes:
            continue
        key, known = axes[axis]

        # "없음" = 그 축 필터 안 함
        if body in ("", "없음"):
            continue

        # 콤마 나열을 낱개로 — 어휘에 실존하는 값만 통과 (지어낸 값은 버린다)
        picked = []
        ghosts = []
        for value in body.split(","):
            value = value.strip()
            if not value:
                continue
            if value in known:
                picked.append(value)
            else:
                ghosts.append(value)
        spec[key] = picked
        if ghosts:
            log.warning("parse_query 검산: 어휘 밖 값 제거 %s=%s", axis, ghosts)

    return spec
