"""애플리케이션 설정 — .env 에서 로드 (하드코딩 금지).

값 우선순위: 환경변수 > .env(프로젝트 루트) > 아래 기본값.
배포마다 다른 값(포트·접속정보 등)은 기본값 없이 필수 → 누락 시 부팅 실패(fail-fast).
도메인 상수(프롬프트·어휘·랭크 가중·컷 레시피)는 여기 두지 않는다 — domains/* 소유.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트(src/config.py → 부모의 부모). .env 를 절대경로로 고정해 실행 CWD 무관.
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 서버 바인딩(API 리슨) ---
    app_host: str = "127.0.0.1"
    app_port: int                       # 배포별 필수

    # --- DB (MariaDB) ---
    db_ip: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "sm_db"
    db_pw: str = ""
    db_name: str = "sm_db"
    db_pool_max: int = 30
    db_pool_recycle: int = 3600         # NAT 경유 dev 는 짧게(예: 240)

    # --- chat LLM (질의 해석·선곡 — openai 호환 /v1/chat/completions) ---
    llm_base_url: str                   # 예: http://{host}:{port}/v1 — 배포별 필수
    llm_model: str = "qwen"
    llm_timeout: float = 600.0          # 전송 상한 — thinking 가드(480s)보다 커야 한다
    llm_concurrency: int = 12           # 동시 전송 상한 — 서버 --max-num-seqs 와 맞춘다
    llm_thinking: bool = False           # 전역 스위치 — 끄면 전 콜 thinking 비활성
                                        # (켜는 콜은 코드가 정한다 — 현재 select_clips 만)
    # thinking 콜의 사고량(low/medium/xhigh — 서버가 정의). 값이 있으면 그 콜은 enable_thinking
    # 토글 대신 reasoning_effort 로 보낸다 — 서버(vLLM qwen)가 effort 로 사고량을
    # 정한다. 빈 값이면 기존 enable_thinking 방식 그대로.
    llm_reasoning_effort: str | None = None
    # 선곡 콜 1건의 입력 토큰 상한 — 초과하면 맵-리듀스(청크 병렬 선곡 → 합집합) 분기.
    # 컨텍스트 한도(131k)가 아니라 선곡 품질(후보 많으면 중간을 놓친다)·속도 기준의 값.
    select_tokens_max: int = 5000

    # --- 편성 마감(trim_budget) ---
    # 예산 여유율 — 허용 상한 = budget_sec × (1 + budget_margin). 예산에 딱 맞춰
    # 자르면 대체로 예산 '이하'에서 끝난다 — 목표 분량을 채우는 쪽이 편성 의도에
    # 가까워 살짝 넘기는 것을 허용한다 (예: 0.1 이면 300초 요청 → 330초까지).
    budget_margin: float = 0.1

    # --- embedding (질의 임베딩 — openai 호환 /v1/embeddings) ---
    # 색인(agent-vision)과 **같은 서버·모델**이어야 한다 — 다르면 벡터 공간이 어긋나
    # 검색이 조용히 무너진다.
    embed_base_url: str                 # 예: http://{host}:{port}/v1 — 배포별 필수
    embed_model: str = "qwen-embed"
    embed_timeout: float = 60.0

    # --- Milvus (증거 색인 — 쓰기는 agent-vision 소유, 여기는 검색·메타 조회만) ---
    milvus_uri: str                     # 예: http://{host}:{port} — 접속 불가는 readyz 가 노출
    milvus_db: str = "sm_db"            # 팀 관례: 컬렉션은 sm_db 데이터베이스 아래
    milvus_collection: str = "sm_sport_baseball"
    vector_top_k: int = 50              # 검색어·종류 1건당 상위 히트 수 — 넉넉히 받고
                                        # 같은 구간·같은 내용 반복은 dedup 이 걸러낸다

    # --- worker-render (편성 → mp4 렌더링 — GPU 워커) ---
    # readyz 프로브 제외 — GPU 야간 자동 중지로 서비스 전체가 not-ready 로 뒤집히는 오탐 방지.
    render_base_url: str                # 예: http://{host}:{port} — 배포별 필수
    render_timeout: float = 600.0       # 접수 전송 상한 + 백그라운드 감시 총 상한(초)
    render_poll_interval: float = 5.0   # 렌더 상태 폴링 주기(초)

    # --- 로깅 ---
    log_level: str = "INFO"
    log_path: str | None = None
    # LLM 트레이스 — 편성 실행마다 하위 디렉터리 {v_id}_{comp_id}/ 를 만들고 그 안에
    # 노드별 md 파일({v_id}_{comp_id}_{node}.md)을 쓴다. 빈 값이면 수집 자체를 안 한다.
    # 프롬프트가 커서 운영 로그에는 흘리지 않는다. 상대 경로는 레포 루트 기준.
    trace_dir: str | None = "logs"

    def __str__(self) -> str:
        """설정을 [key] = value 로 나열(디버깅·로깅용). 비밀 필드는 마스킹."""
        data = self.model_dump()
        for secret in ("db_pw",):           # 비밀 필드 추가 시 여기에 등록
            if data.get(secret):
                data[secret] = "***"
        width = max(len(k) for k in data)
        body = "\n".join(f"  [{k:<{width}}] = {v!r}" for k, v in data.items())
        return f"Settings(\n{body}\n)"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴 — 첫 호출에서 .env 로드, 이후 캐시."""
    return Settings()
