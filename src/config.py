"""애플리케이션 설정 — .env 에서 로드 (하드코딩 금지).

값 우선순위: 환경변수 > .env(프로젝트 루트) > 아래 기본값.
배포마다 다른 값(LLM·임베딩·Milvus·DB)은 기본값 없이 필수 → 누락 시 부팅 실패(fail-fast).
도메인 상수(프롬프트·랭크 가중·컷 레시피)는 여기 두지 않는다 — flow/ 모듈 소유
(agent-vision3 규칙 계승).

agent-compose 특성: LLM 은 chat 1대(text 전용 — 판정·선곡)와 embed 1대(색인·질의)로
역할이 갈린다. bench4 의 fail-open(Milvus 없어도 조용히 통과)은 서비스에선 폐기 —
접속 불가는 /readyz 가 드러낸다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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

    # --- chat LLM (선곡·끝보정·검수 — text 전용) ---
    llm_base_url: str                   # 예: http://host:8002/v1
    llm_model: str = "qwen"
    llm_timeout: float = 240.0
    llm_thinking: bool = True           # plan 노드만 thinking — flow 쪽에서 선별 적용

    # --- embedding (색인·질의 — openai 호환 /v1/embeddings) ---
    embed_base_url: str                 # 예: http://host:8003/v1
    embed_model: str = "qwen-embed"
    embed_timeout: float = 60.0
    embed_batch: int = 64               # 색인 배치 크기 — 문서 수백 건을 나눠 보낸다

    # --- Milvus ---
    milvus_uri: str                     # 예: http://host:19530 — 접속 불가는 readyz 가 노출
    milvus_db: str = "sm_db"            # 팀 관례: 컬렉션은 sm_db 데이터베이스 아래
    milvus_collection: str = "sm_scene_evidence"
    vector_top_k: int = 20              # 검색 상위 히트 수 (bench4 운영값)

    # --- DB (MariaDB) ---
    db_ip: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "sm_db"
    db_pw: str = ""
    db_name: str = "sm_db"
    db_pool_max: int = 30
    db_pool_recycle: int = 3600         # NAT 경유 dev 는 짧게(예: 240)

    # --- 로깅 ---
    log_level: str = "INFO"
    log_path: str | None = None

    @field_validator("llm_base_url", "embed_base_url", "milvus_uri")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        """끝 슬래시 제거 — 경로 이어 붙일 때 `//` 방지."""
        return v.rstrip("/")

    def __str__(self) -> str:
        """설정을 [key] = value 로 나열(디버깅·로깅용). 비밀(db_pw)은 마스킹."""
        data = self.model_dump()
        data["db_pw"] = "***" if data.get("db_pw") else ""
        width = max(len(k) for k in data)
        body = "\n".join(f"  [{k:<{width}}] = {v!r}" for k, v in data.items())
        return f"Settings(\n{body}\n)"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴 — 첫 호출에서 .env 로드, 이후 캐시."""
    return Settings()
