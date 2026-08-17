"""실행 엔트리포인트 — .env(config.Settings)의 APP_HOST/APP_PORT 로 uvicorn 서버를 띄운다.

바인딩(호스트·포트)을 CLI 인자가 아니라 .env 에서 읽는다 — 설정을 config.Settings 한 곳에
모으고(하드코딩 금지), 포트번호를 소스에 노출하지 않는다(레거시 run.py 계승).
"""

# 서드파티
import uvicorn

# 로컬
from config import get_settings


def main() -> None:
    """config(.env)의 app_host/app_port 로 app:app 을 서빙한다."""
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        log_config=None,  # 로깅은 app 의 setup_logging 가 구성 — uvicorn 기본과 중복 방지
        # SIGTERM 후 무기한 대기 금지 — 진행 중 백그라운드 플로우가 살아남아
        # 새 프로세스와 DB 를 경합 오염시킨 실사고(레거시) 방지. 상한 내 미종료 시 강제 취소.
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
