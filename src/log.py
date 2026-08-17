"""
로깅 설정과 헬퍼 — agent-vision src/log.py 관례 계승 (RotatingFileHandler 10MB×5).

agent-vision3 추가분: v_id 컨텍스트 태깅.
서비스는 여러 v_id 분석 플로우를 동시에 돌리므로(BackgroundTasks),
플로우 엔진이 bind_v_id()로 묶어 두면 그 코루틴 안의 모든 로그 줄에
[v123] 프리픽스가 자동으로 붙는다. contextvars 기반이라 asyncio 태스크 간에 섞이지 않는다.
"""

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

# v_id 태그가 %(v_tag)s 자리에 들어간다. 미바인딩 시 빈 문자열이라 포맷이 흐트러지지 않는다.
_FORMAT = "%(asctime)s[%(levelname)s] %(v_tag)s%(filename)s:%(lineno)d | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 앱이 DEBUG 여도 저수준 통신 로그를 도배하는 서드파티 로거 — WARNING 으로 눌러 둔다.
# httpx 도 포함 — LOG_LEVEL=DEBUG(프롬프트 덤프)일 때 "HTTP Request 200" 줄이 프롬프트를 묻지 않게.
_NOISY_LOGGERS = ("httpcore", "httpx")

# 현재 asyncio 태스크(=분석 플로우 1건)에 묶인 v_id. None 이면 태그 생략.
_v_id_ctx: ContextVar[int | None] = ContextVar("v_id", default=None)


class _VidFilter(logging.Filter):
    """모든 레코드에 v_tag 필드를 주입한다 (바인딩 없으면 빈 문자열)."""

    def filter(self, record: logging.LogRecord) -> bool:
        v_id = _v_id_ctx.get()
        record.v_tag = f"[v{v_id}] " if v_id is not None else ""
        return True


@contextmanager
def bind_v_id(v_id: int):
    """
    Summary:
        with 블록(보통 플로우 엔진의 실행 단위) 안의 모든 로그에 [v{v_id}] 태그를 붙인다.
    Args:
        v_id (int): 분석 대상 영상 id.
    Description:
        - contextvars 기반이라 동시 실행 중인 다른 v_id 플로우의 로그와 섞이지 않는다.
        - 블록을 벗어나면 이전 상태로 복원된다 (중첩 안전).
    """
    token = _v_id_ctx.set(v_id)
    try:
        yield
    finally:
        _v_id_ctx.reset(token)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Summary:
        루트 로거를 1회 구성한다. app 부팅 시점에 호출.
    Args:
        level (str): 로그 레벨 (예: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file (str | None): 로그 파일 경로. None 이면 파일 로깅 비활성화.
    Description:
        - 콘솔(stdout)은 항상 출력.
        - log_file 을 주면 같은 포맷으로 파일에도 동시 기록(콘솔+파일).
        - 파일은 RotatingFileHandler 로 10MB×5개 순환 — 무한 증가 방지.
        - _NOISY_LOGGERS(httpcore 등)는 WARNING 으로 눌러 DEBUG 도배를 막는다.
        - 모든 핸들러에 v_id 태그 필터를 걸어 bind_v_id() 태깅을 지원한다.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # logs/ 자동 생성
        handlers.append(
            RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
        )

    vid_filter = _VidFilter()
    for h in handlers:
        h.addFilter(vid_filter)  # 필터는 핸들러에 걸어야 전파 로그에도 적용된다

    logging.basicConfig(
        level=level.upper(),
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,  # 재호출(reload) 시 핸들러 깨끗이 재구성
    )

    # 서드파티 저수준 통신 로거는 한 단계 눌러 DEBUG 도배 방지(앱 DEBUG 와 무관하게).
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Summary:
        모듈용 로거를 반환한다.
    Args:
        name (str): 로거 이름. 보통 호출 모듈의 __name__.
    Returns:
        logging.Logger: 해당 이름의 로거 인스턴스.
    Description:
        - setup_logging 으로 구성한 루트 로거의 핸들러·포맷을 그대로 상속한다.
        - 모듈마다 get_logger(__name__) 으로 받으면 로그에 모듈명이 찍힌다.
    """
    return logging.getLogger(name)
