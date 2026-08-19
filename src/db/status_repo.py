"""t_video.status_code 기록 — 파이프라인 상태 코드(4000번대, t_code 사전) 갱신.

대역 규약 (t_code): 4000번대 = compose 서비스 소유. result -1=에러 / 0=성공 / 1=진행중.
4000 = 전체 수행 완료, 49xx = 에러. UI 는 t_video.status_code 를 t_code 와 조인해
이름·설명을 노출한다 (STT 2000번대·VISION 3000번대와 동일 방식).

comment 는 에러 상세 전달용 — 에러 코드를 쓸 때만 함께 갱신한다 (평시 덮어쓰기 금지).
"""

from db.pool import Database
from log import get_logger

log = get_logger(__name__)

# ── 4000번대 코드 (t_code 등록본과 1:1 — 값 변경 시 t_code 도 함께) ──
COMPOSE_OK = 4000            # 전체 수행 완료 (색인 완료·편성/렌더 완주 공용 종결)
COMPOSE_EMPTY = 4001         # 조건 부합 장면 없음 (빈 편성 정상 종결)
COMPOSE_INGEST = 4010        # 증거 색인 중
COMPOSE_PLAN = 4020          # 장면 선곡 중 (retrieve·plan·재선곡)
COMPOSE_CUT = 4030           # 클립 구성 중 (cutrank·backfill·endfix)
COMPOSE_VERIFY = 4040        # 편성 검수 중
COMPOSE_RENDER = 4050        # mp4 렌더링 중
COMPOSE_ERROR = 4900         # 편성 실패 (일반)
COMPOSE_ERROR_SOURCE = 4910  # 발행본 없음 (publish 선행 필요)
COMPOSE_ERROR_INGEST = 4920  # 색인 실패
COMPOSE_ERROR_RENDER = 4950  # 렌더 실패 (편성은 저장됨 — 재렌더 가능)
COMPOSE_ERROR_STAMP = 4960   # mp4 는 생성됐으나 t_compose 완료 기록 실패 (수동 확인 필요)


class StatusRepo:
    """t_video 상태 코드 갱신 전담 (compose 서비스의 유일한 t_video 쓰기 지점)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def set(self, v_id: int, code: int, comment: str | None = None) -> None:
        """
        Summary:
            t_video.status_code 를 갱신한다 (comment 는 에러 상세 시에만).
        Args:
            v_id (int): 대상 영상 id.
            code (int): 4000번대 상태 코드 (위 상수).
            comment (str | None): 에러 상세 (200자 절단). None 이면 comment 미변경.
        Description:
            - 상태 갱신 실패가 본 작업을 죽이면 안 되므로 예외는 삼키고 로그만 남긴다
              (표시용 부가 정보 — fail-open 금지 원칙의 예외로 의도된 결정).
        """
        try:
            async with self._db.acquire() as conn, conn.cursor() as cur:
                if comment is None:
                    await cur.execute(
                        "UPDATE t_video SET status_code = %s WHERE v_id = %s", (code, v_id))
                else:
                    await cur.execute(
                        "UPDATE t_video SET status_code = %s, comment = %s WHERE v_id = %s",
                        (code, comment[:200], v_id))
            log.debug("status_code=%s 기록 (v_id=%s)", code, v_id)
        except Exception as e:                       # noqa: BLE001 — 표시용 갱신은 본 작업 비차단
            log.warning("status_code 갱신 실패 (v_id=%s, code=%s): %s", v_id, code, e)
