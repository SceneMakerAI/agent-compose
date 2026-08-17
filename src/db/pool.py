"""DB 연결 계층 — MariaDB(asyncmy) 커넥션 풀.

앱 수명 동안 풀 하나를 공유한다(InferPool 과 동일 패턴, app.state.db).
이 모듈은 도메인을 모른다 — SQL 실행 통로(커넥션)만 내어주고,
실제 쿼리는 repository(videos 등)가 갖는다(전송/도메인 분리, 레거시 persistence/db.py 계승).
"""

# 서드파티
import asyncmy

# 로컬
from config import Settings
from log import get_logger

log = get_logger(__name__)


class Database:
    """
    Summary:
        MariaDB 비동기 커넥션 풀 래퍼.
    Description:
        - asyncmy 풀 하나를 앱 수명 동안 재사용한다(요청마다 연결하지 않음).
        - 커서 종류(DictCursor 등)는 호출하는 repository 가 정한다.
        - 동시 실행 상한은 풀(maxsize)이 담당한다.
    """

    def __init__(self, pool: asyncmy.Pool) -> None:
        """
        Summary:
            생성된 커넥션 풀을 보관한다.
        Args:
            pool (asyncmy.Pool): 이미 생성된 커넥션 풀.
        """
        self._pool = pool

    @classmethod
    async def connect(cls, settings: Settings) -> "Database":
        """
        Summary:
            Settings 의 접속 정보로 커넥션 풀을 만들어 Database 를 반환한다.
        Args:
            settings (Settings): 애플리케이션 설정(DB 접속 정보).
        Returns:
            Database: 풀이 준비된 인스턴스.
        Description:
            - lifespan 에서 app.state.db 에 보관할 인스턴스를 만드는 진입점.
            - autocommit=True — 단건 UPDATE 위주라 명시 트랜잭션 없이 운용한다
              (단건 UPDATE 7.8ms 실측 — poc-vision-flow).
            - pool_recycle 은 NAT 경유 dev 에서 짧게(240s) — 유휴 커넥션 강제 종료 대비.
        """
        pool = await asyncmy.create_pool(
            host=settings.db_ip,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_pw,
            db=settings.db_name,
            charset="utf8mb4",
            autocommit=True,
            minsize=1,
            maxsize=settings.db_pool_max,
            pool_recycle=settings.db_pool_recycle,
        )
        log.info("DB 풀 생성: %s@%s:%s/%s (max=%d)",
                 settings.db_user, settings.db_ip, settings.db_port,
                 settings.db_name, settings.db_pool_max)
        return cls(pool)

    def acquire(self):
        """
        Summary:
            풀에서 커넥션 하나를 빌리는 컨텍스트매니저를 돌려준다.
        Returns:
            asyncmy 커넥션 컨텍스트매니저 (`async with db.acquire() as conn:`).
        Description:
            - repository 가 `async with db.acquire() as conn:` 로 사용한다.
            - with 블록을 벗어나면 커넥션은 풀로 반납된다.
        """
        return self._pool.acquire()

    async def ping(self) -> bool:
        """
        Summary:
            DB 가 응답 가능한 상태인지 확인한다(SELECT 1).
        Returns:
            bool: 정상이면 True, 예외면 False.
        Description:
            - /readyz 프로브 등에서 호출한다. 예외는 삼키고 False 로 변환.
        """
        try:
            async with self._pool.acquire() as conn, conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
            return True
        except Exception as e:  # noqa: BLE001 — 연결 불가·인증 실패 등 종류 불문 False 가 목적
            log.warning("DB ping 실패: %s", e)
            return False

    async def close(self) -> None:
        """
        Summary:
            커넥션 풀을 닫는다.
        Description:
            - lifespan shutdown 에서 호출해 모든 커넥션을 정리한다.
        """
        self._pool.close()
        await self._pool.wait_closed()
