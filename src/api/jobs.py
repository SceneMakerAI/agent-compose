"""백그라운드 잡 공통 — 202 접수 패턴의 프로세스 내 상태. 라우터마다 레지스트리를
따로 들지 않고 여기로 모은다.

단일 프로세스 전제(워커 다중화 시 DB 락으로 교체). 프로세스 재시작 시 비워지므로
고아 락이 없다 — 대신 잡 이력도 함께 사라진다(영속 이력은 DB 소유: t_compose 등).
검사→등록 사이에 await 가 없어야 한다 — asyncio 단일 스레드 전제의 무락 가드.
"""

from collections.abc import Hashable

from log import get_logger

log = get_logger(__name__)


class RunningGuard:
    """키(v_id·comp_id 등) 단위 이중 실행 방지 — 진행 중 키 집합.

    사용 규약: try_acquire 가 True 일 때만 작업을 시작하고, 종료 경로(성공·실패·취소)
    에서 반드시 release 한다.
    """

    def __init__(self, name: str) -> None:
        """name 은 로그 식별용 (예: 'ingest', 'render')."""
        self._name = name
        self._keys: set[Hashable] = set()

    def try_acquire(self, key: Hashable) -> bool:
        """키 선점 시도 — 이미 진행 중이면 False (호출부가 409 로 변환)."""
        if key in self._keys:
            return False
        self._keys.add(key)
        return True

    def release(self, key: Hashable) -> None:
        """키 해제 — 종료 경로에서 반드시 호출 (미보유 키 해제는 무해)."""
        self._keys.discard(key)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._keys


class JobStore:
    """키((v_id, comp_id) 등) → 상태 dict 캐시 — 202 접수 후 폴링 조회용 (프로세스 수명).

    키는 호출부가 정한다 — 대상 식별자를 그대로 써서 별도 잡 ID 발급이 없다.
    완료 잡이 cap 을 넘으면 오래된 것부터 정리한다 — 진행 중(running) 잡은 지우지
    않는다.
    """

    def __init__(self, cap: int = 200, sweep: int = 50) -> None:
        """cap: 보관 상한. sweep: 초과 시 한 번에 정리할 완료 잡 수."""
        self._jobs: dict[Hashable, dict] = {}
        self._cap = cap
        self._sweep = sweep

    def create(self, key: Hashable, **fields) -> None:
        """새 잡 등록 — status='running' 으로 시작."""
        self._jobs[key] = {"status": "running", **fields}
        if len(self._jobs) > self._cap:
            done = [k for k, j in self._jobs.items() if j.get("status") != "running"]
            for k in done[:self._sweep]:
                self._jobs.pop(k, None)

    def get(self, key: Hashable) -> dict | None:
        """잡 상태 조회 — 없으면 None (호출부가 폴백·404 로 변환)."""
        return self._jobs.get(key)

    def replace(self, key: Hashable, job: dict) -> None:
        """잡 상태 교체 — 완료·실패 확정 시 결과 통째로 갱신 (부분 갱신이 아니라 교체다)."""
        self._jobs[key] = job
