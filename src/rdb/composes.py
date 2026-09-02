"""t_compose·t_compose_clip repository — 편성 결과 저장·조회 (compose 서비스 소유).

계약:
- PK 는 (v_id, comp_id) 복합 — comp_id 는 v_id 안에서 1부터 발급한다.
  발급은 create() 트랜잭션 안에서 MAX(comp_id)+1 (FOR UPDATE 로 같은 v_id 동시 접수 직렬화).
- 좌표는 정수 초 그대로 저장한다 (time 형·초 왕복 변환을 두지 않는다).
- status_code 는 t_code FK(4000번대) — ComposeStatus(이 파일 소유)가 미러다.
- 생애주기: 접수 시 create()가 헤더를 선-INSERT(PLAN) → 국면 전환마다 set_status() →
  종결 시 finish()가 클립 INSERT + 최종 코드·집계 UPDATE. 실패도 ERROR(4900) 행으로
  남는다 (이력 — 빈 결과도, 실패도 결과다).
"""

from enum import IntEnum

from asyncmy.cursors import DictCursor

from log import get_logger
from rdb.pool import Database

log = get_logger(__name__)


class ComposeStatus(IntEnum):
    """t_compose.status_code — DB t_code(4000번대 COMPOSE 대역)의 미러.

    status_code 는 t_code FK — 미등록 코드는 저장이 거부된다(신규 코드는 t_code 등록 선행).
    멤버 이름·값은 t_code 를 그대로 따른다 — 임의 개명 금지.
    코드는 그래프 노드가 아니라 **국면**이다 — 노드가 늘어도 값은 그대로다.
    4010(색인)은 agent-vision 소유가 되어 안 쓴다. 렌더 완료 스탬프 컬럼은 따로 없다 —
    렌더까지 끝나면 OK(4000)로 복귀한다 (t_code 설명대로 편성·렌더 공용 종결.
    재렌더는 언제든 허용 — 같은 comp_id 출력 덮어쓰기).
    """

    OK = 4000            # COMPOSE-OK — 편성(·렌더) 완료
    EMPTY = 4001         # COMPOSE-EMPTY — 조건 부합 장면 없음 (빈 편성 정상 종결)
    PLAN = 4020          # COMPOSE-PLAN — 질의 해석·선곡 중 (접수~select_clips)
    CUT = 4030           # COMPOSE-CUT — 클립 구간 확정 중 (select_end_point)
    VERIFY = 4040        # COMPOSE-VERIFY — 검수·예산 절단 중 (trim_budget~저장)
    RENDER = 4050        # COMPOSE-RENDER — mp4 렌더링 중 (워커 접수~완료 감시)
    ERROR = 4900         # COMPOSE-ERROR — 편성 실패 (사유는 잡·로그가 소유)
    ERROR_RENDER = 4950  # COMPOSE-ERROR-RENDER — 렌더 실패 (편성은 저장됨 — 재렌더 가능)


class ComposeRepo:
    """편성 결과 저장·조회 전담."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def create(self, v_id: int, query: str, budget_sec: int | None,
                     bumper: bool) -> int:
        """
        Summary:
            편성 헤더 선-INSERT — comp_id 를 발급해 반환한다 (status=PLAN).
        Args:
            v_id (int): 대상 영상 id.
            query (str): 사용자 질의 원문.
            budget_sec (int | None): 요청 목표 분량(초) — 미지정이면 NULL.
            bumper (bool): 렌더 시 이닝 그룹 사이 범퍼 삽입 여부 (Y/N 으로 저장).
        Returns:
            int: 발급된 comp_id (v_id 안에서 1부터).
        Description:
            - 접수 시점에 행을 만들어 진행 국면이 status_code 로 드러나게 한다.
              클립·집계는 finish() 가 채운다.
        """
        async with self._db.acquire() as conn:
            async with conn.cursor() as cur:
                # comp_id 발급 — 같은 v_id 의 동시 접수는 FOR UPDATE 가 직렬화한다
                await cur.execute(
                    "SELECT COALESCE(MAX(comp_id), 0) + 1 FROM t_compose "
                    "WHERE v_id = %s FOR UPDATE", (v_id,))
                (comp_id,) = await cur.fetchone()
                comp_id = int(comp_id)      # 집계 결과는 Decimal 로 온다

                await cur.execute(
                    "INSERT INTO t_compose (v_id, comp_id, query, budget_sec, "
                    "  status_code, bumper_yn) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (v_id, comp_id, query, budget_sec,
                     int(ComposeStatus.PLAN), "Y" if bumper else "N"))
            await conn.commit()
        log.info("t_compose 접수: v_id=%s comp_id=%s %r", v_id, comp_id, query)
        return comp_id

    async def set_status(self, v_id: int, comp_id: int,
                         status: ComposeStatus) -> None:
        """
        Summary:
            진행 국면 전환 — status_code 만 UPDATE 한다.
        Description:
            - 표시용 갱신이라 실패가 본 작업을 죽이면 안 된다 — 예외는 삼키고 로그만
              (fail-open 금지 원칙의 의도된 예외, rdb.videos.set_status 와 같은 결정).
        """
        try:
            async with self._db.acquire() as conn, conn.cursor() as cur:
                await cur.execute(
                    "UPDATE t_compose SET status_code = %s "
                    "WHERE v_id = %s AND comp_id = %s",
                    (int(status), v_id, comp_id))
            log.debug("status_code=%s 기록 (v_id=%s comp_id=%s)",
                      int(status), v_id, comp_id)
        except Exception as e:               # noqa: BLE001 — 표시용 갱신은 본 작업 비차단
            log.warning("status_code 갱신 실패 (v_id=%s comp_id=%s code=%s): %s",
                        v_id, comp_id, int(status), e)

    async def set_bumper(self, v_id: int, comp_id: int, bumper: bool) -> None:
        """
        Summary:
            bumper_yn 갱신 — 렌더 요청이 범퍼를 명시했을 때만 호출한다 (재렌더 시 변경 용도).
        """
        async with self._db.acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE t_compose SET bumper_yn = %s "
                "WHERE v_id = %s AND comp_id = %s",
                ("Y" if bumper else "N", v_id, comp_id))

    async def finish(self, v_id: int, comp_id: int, status: ComposeStatus,
                     clips: list[dict]) -> None:
        """
        Summary:
            편성 종결 — 클립 N행 INSERT + 최종 코드·집계 UPDATE (한 트랜잭션).
        Args:
            v_id (int): 대상 영상 id.
            comp_id (int): create() 가 발급한 편성 id.
            status (ComposeStatus): OK | EMPTY. empty 편성도 이력으로 남긴다.
            clips (list[dict]): {scene_no, start, end, tags, labels, inning} 시간순.
        """
        duration = 0
        for clip in clips:
            duration += clip["end"] - clip["start"]

        async with self._db.acquire() as conn:
            async with conn.cursor() as cur:
                if clips:
                    rows = []
                    for seq, clip in enumerate(clips, 1):
                        rows.append((v_id, comp_id, seq, clip["scene_no"],
                                     clip["start"], clip["end"],
                                     clip["tags"], clip["labels"], clip["inning"]))
                    await cur.executemany(
                        "INSERT INTO t_compose_clip (v_id, comp_id, clip_seq, "
                        "  scene_no, start_sec, end_sec, tags, labels, inning) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        rows)
                await cur.execute(
                    "UPDATE t_compose "
                    "SET status_code = %s, duration_sec = %s, clip_cnt = %s "
                    "WHERE v_id = %s AND comp_id = %s",
                    (int(status), duration, len(clips), v_id, comp_id))
            await conn.commit()
        log.info("t_compose 종결: v_id=%s comp_id=%s status=%s (%d클립, %ds)",
                 v_id, comp_id, status.name, len(clips), duration)

    async def fetch(self, v_id: int, comp_id: int) -> dict | None:
        """
        Summary:
            저장된 편성 재조회 — 헤더 + 클립 목록 (시간순).
        Args:
            v_id (int): 대상 영상 id — comp_id 가 v_id 안 시퀀스라 둘 다 필요하다.
            comp_id (int): 편성 id.
        Returns:
            dict | None: {v_id, comp_id, query, …, clips: [...]}. 없으면 None.
        Description:
            - 진행 중(4020~4040)·실패(4900) 행도 그대로 나온다 — status_code 가 국면이다.
        """
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM t_compose WHERE v_id = %s AND comp_id = %s",
                (v_id, comp_id))
            head = await cur.fetchone()
            if not head:
                return None
            await cur.execute(
                "SELECT clip_seq, scene_no, start_sec, end_sec, tags, labels, inning "
                "FROM t_compose_clip WHERE v_id = %s AND comp_id = %s "
                "ORDER BY clip_seq", (v_id, comp_id))
            clips = list(await cur.fetchall())

        head["reg_datetime"] = str(head.get("reg_datetime"))
        return {**head, "clips": clips}
