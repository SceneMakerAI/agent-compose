"""렌더 라우트 — 저장된 편성 1건을 worker-render 로 mp4 렌더링 (비동기 접수).

- POST 는 워커에 접수만 하고 202 를 돌려준다 (렌더는 GPU 수 분 — 호출자를 잡아두지 않는다).
  렌더 국면은 t_compose.status_code 가 소유한다: 접수 시 4050(RENDER),
  완료면 4000(OK) 복귀, 실패·타임아웃이면 4950(ERROR_RENDER).
- 렌더 가능은 status_code 가 OK·ERROR_RENDER 일 때만 — 재렌더는 언제든 허용한다
  (같은 comp_id 출력 덮어쓰기). empty·편성 진행 중·편성 실패는 워커 호출 전에 409 로
  사전 차단한다 — 조용히 빈/어긋난 렌더를 만드는 것보다 드러내는 게 원칙.
- 완료 감지는 두 경로: 접수 직후 뜨는 백그라운드 폴러(정상 경로), 그리고
  GET /render 조회 시 status_code 보정(복구 경로 — 재기동으로 폴러가 유실돼도
  조회 한 번이면 되살아난다).
"""

import asyncio
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, Request, status
from pydantic import BaseModel

from api.errors import (
    ComposeNotFoundError,
    ComposeNotRenderableError,
    InvalidInningError,
    RenderInProgressError,
    RenderWorkerError,
)
from api.jobs import RunningGuard
from log import bind_v_id, get_logger
from rdb.composes import ComposeRepo, ComposeStatus
from render.payload import build_request

log = get_logger(__name__)
router = APIRouter(tags=["render"])

_guard = RunningGuard("render")     # 이 프로세스가 감시 중인 (v_id, comp_id)
_POLL_FAIL_MAX = 5                  # 연속 조회 실패 상한 (일시 장애는 넘기고, 지속되면 포기)


class RenderRequest(BaseModel):
    """렌더 요청 — 대상 편성과 범퍼 옵션."""

    v_id: int
    comp_id: int
    # 이닝 그룹 사이 범퍼. 생략하면 편성 헤더의 bumper_yn 을 쓰고,
    # 명시하면 헤더를 갱신해 쓴다 (재렌더 시 범퍼 변경 용도).
    bumper: bool | None = None


@router.post("/render", status_code=status.HTTP_202_ACCEPTED)
async def post_render(req: RenderRequest, request: Request,
                      background: BackgroundTasks) -> dict:
    """
    Summary:
        (v_id, comp_id) 편성을 worker-render 에 접수 — 202 반환, 완료는 백그라운드 감시.
    Returns:
        dict: {v_id, comp_id, status_code, ...워커 접수 응답}.
    """
    st = request.app.state
    repo = ComposeRepo(st.db)
    comp = await _renderable(st, repo, req.v_id, req.comp_id)

    # 범퍼 확정 — 명시가 오면 헤더를 갱신해 다음 재렌더의 기본값도 바뀐다
    if req.bumper is None:
        bumper = comp["bumper_yn"] == "Y"
    else:
        bumper = req.bumper
        await repo.set_bumper(req.v_id, req.comp_id, bumper)

    try:
        payload = build_request(req.v_id, req.comp_id, comp["clips"], bumper)
    except ValueError as e:
        # 이닝 결손 = 상류 발행 데이터 결함 — 렌더로 덮지 않고 드러낸다
        raise InvalidInningError(str(e), v_id=req.v_id, comp_id=req.comp_id) from e

    # 검사 통과 즉시 선점 — 아래 await 마다 양보 지점이라, 워커 왕복 뒤에 등록하면
    # 그 사이 들어온 같은 편성 요청이 위 검사를 그대로 통과한다(중복 렌더)
    if not _guard.try_acquire((req.v_id, req.comp_id)):
        raise RenderInProgressError(v_id=req.v_id, comp_id=req.comp_id)
    try:
        with bind_v_id(req.v_id):
            log.info("렌더 접수 요청: comp_id=%s 클립 %d건 이닝 %s bumper=%s",
                     req.comp_id, len(comp["clips"]), list(payload["innings"]), bumper)
            await repo.set_status(req.v_id, req.comp_id, ComposeStatus.RENDER)
            try:
                result = await st.render.render(payload)
            except httpx.HTTPError as e:
                log.error("렌더 접수 실패: comp_id=%s %s: %s",
                          req.comp_id, type(e).__name__, e)
                await repo.set_status(req.v_id, req.comp_id, ComposeStatus.ERROR_RENDER)
                raise RenderWorkerError(f"{type(e).__name__}: {e}",
                                        v_id=req.v_id, comp_id=req.comp_id) from e
            log.info("렌더 접수 응답: %s", result)
    except BaseException:
        # 감시가 뜨기 전에 실패했으면 여기서 풀어야 한다(_watch 의 finally 가 못 돈다)
        _guard.release((req.v_id, req.comp_id))
        raise
    background.add_task(_watch, request, req.v_id, req.comp_id)
    return {"v_id": req.v_id, "comp_id": req.comp_id,
            "status_code": int(ComposeStatus.RENDER), **result}


@router.get("/render")
async def get_render(v_id: int, comp_id: int, request: Request) -> dict:
    """
    Summary:
        렌더 상태 조회 — 워커 상태를 그대로 노출하고, 렌더 중(4050) 편성은 보정한다.
    Returns:
        dict: {v_id, comp_id, status_code, status, ...}. status 는 워커의
            accepted / running / done / error 또는 not_requested / unknown.
    Description:
        - 재기동으로 폴러가 유실돼도 이 조회가 done/error 를 status_code 에 확정한다.
    """
    st = request.app.state
    repo = ComposeRepo(st.db)
    comp = await repo.fetch(v_id, comp_id)
    if comp is None:
        raise ComposeNotFoundError(v_id=v_id, comp_id=comp_id)
    code = comp["status_code"]

    try:
        res = await st.render.status(v_id, comp_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # 워커가 모르는 건 = 접수된 적 없음 (미렌더 편성의 정상 응답).
            # 렌더 중(4050)으로 남아 있다면 고아 — 실패로 확정한다
            if code == int(ComposeStatus.RENDER):
                await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
                code = int(ComposeStatus.ERROR_RENDER)
            return {"v_id": v_id, "comp_id": comp_id, "status_code": code,
                    "status": "not_requested"}
        return _unknown(v_id, comp_id, code, e)
    except httpx.HTTPError as e:
        # 워커 미기동(GPU 야간 중지 등) — 감추지 않고 unknown 으로 드러낸다
        return _unknown(v_id, comp_id, code, e)

    # 렌더 중으로 남은 편성만 보정 — 종결 코드는 조회가 덮지 않는다
    if code == int(ComposeStatus.RENDER):
        if res.get("status") == "done":
            await repo.set_status(v_id, comp_id, ComposeStatus.OK)
            code = int(ComposeStatus.OK)
        elif res.get("status") == "error":
            await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
            code = int(ComposeStatus.ERROR_RENDER)
    return {"v_id": v_id, "comp_id": comp_id, "status_code": code, **res}


async def _renderable(st, repo: ComposeRepo, v_id: int, comp_id: int) -> dict:
    """렌더 가능 검사 — 통과하면 편성(클립 포함)을 돌려준다.

    404(편성 없음) / 409(진행 중 — 고아면 워커에 물어 정정 후 진행) /
    409(empty·편성 중·편성 실패). 재렌더(OK·ERROR_RENDER)는 허용.
    """
    comp = await repo.fetch(v_id, comp_id)
    if comp is None:
        raise ComposeNotFoundError(v_id=v_id, comp_id=comp_id)
    code = comp["status_code"]

    if code == int(ComposeStatus.RENDER):
        # 이 프로세스가 감시 중이면 진짜 진행 중. 아니면 재기동으로 감시가 끊긴
        # 고아일 수 있다 — 워커에게 물어 정정하고, 판단 불가면 막는 쪽으로 둔다
        # (실제로 도는 렌더에 같은 출력 경로로 한 번 더 보내는 것이 더 나쁘다)
        if (v_id, comp_id) in _guard or not await _reconcile(st, repo, v_id, comp_id):
            raise RenderInProgressError(v_id=v_id, comp_id=comp_id)
        comp = await repo.fetch(v_id, comp_id)
        code = comp["status_code"]

    if code not in (int(ComposeStatus.OK), int(ComposeStatus.ERROR_RENDER)):
        raise ComposeNotRenderableError(
            f"status_code={code} 클립 {len(comp['clips'])}건 — 렌더 대상 아님",
            v_id=v_id, comp_id=comp_id)
    return comp


async def _reconcile(st, repo: ComposeRepo, v_id: int, comp_id: int) -> bool:
    """렌더 중(4050)으로 남은 편성을 워커에 물어 정정 — True 면 더는 진행 중이 아니다."""
    try:
        res = await st.render.status(v_id, comp_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            log.warning("렌더 고아 정정: comp_id=%s — 워커가 모르는 작업(실패 처리)", comp_id)
            await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
            return True
        return False
    except httpx.HTTPError:
        return False            # 워커 미기동 등 — 판단 불가면 막는다

    worker_status = res.get("status")
    if worker_status in ("accepted", "running"):
        return False
    if worker_status == "done":
        await repo.set_status(v_id, comp_id, ComposeStatus.OK)
        return True
    log.warning("렌더 고아 정정: comp_id=%s — 워커 상태 %s (실패 처리)",
                comp_id, worker_status)
    await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
    return True


def _unknown(v_id: int, comp_id: int, code: int, e: Exception) -> dict:
    """워커 상태 조회 실패 — 상태를 지어내지 않고 unknown 으로 반환."""
    log.warning("렌더 상태 조회 실패: comp_id=%s %s: %s", comp_id, type(e).__name__, e)
    return {"v_id": v_id, "comp_id": comp_id, "status_code": code,
            "status": "unknown", "error": f"{type(e).__name__}: {e}"}


async def _watch(request: Request, v_id: int, comp_id: int) -> None:
    """백그라운드 폴러 — 워커에 주기적으로 물어 done/error 를 status_code 에 확정한다.

    타임아웃·연속 조회 실패는 4950 으로 남긴다 — 워커가 나중에 끝냈다면
    GET /render 조회의 보정 경로가 되살린다.
    """
    st = request.app.state
    repo = ComposeRepo(st.db)
    deadline = time.monotonic() + st.settings.render_timeout
    fails = 0
    try:
        with bind_v_id(v_id):
            while time.monotonic() < deadline:
                await asyncio.sleep(st.settings.render_poll_interval)
                try:
                    res = await st.render.status(v_id, comp_id)
                except httpx.HTTPError as e:
                    fails += 1
                    log.warning("렌더 상태 조회 실패(%d/%d): comp_id=%s %s",
                                fails, _POLL_FAIL_MAX, comp_id, e)
                    if fails >= _POLL_FAIL_MAX:
                        await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
                        return
                    continue
                fails = 0
                worker_status = res.get("status")
                if worker_status == "done":
                    log.info("렌더 완료: comp_id=%s %s", comp_id, res.get("output_path"))
                    await repo.set_status(v_id, comp_id, ComposeStatus.OK)
                    return
                if worker_status == "error":
                    log.error("렌더 실패: comp_id=%s %s", comp_id, res.get("error"))
                    await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
                    return
            log.error("렌더 감시 타임아웃: comp_id=%s (%.0f초)",
                      comp_id, st.settings.render_timeout)
            await repo.set_status(v_id, comp_id, ComposeStatus.ERROR_RENDER)
    finally:
        # 취소(종료 등)로 빠져나가도 진행 중 표시는 반드시 푼다
        _guard.release((v_id, comp_id))
