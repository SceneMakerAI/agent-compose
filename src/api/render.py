"""렌더 라우트 — 저장된 편성(comp_id) 1건을 worker-render 로 mp4 렌더링 (비동기 접수).

POST 는 워커에 sync_yn=False 로 접수만 하고 202 를 돌려준다 (렌더는 GPU 수 분 —
호출자를 잡아두지 않는다). 완료 감지는 두 경로:
  1) 접수 직후 뜨는 백그라운드 폴러가 워커에 주기적으로 물어 done 이면 스탬프 (정상 경로)
  2) GET /render/{comp_id} 조회 시 미스탬프면 그 자리에서 워커에 물어 보정 (복구 경로 —
     배포·재기동으로 폴러가 유실돼도 조회 한 번이면 되살아난다)
성공 시 t_compose.render_datetime 에 완료 시각을 남긴다 — 뷰어(ui-sbs-viwer)가
"영상 준비됨" 배지와 중복 렌더 차단을 이 한 컬럼으로 판정하고, 뷰어 DB 계정은
SELECT 전용이라 쓰기는 이쪽이 소유한다.
empty 편성·이닝 결손·이미 렌더된 편성·진행 중 편성은 워커 호출 **전에** 이쪽에서 4xx 로
차단한다 — 조용히 빈/어긋난/중복 렌더를 만드는 것보다 사전에 드러내는 게 원칙
(fail-open 금지 계승).
"""

import asyncio
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from db.status_repo import COMPOSE_ERROR_RENDER, COMPOSE_ERROR_STAMP, COMPOSE_OK, COMPOSE_RENDER
from log import bind_v_id, get_logger
from render.payload import build_request

log = get_logger(__name__)
router = APIRouter()

_RENDERING: dict[int, int] = {}     # comp_id → v_id (진행 중 — 단일 워커 전제, ingest 와 동일)
_POLL_FAIL_MAX = 5                  # 연속 조회 실패 상한 (일시 장애는 넘기고, 지속되면 포기)


class RenderRequest(BaseModel):
    """렌더 요청 — 대상 편성과 범퍼 옵션."""

    comp_id: int
    bumper: bool = True     # 이닝 그룹 사이 범퍼 (워커 기본값은 false — 명시 전달)
    force: bool = False     # 이미 렌더된 편성 재렌더 (범퍼 변경 등 운영용 — 뷰어 UI 미사용)


@router.post("/render", status_code=202)
async def post_render(req: RenderRequest, request: Request,
                      background: BackgroundTasks) -> dict:
    """
    Summary:
        comp_id 편성을 worker-render 에 접수 — 202 반환 후 완료는 백그라운드에서 확인.
    Returns:
        dict: {comp_id, v_id, status="accepted", output_path(예정 경로)}.
    """
    st = request.app.state
    comp = await st.compose_repo.fetch(req.comp_id)
    if not comp:
        raise HTTPException(404, detail={"code": "COMPOSE_NOT_FOUND", "comp_id": req.comp_id})

    if comp["status"] != "ok" or not comp["clips"]:
        # empty 편성은 렌더하지 않는다 — 사전 차단 (빈 mp4 를 만들 이유가 없다)
        raise HTTPException(409, detail={
            "code": "COMPOSE_NOT_RENDERABLE",
            "message": (
                f"comp_id={req.comp_id} status={comp['status']} "
                f"클립 {len(comp['clips'])}건 — 렌더 대상 아님"),
        })

    if comp["render_datetime"] and not req.force:
        # 이미 산출물이 있는 편성 — 렌더 1건이 GPU 수 분·수백 MB 라 중복은 순수 낭비.
        # 경쟁 상황(다른 창에서 먼저 렌더)이 정상 경로라 뷰어는 이를 에러로 표시하지 않는다.
        raise HTTPException(409, detail={
            "code": "COMPOSE_ALREADY_RENDERED",
            "comp_id": req.comp_id,
            "rendered_at": comp["render_datetime"].isoformat(),
        })

    if req.comp_id in _RENDERING:
        # 비동기라 완료 전에는 render_datetime 이 NULL — 진행 중 중복은 이쪽이 막는다
        raise HTTPException(409, detail={
            "code": "RENDER_IN_PROGRESS",
            "comp_id": req.comp_id,
            "message": "이미 렌더가 진행 중입니다.",
        })

    try:
        payload = build_request(comp["v_id"], req.comp_id, comp["clips"], req.bumper, sync=False)
    except ValueError as e:
        # 이닝 결손 = 상류 발행 데이터 결함 — 렌더로 덮지 않고 드러낸다
        raise HTTPException(422, detail={"code": "COMPOSE_INVALID_INNING", "message": str(e)}) from e

    # 검사 통과 즉시 선점 — 아래 await 마다 양보 지점이라, 워커 왕복 뒤에 등록하면
    # 그 사이 들어온 같은 comp_id 요청이 위 검사를 그대로 통과한다(중복 렌더).
    _RENDERING[req.comp_id] = comp["v_id"]
    try:
        with bind_v_id(comp["v_id"]):
            log.info("렌더 접수 요청: comp_id=%s 클립 %d건 이닝 %s",
                     req.comp_id, len(comp["clips"]), list(payload["innings"]))
            await st.status.set(comp["v_id"], COMPOSE_RENDER)
            try:
                result = await st.render.render(payload)
            except httpx.HTTPError as e:
                log.error("렌더 접수 실패: comp_id=%s %s: %s", req.comp_id, type(e).__name__, e)
                await st.status.set(comp["v_id"], COMPOSE_ERROR_RENDER, f"{type(e).__name__}: {e}")
                raise HTTPException(502, detail={
                    "code": "RENDER_FAILED",
                    "message": f"{type(e).__name__}: {e}",
                }) from e
            log.info("렌더 접수 응답: %s", result)

        # 실제로 보낸 범퍼 값은 접수 시점에 확정 — 완료 스탬프(render_datetime)와 분리
        await st.compose_repo.mark_render_started(req.comp_id, req.bumper)
    except BaseException:
        # 감시가 뜨기 전에 실패했으면 여기서 풀어야 한다(_watch 의 finally 가 못 돈다)
        _RENDERING.pop(req.comp_id, None)
        raise
    background.add_task(_watch, request, req.comp_id, comp["v_id"])
    return {"comp_id": req.comp_id, "v_id": comp["v_id"], **result}


@router.get("/render/{comp_id}")
async def get_render(comp_id: int, request: Request) -> dict:
    """
    Summary:
        렌더 상태 조회 — 이미 스탬프됐으면 DB 로, 아니면 워커에 물어 보정까지 한다.
    Returns:
        dict: {comp_id, v_id, status, output_path, error, rendered_at?}.
            status 는 done / running / accepted / error / not_requested / unknown.
    """
    st = request.app.state
    comp = await st.compose_repo.fetch(comp_id)
    if not comp:
        raise HTTPException(404, detail={"code": "COMPOSE_NOT_FOUND", "comp_id": comp_id})

    v_id = comp["v_id"]
    if comp["render_datetime"]:
        return {"comp_id": comp_id, "v_id": v_id, "status": "done",
                "rendered_at": comp["render_datetime"].isoformat()}

    try:
        res = await st.render.status(v_id, comp_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # 워커가 모르는 건 = 접수된 적 없음 (미렌더 편성의 정상 응답)
            return {"comp_id": comp_id, "v_id": v_id, "status": "not_requested"}
        return _unknown(comp_id, v_id, e)
    except httpx.HTTPError as e:
        # 워커 미기동(GPU 야간 중지 등) — 감추지 않고 unknown 으로 드러낸다
        return _unknown(comp_id, v_id, e)

    if res.get("status") == "done":
        # 폴러가 유실된 경우의 보정 지점 — 조회 한 번으로 스탬프가 되살아난다
        stamped = await _stamp_done(st, comp_id, v_id)
        return {"comp_id": comp_id, "v_id": v_id, **res,
                **({} if stamped else {"stamped": False,
                                       "message": "렌더는 완료됐으나 완료 기록에 실패했습니다."})}
    return {"comp_id": comp_id, "v_id": v_id, **res}


def _unknown(comp_id: int, v_id: int, e: Exception) -> dict:
    """워커 상태 조회 실패 — 상태를 지어내지 않고 unknown 으로 반환."""
    log.warning("렌더 상태 조회 실패: comp_id=%s %s: %s", comp_id, type(e).__name__, e)
    return {"comp_id": comp_id, "v_id": v_id, "status": "unknown",
            "error": f"{type(e).__name__}: {e}"}


async def _stamp_done(st, comp_id: int, v_id: int) -> bool:
    """완료 스탬프 + 상태 코드 — 스탬프 실패는 4960 으로 남기고 False 를 돌려준다.

    스탬프가 빠지면 뷰어가 같은 편성을 다시 렌더하게 되므로(GPU 수 분 낭비), 실패를
    조용히 넘기지 않고 상태 코드로 드러낸다 — mp4 는 이미 존재한다는 뜻이다.
    """
    try:
        await st.compose_repo.mark_rendered(comp_id)
    except Exception as e:              # 실패 사실을 상태 코드로 드러내고 계속
        log.exception("렌더 완료 기록 실패: comp_id=%s", comp_id)
        await st.status.set(v_id, COMPOSE_ERROR_STAMP, f"{type(e).__name__}: {e}")
        return False
    await st.status.set(v_id, COMPOSE_OK)
    return True


async def _watch(request: Request, comp_id: int, v_id: int) -> None:
    """백그라운드 폴러 — 워커에 주기적으로 물어 done/error 를 확정한다.

    타임아웃·연속 조회 실패는 4950 으로 남긴다 (mp4 유무는 GET 조회로 재확인 가능 —
    워커가 나중에 끝냈다면 조회 시 보정 경로가 스탬프한다).
    """
    st = request.app.state
    deadline = time.monotonic() + st.settings.render_timeout
    interval = st.settings.render_poll_interval
    fails = 0
    try:
        with bind_v_id(v_id):
            while time.monotonic() < deadline:
                await asyncio.sleep(interval)
                try:
                    res = await st.render.status(v_id, comp_id)
                except httpx.HTTPError as e:
                    fails += 1
                    log.warning("렌더 상태 조회 실패(%d/%d): comp_id=%s %s",
                                fails, _POLL_FAIL_MAX, comp_id, e)
                    if fails >= _POLL_FAIL_MAX:
                        await st.status.set(v_id, COMPOSE_ERROR_RENDER,
                                            f"렌더 상태 조회 연속 실패: {type(e).__name__}: {e}")
                        return
                    continue
                fails = 0
                status = res.get("status")
                if status == "done":
                    log.info("렌더 완료: comp_id=%s %s", comp_id, res.get("output_path"))
                    await _stamp_done(st, comp_id, v_id)
                    return
                if status == "error":
                    log.error("렌더 실패: comp_id=%s %s", comp_id, res.get("error"))
                    await st.status.set(v_id, COMPOSE_ERROR_RENDER, res.get("error") or "")
                    return
            log.error("렌더 감시 타임아웃: comp_id=%s (%.0f초)", comp_id, st.settings.render_timeout)
            await st.status.set(v_id, COMPOSE_ERROR_RENDER,
                                f"렌더 {st.settings.render_timeout:.0f}초 내 미완료 (워커 확인 필요)")
    finally:
        # 취소(종료 등)로 빠져나가도 진행 중 표시는 반드시 푼다
        _RENDERING.pop(comp_id, None)
