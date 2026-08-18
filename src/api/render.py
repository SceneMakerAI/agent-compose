"""렌더 라우트 — 저장된 편성(comp_id) 1건을 worker-render 로 mp4 렌더링 (동기).

상태 저장 없음 — 요청-응답으로 끝난다 (렌더 이력이 필요해지면 그때 재설계).
empty 편성·이닝 결손은 워커 호출 **전에** 이쪽에서 4xx 로 차단한다 — 조용히
빈/어긋난 렌더를 만드는 것보다 사전에 드러내는 게 원칙 (fail-open 금지 계승).
"""

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from log import bind_v_id, get_logger
from render.payload import build_request

log = get_logger(__name__)
router = APIRouter()


class RenderRequest(BaseModel):
    """렌더 요청 — 대상 편성과 범퍼 옵션."""

    comp_id: int
    bumper: bool = True     # 이닝 그룹 사이 범퍼 (worker-render 기본값과 동일)


@router.post("/render")
async def post_render(req: RenderRequest, request: Request) -> dict:
    """
    Summary:
        comp_id 편성을 worker-render 에 동기 렌더 요청 — 완주 후 결과 반환.
    Returns:
        dict: {comp_id, v_id, status, output_path, error} (워커 응답 중계).
    """
    st = request.app.state
    comp = await st.compose_repo.fetch(req.comp_id)
    if not comp:
        raise HTTPException(404, detail={"code": "COMPOSE_NOT_FOUND", "comp_id": req.comp_id})

    if comp["status"] != "ok" or not comp["clips"]:
        # empty 편성은 렌더하지 않는다 — 사전 차단 (빈 mp4 를 만들 이유가 없다)
        raise HTTPException(409, detail={
            "code": "COMPOSE_NOT_RENDERABLE",
            "message": (f"comp_id={req.comp_id} status={comp['status']} "
                        f"클립 {len(comp['clips'])}건 — 렌더 대상 아님"),
        })

    try:
        payload = build_request(comp["v_id"], req.comp_id, comp["clips"], req.bumper)
    except ValueError as e:
        # 이닝 결손 = 상류 발행 데이터 결함 — 렌더로 덮지 않고 드러낸다
        raise HTTPException(422, detail={"code": "COMPOSE_INVALID_INNING",
                                         "message": str(e)}) from e

    with bind_v_id(comp["v_id"]):
        log.info("렌더 요청: comp_id=%s 클립 %d건 이닝 %s",
                 req.comp_id, len(comp["clips"]), list(payload["innings"]))
        try:
            result = await st.render.render(payload)
        except httpx.HTTPError as e:
            log.error("렌더 실패: comp_id=%s %s: %s", req.comp_id, type(e).__name__, e)
            raise HTTPException(502, detail={
                "code": "RENDER_FAILED",
                "message": f"{type(e).__name__}: {e}"}) from e
        log.info("렌더 응답: %s", result)

    return {"comp_id": req.comp_id, "v_id": comp["v_id"], **result}
