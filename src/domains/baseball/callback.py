"""편성 결과 통보 — 요청이 준 callback_url 로 1회 POST.

본문의 scenes 는 야구 편성 결과 표현이다 (text 가 이닝) — 도메인 소유.
보조 단계라 본 작업을 죽이지 않는다: 실패는 로그만 남기고 재시도하지 않는다
(결과는 이미 DB 에 있고 GET /compose 로 받아갈 수 있다).
"""

import httpx

from log import get_logger

log = get_logger(__name__)


def hms(sec: int) -> str:
    """초 → 'hh:mm:ss'."""
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build(v_id: int, stream_id: str, search_id: str, query: str,
          code: int, result: str, clips: list[dict]) -> dict:
    """
    Summary:
        통보 본문 조립 — 좌표는 청크 축(그 클립이 속한 stream_id 기준)이다.
    Args:
        clips (list[dict]): _clip_rows 산출 (stream_id·start·end·inning 시간순).
    Returns:
        dict: {v_id, stream_id, search_id, query, result, code, scenes}.
    """
    scenes = []
    for clip in clips:
        scenes.append({
            "v_id": v_id,
            "stream_id": clip["stream_id"],
            "start_time": hms(clip["start"]),
            "end_time": hms(clip["end"]),
            "start_sec": clip["start"],
            "end_sec": clip["end"],
            "text": clip["inning"],
        })
    return {
        "v_id": v_id,
        "stream_id": stream_id,
        "search_id": search_id,
        "query": query,
        "result": result,
        "code": code,
        "scenes": scenes,
    }


async def send(url: str, payload: dict, timeout: float) -> None:
    """통보 1회 — 실패는 삼키고 로그만 (본 작업 비차단)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        log.info("콜백 전송: %s → %s (scenes %d)",
                 url, resp.status_code, len(payload["scenes"]))
    except Exception as e:            # noqa: BLE001 — 보조 단계, 죽이지 않는다
        log.warning("콜백 전송 실패(무시): %s — %s: %s", url, type(e).__name__, e)
