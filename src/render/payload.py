"""worker-render 요청 변환 — t_compose_clip 행 → 렌더 페이로드 (순수 함수, 네트워크 무관).

worker-render 계약 (2026-08-18 스펙):
- file_name: 원본 파일명. worker-prep-stt 가 항상 /mnt/nvme/vod/{v_id}/source.mp4 로
  만들므로 "source.mp4" 고정 — 경로가 아니라 파일명만 보낸다.
- innings: {"{이닝번호}_{top|bot}": [{start_sec, end_sec, start_hms, end_hms}, …]}.
  렌더링은 sec 값만 사용, hms 는 디버깅 표기용.
- bumper: 이닝 그룹 사이 범퍼 삽입. 스펙 예제(bumper)와 표(bumper_yn)의 필드명이
  달라 예제 기준으로 시작 — 실렌더 테스트로 확정 예정.
- sync_yn=True: 렌더 완주 후 응답 (이 플로우는 상태 저장 없이 요청-응답으로 끝낸다).

inning 이 비거나 형식 밖인 클립은 ValueError — 상류 발행 데이터 결함 신호라서,
조용히 빼고 렌더하면 편성과 결과물이 어긋난다 (사전 차단, 호출부가 4xx 로 변환).
"""

import re

SOURCE_FILE_NAME = "source.mp4"   # worker-prep-stt 산출 고정 파일명 (교차 서비스 계약)

_INNING_RE = re.compile(r"(\d+)\s*회\s*(초|말)")


def inning_key(inning: str) -> str:
    """t_compose_clip.inning("3회 초") → worker-render 이닝 키("3_top").

    Args:
        inning (str): "N회 초|말" 형식 (공백 유무 무관, 연장 이닝 포함).
    Returns:
        str: "{N}_top" 또는 "{N}_bot".
    Raises:
        ValueError: 형식 밖(빈 문자열 포함) — 발행 데이터 결함.
    """
    m = _INNING_RE.match((inning or "").strip())
    if not m:
        raise ValueError(f"이닝 형식 밖: {inning!r}")
    return f"{m.group(1)}_{'top' if m.group(2) == '초' else 'bot'}"


def sec_to_hms(sec: float) -> str:
    """초 → "hh:mm:ss.f" (소수 1자리 — worker-render 디버깅 표기 규격)."""
    h, rem = divmod(float(sec), 3600)
    m, rem = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{rem:04.1f}"


def build_request(v_id: int, c_id: int, clips: list[dict], bumper: bool) -> dict:
    """
    Summary:
        편성 클립들 → POST /render/sports/baseball 요청 본문.
    Args:
        v_id (int): 영상 id.
        c_id (int): 편성 id (comp_id — 결과 파일 c_{c_id}.mp4 로 추적).
        clips (list[dict]): t_compose_clip 행 (scene_id·inning·start·end, 시간순 전제).
        bumper (bool): 이닝 그룹 사이 범퍼 삽입 여부.
    Returns:
        dict: worker-render 요청 본문.
    Raises:
        ValueError: 클립 0건, 또는 이닝 없는/형식 밖 클립 존재 (scene_id 나열).
    """
    if not clips:
        raise ValueError("클립 0건 — 렌더 대상 없음")
    bad = []
    for c in clips:
        try:
            inning_key(c.get("inning") or "")
        except ValueError:
            bad.append(c["scene_id"])
    if bad:
        raise ValueError(f"이닝 없는 클립 scene_id={bad} — 발행 데이터 확인 필요 (렌더 중단)")

    innings: dict[str, list[dict]] = {}
    for c in clips:                       # 시간순 입력이라 이닝 그룹·그룹 내 순서 자연 유지
        innings.setdefault(inning_key(c["inning"]), []).append({
            "start_sec": float(c["start"]),
            "end_sec": float(c["end"]),
            "start_hms": sec_to_hms(c["start"]),
            "end_hms": sec_to_hms(c["end"]),
        })
    return {
        "v_id": v_id,
        "c_id": c_id,
        "file_name": SOURCE_FILE_NAME,
        "sync_yn": True,
        "bumper": bumper,
        "innings": innings,
    }
