"""worker-render 요청 변환 — t_compose_clip 행 → 렌더 페이로드 (순수 함수, 네트워크 무관).

요청 본문 (POST /render/sports/baseball):
    {
      "v_id": 1006,
      "c_id": 4,                        # 편성 id (comp_id)
      "file_name": "1006/source.mp4",   # 워커 vod 루트 기준 상대 경로 (worker-prep-stt 산출)
      "sync_yn": false,                 # 항상 false — 접수만 하고 GET /render/{v_id}/{c_id} 폴링
      "bumper_yn": true,                # 이닝 그룹 사이 범퍼 — 워커 기본이 false 라 항상 명시
      "innings": {                      # "{이닝번호}_{top|bot}" 키, 배열 순서 = 렌더 순서
        "1_top": [{"start_sec": 903.0, "end_sec": 906.0,          # 렌더는 sec 만 사용
                   "start_hms": "00:15:03.0", "end_hms": "00:15:06.0"}]  # hms 는 표기용
      }
    }

inning 이 비거나 형식 밖('-1' 미인식 포함)인 클립은 ValueError — 상류 발행 데이터
결함 신호라서, 조용히 빼고 렌더하면 편성과 결과물이 어긋난다 (호출부가 422 로 변환).
"""

import re

SOURCE_FILE_NAME = "source.mp4"   # worker-prep-stt 산출 고정 파일명 (교차 서비스 계약)

_INNING_RE = re.compile(r"(\d+)\s*회\s*(초|말)")


def inning_key(inning: str) -> str:
    """
    Summary:
        클립 이닝("6회초") → worker-render 이닝 키("6_top").
    Args:
        inning (str): "N회초|말" 형식 (공백 유무 무관, 연장 이닝 포함).
    Returns:
        str: "{N}_top" 또는 "{N}_bot".
    Raises:
        ValueError: 형식 밖(빈 문자열·'-1' 미인식 포함) — 발행 데이터 결함.
    """
    m = _INNING_RE.match((inning or "").strip())
    if not m:
        raise ValueError(f"이닝 형식 밖: {inning!r}")
    if m.group(2) == "초":
        return f"{m.group(1)}_top"
    return f"{m.group(1)}_bot"


def sec_to_hms(sec: float) -> str:
    """초 → "hh:mm:ss.f" (소수 1자리 — worker-render 디버깅 표기 규격)."""
    h, rem = divmod(float(sec), 3600)
    m, rem = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{rem:04.1f}"


def build_request(v_id: int, comp_id: int, clips: list[dict], bumper: bool) -> dict:
    """
    Summary:
        편성 클립들 → POST /render/sports/baseball 요청 본문.
    Args:
        v_id (int): 영상 id.
        comp_id (int): 편성 id — 워커의 c_id (결과 파일 추적 키).
        clips (list[dict]): t_compose_clip 행 (scene_no·inning·start_sec·end_sec,
            시간순 전제 — 이닝 그룹·그룹 내 순서가 입력 순서로 유지된다).
        bumper (bool): 이닝 그룹 사이 범퍼 삽입 여부.
    Returns:
        dict: worker-render 요청 본문 (sync_yn=False 고정).
    Raises:
        ValueError: 클립 0건, 또는 이닝 형식 밖 클립 존재 (scene_no 나열).
    """
    if not clips:
        raise ValueError("클립 0건 — 렌더 대상 없음")

    # 이닝 검사를 먼저 전 건 수행 — 결함 클립을 한 번에 전부 드러낸다
    bad_scene_nos = []
    for clip in clips:
        try:
            inning_key(clip.get("inning") or "")
        except ValueError:
            bad_scene_nos.append(clip["scene_no"])
    if bad_scene_nos:
        raise ValueError(
            f"이닝 없는 클립 scene_no={bad_scene_nos} — 발행 데이터 확인 필요 (렌더 중단)")

    innings: dict[str, list[dict]] = {}
    for clip in clips:
        key = inning_key(clip["inning"])
        if key not in innings:
            innings[key] = []
        innings[key].append({
            "start_sec": float(clip["start_sec"]),
            "end_sec": float(clip["end_sec"]),
            "start_hms": sec_to_hms(clip["start_sec"]),
            "end_hms": sec_to_hms(clip["end_sec"]),
        })

    return {
        "v_id": v_id,
        "c_id": comp_id,
        "file_name": f"{v_id}/{SOURCE_FILE_NAME}",
        "sync_yn": False,
        "bumper_yn": bumper,
        "innings": innings,
    }
