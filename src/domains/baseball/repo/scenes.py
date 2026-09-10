"""scene 구간 repository — t_scene_baseball 읽기 전용 (편성 인벤토리 재료).

이 테이블은 agent-vision scene 단계의 산출물이다 — 쓰기는 agent-vision 소유, 여기는
소비만 한다. 행 표현 규약도 agent-vision 를 따른다: 이닝 '1회초', tags·labels 는 콤마
연결 문자열, end_idxs 는 콤마 연결 초 목록(최대 2개).

시작 앵커(pitch_idx)·끝 후보(end_idxs)는 상류가 이미 계산해 담아 둔다 —
compose 는 경계를 다시 추론하지 않고 이 값으로 클립 좌표를 정한다.

좌표는 두 축이다 — 청크 축(start·end·pitch_idx·end_idxs)으로 계산하고, 전체 영상
축(*_whole)은 저장·표시용으로 들고만 간다. 두 축 다 상류가 실어 주므로 유도하지
않는다(오프셋 덧셈 금지) — end 후보는 두 목록의 인덱스가 1:1 이다.
번호도 둘 — 정본 키는 (stream_id, scene_stream_seq), 표시·정렬은 scene_seq.
"""

from dataclasses import dataclass

from asyncmy.cursors import DictCursor

from log import get_logger
from rdb.pool import Database

log = get_logger(__name__)


@dataclass(frozen=True)
class Scene:
    """t_scene_baseball 1행 — 편성 인벤토리의 구간 단위 (읽기 전용 스냅샷).

    노드가 절대 수정하지 않는다(frozen) — 편성 중간 산출은 별도 dict 로 만든다.
    """

    stream_id: str                  # 청크 id — t_video_file.stream_id (VOD 는 'VOD')
    scene_stream_seq: int           # 청크 안 시간순 일련번호 (1부터) — 정본 키
    scene_seq: int                  # 영상 통산 시간순 일련번호 (1부터) — 표시·정렬
    start: int                      # 구간 시작초 (이전 전광판 관측 행) — 청크 축
    end: int                        # 구간 끝초 (결과 전광판 관측 행) — 청크 축
    pitch_idx: int | None           # 구간 안 투구 시점 초 (미탐지 None) — 클립 시작 앵커
    end_idxs: tuple[int, ...]       # 클립 종료 후보 초 (최대 2개, 시간순 — 없으면 빈 튜플)
    start_whole: int                # 위 네 값의 전체 영상 축 짝 (저장·표시용)
    end_whole: int
    pitch_whole: int | None
    end_idxs_whole: tuple[int, ...]
    inning: str                     # 예: '1회초' (미인식 '-1' 규약 그대로)
    home_team: str
    away_team: str
    score_home: int                 # 구간 종료 시점 스코어
    score_away: int
    tags: tuple[str, ...]           # 전광판 사건 태그 (아웃·득점·베이스 변화)
    labels: tuple[str, ...]         # LLM 판정 규정 용어 (없으면 빈 튜플)
    diff_out: int                   # 구간 시작→끝 아웃 변화량
    diff_base: str | None           # 루 점유 변화 '100>010' — 변화 없으면 None
    diff_score: int                 # 구간 시작→끝 점수(합) 변화량


def _split(text: str | None) -> tuple[str, ...]:
    """콤마 연결 문자열 → 튜플. NULL·빈 칸은 빈 튜플 (등장 순서 유지)."""
    return tuple(x.strip() for x in (text or "").split(",") if x.strip())


def _secs(text: str | None) -> tuple[int, ...]:
    """콤마 연결 초 목록 → 정수 튜플. NULL·빈 칸은 빈 튜플 (시간순 유지)."""
    return tuple(int(x) for x in _split(text))


class SceneRepo:
    """t_scene_baseball 조회 전담 (읽기 전용 — 쓰기는 agent-vision 소유)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def fetch(self, v_id: int) -> list[Scene]:
        """
        Summary:
            영상 1건의 scene 구간을 시간순으로 읽는다 — 편성 인벤토리.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            list[Scene]: 구간 목록 — 없으면 빈 목록 (호출부가 발행 선행 오류로 변환).
        Description:
            - 범위는 v_id 전체 — 청크를 가리지 않는다. 스트림이 아직 도착 중이면
              그 시점까지 적재된 청크만 담겨 온다 (부분 편성이 정상 동작이다).
        """
        sql = (
            "SELECT stream_id, scene_stream_seq, scene_seq, "
            "       start_scb_stream_sec, end_scb_stream_sec, pitch_stream_sec, end_stream_secs, "
            "       start_scb_sec, end_scb_sec, pitch_sec, end_secs, "
            "       inning, home_team, away_team, score_home, score_away, "
            "       tags, labels, diff_out, diff_base, diff_score "
            "FROM t_scene_baseball WHERE v_id = %s ORDER BY scene_seq"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())
        scenes = [Scene(
            stream_id=r["stream_id"],
            scene_stream_seq=r["scene_stream_seq"],
            scene_seq=r["scene_seq"],
            start=r["start_scb_stream_sec"],
            end=r["end_scb_stream_sec"],
            pitch_idx=r["pitch_stream_sec"],
            end_idxs=_secs(r["end_stream_secs"]),
            start_whole=r["start_scb_sec"],
            end_whole=r["end_scb_sec"],
            pitch_whole=r["pitch_sec"],
            end_idxs_whole=_secs(r["end_secs"]),
            inning=r["inning"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            score_home=r["score_home"],
            score_away=r["score_away"],
            tags=_split(r["tags"]),
            labels=_split(r["labels"]),
            diff_out=r["diff_out"],
            diff_base=r["diff_base"],
            diff_score=r["diff_score"],
        ) for r in rows]
        log.info("t_scene_baseball 조회: v_id=%s (%d구간)", v_id, len(scenes))
        return scenes
