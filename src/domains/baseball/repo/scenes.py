"""scene 구간 repository — t_scene_baseball 읽기 전용 (편성 인벤토리 재료).

이 테이블은 agent-vision scene 단계의 산출물이다 — 쓰기는 agent-vision 소유, 여기는
소비만 한다. 행 표현 규약도 agent-vision 를 따른다: 이닝 '1회초', tags·labels 는 콤마
연결 문자열, end_idxs 는 콤마 연결 초 목록(최대 2개).

시작 앵커(pitch_idx)·끝 후보(end_idxs)는 상류가 이미 계산해 담아 둔다 —
compose 는 경계를 다시 추론하지 않고 이 값으로 클립 좌표를 정한다.
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

    scene_no: int                   # 영상 안 시간순 일련번호 (1부터)
    start: int                      # 구간 시작초 (idx_start_scb — 이전 전광판 관측 행)
    end: int                        # 구간 끝초 (idx_end_scb — 결과 전광판 관측 행)
    pitch_idx: int | None           # 구간 안 투구 시점 초 (미탐지 None) — 클립 시작 앵커
    end_idxs: tuple[int, ...]       # 클립 종료 후보 초 (최대 2개, 시간순 — 없으면 빈 튜플)
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


class SceneRepo:
    """t_scene_baseball 조회 전담 (읽기 전용 — 쓰기는 agent-vision 소유)."""

    def __init__(self, db: Database) -> None:
        """Database(커넥션 풀 래퍼)를 주입받는다."""
        self._db = db

    async def fetch(self, v_id: int) -> list[Scene]:
        """
        Summary:
            영상 1건의 scene 구간을 scene_no 순으로 읽는다 — 편성 인벤토리.
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            list[Scene]: 구간 목록 — 없으면 빈 목록 (호출부가 발행 선행 오류로 변환).
        """
        sql = (
            "SELECT scene_no, idx_start_scb, idx_end_scb, pitch_idx, end_idxs, "
            "       inning, home_team, away_team, score_home, score_away, "
            "       tags, labels, diff_out, diff_base, diff_score "
            "FROM t_scene_baseball WHERE v_id = %s ORDER BY scene_no"
        )
        async with self._db.acquire() as conn, conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, (v_id,))
            rows = list(await cur.fetchall())
        scenes = [Scene(
            scene_no=r["scene_no"],
            start=r["idx_start_scb"],
            end=r["idx_end_scb"],
            pitch_idx=r["pitch_idx"],
            end_idxs=tuple(int(x) for x in (r["end_idxs"] or "").split(",") if x),
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
