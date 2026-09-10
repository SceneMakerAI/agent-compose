-- 편성 결과 — 질의 기반 편성 헤더 + 클립 (agent-compose 소유).
--
-- - PK 는 (v_id, comp_id) 복합 — comp_id 는 v_id 안에서 1부터 발급 (전역 시퀀스 아님).
--   발급은 저장 트랜잭션이 MAX(comp_id)+1 로 계산한다 (rdb/composes.py 소유).
-- - 좌표는 정수 초 — agent-vision 파이프라인 좌표계 그대로 (time 형 변환 없음).
--   청크 축(*_stream_sec)·전체 영상 축(*_sec) 두 개를 함께 보존한다 — 상류가 둘 다
--   주므로 한쪽에서 다른 쪽을 유도하지 않는다 (2026-09-10 스트림 모드 대응).
-- - 구간 신원도 상류 2단 키를 그대로 보존한다: 정본은 (stream_id, scene_stream_seq),
--   표시·정렬은 scene_seq. 유니크 키를 정본 쪽에 거는 이유는 scene_seq 가 앞 청크
--   재처리 때 낡을 수 있는 파생값이기 때문이다.
-- - 스펙 상세(필터·선곡 근거)는 트레이스 파일(logs/) 소유 — 헤더에 두지 않는다.

CREATE TABLE IF NOT EXISTS t_compose (
    v_id         MEDIUMINT UNSIGNED NOT NULL COMMENT 't_video.v_id',
    comp_id      SMALLINT UNSIGNED  NOT NULL COMMENT '편성 id — v_id 안에서 1부터',
    query        VARCHAR(200)       NOT NULL COMMENT '사용자 질의 원문',
    budget_sec   SMALLINT UNSIGNED  NULL     COMMENT '요청 목표 분량(초) — NULL=미지정(절단 없음)',
    status_code  SMALLINT           NOT NULL COMMENT 't_code 4000번대 — 4020~4040 편성 진행 / 4050 렌더 진행 / 4000 완료 / 4001 빈 편성 / 4900 편성 실패 / 4950 렌더 실패',
    bumper_yn    CHAR(1)            NOT NULL DEFAULT 'Y' COMMENT '미사용 — 옛 렌더 범퍼 옵션. 컬럼만 남긴다 (2026-09-10 결정)',
    duration_sec SMALLINT UNSIGNED  NOT NULL DEFAULT 0 COMMENT '최종 클립 길이 합(초)',
    clip_cnt     SMALLINT UNSIGNED  NOT NULL DEFAULT 0 COMMENT '최종 클립 수',
    reg_datetime DATETIME           NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (v_id, comp_id),
    CONSTRAINT fk_t_compose_t_code FOREIGN KEY (status_code) REFERENCES t_code (code)
) COMMENT='질의 기반 편성 헤더 (agent-compose)';

CREATE TABLE IF NOT EXISTS t_compose_clip (
    v_id             MEDIUMINT UNSIGNED NOT NULL COMMENT 't_video.v_id',
    comp_id          SMALLINT UNSIGNED  NOT NULL COMMENT 't_compose.comp_id (v_id 안 시퀀스)',
    clip_seq         SMALLINT UNSIGNED  NOT NULL COMMENT '재생 순서 (시간순, 1부터)',
    stream_id        VARCHAR(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
                     NOT NULL DEFAULT 'VOD'
                     COMMENT '스트림 청크 id — t_video_file.stream_id (VOD 는 ''VOD''). 상류와 콜레이션 일치(utf8mb4_bin)',
    scene_stream_seq SMALLINT UNSIGNED  NOT NULL COMMENT '구간 번호(청크 안) — (v_id, stream_id, 이 값)이 t_scene_baseball 정본 키',
    scene_seq        SMALLINT UNSIGNED  NOT NULL COMMENT '구간 번호(전체 통산) — 표시·정렬용. 앞 청크 재처리로 낡을 수 있어 매칭에 쓰지 않는다',
    start_stream_sec INT UNSIGNED       NOT NULL COMMENT '클립 시작 초 — 청크 파일 기준 (pitch 앵커)',
    end_stream_sec   INT UNSIGNED       NOT NULL COMMENT '클립 끝 초 — 청크 파일 기준',
    start_sec        INT UNSIGNED       NOT NULL COMMENT '클립 시작 초 — 전체 영상 기준',
    end_sec          INT UNSIGNED       NOT NULL COMMENT '클립 끝 초 — 전체 영상 기준',
    tags             VARCHAR(255)       NULL     COMMENT '전광판 사건 태그 콤마 (표시용 사본)',
    labels           VARCHAR(255)       NULL     COMMENT '구간 판정 라벨 콤마 (표시용 사본)',
    inning           VARCHAR(10)        NOT NULL COMMENT '이닝 (1회초…) — 그룹핑 키',
    PRIMARY KEY (v_id, comp_id, clip_seq),
    UNIQUE KEY uk_compose_clip_scene (v_id, comp_id, stream_id, scene_stream_seq)
) COMMENT='편성 클립 (agent-compose) — 좌표 정수 초, 청크/전체 두 축';
