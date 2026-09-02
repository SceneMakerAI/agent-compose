-- 편성 결과 — 질의 기반 편성 헤더 + 클립 (agent-compose 소유).
--
-- - PK 는 (v_id, comp_id) 복합 — comp_id 는 v_id 안에서 1부터 발급 (전역 시퀀스 아님).
--   발급은 저장 트랜잭션이 MAX(comp_id)+1 로 계산한다 (rdb/composes.py 소유).
-- - 좌표는 정수 초 — agent-vision 파이프라인 좌표계 그대로 (time 형 변환 없음).
-- - 스펙 상세(필터·선곡 근거)는 트레이스 파일(logs/) 소유 — 헤더에 두지 않는다.

CREATE TABLE IF NOT EXISTS t_compose (
    v_id         MEDIUMINT UNSIGNED NOT NULL COMMENT 't_video.v_id',
    comp_id      SMALLINT UNSIGNED  NOT NULL COMMENT '편성 id — v_id 안에서 1부터',
    query        VARCHAR(200)       NOT NULL COMMENT '사용자 질의 원문',
    budget_sec   SMALLINT UNSIGNED  NULL     COMMENT '요청 목표 분량(초) — NULL=미지정(절단 없음)',
    status_code  SMALLINT           NOT NULL COMMENT 't_code 4000번대 — 4020~4040 진행 국면 / 4000 완료 / 4001 빈 편성 / 4900 실패',
    bumper_yn    CHAR(1)            NOT NULL DEFAULT 'Y' COMMENT '렌더 시 이닝 그룹 사이 범퍼 삽입 여부',
    duration_sec SMALLINT UNSIGNED  NOT NULL DEFAULT 0 COMMENT '최종 클립 길이 합(초)',
    clip_cnt     SMALLINT UNSIGNED  NOT NULL DEFAULT 0 COMMENT '최종 클립 수',
    reg_datetime DATETIME           NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (v_id, comp_id),
    CONSTRAINT fk_t_compose_t_code FOREIGN KEY (status_code) REFERENCES t_code (code)
) COMMENT='질의 기반 편성 헤더 (agent-compose)';

CREATE TABLE IF NOT EXISTS t_compose_clip (
    v_id      MEDIUMINT UNSIGNED NOT NULL COMMENT 't_video.v_id',
    comp_id   SMALLINT UNSIGNED  NOT NULL COMMENT 't_compose.comp_id (v_id 안 시퀀스)',
    clip_seq  SMALLINT UNSIGNED  NOT NULL COMMENT '재생 순서 (시간순, 1부터)',
    scene_no  SMALLINT UNSIGNED  NOT NULL COMMENT 't_scene_baseball.scene_no',
    start_sec INT UNSIGNED       NOT NULL COMMENT '클립 시작 초 (pitch 앵커 기반)',
    end_sec   INT UNSIGNED       NOT NULL COMMENT '클립 끝 초 (end_idxs 선택)',
    tags      VARCHAR(255)       NULL     COMMENT '전광판 사건 태그 콤마 (표시용 사본)',
    labels    VARCHAR(255)       NULL     COMMENT '구간 판정 라벨 콤마 (표시용 사본)',
    inning    VARCHAR(10)        NOT NULL COMMENT '이닝 (1회초…) — 렌더 이닝 그룹핑 키',
    PRIMARY KEY (v_id, comp_id, clip_seq)
) COMMENT='편성 클립 (agent-compose) — 좌표는 정수 초 (agent-vision 좌표계 그대로)';
