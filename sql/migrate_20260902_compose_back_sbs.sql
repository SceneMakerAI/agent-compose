-- 레거시 편성(t_compose_back·t_compose_clip_back) → 신규 테이블(t_compose·t_compose_clip) 이관.
-- 대상은 t_video.is_sbs = 1 영상의 편성만 — 고객사 뷰어(ui-sbs-viwer)가 신규 테이블을
-- 읽으므로 노출 대상 레거시 편성을 신규 구조로 옮긴다.
--
-- 컬럼 매핑:
-- - comp_id      : 레거시 전역 시퀀스 → v_id 안에서 1부터 재발급 (레거시 comp_id 순서 유지)
-- - status       : 'ok'+클립 있음 → 4000 / 'ok'+클립 없음 → 4001 / 그 외 → 4900
-- - budget_sec   : 레거시 0(미지정) → NULL
-- - bumper_yn    : tinyint → char — 0 → 'N', 1·NULL → 'Y' (신규 기본값)
-- - scene_id     → scene_no
-- - start/end    : time(3) → 정수 초 — 시작은 내림, 끝은 올림 (구간이 잘리지 않게)
-- - scene_type   → tags (표시용 사본 — 전광판 사건 태그와 같은 어휘)
-- - duration_sec·clip_cnt : 변환된 클립에서 재계산 (초 단위 변환과 일치시키기 위해)
-- - h_id·score_before/after·mode·view_side·targets·job_id·render_* : 신규 스키마에 없음 — 버린다
--
-- 이미 t_compose 에 행이 있는 v_id 는 건너뛴다 (재실행 안전 — comp_id 충돌 방지).

START TRANSACTION;

INSERT INTO t_compose
    (v_id, comp_id, query, budget_sec, status_code, bumper_yn, duration_sec, clip_cnt, reg_datetime)
WITH mapping AS (
    SELECT cb.comp_id AS old_comp_id,
           cb.v_id,
           ROW_NUMBER() OVER (PARTITION BY cb.v_id ORDER BY cb.comp_id) AS new_comp_id
    FROM t_compose_back cb
    JOIN t_video v ON v.v_id = cb.v_id AND v.is_sbs = 1
    WHERE NOT EXISTS (SELECT 1 FROM t_compose c WHERE c.v_id = cb.v_id)
)
SELECT m.v_id,
       m.new_comp_id,
       cb.query,
       NULLIF(cb.budget_sec, 0),
       CASE
           WHEN cb.status = 'ok' AND agg.cnt > 0 THEN 4000
           WHEN cb.status = 'ok'                 THEN 4001
           ELSE 4900
       END,
       CASE WHEN cb.bumper_yn = 0 THEN 'N' ELSE 'Y' END,
       COALESCE(agg.dur, 0),
       COALESCE(agg.cnt, 0),
       cb.reg_datetime
FROM mapping m
JOIN t_compose_back cb ON cb.comp_id = m.old_comp_id
LEFT JOIN (
    SELECT ccb.comp_id,
           SUM(CEILING(TIME_TO_SEC(ccb.end_time)) - FLOOR(TIME_TO_SEC(ccb.start_time))) AS dur,
           COUNT(*) AS cnt
    FROM t_compose_clip_back ccb
    GROUP BY ccb.comp_id
) agg ON agg.comp_id = m.old_comp_id;

INSERT INTO t_compose_clip
    (v_id, comp_id, clip_seq, scene_no, start_sec, end_sec, tags, labels, inning)
WITH mapping AS (
    SELECT cb.comp_id AS old_comp_id,
           cb.v_id,
           ROW_NUMBER() OVER (PARTITION BY cb.v_id ORDER BY cb.comp_id) AS new_comp_id
    FROM t_compose_back cb
    JOIN t_video v ON v.v_id = cb.v_id AND v.is_sbs = 1
    WHERE NOT EXISTS (SELECT 1 FROM t_compose_clip c WHERE c.v_id = cb.v_id)
)
SELECT m.v_id,
       m.new_comp_id,
       ccb.clip_seq,
       ccb.scene_id,
       FLOOR(TIME_TO_SEC(ccb.start_time)),
       CEILING(TIME_TO_SEC(ccb.end_time)),
       NULLIF(ccb.scene_type, ''),
       NULLIF(ccb.labels, ''),
       ccb.inning
FROM mapping m
JOIN t_compose_clip_back ccb ON ccb.comp_id = m.old_comp_id;

COMMIT;
