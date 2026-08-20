-- t_compose_clip 클립 경계를 밀리초 정밀도로 넓힌다.
--
-- 왜: start_time/end_time 이 TIME(정밀도 0)이라 SEC_TO_TIME(4369.3) 이 4369s 로
-- 내려갔다. v201 은 상류 t_segment 경계가 소수라 그 0.3초가 샷을 가른다 —
-- 4369.3s 는 '투구' 샷 시작인데 4369s 는 직전 '리액션' 끝자락이다. 그래서 감사에서
-- "투구 시작 0%"로 보였지만 실제 cut 산출은 93%가 투구 시작이었다(2026-08-20 실측).
--
-- 파급: TIME_TO_SEC 은 time(3) 에서 소수 포함 DECIMAL 을 돌려준다.
--   · agent-compose  db/compose_repo.fetch — float() 로 환산 (같은 커밋에 반영)
--   · ui-sbs-viwer   lib/server/composes.ts:114 — TIME_TO_SEC 결과가 DECIMAL 이라
--     드라이버가 문자열로 넘길 수 있다. 별도 레포라 미반영 — 확인 필요.
-- 기존 행은 .000 이 되며 값은 변하지 않는다.

ALTER TABLE t_compose_clip
  MODIFY start_time time(3) NOT NULL,
  MODIFY end_time   time(3) NOT NULL;
