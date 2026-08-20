-- t_code 4030·4040 설명문 갱신 — 노드 개편으로 두 국면의 경계가 옮겨졌다.
--
-- 구 배선은 컷 단계에서 예산까지 채우고 마지막에 검수했다. 지금은
--   set_bounds·refine_bounds  → 구간만 확정 (4030)
--   score_match·drop_unmatched·order_clips·fill_budget → 채점하고 순서·길이 확정 (4040)
-- 로 갈렸다. 4030 의 "예산에 맞춰 채우고"와 4040 의 "완성된 편성이" 가 둘 다 틀린 말이
-- 됐다 — 예산은 4040 에서 채우고, 4040 시점의 편성은 아직 완성 전이다.
--
-- 코드 값(4030·4040)과 object·name 은 그대로다. description 만 고친다.
-- 적용: mysql -h <db> -P 13306 -u sm_db -p sm_db < 이 파일

UPDATE t_code
   SET description = '고른 장면을 화면 전환에 맞춰 클립 구간으로 확정하고 있습니다.'
 WHERE code = 4030;

UPDATE t_code
   SET description = '클립이 질의에 맞는지 채점하고, 순서와 전체 길이를 확정하고 있습니다.'
 WHERE code = 4040;
