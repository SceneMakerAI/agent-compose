-- t_compose 에 렌더 진행 상태 컬럼을 둔다.
--
-- 왜: render_datetime 은 "완료 시각"이라 **완료/미완료** 두 상태만 표현한다. 렌더가
-- 비동기 접수로 바뀌면서 그 사이의 '진행중'과 '실패'가 DB 어디에도 남지 않는다 —
-- 진행중은 서비스 프로세스 메모리(_RENDERING)에만 있어 재기동하면 사라지고, 실패는
-- t_video.status_code 4950 으로만 남아 **어느 편성이** 실패했는지 알 수 없다.
-- 뷰어는 "영상 준비됨 / 만드는 중 / 실패" 를 배지로 구분해야 하는데 그 근거가 없었다.
--
-- 값 규약은 t_code.result 와 **같은 뜻**으로 맞춘다 (0 이 '끝난 것'):
--   NULL  렌더 요청된 적 없음  → 렌더 버튼 노출
--      1  진행중 (접수~완료 전) → "만드는 중" · 중복 요청 차단
--      0  성공                  → 영상 준비됨 (render_datetime 도 함께 채워진다)
--     -1  실패                  → 재렌더 가능 (실패가 편성을 잠그지 않는다)
--
-- 기존 행 백필: render_datetime 이 있으면 성공(0), 없으면 요청 없음(NULL).
-- 적용: mysql -h <db> -P 13306 -u sm_db -p sm_db < 이 파일

ALTER TABLE t_compose
  ADD COLUMN IF NOT EXISTS render_status tinyint(4) NULL DEFAULT NULL
      COMMENT 'NULL=요청 없음, 1=진행중, 0=성공, -1=실패 (t_code.result 와 같은 규약)'
      AFTER render_datetime;

UPDATE t_compose SET render_status = 0 WHERE render_datetime IS NOT NULL;
