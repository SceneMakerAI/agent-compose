-- 야구 팀명 사전 — 팀 1행: team_id(대표) + alias(전 표기 콤마 나열).
--
-- team_id 는 KBO 전광판 표기 그대로다 ('LG'·'KIA'·'SSG'…) — 구간의
-- home_team/away_team 값과 직접 일치한다 (필터 매칭 기준).
-- alias 는 team_id 자신을 포함한 모든 표기(한글 발음·마스코트명·영문·전광판 변형)를
-- 담는다 — LLM 은 질의의 표기가 alias 에 있으면 대표(team_id)를 고른다.
-- 이 파일은 재생성용 스냅샷 — 별칭 추가·수정은 DB 와 이 파일을 같이 갱신한다.

CREATE TABLE IF NOT EXISTS t_team_baseball (
    team_id VARCHAR(16)  NOT NULL COMMENT 'KBO 전광판 표기 (LG·KIA·SSG…) = 대표',
    alias   VARCHAR(255) NOT NULL COMMENT '전 표기 콤마 나열 (team_id 포함 — 한글·마스코트·영문·전광판 변형)',
    PRIMARY KEY (team_id)
) COMMENT='야구 팀명 사전 — 질의의 다양한 팀 표기를 전광판 표기(team_id)로 정규화';

INSERT IGNORE INTO t_team_baseball (team_id, alias) VALUES
  ('KIA', 'KIA,기아,타이거즈,기아타이거즈,KIA타이거즈'),
  ('KT',  'KT,케이티,위즈,KT위즈'),
  ('LG',  'LG,엘지,트윈스,LG트윈스,엘지트윈스'),
  ('NC',  'NC,엔씨,다이노스,NC다이노스'),
  ('SSG', 'SSG,쓱,랜더스,신세계,SSG랜더스,신세계랜더스'),
  ('두산', '두산,베어스,두산베어스'),
  ('롯데', '롯데,자이언츠,롯데자이언츠,LOTTE,LOTTEE'),
  ('삼성', '삼성,라이온즈,삼성라이온즈,SAMSUNG'),
  ('키움', '키움,히어로즈,키움히어로즈'),
  ('한화', '한화,이글스,한화이글스');
