# compose 빈틈·모순 감사 (2026-08-19)

전 계층(그래프·결정·API/DB·검색) 코드 감사 결과. **플로우 순서**로 배열했다 —
한 단계씩 짚어가며 개선 여부를 정하기 위한 작업 목록이다.

각 항목의 상태 표기:
- **실측** = 운영 로그·DB·Milvus 로 재현 확인
- **코드확인** = 코드를 읽어 성립을 확인, 아직 실물로 터지진 않음
- **추측** = 코드 추론까지만, 검증 안 됨

수정할 때마다 해당 항목에 결정(고침/보류/무시)과 근거를 적어 남긴다.

---

## 0. 요청 접수 — `api/compose.py`

### 0-1. 잡 캐시가 유일한 comp_id 전달 통로인데 유실된다 [코드확인]
`_JOBS` 는 인메모리 dict. 정리는 POST 시점에 `status != "running"` 인 잡을 최대 50개
pop 하는 방식이라, 완료 잡 200개가 찬 뒤 새 요청이 들어오면 **가장 오래된 완료 잡부터
증발**한다. 폴링하던 클라이언트는 404 `JOB_NOT_FOUND` 를 받고 comp_id 를 영영 못 받는다.
TTL 개념이 없어 "완료 후 N분 보존" 같은 계약도 줄 수 없다.

### 0-2. `budget` 입력 검증 부재 [코드확인]
`ComposeRequest.budget` 에 `ge` 제약이 없다. `budget=0` 은 `if st.get("budget")` 에서
falsy 로 걸러져 **명시 입력이 조용히 무시**되고(주석은 "명시 입력이 우선"이라 단언),
음수는 통과해서 아무것도 못 담아 `status=empty` 로 끝난다. 잘못된 입력이 검증 오류가
아니라 "조건 맞는 장면 없음" 으로 보고된다.

### 0-3. 같은 v_id 동시 편성 시 status_code 가 서로를 덮어쓴다 [코드확인]
compose 에 중복 방지가 없는 건 의도된 설계(요청마다 새 comp_id)지만, docstring 의 근거
"읽기 전용 + comp_id 신규 발급이라 충돌 없음" 은 더 이상 사실이 아니다. `t_video` 는
v_id 단위 한 행이라 잡 A 가 4040 을 찍은 뒤 잡 B 가 4020 으로 되돌린다. 원샷이면 GPU
렌더도 두 번 돈다. (ingest 는 `_RUNNING` 집합으로 409 차단 — 단일 워커 전제에서 실동작.)

---

## 1. retrieve — `vector/store.py`, `flow/graph.py`

### 1-1. 색인 신선도를 아무도 검증하지 못한다 [실측(구조)]
컬렉션 스키마에 **발행 버전도 색인 시각도 없다**. 트리거는 vision3 의 fire-and-forget
훅뿐이라, 훅이 실패하면 재색인은 영원히 일어나지 않고 아무도 모른다.

publish 재실행으로 scene_id 가 재번호되면(2026-08-19 실제 발생) 낡은 번호가 남고,
`plan.parse` 는 **번호가 현재 인벤토리에 존재하기만 하면 통과**시키므로 전혀 다른 장면이
"증거 있는 장면" 으로 편성된다. 예외도 경고도 없다.
현재는 v_id 별 max(scene_id) 가 발행 행수와 일치 — 우연히 맞아 있는 상태.

### 1-2. kind 필터·유사도 하한이 없어 한 종류가 top-K 를 점령한다 [실측]
필터는 `v_id` 뿐. v201 실측(top_k=20):
- "구자욱 홈런" → etc 15 / stt 5, orphan 10, 후보 장면 **4개**뿐
- "다이빙 캐치" → shot 17 / stt 3, orphan 4, 장면 10개
- "역전 장면" → stt 19 / shot 1, orphan 10, 장면 8개

컬렉션 전체 orphan 비율 59~70% 라 이 잠식은 상시적이고, `EVIDENCE_SCENES_MAX=8` 슬롯을
채우지도 못한다. 유사도 하한도 없어 무관한 질의에도 항상 20건이 "증거" 로 반환된다.

### 1-3. 정렬 키 `(-hits, -sim)` 가 OCR 변형을 이긴 것으로 만든다 [코드확인]
`merge_etc` 가 완전 일치만 병합하므로 같은 자막이 변형별로 여러 행이 된다. 그 장면은
"히트 3건" 으로 계산돼, 유사도가 훨씬 높은 단일 STT 히트 장면을 이긴다.

---

## 2. plan — `flow/graph.py`, `flow/plan.py`, `flow/prompts.py`

### 2-1. ★최우선★ 호출자 예산이 LLM 에 전달되지 않는다 [실측]
`graph.py:55-61` 이 프롬프트의 `{budget}` 자리와 `plan_user` 인자에 **둘 다
`plan.DEFAULT_BUDGET_SEC`(180) 를 하드코딩**한다. `st["budget"]` 은 응답 파싱 **후**
덮어쓰기만 한다(`:72`). 게다가 `plan_user` 의 `budget` 파라미터는 f-string 본문에서
한 번도 쓰이지 않는 **죽은 인자**다.

실측 로그 — 900 초로 요청한 모든 편성에서 LLM 응답이 `예산: 180`:
```
plan 응답: '모드: collection\n대상: ...\n예산: 180\n선곡: 3, 9, 15, ...'
```
2026-08-19 에 만든 편성 5건 전부 모델이 180 초짜리를 만든다고 믿고 고른 결과다.
v201 이닝 커버리지 9/18 도 프롬프트 문제가 아니라 이 배선이 원인일 공산이 크다.

수정 지점은 프롬프트가 아니라 graph 배선 — `tests/test_flow_deterministic.py` 가
`plan_user` 출력의 bench4 byte 등가를 고정하고 있으므로 프롬프트 본문은 건드리지 말 것.

### 2-2. 응답 형식 검증이 없어 LLM 장애가 "장면 없음" 으로 둔갑 [코드확인]
`parse_verify` 에는 `startswith("판정")` 가드가 있는데 `parse` 에는 없다. 빈 문자열·산문을
받아도 기본 spec + `picked=[]` 로 통과 → `status=empty` 저장. **LLM 장애와 "정말 장면이
없음" 이 결과상 구분되지 않는다.**

### 2-3. 예산 파싱이 첫 숫자만 집는다 [코드확인]
`re.search(r"\d+", line)`. `"예산: 1분 30초"` → **budget=1** → 어떤 클립도 못 들어감 →
재선곡 → `status=empty`. 로그 어디에도 예산 파싱이 원인이라는 단서가 없다.
상한 검증도 없어 `"예산: 99999"` 도 통과한다.

### 2-4. 선곡 번호가 중복 제거되지 않는다 [코드확인]
`plan.py:60` 의 `picked` 에도, `rank.order`(`rank.py:26`)에도 dedupe 가 없다.
`선곡: 3, 7, 3` → 장면 3이 EDL 에 두 번, 예산도 두 번 소모. `_apply_endfix` 의 `by_id` 는
한 사본만 수정해 두 클립의 끝이 서로 달라진다. (현 편성 데이터에 실제 중복은 없었음.)

### 2-5. `<think>` 미종료 시 사고 과정이 답으로 파싱된다 [코드확인]
`_THINK` 정규식이 닫는 태그를 요구한다. 사고가 `max_tokens` 에 잘려 `</think>` 가 없으면
치환이 일어나지 않고, `text` 가 비지 않아 "본문 없음 → thinking 끄고 재시도" 안전망도
발동하지 않는다. 2-2 와 겹쳐, 사고 과정 안의 후보 나열이 최종 선곡이 될 수 있다.
`finish_reason` 을 어디서도 보지 않는다. 재시도 경로는 `max_tokens=512` 로 떨어져
인벤토리가 큰 경기에서 `선곡:` 줄이 잘려도 조용히 "선곡 없음" 이 된다.

---

## 3. cutrank / route / backfill — `flow/rank.py`, `flow/cut.py`, `flow/graph.py`

### 3-1. 장면 하나가 예산보다 길면 "장면 없음" 으로 끝난다 [코드확인]
`_assemble` 은 `total + d <= budget` 이라 예산보다 긴 클립을 전부 `spare` 로 보낸다.
후보가 그것뿐이면 `picked=[]` → feedback → 재선곡 → 같은 결과 → `status=empty`.
게다가 `feedback_node` 메시지가 사실과 어긋난다 — 원인은 검산이 아니라 예산 절단인데
"어휘를 다시 번역하라" 고 지시하므로 재선곡이 문제를 고칠 수 없다.

### 3-2. `spare` 가 남았다는 이유로 심한 미달에서도 backfill 을 건너뛴다 [코드확인]
`underfill = (not spare and total < budget*0.7)`. 주석은 "예비가 남으면 예산이 한계였던
것" 이라 하지만, 예비가 남는 이유는 "길어서 안 들어간 것" 일 수도 있다.
풀이 `[200s, 20s]`·예산 180 이면 총 20 초짜리 결과가 나간다(미달률 11%).

### 3-3. backfill 의 조용한 무동작 [코드확인]
`view != "전체"` 이거나 `targets` 가 비면 아무것도 안 하고 반환하는데 로그가 없다.
route 는 "예산 미달 … backfill 만" 을 찍어놔서, 로그만 보면 충원을 시도한 것처럼 읽힌다.
`targets` 가 어휘와 대조되지 않아 LLM 이 "장타" 같은 어휘 밖 표현을 쓰면 매칭 0건인데,
그것도 로그가 없다(`if added:` 라 0건이면 무음).

### 3-4. rank 가산의 비대칭과 공백 [코드확인]
라벨은 `sum`, 태그는 `max` — 규칙이 다른데 문서화가 없다(bench4 계승).
`RANK_LABEL_BONUS` = 역전4·동점2·적시타1·병살2 뿐이라 **삼중살·견제사·도루사·진루타·
밀어내기·경기 종료·끝내기는 전부 0**. 야구에서 가장 드문 삼중살이 병살보다 낮게 평가된다.
`RANK_TAG_BONUS` 에도 견제·보크·실책이 없다.

예: 9회 말 삼중살(라벨 삼중살, 태그 범타) = `delta*3 + 0 + 0 + 3`.
같은 이닝 평범한 적시타(delta=1) = `3+1+0+3 = 7` 로 앞선다 → 삼중살이 spare 로 탈락.

### 3-5. 견제·견제사가 컷 레시피에서 통째로 빠졌다 [코드확인]
`CUT_RECIPE` 에 `견제` 없음 → `allow=∅` → 앵커 투구 샷 하나만.
`LABEL_EXTRA_SHOTS` 에 `견제사` 없음 → 주루 샷 보강도 없음.
결과: 견제사 장면이 "투수가 공 던지는 3초" 로 나가고 정작 주자 태그아웃이 빠진다.
`삼중살` 도 `LABEL_EXTRA_SHOTS` 에 없다(병살만 있음) → 여운 샷 미부착.

### 3-6. `LABEL_EXTRA_MAX_SEC` 가 부분 상한이 아니라 체인 종결자 [코드확인]
상한을 넘는 extra 샷은 채택되지 않고 `else: break` 로 떨어져 **그 뒤 `allow` 샷까지
통째로 끊긴다**. 상한 초과 샷 하나가 뒤따르는 정상 주루 샷을 같이 죽인다.

### 3-7. `rank.score` 의 `max()` 빈 시퀀스는 "우연히" 막혀 있다 [코드확인]
`r["tags"]` 가 빈 리스트면 `ValueError` 로 편성 전체가 죽는다. 지금 안전한 이유는
두 겹의 우연 — vision3 가 태그 전멸 시 `[판별불가]` 로 대체하고, `repos.py` 가
`(scene_type or "").split(",")` 로 `[""]` 를 주기 때문. `label_list` 는 빈 리스트를
명시적으로 방어하는데 tags 만 안 하는 비대칭.

---

## 4. endfix — `flow/graph.py`

### 4-1. 예산 재검사가 전혀 없다 [실측]
`_assemble` 은 예산을 엄격히 지키는데, endfix 는 클립마다 최대 `ENDFIX_MAX_EXT_SEC=12`
초를 늘리고 `total` 을 **재계산만** 할 뿐 예산과 비교하지 않는다. 이후 검사도 없다.

실측 — 예산 900 초 요청의 결과 길이:
| comp | v_id | 결과 | 초과 |
|---|---|---|---|
| 11 | 200 | 949s | +49 |
| 12 | 201 | 947s | +47 |
| 13 | 202 | 964s | +64 |
| 14 | 203 | 1018s | +118 |
| 15 | 1003 | 977s | +77 |

초과를 알리는 로그도 없다.

### 4-2. 장면 끝 상한도, 클립 간 겹침 검사도 없다 [코드확인]
후보·수용 조건 모두 `ce + 12s` 만 본다. 연속 두 장면이 채택되고 간격이 5 초일 때 앞
클립이 12 초 연장되면 뒤 클립과 7 초 겹친다 → 같은 초가 EDL 에 두 번. `_assemble` 은
정렬만 하고 겹침을 보지 않는다.

---

## 5. verify — `flow/graph.py`, `flow/plan.py`

### 5-1. "소견 전용" 단계의 실패가 완성된 편성을 통째로 날린다 [코드확인]
`llm.chat` 은 `raise_for_status()` 로 HTTP 오류를 전파한다. verify 는 모듈 docstring 이
"소견 전용 — 기각권 없음" 이라 명시했고 endfix 도 미세 보정인데, 둘 중 하나만 실패해도
`COMPOSE_ERROR` 가 되고 **이미 확정된 클립이 저장되지 않는다**.
retrieve 실패 전파는 설계 결정으로 명시돼 있지만, 이 두 단계의 전파는 근거가 없다.

예: plan·cutrank 정상 완료(70클립 확정) → GPU 야간 중지로 verify 타임아웃 → 결과 0건.

### 5-2. `_packets` 만 score 가드가 없다 [코드확인]
`graph.py:308` 이 `c['score'].split()[1]` 을 무방비로 쓴다. `plan.render_inventory` 와
`_clip_row` 는 `if r["score"] else "?"` 로 방어한다. score 가 NULL 인 행 1건이 채택되면
verify 직전에 예외 → 5-1 과 결합해 편성 전체 실패. 같은 행이 인벤토리 렌더에서는
"?" 로 무사히 지나간다.

### 5-3. `판정` 필드는 소비되지 않고 기각 번호는 검증되지 않는다 [코드확인]
`VERIFY_SYSTEM` 이 `판정: ok|부족` 을 요구하지만 코드는 존재 여부만 보고 값을 버린다.
`rejected` 의 scene_id 를 채택 클립과 대조하지 않아, 인벤토리에 없는 번호가 그대로
API 응답 `suspicions` 에 실린다. plan 의 선곡은 ghost 제거·경고까지 하는데 여기는 안 함.

### 5-4. `기각:` 줄 자체가 공통 사유로 섞인다 [코드확인]
`기각:` 분기가 `continue` 없이 흘러가 `common.append(line)` 에 걸린다. 개별 사유가 없는
장면은 폴백으로 `"기각: 12, 34"` 가 사용자에게 노출된다. docstring("매핑 안 되는 **사유**
줄은 공통 소견으로 강등")과 어긋난다 — 기각 줄은 사유 줄이 아니다.

---

## 6. 저장 — `db/compose_repo.py`, `db/pool.py`

### 6-1. 트랜잭션이 없는데 있는 것처럼 보인다 [코드확인]
풀이 `autocommit=True` 라 `await conn.commit()` 은 사실상 no-op. `BEGIN` 은 어디에도 없다.
헤더 INSERT 가 즉시 확정 커밋되고 클립 `executemany` 가 별개로 커밋되므로, 중간 실패 시
`clip_cnt=200, duration=1800, status='ok'` 인데 클립 0행인 **고아 헤더**가 남는다.
완화: 그 comp_id 로 렌더하면 `not comp["clips"]` 에 걸려 409 — 빈 mp4 는 안 만들어진다.
문제는 헤더 통계가 거짓말하는 행이 DB 에 영구히 남는 것, 그리고 `commit()` 이 코드에
보여서 트랜잭션이 있는 것처럼 읽히는 것.

### 6-2. ValueError 를 전부 "발행본 없음" 으로 뭉갠다 [코드확인]
`code = COMPOSE_ERROR_SOURCE if isinstance(e, ValueError) else COMPOSE_ERROR`.
의도한 건 "t_scene_baseball 비어 있음" 하나인데, 그래프 어디서 나온 ValueError 든
(예: 렌더 응답 JSON 깨짐 → `JSONDecodeError`) 4910 "발행본 없음(publish 선행 필요)" 이
된다. 운영자는 publish 를 다시 돌리는 헛수고를 한다.

---

## 7. 렌더 — `api/render.py`, `render/payload.py`, `render/client.py`

> 이 절은 미커밋 작업분(동기 → 비동기 접수 + 폴러 전환) 기준.

### 7-1. 접수 성공 후 스탬프 실패 시 감시자가 안 뜬다 → 중복 GPU 렌더 [코드확인]
워커 접수 성공 → `mark_render_started` 가 DB 순간 장애로 예외 → `except BaseException` 이
`_RENDERING` 을 pop 하고 re-raise → `background.add_task(_watch, ...)` 가 **실행되지 않는다**.
워커는 이미 렌더 중인데 서버엔 흔적이 없어, 재시도하면 같은 c_id 를 동시에 두 번 렌더한다.

### 7-2. 원샷(compose render=true)이 중복 방지 게이트를 전부 우회한다 [코드확인]
`render.py` 모듈 docstring 은 "이미 렌더된·진행 중 편성은 워커 호출 전에 4xx 로 차단"
이라 선언하지만, `_render_after` 는 `_RENDERING` 등록도 조회도 하지 않고 `render_datetime`
도 보지 않는다. 원샷이 렌더 중인 사이 UI 가 `POST /render` 를 걸면 이중 렌더.

### 7-3. 원샷 성공 판정이 화이트리스트가 아니라 블랙리스트 [코드확인]
`status not in ("error","skipped")` 이라 워커가 `{"status":"accepted"}` 를 주면 그것도
성공으로 간주해 `mark_rendered` 가 돈다. 존재하지 않는 mp4 에 `render_datetime` 이 찍히고
이후 렌더는 409 로 영구 차단된다(force 를 아는 운영자만 탈출). `_watch`/`get_render` 는
`== "done"` 을 쓴다 — **같은 판정에 기준이 둘**인 것 자체가 신호.

### 7-4. force 재렌더 중 GET 이 옛 완료 시각으로 "done" 을 답한다 — **조치 완료** (`445a1e0`)
`force=True` 가 `render_datetime` 을 비우지 않아, 재렌더 수 분 동안 GET 이 즉시 단락되어
이전 `rendered_at` 과 함께 `status:"done"` 을 돌려줬다. 뷰어가 덮어써지는 중인 파일을 튼다.
→ DB 지름길 조건에 `comp_id not in _RENDERING` 을 붙여, 진행 중이면 워커에 직접 묻는다.
README 의 "`render_datetime` NOT NULL = 영상 준비됨" 계약이 이제 진행 중 구간에서도 유지된다.

### 7-5. 원샷 경로의 DB 예외가 편성 성공을 error 로 뒤집는다 [코드확인]
`try` 가 `httpx.HTTPError` 만 잡는다. `st.status.set` 은 예외를 삼키지만
`mark_render_started` 는 그대로 올라가 잡이 error 로 덮인다. 편성은 저장돼 comp_id 가
존재하는데 응답엔 comp_id 가 없다. docstring 의 "렌더 실패가 편성 성공을 뒤집지 않는다"
가 이 경로에서 거짓.

### 7-6. `_RENDERING` 이 영구 잠금이 될 수 있다 [코드확인]
해제가 `_watch` 의 `finally` 하나뿐인데, `_watch` 본체가 `try` 진입 **전** 줄에서 예외를
내면 해제되지 않는다. 그 comp_id 는 프로세스 재시작까지 모든 렌더가 409.

### 7-7. 길이 0 클립이 렌더 요청에 들어간다 [코드확인]
`api/compose.py:215` 가 `int(cs)`·`int(ce)` 로 **양쪽 다 내림**한다. cut 좌표는 소수 초라
`cs=100.4, ce=100.6` → `100, 100`. `payload.build_request` 에 `start < end` 검증이 없어
`{"start_sec":100.0,"end_sec":100.0}` 이 그대로 워커에 간다.
근거가 코드 주석에 이미 있다 — `cut.py:33` 의 "v203 장면 6: 0.2초 볼넷 클립".
부수 효과: `ce` 내림으로 항상 최대 1초 꼬리가 잘린다.

### 7-8. 이닝 NULL 1건이 편성 전체의 렌더를 막는다 [코드확인]
클립 하나라도 이닝 파싱 실패면 422 `COMPOSE_INVALID_INNING`, 부분 렌더 경로가 없다.
vision3 는 `inning_of.get(board_sec)` 가 비면 NULL 로 발행하고, compose 는 선곡 단계에서
걸러내지 않는다. 편성은 정상 저장되고 **렌더 버튼을 누르는 순간에만** 실패하며,
재편성해도 같은 장면이 뽑히면 계속 막힌다.

### 7-9. output_path 를 DB 에 기록하지 않는다 [실측]
워커가 돌려주는 경로는 로그와 일시적 API 응답에만 있다. `mark_rendered` 는
`render_datetime` 만 찍는다. 소비 측이 파일 경로를 알 방법이 없어 명명 규약을
재구현해야 하고, 워커가 규약을 바꾸면 조용히 깨진다.

실제로 2026-08-19 에 t_compose 를 비운 시점에 gpu-00 의 mp4 8건(6.1GB)이 **출처를 잃었다.**

---

## 8. 색인 (ingest) — `api/ingest.py`, `vector/ingest.py`, `vector/store.py`

### 8-1. `replace` 가 원자적이지 않다 — 반쪽 색인이 그대로 검색된다 [코드확인]
`delete` 후 500 건 단위 insert 루프 중간에 예외가 나면 롤백이 없다. 상태는 4920 으로
남지만 **compose 검색은 그 반쪽 색인을 아무 검사 없이 계속 쓴다**.
유발 트리거가 바로 옆에 있다 — `zip(rows, vecs)` 에 **길이 검증이 없어** 임베딩 응답이
한 건이라도 모자라면 뒤쪽 행에 `vector` 키가 없는 채로 insert 에 들어간다.
`embed_docs` 에 재시도가 없어 vLLM 일시 503 하나로 전체 색인이 4920 이 된다
(v201 = 2,178행 / batch 32 → 약 68회 연속 호출이 전부 성공해야 함).

### 8-2. 0 길이 증거는 절대 귀속되지 않고 STT 는 이전 장면으로 오귀속 [실측]
`o > ov` (strict) + 초기 `ov=0.0` 이라 `s == e` 인 증거는 겹침이 항상 0 → 귀속 실패.
실측: `t_dialogue` 에 `start_time = end_time` 인 발화가 영상당 1~4건 존재
(200:4, 201:1, 202:2, 203:4, 1002:1, 1003:1). 장면 40 한복판의 발화가 겹침 0 이 되어
STT 폴백으로 **직전 장면(39)에 귀속**된다 — orphan 보다 나쁜 조용한 오귀속.
경계 정확 일치(e == scene.s)도 같은 이유로 orphan. 동점 겹침은 낮은 scene_id 가 이긴다
(결정적이지만 근거 없는 편향, 문서화 없음).

### 8-3. `GET /api/v1/ingest` 행 수 과다 보고 [실측]
`get_collection_stats().row_count` 는 삭제 표시만 되고 컴팩션 전인 엔티티를 포함한다.
실측 `row_count = 11,995` vs 실제 조회 가능 **10,468**. 1,527 행이 유령이다.
v_id 지정 경로(`query count(*)`)는 정확하다.

### 8-4. v_id 색인 삭제 경로가 없다 + 발행이 비면 낡은 색인이 살아남는다 [코드확인]
삭제는 `replace` 내부 한 곳뿐, DELETE 라우트 없음. 영상이 MySQL 에서 지워져도 색인은 남는다.
더 나쁜 조합: 재발행 실패로 발행본이 0행이면 `ValueError` 가 **`replace`(=delete) 이전에**
터져 "발행본은 없는데 색인은 옛 장면 그대로 검색되는" 상태가 고정된다.

### 8-5. kind 통째 누락에 아무 신호가 없다 [실측]
v1002 는 컬렉션에 **etc 0건**(shot 765 / stt 364). ETC 자막이 비어도 경고 없이 성공하고
요약 `by_kind` 에 키가 빠질 뿐이다. 인물 질의의 주 재료가 통째로 없는 색인이 "정상 완료".

### 8-6. 연장 경기가 들어오면 즉시 터지는 절단 [실측(스키마)]
`inning` 필드 한도 8 바이트. `"10회 초"` 는 **9 바이트** → 절단 + `errors="ignore"` 로
`"10회 "` 가 되어 **초/말이 사라진다**. 현재 색인은 9회까지라 미발현.
같은 축의 잠재 건:
- `shot_type` 은 **유일하게 `_trunc_bytes` 가 안 걸린 VARCHAR**(한도 16B). 실측 최대 14B
  ("타구·수비")라 아슬아슬 통과 — vision3 `SHOT_TYPES` 에 6자 이상 유형이 하나 추가되면
  insert 가 거부되고 색인 전체가 실패한다.
- `labels` 컬럼은 varchar(100) 인데 Milvus 필드는 64B — 스키마상 초과 가능(실측 최대 16B).
- 절단이 발생하면 `"역전,병"` 같은 **반토막 태그**가 되는데 경고가 없다.
- `repos.py` 의 `h_id`·`start_ms`·`end_ms`·`scene_type` 은 스키마상 NULL 허용(실측 0건).
  NULL 이면 `float(None)` 또는 Milvus insert 실패로 색인 전체가 죽는다.

---

## 9. 운영·재기동

### 9-1. ★재기동이 진행 중 잡을 10초 만에 죽인다★ [코드확인]
`run.py` 의 `timeout_graceful_shutdown=10` + Starlette `BackgroundTasks`.
plan thinking 만 6분 넘게 걸리는데(Qwen3.8 실측), 배포 한 번이 진행 중 편성을 취소한다.
그때 `_run` 은 `CancelledError` 를 **상태를 손대지 않고 re-raise** 하므로
`t_video.status_code` 가 4020~4050 인 채 영원히 남고, `_JOBS` 도 사라져 comp_id 를
회수할 수 없다. ingest 도 4010 고착.

**비대칭이 핵심**: render 만 `GET /render/{comp_id}` 보정 경로가 있어 되살아난다.
compose·ingest 에는 그런 경로도, 부팅 시 "진행 중 상태로 남은 v_id 정리" 도 없다.

2026-08-19 에 compose 를 두 번 재기동했는데 마침 돌던 잡이 없어 사고를 면했다.

### 9-2. 설정 검증 공백 [코드확인]
- `config.py` docstring 은 "DB 는 기본값 없이 필수 → 누락 시 부팅 실패" 라 하지만
  실제로는 `db_ip="127.0.0.1"`·`db_pw=""` 기본값이 있어 누락해도 부팅한다.
- `render_poll_interval` 에 하한이 없다 — `0` 이면 워커를 초당 수천 회 두드린다.
- `render_timeout` 하나가 httpx 클라이언트 타임아웃(원샷 sync 대기)과 `_watch` 폴링
  데드라인을 겸한다. 성격이 달라 한쪽을 늘리면 다른 쪽이 끌려간다.
- `.env.example` 이 `EMBED_BASE_URL` 과 `RENDER_BASE_URL` 을 **같은 8003** 으로 안내한다.
  포트 정책상 embed 는 8500. 예시대로 배포하면 embed 요청이 렌더 워커로 간다.
- `.env.example` 의 `EMBED_BATCH=64` vs 운영 `.env` 32 — 2배 어긋남.

### 9-3. `.env` 임시 상태 [실측]
2026-08-19 에 `LLM_TIMEOUT` 을 240 → 900 으로 올려둔 상태다(Qwen3.8 의 plan 이 6분+).
되돌리면 매 편성이 타임아웃된다. 백업: sm-api-01 `/tmp/agent-compose.env.bak-1927`.

또한 GA(166.117.29.29) 경유로는 이 호출이 **구조적으로 불가능**하다 — Global Accelerator
TCP 유휴 타임아웃 340 초 < thinking 응답 시간. compose LLM 은 직접 IP 를 써야 한다.

---

## 10. 어휘 동기화 (2026-08-19 조치 완료)

vision3 가 정본이고 compose 는 복제본인데, 감시가 태그 한 방향뿐이라 라벨이 통째로 샜다.

- `보크` 태그 누락 → 추가 (`6c1527d`)
- `희생플라이` → `진루타` 개명 미반영. `LABEL_EXTRA_SHOTS` 키가 죽은 이름으로 남아
  **진루타 클립이 득점·홈인/리액션 여운을 못 붙였다** → 수정 (`6c1527d`)
- `삼중살`·`견제사`·`경기 종료` 누락 → 추가 (`6c1527d`)
- `끝내기` 누락(위 조치에서도 놓침) → 추가 + **양방향 등가 테스트** (`ff7f228`)
- `prompts.TAG_VOCAB` 하드코딩이 드리프트의 통로였다 → vocab 렌더로 전환 (`6c1527d`)

남은 것: 새 라벨들의 `RANK_LABEL_BONUS`·`LABEL_EXTRA_SHOTS` 가중은 **미정**
(실측 근거 없이 값을 바꾸지 않는다는 vocab.py 원칙 — 3-4·3-5 항목에서 함께 결정).

---

## 11. 문서·주석이 코드와 어긋난 곳

- `plan.py:4` "LLM 호출은 nodes 가 한다" / `prompts.py:70` "(nodes._apply_endfix)"
  — `flow/nodes.py` 는 **존재하지 않는다**. 전부 `flow/graph.py` 에 있다.
- `render/client.py` 클래스·메서드 docstring 이 "동기 완주" 계약 그대로 — 미커밋 작업분에서
  기본이 비동기 접수로 바뀌었다(모듈 docstring 만 갱신됨).
- `render/payload.py:54` 산출물을 `c_{c_id}.mp4` 로 적었으나 실제 파일은
  `{v_id}_{c_id}.mp4`(실측 `201_16.mp4`). README 쪽이 맞다.
- `rank.py:19` "이닝 가중 — 후반일수록 +1~+4" — 실제는 `inning_no // 3` 이라
  1~2회=**0**, 9회=3. +4 는 12회 이상에서만.
- `store.py:79` "delete-insert 멱등 — DB 관례와 동일" — 멱등은 맞지만 **원자적이지 않다**(8-1).
- `store.py:102` search 반환 설명에 `h_id` 누락. 실제 `output_fields` 에도 없다.
  색인은 `h_id` 를 채우는데 검색 경로에서 한 번도 읽히지 않는 **죽은 필드**.
  `tags`·`labels`·`shot_type`·`inning` 도 `output_fields` 엔 있으나 `_group_hits` 가
  `text`·`scene_id`·`distance` 만 쓰고 버린다 — "SQL 재조회 없이 프롬프트에 렌더한다" 는
  vector-collection.md §3 의 명분이 현재 코드에서 실현되지 않았다.
- `repos.py:61` "caption 있는 행만" 인데 SQL 은 `summary IS NOT NULL` 뿐 — 빈 문자열 통과
  (실측 0건이라 무해).
- `.aidoc/vector-collection.md` 가 **구 테이블명**(`t_scene`, `t_frame_board_detail`)을 쓴다.
  실적 수치도 낡음 — 문서 "v201 2,095행(shot 524)" vs 실측 **2,178행(shot 607)**.
- `compose.py` docstring 의 "읽기 전용 + comp_id 신규 발급이라 충돌 없음" — 0-3 참조.
- `render.py` 모듈 docstring 의 "워커 호출 전에 4xx 차단" — 7-2 참조.
- `compose_repo.py:79` "실패는 삼키지 않는다" — repo 계약으로는 맞지만 호출부 두 곳이
  전부 잡아 4960 으로 바꾼다.
- `compose_repo.py` 가 `reg_datetime` 만 `str()` 정규화하고 `render_datetime` 은 datetime
  그대로 반환 — `GET /compose` 와 `GET /render/{comp_id}`(isoformat)의 시각 형식이 다르다.

---

## 12. 테스트 공백

`tests/` 에 `test_flow_*`·`test_ingest`·`test_render_payload` 만 있고 **`src/api/` 의 상태
기계를 검증하는 테스트가 하나도 없다**. 미커밋 작업분의 위험이 거의 전부 거기(잡 수명,
`_RENDERING` 게이트, 스탬프 판정, 취소 경로)에 몰려 있는데 회귀 그물이 없다.

---

## 12-A. 재설계 판단용 기준선 (2026-08-19, 예산 배선 수정 **이전**)

> `t_compose` 를 비웠으므로 이 표가 유일한 기록이다. 2-1(예산 미전달) 수정 후 같은 질의로
> 재실행해 이 값과 비교하는 것이 "현 flow 가 최적인가" 를 판정하는 단일 실험이다.

질의 `이닝별 최고의 공격/수비 장면`, budget=900, render=true (comp 16 만 질의에
"1회부터 9회까지 모든 이닝을 골고루 담을 것" 추가).

| comp | v_id | 모드 | 결과 | 초과 | 클립 | 이닝 커버 | 모델 |
|---|---|---|---|---|---|---|---|
| 11 | 200 | collection | 949s | +49 | 38 | 17/18 | Qwen3.6 |
| 12 | 201 | collection | 947s | +47 | 18 | 9/18 | Qwen3.6 |
| 13 | 202 | collection | 964s | +64 | 27 | 13/17 | Qwen3.6 |
| 14 | 203 | collection | 1018s | +118 | 42 | 17/18 | Qwen3.8 |
| 15 | 1003 | compose | 977s | +77 | 25 | 12/18 | Qwen3.8 |
| 16 | 201 | collection | 940s | +40 | 21 | 11/18 | Qwen3.8 |

**단일 변수 비교가 가능한 유일한 쌍은 comp 16 ↔ 재실행**(둘 다 Qwen3.8, 같은 v201).
18:37 에 gpu-01 모델이 Qwen3.6 → Qwen3.8 로 교체됐으므로 comp 11~13 과의 비교에는
모델 교체가 섞인다.

### v201 선곡 품질 실측 (comp 16 기준)

- 발행 71건 중 **득점 발생 장면 15건**. 편성 포함 13건, 누락 2건:
  - `#26` 4회 말 안타·적시타 1-3 (72초)
  - `#57` 7회 말 안타·적시타 9-5 (**166초** — 발행분 최장, 예산 압박에 밀린 것으로 추정)
- 빠진 7개 이닝(1말·2말·3초·4초·5말·6말·8초)은 **전부 득점 0** — 정당한 제외로 확인.
- 6회 초 5클립·7회 말 5클립 쏠림도 실제 득점 구간(삼성 6점·롯데 4점) — 정확한 판단.

→ 선곡기의 판단 자체는 작동하고 있다. 실패는 입력(예산)과 어휘 드리프트에서 왔다.

### 재실행 시 비교할 지표

1. 이닝 커버리지 (11/18 대비)
2. **득점 장면 포함률** (13/15 대비 — #26·#57 이 들어오는가)
3. 예산 준수 (940s / +40 대비)
4. 클립 수·길이 분포 (21클립, 평균 45초)

---

## 13. 레포만으로 확인 불가 (남은 확인 과제)

- `render_datetime`·`bumper_yn` 의 **마이그레이션 파일이 레포에 없다**(`sql/` 디렉터리 자체가
  없음). 운영 DB 에는 두 컬럼이 **실재함을 확인**했으나, ALTER 없이 코드만 배포하는 순서
  사고가 나면 `head["render_datetime"]` KeyError → 500 `INTERNAL_ERROR` 로 원인이 숨는다.
  vision3 는 `sql/migration_*.sql` 로 추적한다 — 같은 관례를 도입할지 결정 필요.
- worker-render 의 상태 엔드포인트가 `sync_yn=True` 잡이나 워커 재시작 이후 무엇을
  돌려주는지. 워커가 잡 이력을 잃고 404 를 주면 완료된 렌더를 `not_requested` 로 답해
  뷰어가 재렌더를 건다.
- 6-1 의 병합(consolidate) 장면 `obs_sec` 문제(추측): vision3 병합이 원장 `t_play_baseball`
  을 다시 쓰지 않아 `_obs_floor` 가 앞쪽 아웃 관측 시각을 쓴다. v201 병합 0건이라 미검증.
