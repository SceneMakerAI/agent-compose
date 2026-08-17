# Milvus 컬렉션 `sm_db/sm_scene_evidence` — MySQL 원천 기준 명세

색인은 MySQL 발행 산출의 **파생 캐시**다 (진실은 MySQL, 컬렉션은 v_id 단위 delete-insert
로 언제든 재생성). 이 문서의 예시는 전부 v201(MBC 삼성 vs 롯데) 실데이터다 (2026-08-18).
설계 근거(왜 증거를 색인하나·왜 비정규화하나)는 design.md §4, 상수 근거는
`src/vector/ingest.py` docstring.

## 1. 원천 테이블 → 색인 행 매핑

```
MySQL                                          Milvus sm_scene_evidence
────────────────────────────────────────       ─────────────────────────────
t_scene (source='board')          ──귀속 기준──▶ scene_id·h_id·tags·labels·
  발행 장면 (시간 구간·태그·라벨)                 score_delta·inning (플랫 복사)
t_segment (scene-cut, summary≠∅)  ──kind=shot─▶ 샷 캡션 1행 = 색인 1행
t_dialogue                        ──kind=stt ─▶ 인접 발화 병합 청크 = 색인 1행
t_frame_board_detail (kind='ETC') ──kind=etc ─▶ 동일 자막 런 병합 = 색인 1행
```

v201 실적: shot 524 + stt 706 + etc 865 = 2,095행 (장면 귀속 754 / orphan 1,341).

## 2. kind 별 변환 — 실데이터 예시

### shot — `t_segment.summary` (1행 = 1행, 변환 없음)

MySQL (scene-cut 샷, VLM 캡션):

| s | e | shot_type | summary |
|---|---|---|---|
| 10977 | 10982 | 타구·수비 | 야수가 외야 잔디밭에 엎드려 다이빙 캐치를 시도하는 장면 |

→ Milvus 행 (장면60에 귀속돼 장면 메타가 붙는다):

```json
{"kind": "shot", "s": 10977.0, "e": 10982.0, "scene_id": 60, "h_id": 61,
 "shot_type": "타구·수비", "tags": "범타", "labels": "", "inning": "8회 초",
 "text": "야수가 외야 잔디밭에 엎드려 다이빙 캐치를 시도하는 장면", "vector": [2560차원]}
```

### stt — `t_dialogue` (인접 발화 병합 청크)

MySQL 발화는 문장 단위로 잘게 쪼개져 있다. 간격 ≤2초·합계 ≤300자면 한 청크로 병합,
병합 후에도 6자 미만이면 폐기 (실측: 발화 단위 그대로면 "네"·"칠구" 같은 0.1~1s
파편이 코사인 상위를 점령). 예 — 장면60 앞 해설 여러 발화가 한 청크로:

```json
{"kind": "stt", "s": 10945.1, "e": 10972.9, "scene_id": 60, "tags": "범타",
 "text": "확실히 최근에 구자욱 선수를 보면 타석에 서 있는 동안에는 쉽게 아웃될 것
          같지가 않아요. 그러니까 오히려 안타를 칠 것만 같은... (병합된 4~5개 발화)"}
```

**여운 귀속 (stt 전용)**: 장면과 시간이 안 겹쳐도 직전 장면 끝 +30초 이내 시작이면
직전 장면에 귀속. 예 — 장면60은 10970~10982초인데, 다이빙 캐치 콜은 **11011초**
(장면 끝 +29초):

| MySQL t_dialogue | → Milvus |
|---|---|
| s=11011.2, "야, 이거 뭐 김호령, 박해민이 부럽지 않은 레이스의 멋진 다이빙 캐치가 나왔습니다." | scene_id=**60** 귀속 (orphan 아님) — "다이빙 캐치" 질의 1위 히트가 이 행 |

### etc — `t_frame_board_detail(kind='ETC')` (동일 자막 런 병합)

MySQL 은 2초 간격 프레임마다 같은 자막이 반복 판독돼 있다. **완전 일치**하는 연속
구간(간격 ≤3초)을 런 하나로 병합, 5자 미만 제외. OCR 변형은 별개 런으로 남긴다
(변형까지 뭉치면 타석 매치업 교체 순간을 잃는다):

| MySQL (idx, txt) | → Milvus 런 |
|---|---|
| 702·704·706·708 "2026 구자욱 .346 ▶ 10홈런 71타점 OPS.955" | 1행: s=702, e=709, text=그 자막 |
| 710 "...346 **/** 10홈런..." (OCR 변형) | 별개 1행 |
| 712 "▶ 구자욱 시즌 사직 .500 6안타 1홈런 7타점" | 별개 1행 |

인물 질의("구자욱 하이라이트")를 캡션(이름 환각 가능)·STT(전사 깨짐)가 아니라
자막 사실로 잡는 재료다.

## 3. 필드 명세 (스키마 원본: `src/vector/store.py::ensure_collection`)

| 필드 | 타입 | 채우는 값 (MySQL 근거) |
|---|---|---|
| id | INT64 PK auto | — |
| v_id | INT64 | 대상 영상. 모든 검색에 필터 강제, 색인도 v_id 단위 delete-insert |
| kind | VARCHAR(8) | shot / stt / etc |
| s, e | FLOAT | 증거 시간 구간(초). shot·stt 는 TIME_TO_SEC, etc 는 idx(=초) 런 |
| scene_id | INT64 | **t_scene 과 시간 겹침 최대치** 귀속 (stt 는 +30s 여운 폴백). 없으면 **-1**(orphan — 발행 누락 신호로 보존) |
| h_id | INT64 | 귀속 장면의 t_scene.h_id → t_play 원장 역추적 |
| shot_type | VARCHAR(16) | shot 만: t_segment.shot_type (투구/타구·수비/…) |
| tags | VARCHAR(64) | 귀속 장면의 t_scene.scene_type (쉼표 나열) 플랫 복사 |
| labels | VARCHAR(64) | t_scene.labels (역전·병살…) 플랫 복사 |
| score_delta | INT16 | t_scene.score_delta |
| inning | VARCHAR(8) | t_scene.inning ("8회 초") |
| text | VARCHAR(1024) | 증거 원문 — **임베딩 입력이자 검색 스니펫**. ⚠️ VARCHAR 한도는 **UTF-8 바이트** 기준 (실측 v200 실패) — `_trunc_bytes` 로 문자 경계 절단 |
| vector | FLOAT_VECTOR **2560** | Qwen3-Embedding-4B(text 원문, instruction 없음). 인덱스 AUTOINDEX·COSINE |

플랫 복사(tags~inning)는 Milvus 에 조인이 없어서다 — 검색 히트를 SQL 재조회 없이
프롬프트에 렌더한다. 복사본의 신선도는 "재발행 → 재ingest" 멱등 규약이 보장한다.

## 4. 갱신·검색 규약

- **쓰기**: `POST /api/v1/ingest {v_id}` — vision3 publish 직후 호출. 수집(§2 변환)
  → 임베딩(gpu 8003) → v_id delete-insert. 같은 v_id 동시 요청 409.
- **읽기**: 질의에만 `QUERY_INSTRUCT` 프리픽스 → v_id 필터 top-20 → scene_id 그룹핑
  (히트수→유사도) 상위 8장면을 plan 후보로. 유사도는 rank 점수에 섞지 않는다
  (벡터는 후보 발견까지 — LLM 이 제안, 사실이 처분).
- 위치: **`sm_db` 데이터베이스** (default 아님 — 팀 관례, 2026-08-18 이전 완료).
  구세대 sm_1024·sm_2560 은 별개(agent-search 레거시)로 보존 중.
