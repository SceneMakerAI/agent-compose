# agent-compose 설계 (v0.1 — 2026-08-17)

질의 기반 하이라이트 편성 API. 원본: `poc/poc-search-bench4` (데모 검증 완료)의 서비스 승격.
소유 산출물: `t_compose`(편성 헤더) + `t_compose_clip`(클립 목록) + Milvus `sm_scene_evidence`.

## 1. 역할·경계

```
[상류] agent-vision3 publish → t_scene(source='board') 발행
          │ (발행 직후 HTTP 훅)
          ▼
POST /api/v1/ingest {v_id}     증거 수집→임베딩→Milvus 교체 (v_id 단위 멱등)
POST /api/v1/compose {...}     질의 → LangGraph flow → t_compose(_clip) + 응답
```

- **소비만 하는 것**: t_scene / t_segment(scene-cut, caption) / t_dialogue /
  t_frame_board_detail(ETC). seg_id 조인 금지(양쪽 delete-insert — ID 어긋남).
- **쓰는 것**: t_compose·t_compose_clip (요청마다 새 comp_id — 이력 보존),
  Milvus 컬렉션(v_id 단위 delete-insert).
- 어휘(태그·라벨)의 원천은 vision3 `sports/baseball/vocab.py` — bench4 는 prompt 문자열과
  CUT_RECIPE 키 두 곳에 하드코딩돼 드리프트 확정 상태였다. 이식 시 단일 원천으로 통합한다
  (방법: vocab 공유 패키지화 vs 복제+회귀 테스트 — 이식 단계에서 결정).

## 2. bench4 대비 확정 변경 (서비스 승격 결정)

1. **fail-open 폐기** — Milvus·embed 접속 불가를 조용히 통과시키지 않는다.
   /readyz 가 DB·embed·Milvus 3종을 드러내고, 실패는 예외로 전파.
   (POC 실측: localhost 미러 URI 오류에도 "벡터 없는 모드"로 무음 저하)
2. **클라이언트 lifespan 공유** — POC 는 호출마다 OpenAI/Milvus 생성·미반납(누수).
   Database 풀·Embedder·VectorStore 를 app.state 1식으로.
3. **Milvus 는 원격**(sm-db-01:19530) — localhost 는 개발 미러였다.
4. **색인은 API**(ingest) — 수동 스크립트(tools/index_evidence.py) 대체.
   vision3 발행 훅이 호출, 실패해도 발행은 유효(재색인 가능).
5. ETC 색인의 BASE 정규형 전제 등 판독 계약은 vision3 baseball.md 상류 계약을 따른다.

## 3. LangGraph flow (bench4 현행 — 이식 대상)

```
START → retrieve → plan★ → cutrank → endfix★ → route ─┬→ verify★ → END
        (벡터검색)  (선곡)   (컷+예산)  (끝보정)         ├→ backfill → verify★ → END
                     ▲                                  ├→ feedback → plan (0건 재선곡 1회)
                     └──────────────────────────────────┘└→ empty → END
```

LLM 콜: happy path 2~3회(plan+verify, endfix 시 +1) + 임베딩 1회.
원칙: **LLM 이 제안, 사실이 처분** — 선곡은 실존 scene_id 검산 통과분만, endfix 는
검증기 수용 발화만 제시(±1s 일치·연장 상한), 마지막 채움(backfill)은 결정적 계산.

### 이식하며 고칠 결함 (bench4 실사 2026-08-17)

- **(A1) backfill 클립이 endfix 를 못 받음** — 노드 순서 재배치 (endfix 를 route 뒤로).
- **(A2) verify 의심 사유가 클립별 아님** — 파서가 마지막 한 줄을 전 클립에 복사.
- **(A3) backfill 이 state.total 미갱신** — 시한폭탄.
- **(B4) 공유 가변 상태** — cutrank/endfix 가 scenes dict 를 in-place 수정.
  State 를 불변(행 복사)으로 재설계 — 서버 동시 요청의 전제.
- (B6) retrieve 1회 단방향 — 질의 재작성 루프는 등가 이식 검증 후 별도 단계.
- (B7) verify 는 기각권 없음(삼진 22건 오기각 실측 — 되돌리려면 반증 필요).
  콜 값이 낮아 비동기/샘플링 후보.

### 재설계 제약 (bench4 실측 확정 — 반증 없이 못 되돌림)

verify 기각권 박탈 / 스테일 투구 승격 금지·임의 백오프 금지 / STT 는 원문만·대사 조회는
장면 전체 구간 / seg_id 조인 금지 / **LLM 출력 JSON 금지(줄 형식)** / 마지막 채움은 계산.

## 4. Milvus 색인 (sm_scene_evidence)

kind 3종: shot(t_segment.summary) / stt(발화 청크 병합 — 파편 색인은 실측 실패) /
etc(ETC 자막 런 병합 — 완전 일치만). scene_id 귀속은 시간 겹침 최대치 + STT 여운
귀속(+30s). 스키마·상수 근거는 `src/vector/{store,ingest}.py` docstring.
질의만 instruction 프리픽스(Qwen3-Embedding 권장), 문서는 원문.
차원 2560(Qwen3-Embedding-4B) — 모델 교체 = 컬렉션 재생성.

## 5. 로드맵

- **Phase 1 (완료 2026-08-17)**: 골격(config/log/db/vector/api) + ingest API 실색인 검증.
- **Phase 2**: compose flow 이식 (A1~A3, B4 수정 포함) — bench4 산출과 등가 검증
  (같은 질의·같은 v_id 로 클립 목록 대조, LLM 비결정 필드 제외).
- **Phase 3**: vision3 발행 훅 연결, sm-api-01 배포(systemd), t_compose 소비자(ui) 확인.
- **Phase 4**: flow 개선(B6 질의 재작성, B7 verify 재배치) — 등가 검증 후.
