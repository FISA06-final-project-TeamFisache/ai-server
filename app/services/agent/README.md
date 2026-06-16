# AI 에이전트 (app/services/agent)

개인 재무 관리 AI **Pori**의 에이전트 모듈 모음입니다.
모든 에이전트는 LLM 기반이며, FastAPI 라우터(`app/routers`)에서 호출되어 백엔드/프론트로 결과를 반환합니다.

---

## 전체 아키텍처

```
                    ┌──────────────────────────────────────────────┐
   Spring 백엔드 ──▶ │             FastAPI (app/main.py)            │
   /프론트          │     라우터: asyncio.wait_for 타임아웃 가드     │
                    └──────────────────────────────────────────────┘
                          │            │            │            │
          ┌───────────────┘   ┌────────┘     ┌──────┘      ┌─────┘
          ▼                   ▼              ▼             ▼
   ┌────────────┐    ┌──────────────┐  ┌──────────┐  ┌──────────┐
   │ /portfolio │    │ /salary      │  │ /report  │  │/mini_... │
   │ /consultant│    │              │  │          │  │ /nag     │
   └─────┬──────┘    └──────┬───────┘  └────┬─────┘  └────┬─────┘
         │                  │               │             │
         ▼                  ▼               ▼             ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  app/services/agent (에이전트 계층)            │
   │                                                              │
   │  LangGraph 그래프            단일/병렬 LLM 에이전트            │
   │  ┌────────────────┐         ┌──────────────────────────┐     │
   │  │ asset_portfolio│         │ portfolio_profile        │     │
   │  │   (PEV 병렬)    │         │ consultant / salary_rb   │     │
   │  │ rebalance      │         │ mini_challenge / nag     │     │
   │  │   (Plan-Reflect)│        │ report                   │     │
   │  └────────────────┘         └──────────────────────────┘     │
   │            │                          │                      │
   │            └────────────┬─────────────┘                      │
   │                         ▼                                    │
   │   ┌──────────────────────────────────────────────────┐      │
   │   │ 공통 레이어                                         │      │
   │   │  llm.py (ainvoke_structured + 3중 폴백)            │      │
   │   │  porti_types.py (PorTI 성향 매핑)                  │      │
   │   │  tools.py (@tool · 계산 유틸)                       │      │
   │   │  gather_products.py (정적 상품 데이터)             │      │
   │   └──────────────────────────────────────────────────┘      │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
        ┌────────────┬───────────┼────────────┬─────────────┐
        ▼            ▼           ▼            ▼             ▼
   ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐  ┌──────────┐
   │ OpenAI  │  │ pgvector│ │  MySQL  │  │  Redis  │  │ 외부 API │
   │ GPT-4o· │  │(asyncpg)│ │(pymysql)│  │ 세션 7일 │  │ yfinance │
   │ 임베딩   │  │ ETF검색  │ │etf_가격 │  │         │  │ Tavily   │
   └─────────┘  └─────────┘ └─────────┘  └─────────┘  └──────────┘
     LLM 호출    ETF 벡터검색  HRP 비중     미니챌린지    주가/뉴스
                  (RAG)      최적화       제안 이력
```

**데이터 흐름 요약**
1. Spring 백엔드/프론트가 FastAPI 라우터로 요청 → 라우터가 타임아웃을 걸고 에이전트 호출
2. 에이전트는 공통 레이어(`llm`/`tools`/`porti_types`)를 거쳐 LLM·외부 데이터 소스와 통신
3. 결정론적 계산(비율·복리·HRP)은 코드로, 맥락 판단·코멘트는 LLM으로 분리 처리
4. 결과를 Pydantic 응답 스키마로 변환해 반환

**외부 의존성**

| 소스 | 클라이언트 | 용도 | 미연결 시 |
|------|-----------|------|-----------|
| OpenAI | `langchain_openai` | LLM 추론·임베딩 | 필수 |
| pgvector | `asyncpg` (`rag/db.py`) | ETF 벡터 검색(RAG) | 빈 결과 폴백 |
| MySQL | `pymysql` | `etf_prices` HRP 가격 이력 | 균등 배분 폴백 |
| Redis | `redis.asyncio` (`session.py`) | 미니챌린지 세션(7일 TTL) | 빈 세션 폴백 |
| yfinance | `yfinance` | 실시간 주가(5분 캐시) | 종목 조회 실패 처리 |
| Tavily | `tavily` | 리포트용 시장 뉴스 | 뉴스 없이 진행 |

---

## 공통 인프라

### `llm.py` — LLM 호출 공통 레이어
- `get_llm(temperature)` : `ChatOpenAI` 인스턴스 생성. 환경변수 `LLM_MODEL`(기본 `gpt-4o`), `LLM_TEMPERATURE`(기본 0.2), `OPENAI_API_KEY` 사용. `max_tokens=4096`.
- `invoke_structured` / `ainvoke_structured(messages, schema, ...)` : **structured output 호출의 핵심 헬퍼**.
  1. 1차: `with_structured_output(schema)` (function calling) 시도
  2. 2차(폴백): 일반 텍스트 응답에서 정규식 `\{.*\}`로 JSON 추출 → Pydantic 검증
  3. 둘 다 실패 시 `None` 반환 → 호출부에서 폴백 처리
- 거의 모든 에이전트가 이 함수를 통해 LLM과 통신합니다.

### `porti_types.py` — PorTI 투자 성향 enum 매핑 (모든 에이전트 공유)
Spring `PortiType` enum을 한국어 설명/배분 전략으로 변환합니다.

| 코드 | 성향 |
|------|------|
| `SWIMMING` | 안전형·단기 |
| `ARCHERY` | 안전형·장기 |
| `JUDO` | 중립형·단기 |
| `RHYTHMIC` | 중립형·장기 |
| `FENCING` | 투자형·단기 |
| `CYCLING` | 투자형·장기 |

- `PORTI_TYPE_DESC` / `PORTI_GUIDANCE` : 성향 설명 / 배분 전략 가이드 텍스트
- `STABLE_/NEUTRAL_/INVEST_PORTI_TYPES` : 성향 집합 (투자 비중·계좌 배정 판단용)
- `porti_label()` / `porti_detail()` : LLM 프롬프트용 라벨·전략 문자열 생성

### `tools.py` — LangChain `@tool` 및 계산 유틸
- `get_stock_prices` / `get_all_prices` : yfinance로 워치리스트 현재가 조회 (5분 캐시, ThreadPool 비동기화)
- `pick_stock` : 절약 금액으로 살 수 있는 최적 종목 선택
- `search_etfs` : **pgvector 하이브리드 검색** — 거래대금 필터 + 관심사 임베딩 유사도로 ETF 검색 (`OpenAIEmbeddings` text-embedding-3-small)
- `calculate_hrp_weights` : **HRP(계층적 위험 균형)**로 ETF 비중 최적화 + 연 기대수익률. 데이터 부족 시 균등 배분 폴백 (MySQL `etf_prices` 사용)
- `compound_interest` : 복리 적립식 미래가치 계산
- `normalize_ratios` / `normalize_amounts` / `normalize_to_thousands` : 비율·금액 합계 보정 유틸

### `gather_products.py`
모으기 계좌 추천용 **정적 상품 데이터** (사용자가 해당 유형 계좌 미보유 시 신규 개설 추천에 사용).

---

## 에이전트별 정리

| 에이전트 파일 | 엔드포인트 | 역할 | 구조 |
|---------------|-----------|------|------|
| `asset_portfolio.py` | `POST /portfolio/asset-portfolio` | 투자 흐름 설계 + ETF 포트폴리오 추천 | **LangGraph PEV** |
| `rebalance.py` | `POST /portfolio/rebalance` | 월급 배분(계좌별) 재설계 | **LangGraph Plan-Reflect** |
| `salary_rebalance.py` | `POST /salary` | 월급 변동분(잉여/결손) 재배분 | 3-step 순차 |
| `portfolio_profile.py` | `POST /portfolio/profile` | 소비·투자 현황 진단 코멘트 | 병렬 2-LLM |
| `consultant.py` | `POST /consultant/analyze`·`/propose` | 목표 분석 → salary/portfolio 재설정 제안 | 단일 LLM ×2 |
| `mini_challenge_agent.py` | `POST /mini_challenge`·`/adjust` | 소비 절약 미니 챌린지 제안/조정 | 단일 LLM + 세션 |
| `nag_agent.py` | `POST /mini_challenge/nag` | 챌린지 달성률 독려 메시지 | 단일 LLM |
| `report.py` | `POST /report` | 월간 재무 리포트 코멘트 | 단일 LLM + Tavily |

> 모든 라우터는 `asyncio.wait_for`로 타임아웃을 걸고, 초과 시 504를 반환합니다.

---

### 1. `asset_portfolio.py` — 자산 포트폴리오 추천 ⭐
**LangGraph PEV(Planner–Executor–Verifier) 패턴**으로 흐름별 병렬 실행.

```
planner ──▶ (Send 분기) ──▶ [ flow_branch: executor ─▶ verifier ] × N (병렬)
```

- **Planner** (`_node_planner`) : 성향·관심사·보유 계좌를 종합해 투자 흐름(단기/중기/장기) 설계, 각 흐름에 모으기 계좌 배정, ratio 합계 100·계좌 중복 제거 검증. LLM 실패 시 `_FALLBACK_FLOWS` 사용.
- **Executor** (`_node_executor`) : `create_agent` + `search_etfs` 툴로 흐름별 ETF 선택. 투자 불가 계좌(적립 전용)는 즉시 빈 포트폴리오. 실패 시 유사도 검색 폴백.
- **Verifier** (`_node_verifier`) : `calculate_hrp_weights`로 비중 최적화 → `compound_interest`로 기대 수익 계산 → 최종 `investment_flow` 조립.

### 2. `rebalance.py` — 월급 배분 재설계 ⭐
**LangGraph Plan–Reflect 패턴** (최대 2회 재계획).

```
plan ──▶ reflect ──▶ (approved? END : plan 재시도, 최대 2회)
```

- **Plan** (`_plan_rebalance`) : 소비 패턴·계좌 잔액 분석 후 계좌별 금액 배분. PorTI 성향별 투자금 허용 범위(`_INVEST_RATIO`) 강제, `asset_id` 검증·천원 단위 정규화.
- **Reflect** (`_reflect`) : 배분 결과 감수 — 분산 실패/투자금 과다/용도 불일치 시 거부(feedback) 후 재계획, 통과 시 reasoning·comment 재작성 및 극단 배분 인라인 교정.

### 3. `salary_rebalance.py` — 월급 변동 재배분
월급이 변동(잉여금/결손금)했을 때 기존 배분을 조정. **3-step 순차**.
1. **ratio 결정** : LLM이 각 항목별 배분 비율(합계 1.0) 산출. 잉여금은 투자 우선, 결손금은 저축/투자 먼저 차감. 검증 실패 시 flow 우선 균등 폴백.
2. **금액 계산** (`_apply_ratios`) : ratio × 변동액 적용, 결손금이 잔액 초과 시 다른 항목으로 재분배.
3. **코멘트 생성** : 실제 delta 금액 기반 `~습니다` 체 설명.

### 4. `portfolio_profile.py` — 소비·투자 진단
소비/투자 현황을 **병렬 2개 LLM**(`asyncio.gather`)으로 진단.
- 지출·자산 비율은 LLM 없이 순수 계산(`_diagnose_expense/_invest`)
- 소비 코멘트(120자) + 투자 코멘트(70자)를 reasoning-then-comment 방식으로 생성

### 5. `consultant.py` — 목표 기반 재설정 컨설턴트
- `analyze_goal` : 사용자 목표를 분석해 `salary`(월급 배분) vs `portfolio`(투자) 중 적합한 액션 추천
- `propose_reset` : 선택된 액션에 따라 구체적 배분/포트폴리오 비율 제안 (ratio 합계 100)

### 6. `mini_challenge_agent.py` — 소비 절약 미니 챌린지
소비 패턴 분석 → 절약 챌린지 제안. **세션**(`get/save_session`)에 제안 이력 저장.
- `propose_mini_challenge` : sub_type 선정(COFFEE/DELIVERY/… 7종) → 챌린지 설계 → 현재 주가에서 ticker 선택
- `adjust_challenge` : 사용자 피드백("더 쉽게/어렵게/주제 변경")에 맞게 조정, 이전 제안 반복 금지
- (reward 처리는 라우터에서 `pick_stock`으로 절약금↔주식 환산)

### 7. `nag_agent.py` — 챌린지 독려
챌린지 달성률(50/80/90%)에 맞춘 친근한 잔소리성 메시지 1~2문장 생성.

### 8. `report.py` — 월간 재무 리포트
거래내역·자산 스냅샷 집계(순수 계산) + **Tavily**로 시장 뉴스 검색 → 추세/챌린지/시장/카테고리 hover/가이드라인 코멘트 생성. `TAVILY_API_KEY` 없으면 뉴스 없이 진행.

---

## 설계 패턴 요약

- **공통 LLM 레이어 + 폴백** : 모든 LLM 호출은 `ainvoke_structured`로 통일, structured output 실패 시 정규식 파싱 → 기본값까지 3중 폴백.
- **계산은 코드, 판단은 LLM** : 비율·금액·복리·HRP는 결정론적 코드로, 맥락 판단·코멘트는 LLM으로 분리.
- **LangGraph 활용** : 복잡한 흐름(asset_portfolio=PEV 병렬, rebalance=Plan-Reflect 루프)에만 그래프 사용, 단순 흐름은 직접 호출.
- **타임아웃 가드** : 라우터마다 `asyncio.wait_for`로 504 처리.
