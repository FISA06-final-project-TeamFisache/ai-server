# Pori AI Server (FastAPI)

개인 재무 관리 AI **Pori**의 FastAPI 기반 AI 서버입니다.  
Spring 백엔드로부터 AI 추론 요청을 위임받아 처리하며, LangGraph·LangChain·OpenAI GPT-4o를 사용해 투자 포트폴리오 설계, 월급 배분, 소비 챌린지 등 다양한 재무 에이전트를 구동합니다.

> 실행 순서: **인프라(Docker) → 백엔드(Spring) → AI 서버(FastAPI) → 목서버**

---

## 목차

- [전체 아키텍처](#전체-아키텍처)
- [핵심 서비스](#핵심-서비스)
- [외부 의존성](#외부-의존성)
- [공통 인프라](#공통-인프라)
- [에이전트 목록](#에이전트-목록)
- [실행 방법](#실행-방법)
- [환경 변수](#환경-변수)

---

## 전체 아키텍처

```
                ┌──────────────────────────────────────────────┐
 Spring 백엔드 ──▶ │           FastAPI (app/main.py)              │
 /프론트         │    라우터: asyncio.wait_for 타임아웃 가드       │
                └──────────────────────────────────────────────┘
                      │            │            │            │
      ┌───────────────┘   ┌────────┘     ┌──────┘      ┌─────┘
      ▼                   ▼              ▼             ▼
 ┌────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐
 │ /portfolio │  │ /salary      │  │ /report  │  │/mini_challenge│
 └─────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘
       │                │               │               │
       ▼                ▼               ▼               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │               app/services/agent (에이전트 계층)              │
 │                                                              │
 │  LangGraph 그래프              단일/병렬 LLM 에이전트          │
 │  ┌────────────────┐           ┌──────────────────────────┐   │
 │  │ asset_portfolio│           │ portfolio_profile        │   │
 │  │   (PEV 병렬)    │           │ consultant / salary_rb   │   │
 │  │ rebalance      │           │ mini_challenge / nag     │   │
 │  │  (Plan-Reflect)│           │ report                   │   │
 │  └────────────────┘           └──────────────────────────┘   │
 │              │                          │                    │
 │              └──────────────────────────┘                    │
 │                             ▼                                │
 │   ┌──────────────────────────────────────────────────┐      │
 │   │ 공통 레이어                                         │      │
 │   │  llm.py (ainvoke_structured + 2중 폴백)            │      │
 │   │  porti_types.py (PorTI 성향 매핑)                  │      │
 │   │  tools.py (@tool · 계산 유틸)                       │      │
 │   │  gather_products.py (정적 상품 데이터)              │      │
 │   └──────────────────────────────────────────────────┘      │
 └─────────────────────────────┬───────────────────────────────┘
                               │
      ┌────────────┬───────────┼────────────┬─────────────┐
      ▼            ▼           ▼            ▼             ▼
 ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐  ┌──────────┐
 │ OpenAI  │  │pgvector │ │  MySQL  │  │  Redis  │  │ 외부 API │
 │ GPT-4o  │  │(asyncpg)│ │(pymysql)│  │세션 7일  │  │ yfinance │
 │ 임베딩   │  │ ETF검색  │ │etf_가격 │  │         │  │ Tavily   │
 └─────────┘  └─────────┘ └─────────┘  └─────────┘  └──────────┘
```

**데이터 흐름 요약**

1. Spring 백엔드/프론트가 FastAPI 라우터로 요청 → 라우터가 타임아웃을 걸고 에이전트 호출
2. 에이전트는 공통 레이어(`llm` / `tools` / `porti_types`)를 거쳐 LLM·외부 데이터 소스와 통신
3. 결정론적 계산(비율·복리·HRP)은 코드로, 맥락 판단·코멘트는 LLM으로 분리 처리
4. 결과를 Pydantic 응답 스키마로 변환해 반환

---

## 핵심 서비스

### 1. 투자 포트폴리오 설계 에이전트 — `asset_portfolio.py`

**엔드포인트:** `POST /portfolio/asset-portfolio`

사용자의 PorTI 투자 성향, 인생 목표, 보유 계좌를 분석해 단기·중기·장기 **투자 흐름**을 설계하고, 각 흐름에 맞는 ETF 포트폴리오를 추천합니다.

**LangGraph PEV(Planner–Executor–Verifier) 패턴**으로 각 투자 흐름을 병렬 처리합니다.

```
Planner ──▶ (Send 분기) ──▶ [ Executor ─▶ Verifier ] × N (흐름별 병렬)
```

| 노드 | 역할 |
|------|------|
| Planner | 성향·목표·보유 계좌를 종합해 투자 흐름(단기/중기/장기)과 모으기 계좌 설계. ratio 합계 100 보정, 노후 흐름 구조적 가드 |
| Executor | `search_etfs` 툴(pgvector 하이브리드 검색)로 흐름별 ETF 선택. 투자 불가 계좌는 즉시 빈 포트폴리오 반환 |
| Verifier | `calculate_hrp_weights`(HRP 최적화) + `compound_interest`(복리 기대 수익) 계산 후 최종 흐름 조립 |

---

### 2. 월급 관리 에이전트 — `rebalance.py`

**엔드포인트:** `POST /portfolio/rebalance`

월급에서 고정지출을 제외한 **가처분소득**을 사용자의 소비 패턴과 보유 계좌에 맞게 생활비·비상금·예비비·투자로 재배분합니다.

**LangGraph Plan–Reflect 패턴** (최대 2회 재계획)으로 배분 결과의 품질을 자동 검증합니다.

```
Plan ──▶ Reflect ──▶ (승인? END : Plan 재시도, 최대 2회)
```

| 노드 | 역할 |
|------|------|
| Plan | 소비 패턴·계좌 잔액 분석 후 계좌별 금액 배분. PorTI 성향별 투자금 허용 범위 강제, asset_id 검증·천원 단위 정규화 |
| Reflect | 배분 결과 감수 — 분산 실패/투자금 과다/용도 불일치 시 feedback 반환 후 재계획, 통과 시 승인 |

---

### 3. 소비 내역 기반 미니 챌린지 제안 에이전트 — `mini_challenge_agent.py`

**엔드포인트:** `POST /mini_challenge`, `POST /mini_challenge/adjust`

이번 달 소비 패턴을 분석해 가장 절약 효과가 큰 카테고리의 **미니 챌린지**를 제안하고, 절약한 금액을 주식 투자 리워드로 연결합니다.

Redis 세션(7일 TTL)에 제안 이력을 저장해 동일 챌린지 반복을 방지하며, 사용자 피드백("더 쉽게/어렵게/주제 변경")에 따라 챌린지를 실시간으로 조정합니다.

| 함수 | 역할 |
|------|------|
| `propose_mini_challenge` | 소비 상위 카테고리 선정 → 챌린지 설계(횟수/금액 제한) → 관심 테마 주식 ticker 선택 |
| `adjust_challenge` | 피드백 기반 챌린지 난이도·주제 조정, 이전 제안 반복 금지 |
| `/reward` (라우터) | 절약 금액 기반 `pick_stock`으로 실시간 주식 환산 + Kafka 이벤트 발행 |

---

## 외부 의존성

| 소스 | 클라이언트 | 용도 | 미연결 시 |
|------|-----------|------|-----------|
| OpenAI | `langchain_openai` | LLM 추론·임베딩 | 필수 |
| pgvector | `asyncpg` | ETF 벡터 검색(RAG) | 빈 결과 폴백 |
| MySQL | `pymysql` | `etf_prices` HRP 가격 이력 | 균등 배분 폴백 |
| Redis | `redis.asyncio` | 미니챌린지 세션(7일 TTL) | 빈 세션 폴백 |
| yfinance | `yfinance` | 실시간 주가(5분 캐시) | 종목 조회 실패 처리 |
| Tavily | `tavily` | 리포트용 시장 뉴스 | 뉴스 없이 진행 |
| Kafka | `aiokafka` | 사용자 행동 이벤트 로그 | 로그 전송 비활성화 |

---

## 공통 인프라

### `llm.py` — LLM 호출 공통 레이어

모든 에이전트가 공유하는 structured output 헬퍼. 2중 폴백으로 안정성 보장.

1. 1차: `with_structured_output(schema)` (function calling) 시도
2. 2차(폴백): 일반 텍스트 응답 → `\{.*\}` 정규식으로 JSON 추출 → Pydantic 검증
3. 둘 다 실패 시 `None` 반환 → 호출부에서 폴백 처리

### `porti_types.py` — PorTI 투자 성향 매핑

| 코드 | 성향 |
|------|------|
| `SWIMMING` | 안전형·단기 |
| `ARCHERY` | 안전형·장기 |
| `JUDO` | 중립형·단기 |
| `RHYTHMIC` | 중립형·장기 |
| `FENCING` | 투자형·단기 |
| `CYCLING` | 투자형·장기 |

### `tools.py` — LangChain `@tool` 및 계산 유틸

| 함수 | 역할 |
|------|------|
| `search_etfs` | pgvector 하이브리드 검색 — 거래대금 필터 + 임베딩 유사도 |
| `calculate_hrp_weights` | HRP(계층적 위험 균형)로 ETF 비중 최적화 + 기대 수익률 |
| `compound_interest` | 복리 적립식 미래가치 계산 |
| `get_stock_prices` / `get_all_prices` | yfinance 현재가 조회 (5분 캐시) |
| `pick_stock` | 절약 금액으로 살 수 있는 최적 종목 선택 |

---

## 에이전트 목록

| 파일 | 엔드포인트 | 역할 | 구조 |
|------|-----------|------|------|
| `asset_portfolio.py` | `POST /portfolio/asset-portfolio` | 투자 흐름 설계 + ETF 포트폴리오 추천 | LangGraph PEV |
| `rebalance.py` | `POST /portfolio/rebalance` | 월급 배분(계좌별) 재설계 | LangGraph Plan-Reflect |
| `salary_rebalance.py` | `POST /salary` | 월급 변동분(잉여/결손) 재배분 | 3-step 순차 |
| `portfolio_profile.py` | `POST /portfolio/profile` | 소비·투자 현황 진단 코멘트 | 병렬 2-LLM |
| `mini_challenge_agent.py` | `POST /mini_challenge`, `/adjust` | 소비 절약 미니 챌린지 제안/조정 | 단일 LLM + 세션 |
| `nag_agent.py` | `POST /mini_challenge/nag` | 챌린지 달성률 독려 메시지 | 단일 LLM |
| `report.py` | `POST /report` | 월간 재무 리포트 코멘트 | 단일 LLM + Tavily |

> 모든 라우터는 `asyncio.wait_for`로 타임아웃(30~120초)을 걸고, 초과 시 504를 반환합니다.

---

## 실행 방법

```powershell
# 가상환경 활성화 (터미널 새로 열 때마다)
.\.venv\Scripts\activate

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

### 포트

| 서비스 | 포트 |
|--------|------|
| AI Server (FastAPI) | 8000 |
| Prometheus metrics | 8000/metrics |

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI API 키 (필수) |
| `LLM_MODEL` | `gpt-4o` | 사용할 LLM 모델 |
| `LLM_TEMPERATURE` | `0.2` | LLM 온도 |
| `TAVILY_API_KEY` | — | Tavily 뉴스 검색 (선택) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` | 허용 프론트 origin (콤마 구분) |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka 브로커 주소 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

자세한 에이전트 설계는 [app/services/agent/README.md](app/services/agent/README.md)를 참고하세요.
