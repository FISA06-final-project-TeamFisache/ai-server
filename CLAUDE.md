# AI Server — CLAUDE.md

## 1. Commands

```bash
# 서버 실행
uvicorn app.main:app --reload --port 8000

# venv 활성화 (macOS)
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 테스트 전체 실행
pytest app/tests/ -v

# 테스트 + 커버리지
pytest app/tests/ -v --cov=app --cov-report=term-missing

# 특정 테스트만
pytest app/tests/test_agent.py -v

# 상품 데이터 DB 적재 (KRX ETF + FSS 예적금)
python scripts/load_products.py

# 타입 체크
.venv/bin/python -m mypy app/ --ignore-missing-imports
```

---

## 2. Testing

| 항목 | 내용 |
|---|---|
| 프레임워크 | pytest + pytest-asyncio |
| 테스트 위치 | `app/tests/` |
| 설정 파일 | `app/tests/conftest.py` |
| 커버리지 기준 | agent 서비스 함수 주요 경로 커버 |

```bash
# 비동기 테스트 실행 시
pytest app/tests/ -v --asyncio-mode=auto
```

**테스트 작성 규칙:**
- agent 노드는 개별 함수 단위로 단위 테스트
- LLM 호출이 포함된 통합 테스트는 `OPENROUTER_API_KEY` 환경변수 필요
- yfinance / DB 의존 테스트는 mock 또는 skipif 처리

---

## 3. Project Structure

```
ai-server/
├── app/
│   ├── core/
│   │   ├── config.py          # 상수 (타임아웃, 모델명 등)
│   │   └── exceptions.py      # 공통 예외 핸들러
│   ├── db/
│   │   └── connection.py      # asyncpg 커넥션 풀
│   ├── routers/               # FastAPI 라우터 (엔드포인트 정의)
│   │   ├── portfolio.py       # /portfolio/* (rebalance, asset-portfolio)
│   │   ├── salary.py          # /salary/*
│   │   ├── consultant.py      # /consultant/*
│   │   ├── report.py          # /report/*
│   │   └── mini_challenge.py  # /mini-challenge/*
│   ├── schemas/               # Pydantic 입출력 모델
│   ├── services/
│   │   ├── agent/             # ★ LangGraph Agent 로직
│   │   │   ├── asset_portfolio.py   # 투자 포트폴리오 추천 (핵심)
│   │   │   ├── rebalance.py         # 월급 쪼개기
│   │   │   ├── salary_rebalance.py  # 월급 리밸런싱
│   │   │   ├── consultant.py        # 목표 분석·제안
│   │   │   ├── tools.py             # LangChain 도구 모음
│   │   │   └── llm.py               # LLM 클라이언트 (OpenRouter)
│   │   ├── rag/               # RAG (pgvector 임베딩 검색)
│   │   │   ├── db.py
│   │   │   └── retriever.py
│   │   └── stock.py           # yfinance 현재가 조회
│   └── tests/
│       ├── conftest.py
│       ├── test_agent.py
│       └── test_health.py
├── scripts/
│   └── load_products.py       # KRX ETF + FSS 상품 DB 적재 스크립트
├── models/                    # ML 모델 파일 (.pkl)
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

---

## 4. Code Style

**비동기:** 모든 agent 노드 함수는 `async def` 사용

```python
# ✅ 올바른 노드 함수 시그니처
async def _investment_agent(state: AssetPortfolioState) -> AssetPortfolioState:
    ...
    return {**state, "flow_products": flow_products}

# ✅ LLM 구조화 출력 호출
result = await ainvoke_structured(messages, OutputSchema)
if result is None:
    # 항상 폴백 처리
    ...
```

**LangGraph 상태 업데이트:** 반드시 `{**state, "key": value}` 패턴 사용 (직접 변이 금지)

```python
# ✅
return {**state, "flow_products": flow_products}

# ❌
state["flow_products"] = flow_products
return state
```

**블로킹 IO:** yfinance 등 동기 라이브러리는 반드시 `asyncio.to_thread()` 로 실행

```python
# ✅
returns = await asyncio.to_thread(_download)

# ❌
returns = yf.download(...)  # async 함수 안에서 직접 호출 금지
```

**폴백 필수:** LLM 응답이 None이거나 외부 API 실패 시 항상 안전한 폴백 반환

```python
result = await ainvoke_structured(messages, Schema)
if not result:
    return {**state, "field": fallback_value}
```

**네이밍:**
- 내부 함수/클래스: `_언더스코어` prefix (외부 노출 아님)
- LangGraph 노드: `_동사_명사` (예: `_define_flows`, `_select_accounts`)
- Pydantic 내부 스키마: `_파스칼케이스` (예: `_FlowsAIOutput`)
- 공개 entry point: `동사_명사` (예: `recommend_asset_portfolio`)

---

## 5. Agent 설계 스펙 (feat/agent-tool-calling 브랜치)

### 핵심 원칙
> LLM은 **판단**하고, 수학/도구는 **계산**한다.
> LLM이 숫자를 직접 생성하면 신뢰도 낮음 → 반드시 도구를 통해 근거 있는 결과 도출

### `asset_portfolio.py` 그래프 구조

```
preprocess (RAG 벡터 검색)
    ↓
define_flows (LLM: 단기/중기/장기 흐름 비중 및 제목 결정)
    ↓
select_accounts (Rule: 계좌 타입 우선순위 매핑)
    ↓
[search_trends?] (Tavily: 관심사 없을 때 조건부 실행)
    ↓
investment_agent ← ★ 핵심 노드
  Step 1. LLM: 흐름별 후보 ETF 선택 (어떤 상품인지 판단)
  Step 2. Tool: yfinance 2년치 수익률 조회
  Step 3. Tool: HRP(계층적 리스크 패리티)로 비중 계산
          → ratio 합계 100% 수학적 보장
    ↓
calculate (복리 적립식 FV 계산)
    ↓
END
```

**제거된 노드:** `select_products`, `reflect`, `refine`
- 제거 이유: LLM이 비중을 직접 생성하던 노드들. HRP 도구로 대체하여 불필요해짐.

### `tools.py` 도구 목록

| 도구 | 입력 | 출력 | 용도 |
|---|---|---|---|
| `compound_interest` | 원금, 월이율, 기간 | 최종금액 | 복리 계산 |
| `monthly_savings_needed` | 목표액, 현재액, 기간 | 월 저축액 | 목표 달성 저축액 |
| `normalize_ratios` | 주식/채권/현금 | 정규화 비중 | 비중 합계 100% |
| `rebalance_diff` | 현재/목표 비율, 총자산 | 매수/매도 금액 | 리밸런싱 차이 |
| `calculate_hrp_weights` | 수익률 시계열 JSON | 비중 JSON | **HRP 최적화** |

### HRP 폴백 전략

```
yfinance 데이터 조회 성공 + 행 ≥ 20 + 상품 ≥ 2개
    → HRP 계산
데이터 부족 (< 20일) 또는 상품 1개
    → 균등 가중치
yfinance 실패
    → 균등 가중치
HRP 예외
    → 균등 가중치
LLM 후보 선택 실패
    → 상위 ETF 3개 균등 배분
```

### 환경변수 (`.env` 필수)

```bash
OPENROUTER_API_KEY=...      # LLM 호출 (필수)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.2
DB_URL=postgresql://...     # pgvector DB (필수)
TAVILY_API_KEY=...          # 트렌드 검색 (선택, 없으면 스킵)
KRX_API_KEY=...             # 상품 적재 스크립트용 (선택)
FSS_API_KEY=...             # 상품 적재 스크립트용 (선택)
OPENAI_API_KEY=...          # 임베딩용 (필수)
EMBEDDING_MODEL=text-embedding-3-small
```

---

## 6. Git 워크플로우

**브랜치 규칙:**

```
main                        # 프로덕션
feat/<기능명>               # 기능 개발
refactoring/<내용>          # 리팩토링
chore/<내용>                # 설정/빌드
```

**현재 작업 브랜치:** `feat/agent-tool-calling`

**커밋 메시지 포맷:**

```
feat: 한 줄 요약 (50자 이내)

- 변경 이유나 비고 (선택)
```

**PR 요건:**
- [PR템플릿](.github/pull_request_template.md) 양식 참고
- main 직접 push 금지, 반드시 PR
- PR 제목 = 커밋 메시지 형식 동일
- Co-Authored-By 라인 커밋 메시지에 포함 금지
---

## 7. Boundaries (절대 건드리면 안 되는 것)

```
❌ .env 파일 커밋 금지 (API 키 포함)
❌ models/*.pkl 파일 직접 수정 금지 (ML 모델 바이너리)
❌ scripts/load_products.py 실행 시 프로덕션 DB_URL 사용 금지
❌ LangGraph 상태(state) 직접 변이 금지 → {**state, ...} 패턴만 허용
❌ ainvoke_structured() 폴백 처리 생략 금지 → None 반환 항상 처리
❌ async 함수 내 블로킹 IO 직접 호출 금지 → asyncio.to_thread() 사용
❌ LLM에게 ratio 숫자 직접 생성 요청 금지 → 도구(HRP/수학함수) 사용
```
