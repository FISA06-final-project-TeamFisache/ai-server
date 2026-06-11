# Instructor 도입 가이드

## 왜 Instructor인가

### 현재 방식의 문제

현재 `llm.py`는 LangChain의 `with_structured_output`으로 구조화된 응답을 받고,
실패하면 regex로 JSON을 파싱하는 2단계 폴백을 **직접** 구현하고 있다.

```
1차: llm.with_structured_output(schema).ainvoke(messages)
2차: llm.ainvoke(messages) → regex로 JSON 추출 → Pydantic 검증
```

이 방식의 한계:

- **Pydantic validator를 어겨도 알 수 없다.**  
  LLM이 `target = 500,000`을 뱉어도 타입 검사만 통과하면 그냥 반환된다.  
  비즈니스 규칙("target < 소비 총액")을 강제할 수단이 프롬프트뿐이다.

- **폴백 로직을 직접 관리해야 한다.**  
  2차 파싱이 실패하면 `None` 반환 후 caller에서 default 응답으로 처리한다.

### Instructor가 해결하는 것

Instructor는 OpenAI 클라이언트를 래핑해 **Pydantic validator 실패 시
실패 메시지를 LLM에 그대로 피드백하고 자동 재시도**한다.

```
LLM 응답 → Pydantic validator 실행
              ├─ 통과 → 반환
              └─ 실패 → 오류 메시지를 LLM에 전달 → 재시도 (최대 N회)
```

예시: `target = 500,000`, 소비 총액 = 450,000원일 때

```
[Instructor → LLM]
"ValidationError: target(500,000원)이 소비 총액(450,000원) 이상입니다.
 반드시 소비 총액 미만으로 설정하세요."

[LLM 재시도]
target = 360,000  ← 수정된 응답
```

프롬프트로 설명하는 것보다 훨씬 확실하게 제약을 강제할 수 있다.

---

## 변경이 필요한 파일

전부 `ai-server/` 안에 있다.

### 1. `requirements.txt`

```diff
+ instructor>=1.0.0
```

`openai>=1.0.0`은 이미 있으므로 추가 의존성은 없다.

---

### 2. `app/services/agent/llm.py` — 핵심 변경

#### Before

```python
from langchain_openai import ChatOpenAI

def get_llm(temperature=None) -> ChatOpenAI:
    return ChatOpenAI(model=..., temperature=..., openai_api_key=...)

async def ainvoke_structured(messages, schema, temperature=None, max_tokens=None):
    llm = get_llm(temperature)
    # 1차: structured output
    result = await llm.with_structured_output(schema).ainvoke(messages)
    # 2차: regex fallback
    raw = (await llm.ainvoke(messages)).content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return schema.model_validate(json.loads(match.group()))
```

#### After

```python
import instructor
from openai import AsyncOpenAI, OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    role_map = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
    return [{"role": role_map[type(m)], "content": m.content} for m in messages]

async def ainvoke_structured(messages, schema, temperature=None, max_tokens=None):
    client = instructor.from_openai(
        AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    )
    return await client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        temperature=temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        max_tokens=max_tokens or 4096,
        response_model=schema,
        messages=_to_openai_messages(messages),
        max_retries=3,
    )
```

- 호출부(`mini_challenge_agent.py` 등) **시그니처 변경 없음**
- LangChain의 `SystemMessage`, `HumanMessage`는 그대로 유지

---

### 3. `app/services/agent/mini_challenge_agent.py` — validator 추가

Instructor의 retry가 의미를 가지려면 Pydantic 모델에 비즈니스 규칙을 validator로 명시해야 한다.

```python
from pydantic import model_validator

class _MiniChallengeAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: ChallengeType
    target: int
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: _SubType

    # 소비 총액은 호출 시점에 context로 주입
    spending_total: int = 0

    @model_validator(mode='after')
    def validate_amount_target(self):
        if (
            self.challenge_type == ChallengeType.AMOUNT
            and self.spending_total > 0
            and self.target >= self.spending_total
        ):
            raise ValueError(
                f"target({self.target:,}원)이 해당 카테고리 소비 총액({self.spending_total:,}원) "
                f"이상입니다. 소비를 줄이는 챌린지이므로 target은 반드시 소비 총액 미만이어야 합니다."
            )
        return self
```

`spending_total` 주입 방법:

```python
# propose_mini_challenge() 안에서
total = sum(c.amount for c in req.category_expense)
schema = _MiniChallengeAIOutput.model_copy(
    update={"spending_total": total}
)
# 또는 Pydantic partial / ClassVar 활용
```

> `_AdjustAIOutput`도 동일하게 validator 추가 필요.

---

## 변경하지 않아도 되는 파일

| 파일 | 이유 |
|------|------|
| `asset_portfolio.py` | `ainvoke_structured` 시그니처 동일 |
| `consultant.py` | 동일 |
| `portfolio_profile.py` | 동일 |
| `rebalance.py` | 동일 |
| `report.py` | 동일 |
| `salary_rebalance.py` | 동일 |
| `nag_agent.py` | 동일 |
| `tools.py` | LangChain `@tool` 사용, LLM 호출 없음 |
| 모든 `routers/` | 변경 없음 |

---

## 주의사항

- **retry 발생 시 토큰 추가 소모**: validator 실패 1회당 요청이 1번 더 나간다.  
  `max_retries=3` 기준 최대 4회 호출 가능. 비용 민감한 엔드포인트는 `max_retries=2`도 고려.

- **`spending_total` 주입 방식**: Pydantic v2에서 외부 컨텍스트를 validator에 넘기는 방법이  
  여러 가지(`model_validator`, `ClassVar`, `PrivateAttr`)이므로 구현 시 확인 필요.

- **`invoke_structured`(동기)도 동일하게 교체** 필요:  
  `instructor.from_openai(OpenAI(...))` 사용.
