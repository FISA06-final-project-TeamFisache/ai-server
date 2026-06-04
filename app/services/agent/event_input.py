from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.schemas.event import EventInputRequest, EventInputResponse
from app.services.agent.llm import invoke_structured


class _EventInputSchema(BaseModel):
    title: str = Field(default="목표 자금 마련")
    target_amount: int = Field(default=500_000)
    deadline_months: int = Field(default=6)


class EventInputState(TypedDict):
    user_input: str
    title: str
    target_amount: int
    deadline_months: int


_SYSTEM = (
    "사용자의 자연어 목표 입력에서 이벤트 정보를 추출하세요.\n\n"
    "추출 순서:\n"
    "1. 입력에서 금액 단서를 찾아 target_amount를 결정하세요. (없으면 500000)\n"
    "2. 기간 단서를 찾아 deadline_months를 결정하세요. (없으면 6)\n"
    "3. 목표를 한 문장으로 요약해 title을 작성하세요.\n\n"
    "예시 1:\n"
    "입력: '6개월 안에 제주도 여행 150만원 모으고 싶어'\n"
    "→ title: '제주도 여행 자금 마련', target_amount: 1500000, deadline_months: 6\n\n"
    "예시 2:\n"
    "입력: '내년 여름까지 노트북 살 돈 모으기'\n"
    "→ title: '노트북 구매 자금', target_amount: 500000, deadline_months: 12\n\n"
    "예시 3:\n"
    "입력: '3년 후 결혼 준비금 3000만원'\n"
    "→ title: '결혼 준비 자금', target_amount: 30000000, deadline_months: 36"
)


def _parse_event(state: EventInputState) -> EventInputState:
    result = invoke_structured(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=state["user_input"])],
        _EventInputSchema,
        temperature=0.0,
    )
    if result is None:
        return {**state, "title": "목표 자금 마련", "target_amount": 500_000, "deadline_months": 6}
    return {
        **state,
        "title": result.title,
        "target_amount": result.target_amount,
        "deadline_months": result.deadline_months,
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(EventInputState)
    graph.add_node("parse", _parse_event)
    graph.set_entry_point("parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


async def analyze_event_input(request: EventInputRequest) -> EventInputResponse:
    initial_state: EventInputState = {
        "user_input": request.user_input,
        "title": "",
        "target_amount": 0,
        "deadline_months": 6,
    }

    final_state: EventInputState = await _graph.ainvoke(initial_state)

    months = max(1, final_state["deadline_months"])
    deadline = datetime.now(timezone.utc) + timedelta(days=months * 30)

    return EventInputResponse(
        created_at=datetime.now(timezone.utc),
        title=final_state["title"],
        target_amount=final_state["target_amount"],
        deadline=deadline,
    )
