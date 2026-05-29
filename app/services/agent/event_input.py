import json
import re
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.event import EventInputRequest, EventInputResponse
from app.services.agent.llm import get_llm


class EventInputState(TypedDict):
    user_input: str
    title: str
    target_amount: int
    deadline_months: int
    strategy_raw: str


def _parse_event(state: EventInputState) -> EventInputState:
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=(
            "사용자의 자연어 목표 입력에서 이벤트 정보를 추출하세요.\n\n"
            "규칙:\n"
            "- title: 목표를 한 문장으로 정리한 이름 (예: '제주도 여행 자금 마련')\n"
            "- target_amount: 목표 금액(원, 정수). 명시 없으면 500000\n"
            "- deadline_months: 목표까지 남은 개월 수(정수). 명시 없으면 6\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"title":"이벤트명","target_amount":1500000,"deadline_months":6}'
        )),
        HumanMessage(content=state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_result(state: EventInputState) -> EventInputState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    return {
        **state,
        "title": data.get("title", "목표 자금 마련"),
        "target_amount": int(data.get("target_amount", 500000)),
        "deadline_months": int(data.get("deadline_months", 6)),
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(EventInputState)
    graph.add_node("parse", _parse_event)
    graph.add_node("extract", _parse_result)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", END)
    return graph.compile()


_graph = _build_graph()


async def analyze_event_input(request: EventInputRequest) -> EventInputResponse:
    initial_state: EventInputState = {
        "user_input": request.user_input,
        "title": "",
        "target_amount": 0,
        "deadline_months": 6,
        "strategy_raw": "",
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
