import json
import re
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.kafka import KafkaAnomalyAlert, KafkaTransactionMessage
from app.services.agent.llm import get_llm


class AnomalyState(TypedDict):
    asset_number: str
    amount: int
    category: str
    sender_name: str
    transaction_at: datetime
    is_anomaly: bool
    alert_content: str


def _analyze(state: AnomalyState) -> AnomalyState:
    llm = get_llm(temperature=0.0)
    hour = state["transaction_at"].hour
    messages = [
        SystemMessage(content=(
            "당신은 이상 거래 탐지 전문가입니다. 아래 거래 정보를 분석해 이상 여부를 판단하세요.\n\n"
            "이상 거래 판단 기준:\n"
            "- 카테고리 대비 과도한 금액 (예: 편의점에서 200,000원 이상)\n"
            "- 심야 시간대(자정~오전 6시) 고액 거래\n"
            "- 가맹점명이 불분명하거나 비정상적인 경우\n\n"
            "반드시 JSON 형식으로만 응답하세요:\n"
            '{"is_anomaly": true or false, "reason": "이상 사유 (정상이면 빈 문자열)"}'
        )),
        HumanMessage(content=(
            f"거래 정보:\n"
            f"- 금액: {state['amount']:,}원\n"
            f"- 카테고리: {state['category']}\n"
            f"- 가맹점: {state['sender_name']}\n"
            f"- 거래 시각: {state['transaction_at'].strftime('%Y-%m-%d %H:%M')} ({hour}시)"
        )),
    ]
    result = llm.invoke(messages)
    try:
        match = re.search(r"\{.*?\}", result.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    return {
        **state,
        "is_anomaly": bool(data.get("is_anomaly", False)),
        "alert_content": data.get("reason", ""),
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(AnomalyState)
    graph.add_node("analyze", _analyze)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", END)
    return graph.compile()


_graph = _build_graph()


async def detect_anomaly_agent(transaction: KafkaTransactionMessage) -> KafkaAnomalyAlert | None:
    initial_state: AnomalyState = {
        "asset_number": transaction.asset_number,
        "amount": transaction.amount,
        "category": transaction.category,
        "sender_name": transaction.sender_name,
        "transaction_at": transaction.transactionAt,
        "is_anomaly": False,
        "alert_content": "",
    }
    final_state: AnomalyState = await _graph.ainvoke(initial_state)

    if not final_state["is_anomaly"]:
        return None

    return KafkaAnomalyAlert(
        asset_number=final_state["asset_number"],
        content=final_state["alert_content"],
        created_at=datetime.now(),
    )
