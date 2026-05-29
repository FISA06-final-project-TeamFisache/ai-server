import asyncio
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.portfolio import ProfileRequest, ProfileResponse
from app.services.agent.llm import get_llm


class ProfileState(TypedDict):
    porti_type: str
    porti_comment: str
    expense_summary: str   # "식비 450,000원, 교통 150,000원 ..."
    asset_summary: str     # "위험자산 35% / 안전자산 65%, 총 자산 18,500,000원"
    savings_summary: str   # "입출금 6,800,000원, 적금 2,400,000원 ..."
    expense_comment: str
    invest_comment: str
    savings_comment: str


def _generate_expense_comment(state: ProfileState) -> ProfileState:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 친절한 금융 어드바이저입니다. "
            "사용자의 소비 내역과 투자 성향을 바탕으로 지출 패턴을 2~3문장으로 분석하고 "
            "개선 포인트를 제안하세요. 친근하고 구체적인 말투로 작성하세요."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {state['porti_type']}\n"
            f"성향 설명: {state['porti_comment']}\n"
            f"월 평균 카테고리별 지출: {state['expense_summary']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "expense_comment": result.content.strip()}


def _generate_invest_comment(state: ProfileState) -> ProfileState:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 투자 성향 분석 전문가입니다. "
            "PorTI 설문 결과와 실제 계좌 구성을 비교해 "
            "투자 성향을 2~3문장으로 분석하세요. 친근하고 따뜻한 말투로 작성하세요."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {state['porti_type']}\n"
            f"실제 자산 구성: {state['asset_summary']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "invest_comment": result.content.strip()}


def _generate_savings_comment(state: ProfileState) -> ProfileState:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 저축 컨설턴트입니다. "
            "사용자의 저축 현황을 바탕으로 저축 패턴을 2~3문장으로 분석하고 "
            "유동성 관리 조언을 제공하세요. 친근하고 따뜻한 말투로 작성하세요."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {state['porti_type']}\n"
            f"저축·예금 현황: {state['savings_summary']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "savings_comment": result.content.strip()}


def _build_graph() -> StateGraph:
    graph = StateGraph(ProfileState)

    graph.add_node("expense", _generate_expense_comment)
    graph.add_node("invest", _generate_invest_comment)
    graph.add_node("savings", _generate_savings_comment)

    graph.set_entry_point("expense")
    graph.add_edge("expense", "invest")
    graph.add_edge("invest", "savings")
    graph.add_edge("savings", END)

    return graph.compile()


_graph = _build_graph()


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    # 지출 요약 구성
    if request.category_expense:
        expense_summary = ", ".join(
            f"{e.name} {e.expense:,}원" for e in request.category_expense
        )
    else:
        expense_summary = "거래 내역 없음"

    # 자산 구성 분석
    RISK_TYPES = {"STOCK", "IRP", "ISA"}
    total_balance = sum(a.balance for a in request.assets)
    risk_balance = sum(a.balance for a in request.assets if a.asset_type in RISK_TYPES)
    safe_balance = total_balance - risk_balance
    risk_ratio = round(risk_balance * 100 / total_balance) if total_balance > 0 else 0
    asset_summary = (
        f"위험자산(주식·IRP·ISA) {risk_ratio}% / "
        f"안전자산 {100 - risk_ratio}%, "
        f"총 자산 {total_balance:,}원"
    )

    # 저축 요약
    SAVINGS_TYPES = {"SAVINGS", "DEPOSIT", "PARKING", "CHECKING", "CMA"}
    savings_assets = [a for a in request.assets if a.asset_type in SAVINGS_TYPES]
    if savings_assets:
        savings_summary = ", ".join(
            f"{a.account_name}({a.asset_type}) {a.balance:,}원" for a in savings_assets
        )
    else:
        savings_summary = "저축 계좌 없음"

    initial_state: ProfileState = {
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "expense_summary": expense_summary,
        "asset_summary": asset_summary,
        "savings_summary": savings_summary,
        "expense_comment": "",
        "invest_comment": "",
        "savings_comment": "",
    }

    final_state: ProfileState = await _graph.ainvoke(initial_state)

    return ProfileResponse(
        created_at=datetime.now(timezone.utc),
        expense_comment=final_state["expense_comment"],
        invest_comment=final_state["invest_comment"],
        savings_comment=final_state["savings_comment"],
    )
