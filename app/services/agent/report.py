import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.report import HoverDescription, ReportRequest, ReportResponse
from app.services.agent.llm import get_llm


class ReportState(TypedDict):
    year: int
    month: int
    title: str
    target_amount: int
    goal_progress: int
    expense_summary: str
    asset_trend: str
    strategy_raw: str
    trend_comment: str
    event_comment: str
    market_condition: str
    hover_descriptions: list[dict]
    guideline: str
    performance_status: str
    performance_comment: str


def _analyze_report(state: ReportState) -> ReportState:
    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=(
            "당신은 개인 재무 월간 리포트 작성 전문가입니다.\n"
            "사용자의 이번 달 소비/자산 데이터를 분석해 리포트 코멘트를 작성하세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            "{\n"
            '  "trend_comment": "전월 대비 자산/소비 변화 한 줄 분석",\n'
            '  "event_comment": "목표 달성률 기반 한 줄 코멘트",\n'
            '  "market_condition": "현재 시장 상황 한 줄 요약",\n'
            '  "hover_descriptions": [{"category":"소비항목","content":"한 줄 설명"}],\n'
            '  "guideline": "다음 달 소비/저축 가이드라인 한 줄",\n'
            '  "performance_status": "OUTPERFORM 또는 UNDERPERFORM 또는 ON_TRACK",\n'
            '  "performance_comment": "성과 한 줄 코멘트"\n'
            "}"
        )),
        HumanMessage(content=(
            f"{state['year']}년 {state['month']}월 리포트\n"
            f"이벤트 목표: {state['title']} / 목표금액: {state['target_amount']:,}원 / "
            f"달성률: {state['goal_progress']}%\n\n"
            f"자산 변화:\n{state['asset_trend']}\n\n"
            f"이번 달 지출 요약:\n{state['expense_summary']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_report(state: ReportState) -> ReportState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    status = data.get("performance_status", "ON_TRACK")
    if status not in {"OUTPERFORM", "UNDERPERFORM", "ON_TRACK"}:
        status = "ON_TRACK"

    hover = [
        {"category": h.get("category", "기타"), "content": h.get("content", "")}
        for h in (data.get("hover_descriptions") or [])
    ]
    if not hover:
        hover = [{"category": "소비", "content": "이번 달 소비 패턴을 분석했어요."}]

    return {
        **state,
        "trend_comment": data.get("trend_comment", "전월 대비 자산 변화를 분석했어요."),
        "event_comment": data.get("event_comment", f"목표 달성률 {state['goal_progress']}%예요."),
        "market_condition": data.get("market_condition", "현재 시장은 안정적인 흐름입니다."),
        "hover_descriptions": hover,
        "guideline": data.get("guideline", "다음 달도 꾸준한 저축을 이어가세요."),
        "performance_status": status,
        "performance_comment": data.get("performance_comment", "이번 달 재무 목표를 잘 유지했어요."),
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(ReportState)
    graph.add_node("analyze", _analyze_report)
    graph.add_node("parse", _parse_report)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


async def generate_report(request: ReportRequest) -> ReportResponse:
    expense_by_category: dict[str, int] = {}
    for tx in request.transaction_log:
        if tx.amount < 0:
            expense_by_category[tx.category] = (
                expense_by_category.get(tx.category, 0) + abs(tx.amount)
            )

    expense_summary = "\n".join(
        f"  - {cat}: {amt:,}원" for cat, amt in sorted(expense_by_category.items(), key=lambda x: -x[1])
    ) or "  지출 내역 없음"

    if len(request.asset_snapshots) >= 2:
        first = request.asset_snapshots[0].total_amount
        last = request.asset_snapshots[-1].total_amount
        diff = last - first
        asset_trend = f"  기간 중 자산 변화: {first:,}원 → {last:,}원 ({'+' if diff >= 0 else ''}{diff:,}원)"
    elif request.asset_snapshots:
        snap = request.asset_snapshots[0]
        asset_trend = f"  현재 총 자산: {snap.total_amount:,}원 (저축: {snap.savings_amount:,}원 / 투자: {snap.invest_amount:,}원)"
    else:
        asset_trend = "  자산 스냅샷 없음"

    initial_state: ReportState = {
        "year": request.year,
        "month": request.month,
        "title": request.title,
        "target_amount": request.target_amount,
        "goal_progress": request.goal_progress,
        "expense_summary": expense_summary,
        "asset_trend": asset_trend,
        "strategy_raw": "",
        "trend_comment": "",
        "event_comment": "",
        "market_condition": "",
        "hover_descriptions": [],
        "guideline": "",
        "performance_status": "ON_TRACK",
        "performance_comment": "",
    }

    final_state: ReportState = await _graph.ainvoke(initial_state)

    return ReportResponse(
        created_at=datetime.now(timezone.utc),
        trend_comment=final_state["trend_comment"],
        event_comment=final_state["event_comment"],
        market_condition=final_state["market_condition"],
        hover_description=[
            HoverDescription(category=h["category"], content=h["content"])
            for h in final_state["hover_descriptions"]
        ],
        guideline=final_state["guideline"],
        performance_status=final_state["performance_status"],
        performance_comment=final_state["performance_comment"],
    )
