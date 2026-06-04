import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.schemas.report import HoverDescription, ReportRequest, ReportResponse
from app.services.agent.llm import get_llm


def _fetch_market_news() -> str:
    """Tavily로 최신 한국 금융 시장 뉴스 검색. 키 없으면 빈 문자열 반환."""
    if not settings.tavily_api_key:
        return ""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(
            query="한국 주식 금리 경제 시장 동향",
            search_depth="basic",
            max_results=3,
        )
        snippets = [r.get("content", "") for r in results.get("results", []) if r.get("content")]
        return "\n".join(snippets[:3])
    except Exception:
        return ""


class ReportState(TypedDict):
    year: int
    month: int
    mini_challenges_summary: str
    income_summary: str
    expense_summary: str
    expense_categories: list[str]
    asset_trend: str
    market_news: str
    strategy_raw: str
    trend_comment: str
    challenge_comment: str
    market_condition: str
    hover_descriptions: list[dict]
    guideline: str


def _analyze_report(state: ReportState) -> ReportState:
    llm = get_llm(temperature=0.3)

    market_section = (
        f"\n최신 시장 뉴스:\n{state['market_news']}"
        if state["market_news"]
        else "\n(시장 데이터 없음 — market_condition은 사용자 데이터 기반으로만 작성)"
    )

    messages = [
        SystemMessage(content=(
            "당신은 개인 재무 월간 리포트 작성 전문가입니다.\n"
            "사용자의 이번 달 소비/자산 데이터를 분석해 리포트 코멘트를 작성하세요.\n"
            "이모지나 이모티콘은 사용하지 마세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            "{\n"
            '  "trend_comment": "전월 대비 자산/소비 변화 한 줄 분석",\n'
            '  "challenge_comment": "미니 챌린지 달성 현황 기반 한 줄 코멘트",\n'
            '  "market_condition": "시장 뉴스 기반 한 줄 요약 (뉴스 없으면 사용자 자산 흐름 기반으로 작성)",\n'
            '  "hover_descriptions": [{"category": "실제 지출 카테고리명", "content": "한 줄 설명"}],\n'
            '  "guideline": "다음 달 소비/저축 가이드라인 한 줄",\n'
            "}\n\n"
            "hover_descriptions는 반드시 아래 실제 지출 카테고리만 사용하세요."
        )),
        HumanMessage(content=(
            f"{state['year']}년 {state['month']}월 리포트\n\n"
            f"미니 챌린지 현황:\n{state['mini_challenges_summary']}\n\n"
            f"자산 변화:\n{state['asset_trend']}\n\n"
            f"이번 달 수입 요약:\n{state['income_summary']}\n\n"
            f"이번 달 지출 요약 (hover_descriptions 카테고리는 아래 항목만 사용):\n{state['expense_summary']}"
            f"{market_section}"
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
        "challenge_comment": data.get("challenge_comment", "이번 달 챌린지 현황을 확인했어요."),
        "market_condition": data.get("market_condition", "시장 데이터를 불러올 수 없었어요."),
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
    NON_EXPENSE_CATEGORIES = {"저축", "투자", "이체", "자동이체", "적금"}

    # 수입/지출 분리 집계
    income_by_category: dict[str, int] = {}
    expense_by_category: dict[str, int] = {}
    for tx in request.transaction_log:
        if tx.amount > 0:
            income_by_category[tx.category] = income_by_category.get(tx.category, 0) + tx.amount
        elif tx.category not in NON_EXPENSE_CATEGORIES:
            expense_by_category[tx.category] = expense_by_category.get(tx.category, 0) + abs(tx.amount)

    income_summary = "\n".join(
        f"  - {cat}: {amt:,}원" for cat, amt in sorted(income_by_category.items(), key=lambda x: -x[1])
    ) or "  수입 내역 없음"

    expense_summary = "\n".join(
        f"  - {cat}: {amt:,}원" for cat, amt in sorted(expense_by_category.items(), key=lambda x: -x[1])
    ) or "  지출 내역 없음"

    # 자산 변화 (savings/invest 분리)
    if len(request.asset_snapshots) >= 2:
        first = request.asset_snapshots[0]
        last = request.asset_snapshots[-1]
        diff = last.total_amount - first.total_amount
        asset_trend = (
            f"  총 자산: {first.total_amount:,}원 → {last.total_amount:,}원 "
            f"({'+' if diff >= 0 else ''}{diff:,}원)\n"
            f"  저축: {first.savings_amount:,}원 → {last.savings_amount:,}원 / "
            f"투자: {first.invest_amount:,}원 → {last.invest_amount:,}원"
        )
    elif request.asset_snapshots:
        snap = request.asset_snapshots[0]
        asset_trend = (
            f"  현재 총 자산: {snap.total_amount:,}원 "
            f"(저축: {snap.savings_amount:,}원 / 투자: {snap.invest_amount:,}원)"
        )
    else:
        asset_trend = "  자산 스냅샷 없음"

    # 미니 챌린지 요약
    if request.mini_challenges:
        mini_challenges_summary = "\n".join(
            f"  - [{c.status}] {c.title} ({c.challenge_type}, 목표 {c.target})"
            + (f" — 완료: {c.completed_at.strftime('%Y-%m-%d')}" if c.completed_at else "")
            for c in request.mini_challenges
        )
    else:
        mini_challenges_summary = "  챌린지 없음"

    market_news = _fetch_market_news()

    initial_state: ReportState = {
        "year": request.year,
        "month": request.month,
        "mini_challenges_summary": mini_challenges_summary,
        "expense_categories": list(expense_by_category.keys()),
        "income_summary": income_summary,
        "expense_summary": expense_summary,
        "asset_trend": asset_trend,
        "market_news": market_news,
        "strategy_raw": "",
        "trend_comment": "",
        "challenge_comment": "",
        "market_condition": "",
        "hover_descriptions": [],
        "guideline": "",
        "performance_status": "ON_TRACK",
        "performance_comment": "",
    }

    final_state: ReportState = await _graph.ainvoke(initial_state)

    valid_hover = [
        HoverDescription(category=h["category"], content=h["content"])
        for h in final_state["hover_descriptions"]
        if h.get("category") not in NON_EXPENSE_CATEGORIES
    ]
    if not valid_hover:
        valid_hover = [
            HoverDescription(category=h["category"], content=h["content"])
            for h in final_state["hover_descriptions"]
        ]

    return ReportResponse(
        created_at=datetime.now(timezone.utc),
        trend_comment=final_state["trend_comment"],
        challenge_comment=final_state["challenge_comment"],
        market_condition=final_state["market_condition"],
        hover_description=valid_hover,
        guideline=final_state["guideline"],
    )
