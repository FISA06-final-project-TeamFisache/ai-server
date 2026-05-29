import json
import re
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.event import (
    EventAssetPortfolioRequest,
    EventAssetPortfolioResponse,
    FundingSource,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import get_llm


class EventAssetPortfolioState(TypedDict):
    title: str
    target_amount: int
    months_left: int
    invest_amount: int
    porti_type: str
    porti_comment: str
    existing_flows_text: str
    products_text: str
    assets_text: str
    asset_list: list[dict]
    product_list: list[dict]
    existing_flows: list[dict]
    strategy_raw: str
    new_flows: list[dict]


def _plan_event_portfolio(state: EventAssetPortfolioState) -> EventAssetPortfolioState:
    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=(
            "당신은 이벤트(목표) 기반 자산 포트폴리오 전문가입니다.\n"
            "기존 투자 흐름에 이벤트 목표 달성을 위한 새 흐름을 추가하세요.\n\n"
            "규칙:\n"
            "- 기존 흐름은 그대로 유지\n"
            "- 이벤트 목표를 위한 흐름 1개 추가 (term: 단기/중기/장기 결정)\n"
            "- 단기(≤6개월): 안전 자산(예금/적금) 위주\n"
            "- 중장기: 성향에 따라 주식/채권 혼합\n"
            "- gathering_asset_type: 보유 계좌 유형 중 선택\n"
            "- product_names: 아래 상품 목록에 있는 이름 정확히 사용\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"new_flow":{"title":"흐름명","term":"단기","summary":"설명",'
            '"gathering_asset_type":"SAVING","product_names":["상품명"],"monthly_ratio":30}}'
        )),
        HumanMessage(content=(
            f"이벤트: {state['title']}\n"
            f"목표 금액: {state['target_amount']:,}원\n"
            f"남은 기간: {state['months_left']}개월\n"
            f"월 투자금: {state['invest_amount']:,}원\n"
            f"PorTI: {state['porti_type']} — {state['porti_comment']}\n\n"
            f"기존 흐름:\n{state['existing_flows_text']}\n\n"
            f"보유 계좌:\n{state['assets_text']}\n\n"
            f"투자 가능 상품:\n{state['products_text']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_event_portfolio(state: EventAssetPortfolioState) -> EventAssetPortfolioState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    new_flow_raw = data.get("new_flow") or {}
    asset_list = state["asset_list"]
    product_map = {p["name"]: p for p in state["product_list"]}

    asset_by_type: dict[str, list[dict]] = {}
    for a in asset_list:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    used_gathering = {f.get("gathering_account") for f in state["existing_flows"]}

    def pick_asset(asset_type: str) -> str | None:
        for a in asset_by_type.get(asset_type, []):
            if a["asset_id"] not in used_gathering:
                return a["asset_id"]
        for assets in asset_by_type.values():
            for a in assets:
                if a["asset_id"] not in used_gathering:
                    return a["asset_id"]
        return asset_list[0]["asset_id"] if asset_list else None

    monthly_ratio = max(10, int(new_flow_raw.get("monthly_ratio", 30)))
    monthly_amount = round(state["invest_amount"] * monthly_ratio / 100)
    gathering_type = new_flow_raw.get("gathering_asset_type", "SAVING")
    gathering_id = pick_asset(gathering_type)

    source_id = next(
        (a["asset_id"] for assets in asset_by_type.values()
         for a in assets if a["asset_id"] != gathering_id),
        gathering_id,
    )

    raw_names: list[str] = new_flow_raw.get("product_names") or []
    valid_names = [n for n in raw_names if n in product_map]
    if not valid_names and state["product_list"]:
        valid_names = [state["product_list"][0]["name"]]

    if valid_names:
        per = 100 // len(valid_names)
        remainder = 100 - per * len(valid_names)
        portfolio = [
            {"name": n, "ratio": per + (remainder if i == 0 else 0)}
            for i, n in enumerate(valid_names)
        ]
    else:
        portfolio = []

    new_event_flow = {
        "title": new_flow_raw.get("title", f"{state['title']} 목표 흐름"),
        "term": new_flow_raw.get("term", "단기"),
        "summary": new_flow_raw.get("summary", f"{state['title']}을 위한 전용 저축 흐름이에요"),
        "funding_sources": [{"asset_id": source_id, "amount": monthly_amount,
                              "account_name": ""}],
        "gathering_account": gathering_id,
        "amount": monthly_amount,
        "portfolio": portfolio,
    }

    all_flows = list(state["existing_flows"]) + [new_event_flow]
    return {**state, "new_flows": all_flows}


def _build_graph() -> StateGraph:
    graph = StateGraph(EventAssetPortfolioState)
    graph.add_node("plan", _plan_event_portfolio)
    graph.add_node("parse", _parse_event_portfolio)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


async def asset_portfolio_event(
    request: EventAssetPortfolioRequest,
) -> EventAssetPortfolioResponse:
    now = datetime.now(timezone.utc)
    months_left = max(1, int((request.deadline - now).days / 30))

    existing_flows = [
        {
            "title": f.title, "term": f.term, "summary": f.summary,
            "gathering_account": str(f.gathering_account),
            "amount": f.amount,
            "funding_sources": [
                {"asset_id": str(s.asset_id), "amount": s.amount, "account_name": s.account_name}
                for s in f.funding_sources
            ],
            "portfolio": [{"name": p.name, "ratio": p.ratio} for p in f.portfolio],
        }
        for f in request.investment_flows
    ]

    existing_flows_text = "\n".join(
        f"- [{f['term']}] {f['title']}: {f['summary']}"
        for f in existing_flows
    ) or "기존 흐름 없음"

    assets_text = "\n".join(
        f"- {a.account_name} ({a.asset_type}): {a.balance:,}원"
        for a in request.invest_assets
    ) or "보유 계좌 없음"

    products_text = "\n".join(
        f"- [{p.product_type}] {p.institution} '{p.name}' 연 {p.interest_rate}%"
        for p in request.products
    ) or "상품 없음"

    asset_list = [
        {"asset_id": str(a.asset_id), "asset_type": a.asset_type, "account_name": a.account_name}
        for a in request.invest_assets
    ]
    product_list = [{"name": p.name, "product_type": p.product_type} for p in request.products]

    initial_state: EventAssetPortfolioState = {
        "title": request.title,
        "target_amount": request.target_amount,
        "months_left": months_left,
        "invest_amount": request.invest_amount,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "existing_flows_text": existing_flows_text,
        "assets_text": assets_text,
        "products_text": products_text,
        "asset_list": asset_list,
        "product_list": product_list,
        "existing_flows": existing_flows,
        "strategy_raw": "",
        "new_flows": [],
    }

    final_state: EventAssetPortfolioState = await _graph.ainvoke(initial_state)

    investment_flows = []
    for f in final_state["new_flows"]:
        try:
            gathering_id = UUID(f["gathering_account"]) if f.get("gathering_account") else (
                request.invest_assets[0].asset_id if request.invest_assets else None
            )
            if gathering_id is None:
                continue
            funding_sources = [
                FundingSource(
                    account_name=s.get("account_name", ""),
                    asset_id=UUID(s["asset_id"]) if s.get("asset_id") else request.invest_assets[0].asset_id,
                    amount=s.get("amount", 0),
                )
                for s in f.get("funding_sources", [])
                if s.get("asset_id")
            ]
            investment_flows.append(InvestmentPlan(
                title=f["title"],
                term=f["term"],
                summary=f["summary"],
                funding_sources=funding_sources,
                gathering_account=gathering_id,
                amount=f.get("amount", 0),
                portfolio=[PortfolioItem(name=p["name"], ratio=p["ratio"]) for p in f.get("portfolio", [])],
            ))
        except Exception:
            continue

    if not investment_flows and request.investment_flows:
        investment_flows = list(request.investment_flows)

    return EventAssetPortfolioResponse(created_at=now, investment_flows=investment_flows)
