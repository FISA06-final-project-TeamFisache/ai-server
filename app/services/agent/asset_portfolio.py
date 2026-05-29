import json
import re
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    FundingSource,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import get_llm

# 출금 우선 계좌 유형 순서 (자금을 끌어오는 원천)
_SOURCE_PRIORITY = ["CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT"]


class AssetPortfolioState(TypedDict):
    invest_amount: int
    porti_type: str
    porti_comment: str
    assets_text: str
    products_text: str
    asset_list: list[dict]
    product_list: list[dict]
    strategy_raw: str
    investment_flows: list[dict]


def _plan_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=(
            "당신은 개인 자산 포트폴리오 전문가입니다.\n"
            "사용자의 월 투자금을 단기·중기·장기 흐름으로 나눠 투자 계획을 설계하세요.\n\n"
            "규칙:\n"
            "- 2~3개 흐름(flow) 생성. 각 흐름의 ratio 합계는 반드시 100\n"
            "- term: 단기(6개월 이내), 중기(1년), 장기(4년 이상)\n"
            "- 공격적 성향(AGGRESSIVE): 주식·ETF 비중 높게 / 안정형(CONSERVATIVE): 예금·채권 위주\n"
            "- gathering_asset_type: 자금이 모이는 계좌 유형 (보유 계좌 목록의 asset_type 중 선택)\n"
            "- product_names: 아래 상품 목록에 있는 이름을 정확히 사용\n"
            "- expected_return_rate: 해당 흐름의 예상 연 수익률(%, 소수점 한 자리)\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"flows":['
            '{"title":"흐름명","term":"단기","summary":"한 줄 설명",'
            '"ratio":30,"gathering_asset_type":"SAVING",'
            '"product_names":["상품명1","상품명2"],'
            '"expected_return_rate":3.5}'
            "]}"
        )),
        HumanMessage(content=(
            f"PorTI 유형: {state['porti_type']}\n"
            f"투자 성향: {state['porti_comment']}\n"
            f"월 투자금: {state['invest_amount']:,}원\n\n"
            f"보유 계좌:\n{state['assets_text']}\n\n"
            f"투자 가능 상품:\n{state['products_text']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    raw = state["strategy_raw"]
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    flows_raw: list[dict] = data.get("flows") or []
    if not flows_raw:
        flows_raw = [
            {
                "title": "분산 투자",
                "term": "중기",
                "summary": "안정적으로 자금을 불려요",
                "ratio": 100,
                "gathering_asset_type": "SAVING",
                "product_names": [],
                "expected_return_rate": 5.0,
            }
        ]

    invest_amount = state["invest_amount"]
    asset_list = state["asset_list"]
    product_list = state["product_list"]

    # 유형별 자산 맵
    asset_by_type: dict[str, list[dict]] = {}
    for a in asset_list:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    # 상품명 → 상품 정보 맵
    product_by_name: dict[str, dict] = {p["name"]: p for p in product_list}

    # 출금 원천 계좌: 우선순위 순으로 첫 번째 매칭
    source_asset = next(
        (a for t in _SOURCE_PRIORITY for a in asset_by_type.get(t, [])),
        asset_list[0] if asset_list else None,
    )

    used_gathering: set[str] = set()

    def pick_gathering(asset_type: str) -> str | None:
        for a in asset_by_type.get(asset_type, []):
            if a["asset_id"] not in used_gathering:
                used_gathering.add(a["asset_id"])
                return a["asset_id"]
        # 해당 유형 없으면 미사용 계좌 중 첫 번째
        for assets in asset_by_type.values():
            for a in assets:
                if a["asset_id"] not in used_gathering:
                    used_gathering.add(a["asset_id"])
                    return a["asset_id"]
        return asset_list[0]["asset_id"] if asset_list else None

    investment_flows = []
    for flow in flows_raw:
        ratio = max(1, int(flow.get("ratio", 100)))
        monthly_amount = round(invest_amount * ratio / 100)
        gathering_id = pick_gathering(flow.get("gathering_asset_type", "SAVING"))

        # 자금 출처: 원천 계좌 (gathering과 다른 경우), 아니면 아무 미사용 계좌
        if source_asset and gathering_id and source_asset["asset_id"] != gathering_id:
            funding_id = source_asset["asset_id"]
        else:
            funding_id = next(
                (a["asset_id"] for assets in asset_by_type.values()
                 for a in assets if a["asset_id"] != gathering_id),
                gathering_id,
            )

        # 포트폴리오 상품 배분
        raw_names: list[str] = flow.get("product_names") or []
        valid_names = [n for n in raw_names if n in product_by_name]
        if not valid_names and product_list:
            valid_names = [product_list[0]["name"]]

        if valid_names:
            per = 100 // len(valid_names)
            remainder = 100 - per * len(valid_names)
            portfolio = [
                {"name": n, "ratio": per + (remainder if i == 0 else 0)}
                for i, n in enumerate(valid_names)
            ]
        else:
            portfolio = []

        investment_flows.append({
            "title": flow.get("title", "투자 플랜"),
            "term": flow.get("term", "중기"),
            "summary": flow.get("summary", ""),
            "funding_sources": [{"asset_id": funding_id, "amount": monthly_amount}],
            "gathering_account": gathering_id,
            "portfolio": portfolio,
        })

    return {**state, "investment_flows": investment_flows}


def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("plan", _plan_flows)
    graph.add_node("parse", _parse_flows)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    assets_text = "\n".join(
        f"- {a.account_name} ({a.asset_type}): 잔액 {a.balance:,}원"
        for a in request.invest_assets
    ) or "보유 계좌 없음"

    products_text = "\n".join(
        f"- [{p.product_type}] {p.institution} '{p.name}' 연 {p.interest_rate}% — {p.description}"
        for p in request.products
    ) or "상품 없음"

    asset_list = [
        {
            "asset_id": str(a.asset_id),
            "asset_type": a.asset_type,
            "account_name": a.account_name,
            "balance": a.balance,
        }
        for a in request.invest_assets
    ]
    product_list = [
        {"name": p.name, "product_type": p.product_type, "interest_rate": p.interest_rate}
        for p in request.products
    ]

    initial_state: AssetPortfolioState = {
        "invest_amount": request.invest_amount,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "assets_text": assets_text,
        "products_text": products_text,
        "asset_list": asset_list,
        "product_list": product_list,
        "strategy_raw": "",
        "investment_flows": [],
    }

    final_state: AssetPortfolioState = await _graph.ainvoke(initial_state)

    return AssetPortfolioResponse(
        created_at=datetime.now(timezone.utc),
        investment_flows=[
            InvestmentPlan(
                title=f["title"],
                term=f["term"],
                summary=f["summary"],
                funding_sources=[
                    FundingSource(asset_id=UUID(s["asset_id"]), amount=s["amount"])
                    for s in f["funding_sources"]
                    if s.get("asset_id")
                ],
                gathering_account=(
                    UUID(f["gathering_account"])
                    if f.get("gathering_account")
                    else request.invest_assets[0].asset_id
                ),
                portfolio=[
                    PortfolioItem(name=p["name"], ratio=p["ratio"])
                    for p in f["portfolio"]
                ],
            )
            for f in final_state["investment_flows"]
        ],
    )
