from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    FundingSource,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import invoke_structured

_SOURCE_PRIORITY = ["CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT"]
_MAX_RETRIES = 2


# ── Pydantic 스키마 ────────────────────────────────────────────────────────────

class _ProductItem(BaseModel):
    name: str = Field(default="")
    ratio: int = Field(default=0)


class _FlowItem(BaseModel):
    title: str = Field(default="투자 플랜")
    term: str = Field(default="중기")
    summary: str = Field(default="")
    ratio: int = Field(default=0)
    gathering_asset_type: str = Field(default="SAVING")
    products: list[_ProductItem] = Field(default_factory=list)
    expected_return_rate: float = Field(default=0.0)


class _AssetPortfolioPlan(BaseModel):
    flows: list[_FlowItem] = Field(default_factory=list)


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    porti_type: str
    porti_comment: str
    assets_text: str
    products_text: str
    asset_list: list[dict]
    product_list: list[dict]
    raw_flows: list[dict]        # LLM 출력 (해석 전)
    investment_flows: list[dict] # 최종 해석 완료 결과
    retry_count: int
    feedback: str                # 검증 실패 시 피드백 메시지


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_SYSTEM = (
    "당신은 개인 자산 포트폴리오 전문가입니다.\n"
    "사용자의 월 투자금을 반드시 4개 흐름으로 나눠 투자 계획을 설계하세요.\n\n"
    "추론 순서:\n"
    "1. PorTI 성향을 확인하고 위험 선호도를 결정하세요.\n"
    "2. 4개 흐름 배분 비율을 결정하세요 — flows[*].ratio 합계는 반드시 100.\n"
    "   - 단기(6개월): 안전 자산 위주 (예금·적금)\n"
    "   - 중기(1년): 균형 (채권·혼합)\n"
    "   - 장기1(4년·IRP): IRP 계좌 활용, 세액공제 목적\n"
    "   - 장기2(4년·ISA): ISA 계좌 활용, 비과세 목적\n"
    "3. 각 흐름에 상품 목록에서 정확히 2개를 골라 products를 채우세요.\n"
    "   products[*].ratio 합계도 반드시 100.\n"
    "4. 공격적(AGGRESSIVE): 주식·ETF 비중 높게 / 안정형(CONSERVATIVE): 예금·채권 위주\n\n"
    "예시 (균형형, 월 투자금 120만원):\n"
    "flows[0]: 단기, ratio=30, products=[{예금A,70},{적금B,30}], return=3.5\n"
    "flows[1]: 중기, ratio=30, products=[{채권C,60},{ETF_D,40}], return=5.0\n"
    "flows[2]: 장기(IRP), ratio=20, gathering=IRP, products=[{펀드E,50},{펀드F,50}], return=6.0\n"
    "flows[3]: 장기(ISA), ratio=20, gathering=ISA, products=[{ETF_G,60},{채권H,40}], return=6.5\n"
    "→ flows ratio 합계: 30+30+20+20 = 100 ✓\n"
    "→ 각 products ratio 합계: 100 ✓"
)


# ── 노드 ──────────────────────────────────────────────────────────────────────

def _plan_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    human_parts = [
        f"PorTI 유형: {state['porti_type']}",
        f"투자 성향: {state['porti_comment']}",
        f"월 투자금: {state['invest_amount']:,}원",
        f"\n보유 계좌:\n{state['assets_text']}",
        f"\n투자 가능 상품:\n{state['products_text']}",
    ]

    if state["retry_count"] > 0 and state["feedback"]:
        human_parts.append(
            f"\n⚠️ 이전 응답의 오류 — 반드시 수정하세요:\n{state['feedback']}"
        )

    result = invoke_structured(
        [SystemMessage(content=_SYSTEM), HumanMessage(content="\n".join(human_parts))],
        _AssetPortfolioPlan,
        temperature=0.2,
    )
    raw_flows = [f.model_dump() for f in result.flows] if result else []

    return {**state, "raw_flows": raw_flows, "retry_count": state["retry_count"] + 1}


def _validate(state: AssetPortfolioState) -> AssetPortfolioState:
    flows = state["raw_flows"]
    errors: list[str] = []

    if len(flows) != 4:
        errors.append(f"flows는 정확히 4개여야 합니다. 현재 {len(flows)}개.")

    ratio_sum = sum(f.get("ratio", 0) for f in flows)
    if ratio_sum != 100:
        errors.append(
            f"flows ratio 합계가 {ratio_sum}입니다. 반드시 100이 되도록 각 ratio를 조정하세요."
        )

    for i, f in enumerate(flows):
        products = f.get("products", [])
        if len(products) < 2:
            errors.append(
                f"flows[{i}] '{f.get('title', '')}': products가 {len(products)}개입니다. 정확히 2개 필요."
            )
        else:
            p_sum = sum(p.get("ratio", 0) for p in products)
            if p_sum != 100:
                errors.append(
                    f"flows[{i}] '{f.get('title', '')}': products ratio 합계가 {p_sum}입니다. 100이어야 합니다."
                )

    return {**state, "feedback": "\n".join(f"- {e}" for e in errors)}


def _route(state: AssetPortfolioState) -> str:
    """검증 통과 or 최대 재시도 도달 → resolve, 아니면 → plan"""
    if not state["feedback"]:
        return "resolve"
    if state["retry_count"] >= _MAX_RETRIES:
        return "resolve"
    return "plan"


def _resolve_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    flows_raw = state["raw_flows"]

    _DEFAULTS = [
        {"title": "단기 자금 운용", "term": "단기", "summary": "비상금·생활비 베이스를 단단히 다져요",
         "ratio": 0, "gathering_asset_type": "CHECKING", "products": [], "expected_return_rate": 3.0},
        {"title": "중기 목돈 만들기", "term": "중기", "summary": "중기 목표를 위한 균형 성장 전략",
         "ratio": 0, "gathering_asset_type": "SAVING", "products": [], "expected_return_rate": 5.0},
        {"title": "노후 대비 IRP", "term": "장기", "summary": "IRP로 매년 연말정산 환급까지 챙겨요",
         "ratio": 0, "gathering_asset_type": "IRP", "products": [], "expected_return_rate": 6.0},
        {"title": "절세 ISA 투자", "term": "장기", "summary": "ISA 비과세 한도로 수익률을 더 챙겨요",
         "ratio": 0, "gathering_asset_type": "ISA", "products": [], "expected_return_rate": 7.0},
    ]
    while len(flows_raw) < 4:
        flows_raw.append(_DEFAULTS[len(flows_raw)])

    invest_amount = state["invest_amount"]
    asset_list    = state["asset_list"]
    product_list  = state["product_list"]

    asset_by_type: dict[str, list[dict]] = {}
    for a in asset_list:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    product_by_name: dict[str, dict] = {p["name"]: p for p in product_list}

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
        for assets in asset_by_type.values():
            for a in assets:
                if a["asset_id"] not in used_gathering:
                    used_gathering.add(a["asset_id"])
                    return a["asset_id"]
        return asset_list[0]["asset_id"] if asset_list else None

    def resolve_product_name(ai_name: str) -> str | None:
        if ai_name in product_by_name:
            return ai_name
        for db_name in product_by_name:
            if ai_name in db_name or db_name in ai_name:
                return db_name
        return None

    # ratio 합계가 100이 아니면 균등 재분배 (검증 실패 후 max retry 도달 시 안전망)
    total_ratio = sum(max(1, int(f.get("ratio", 0))) for f in flows_raw[:4])
    if total_ratio != 100:
        per = 100 // 4
        for i, f in enumerate(flows_raw[:4]):
            f["ratio"] = per + (100 - per * 4 if i == 0 else 0)

    investment_flows = []
    for flow in flows_raw[:4]:
        ratio = max(1, int(flow.get("ratio", 25)))
        monthly_amount = round(invest_amount * ratio / 100)
        gathering_id = pick_gathering(flow.get("gathering_asset_type", "SAVING"))

        funding_id = (
            source_asset["asset_id"]
            if source_asset and gathering_id and source_asset["asset_id"] != gathering_id
            else next(
                (a["asset_id"] for assets in asset_by_type.values()
                 for a in assets if a["asset_id"] != gathering_id),
                gathering_id,
            )
        )

        raw_products: list[dict] = flow.get("products") or []
        seen: set[str] = set()
        valid_products: list[dict] = []

        for item in raw_products:
            resolved = resolve_product_name(item.get("name", ""))
            ai_ratio = int(item.get("ratio", 0))
            if resolved and resolved not in seen and ai_ratio > 0:
                valid_products.append({"name": resolved, "ratio": ai_ratio})
                seen.add(resolved)

        for p in product_list:
            if len(valid_products) >= 2:
                break
            if p["name"] not in seen:
                valid_products.append({"name": p["name"], "ratio": 0})
                seen.add(p["name"])

        total_p = sum(p["ratio"] for p in valid_products)
        if not valid_products:
            portfolio = []
        elif total_p != 100 or any(p["ratio"] == 0 for p in valid_products):
            per = 100 // len(valid_products)
            rem = 100 - per * len(valid_products)
            portfolio = [
                {"name": p["name"], "ratio": per + (rem if i == 0 else 0)}
                for i, p in enumerate(valid_products)
            ]
        else:
            portfolio = valid_products

        investment_flows.append({
            "title": flow.get("title", "투자 플랜"),
            "term": flow.get("term", "중기"),
            "summary": flow.get("summary", ""),
            "funding_sources": [{"asset_id": funding_id, "amount": monthly_amount}],
            "gathering_account": gathering_id,
            "portfolio": portfolio,
        })

    return {**state, "investment_flows": investment_flows}


# ── 그래프 ────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)

    graph.add_node("plan",     _plan_flows)
    graph.add_node("validate", _validate)
    graph.add_node("resolve",  _resolve_flows)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges(
        "validate",
        _route,
        {"plan": "plan", "resolve": "resolve"},
    )
    graph.add_edge("resolve", END)

    return graph.compile()


_graph = _build_graph()


# ── 진입점 ────────────────────────────────────────────────────────────────────

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
        {"asset_id": str(a.asset_id), "asset_type": a.asset_type,
         "account_name": a.account_name, "balance": a.balance}
        for a in request.invest_assets
    ]
    product_list = [
        {"name": p.name, "product_type": p.product_type, "interest_rate": p.interest_rate}
        for p in request.products
    ]

    initial_state: AssetPortfolioState = {
        "invest_amount": request.invest_amount,
        "porti_type":    request.porti_type,
        "porti_comment": request.porti_comment,
        "assets_text":   assets_text,
        "products_text": products_text,
        "asset_list":    asset_list,
        "product_list":  product_list,
        "raw_flows":     [],
        "investment_flows": [],
        "retry_count":   0,
        "feedback":      "",
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
                    for s in f["funding_sources"] if s.get("asset_id")
                ],
                gathering_account=(
                    UUID(f["gathering_account"])
                    if f.get("gathering_account")
                    else request.invest_assets[0].asset_id
                ),
                amount=sum(s["amount"] for s in f["funding_sources"] if s.get("asset_id")),
                portfolio=[
                    PortfolioItem(name=p["name"], ratio=p["ratio"])
                    for p in f["portfolio"]
                ],
            )
            for f in final_state["investment_flows"]
        ],
    )
