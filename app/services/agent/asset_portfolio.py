import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

import httpx
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    GatheringAccount,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import get_llm, invoke_structured, openrouter_api_key, openrouter_base_url
from app.services.rag.db import get_pool

logger = logging.getLogger(__name__)

_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

_INVEST_PRODUCT_TYPES = ["STOCK", "ETF", "BOND"]
_GATHER_PRODUCT_TYPES = ["CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT", "ISA", "IRP", "PENSION_SAVINGS"]

# 4개 고정 흐름 스펙
# can_invest_fixed: True=항상, False=항상 아님, None=ISA일 때만
_FLOW_SPECS = [
    {
        "flow_type": "단기",
        "term": "단기",
        "investment_months": 6,
        "gather_priority": ["CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT"],
        "can_invest_fixed": False,
    },
    {
        "flow_type": "중기",
        "term": "중기",
        "investment_months": 60,
        "gather_priority": ["ISA", "SAVING", "DEPOSIT", "CHECKING"],
        "can_invest_fixed": None,
    },
    {
        "flow_type": "장기1",
        "term": "장기",
        "investment_months": 240,
        "gather_priority": ["PENSION_SAVINGS"],
        "can_invest_fixed": True,
    },
    {
        "flow_type": "장기2",
        "term": "장기",
        "investment_months": 240,
        "gather_priority": ["IRP"],
        "can_invest_fixed": True,
    },
]

# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    interest: str
    invest_interest: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    asset_by_type: dict[str, list[dict]]
    top_invest_products: list[dict]
    gather_products: list[dict]
    invest_products_text: str
    gather_products_text: str
    flow_defs: list[dict]
    flow_accounts: list[dict]
    trend_context: str
    flow_products: list[dict]
    reflection: dict
    investment_flows: list[dict]

# ── 노드 ──────────────────────────────────────────────────────────────────────

# ── 임베딩 ─────────────────────────────────────────────────────────────────────

async def _get_embedding(text: str) -> list[float] | None:
    if not openrouter_api_key or not text.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{openrouter_base_url}/embeddings",
                headers={"Authorization": f"Bearer {openrouter_api_key}"},
                json={"model": _EMBED_MODEL, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
    except httpx.HTTPError as e:
        logger.warning("임베딩 생성 실패: %s", e)
        return None


# ── Node: preprocess ──────────────────────────────────────────────────────────

async def _preprocess(state: AssetPortfolioState) -> AssetPortfolioState:
    asset_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    query_parts = list(state["invest_interest"]) + ([state["interest"]] if state["interest"] else [])
    query_vector = await _get_embedding(" ".join(query_parts))

    pool = await get_pool()
    top_invest_products: list[dict] = []
    gather_products: list[dict] = []

    if pool:
        try:
            if query_vector:
                vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, interest_rate, description "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY embedding <=> $2::vector "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES, vec_str,
                )
            else:
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, interest_rate, description "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY interest_rate DESC NULLS LAST "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES,
                )
            top_invest_products = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("투자 상품 조회 실패: %s", e)

        try:
            rows = await pool.fetch(
                "SELECT product_type, institution, name, interest_rate, description "
                "FROM products "
                "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST",
                _GATHER_PRODUCT_TYPES,
            )
            gather_products = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("모으기 상품 조회 실패: %s", e)

    invest_products_text = "\n".join(
        f"- [{p['product_type']}] {p['institution']} '{p['name']}' "
        f"연 {p['interest_rate'] or '-'}% — {(p['description'] or '')[:60]}"
        for p in top_invest_products
    ) or "상품 없음"

    gather_products_text = "\n".join(
        f"- [{p['product_type']}] {p['institution']} '{p['name']}' 연 {p['interest_rate'] or '-'}%"
        for p in gather_products
    ) or "상품 없음"

    return {
        **state,
        "asset_by_type": asset_by_type,
        "top_invest_products": top_invest_products,
        "gather_products": gather_products,
        "invest_products_text": invest_products_text,
        "gather_products_text": gather_products_text,
    }


# ── Node: define_flows ────────────────────────────────────────────────────────

async def _define_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=(
            "당신은 개인 자산관리 전문가입니다.\n"
            "아래 4개 고정 투자 흐름에 사용자 맞춤 제목과 한 줄 요약을 작성하세요.\n\n"
            "- 단기 (6개월): 유동성 확보·단기 목돈 마련\n"
            "- 중기 (60개월): 5년 중기 목표 달성\n"
            "- 장기1 (240개월): 연금저축계좌 활용 20년 노후 대비\n"
            "- 장기2 (240개월): IRP 활용 20년 노후 대비\n\n"
            "규칙:\n"
            "- title: 사용자 관심사·성향 반영, 15자 이내\n"
            "- summary: 이 흐름의 목적과 전략 1문장\n"
            "- ratio: 이 흐름에 배분할 투자금 비중(%), 4개 합계 반드시 100\n"
            "  · 단기 비중: 유동성 필요도·단기 목표 여부로 판단 (보통 10~30%)\n"
            "  · 중기 비중: 5년 내 목표 크기로 판단 (보통 20~40%)\n"
            "  · 장기1+장기2: 노후 대비 중요도로 판단, 합산 40~70% 권장\n\n"
            "반드시 JSON만 응답:\n"
            '{"flows":['
            '{"flow_type":"단기","title":"","summary":"","ratio":20},'
            '{"flow_type":"중기","title":"","summary":"","ratio":30},'
            '{"flow_type":"장기1","title":"","summary":"","ratio":25},'
            '{"flow_type":"장기2","title":"","summary":"","ratio":25}'
            "]}"
        )),
        HumanMessage(content=(
            f"PorTI 유형: {state['porti_type']}\n"
            f"투자 성향: {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interest']) or '없음'}\n"
            f"월 투자금: {state['invest_amount']:,}원"
        )),
    ]
    result = await llm.ainvoke(messages)

    try:
        match = re.search(r"\{.*\}", result.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        llm_map = {f["flow_type"]: f for f in data.get("flows", [])}
    except Exception as e:
        logger.warning("흐름 정의 JSON 파싱 실패: %s", e)
        llm_map = {}

    # ratio 합계가 100이 아니면 균등 분배로 fallback
    raw_ratios = [int(float(llm_map.get(s["flow_type"], {}).get("ratio") or 25)) for s in _FLOW_SPECS]
    total_ratio = sum(raw_ratios)
    if total_ratio != 100:
        raw_ratios = [25, 25, 25, 25]

    flow_defs = []
    for spec, ratio in zip(_FLOW_SPECS, raw_ratios):
        ft = spec["flow_type"]
        llm_f = llm_map.get(ft, {})
        flow_defs.append({
            "flow_type": ft,
            "term": spec["term"],
            "investment_months": spec["investment_months"],
            "title": llm_f.get("title") or f"{ft} 투자 플랜",
            "summary": llm_f.get("summary") or "",
            "ratio": ratio,
        })

    return {**state, "flow_defs": flow_defs}


# ── Node: select_accounts ─────────────────────────────────────────────────────

def _select_accounts(state: AssetPortfolioState) -> AssetPortfolioState:
    asset_by_type = state["asset_by_type"]

    gather_by_type: dict[str, list[dict]] = {}
    for p in state["gather_products"]:
        gather_by_type.setdefault(p["product_type"], []).append(p)

    used_ids: set[str] = set()
    flow_accounts = []

    for spec in _FLOW_SPECS:
        ft = spec["flow_type"]

        matched_asset: dict | None = None
        matched_type: str | None = None
        for atype in spec["gather_priority"]:
            candidates = [a for a in asset_by_type.get(atype, []) if a["asset_id"] not in used_ids]
            if candidates:
                matched_asset = candidates[0]
                matched_type = atype
                used_ids.add(matched_asset["asset_id"])
                break

        # 우선순위 계좌 없으면 임의 선택
        if not matched_asset:
            for atype, assets in asset_by_type.items():
                for a in assets:
                    if a["asset_id"] not in used_ids:
                        matched_asset = a
                        matched_type = atype
                        used_ids.add(a["asset_id"])
                        break
                if matched_asset:
                    break

        # can_invest 결정
        fixed = spec["can_invest_fixed"]
        if fixed is True:
            can_invest = True
        elif fixed is False:
            can_invest = False
        else:
            can_invest = matched_type == "ISA"

        # 해당 유형 최고 금리 추천 상품
        best_product = (gather_by_type.get(matched_type or "SAVING") or [None])[0]

        ga: dict = {
            "name": (
                best_product["name"] if best_product
                else (matched_asset["account_name"] if matched_asset else "자유적금")
            ),
            "type": matched_type or "SAVING",
            "institution": best_product["institution"] if best_product else "",
            "interest_rate": float(best_product["interest_rate"] or 0.0) if best_product else 0.0,
        }

        flow_accounts.append({
            "flow_type": ft,
            "gathering_asset_id": matched_asset["asset_id"] if matched_asset else None,
            "gathering_asset_type": matched_type,
            "gathering_account": ga,
            "can_invest": can_invest,
        })

    return {**state, "flow_accounts": flow_accounts}


# ── Node: search_trends ───────────────────────────────────────────────────────

async def _search_trends(state: AssetPortfolioState) -> AssetPortfolioState:
    trend_context = ""
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_api_key:
        try:
            tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
            results = await tool.ainvoke({"query": "2025 ETF 채권 투자 트렌드 한국 추천"})
            if isinstance(results, list):
                trend_context = "\n".join(
                    r.get("content", "")[:200] for r in results[:3]
                )
        except Exception as e:
            logger.warning("트렌드 검색 실패: %s", e)
    return {**state, "trend_context": trend_context}


# ── Node: select_products ─────────────────────────────────────────────────────

async def _select_products(state: AssetPortfolioState) -> AssetPortfolioState:
    llm = get_llm(temperature=0.3)

    can_invest_map = {fa["flow_type"]: fa["can_invest"] for fa in state["flow_accounts"]}
    invest_flows = [fd for fd in state["flow_defs"] if can_invest_map.get(fd["flow_type"])]

    if not invest_flows:
        flow_products = [{"flow_type": spec["flow_type"], "portfolio": []} for spec in _FLOW_SPECS]
        return {**state, "flow_products": flow_products}

    context = state["invest_products_text"]
    if state.get("trend_context"):
        context += f"\n\n[시장 트렌드]\n{state['trend_context']}"

    flows_desc = "\n".join(
        f'- {fd["flow_type"]} ({fd["term"]}, {fd["investment_months"]}개월): {fd["summary"]}'
        for fd in invest_flows
    )
    target_keys = ", ".join(f'"{fd["flow_type"]}"' for fd in invest_flows)

    messages = [
        SystemMessage(content=(
            "당신은 포트폴리오 전문가입니다.\n"
            "투자 가능 흐름 각각에 ETF·채권 포트폴리오를 구성하세요.\n\n"
            "규칙:\n"
            "- 상품명은 반드시 아래 상품 목록에 있는 이름만 사용\n"
            "- 각 흐름 portfolio의 ratio 합계 = 100\n"
            "- comment: 이 상품을 선택한 이유 1문장 (한국어)\n"
            "- 공격적 성향: 주식형 ETF 비중 높게 / 안정 성향: 채권·배당 ETF 위주\n"
            "- 장기 흐름: 성장형 ETF 높게 / 단기·중기: 안정형 혼합\n\n"
            f"[투자 가능 흐름]\n{flows_desc}\n\n"
            f"[투자 상품 목록]\n{context}\n\n"
            f"반드시 JSON만 응답. 포함할 flow_type: [{target_keys}]\n"
            '{"flow_products":['
            '{"flow_type":"장기1","portfolio":[{"name":"상품명","ratio":60,"comment":"이유"}]}'
            "]}"
        )),
        HumanMessage(content=(
            f"PorTI: {state['porti_type']} / {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interest']) or '없음'}"
        )),
    ]
    result = await llm.ainvoke(messages)

    try:
        match = re.search(r"\{.*\}", result.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        llm_map = {fp["flow_type"]: fp["portfolio"] for fp in data.get("flow_products", [])}
    except Exception as e:
        logger.warning("상품 선택 JSON 파싱 실패: %s", e)
        llm_map = {}

    flow_products = [
        {
            "flow_type": spec["flow_type"],
            "portfolio": llm_map.get(spec["flow_type"], []) if can_invest_map.get(spec["flow_type"]) else [],
        }
        for spec in _FLOW_SPECS
    ]

    return {**state, "flow_products": flow_products}


# ── Node: reflect ─────────────────────────────────────────────────────────────

async def _reflect(state: AssetPortfolioState) -> AssetPortfolioState:
    llm = get_llm(temperature=0.1)

    valid_names = {p["name"] for p in state["top_invest_products"]}
    lines = []
    for fp in state["flow_products"]:
        if not fp["portfolio"]:
            continue
        total_ratio = sum(item.get("ratio", 0) for item in fp["portfolio"])
        invalid = [item["name"] for item in fp["portfolio"] if item.get("name") not in valid_names]
        lines.append(
            f"[{fp['flow_type']}] ratio합={total_ratio}, "
            f"상품={[i['name'] for i in fp['portfolio']]}"
            + (f", 미존재상품={invalid}" if invalid else "")
        )

    if not lines:
        return {**state, "reflection": {"is_valid": True, "issues": [], "suggestions": []}}

    messages = [
        SystemMessage(content=(
            "포트폴리오 검증자입니다. 아래를 확인하고 JSON으로 응답하세요.\n"
            "1. 각 흐름 ratio 합계가 100인가?\n"
            "2. 모든 상품명이 유효한가?\n"
            "3. 투자 성향에 맞는 비중인가?\n\n"
            '{"is_valid":true,"issues":[],"suggestions":[]}'
        )),
        HumanMessage(content=(
            f"성향: {state['porti_type']} / {state['porti_comment']}\n\n"
            + "\n".join(lines)
        )),
    ]
    result = await llm.ainvoke(messages)

    try:
        match = re.search(r"\{.*\}", result.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception as e:
        logger.warning("포트폴리오 검증 JSON 파싱 실패: %s", e)
        data = {}

    return {**state, "reflection": {
        "is_valid": data.get("is_valid", True),
        "issues": data.get("issues", []),
        "suggestions": data.get("suggestions", []),
    }}


# ── Node: refine ──────────────────────────────────────────────────────────────

async def _refine(state: AssetPortfolioState) -> AssetPortfolioState:
    llm = get_llm(temperature=0.3)
    reflection = state["reflection"]

    current_json = json.dumps(
        [fp for fp in state["flow_products"] if fp["portfolio"]],
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(content=(
            "포트폴리오 수정 전문가입니다. 검토 의견을 반영해 포트폴리오를 수정하세요.\n\n"
            f"지적 사항: {', '.join(reflection.get('issues', []))}\n"
            f"개선 제안: {', '.join(reflection.get('suggestions', []))}\n\n"
            f"[투자 가능 상품]\n{state['invest_products_text']}\n\n"
            "JSON만 응답 (입력과 동일한 구조):\n"
            '{"flow_products":[{"flow_type":"","portfolio":[{"name":"","ratio":0,"comment":""}]}]}'
        )),
        HumanMessage(content=f"현재 포트폴리오:\n{current_json}"),
    ]
    result = await llm.ainvoke(messages)

    try:
        match = re.search(r"\{.*\}", result.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        refined_map = {fp["flow_type"]: fp["portfolio"] for fp in data.get("flow_products", [])}
    except Exception as e:
        logger.warning("포트폴리오 수정 JSON 파싱 실패: %s", e)
        refined_map = {}

    new_flow_products = [
        {
            "flow_type": fp["flow_type"],
            "portfolio": refined_map.get(fp["flow_type"], fp["portfolio"]),
        }
        for fp in state["flow_products"]
    ]

    return {**state, "flow_products": new_flow_products}


# ── Node: calculate ───────────────────────────────────────────────────────────

def _calculate(state: AssetPortfolioState) -> AssetPortfolioState:
    invest_amount = state["invest_amount"]

    product_by_name: dict[str, dict] = {}
    for p in state["top_invest_products"] + state["gather_products"]:
        product_by_name.setdefault(p["name"], p)

    flow_defs_map = {fd["flow_type"]: fd for fd in state["flow_defs"]}
    flow_accounts_map = {fa["flow_type"]: fa for fa in state["flow_accounts"]}
    flow_products_map = {fp["flow_type"]: fp["portfolio"] for fp in state["flow_products"]}

    def resolve_product_name(ai_name: str) -> str | None:
        if ai_name in product_by_name:
            return ai_name
        for db_name in product_by_name:
            if ai_name in db_name or db_name in ai_name:
                return db_name
        return None

    # ratio 합계가 100이 아니면 균등 재분배 (검증 실패 후 max retry 도달 시 안전망)
    flow_defs_list = state["flow_defs"]
    total_ratio = sum(max(1, int(f.get("ratio", 0))) for f in flow_defs_list[:4])
    if total_ratio != 100:
        per = 100 // 4
        for i, f in enumerate(flow_defs_list[:4]):
            f["ratio"] = per + (100 - per * 4 if i == 0 else 0)

    investment_flows = []
    for spec in _FLOW_SPECS:
        ft = spec["flow_type"]
        fd = flow_defs_map.get(ft, {})
        fa = flow_accounts_map.get(ft, {})
        portfolio_raw = flow_products_map.get(ft, [])
        months = spec["investment_months"]
        amount = round(invest_amount * fd.get("ratio", 25) / 100)

        ga = fa.get("gathering_account", {})
        ga_rate = float(ga.get("interest_rate", 0.0) or 0.0)

        # 기대 수익률: 투자 포트폴리오 가중 평균 or 모으기 계좌 금리
        if portfolio_raw:
            weighted = sum(
                float(product_by_name.get(item.get("name", ""), {}).get("interest_rate") or 5.0)
                * item.get("ratio", 0) / 100
                for item in portfolio_raw
            )
            expected_rr = weighted if weighted > 0 else 5.0
        else:
            expected_rr = ga_rate

        # 적립식 복리 FV: PMT × ((1+r_m)^n - 1) / r_m
        r_m = expected_rr / 100 / 12
        if r_m > 0:
            expected_amount = amount * ((math.pow(1 + r_m, months) - 1) / r_m)
        else:
            expected_amount = float(amount * months)

        # PortfolioItem 구성 + ratio 정규화
        portfolio_items: list[dict] = []
        if portfolio_raw:
            for item in portfolio_raw:
                p = product_by_name.get(item.get("name", ""), {})
                portfolio_items.append({
                    "type": p.get("product_type", "STOCK"),
                    "name": item.get("name", ""),
                    "ratio": item.get("ratio", 0),
                    "interest_rate": float(p.get("interest_rate") or 0.0),
                    "comment": item.get("comment", ""),
                })
            total_ratio = sum(i["ratio"] for i in portfolio_items)
            if 0 < total_ratio != 100:
                for i in portfolio_items:
                    i["ratio"] = round(i["ratio"] * 100 / total_ratio)
                diff = 100 - sum(i["ratio"] for i in portfolio_items)
                if portfolio_items and diff:
                    portfolio_items[0]["ratio"] += diff

        investment_flows.append({
            "flow_type": ft,
            "title": fd.get("title", f"{ft} 투자 플랜"),
            "term": spec["term"],
            "summary": fd.get("summary", ""),
            "gathering_id": fa.get("gathering_asset_id"),
            "gathering_account": ga,
            "amount": amount,
            "account_comment": (
                f"{ga.get('type', '')} 계좌({ga.get('name', '')})에 "
                f"매월 {amount:,}원씩 납입하세요."
            ),
            "portfolio": portfolio_items,
            "expected_rr_pct": round(expected_rr, 1),
            "investment_months": months,
            "expected_amount": round(expected_amount),
            "rr_comment": (
                f"연 {expected_rr:.1f}% 기준 {months}개월 적립식 복리 시 "
                f"약 {round(expected_amount):,}원 예상"
            ),
        })

    return {**state, "investment_flows": investment_flows}


# ── 조건 엣지 ─────────────────────────────────────────────────────────────────

def _should_search_trends(state: AssetPortfolioState) -> str:
    has_can_invest = any(fa["can_invest"] for fa in state["flow_accounts"])
    needs_search = has_can_invest and not state["invest_interest"]
    return "search_trends" if needs_search else "select_products"


def _should_refine(state: AssetPortfolioState) -> str:
    return "refine" if not state["reflection"].get("is_valid", True) else "calculate"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("preprocess", _preprocess)
    graph.add_node("define_flows", _define_flows)
    graph.add_node("select_accounts", _select_accounts)
    graph.add_node("search_trends", _search_trends)
    graph.add_node("select_products", _select_products)
    graph.add_node("reflect", _reflect)
    graph.add_node("refine", _refine)
    graph.add_node("calculate", _calculate)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "define_flows")
    graph.add_edge("define_flows", "select_accounts")
    graph.add_conditional_edges(
        "select_accounts",
        _should_search_trends,
        {"search_trends": "search_trends", "select_products": "select_products"},
    )
    graph.add_edge("search_trends", "select_products")
    graph.add_edge("select_products", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _should_refine,
        {"refine": "refine", "calculate": "calculate"},
    )
    graph.add_edge("refine", "calculate")
    graph.add_edge("calculate", END)

    return graph.compile()


_graph = _build_graph()


# ── Entry point ───────────────────────────────────────────────────────────────

async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    asset_list = [
        {"asset_id": str(a.asset_id), "asset_type": a.asset_type,
         "account_name": a.account_name, "balance": a.balance}
        for a in request.invest_assets
    ]

    initial_state: AssetPortfolioState = {
        "invest_amount": request.invest_amount,
        "interest": request.interest,
        "invest_interest": request.invest_interest,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "asset_list": asset_list,
        "asset_by_type": {},
        "top_invest_products": [],
        "gather_products": [],
        "invest_products_text": "",
        "gather_products_text": "",
        "flow_defs": [],
        "flow_accounts": [],
        "trend_context": "",
        "flow_products": [],
        "reflection": {},
        "investment_flows": [],
    }

    final_state: AssetPortfolioState = await _graph.ainvoke(initial_state)

    default_id = request.invest_assets[0].asset_id if request.invest_assets else UUID(int=0)

    return AssetPortfolioResponse(
        created_at=datetime.now(timezone.utc),
        investment_flows=[
            InvestmentPlan(
                title=f["title"],
                term=f["term"],
                summary=f["summary"],
                gathering_id=UUID(f["gathering_id"]) if f.get("gathering_id") else default_id,
                gathering_account=GatheringAccount(
                    name=f["gathering_account"].get("name", "자유적금"),
                    type=f["gathering_account"].get("type", "SAVING"),
                    institution=f["gathering_account"].get("institution", ""),
                    interest_rate=float(f["gathering_account"].get("interest_rate", 0.0)),
                ),
                amount=f["amount"],
                account_comment=f["account_comment"],
                portfolio=[
                    PortfolioItem(
                        type=p["type"],
                        name=p["name"],
                        ratio=p["ratio"],
                        interest_rate=p["interest_rate"],
                        comment=p["comment"],
                    )
                    for p in f["portfolio"]
                ],
                expected_rr_pct=f["expected_rr_pct"],
                investment_months=f["investment_months"],
                expected_amount=float(f["expected_amount"]),
                rr_comment=f["rr_comment"],
            )
            for f in final_state["investment_flows"]
        ],
    )
