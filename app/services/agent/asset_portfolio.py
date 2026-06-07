import logging
import math
import os
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    GatheringAccount,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import ainvoke_structured
from app.services.rag.db import get_pool

logger = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"

_INVEST_PRODUCT_TYPES = ["STOCK", "ETF", "BOND"]
_GATHER_PRODUCT_TYPES = ["CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT", "ISA", "IRP", "PENSION_SAVINGS"]

_WOORI_BANK = "우리은행"
_WOORI_INVEST = "우리투자증권"

from app.services.agent.porti_types import STABLE_PORTI_TYPES as _STABLE_PORTI_TYPES, porti_label as _porti_label


def _fmt_mktcap(v) -> str:
    if not v:
        return "-"
    v = int(v)
    if v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.1f}조"
    if v >= 100_000_000:
        return f"{v // 100_000_000}억"
    return f"{v:,}"

_FLOW_SPECS = [
    {"flow_type": "단기",  "term": "단기",  "investment_months": 6},
    {"flow_type": "중기",  "term": "중기",  "investment_months": 60},
    {"flow_type": "장기1", "term": "장기",  "investment_months": 240},
    {"flow_type": "장기2", "term": "장기",  "investment_months": 240},
]

# ── AI 출력 스키마 ─────────────────────────────────────────────────────────────

class _FlowItem(BaseModel):
    flow_type: str
    title: str
    summary: str
    ratio: int


class _FlowsAIOutput(BaseModel):
    flows: list[_FlowItem]


class _AccountCommentItem(BaseModel):
    flow_type: str
    comment: str


class _AccountCommentsOutput(BaseModel):
    comments: list[_AccountCommentItem]


class _AIPortfolioItem(BaseModel):
    name: str
    ticker: str
    ratio: int
    comment: str


class _FlowProductItem(BaseModel):
    flow_type: str
    portfolio: list[_AIPortfolioItem]


class _ProductsAIOutput(BaseModel):
    flow_products: list[_FlowProductItem]


class _ReflectionItem(BaseModel):
    flow_type: str
    is_aligned: bool
    feedback: str = ""


class _ReflectionOutput(BaseModel):
    reflections: list[_ReflectionItem]


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    asset_by_type: dict[str, list[dict]]
    etf_candidates: list[dict]       # DB에서 확정된 투자 상품 목록
    gather_products: list[dict]      # DB에서 조회된 모으기 상품 목록
    flow_defs: list[dict]
    flow_accounts: list[dict]
    flow_products: list[dict]
    investment_flows: list[dict]


# ── Tool functions (future @tool 전환 가능) ────────────────────────────────────

def _gather_rule(flow_type: str, porti_type: str) -> tuple[str, str]:
    """흐름 타입과 투자 성향으로 (계좌 타입, fallback 기관) 결정"""
    if flow_type == "단기":
        return "DEPOSIT", _WOORI_BANK
    elif flow_type == "중기":
        if porti_type in _STABLE_PORTI_TYPES:
            return "SAVING", _WOORI_BANK
        return "ISA", _WOORI_INVEST
    elif flow_type == "장기1":
        return "PENSION_SAVINGS", _WOORI_INVEST
    else:  # 장기2
        return "IRP", _WOORI_INVEST


def _can_invest(account_type: str) -> bool:
    return account_type in {"ISA", "PENSION_SAVINGS", "IRP"}


def _find_user_asset(asset_by_type: dict, account_type: str, used_ids: set) -> dict | None:
    for a in asset_by_type.get(account_type, []):
        if a["asset_id"] not in used_ids:
            return a
    return None


def _find_best_product(gather_products: list[dict], account_type: str, prefer_institution: str) -> dict | None:
    candidates = [p for p in gather_products if p["product_type"] == account_type]
    if not candidates:
        return None
    woori = [p for p in candidates if prefer_institution in (p.get("institution") or "")]
    return woori[0] if woori else candidates[0]


# ── 임베딩 ─────────────────────────────────────────────────────────────────────

async def _get_embedding(text: str) -> list[float] | None:
    if not text.strip():
        return None
    try:
        return await OpenAIEmbeddings(model=_EMBED_MODEL).aembed_query(text)
    except Exception as e:
        logger.warning("임베딩 생성 실패: %s", e)
        return None


# ── Node: preprocess ──────────────────────────────────────────────────────────

async def _preprocess(state: AssetPortfolioState) -> AssetPortfolioState:
    asset_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    query_parts = list(state["invest_interests"]) + ([state["interest"]] if state["interest"] else [])
    query_vector = await _get_embedding(" ".join(query_parts))

    pool = await get_pool()
    etf_candidates: list[dict] = []
    gather_products: list[dict] = []

    if pool:
        try:
            if query_vector:
                vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, ticker, interest_rate, description, "
                    "mktcap, avg_trading_value "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY embedding <=> $2::vector "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES, vec_str,
                )
            else:
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, ticker, interest_rate, description, "
                    "mktcap, avg_trading_value "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY interest_rate DESC NULLS LAST "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES,
                )
            etf_candidates = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("투자 상품 조회 실패: %s", e)

        try:
            rows = await pool.fetch(
                "SELECT product_type, institution, name, ticker, interest_rate, description "
                "FROM products "
                "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST",
                _GATHER_PRODUCT_TYPES,
            )
            gather_products = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("모으기 상품 조회 실패: %s", e)

    return {
        **state,
        "asset_by_type": asset_by_type,
        "etf_candidates": etf_candidates,
        "gather_products": gather_products,
    }


# ── Node: define_flows ────────────────────────────────────────────────────────

async def _define_flows(state: AssetPortfolioState) -> AssetPortfolioState:
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
            f"PorTI 유형: {_porti_label(state['porti_type'])}\n"
            f"투자 성향 설명: {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interests']) or '없음'}\n"
            f"월 투자금: {state['invest_amount']:,}원"
        )),
    ]
    ai_result = await ainvoke_structured(messages, _FlowsAIOutput)

    if ai_result:
        llm_map = {f.flow_type: {"title": f.title, "summary": f.summary, "ratio": f.ratio} for f in ai_result.flows}
    else:
        llm_map = {}

    raw_ratios = [int(float(llm_map.get(s["flow_type"], {}).get("ratio") or 25)) for s in _FLOW_SPECS]
    if sum(raw_ratios) != 100:
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

async def _select_accounts(state: AssetPortfolioState) -> AssetPortfolioState:
    asset_by_type = state["asset_by_type"]
    gather_products = state["gather_products"]
    porti_type = state["porti_type"]
    used_ids: set = set()
    flow_accounts = []

    for spec in _FLOW_SPECS:
        ft = spec["flow_type"]
        account_type, fallback_institution = _gather_rule(ft, porti_type)

        user_asset = _find_user_asset(asset_by_type, account_type, used_ids)
        if user_asset:
            used_ids.add(user_asset["asset_id"])

        best_product = _find_best_product(gather_products, account_type, fallback_institution)

        ga: dict = {
            "name": (
                best_product["name"] if best_product
                else (user_asset["account_name"] if user_asset else account_type)
            ),
            "type": account_type,
            "institution": (
                best_product["institution"] if best_product
                else (fallback_institution if not user_asset else "")
            ),
            "interest_rate": float(best_product["interest_rate"] or 0.0) if best_product else 0.0,
        }

        flow_accounts.append({
            "flow_type": ft,
            "gathering_asset_id": user_asset["asset_id"] if user_asset else None,
            "has_user_account": user_asset is not None,
            "account_type": account_type,
            "fallback_institution": fallback_institution,
            "gathering_account": ga,
            "can_invest": _can_invest(account_type),
            "account_comment": "",
        })

    # 계좌 추천 이유 — 1회 배치 LLM 호출
    flow_accounts = await _generate_account_comments(flow_accounts, state)

    return {**state, "flow_accounts": flow_accounts}


async def _generate_account_comments(flow_accounts: list[dict], state: dict) -> list[dict]:
    flows_desc = "\n".join(
        f'- {fa["flow_type"]}: {fa["account_type"]} | {fa["gathering_account"]["name"]} '
        f'({fa["gathering_account"]["institution"]}) | 기존 계좌 {"있음" if fa["has_user_account"] else "없음 → 신규 개설 필요"}'
        for fa in flow_accounts
    )
    messages = [
        SystemMessage(content=(
            "개인 자산관리 전문가입니다. 각 투자 흐름의 모으기 계좌 선택 이유를 1~2문장으로 설명하세요.\n"
            "- 기존 계좌 없음: 왜 이 계좌를 새로 개설해야 하는지\n"
            "- 기존 계좌 있음: 왜 이 계좌를 이 흐름에 활용하는지\n"
            "단정적 수익보장 표현 금지. 각 comment는 50자 이내.\n\n"
            '{"comments":[{"flow_type":"단기","comment":"..."}]}'
        )),
        HumanMessage(content=(
            f"PorTI: {_porti_label(state['porti_type'])} / {state['porti_comment']}\n\n"
            f"[흐름별 계좌 현황]\n{flows_desc}"
        )),
    ]
    ai_result = await ainvoke_structured(messages, _AccountCommentsOutput)

    if ai_result:
        comment_map = {c.flow_type: c.comment for c in ai_result.comments}
        for fa in flow_accounts:
            fa["account_comment"] = comment_map.get(fa["flow_type"], "")

    return flow_accounts


# ── Node: select_products ─────────────────────────────────────────────────────

async def _select_products(state: AssetPortfolioState) -> AssetPortfolioState:
    can_invest_map = {fa["flow_type"]: fa["can_invest"] for fa in state["flow_accounts"]}
    account_type_map = {fa["flow_type"]: fa["account_type"] for fa in state["flow_accounts"]}
    invest_flows = [fd for fd in state["flow_defs"] if can_invest_map.get(fd["flow_type"])]

    if not invest_flows:
        flow_products = [{"flow_type": spec["flow_type"], "portfolio": []} for spec in _FLOW_SPECS]
        return {**state, "flow_products": flow_products}

    # 확정된 상품 목록 (truth source)
    confirmed_by_name: dict[str, dict] = {p["name"]: p for p in state["etf_candidates"]}

    candidates_text = "\n".join(
        f"- [{p['product_type']}] {p['institution']} | {p['name']} "
        f"| ticker:{p.get('ticker') or ''} | 연 {p['interest_rate'] or '-'}% "
        f"| 시가총액:{_fmt_mktcap(p.get('mktcap'))} | 일평균거래대금:{_fmt_mktcap(p.get('avg_trading_value'))} "
        f"| {(p['description'] or '')[:80]}"
        for p in state["etf_candidates"]
    ) or "상품 없음"

    flows_desc = "\n".join(
        f'- {fd["flow_type"]} ({fd["term"]}, {fd["investment_months"]}개월'
        f', 계좌:{account_type_map.get(fd["flow_type"], "")}): {fd["summary"]}'
        for fd in invest_flows
    )
    target_keys = ", ".join(f'"{fd["flow_type"]}"' for fd in invest_flows)

    messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "아래 [선택 가능 상품 목록]에서만 골라 각 투자 흐름의 포트폴리오를 구성하세요.\n\n"
            "규칙:\n"
            "- name: 목록의 정확한 상품명 그대로 사용 (변형·새 이름 생성 절대 금지)\n"
            "- ticker: 목록에 표시된 ticker 그대로 사용\n"
            "- ratio: 각 흐름 합계 = 100\n"
            "- comment: 이 상품 선택 이유 1문장 (수익보장·단정 표현 금지)\n"
            "- 공격적 성향: 주식형 ETF 비중 높게 / 안정 성향: 채권·배당 ETF 위주\n"
            "- ISA: 국내 ETF 우선 / IRP·연금저축: 장기 분산 포트폴리오\n"
            f"- 반드시 아래 [{target_keys}] 모든 flow_type에 대해 portfolio를 빠짐없이 작성하세요.\n\n"
            f"[투자 가능 흐름]\n{flows_desc}\n\n"
            f"[선택 가능 상품 목록]\n{candidates_text}\n\n"
            f"JSON만 응답. 포함할 flow_type: [{target_keys}]\n"
            '{"flow_products":[{"flow_type":"장기1","portfolio":'
            '[{"name":"정확한상품명","ticker":"종목코드","ratio":60,"comment":"이유"}]}]}'
        )),
        HumanMessage(content=(
            f"PorTI: {_porti_label(state['porti_type'])} / {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interests']) or '없음'}"
        )),
    ]
    ai_result = await ainvoke_structured(messages, _ProductsAIOutput)

    llm_map: dict[str, list[_AIPortfolioItem]] = {}
    if ai_result:
        llm_map = {fp.flow_type: fp.portfolio for fp in ai_result.flow_products}

    flow_products = []
    for spec in _FLOW_SPECS:
        ft = spec["flow_type"]
        if not can_invest_map.get(ft):
            flow_products.append({"flow_type": ft, "portfolio": []})
            continue

        validated: list[dict] = []
        for item in llm_map.get(ft, []):
            name = item.name
            # cross-validation: 목록에 없는 상품명 제거 (fuzzy fallback)
            if name not in confirmed_by_name:
                matched = next(
                    (n for n in confirmed_by_name if name in n or n in name), None
                )
                if not matched:
                    logger.warning("LLM 생성 상품명 — 목록 미존재, 제거: %s", name)
                    continue
                name = matched

            p = confirmed_by_name[name]
            validated.append({
                "name": name,
                "ticker": p.get("ticker") or item.ticker or "",
                "ratio": item.ratio,
                "comment": item.comment,
            })

        # LLM이 이 flow_type을 응답에서 누락했거나 cross-validation 후 비어있으면 상위 상품으로 채움
        if not validated and state["etf_candidates"]:
            logger.warning("LLM 포트폴리오 응답 없음(flow=%s) — etf_candidates 상위 항목으로 대체", ft)
            top = state["etf_candidates"][:3]
            base = 100 // len(top)
            rem = 100 - base * len(top)
            validated = [
                {
                    "name": p["name"],
                    "ticker": p.get("ticker") or "",
                    "ratio": base + (rem if i == 0 else 0),
                    "comment": "분산 투자를 위한 기본 추천 상품입니다.",
                }
                for i, p in enumerate(top)
            ]

        # ratio 정규화
        total = sum(v["ratio"] for v in validated)
        if 0 < total != 100:
            for v in validated:
                v["ratio"] = round(v["ratio"] * 100 / total)
            diff = 100 - sum(v["ratio"] for v in validated)
            if validated and diff:
                validated[0]["ratio"] += diff

        flow_products.append({"flow_type": ft, "portfolio": validated})

    return {**state, "flow_products": flow_products}


# ── Node: reflect_products ────────────────────────────────────────────────────

def _validate_and_merge(
    raw_items: list[_AIPortfolioItem],
    confirmed_by_name: dict[str, dict],
    etf_candidates: list[dict],
) -> list[dict]:
    """cross-validation + ratio 정규화 + 빈 경우 fallback — select_products와 동일 로직"""
    validated: list[dict] = []
    for item in raw_items:
        name = item.name
        if name not in confirmed_by_name:
            matched = next((n for n in confirmed_by_name if name in n or n in name), None)
            if not matched:
                logger.warning("reflect refine — 목록 미존재 상품명 제거: %s", name)
                continue
            name = matched
        p = confirmed_by_name[name]
        validated.append({
            "name": name,
            "ticker": p.get("ticker") or item.ticker or "",
            "ratio": item.ratio,
            "comment": item.comment,
        })

    if not validated and etf_candidates:
        top = etf_candidates[:3]
        base = 100 // len(top)
        rem = 100 - base * len(top)
        validated = [
            {
                "name": p["name"],
                "ticker": p.get("ticker") or "",
                "ratio": base + (rem if i == 0 else 0),
                "comment": "분산 투자를 위한 기본 추천 상품입니다.",
            }
            for i, p in enumerate(top)
        ]

    total = sum(v["ratio"] for v in validated)
    if 0 < total != 100:
        for v in validated:
            v["ratio"] = round(v["ratio"] * 100 / total)
        diff = 100 - sum(v["ratio"] for v in validated)
        if validated and diff:
            validated[0]["ratio"] += diff

    return validated


async def _reflect_products(state: AssetPortfolioState) -> AssetPortfolioState:
    invest_flow_types = {
        fa["flow_type"] for fa in state["flow_accounts"] if fa["can_invest"]
    }
    invest_flows_products = [
        fp for fp in state["flow_products"] if fp["flow_type"] in invest_flow_types and fp["portfolio"]
    ]

    if not invest_flows_products:
        return state

    portfolio_desc = "\n".join(
        f'- {fp["flow_type"]}: '
        + ", ".join(
            f'{p["name"]}({p["ratio"]}%)'
            for p in fp["portfolio"]
        )
        for fp in invest_flows_products
    )

    eval_messages = [
        SystemMessage(content=(
            "포트폴리오 검토 전문가입니다.\n"
            "사용자 투자 성향과 각 흐름의 포트폴리오 구성이 적절한지 평가하세요.\n\n"
            "평가 기준:\n"
            "- 안전형(SWIMMING·ARCHERY): 채권·배당 ETF 위주, 주식 비중 30% 이하\n"
            "- 중립형(JUDO·RHYTHMIC): 주식·채권 균형, 주식 30~60%\n"
            "- 투자형(FENCING·CYCLING): 주식형 ETF 위주, 주식 60% 이상\n"
            "- IRP·연금저축(장기1·2): 장기 분산 구성인지\n\n"
            "is_aligned=false 시 feedback에 구체적 개선 방향 기술.\n"
            '{"reflections":[{"flow_type":"장기1","is_aligned":true,"feedback":""}]}'
        )),
        HumanMessage(content=(
            f"PorTI: {_porti_label(state['porti_type'])} / {state['porti_comment']}\n\n"
            f"[현재 포트폴리오]\n{portfolio_desc}"
        )),
    ]
    eval_result = await ainvoke_structured(eval_messages, _ReflectionOutput)

    if not eval_result:
        return state

    misaligned = [r for r in eval_result.reflections if not r.is_aligned]
    if not misaligned:
        logger.info("reflect_products: 모든 흐름 성향 일치 — refine 생략")
        return state

    logger.info("reflect_products: 불일치 흐름 %s — refine 시작", [r.flow_type for r in misaligned])

    confirmed_by_name: dict[str, dict] = {p["name"]: p for p in state["etf_candidates"]}
    account_type_map = {fa["flow_type"]: fa["account_type"] for fa in state["flow_accounts"]}

    misaligned_map = {r.flow_type: r.feedback for r in misaligned}
    refine_flows_desc = "\n".join(
        f'- {ft} (계좌:{account_type_map.get(ft,"")}) 개선 필요: {fb}'
        for ft, fb in misaligned_map.items()
    )
    candidates_text = "\n".join(
        f"- [{p['product_type']}] {p['institution']} | {p['name']} "
        f"| ticker:{p.get('ticker') or ''} | 연 {p['interest_rate'] or '-'}% "
        f"| 시가총액:{_fmt_mktcap(p.get('mktcap'))} | 일평균거래대금:{_fmt_mktcap(p.get('avg_trading_value'))} "
        f"| {(p['description'] or '')[:80]}"
        for p in state["etf_candidates"]
    )
    target_keys = ", ".join(f'"{ft}"' for ft in misaligned_map)

    refine_messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "아래 [선택 가능 상품 목록]에서만 골라 개선된 포트폴리오를 구성하세요.\n"
            "name은 목록의 정확한 상품명 그대로, ratio 합계=100.\n\n"
            f"[개선 대상 흐름 및 피드백]\n{refine_flows_desc}\n\n"
            f"[선택 가능 상품 목록]\n{candidates_text}\n\n"
            f"JSON만 응답. 포함할 flow_type: [{target_keys}]\n"
            '{"flow_products":[{"flow_type":"장기1","portfolio":'
            '[{"name":"정확한상품명","ticker":"코드","ratio":60,"comment":"이유"}]}]}'
        )),
        HumanMessage(content=(
            f"PorTI: {_porti_label(state['porti_type'])} / {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interests']) or '없음'}"
        )),
    ]
    refine_result = await ainvoke_structured(refine_messages, _ProductsAIOutput)

    if not refine_result:
        return state

    refine_map = {fp.flow_type: fp.portfolio for fp in refine_result.flow_products}

    updated_products = []
    for fp in state["flow_products"]:
        ft = fp["flow_type"]
        if ft in refine_map:
            new_portfolio = _validate_and_merge(
                refine_map[ft], confirmed_by_name, state["etf_candidates"]
            )
            updated_products.append({"flow_type": ft, "portfolio": new_portfolio})
        else:
            updated_products.append(fp)

    return {**state, "flow_products": updated_products}


# ── Node: calculate ───────────────────────────────────────────────────────────

def _calculate(state: AssetPortfolioState) -> AssetPortfolioState:
    invest_amount = state["invest_amount"]

    product_by_name: dict[str, dict] = {}
    for p in state["etf_candidates"] + state["gather_products"]:
        product_by_name.setdefault(p["name"], p)

    flow_defs_map = {fd["flow_type"]: fd for fd in state["flow_defs"]}
    flow_accounts_map = {fa["flow_type"]: fa for fa in state["flow_accounts"]}
    flow_products_map = {fp["flow_type"]: fp["portfolio"] for fp in state["flow_products"]}

    # ratio 합계 보정 (LLM fallback 후 안전망)
    total_ratio = sum(max(1, int(f.get("ratio", 0))) for f in state["flow_defs"][:4])
    if total_ratio != 100:
        per = 100 // 4
        for i, f in enumerate(state["flow_defs"][:4]):
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
        _ga_rate_raw = float(ga.get("interest_rate", 0.0) or 0.0)
        # interest_rate 미입력(0.0)이면 기준금리 2.5% 적용
        ga_rate = _ga_rate_raw if _ga_rate_raw > 0.0 else 2.5

        if portfolio_raw:
            weighted = sum(
                (float(product_by_name.get(item.get("name", ""), {}).get("interest_rate") or 0.0) or 2.5)
                * item.get("ratio", 0) / 100
                for item in portfolio_raw
            )
            expected_rr = weighted if weighted > 0 else 2.5
        else:
            expected_rr = ga_rate

        r_m = expected_rr / 100 / 12
        expected_amount = (
            amount * ((math.pow(1 + r_m, months) - 1) / r_m)
            if r_m > 0 else float(amount * months)
        )

        portfolio_items: list[dict] = []
        for item in portfolio_raw:
            p = product_by_name.get(item.get("name", ""), {})
            portfolio_items.append({
                "type": p.get("product_type", "ETF"),
                "name": item.get("name", ""),
                "ticker": item.get("ticker", ""),
                "ratio": item.get("ratio", 0),
                "interest_rate": float(p.get("interest_rate") or 0.0),
                "comment": item.get("comment", ""),
            })

        rr_comment = (
            f"연 {expected_rr:.1f}% 기준 {months}개월 적립식 복리 시 약 {round(expected_amount):,}원 예상."
        )

        investment_flows.append({
            "flow_type": ft,
            "title": fd.get("title", f"{ft} 투자 플랜"),
            "term": spec["term"],
            "summary": fd.get("summary", ""),
            "gathering_id": fa.get("gathering_asset_id"),
            "gathering_account": ga,
            "amount": amount,
            "account_comment": fa.get("account_comment", ""),
            "portfolio": portfolio_items,
            "expected_rr_pct": round(expected_rr, 1),
            "investment_months": months,
            "expected_amount": round(expected_amount),
            "rr_comment": rr_comment,
        })

    return {**state, "investment_flows": investment_flows}


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("preprocess", _preprocess)
    graph.add_node("define_flows", _define_flows)
    graph.add_node("select_accounts", _select_accounts)
    graph.add_node("select_products", _select_products)
    graph.add_node("reflect_products", _reflect_products)
    graph.add_node("calculate", _calculate)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "define_flows")
    graph.add_edge("define_flows", "select_accounts")
    graph.add_edge("select_accounts", "select_products")
    graph.add_edge("select_products", "reflect_products")
    graph.add_edge("reflect_products", "calculate")
    graph.add_edge("calculate", END)

    return graph.compile()


_graph = _build_graph()


# ── Entry point ───────────────────────────────────────────────────────────────

async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    asset_list = [
        {
            "asset_id": str(a.asset_id),
            "asset_type": a.asset_type,
            "account_name": a.account_name,
            "balance": a.balance,
        }
        for a in request.invest_assets
    ]

    initial_state: AssetPortfolioState = {
        "invest_amount": request.invest_amount,
        "interest": request.interest,
        "invest_interests": request.invest_interests,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "asset_list": asset_list,
        "asset_by_type": {},
        "etf_candidates": [],
        "gather_products": [],
        "flow_defs": [],
        "flow_accounts": [],
        "flow_products": [],
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
                # 사용자 계좌 있음: gathering_id만, gathering_account=null
                # 사용자 계좌 없음: gathering_account만(신규 개설 추천), gathering_id=null
                gathering_id=UUID(f["gathering_id"]) if f.get("gathering_id") else None,
                gathering_account=(
                    None if f.get("gathering_id") else GatheringAccount(
                        name=f["gathering_account"].get("name", "자유적금"),
                        type=f["gathering_account"].get("type", "SAVING"),
                        institution=f["gathering_account"].get("institution", ""),
                        interest_rate=float(f["gathering_account"].get("interest_rate", 0.0)),
                    )
                ),
                amount=f["amount"],
                account_comment=f["account_comment"],
                portfolio=[
                    PortfolioItem(
                        type=p["type"],
                        name=p["name"],
                        ticker=p["ticker"],
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
