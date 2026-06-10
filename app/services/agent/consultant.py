from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.consultant import (
    FlowProductItem,
    FlowUpdate,
    ResetAllocation,
    ResetAnalyzeRequest,
    ResetAnalyzeResponse,
    ResetPortfolioItem,
    ResetProposeRequest,
    ResetProposeResponse,
)
from app.services.agent.llm import ainvoke_structured

logger = logging.getLogger(__name__)

_ANALYZE_SYSTEM = (
    "당신은 개인 재무 관리 AI Pori입니다.\n"
    "사용자 목표를 분석해 월급 배분 재설정(salary)과 투자 포트폴리오 재설정(portfolio) 중 더 적합한 것을 추천합니다.\n\n"
    '반드시 아래 JSON 형식으로만 응답하세요:\n{"action": "salary" 또는 "portfolio", "reasoning": "추천 이유 1문장"}\n\n'
    "기준:\n"
    "- salary: 저축 목표 달성, 지출 구조 조정, 특정 목적 자금 마련에 효과적\n"
    "- portfolio: 투자 수익 개선, 리스크 조정, 자산 배분 최적화에 효과적"
)

_SALARY_SYSTEM = (
    "당신은 개인 재무 관리 AI Pori입니다.\n"
    "사용자의 목표와 현재 재무 상황을 바탕으로 월급 배분 비율을 재설정해 주세요.\n\n"
    "반드시 아래 JSON 형식으로만 응답하세요:\n"
    '{"summary":"변경 요약 1문장","explanation":"배분 기준과 기대 효과 2-3문장",'
    '"salary_allocations":[{"purpose":"생활비","plannedAmount":1500000,"ratio":50}],"portfolio":[]}\n\n'
    "규칙:\n"
    "- 모든 ratio 합계 = 100\n"
    "- plannedAmount = 월 소득 x (ratio / 100), 원 단위 반올림\n"
    "- 생활비·저축·투자·목표 적금 등 현실적 항목으로 구성 (3~5개)"
)

_PORTFOLIO_SYSTEM = (
    "당신은 개인 재무 관리 AI Pori입니다.\n"
    "사용자의 목표와 현재 투자 흐름을 바탕으로 각 흐름 내 상품의 투자 비율을 재조정해 주세요.\n\n"
    "반드시 아래 JSON 형식으로만 응답하세요:\n"
    '{"summary":"각 투자 상품의 비율을 어떻게 조정했는지 1문장으로 (월급·배분·급여 언급 금지)",'
    '"explanation":"상품별 비율 조정 근거와 기대 효과 2-3문장 (투자 상품 관련 내용만)",'
    '"salary_allocations":[],"portfolio":[],'
    '"flows":[{"flowId":"현재 투자 흐름의 flowId 그대로","products":[{"productId":"상품의 productId 그대로","productRatio":60}]}]}\n\n'
    "규칙:\n"
    "- summary와 explanation은 반드시 투자 상품 비율 재조정에 관한 내용만 작성 (월급·급여·배분 언급 금지)\n"
    "- 각 흐름(flow) 내 모든 productRatio 합계 = 100\n"
    "- flowId, productId 는 반드시 '현재 투자 흐름' 에 표시된 값을 그대로 사용 (변경·추가 금지)\n"
    "- 보유하지 않은 flowId 또는 productId는 절대 사용하지 마세요\n"
    "- 흐름이 없으면 flows = []"
)


def _fmt_dashboard(snap: dict[str, Any]) -> str:
    income = snap.get("salaryPlan", {}).get("monthlyIncome", 0)
    allocs = snap.get("salaryPlan", {}).get("allocations", [])
    flows = snap.get("flows", [])

    alloc_lines = "\n".join(
        f"  - {a.get('purpose','기타')}: {a.get('plannedAmount',0):,}원"
        for a in allocs
    ) or "  (없음)"

    flow_lines_parts = []
    for f in flows:
        header = f"  [{f.get('title','?')}] flowId={f.get('flowId','?')}"
        product_lines = "\n".join(
            f"    - {p.get('productName','?')} ({p.get('productType','?')}) "
            f"productId={p.get('productId','?')} 현재비율={p.get('productRatio',0)}%"
            for p in f.get("products", [])
        ) or "    (상품 없음)"
        flow_lines_parts.append(f"{header}\n{product_lines}")
    flow_lines = "\n".join(flow_lines_parts) or "  (없음)"

    return (
        f"월 소득: {income:,}원\n"
        f"현재 월급 배분:\n{alloc_lines}\n"
        f"현재 투자 흐름:\n{flow_lines}\n"
        f"이번 달 총 지출: {snap.get('totalExpense', 0):,}원"
    )


class _AnalyzeAI(ResetAnalyzeResponse):
    pass


class _ProposeAI(ResetProposeResponse):
    pass


async def analyze_goal(req: ResetAnalyzeRequest) -> ResetAnalyzeResponse:
    context = f"사용자 목표: {req.user_goal}\n\n{_fmt_dashboard(req.dashboard_snapshot)}"
    result = await ainvoke_structured(
        [SystemMessage(content=_ANALYZE_SYSTEM), HumanMessage(content=context)],
        _AnalyzeAI,
        max_tokens=256,
    )
    if result is None:
        raise ValueError("분석 응답 파싱 실패")
    return ResetAnalyzeResponse(action=result.action, reasoning=result.reasoning)


async def propose_reset(req: ResetProposeRequest) -> ResetProposeResponse:
    system = _SALARY_SYSTEM if req.action == "salary" else _PORTFOLIO_SYSTEM
    income = req.dashboard_snapshot.get("salaryPlan", {}).get("monthlyIncome", 0)
    context = (
        f"사용자 목표: {req.user_goal}\n\n"
        f"{_fmt_dashboard(req.dashboard_snapshot)}\n\n"
        f"월 소득({income:,}원) 기준으로 {'월급 배분' if req.action == 'salary' else '포트폴리오'}을 재설정해 주세요."
    )
    result = await ainvoke_structured(
        [SystemMessage(content=system), HumanMessage(content=context)],
        _ProposeAI,
        max_tokens=512,
    )
    if result is None:
        raise ValueError("제안 응답 파싱 실패")

    response = ResetProposeResponse(
        summary=result.summary,
        explanation=result.explanation,
        salary_allocations=[ResetAllocation(**a.model_dump()) for a in result.salary_allocations],
        portfolio=[ResetPortfolioItem(**p.model_dump()) for p in result.portfolio],
        flows=[
            FlowUpdate(
                flowId=f.flowId,
                amount=f.amount,
                products=[FlowProductItem(**p.model_dump()) for p in f.products],
            )
            for f in result.flows
        ],
    )

    return response
