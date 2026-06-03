from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.propose import ProposeRequest, ProposalResponse
from app.services.agent.llm import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = """당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.
사용자의 현재 대시보드 데이터와 요청을 분석해 재무 계획 변경을 제안합니다.

반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이 순수 JSON):
{
  "summary": "한 줄 요약",
  "explanation": "2-3문장 설명 (왜 이 제안을 했는지 포함)",
  "changes": {
    "events": [
      {
        "title": "목표명",
        "targetAmount": "목표금액(원, 숫자 문자열)",
        "deadline": "YYYY-MM-DD",
        "userInput": "목표 간단 설명"
      }
    ],
    "salaryAllocations": [
      {
        "purpose": "용도명",
        "plannedAmount": 금액정수
      }
    ],
    "portfolio": [
      {
        "assetType": "STOCK|BOND|CASH|IRP|SAVING",
        "assetAmount": 금액정수
      }
    ]
  }
}

규칙:
- 목표 추가 요청 → events에 추가, 필요한 월 저축액은 salaryAllocations에도 추가
- 투자 조정 요청 → portfolio 변경
- 잔여액(surplus)이 부족하면 다른 항목 줄이는 방안도 함께 제안
- 변경 없는 항목은 빈 배열로 유지
- 모든 금액은 원 단위 정수
- deadline은 현재 날짜 기준으로 합리적인 날짜 계산"""


async def propose(req: ProposeRequest) -> ProposalResponse:
    sp = req.dashboard_snapshot
    assets = sp.get("assetsSummary", {})
    salary = sp.get("salaryPlan", {})
    consumption = sp.get("consumption", {})
    portfolio = sp.get("portfolio", [])

    context = f"""사용자 요청: {req.user_message}

현재 재무 현황:
- 총 자산: {assets.get("totalBalance", 0):,}원
- 투자 자산: {assets.get("investmentBalance", 0):,}원
- 현금성 자산: {assets.get("cashBalance", 0):,}원
- 월 급여: {salary.get("monthlyIncome", 0):,}원
- 월 투자 금액: {salary.get("investmentAmount") or 0:,}원
- 월 잉여액: {salary.get("surplus") or 0:,}원
- 현재 배분 항목: {json.dumps(salary.get("allocations", []), ensure_ascii=False)}
- 이번달 지출: {consumption.get("totalExpense", 0):,}원
- 포트폴리오: {json.dumps(portfolio, ensure_ascii=False)}"""

    llm = get_llm(temperature=0.3)
    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=context),
    ])

    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[-2] if raw.count("```") >= 2 else raw
        raw = raw.lstrip("json").strip()

    data = json.loads(raw)
    return ProposalResponse(**data)
