import asyncio
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.portfolio import ProfileRequest, ProfileResponse
from app.services.agent.llm import get_llm
from app.services.agent.porti_types import porti_label


async def _generate_expense_comment(porti_type: str, porti_comment: str, expense_summary: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 사용자의 소비 패턴을 함께 들여다보는 금융 친구예요.\n"
            "데이터를 보고 자연스럽게 한두 마디 건네는 느낌으로 써주세요.\n\n"
            "- 가장 눈에 띄는 지출 하나를 콕 집어 말해주세요.\n"
            "- 뻔한 조언('절약하세요', '관리하세요') 대신 구체적인 수치나 행동을 담아주세요.\n"
            "- 120자 내외로 짧게. 줄바꿈 없이 한 흐름으로.\n"
            "- 이모지·특수문자 금지."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {porti_label(porti_type)}\n"
            f"성향 설명: {porti_comment}\n"
            f"월 평균 지출: {expense_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def _generate_invest_comment(porti_type: str, asset_summary: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 사용자의 투자 현황을 솔직하게 짚어주는 금융 친구예요.\n"
            "성향과 실제 자산이 얼마나 맞는지 자연스럽게 말해주세요.\n\n"
            "- 성향과 포트폴리오가 맞으면 칭찬, 다르면 어떻게 다른지 가볍게 알려주세요.\n"
            "- 수치를 활용하되, 보고서 말투가 아닌 대화 말투로.\n"
            "- 70자 내외로 짧게. 줄바꿈 없이 한 흐름으로.\n"
            "- 이모지·특수문자 금지."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {porti_label(porti_type)}\n"
            f"실제 자산 구성: {asset_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    expense_summary = ", ".join(
        f"{e.name} {e.expense:,}원" for e in request.category_expense
    ) or "거래 내역 없음"

    total = request.assets_safe + request.assets_moderate + request.assets_risky
    risk_ratio     = round(request.assets_risky    * 100 / total) if total > 0 else 0
    moderate_ratio = round(request.assets_moderate * 100 / total) if total > 0 else 0
    safe_ratio     = 100 - risk_ratio - moderate_ratio
    asset_summary = (
        f"위험자산(주식·IRP·ISA) {risk_ratio}% / 중립자산 {moderate_ratio}% / 안전자산 {safe_ratio}%, "
        f"총 자산 {total:,}원 "
        f"(안전 {request.assets_safe:,}원 / 중립 {request.assets_moderate:,}원 / 위험 {request.assets_risky:,}원)"
    )

    # 2개 LLM 호출 병렬 실행
    expense_comment, invest_comment = await asyncio.gather(
        _generate_expense_comment(request.porti_type, request.porti_comment, expense_summary),
        _generate_invest_comment(request.porti_type, asset_summary),
    )

    return ProfileResponse(
        created_at=datetime.now(timezone.utc),
        expense_comment=expense_comment,
        invest_comment=invest_comment,
    )
