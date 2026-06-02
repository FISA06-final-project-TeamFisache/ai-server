import asyncio
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.portfolio import ProfileRequest, ProfileResponse
from app.services.agent.llm import get_llm


async def _generate_expense_comment(porti_type: str, porti_comment: str, expense_summary: str) -> str:
    llm = get_llm(temperature=0.5)
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
            f"PorTI 유형: {porti_type}\n"
            f"성향: {porti_comment}\n"
            f"월 평균 지출: {expense_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def _generate_invest_comment(porti_type: str, asset_summary: str) -> str:
    llm = get_llm(temperature=0.5)
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
            f"PorTI 유형: {porti_type}\n"
            f"실제 자산 구성: {asset_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def _generate_savings_comment(porti_type: str, savings_summary: str) -> str:
    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content=(
            "당신은 사용자의 저축 상황을 편하게 이야기해주는 금융 친구예요.\n"
            "비상금이나 저축 구성을 보고 솔직하게 한마디 건네주세요.\n\n"
            "- 현재 상태를 긍정적으로 인정하되, 빠진 게 있다면 가볍게 짚어주세요.\n"
            "- 비상금 기준(월 지출 3~6배)이 있으면 실제와 비교해 알려주세요.\n"
            "- 70자 내외로 짧게. 줄바꿈 없이 한 흐름으로.\n"
            "- 이모지·특수문자 금지."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {porti_type}\n"
            f"저축·예금 현황: {savings_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    expense_summary = ", ".join(
        f"{e.name} {e.expense:,}원" for e in request.category_expense
    ) or "거래 내역 없음"

    RISK_TYPES = {"STOCK", "IRP", "ISA"}
    total_balance = sum(a.balance for a in request.assets)
    risk_balance = sum(a.balance for a in request.assets if a.asset_type in RISK_TYPES)
    risk_ratio = round(risk_balance * 100 / total_balance) if total_balance > 0 else 0
    asset_summary = (
        f"위험자산(주식·IRP·ISA) {risk_ratio}% / "
        f"안전자산 {100 - risk_ratio}%, "
        f"총 자산 {total_balance:,}원"
    )

    SAVINGS_TYPES = {"SAVINGS", "DEPOSIT", "PARKING", "CHECKING", "CMA"}
    savings_assets = [a for a in request.assets if a.asset_type in SAVINGS_TYPES]
    savings_summary = ", ".join(
        f"{a.account_name}({a.asset_type}) {a.balance:,}원" for a in savings_assets
    ) if savings_assets else "저축 계좌 없음"

    expense_comment, invest_comment, savings_comment = await asyncio.gather(
        _generate_expense_comment(request.porti_type, request.porti_comment, expense_summary),
        _generate_invest_comment(request.porti_type, asset_summary),
        _generate_savings_comment(request.porti_type, savings_summary),
    )

    return ProfileResponse(
        created_at=datetime.now(timezone.utc),
        expense_comment=expense_comment,
        invest_comment=invest_comment,
        savings_comment=savings_comment,
    )
