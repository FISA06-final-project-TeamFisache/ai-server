import asyncio
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.portfolio import ProfileRequest, ProfileResponse
from app.services.agent.llm import get_llm


async def _generate_expense_comment(porti_type: str, porti_comment: str, expense_summary: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 친절한 금융 어드바이저입니다. "
            "사용자의 소비 내역과 투자 성향을 바탕으로 지출 패턴을 2~3문장으로 분석하고 "
            "개선 포인트를 제안하세요. 친근하고 구체적인 말투로 작성하세요.\n"
            "금지: 이모지, 이모티콘, 특수문자 장식(★ ♦ 등) 사용 금지. 반드시 텍스트만 사용하세요."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {porti_type}\n"
            f"성향 설명: {porti_comment}\n"
            f"월 평균 카테고리별 지출: {expense_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def _generate_invest_comment(porti_type: str, asset_summary: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "당신은 투자 성향 분석 전문가입니다. "
            "PorTI 설문 결과와 실제 계좌 구성을 비교해 "
            "투자 성향을 2~3문장으로 분석하세요. 친근하고 따뜻한 말투로 작성하세요.\n"
            "금지: 이모지, 이모티콘, 특수문자 장식(★ ♦ 등) 사용 금지. 반드시 텍스트만 사용하세요."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {porti_type}\n"
            f"실제 자산 구성: {asset_summary}"
        )),
    ]
    result = await llm.ainvoke(messages)
    return result.content.strip()


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    # 지출 요약
    expense_summary = ", ".join(
        f"{e.name} {e.expense:,}원" for e in request.category_expense
    ) or "거래 내역 없음"

    # 자산 구성 분석 (assets_safe / assets_risky 사용)
    total = request.assets_safe + request.assets_risky
    risk_ratio = round(request.assets_risky * 100 / total) if total > 0 else 0
    asset_summary = (
        f"위험자산(주식·IRP·ISA) {risk_ratio}% / "
        f"안전자산 {100 - risk_ratio}%, "
        f"총 자산 {total:,}원 (안전 {request.assets_safe:,}원 / 위험 {request.assets_risky:,}원)"
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
