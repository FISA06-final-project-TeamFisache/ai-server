from app.schemas.agent import RebalanceAllocation, RebalanceRequest, RebalanceResponse


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    # TODO: LangGraph 연동
    recommendations = [
        RebalanceAllocation(
            category=item.category,
            amount=item.amount,
            change=0.0,
            reason="[STUB] 현재 배분 유지",
        )
        for item in request.current_allocations
    ]
    return RebalanceResponse(
        user_id=request.user_id,
        recommended_allocations=recommendations,
        summary="[STUB] 급여 배분 리밸런싱 결과입니다.",
    )
