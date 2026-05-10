from app.schemas.agent import WeddingBudget, WeddingRequest, WeddingResponse, WeddingScale

_VENUE = {WeddingScale.small: 5_000_000, WeddingScale.medium: 15_000_000, WeddingScale.large: 30_000_000}
_HONEYMOON = {WeddingScale.small: 3_000_000, WeddingScale.medium: 8_000_000, WeddingScale.large: 20_000_000}
_SDRME = {WeddingScale.small: 2_000_000, WeddingScale.medium: 5_000_000, WeddingScale.large: 10_000_000}


async def build_wedding(request: WeddingRequest) -> WeddingResponse:
    # TODO: LangGraph 연동 (지역·시기 반영한 실제 시세 조회)
    venue = _VENUE[request.honeymoon_scale]
    honeymoon = _HONEYMOON[request.honeymoon_scale]
    sdrme = _SDRME[request.sdrme_scale]
    return WeddingResponse(
        user_id=request.user_id,
        budget=WeddingBudget(
            venue=venue,
            honeymoon=honeymoon,
            sdrme=sdrme,
            total=venue + honeymoon + sdrme,
        ),
    )
