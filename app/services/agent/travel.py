from app.schemas.agent import TravelBudget, TravelRequest, TravelResponse, TravelStyle

_DAILY_RATES = {
    TravelStyle.budget: {"accommodation": 80_000, "food": 40_000, "transportation": 20_000, "sightseeing": 30_000},
    TravelStyle.luxury: {"accommodation": 300_000, "food": 150_000, "transportation": 80_000, "sightseeing": 100_000},
}
_FLIGHT = {TravelStyle.budget: 600_000, TravelStyle.luxury: 2_000_000}


async def build_travel(request: TravelRequest) -> TravelResponse:
    # TODO: LangGraph 연동 (국가별 실시간 물가·항공권 시세 조회)
    rates = _DAILY_RATES[request.travel_style]
    days = request.travel_days
    accommodation = rates["accommodation"] * days
    food = rates["food"] * days
    transportation = rates["transportation"] * days
    sightseeing = rates["sightseeing"] * days
    flight = _FLIGHT[request.travel_style]
    return TravelResponse(
        budget=TravelBudget(
            accommodation=accommodation,
            flight=flight,
            food=food,
            transportation=transportation,
            sightseeing=sightseeing,
            total=accommodation + flight + food + transportation + sightseeing,
        ),
    )
