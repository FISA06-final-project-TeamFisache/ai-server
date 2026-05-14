import json
import re
from typing import Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.core.config import settings
from app.schemas.agent import WeddingBudget, WeddingRequest, WeddingResponse


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_regional_venue_cost(region: str, month: int, scale: str) -> str:
    """지역, 결혼 월, 규모에 따른 예식장 비용을 조회합니다.

    Args:
        region: 결혼식 예정 지역 (예: 서울 강남, 부산, 경기 성남)
        month: 결혼 예정 월 (1-12)
        scale: 예식 규모 — small(소규모 100명 이하), medium(200명 내외), large(300명 이상)
    """
    base = {"small": 8_000_000, "medium": 18_000_000, "large": 35_000_000}

    region_mult = 1.0
    if any(k in region for k in ["강남", "서초", "송파", "용산"]):
        region_mult = 1.5
    elif "서울" in region:
        region_mult = 1.3
    elif any(k in region for k in ["경기", "인천", "분당", "판교", "수원"]):
        region_mult = 1.1
    elif any(k in region for k in ["부산", "대구", "광주", "대전", "울산"]):
        region_mult = 0.9

    # 5·6·10월 성수기, 1·2·7·8월 비수기
    season_mult = 1.2 if month in [5, 6, 10] else (0.85 if month in [1, 2, 7, 8] else 1.0)

    cost = int(base.get(scale, base["medium"]) * region_mult * season_mult)
    return (
        f"예식장 예상 비용: {cost:,}원\n"
        f"  - 기준 비용({scale}): {base.get(scale, base['medium']):,}원\n"
        f"  - 지역 가중치({region}): ×{region_mult}\n"
        f"  - 시즌 가중치({month}월): ×{season_mult}"
    )


@tool
def get_honeymoon_packages(scale: str) -> str:
    """신혼여행 패키지 예상 비용을 규모에 따라 조회합니다.

    Args:
        scale: 신혼여행 규모 — small(국내/동남아), medium(일본/하와이), large(유럽/몰디브 고급)
    """
    packages = {
        "small":  (2_000_000, 5_000_000,  "국내 제주 또는 동남아(태국·발리·베트남)"),
        "medium": (5_000_000, 12_000_000, "일본·하와이·동유럽 패키지"),
        "large":  (12_000_000, 30_000_000, "몰디브·유럽 허니문 고급 리조트"),
    }
    lo, hi, dest = packages.get(scale, packages["medium"])
    mid = (lo + hi) // 2
    return (
        f"신혼여행 예상 비용: {mid:,}원 (범위 {lo:,}~{hi:,}원)\n"
        f"  - 여행지 유형({scale}): {dest}"
    )


@tool
def get_sdrme_estimate(scale: str) -> str:
    """스드메(스튜디오·드레스·메이크업) 및 예물·예단 예상 비용을 조회합니다.

    Args:
        scale: 규모 — small(기본 패키지), medium(중급), large(프리미엄)
    """
    costs = {
        "small":  {"sdrme": 2_500_000, "gift": 2_000_000},
        "medium": {"sdrme": 6_000_000, "gift": 5_000_000},
        "large":  {"sdrme": 15_000_000, "gift": 15_000_000},
    }
    c = costs.get(scale, costs["medium"])
    total = c["sdrme"] + c["gift"]
    return (
        f"스드메 비용: {c['sdrme']:,}원\n"
        f"예물·예단 비용: {c['gift']:,}원\n"
        f"소계: {total:,}원"
    )


@tool
def calculate_total(venue: int, honeymoon: int, sdrme: int) -> str:
    """세 항목의 비용을 합산해 총 예산을 계산합니다. 반드시 이 도구로 합계를 구하세요.

    Args:
        venue: 예식장 비용 (원, 정수)
        honeymoon: 신혼여행 비용 (원, 정수)
        sdrme: 스드메 + 예물/예단 비용 (원, 정수)
    """
    total = venue + honeymoon + sdrme
    return (
        f"총 예산: {total:,}원\n"
        f"  예식장 {venue:,} + 신혼여행 {honeymoon:,} + 스드메 {sdrme:,}"
    )


_TOOLS = [get_regional_venue_cost, get_honeymoon_packages, get_sdrme_estimate, calculate_total]


# ── Graph state ───────────────────────────────────────────────────────────────

class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Structured output schema ─────────────────────────────────────────────────

class _BudgetOutput(BaseModel):
    venue: int = Field(description="예식장 비용 (원, 정수)")
    honeymoon: int = Field(description="신혼여행 비용 (원, 정수)")
    sdrme: int = Field(description="스드메 + 예물/예단 비용 (원, 정수)")
    total: int = Field(description="venue + honeymoon + sdrme 합계 (원, 정수)")
    reasoning: str = Field(description="지역·시기·규모를 반영한 예산 산출 근거 요약 (한국어, 2~4문장)")


# ── Agent graph ───────────────────────────────────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is not None:
        return _graph

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    ).bind_tools(_TOOLS, parallel_tool_calls=True)

    async def call_model(state: _State) -> dict:
        return {"messages": [await llm.ainvoke(state["messages"])]}

    builder = StateGraph(_State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    _graph = builder.compile()
    return _graph

_SYSTEM_PROMPT = SystemMessage(
    "당신은 한국 결혼 시장 전문가입니다. 반드시 아래 2단계로 도구를 호출하세요.\n\n"
    "[1단계] 아래 세 도구를 반드시 한 번의 응답에서 동시에(병렬로) 호출하세요:\n"
    "  - get_regional_venue_cost\n"
    "  - get_honeymoon_packages\n"
    "  - get_sdrme_estimate\n"
    "세 도구는 서로 독립적이므로 순차 호출하지 말고 반드시 동시 호출해야 합니다.\n\n"
    "[2단계] 1단계 결과를 받은 후 calculate_total을 호출해 합계를 계산하세요. 직접 더하지 말 것.\n\n"
    "모든 도구 호출이 끝나면 반드시 아래 JSON 형식으로만 답변하세요. 다른 설명 없이 JSON만 출력합니다.\n"
    "{\n"
    '  "venue": <예식장 비용 정수>,\n'
    '  "honeymoon": <신혼여행 비용 정수>,\n'
    '  "sdrme": <스드메+예물/예단 비용 정수>,\n'
    '  "total": <합계 정수>,\n'
    '  "reasoning": "<산출 근거 2~4문장>"\n'
    "}"
)


def _parse_budget(content: str) -> _BudgetOutput:
    # 마크다운 코드블록 제거 후 첫 번째 JSON 오브젝트 추출
    text = re.sub(r"```(?:json)?", "", content).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON을 찾을 수 없습니다: {content[:200]}")
    return _BudgetOutput.model_validate(json.loads(match.group()))


async def build_wedding(request: WeddingRequest) -> WeddingResponse:
    user_msg = (
        f"결혼 예산을 산출해 주세요.\n"
        f"- 지역: {request.wedding_region}\n"
        f"- 결혼 예정 월: {request.wedding_month}월\n"
        f"- 신혼여행 규모: {request.honeymoon_scale.value}\n"
        f"- 스드메 규모: {request.sdrme_scale.value}\n"
        f"- 목표 시기: {request.deadline}"
    )

    result = await _get_graph().ainvoke({"messages": [_SYSTEM_PROMPT, ("human", user_msg)]})

    last_content = result["messages"][-1].content
    parsed = _parse_budget(last_content)

    return WeddingResponse(
        budget=WeddingBudget(
            venue=parsed.venue,
            honeymoon=parsed.honeymoon,
            sdrme=parsed.sdrme,
            total=parsed.total,
        ),
        reasoning=parsed.reasoning,
    )
