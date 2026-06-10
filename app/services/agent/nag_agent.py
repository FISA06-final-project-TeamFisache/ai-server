from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.mini_challenge import NagRequest, NagResponse
from app.services.agent.llm import ainvoke_structured


class _NagAIOutput(BaseModel):
    nag_message: str


_NAG_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자의 챌린지 달성률에 맞는 친근한 잔소리성 독려 메시지를 1~2문장으로 생성하세요.\n\n"
    "달성률별 톤:\n"
    "- 50%: 반쯤 왔다는 안도감 + 아직 갈 길 있다는 가벼운 긴장감\n"
    "- 80%: 거의 다 왔다는 흥분 + 마지막 집중력 촉구\n"
    "- 90%: 눈앞에 있다는 강한 격려 + 포기 금지 강조\n"
    "이모지 1~2개 포함, 챌린지 제목과 카테고리를 자연스럽게 언급"
)


async def generate_nag(req: NagRequest) -> NagResponse:
    target_str = f"{req.target}{'회' if req.challenge_type == 'count' else '원'}" if req.target else "-"
    current_str = f"{req.current}{'회' if req.challenge_type == 'count' else '원'}"
    context = (
        f"챌린지: {req.title} | 카테고리: {req.category}\n"
        f"목표: {target_str}, 현재: {current_str}, 달성률: {req.progress_pct}%"
    )

    messages = [SystemMessage(content=_NAG_SYSTEM), HumanMessage(content=context)]
    result = await ainvoke_structured(messages, _NagAIOutput)

    nag_message = result.nag_message if result else "조금만 더 힘내세요! 거의 다 왔어요."
    return NagResponse(created_at=datetime.now(timezone.utc), nag_message=nag_message)
