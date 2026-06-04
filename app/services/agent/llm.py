from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()
import os
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/wooriport",
    "X-Title": "WooriPort AI Server",
}

T = TypeVar("T", bound=BaseModel)


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL"),
        temperature=temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", 0.7)),
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL"),
        default_headers=_OPENROUTER_HEADERS,
        max_tokens=4096,
    )


def invoke_structured(
    messages: list[BaseMessage],
    schema: type[T],
    temperature: float | None = None,
) -> T | None:
    """structured output 시도 후 실패 시 regex JSON 파싱으로 폴백."""
    llm = get_llm(temperature)

    # 1차: function calling 방식 structured output
    try:
        result = llm.with_structured_output(schema).invoke(messages)
        if isinstance(result, schema):
            return result
    except Exception:
        pass

    # 2차: 일반 텍스트 응답 → regex JSON 파싱 → Pydantic 검증
    try:
        raw = llm.invoke(messages).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return schema.model_validate(json.loads(match.group()))
    except Exception:
        pass

    return None


async def ainvoke_structured(
    messages: list[BaseMessage],
    schema: type[T],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> T | None:
    """invoke_structured의 비동기 버전."""
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    llm = get_llm(temperature)
    if kwargs:
        llm = llm.bind(**kwargs)

    # 1차: function calling 방식 structured output
    try:
        result = await llm.with_structured_output(schema).ainvoke(messages)
        if isinstance(result, schema):
            return result
    except Exception:
        pass

    # 2차: 일반 텍스트 응답 → regex JSON 파싱 → Pydantic 검증
    try:
        raw = (await llm.ainvoke(messages)).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return schema.model_validate(json.loads(match.group()))
    except Exception:
        pass

    return None
