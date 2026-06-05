from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        temperature=temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
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
    except Exception as e:
        logger.debug("structured output 1차 시도 실패: %s", e)

    # 2차: 일반 텍스트 응답 → regex JSON 파싱 → Pydantic 검증
    try:
        raw = llm.invoke(messages).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return schema.model_validate(json.loads(match.group()))
    except Exception as e:
        logger.debug("structured output 2차 시도 실패: %s", e)

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
    except Exception as e:
        logger.debug("structured output 1차 시도 실패: %s", e)

    # 2차: 일반 텍스트 응답 → regex JSON 파싱 → Pydantic 검증
    try:
        raw = (await llm.ainvoke(messages)).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return schema.model_validate(json.loads(match.group()))
    except Exception as e:
        logger.debug("structured output 2차 시도 실패: %s", e)

    return None
