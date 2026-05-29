from langchain_openai import ChatOpenAI

from app.core.config import settings

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/wooriport",
    "X-Title": "WooriPort AI Server",
}


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        default_headers=_OPENROUTER_HEADERS,
        max_tokens=4096,
    )
