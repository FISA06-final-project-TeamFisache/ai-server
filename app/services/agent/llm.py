from langchain_openai import ChatOpenAI

from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()
import os
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/wooriport",
    "X-Title": "WooriPort AI Server",
}


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL"),
        temperature=temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", 0.7)),
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL"),
        default_headers=_OPENROUTER_HEADERS,
        max_tokens=4096,
    )
