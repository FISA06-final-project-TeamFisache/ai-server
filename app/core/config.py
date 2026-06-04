from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    use_openai_direct: bool = False  # True면 OpenAI 직접, False면 OpenRouter 경유

    agent_timeout_portfolio: int = 120
    agent_timeout_rebalance: int = 30
    agent_timeout_report: int = 60

    tavily_api_key: str = ""
    internal_api_key: str = ""

    langchain_tracing_v2: str = ""
    langchain_endpoint: str = ""
    langchain_api_key: str = ""
    langchain_project: str = ""

    # DB
    db_url: str = ""


settings = Settings()
