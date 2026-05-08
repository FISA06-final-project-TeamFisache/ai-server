from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    model_dir: str = "./models"
    anomaly_model_file: str = "anomaly_v1.pkl"
    recommend_model_file: str = "recommend_v1.pkl"

    agent_timeout_portfolio: int = 30
    agent_timeout_rebalance: int = 30
    agent_timeout_report: int = 60
    ml_timeout_anomaly: int = 5
    ml_timeout_recommend: int = 5

    internal_api_key: str = ""


settings = Settings()
