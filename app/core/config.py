from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o"
    llm_temperature: float = 0.2

    model_dir: str = "./models"
    anomaly_model_file: str = "anomaly_v1.pkl"
    recommend_model_file: str = "recommend_v1.pkl"

    agent_timeout_portfolio: int = 120
    agent_timeout_rebalance: int = 30
    agent_timeout_report: int = 60
    ml_timeout_anomaly: int = 5
    ml_timeout_recommend: int = 5

    internal_api_key: str = ""

    # Kafka
    kafka_bootstrap_servers: str = ""
    kafka_input_topic: str = ""
    kafka_output_topic: str = ""
    kafka_consumer_group_id: str = ""
    kafka_anomaly_window_seconds: int = 600  # 10분
    kafka_anomaly_max_count: int = 5  # 10분 내 5건 초과 시 이상 탐지


settings = Settings()
