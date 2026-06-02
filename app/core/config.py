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

    tavily_api_key: str = ""
    internal_api_key: str = ""

    langchain_tracing_v2: str = ""
    langchain_endpoint: str = ""
    langchain_api_key: str = ""
    langchain_project: str = ""

    # DB
    db_url: str = ""
    db_transaction_table: str = ""
    db_asset_table: str = ""
    db_asset_pk_col: str = ""
    db_asset_number_col: str = ""
    db_transaction_asset_fk_col: str = ""
    db_amount_col: str = ""
    db_category_col: str = ""
    db_date_col: str = ""

    # Kafka
    kafka_bootstrap_servers: str = ""
    kafka_input_topic: str = ""
    kafka_output_topic: str = ""
    kafka_consumer_group_id: str = ""

    agent_timeout_consume_alert: int = 30


settings = Settings()
