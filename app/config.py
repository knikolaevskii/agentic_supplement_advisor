"""Application settings loaded from environment variables."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ BEFORE any langchain imports so that
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY are visible at import time.
load_dotenv()


class Settings(BaseSettings):
    """Central configuration sourced from .env / environment variables."""

    openai_api_key: str
    tavily_api_key: str

    chroma_persist_dir: str = "./data/chroma"
    general_collection_name: str = "general_kb"

    llm_model: str = "gpt-5.2"
    tavily_timeout: int = 10

    # LangSmith tracing (optional)
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None
    langsmith_project: str = "supplement-advisor"

    model_config = {"env_file": ".env"}


settings = Settings()
