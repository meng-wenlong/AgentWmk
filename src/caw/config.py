import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI client."""
    api_key: str = "EMPTY"
    base_url: str = "http://localhost:8000/v1"
    default_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    watermark_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    
    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            default_model=os.getenv("OPENAI_DEFAULT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        )


# Global configuration instance
_config: Optional[OpenAIConfig] = None


def get_config() -> OpenAIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = OpenAIConfig.from_env()
    return _config


def set_config(config: OpenAIConfig):
    """Set the global configuration instance."""
    global _config
    _config = config