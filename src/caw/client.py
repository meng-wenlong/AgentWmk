from openai import OpenAI, AsyncOpenAI
from typing import Optional
from .config import get_config


_client: Optional[OpenAI] = None


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_async: bool = False,
) -> OpenAI:
    """Get or create a singleton OpenAI client instance.
    
    Args:
        api_key: Optional API key. If not provided, uses config.
        base_url: Optional base URL. If not provided, uses config.
    
    Returns:
        OpenAI client instance
    """
    global _client
    
    config = get_config()
    
    # Use provided values or fall back to config
    actual_api_key = api_key or config.api_key
    actual_base_url = base_url or config.base_url
    
    if _client is None:
        _client = OpenAI(
            api_key=actual_api_key,
            base_url=actual_base_url,
        ) if not use_async else AsyncOpenAI(
            api_key=actual_api_key,
            base_url=actual_base_url,
        )
    
    return _client


def reset_client():
    """Reset the client instance (useful for testing)."""
    global _client
    _client = None