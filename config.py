"""Configuration module for loading environment variables."""

import os
from pathlib import Path


def _load_env_file():
    """Load .env file into os.environ if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def get_openai_api_key() -> str:
    """
    Load and return the OpenAI API key from environment.

    Returns:
        str: The OpenAI API key

    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please ensure .env file exists with OPENAI_API_KEY set."
        )

    return api_key


def get_openrouter_api_key() -> str:
    """
    Load and return the OpenRouter API key from environment.

    Returns:
        str: The OpenRouter API key

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in environment
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment. "
            "Please ensure .env file exists with OPENROUTER_API_KEY set."
        )
    return api_key


def get_anthropic_api_key() -> str:
    """
    Load and return the Anthropic API key from environment.

    Returns:
        str: The Anthropic API key

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set in environment
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Please ensure .env file exists with ANTHROPIC_API_KEY set."
        )
    return api_key


def get_api_key_for_provider(provider: str) -> str:
    """
    Get the appropriate API key for the given provider.

    Args:
        provider: The LLM provider ("openai", "ollama", "unsloth", "openrouter", or "claude")

    Returns:
        str: The API key for the provider

    Raises:
        ValueError: If provider is unknown or required API key is missing
    """
    if provider == "ollama":
        return "ollama"
    elif provider == "unsloth":
        # Local Unsloth Desktop (llama.cpp) server: no real key required.
        return "unsloth"
    elif provider == "openai":
        return get_openai_api_key()
    elif provider == "openrouter":
        return get_openrouter_api_key()
    elif provider == "claude":
        return get_anthropic_api_key()
    else:
        raise ValueError(f"Unknown provider: {provider}")
