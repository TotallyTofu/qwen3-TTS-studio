"""Unified LLM client abstraction supporting OpenAI, Ollama, Unsloth Desktop, OpenRouter, and Claude."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    UNSLOTH = "unsloth"
    OPENROUTER = "openrouter"
    CLAUDE = "claude"


DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-5.2",
    LLMProvider.OLLAMA: "qwen3:8b",
    LLMProvider.UNSLOTH: "",
    LLMProvider.OPENROUTER: "google/gemini-2.5-flash",
    LLMProvider.CLAUDE: "claude-sonnet-4-5-20250929",
}

PROVIDER_BASE_URLS = {
    LLMProvider.OPENAI: "",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
    LLMProvider.UNSLOTH: "http://localhost:8888/v1",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    LLMProvider.CLAUDE: "",
}

PROVIDER_MODEL_OPTIONS: dict[LLMProvider, list[str]] = {
    LLMProvider.OPENAI: [
        "gpt-5.2",
        "gpt-5.1",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    LLMProvider.OLLAMA: [
        "qwen3:8b",
        "qwen3:14b",
        "llama3.1:8b",
    ],
    # Unsloth Desktop serves whatever GGUF model the user loaded in its UI;
    # the model name is user-typed in the studio (allow_custom_value=True).
    LLMProvider.UNSLOTH: [],
    LLMProvider.OPENROUTER: [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.6",
        "openai/gpt-5.2",
        "openai/gpt-5.3-codex",
        "deepseek/deepseek-r1",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    LLMProvider.CLAUDE: [
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-20250514",
    ],
}


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: LLMProvider
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.4


def create_llm_client(config: LLMConfig) -> Any:
    """Create a provider client for the specified LLM provider."""
    if config.provider == LLMProvider.CLAUDE:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for Claude. Install with: pip install anthropic"
            ) from e

        anthropic_kwargs: dict[str, Any] = {}
        if config.api_key:
            anthropic_kwargs["api_key"] = config.api_key
        if config.base_url:
            anthropic_kwargs["base_url"] = config.base_url
        return Anthropic(**anthropic_kwargs)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        ) from e

    kwargs: dict[str, Any] = {}
    if config.provider == LLMProvider.OLLAMA:
        kwargs["api_key"] = "ollama"
    elif config.provider == LLMProvider.UNSLOTH:
        # Local Unsloth Desktop (llama.cpp) server: no real key needed, but
        # the OpenAI client requires a non-empty value.
        kwargs["api_key"] = config.api_key or "unsloth"
    elif config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.4,
    json_mode: bool = True,
    provider: LLMProvider = LLMProvider.OPENAI,
) -> str:
    """Call chat completion API with retry logic.

    Args:
        client: OpenAI-compatible client instance.
        model: Model identifier string.
        messages: Chat messages list.
        temperature: Sampling temperature.
        json_mode: Whether to request JSON output.
        provider: LLM provider for provider-specific handling.

    Returns:
        Response content string.

    Raises:
        ValueError: If the LLM returns empty or invalid content.
        openai errors: After exhausting retries for transient API errors.
    """
    if provider == LLMProvider.CLAUDE:
        return _chat_completion_anthropic(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            json_mode=json_mode,
        )

    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

    max_retries = 3
    backoff_seconds = [1, 2, 4]

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
            except APIError as exc:
                status_code = getattr(exc, "status_code", None)
                response_format_rejected = (
                    json_mode
                    and "response_format" in kwargs
                    and (
                        _is_response_format_error(exc)
                        or (
                            provider != LLMProvider.OPENAI
                            and status_code == 400
                        )
                    )
                )
                if response_format_rejected:
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**fallback_kwargs)
                else:
                    raise

            if not response.choices or response.choices[0].message.content is None:
                raise ValueError(
                    f"LLM returned empty content (provider={provider.value}, model={model})"
                )
            content: str = response.choices[0].message.content

            if json_mode:
                try:
                    json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = _extract_json_from_text(content)

            return content

        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise
        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            is_retryable = status_code is None or status_code >= 500 or status_code == 429
            if is_retryable and attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise

    raise RuntimeError("Unreachable: all retry attempts exhausted without return or raise")


def _extract_json_from_text(text: str) -> str:
    """Extract a JSON object from text that may contain prose or markdown fences.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    import re

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            json.loads(fenced.group(1))
            return fenced.group(1)
        except (json.JSONDecodeError, TypeError):
            pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        pos = text.find("{", idx)
        if pos == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return text[pos:end_idx]
        except json.JSONDecodeError:
            pass
        idx = pos + 1

    raise ValueError("No valid JSON object found in LLM response.")


def _is_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    anth_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
            continue
        normalized_role = role if role in {"user", "assistant"} else "user"
        anth_messages.append({"role": normalized_role, "content": content})
    return "\n\n".join(part for part in system_parts if part), anth_messages


def _extract_text_from_anthropic_response(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _chat_completion_anthropic(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    json_mode: bool,
) -> str:
    from anthropic import APIConnectionError, APIError, APIStatusError, RateLimitError

    max_retries = 3
    backoff_seconds = [1, 2, 4]
    system_prompt, anth_messages = _split_system_messages(messages)
    if json_mode:
        json_instruction = "Return only a valid JSON object with no markdown fences or extra prose."
        system_prompt = f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=temperature,
                system=system_prompt,
                messages=anth_messages,
            )
            content = _extract_text_from_anthropic_response(response)
            if not content:
                raise ValueError(f"LLM returned empty content (provider=claude, model={model})")
            if json_mode:
                try:
                    json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = _extract_json_from_text(content)
            return content
        except (RateLimitError, APIConnectionError):
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            is_retryable = status_code is None or status_code >= 500 or status_code == 429
            if is_retryable and attempt < max_retries - 1:
                time.sleep(backoff_seconds[attempt])
                continue
            raise
        except APIError:
            raise

    raise RuntimeError("Unreachable: all Anthropic retry attempts exhausted")


def get_default_config(
    provider: LLMProvider,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> LLMConfig:
    """Create default LLM configuration for a provider."""
    if not model:
        model = DEFAULT_MODELS[provider]
    if not base_url:
        base_url = PROVIDER_BASE_URLS[provider]
    if provider == LLMProvider.OLLAMA and not api_key:
        api_key = "ollama"
    if provider == LLMProvider.UNSLOTH and not api_key:
        api_key = "unsloth"

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def validate_connection(config: LLMConfig) -> tuple[bool, str]:
    """Validate that the LLM configuration works."""
    try:
        client = create_llm_client(config)
        if config.provider == LLMProvider.CLAUDE:
            response = client.messages.create(
                model=config.model,
                max_tokens=32,
                temperature=0.0,
                messages=[{"role": "user", "content": "Respond with a single word: OK"}],
            )
            content = _extract_text_from_anthropic_response(response)
            if not content:
                return False, "Received empty response"
        else:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": "Respond with a single word: OK"}],
                temperature=0.0,
            )
            if not response.choices or not response.choices[0].message.content:
                return False, "Received empty response"
        return True, f"Successfully connected to {config.provider.value}"

    except ImportError as e:
        return False, f"Missing dependency: {e}"
    except Exception as e:
        return False, f"Connection failed: {e}"
