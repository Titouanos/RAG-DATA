"""Tests des providers LLM : retry/backoff + fabrique (sans appel réseau)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rag_builder.core.llm import build_provider
from rag_builder.core.llm.retry import stream_with_retries


def test_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        yield "a"
        yield "b"

    out = list(stream_with_retries(factory, attempts=3, base_delay=0.0))
    assert out == ["a", "b"] and calls["n"] == 3


def test_retry_gives_up_and_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def factory():
        raise TimeoutError("always")
        yield  # pragma: no cover

    with pytest.raises(TimeoutError):
        list(stream_with_retries(factory, attempts=2, base_delay=0.0))


def test_non_retriable_raises_immediately():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        raise ValueError("auth")
        yield  # pragma: no cover

    with pytest.raises(ValueError):
        list(stream_with_retries(factory, attempts=3, is_retriable=lambda _e: False))
    assert calls["n"] == 1


def test_error_after_first_token_propagates(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def factory():
        yield "x"
        raise RuntimeError("mid-stream")

    gen = stream_with_retries(factory, attempts=3, base_delay=0.0)
    assert next(gen) == "x"
    with pytest.raises(RuntimeError):
        next(gen)


def _settings(**kw):
    base = {
        "mistral_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "ollama_base_url": "http://localhost:11434",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_provider_requires_keys():
    with pytest.raises(ValueError):
        build_provider("mistral", "mistral-large-latest", _settings())
    with pytest.raises(ValueError):
        build_provider("anthropic", "claude-haiku-4-5", _settings())
    with pytest.raises(ValueError):
        build_provider("gemini", "gemini-2.5-flash", _settings())


def test_build_provider_unknown():
    with pytest.raises(ValueError):
        build_provider("openai", "gpt", _settings())


def test_build_provider_ollama_needs_no_key():
    from rag_builder.core.llm.ollama import OllamaProvider

    p = build_provider("ollama", "llama3.1", _settings())
    assert isinstance(p, OllamaProvider)


def test_build_provider_constructs_with_keys():
    from rag_builder.core.llm.anthropic import AnthropicProvider
    from rag_builder.core.llm.mistral import MistralProvider

    assert isinstance(
        build_provider("mistral", "m", _settings(mistral_api_key="x")), MistralProvider
    )
    assert isinstance(
        build_provider("anthropic", "c", _settings(anthropic_api_key="x")), AnthropicProvider
    )
