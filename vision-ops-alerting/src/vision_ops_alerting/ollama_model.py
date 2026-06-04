"""Shared Strands Ollama model factory (requires explicit host)."""

from __future__ import annotations

from strands.models.ollama import OllamaModel

from vision_ops_alerting.config import settings


def make_ollama_model(*, temperature: float | None = None) -> OllamaModel:
    kwargs: dict = {"model_id": settings.ollama_model}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return OllamaModel(settings.ollama_host, **kwargs)
