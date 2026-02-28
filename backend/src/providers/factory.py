"""Unified provider factory — single entry point for creating LLM, TTS, STT instances.

Uses a registry dict to avoid if/elif chains. Adding a new provider =
1. Create the class implementing the base ABC
2. Add one entry to _REGISTRY
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.src.providers.base import BaseLLM, BaseSTT, BaseTTS
from backend.src.providers.llm.azure_llm import AzureLLM
from backend.src.providers.llm.google_llm import GoogleLLM
from backend.src.providers.llm.groq_llm import GroqLLM
from backend.src.providers.tts.elevenlabs_tts import ElevenLabsTTS
from backend.src.providers.tts.sarvam_tts import SarvamTTS
from backend.src.providers.stt.google_stt import GoogleSTT
from backend.src.providers.stt.sarvam_stt import SarvamSTT

if TYPE_CHECKING:
    from backend.src.config import Config

# ── Registry ──
_REGISTRY: dict[str, dict[str, type]] = {
    "llm": {"google": GoogleLLM, "azure": AzureLLM, "groq": GroqLLM},
    "tts": {"elevenlabs": ElevenLabsTTS, "sarvam": SarvamTTS},
    "stt": {"google": GoogleSTT, "sarvam": SarvamSTT},
}


def create_provider(kind: str, config: Config) -> BaseLLM | BaseTTS | BaseSTT:
    """Create a provider instance based on kind ('llm', 'tts', 'stt') and config.

    Reads the `provider` field from the relevant config section
    and instantiates the matching class with provider-specific config.

    Args:
        kind: One of 'llm', 'tts', 'stt'.
        config: Full application Config object.

    Returns:
        Provider instance implementing the relevant base ABC.

    Raises:
        ValueError: If kind or provider name is unknown.
    """
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown provider kind: {kind}. Available: {list(_REGISTRY.keys())}")

    section = getattr(config, kind)
    provider_name = section.provider

    cls = _REGISTRY[kind].get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown {kind} provider: {provider_name}. "
            f"Available: {list(_REGISTRY[kind].keys())}"
        )

    # Get provider-specific config dict
    provider_config = getattr(section, provider_name)
    return cls(provider_config.model_dump() if hasattr(provider_config, "model_dump") else provider_config)


def create_all_providers(config: Config) -> dict[str, BaseLLM | BaseTTS | BaseSTT]:
    """Create all three providers in one call."""
    return {
        "llm": create_provider("llm", config),
        "tts": create_provider("tts", config),
        "stt": create_provider("stt", config),
    }
