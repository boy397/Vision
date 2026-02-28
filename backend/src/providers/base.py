"""Abstract base classes for all providers (LLM, TTS, STT).

Every provider implementation inherits from one of these ABCs.
Swapping providers requires zero code changes — just update config.yml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLM(ABC):
    """Abstract base for vision LLM + chat providers."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    async def analyze_image(self, image: bytes, prompt: str) -> dict:
        """Analyze an image with a mode-specific prompt. Returns structured JSON."""
        ...

    @abstractmethod
    async def chat(self, context: dict, question: str) -> str:
        """Follow-up chat with previous detection context."""
        ...


class BaseTTS(ABC):
    """Abstract base for text-to-speech providers."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech audio. Yields audio chunks for streaming."""
        ...


class BaseSTT(ABC):
    """Abstract base for speech-to-text providers."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes to text."""
        ...
