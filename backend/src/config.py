"""YAML configuration loader with validation and mode management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Path Constants ──
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG = _CONFIG_DIR / "config.yml"


# ── Pydantic Models ──

class DetectionConfig(BaseModel):
    model: str = "yolo26n"
    confidence: float = 0.45
    device: str = "cuda"
    input_size: int = 640
    quantized: bool = True
    state_change_threshold: float = 0.3
    classes: list[str] = Field(default_factory=list)


class LLMProviderConfig(BaseModel):
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 512
    stream: bool = True
    api_version: str = ""


class LLMConfig(BaseModel):
    provider: str = "google"
    google: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    azure: LLMProviderConfig = Field(default_factory=LLMProviderConfig)


class STTProviderConfig(BaseModel):
    model: str = ""
    language: str = "en-IN"


class STTConfig(BaseModel):
    provider: str = "google"
    google: STTProviderConfig = Field(default_factory=STTProviderConfig)
    sarvam: STTProviderConfig = Field(default_factory=STTProviderConfig)


class TTSProviderConfig(BaseModel):
    model: str = ""
    voice_id: str = ""
    stability: float = 0.5
    similarity_boost: float = 0.75
    stream: bool = True
    language: str = ""
    speaker: str = ""


class TTSConfig(BaseModel):
    provider: str = "elevenlabs"
    elevenlabs: TTSProviderConfig = Field(default_factory=TTSProviderConfig)
    sarvam: TTSProviderConfig = Field(default_factory=TTSProviderConfig)


class ModeOverride(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    llm: dict[str, str] = Field(default_factory=dict)
    prompt_file: str = ""


class VoiceConfig(BaseModel):
    vad_aggressiveness: int = 2
    wake_word_enabled: bool = False
    silence_timeout_ms: int = 800


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AppConfig(BaseModel):
    name: str = "Vision Assistive System"
    default_mode: str = "medical"
    trigger: str = "voice"
    fps: int = 10
    language: str = "en"
    debug: bool = False


class Config(BaseModel):
    """Root configuration model."""

    app: AppConfig = Field(default_factory=AppConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    modes: dict[str, ModeOverride] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    model_config = {"arbitrary_types_allowed": True}

    # ── Runtime state (not persisted) ──
    _current_mode: str = "medical"
    _prompts: dict[str, dict] = {}

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def switch_mode(self, mode: str) -> None:
        """Switch active mode (medical/retail). Updates detection config overlays."""
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.modes.keys())}")
        self._current_mode = mode

    def get_active_detection(self) -> DetectionConfig:
        """Return detection config merged with current mode overrides."""
        base = self.detection.model_copy()
        if self._current_mode in self.modes:
            override = self.modes[self._current_mode].detection
            for field in override.model_fields_set:
                setattr(base, field, getattr(override, field))
        return base

    def get_active_llm_model(self) -> str:
        """Return LLM model for active mode (mode override > global default)."""
        if self._current_mode in self.modes:
            mode_llm = self.modes[self._current_mode].llm
            if "model" in mode_llm:
                return mode_llm["model"]
        provider = self.llm.provider
        return getattr(self.llm, provider).model

    def get_active_llm_provider(self) -> str:
        """Return LLM provider for active mode (mode override > global default)."""
        if self._current_mode in self.modes:
            mode_llm = self.modes[self._current_mode].llm
            if "provider" in mode_llm:
                return mode_llm["provider"]
        return self.llm.provider

    def get_prompt(self, prompt_type: str = "vision_system_prompt") -> str:
        """Load and return prompt for current mode."""
        mode = self._current_mode
        if mode not in self._prompts:
            self._load_mode_prompts(mode)
        return self._prompts.get(mode, {}).get(prompt_type, "")

    def _load_mode_prompts(self, mode: str) -> None:
        """Load prompt YAML for a given mode."""
        if mode not in self.modes:
            return
        prompt_file = _CONFIG_DIR / self.modes[mode].prompt_file
        if prompt_file.exists():
            with open(prompt_file) as f:
                self._prompts[mode] = yaml.safe_load(f) or {}


# ── Voice Commands Loader ──

def load_voice_commands(path: Path | None = None) -> dict[str, Any]:
    """Load voice command definitions from YAML."""
    path = path or _CONFIG_DIR / "voice_commands.yml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Config Loader ──

def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from YAML file.

    Environment variable overrides:
      - VISION_CONFIG_PATH: path to config YAML
    """
    if path is None:
        path = Path(os.getenv("VISION_CONFIG_PATH", str(_DEFAULT_CONFIG)))
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = Config(**raw)
    config._current_mode = config.app.default_mode

    # Pre-load prompts for default mode
    config._load_mode_prompts(config._current_mode)

    return config
