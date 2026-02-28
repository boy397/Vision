"""Intent classifier — maps transcribed voice text to actions."""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Voice command intents."""
    SCAN = "scan"
    START_CONTINUOUS = "start_continuous"
    STOP_CONTINUOUS = "stop_continuous"
    SWITCH_MODE = "switch_mode"
    FOLLOW_UP = "follow_up"
    REPEAT = "repeat"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float = 0.0
    params: dict = field(default_factory=dict)
    raw_text: str = ""


class IntentClassifier:
    """Lightweight intent classifier using keyword/fuzzy matching.

    Loaded from voice_commands.yml config.
    """

    def __init__(self, commands_config: dict) -> None:
        self._commands = commands_config.get("commands", {})
        self._threshold = commands_config.get("matching", {}).get("threshold", 0.7)

    def classify(self, text: str) -> IntentResult:
        """Classify transcribed text into an intent.

        Uses token-based similarity matching against configured phrases.
        Falls back to UNKNOWN if no match exceeds threshold.
        """
        if not text:
            return IntentResult(intent=Intent.UNKNOWN, raw_text=text)

        text_lower = text.lower().strip()
        best_match: IntentResult | None = None
        best_score = 0.0

        for _cmd_name, cmd_config in self._commands.items():
            action = cmd_config.get("action", "unknown")
            phrases = cmd_config.get("phrases", [])
            params = cmd_config.get("params", {})

            for phrase in phrases:
                score = self._similarity(text_lower, phrase.lower())
                if score > best_score:
                    best_score = score
                    intent = self._action_to_intent(action)
                    best_match = IntentResult(
                        intent=intent,
                        confidence=score,
                        params=params,
                        raw_text=text,
                    )

        if best_match and best_score >= self._threshold:
            return best_match

        # If no command matched but text is a question/statement, treat as follow-up
        if any(word in text_lower for word in ["is", "can", "does", "what", "how", "why", "should"]):
            return IntentResult(
                intent=Intent.FOLLOW_UP,
                confidence=0.6,
                params={},
                raw_text=text,
            )

        return IntentResult(intent=Intent.UNKNOWN, confidence=best_score, raw_text=text)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Token-set similarity between two strings."""
        tokens_a = set(a.split())
        tokens_b = set(b.split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)

    @staticmethod
    def _action_to_intent(action: str) -> Intent:
        """Map action string from config to Intent enum."""
        mapping = {
            "scan": Intent.SCAN,
            "start_continuous": Intent.START_CONTINUOUS,
            "stop_continuous": Intent.STOP_CONTINUOUS,
            "switch_mode": Intent.SWITCH_MODE,
            "follow_up": Intent.FOLLOW_UP,
            "repeat": Intent.REPEAT,
        }
        return mapping.get(action, Intent.UNKNOWN)
