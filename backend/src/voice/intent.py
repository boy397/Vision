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

        Uses token-based similarity matching + substring containment.
        Strips punctuation for robust matching of STT output.
        Falls back to UNKNOWN if no match exceeds threshold.
        """
        if not text:
            return IntentResult(intent=Intent.UNKNOWN, raw_text=text)

        # Strip punctuation for robust matching (STT often adds periods, commas)
        import re
        text_lower = re.sub(r"[^\w\s]", "", text.lower().strip())
        best_match: IntentResult | None = None
        best_score = 0.0

        for _cmd_name, cmd_config in self._commands.items():
            action = cmd_config.get("action", "unknown")
            phrases = cmd_config.get("phrases", [])
            params = cmd_config.get("params", {})

            for phrase in phrases:
                phrase_clean = re.sub(r"[^\w\s]", "", phrase.lower().strip())
                score = self._similarity(text_lower, phrase_clean)
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
        """Enhanced similarity: token-set overlap + substring containment bonus.

        For short commands like "scan", exact containment in the input
        should give a high score even if there are extra words.
        """
        tokens_a = set(a.split())
        tokens_b = set(b.split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        jaccard = len(intersection) / len(union)

        # Containment bonus: if all tokens of the phrase are in the input,
        # score at least 0.7 (for short phrases, this handles "scan" in
        # "could you scan this please" gracefully)
        containment = len(intersection) / len(tokens_b) if tokens_b else 0.0

        # Also check if the phrase appears as a substring in the input
        if b in a:
            containment = max(containment, 0.9)

        # Return the higher of Jaccard and containment-weighted score
        return max(jaccard, containment * 0.85)

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
