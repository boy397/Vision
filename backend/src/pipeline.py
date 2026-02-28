"""Main orchestration pipeline — ties together detection, LLM, TTS, STT, and voice."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import numpy as np

from backend.src.config import Config, load_voice_commands
from backend.src.detection.detector import Detector, Detection
from backend.src.detection.state_tracker import StateTracker
from backend.src.providers.base import BaseLLM, BaseTTS, BaseSTT
from backend.src.providers.factory import create_all_providers
from backend.src.voice.intent import IntentClassifier, Intent, IntentResult
from backend.src.utils.image import crop_roi, encode_jpeg, resize_frame
from backend.src.utils.logging import latency_tracker

logger = logging.getLogger(__name__)


class PipelineResult:
    """Result from a pipeline scan operation."""

    def __init__(
        self,
        detections: list[dict] | None = None,
        analysis: dict | None = None,
        tts_text: str = "",
        mode: str = "",
        state_changed: bool = True,
    ):
        self.detections = detections or []
        self.analysis = analysis or {}
        self.tts_text = tts_text
        self.mode = mode
        self.state_changed = state_changed

    def to_dict(self) -> dict:
        return {
            "detections": self.detections,
            "analysis": self.analysis,
            "tts_text": self.tts_text,
            "mode": self.mode,
            "state_changed": self.state_changed,
        }


class VisionPipeline:
    """Main pipeline orchestrator.

    Manages the full flow:
      Frame → YOLO Detect → State Check → ROI Crop → LLM Analyze → TTS Speak
      Audio → STT Transcribe → Intent Classify → Action Dispatch
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Detection
        det_config = config.get_active_detection()
        self.detector = Detector(det_config.model_dump())
        self.state_tracker = StateTracker(
            iou_threshold=config.detection.state_change_threshold
        )

        # Providers
        providers = create_all_providers(config)
        self.llm: BaseLLM = providers["llm"]
        self.tts: BaseTTS = providers["tts"]
        self.stt: BaseSTT = providers["stt"]

        # Voice
        voice_cmds = load_voice_commands()
        self.intent_classifier = IntentClassifier(voice_cmds)

        # State
        self._last_analysis: dict = {}
        self._last_tts_text: str = ""
        self._continuous_mode: bool = config.app.trigger == "continuous"

    # ── Frame Processing (Tier 1 + Tier 2) ──

    async def process_frame(self, frame: np.ndarray, force: bool = False) -> PipelineResult:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image as numpy array.
            force: If True, skip state change check (for triggered scans).
        """
        mode = self.config.current_mode
        det_config = self.config.get_active_detection()

        # Tier 1: YOLO Detection
        with latency_tracker("yolo_detect", logger):
            detections = self.detector.detect(
                frame,
                confidence=det_config.confidence,
                filter_classes=det_config.classes or None,
            )

        if not detections:
            return PipelineResult(mode=mode, tts_text="I don't see anything relevant right now.")

        det_dicts = [d.to_dict() for d in detections]

        # State change check (skip if forced/triggered)
        if not force:
            with latency_tracker("state_check", logger):
                state_changed = self.state_tracker.check(det_dicts)
            if not state_changed:
                return PipelineResult(
                    detections=det_dicts,
                    mode=mode,
                    state_changed=False,
                )

        # Tier 2: LLM Vision Analysis
        # Crop ROI from best detection (highest confidence)
        best_det = max(detections, key=lambda d: d.confidence)
        with latency_tracker("roi_crop", logger):
            roi = crop_roi(frame, best_det.bbox)
            roi_bytes = encode_jpeg(resize_frame(roi))

        prompt = self.config.get_prompt("vision_system_prompt")

        with latency_tracker("llm_analyze", logger):
            analysis = await self.llm.analyze_image(roi_bytes, prompt)

        # Generate TTS text from template
        tts_text = self._format_tts(analysis)
        self._last_analysis = analysis
        self._last_tts_text = tts_text

        return PipelineResult(
            detections=det_dicts,
            analysis=analysis,
            tts_text=tts_text,
            mode=mode,
            state_changed=True,
        )

    # ── Voice Processing ──

    async def process_voice(self, audio: bytes) -> dict[str, Any]:
        """Process voice audio: STT → Intent → Action.

        Returns action result dict.
        """
        # STT
        with latency_tracker("stt_transcribe", logger):
            text = await self.stt.transcribe(audio)

        if not text:
            return {"action": "none", "text": "", "message": "Could not understand audio"}

        # Intent Classification
        with latency_tracker("intent_classify", logger):
            result = self.intent_classifier.classify(text)

        logger.info(f"Voice: '{text}' → intent={result.intent.value} (conf={result.confidence:.2f})")

        # Dispatch
        return await self._dispatch_intent(result)

    async def _dispatch_intent(self, result: IntentResult) -> dict[str, Any]:
        """Route intent to appropriate action."""
        match result.intent:
            case Intent.SCAN:
                return {"action": "scan", "text": result.raw_text, "message": "Scan triggered"}

            case Intent.START_CONTINUOUS:
                self._continuous_mode = True
                return {"action": "start_continuous", "message": "Continuous mode started"}

            case Intent.STOP_CONTINUOUS:
                self._continuous_mode = False
                return {"action": "stop_continuous", "message": "Continuous mode stopped"}

            case Intent.SWITCH_MODE:
                mode = result.params.get("mode", "medical")
                self.config.switch_mode(mode)
                # Re-init detector for new mode
                det_config = self.config.get_active_detection()
                self.detector = Detector(det_config.model_dump())
                self.state_tracker.reset()
                return {
                    "action": "switch_mode",
                    "mode": mode,
                    "message": f"Switched to {mode} mode",
                }

            case Intent.FOLLOW_UP:
                return await self._handle_follow_up(result.raw_text)

            case Intent.REPEAT:
                return {
                    "action": "repeat",
                    "tts_text": self._last_tts_text,
                    "message": "Repeating last response",
                }

            case _:
                # Treat as follow-up question if we have context
                if self._last_analysis:
                    return await self._handle_follow_up(result.raw_text)
                return {"action": "unknown", "text": result.raw_text, "message": "Command not recognized"}

    async def _handle_follow_up(self, question: str) -> dict[str, Any]:
        """Handle follow-up chat using last detection context."""
        if not self._last_analysis:
            return {"action": "follow_up", "message": "No previous scan to ask about. Try scanning something first."}

        chat_prompt = self.config.get_prompt("chat_system_prompt")
        context = {
            "system_prompt": chat_prompt,
            **self._last_analysis,
        }

        with latency_tracker("llm_chat", logger):
            response = await self.llm.chat(context, question)

        self._last_tts_text = response
        return {
            "action": "follow_up",
            "response": response,
            "tts_text": response,
            "message": "Follow-up answered",
        }

    # ── TTS ──

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech audio chunks."""
        with latency_tracker("tts_synthesize", logger):
            async for chunk in self.tts.synthesize(text):
                yield chunk

    # ── Helpers ──

    def _format_tts(self, analysis: dict) -> str:
        """Format analysis dict into natural speech text.

        Uses simple template logic — falls back to key extraction.
        """
        if "error" in analysis:
            return "Sorry, I couldn't analyze that clearly. Please try again."

        if "raw_response" in analysis:
            return analysis["raw_response"]

        # Build natural language from structured data
        parts: list[str] = []
        mode = self.config.current_mode

        if mode == "medical":
            parts = self._format_medical(analysis)
        elif mode == "retail":
            parts = self._format_retail(analysis)

        return " ".join(parts) if parts else "I analyzed the image but couldn't extract clear details."

    @staticmethod
    def _format_medical(data: dict) -> list[str]:
        """Format medical analysis for TTS."""
        parts = []
        med = data.get("medicine_name", {})
        if isinstance(med, dict) and med.get("brand"):
            generic = f", also known as {med['generic']}" if med.get("generic") else ""
            parts.append(f"This is {med['brand']}{generic}.")
        if data.get("manufacturer"):
            parts.append(f"By {data['manufacturer']}.")
        if data.get("expiry_date"):
            parts.append(f"Expiry: {data['expiry_date']}.")
        if data.get("dosage"):
            parts.append(f"Dosage: {data['dosage']}.")
        if data.get("allergen_flags"):
            parts.append(f"Allergen alert: {', '.join(data['allergen_flags'])}.")
        if data.get("warnings"):
            parts.append(f"Warning: {'. '.join(data['warnings'])}.")
        return parts

    @staticmethod
    def _format_retail(data: dict) -> list[str]:
        """Format retail analysis for TTS."""
        parts = []
        currency = data.get("currency", {})
        product = data.get("product", {})

        if data.get("item_type") == "currency_note" and currency:
            denom = currency.get("denomination", "")
            parts.append(f"I can see a {denom} rupee note.")
            if currency.get("count", 1) > 1:
                parts.append(f"There are {currency['count']} notes, totaling {currency['total_value']} rupees.")
        elif product:
            name = f"{product.get('brand', '')} {product.get('name', '')}".strip()
            if name:
                parts.append(f"This is {name}.")
            if product.get("weight_volume"):
                parts.append(f"{product['weight_volume']}.")
            if product.get("expiry_date"):
                parts.append(f"Expiry: {product['expiry_date']}.")
            if product.get("price"):
                parts.append(f"Price: {product['price']} rupees.")
        return parts

    @property
    def is_continuous(self) -> bool:
        return self._continuous_mode
