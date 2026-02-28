"""Main orchestration pipeline — ties together detection, LLM, TTS, STT, and voice.

Includes object tracking integration and comprehensive latency debugging.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

import numpy as np

from backend.src.config import Config, load_voice_commands
from backend.src.detection.detector import Detector
from backend.src.detection.state_tracker import StateTracker
from backend.src.providers.base import BaseLLM, BaseTTS, BaseSTT
from backend.src.providers.factory import create_all_providers
from backend.src.voice.intent import IntentClassifier, Intent, IntentResult
from backend.src.utils.image import crop_roi, encode_jpeg, resize_frame
from backend.src.utils.logging import latency_tracker

import pytesseract
from PIL import Image

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
      Frame → YOLO Detect+Track → State Check → ROI Crop → LLM Analyze → TTS Speak
      Audio → STT Transcribe → Intent Classify → Action Dispatch
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Detection + Tracking
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

        # Pipeline stats
        self._total_frames = 0
        self._total_frame_time_ms = 0.0
        self._total_voice_cmds = 0
        self._total_voice_time_ms = 0.0
        self._skipped_by_tracking = 0
        self._skipped_by_state = 0

        logger.info(
            f"[Pipeline] Initialized: mode={config.current_mode}, "
            f"continuous={self._continuous_mode}, "
            f"llm={config.get_active_llm_provider()}, "
            f"tts={config.tts.provider}, stt={config.stt.provider}"
        )

    # ── Frame Processing (Tier 1 + Tier 2) ──

    async def process_frame(self, frame: np.ndarray, force: bool = False) -> PipelineResult:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image as numpy array.
            force: If True, skip state change check (for triggered scans).
        """
        frame_start = time.perf_counter()
        self._total_frames += 1
        frame_num = self._total_frames
        mode = self.config.current_mode
        det_config = self.config.get_active_detection()
        detection_enabled = self.config.app.detection_enabled

        logger.debug(
            f"[Pipeline] Frame #{frame_num}: shape={frame.shape}, "
            f"mode={mode}, force={force}, detection_enabled={detection_enabled}"
        )

        # ── Path A: YOLO disabled — send full frame straight to LLM ──
        if not detection_enabled:
            logger.info(
                f"[Pipeline] Frame #{frame_num}: YOLO DISABLED — direct to LLM "
                f"(using {type(self.llm).__name__}, config says '{self.config.get_active_llm_provider()}')"
            )
            prompt = self.config.get_prompt("vision_system_prompt")
            with latency_tracker("roi_encode", logger):
                frame_bytes = encode_jpeg(resize_frame(frame))
                logger.debug(f"[Pipeline] Full frame encoded: {len(frame_bytes)} bytes")

            with latency_tracker("llm_analyze", logger):
                analysis = await self.llm.analyze_image(frame_bytes, prompt)

            # Log LLM output for debugging
            logger.info(
                f"[Pipeline] Frame #{frame_num}: LLM raw response: "
                f"{str(analysis)[:300]}"
            )

            tts_text = self._format_tts(analysis)
            self._last_analysis = analysis
            self._last_tts_text = tts_text

            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            logger.info(
                f"[Pipeline] Frame #{frame_num}: no-YOLO COMPLETE in {elapsed_ms:.0f}ms, "
                f"tts_text=\"{tts_text[:100]}\""  # Log actual TTS text
            )
            return PipelineResult(
                detections=[],
                analysis=analysis,
                tts_text=tts_text,
                mode=mode,
                state_changed=True,
            )

        # ── Path B: YOLO enabled — detect first ──
        with latency_tracker("yolo_detect", logger):
            detections = self.detector.detect(
                frame,
                confidence=det_config.confidence,
                filter_classes=det_config.classes or None,
            )

        if not detections:
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            self._total_frame_time_ms += elapsed_ms
            logger.info(
                f"[Pipeline] Frame #{frame_num}: no detections ({elapsed_ms:.0f}ms) — "
                "returning 'nothing relevant'"
            )
            return PipelineResult(mode=mode, tts_text="I don't see anything relevant right now.")

        det_dicts = [d.to_dict() for d in detections]

        # Check if all detections are already-tracked objects (skip re-analysis)
        if not force:
            all_tracked = all(
                not self.detector.is_new_object(d) for d in detections
            )
            if all_tracked:
                self._skipped_by_tracking += 1
                elapsed_ms = (time.perf_counter() - frame_start) * 1000
                self._total_frame_time_ms += elapsed_ms
                logger.debug(
                    f"[Pipeline] Frame #{frame_num}: all objects already tracked, "
                    f"skipping LLM ({elapsed_ms:.0f}ms) "
                    f"[skip_count={self._skipped_by_tracking}]"
                )
                return PipelineResult(
                    detections=det_dicts,
                    mode=mode,
                    state_changed=False,
                )

        # State change check (skip if forced/triggered)
        if not force:
            with latency_tracker("state_check", logger):
                state_changed = self.state_tracker.check(det_dicts)
            if not state_changed:
                self._skipped_by_state += 1
                elapsed_ms = (time.perf_counter() - frame_start) * 1000
                self._total_frame_time_ms += elapsed_ms
                logger.debug(
                    f"[Pipeline] Frame #{frame_num}: state unchanged, "
                    f"skipping LLM ({elapsed_ms:.0f}ms) "
                    f"[skip_count={self._skipped_by_state}]"
                )
                return PipelineResult(
                    detections=det_dicts,
                    mode=mode,
                    state_changed=False,
                )

        # Tier 2: LLM Vision Analysis
        # Crop ROI from best detection (highest confidence)
        best_det = max(detections, key=lambda d: d.confidence)
        logger.info(
            f"[Pipeline] Frame #{frame_num}: Tier 2 triggered — "
            f"best={best_det.class_name}(conf={best_det.confidence:.2f}, "
            f"track_id={best_det.track_id}), "
            f"total_dets={len(detections)}"
        )

        with latency_tracker("roi_crop", logger):
            roi = crop_roi(frame, best_det.bbox)
            roi_bytes = encode_jpeg(resize_frame(roi))
            logger.debug(
                f"[Pipeline] ROI: original={roi.shape}, encoded={len(roi_bytes)} bytes"
            )

        # PyTesseract Local OCR (Tier 1 Vision)
        ocr_text = ""
        with latency_tracker("local_ocr", logger):
            try:
                import os
                # Windows fallback for Tesseract if not in PATH
                if os.name == 'nt' and not getattr(pytesseract.pytesseract, 'tesseract_cmd', '').endswith('tesseract.exe'):
                    default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    if os.path.exists(default_path):
                        pytesseract.pytesseract.tesseract_cmd = default_path

                # Convert ROI BGR numpy array to PIL Image (RGB)
                roi_rgb = roi[..., ::-1] # BGR to RGB
                pil_img = Image.fromarray(roi_rgb)
                ocr_text = pytesseract.image_to_string(pil_img).strip()
            except Exception as e:
                logger.error(f"[Pipeline] PyTesseract Error: {e}")

        # If OCR text found, fast-path with local text processing / SLM prompt
        if ocr_text and len(ocr_text) > 3:
            logger.info(f"[Pipeline] Frame #{frame_num}: Local OCR Found Text: {ocr_text[:50]}...")
            
            # Send just the OCR text to the LLM as a faster 'text-only' generation
            prompt = self.config.get_prompt("vision_system_prompt")
            text_prompt = f"Extract structured data from this OCR text: {ocr_text}\n\nTask:\n{prompt}"
            
            # Use Chat API for structured text-to-JSON
            with latency_tracker("llm_text_analyze", logger):
                analysis_text = await self.llm.chat({"system_prompt": "You extract structured data from OCR text in JSON format."}, text_prompt)
                
                # Robustly parse JSON from Chat model response (stripping markdown)
                parsed_analysis = None
                if hasattr(self.llm, "_parse_response"):
                    parsed_analysis = self.llm._parse_response(analysis_text)
                else:
                    try:
                        import json
                        # Strip markdown blocks manually
                        clean_text = analysis_text.strip()
                        if clean_text.startswith("```"):
                            lines = clean_text.split("\n")
                            if lines[0].startswith("```json"):
                                lines = lines[1:]
                            else:
                                lines = lines[1:]
                            if len(lines) > 0 and lines[-1] == "```":
                                lines = lines[:-1]
                            clean_text = "\n".join(lines).strip()
                        parsed_analysis = json.loads(clean_text)
                    except Exception:
                        parsed_analysis = {"raw_response": analysis_text, "parse_error": True}
                
                analysis = parsed_analysis

        else:
            logger.warning(f"[Pipeline] Frame #{frame_num}: Local OCR Failed/Empty. Prompting for realignment.")
            analysis = {"error": "realign"}

        # Mark all detections as tracked/analyzed
        for d in detections:
            self.detector.mark_analyzed(d)

        # Generate TTS text from template
        if analysis.get("error") == "realign":
            tts_text = "I couldn't read the text clearly. Please realign the camera."
        else:
            tts_text = self._format_tts(analysis)
        
        self._last_analysis = analysis
        self._last_tts_text = tts_text

        elapsed_ms = (time.perf_counter() - frame_start) * 1000
        self._total_frame_time_ms += elapsed_ms

        logger.info(
            f"[Pipeline] Frame #{frame_num}: COMPLETE in {elapsed_ms:.0f}ms — "
            f"dets={len(detections)}, tts_len={len(tts_text)}, "
            f"analysis_keys={list(analysis.keys()) if analysis else 'none'}"
        )

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
        voice_start = time.perf_counter()
        self._total_voice_cmds += 1
        cmd_num = self._total_voice_cmds

        logger.info(
            f"[Pipeline] Voice #{cmd_num}: {len(audio)} bytes "
            f"({len(audio) / (16000 * 2) * 1000:.0f}ms audio)"
        )

        # STT
        with latency_tracker("stt_transcribe", logger):
            text = await self.stt.transcribe(audio)

        if not text:
            elapsed_ms = (time.perf_counter() - voice_start) * 1000
            logger.warning(
                f"[Pipeline] Voice #{cmd_num}: STT returned empty ({elapsed_ms:.0f}ms)"
            )
            return {"action": "none", "text": "", "message": "Could not understand audio"}

        logger.info(f"[Pipeline] Voice #{cmd_num}: STT result: \"{text}\"")

        # Intent Classification
        with latency_tracker("intent_classify", logger):
            result = self.intent_classifier.classify(text)

        logger.info(
            f"[Pipeline] Voice #{cmd_num}: "
            f"'{text}' → intent={result.intent.value} "
            f"(conf={result.confidence:.2f}, params={result.params})"
        )

        # Dispatch
        dispatch_result = await self._dispatch_intent(result)

        elapsed_ms = (time.perf_counter() - voice_start) * 1000
        self._total_voice_time_ms += elapsed_ms
        logger.info(
            f"[Pipeline] Voice #{cmd_num}: COMPLETE in {elapsed_ms:.0f}ms — "
            f"action={dispatch_result.get('action')}"
        )

        return dispatch_result

    async def _dispatch_intent(self, result: IntentResult) -> dict[str, Any]:
        """Route intent to appropriate action."""
        match result.intent:
            case Intent.SCAN:
                return {
                    "action": "scan",
                    "text": result.raw_text,
                    "message": "Scan triggered",
                    "tts_ack": "Scanning now.",  # Frontend can speak this immediately
                }

            case Intent.START_CONTINUOUS:
                self._continuous_mode = True
                # Clear tracked objects for fresh continuous scan
                self.detector.clear_tracked()
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
        logger.debug(f"[Pipeline] TTS speak: {len(text)} chars")
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

    @property
    def stats(self) -> dict:
        """Return pipeline statistics for debugging."""
        return {
            "total_frames": self._total_frames,
            "avg_frame_ms": round(
                self._total_frame_time_ms / self._total_frames, 1
            )
            if self._total_frames
            else 0,
            "total_voice_cmds": self._total_voice_cmds,
            "avg_voice_ms": round(
                self._total_voice_time_ms / self._total_voice_cmds, 1
            )
            if self._total_voice_cmds
            else 0,
            "skipped_by_tracking": self._skipped_by_tracking,
            "skipped_by_state": self._skipped_by_state,
            "current_mode": self.config.current_mode,
            "continuous_mode": self._continuous_mode,
        }
