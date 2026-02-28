"""Google Gemini LLM provider for vision analysis and chat."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from backend.src.providers.base import BaseLLM

logger = logging.getLogger(__name__)


class GoogleLLM(BaseLLM):
    """Google Gemini Vision + Chat implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None
        self._model_name = config.get("model", "gemini-2.0-flash")

    def _get_client(self):
        """Lazy-init Gemini client."""
        if self._client is None:
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY env var not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def analyze_image(self, image: bytes, prompt: str) -> dict:
        """Send image + prompt to Gemini Vision, return structured JSON."""
        try:
            client = self._get_client()
            from google.genai import types

            b64 = base64.b64encode(image).decode()

            response = client.models.generate_content(
                model=self._model_name,
                contents=[
                    types.Content(parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=b64
                        )),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    temperature=self.config.get("temperature", 0.3),
                    max_output_tokens=self.config.get("max_tokens", 512),
                    response_mime_type="application/json",
                ),
            )

            return self._parse_response(response.text)

        except Exception as e:
            logger.error(f"Gemini Vision error: {e}")
            return {"error": str(e)}

    async def chat(self, context: dict, question: str) -> str:
        """Follow-up chat with context from previous detection."""
        try:
            client = self._get_client()
            from google.genai import types

            messages = [
                f"Previous detection result:\n{json.dumps(context, indent=2)}",
                f"User asks: {question}",
            ]

            response = client.models.generate_content(
                model=self._model_name,
                contents="\n\n".join(messages),
                config=types.GenerateContentConfig(
                    temperature=self.config.get("temperature", 0.3),
                    max_output_tokens=self.config.get("max_tokens", 512),
                ),
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini Chat error: {e}")
            return f"Sorry, I couldn't process that question. Error: {e}"

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse JSON from Gemini response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_response": text, "parse_error": True}
