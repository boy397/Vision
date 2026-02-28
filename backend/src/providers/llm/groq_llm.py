"""Groq LLM provider for vision analysis and chat."""

from __future__ import annotations

import base64
import json
import logging
import os

from backend.src.providers.base import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """Groq API implementation (OpenAI-compatible)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None
        # Groq Llama 3 Vision model
        self._model_name = config.get("model", "llama-3.2-90b-vision-preview")

    def _get_client(self):
        """Lazy-init Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY env var not set")
            self._client = AsyncGroq(api_key=api_key)
        return self._client

    async def analyze_image(self, image: bytes, prompt: str) -> dict:
        """Send image + prompt to Groq, return structured JSON."""
        try:
            client = self._get_client()
            b64 = base64.b64encode(image).decode("utf-8")
            
            # Use JSON mode if supported, otherwise just parse the text.
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 512),
                response_format={"type": "json_object"}
            )

            text = response.choices[0].message.content
            return self._parse_response(text)

        except Exception as e:
            logger.error(f"Groq Vision error: {e}")
            return {"error": str(e)}

    async def chat(self, context: dict, question: str) -> str:
        """Follow-up chat with context from previous detection."""
        try:
            client = self._get_client()

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Provide concise and accurate answers."
                },
                {
                    "role": "user",
                    "content": f"Previous detection result:\n{json.dumps(context, indent=2)}\n\nUser asks: {question}"
                }
            ]

            response = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 512),
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq Chat error: {e}")
            return f"Sorry, I couldn't process that question. Error: {e}"

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse JSON from Groq response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json"):
                lines = lines[1:]
            else:
                lines = lines[1:]
            if len(lines) > 0 and lines[-1] == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_response": text, "parse_error": True}
