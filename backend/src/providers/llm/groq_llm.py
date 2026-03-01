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
        # Active model from config.yml (switchable at runtime)
        self._model_name = config.get("model", "llama-3.2-11b-vision-preview")

    def _get_client(self):
        """Lazy-init Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY env var not set")
            self._client = AsyncGroq(api_key=api_key)
        return self._client

    async def analyze_image(self, image: bytes | list[bytes], prompt: str) -> dict:
        """Send image(s) + prompt to Groq, return structured JSON.

        Automatically falls back to text-only mode for non-vision models (e.g. Kimi K2).
        Non-vision models require content to be a plain string, not a content-parts array.
        """
        try:
            client = self._get_client()

            # Check if this model supports vision (image inputs)
            # config.get("vision", True) — defaults True; set False for text-only models
            is_vision_model = self.config.get("vision", True)

            if is_vision_model:
                # Normalize to list for uniform handling
                images = image if isinstance(image, list) else [image]

                content_parts: list[dict] = [{"type": "text", "text": prompt}]
                for img in images:
                    b64 = base64.b64encode(img).decode("utf-8")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    })
                content: str | list[dict] = content_parts
            else:
                # Text-only model — send prompt as plain string, no image parts
                logger.info(
                    f"[GroqLLM] Model '{self._model_name}' is text-only — "
                    "skipping image attachment, sending prompt only."
                )
                content = prompt + "\n\n(No image available for this text-only model. Return a JSON object with an explanatory error field.)"

            response = await client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content,
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
