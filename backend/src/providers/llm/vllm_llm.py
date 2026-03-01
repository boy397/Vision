"""Local vLLM provider for vision analysis and chat."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from backend.src.providers.base import BaseLLM

logger = logging.getLogger(__name__)


class VLLMLLM(BaseLLM):
    """Local vLLM API implementation (OpenAI-compatible)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None
        self._model_name = config.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
        self._api_base = config.get("api_base", "http://localhost:8001/v1")

    def _get_client(self):
        """Lazy-init OpenAI Async API client for vLLM."""
        if self._client is None:
            from openai import AsyncOpenAI
            
            self._client = AsyncOpenAI(
                base_url=self._api_base,
                api_key="EMPTY"  # vLLM usually doesn't require an API key by default
            )
        return self._client

    async def analyze_image(self, image: bytes | list[bytes], prompt: str) -> dict[str, Any]:
        """Send image(s) + prompt to vLLM, return structured JSON."""
        try:
            client = self._get_client()

            # Normalize to list for uniform handling
            images = image if isinstance(image, list) else [image]

            content_parts: list[dict] = [
                {"type": "text", "text": prompt + "\n\nOutput Valid JSON only."}
            ]
            for img in images:
                b64 = base64.b64encode(img).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            # Note: We do not pass response_format={"type": "json_object"} 
            # to maximize compatibility with vanilla vLLM deployments.
            # We rely on prompting and robust parsing instead.
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant. You must output your response in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
                temperature=self.config.get("temperature", 0.0), # Low temperature for JSON consistency
                max_tokens=self.config.get("max_tokens", 512),
            )

            text = response.choices[0].message.content
            return self._parse_response(text)

        except Exception as e:
            logger.error(f"vLLM Vision error: {e}")
            return {"error": str(e), "provider": "vllm"}

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

            return response.choices[0].message.content or "No response generated."

        except Exception as e:
            logger.error(f"vLLM Chat error: {e}")
            return f"Sorry, I couldn't process that question. Error: {e}"

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse JSON from response, handling markdown code blocks."""
        if not text:
            return {"raw_response": "", "parse_error": True}
            
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 1 and lines[0].startswith("```"):
                lines = lines[1:]
            
            # Remove trailing codeblocks
            while len(lines) > 0 and lines[-1].strip() == "```":
                lines = lines[:-1]
                
            text = "\n".join(lines).strip()
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback string manipulation just in case standard json load fails on minor issues
            logger.warning(f"JSON decode failed. Raw text: {text[:200]}...")
            return {"raw_response": text, "parse_error": True}
