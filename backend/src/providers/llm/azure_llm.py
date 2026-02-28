"""Azure OpenAI LLM provider for vision analysis and chat (optional)."""

from __future__ import annotations

import base64
import json
import logging
import os

from backend.src.providers.base import BaseLLM

logger = logging.getLogger(__name__)


class AzureLLM(BaseLLM):
    """Azure OpenAI Vision + Chat implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.get("model", "gpt-4o")

    def _get_client(self):
        """Lazy-init Azure OpenAI client."""
        if self._client is None:
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=self.config.get("api_version", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            )
        return self._client

    async def analyze_image(self, image: bytes, prompt: str) -> dict:
        """Send image + prompt to Azure OpenAI Vision."""
        try:
            client = self._get_client()
            b64 = base64.b64encode(image).decode()

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 512),
                response_format={"type": "json_object"},
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Azure Vision error: {e}")
            return {"error": str(e)}

    async def chat(self, context: dict, question: str) -> str:
        """Follow-up chat with context."""
        try:
            client = self._get_client()

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Previous detection result:\n{json.dumps(context, indent=2)}",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 512),
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Azure Chat error: {e}")
            return f"Sorry, I couldn't process that question. Error: {e}"

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse JSON from Azure response."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"raw_response": text, "parse_error": True}
