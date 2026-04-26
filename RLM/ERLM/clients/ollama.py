"""
Ollama Client for local Qwen3 inference
========================================
Wraps the BASELINE OpenAIClient to target a local Ollama server.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1.
No authentication is required.

Thinking mode is disabled via Ollama's native ``think: false`` parameter
passed in extra_body, which prevents Qwen3 from generating internal
chain-of-thought tokens.

Usage:
    client = OllamaClient(model_name="qwen3:8b")
    response = client.completion("What is 2+2?")
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Add BASELINE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

from rlm.clients.openai import OpenAIClient  # noqa: E402

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "qwen3:8b"


class OllamaClient(OpenAIClient):
    """
    LM client targeting a local Ollama server.

    Inherits token tracking and async support from OpenAIClient.
    Disables Qwen3 thinking mode via ``think: false`` in every request.

    Parameters
    ----------
    model_name:
        Ollama model tag (default: ``"qwen3:8b"``).
    base_url:
        Ollama OpenAI-compatible endpoint (default: ``http://localhost:11434/v1``).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = _OLLAMA_BASE_URL,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key="ollama",   # required by openai SDK; not validated by Ollama
            model_name=model_name,
            base_url=base_url,
            **kwargs,
        )

    def _inject_no_think(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepend /no_think to the system message (or insert one) to suppress Qwen3 thinking.

        Qwen3 in Ollama ignores think:false for complex multi-message prompts but
        reliably respects /no_think when it appears at the start of the system content.
        """
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            result = []
            for m in messages:
                if m.get("role") == "system" and "/no_think" not in m.get("content", ""):
                    m = dict(m)
                    m["content"] = "/no_think\n" + m["content"]
                result.append(m)
            return result
        return [{"role": "system", "content": "/no_think"}] + messages

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        messages = self._inject_no_think(messages)
        model = model or self.model_name
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"think": False},
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        messages = self._inject_no_think(messages)
        model = model or self.model_name
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"think": False},
        )
        self._track_cost(response, model)
        return response.choices[0].message.content
