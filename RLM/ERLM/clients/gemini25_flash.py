"""
Gemini 2.5 Flash Client — thinking disabled
============================================
Subclasses the BASELINE GeminiClient to force thinking_budget=0
so Gemini 2.5 Flash behaves like a non-thinking model.

Usage:
    client = Gemini25FlashClient(api_key="...")
    response = client.completion("Explain this code.")
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Add BASELINE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

from google.genai import types                          # noqa: E402
from rlm.clients.gemini import GeminiClient            # noqa: E402

_DEFAULT_MODEL = "gemini-2.5-flash"


class Gemini25FlashClient(GeminiClient):
    """
    Gemini 2.5 Flash with thinking_budget=0.

    Gemini 2.5 Flash has a thinking mode that can generate thousands of
    internal reasoning tokens. Setting thinking_budget=0 suppresses this,
    making the model behave like a fast non-thinking model while retaining
    its improved instruction-following capabilities over 2.0 Flash.

    Parameters
    ----------
    api_key:
        Google AI Studio API key (GEMINI_API_KEY env var is used as fallback).
    model_name:
        Gemini model ID (default: ``"gemini-2.5-flash"``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = _DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        contents, system_instruction = self._prepare_contents(prompt)
        model = model or self.model_name

        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._track_cost(response, model)
        return response.text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        contents, system_instruction = self._prepare_contents(prompt)
        model = model or self.model_name

        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._track_cost(response, model)
        return response.text
