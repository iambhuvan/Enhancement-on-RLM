"""
Gemini25VertexClient — Gemini 2.5 Flash Preview via Vertex AI, thinking disabled
==================================================================================
Subclasses GeminiVertexClient to force thinking_budget=0.

Usage:
    client = Gemini25VertexClient(project="newproject-490100")
    response = client.completion("Explain this code.")
"""

from __future__ import annotations

import os
import sys
from typing import Any

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "BASELINE"))
sys.path.insert(0, _THIS_DIR)  # so we can import sibling gemini_vertex

from google.genai import types                             # noqa: E402
from gemini_vertex import GeminiVertexClient               # noqa: E402

_DEFAULT_PROJECT  = "newproject-490100"
_DEFAULT_LOCATION = "us-central1"
_DEFAULT_MODEL    = "gemini-2.5-flash-preview-09-2025"


class Gemini25VertexClient(GeminiVertexClient):
    """
    Gemini 2.5 Flash Preview via Vertex AI with thinking_budget=0.

    Parameters
    ----------
    project:
        Google Cloud project ID (default: ``"newproject-490100"``).
    location:
        Vertex AI region (default: ``"us-central1"``).
    model_name:
        Gemini model ID (default: ``"gemini-2.5-flash-preview-09-2025"``).
    """

    def __init__(
        self,
        project: str = _DEFAULT_PROJECT,
        location: str = _DEFAULT_LOCATION,
        model_name: str = _DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(project=project, location=location, model_name=model_name, **kwargs)

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
