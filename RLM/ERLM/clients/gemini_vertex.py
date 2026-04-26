"""
GeminiVertexClient — Gemini via Vertex AI (Application Default Credentials)
============================================================================
Uses `gcloud auth application-default login` — no GEMINI_API_KEY needed.

Usage:
    client = GeminiVertexClient(project="newproject-490100")
    response = client.completion("Explain this code.")
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any

# Add BASELINE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

from google import genai                                    # noqa: E402
from google.genai import types                             # noqa: E402
from rlm.clients.base_lm import BaseLM                    # noqa: E402
from rlm.clients.gemini import GeminiClient               # noqa: E402

_DEFAULT_PROJECT  = "newproject-490100"
_DEFAULT_LOCATION = "us-central1"
_DEFAULT_MODEL    = "gemini-2.0-flash-001"


class GeminiVertexClient(GeminiClient):
    """
    Gemini via Vertex AI using Application Default Credentials.

    Inherits all completion/tracking logic from GeminiClient but initialises
    the google-genai SDK in Vertex AI mode (vertexai=True) instead of using
    an API key.

    Parameters
    ----------
    project:
        Google Cloud project ID (default: ``"newproject-490100"``).
    location:
        Vertex AI region (default: ``"us-central1"``).
    model_name:
        Gemini model ID (default: ``"gemini-2.0-flash-001"``).
    """

    def __init__(
        self,
        project: str = _DEFAULT_PROJECT,
        location: str = _DEFAULT_LOCATION,
        model_name: str = _DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        # Skip GeminiClient.__init__ (requires api_key); call BaseLM directly.
        BaseLM.__init__(self, model_name=model_name, **kwargs)

        http_options = types.HttpOptions(timeout=int(self.timeout * 1000))
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=http_options,
        )
        self.model_name = model_name

        # Per-model usage tracking (mirrors GeminiClient)
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
