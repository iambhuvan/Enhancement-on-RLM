"""
AnthropicVertexClient — Claude via Vertex AI (Application Default Credentials)
===============================================================================
Uses `gcloud auth application-default login` — no ANTHROPIC_API_KEY needed.

Usage:
    client = AnthropicVertexClient(project_id="newproject-490100")
    response = client.completion("Explain this code.")
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any

# Add BASELINE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "BASELINE"))

import anthropic                                           # noqa: E402
from rlm.clients.base_lm import BaseLM                    # noqa: E402
from rlm.clients.anthropic import AnthropicClient         # noqa: E402

_DEFAULT_PROJECT = "newproject-490100"
_DEFAULT_REGION  = "global"             # Vertex AI docs use region="global" for Anthropic
_DEFAULT_MODEL   = "claude-haiku-4-5"   # No @version suffix on Vertex AI


class AnthropicVertexClient(AnthropicClient):
    """
    Claude via Vertex AI using Application Default Credentials.

    Inherits all completion/tracking logic from AnthropicClient but initialises
    the Anthropic SDK in Vertex AI mode (AnthropicVertex) instead of using
    a direct API key.

    Parameters
    ----------
    project_id:
        Google Cloud project ID (default: ``"newproject-490100"``).
    region:
        Vertex AI region for Anthropic models (default: ``"global"``).
    model_name:
        Claude model name on Vertex AI (default: ``"claude-haiku-4-5"``).
    max_tokens:
        Maximum output tokens (default: 32768).
    """

    def __init__(
        self,
        project_id: str = _DEFAULT_PROJECT,
        region: str = _DEFAULT_REGION,
        model_name: str = _DEFAULT_MODEL,
        max_tokens: int = 32768,
        **kwargs,
    ) -> None:
        # Skip AnthropicClient.__init__ (requires api_key); call BaseLM directly.
        BaseLM.__init__(self, model_name=model_name, **kwargs)

        self.client = anthropic.AnthropicVertex(
            project_id=project_id,
            region=region,
        )
        try:
            self.async_client = anthropic.AsyncAnthropicVertex(
                project_id=project_id,
                region=region,
            )
        except AttributeError:
            # Older SDK versions may not have AsyncAnthropicVertex
            self.async_client = self.client

        self.model_name = model_name
        self.max_tokens = max_tokens

        # Per-model usage tracking (mirrors AnthropicClient)
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
