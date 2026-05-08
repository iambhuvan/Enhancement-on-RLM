from typing import Any

from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend

load_dotenv()


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Routes a specific backend and the args (as a dict) to the appropriate client if supported.
    Currently supported backends: ['openai']
    """
    if backend == "openai":
        from rlm.clients.openai import OpenAIClient

        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        from rlm.clients.openai import OpenAIClient

        assert "base_url" in backend_kwargs, (
            "base_url is required to be set to local vLLM server address for vLLM"
        )
        return OpenAIClient(**backend_kwargs)
    elif backend == "portkey":
        from rlm.clients.portkey import PortkeyClient

        return PortkeyClient(**backend_kwargs)
    elif backend == "openrouter":
        from rlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "vercel":
        from rlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "anthropic":
        from rlm.clients.anthropic import AnthropicClient

        return AnthropicClient(**backend_kwargs)
    elif backend == "gemini":
        from rlm.clients.gemini import GeminiClient

        return GeminiClient(**backend_kwargs)
    elif backend == "azure_openai":
        from rlm.clients.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(**backend_kwargs)
    elif backend == "ollama":
        import os, sys
        # OllamaClient lives in ERLM/clients/ — add that to path if needed
        _erlm_clients = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ERLM", "clients")
        if os.path.isdir(_erlm_clients) and _erlm_clients not in sys.path:
            sys.path.insert(0, _erlm_clients)
        from ollama import OllamaClient  # noqa: E402

        return OllamaClient(**{k: v for k, v in backend_kwargs.items()
                               if k in ("model_name", "base_url")})
    elif backend == "gemini25":
        import os, sys
        # Gemini25FlashClient lives in ERLM/clients/ — add that to path if needed
        _erlm_clients = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ERLM", "clients")
        if os.path.isdir(_erlm_clients) and _erlm_clients not in sys.path:
            sys.path.insert(0, _erlm_clients)
        from gemini25_flash import Gemini25FlashClient  # noqa: E402

        return Gemini25FlashClient(**{k: v for k, v in backend_kwargs.items()
                                     if k in ("api_key", "model_name")})
    elif backend == "gemini_vertex":
        import os, sys
        _erlm_clients = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ERLM", "clients")
        if os.path.isdir(_erlm_clients) and _erlm_clients not in sys.path:
            sys.path.insert(0, _erlm_clients)
        from gemini_vertex import GeminiVertexClient  # noqa: E402

        return GeminiVertexClient(**{k: v for k, v in backend_kwargs.items()
                                     if k in ("project", "location", "model_name")})
    elif backend == "gemini25_vertex":
        import os, sys
        _erlm_clients = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ERLM", "clients")
        if os.path.isdir(_erlm_clients) and _erlm_clients not in sys.path:
            sys.path.insert(0, _erlm_clients)
        from gemini25_vertex import Gemini25VertexClient  # noqa: E402

        return Gemini25VertexClient(**{k: v for k, v in backend_kwargs.items()
                                      if k in ("project", "location", "model_name")})
    elif backend == "anthropic_vertex":
        import os, sys
        _erlm_clients = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ERLM", "clients")
        if os.path.isdir(_erlm_clients) and _erlm_clients not in sys.path:
            sys.path.insert(0, _erlm_clients)
        from anthropic_vertex import AnthropicVertexClient  # noqa: E402

        return AnthropicVertexClient(**{k: v for k, v in backend_kwargs.items()
                                        if k in ("project_id", "region", "model_name", "max_tokens")})
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: ['openai', 'vllm', 'portkey', 'openrouter', "
            f"'anthropic', 'azure_openai', 'gemini', 'gemini25', 'gemini_vertex', 'gemini25_vertex', "
            f"'anthropic_vertex', 'vercel', 'ollama']"
        )
