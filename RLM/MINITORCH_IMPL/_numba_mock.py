"""
_numba_mock.py — Patches sys.modules so MiniTorch can be imported when
numba is incompatible with the installed NumPy (NumPy ≥ 2.2 breaks
numba < 0.60).

Without this patch:  numba/__init__.py raises ImportError at load time.
With this patch:     numba is replaced by a thin mock that:
  • Makes @njit / @jit identity decorators (functions run as pure Python).
  • Replaces prange with range (loop parallelism is dropped; correctness kept).
  • Provides numba.cuda stubs so device checks return False.
  • Provides numba.typed (ListType etc.) as stubs.

The result is MiniTorch running on its SimpleOps CPU backend without any
JIT acceleration.  For the tiny benchmark model (n_embd=128, 4 layers) this
is fast enough.

Call install_mock() ONCE before the first `import minitorch`.
"""
from __future__ import annotations

import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def install_mock() -> None:
    """Insert the numba mock into sys.modules if numba is broken/absent."""
    try:
        import numba  # type: ignore  # noqa: F401
        return  # real numba works — nothing to do
    except (ImportError, Exception):
        pass

    mock = _build_mock()
    sys.modules["numba"]          = mock
    sys.modules["numba.core"]     = mock
    sys.modules["numba.cuda"]     = mock.cuda      # type: ignore[attr-defined]
    sys.modules["numba.typed"]    = mock.typed     # type: ignore[attr-defined]
    sys.modules["numba.np"]       = types.ModuleType("numba.np")
    sys.modules["numba.np.numpy_support"] = types.ModuleType("numba.np.numpy_support")


# ---------------------------------------------------------------------------
# Mock construction
# ---------------------------------------------------------------------------

def _identity_decorator(*args: Any, **kwargs: Any):
    """Return a decorator that leaves the decorated function unchanged."""
    if args and callable(args[0]):
        # Called as @njit without parens
        return args[0]
    # Called as @njit(nopython=True, ...) — return a decorator
    def _wrap(fn: Any) -> Any:
        return fn
    return _wrap


class _CudaStub:
    """Stub that makes all CUDA availability checks return False."""

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_cuda_array(x: Any) -> bool:
        return False

    @staticmethod
    def to_device(x: Any) -> Any:
        return x

    @staticmethod
    def jit(*args: Any, **kwargs: Any):
        return _identity_decorator(*args, **kwargs)

    # Allow attribute access without AttributeError
    def __getattr__(self, name: str) -> Any:
        return _identity_decorator


class _TypedStub:
    """Stub for numba.typed (List, Dict, etc.)."""

    class List:
        @staticmethod
        def empty_list(*args: Any, **kwargs: Any):
            return []

    class Dict:
        pass

    def __getattr__(self, name: str) -> Any:
        return type(name, (), {})


def _build_mock() -> types.ModuleType:
    mock = types.ModuleType("numba")

    # Core decorators — pure-Python pass-through
    mock.njit    = _identity_decorator     # type: ignore[attr-defined]
    mock.jit     = _identity_decorator     # type: ignore[attr-defined]
    mock.vectorize = _identity_decorator   # type: ignore[attr-defined]
    mock.guvectorize = _identity_decorator # type: ignore[attr-defined]
    mock.stencil = _identity_decorator     # type: ignore[attr-defined]

    # prange → plain range (drop parallelism, keep correctness)
    mock.prange  = range                   # type: ignore[attr-defined]

    # CUDA stub
    mock.cuda    = _CudaStub()             # type: ignore[attr-defined]

    # typed stub
    mock.typed   = _TypedStub()            # type: ignore[attr-defined]

    # Common type aliases used as annotations
    for _t in ("int32", "int64", "float32", "float64", "boolean",
               "uint8", "uint32", "uint64"):
        setattr(mock, _t, _t)

    # Allow arbitrary non-dunder attribute access without crashing.
    # IMPORTANT: use a class-level __getattr__ on a custom ModuleType subclass
    # so that dunder attrs (__file__, __spec__, etc.) are NOT intercepted —
    # torch's inspect machinery requires these to be real strings / None.
    class _NumbaModule(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _identity_decorator

    mock.__class__ = _NumbaModule          # type: ignore[attr-defined]

    return mock
