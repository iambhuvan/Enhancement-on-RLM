"""
ERLM — Enhanced Recursive Language Model
=========================================
Top-level package.  Import ``EnhancedRLM`` from here.

Example
-------
>>> from erlm import EnhancedRLM
>>> model = EnhancedRLM(enable_indexing=True, enable_budget=True)
"""

import os
import sys

# Ensure the ERLM directory itself is on sys.path so that sub-modules
# (optimisations.*) can be imported with a plain ``import optimisations.*``
# style regardless of how Python was invoked.
_ERLM_DIR = os.path.dirname(__file__)
if _ERLM_DIR not in sys.path:
    sys.path.insert(0, _ERLM_DIR)

from erlm import EnhancedRLM  # noqa: E402

__all__ = ["EnhancedRLM"]
