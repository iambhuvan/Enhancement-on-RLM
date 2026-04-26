"""
Baselines for RLM vs ERLM evaluation.

Exports:
    VanillaBaseline    — single-call full-document QA
    CompactionBaseline — parallel chunk extraction then answer
    ReActBaseline      — iterative Thought→Action→Observation loop
    BaselineResult     — shared result dataclass
"""

from .vanilla import VanillaBaseline, BaselineResult
from .compaction import CompactionBaseline
from .react import ReActBaseline

__all__ = [
    "BaselineResult",
    "VanillaBaseline",
    "CompactionBaseline",
    "ReActBaseline",
]
