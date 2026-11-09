"""Backward-compatible exports for legacy scripts.

The implementation lives in src.pipelines.training_utils. Retain this
module to support existing CLI imports and external experiment scripts.
"""

from src.pipelines.training_utils import (
    make_group_split,
    make_groups,
    resolve_device,
)

__all__ = ["make_group_split", "make_groups", "resolve_device"]
