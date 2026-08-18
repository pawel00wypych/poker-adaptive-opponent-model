"""Backward-compatible facade for the validation package.

New code should import from :mod:`src.evaluation.validation` directly.
"""

from src.evaluation import validation as _validation

__all__ = _validation.__all__

globals().update(
    {name: getattr(_validation, name) for name in __all__}
)
