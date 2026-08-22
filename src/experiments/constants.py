"""Model-type vocabulary shared by the experiment CLIs.

``MODEL_TYPE_GENERAL_POLICY`` and ``MODEL_TYPE_SPECIALIST`` are re-exported
from :mod:`src.training.constants` so that experiment scripts have a single
import site for everything that names a model.
"""

from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
)
from src.training.constants import (
    MODEL_TYPE_GENERAL_POLICY,
    MODEL_TYPE_SPECIALIST,
)

MODEL_TYPES = (
    MODEL_TYPE_GENERAL_POLICY,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)

__all__ = [
    "MODEL_TYPES",
    "MODEL_TYPE_GENERAL_POLICY",
    "MODEL_TYPE_SPECIALIST",
]
