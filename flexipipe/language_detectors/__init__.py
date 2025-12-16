"""Built-in language detectors."""

from __future__ import annotations

# Import built-in detectors to register them
from . import fasttext_detector as _fasttext_detector  # noqa: F401
from . import trigram_detector as _trigram_detector  # noqa: F401

