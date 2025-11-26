from __future__ import annotations

from ..language_detector_registry import LanguageDetectorSpec, register_language_detector
from ..language_utils import (
    ensure_fasttext_language_model,
    detect_language_fasttext,
)


def _prepare_fasttext(verbose: bool) -> None:
    try:
        ensure_fasttext_language_model()
    except RuntimeError as exc:
        raise RuntimeError(
            f"fastText language detector unavailable: {exc}"
        ) from exc


def _detect_fasttext(
    text: str,
    min_length: int,
    confidence_threshold: float,
    verbose: bool,
):
    if not text or len(text.strip()) < min_length:
        if verbose:
            print("[flexipipe] fastText detector skipped: input too short.")
        return None
    ensure_fasttext_language_model()
    try:
        return detect_language_fasttext(
            text,
            min_length=min_length,
            confidence_threshold=confidence_threshold,
        )
    except RuntimeError as exc:
        if verbose:
            print(f"[flexipipe] fastText detector error: {exc}")
        return None


register_language_detector(
    LanguageDetectorSpec(
        name="fasttext",
        description="fastText lid.176 language detector (requires fasttext + lid.176.ftz)",
        detect=_detect_fasttext,
        prepare=_prepare_fasttext,
        is_default=True,
    )
)

