from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


DetectFn = Callable[[str, int, float, bool], Optional[Dict[str, Any]]]
PrepareFn = Callable[[bool], None]


@dataclass
class LanguageDetectorSpec:
    name: str
    description: str
    detect: DetectFn
    prepare: Optional[PrepareFn] = None
    is_default: bool = False


_LANGUAGE_DETECTORS: Dict[str, LanguageDetectorSpec] = {}
LANGUAGE_DETECTOR_DEFAULT = "fasttext"
_DETECTORS_IMPORTED = False


def register_language_detector(spec: LanguageDetectorSpec) -> None:
    global _DETECTORS_IMPORTED
    key = spec.name.lower()
    _LANGUAGE_DETECTORS[key] = spec
    # Once something registers, we know imports succeeded
    _DETECTORS_IMPORTED = True


def list_language_detectors(include_none: bool = True) -> Dict[str, LanguageDetectorSpec]:
    _ensure_language_detectors_loaded()
    detectors = dict(_LANGUAGE_DETECTORS)
    if include_none:
        detectors["none"] = LanguageDetectorSpec(
            name="none",
            description="Disable automatic language detection",
            detect=lambda text, min_length, confidence_threshold, verbose: None,
        )
    return detectors


def detect_language_with(
    name: str,
    text: str,
    *,
    min_length: int = 10,
    confidence_threshold: float = 0.5,
    verbose: bool = False,
) -> Optional[Dict[str, object]]:
    _ensure_language_detectors_loaded()
    key = name.lower()
    if key == "none":
        return None
    spec = _LANGUAGE_DETECTORS.get(key)
    if not spec:
        raise ValueError(f"Unknown language detector '{name}'")
    if spec.prepare:
        spec.prepare(verbose)
    return spec.detect(text, min_length, confidence_threshold, verbose)


def get_default_language_detector() -> str:
    _ensure_language_detectors_loaded()
    if LANGUAGE_DETECTOR_DEFAULT in _LANGUAGE_DETECTORS:
        return LANGUAGE_DETECTOR_DEFAULT
    if _LANGUAGE_DETECTORS:
        return next(iter(_LANGUAGE_DETECTORS.keys()))
    return "none"


def _ensure_language_detectors_loaded() -> None:
    global _DETECTORS_IMPORTED
    if _DETECTORS_IMPORTED:
        return
    try:  # pragma: no cover - import side effect
        from . import language_detectors as _language_detectors  # noqa: F401
    except ImportError:
        return
    else:
        _DETECTORS_IMPORTED = True

