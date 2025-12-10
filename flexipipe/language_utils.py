from __future__ import annotations

import os
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_NAME = "lid.176.ftz"
FASTTEXT_ENV_VAR = "FLEXIPIPE_FASTTEXT_MODEL"

_FASTTEXT_MODEL = None
_FASTTEXT_MODEL_PATH: Optional[Path] = None
_FASTTEXT_IMPORT_ERROR: Optional[Exception] = None

try:
    import pycountry  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pycountry = None  # type: ignore

from .model_storage import get_cache_dir

LANGUAGE_FIELD_ISO = "language_iso"
LANGUAGE_FIELD_NAME = "language_name"
LANGUAGE_FIELD_GLOTTO = "language_glotto"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Project names that should be removed from language names (these are corpus/treebank project names, not language qualifiers)
_PROJECT_NAMES = {
    "proiel",  # PROIEL project (Ancient Greek, Old Church Slavonic, Gothic, Latin)
    "ittb",    # ITTB project (Latin)
    "vedic",   # Vedic Sanskrit project
    "ufal",    # UFAL project (various languages)
}

# Special cases where parenthetical text should be moved to the front (language qualifiers, not project names)
_LANGUAGE_QUALIFIER_MAP = {
    "sorbian (upper)": "Upper Sorbian",
    "upper sorbian": "Upper Sorbian",
    "sorbian (lower)": "Lower Sorbian",
    "lower sorbian": "Lower Sorbian",
}


def _normalize_name(value: str) -> str:
    return _NON_ALNUM.sub("", value.lower())


def clean_language_name(language_name: str) -> str:
    """
    Clean a language name by removing project names in parentheses and handling special cases.
    
    Examples:
        "Old Church Slavonic (PROIEL)" -> "Old Church Slavonic"
        "Sorbian (Upper)" -> "Upper Sorbian"
        "Ancient Greek (PROIEL)" -> "Ancient Greek"
        "Sanskrit (Vedic)" -> "Sanskrit"
    
    Args:
        language_name: The language name to clean
        
    Returns:
        Cleaned language name
    """
    if not language_name:
        return language_name
    
    # Check for special cases first (language qualifiers that should be moved to front)
    name_lower = language_name.lower().strip()
    if name_lower in _LANGUAGE_QUALIFIER_MAP:
        return _LANGUAGE_QUALIFIER_MAP[name_lower]
    
    # Check if it matches a special case pattern (e.g., "Sorbian (Upper)")
    for pattern, replacement in _LANGUAGE_QUALIFIER_MAP.items():
        if name_lower == pattern:
            return replacement
    
    # Remove parenthetical project names
    # Pattern: "Language Name (PROJECT)" -> "Language Name"
    paren_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", language_name.strip())
    if paren_match:
        base_name = paren_match.group(1).strip()
        paren_content = paren_match.group(2).strip().lower()
        
        # Check if it's a project name (should be removed)
        if paren_content in _PROJECT_NAMES:
            return base_name
        
        # Check if it's a language qualifier that should be moved to front
        # This handles cases like "Sorbian (Upper)" -> "Upper Sorbian"
        full_pattern = f"{base_name.lower()} ({paren_content})"
        if full_pattern in _LANGUAGE_QUALIFIER_MAP:
            return _LANGUAGE_QUALIFIER_MAP[full_pattern]
        
        # If it's not a known project name and not a known qualifier, keep it as-is
        # (might be a legitimate part of the language name)
        return language_name.strip()
    
    return language_name.strip()


if pycountry:
    _LANGUAGE_BY_CODE = {}
    _LANGUAGE_BY_NAME = {}
    for _lang in pycountry.languages:
        fields = getattr(_lang, "_fields", {})
        for attr in ("alpha_2", "alpha_3", "bibliographic", "terminology"):
            code = fields.get(attr)
            if code:
                _LANGUAGE_BY_CODE.setdefault(code.lower(), _lang)
        for attr in ("name", "common_name", "inverted_name"):
            name_val = fields.get(attr)
            if name_val:
                _LANGUAGE_BY_NAME.setdefault(_normalize_name(name_val), _lang)
else:  # pragma: no cover - fallback when pycountry missing
    _LANGUAGE_BY_CODE = {}
    _LANGUAGE_BY_NAME = {}


def _lookup_language_by_code(code: str):
    if not pycountry:
        return None
    code_key = code.lower()
    if code_key in _LANGUAGE_BY_CODE:
        return _LANGUAGE_BY_CODE[code_key]
    try:
        return pycountry.languages.lookup(code)
    except LookupError:
        return None


def _lookup_language_by_name(name: str):
    if not pycountry:
        return None
    normalized = _normalize_name(name)
    if normalized in _LANGUAGE_BY_NAME:
        return _LANGUAGE_BY_NAME[normalized]
    variants = {
        name,
        name.replace("_", " "),
        name.replace("_", " ").title(),
        name.replace("_", " ").lower(),
    }
    for variant in variants:
        try:
            return pycountry.languages.lookup(variant)
        except LookupError:
            continue
    for key, lang in _LANGUAGE_BY_NAME.items():
        if key.startswith(normalized) or normalized.startswith(key):
            return lang
    for key, lang in _LANGUAGE_BY_NAME.items():
        if normalized in key:
            return lang
    return None


def standardize_language_metadata(
    language_code: Optional[str],
    language_name: Optional[str],
) -> dict:
    iso_code = None
    resolved_name = language_name
    lang_obj = None

    code_candidate = (language_code or "").strip()
    if code_candidate and code_candidate.lower() not in {"xx", "und", "unknown"}:
        lang_obj = _lookup_language_by_code(code_candidate)

    if not lang_obj and language_name:
        lang_obj = _lookup_language_by_name(language_name)

    if lang_obj:
        fields = getattr(lang_obj, "_fields", {})
        iso_code = fields.get("alpha_2") or fields.get("alpha_3")
        if not resolved_name:
            resolved_name = fields.get("name")
    elif code_candidate:
        iso_code = code_candidate.lower()

    return {
        LANGUAGE_FIELD_ISO: iso_code.lower() if iso_code else None,
        LANGUAGE_FIELD_NAME: resolved_name,
        LANGUAGE_FIELD_GLOTTO: None,
    }


def normalize_language_value(value: Optional[str]) -> str:
    if not value:
        return ""
    return _normalize_name(value)


def resolve_language_query(value: str) -> Dict[str, Set[str]]:
    iso_candidates: Set[str] = set()
    name_candidates: Set[str] = set()
    normalized_candidates: Set[str] = set()
    raw_normalized = normalize_language_value(value)

    # If this value isn't known to the current mapping cache, force a reload (with refresh)
    # so newly added languages are picked up without needing a separate CLI refresh.
    from .language_mapping import _LANGUAGE_BY_CODE, _LANGUAGE_BY_NAME, _build_language_mappings
    lowered = value.strip().lower() if value else ""
    if lowered and lowered not in _LANGUAGE_BY_CODE and lowered not in _LANGUAGE_BY_NAME and raw_normalized not in _LANGUAGE_BY_NAME:
        _build_language_mappings(force_reload=True, refresh_cache=True, verbose=False)
    
    if raw_normalized:
        normalized_candidates.add(raw_normalized)

    if value:
        if len(lowered) in {2, 3} and lowered.isalpha():
            iso_candidates.add(lowered)

    lang_obj = _lookup_language_by_code(value) if value else None
    if not lang_obj and value:
        lang_obj = _lookup_language_by_name(value)

    if lang_obj:
        fields = getattr(lang_obj, "_fields", {})
        for attr in ("alpha_2", "alpha_3"):
            code = fields.get(attr)
            if code:
                iso_candidates.add(code.lower())
        for attr in ("name", "common_name", "inverted_name"):
            name_val = fields.get(attr)
            if name_val:
                name_candidates.add(name_val)
                normalized = normalize_language_value(name_val)
                if normalized:
                    normalized_candidates.add(normalized)

    if value:
        name_candidates.add(value)

    return {
        "iso": iso_candidates,
        "names": name_candidates,
        "normalized": normalized_candidates,
        "raw": value,
        "raw_normalized": raw_normalized,
    }


def language_matches_entry(
    entry: Dict[str, Any],
    query: Dict[str, Set[str]],
    *,
    allow_fuzzy: bool = False,
    fuzzy_threshold: float = 0.65,
) -> bool:
    iso_value = (entry.get(LANGUAGE_FIELD_ISO) or "").lower()
    name_value = entry.get(LANGUAGE_FIELD_NAME) or ""
    display_value = entry.get("language_display") or ""
    normalized_name = normalize_language_value(name_value)
    normalized_display = normalize_language_value(display_value)

    if iso_value and iso_value in query["iso"]:
        return True
    if normalized_name and normalized_name in query["normalized"]:
        return True
    if normalized_display and normalized_display in query["normalized"]:
        return True

    for candidate in query["names"]:
        if candidate and candidate.lower() == name_value.lower():
            return True
        if candidate and candidate.lower() == display_value.lower():
            return True

    raw_norm = query.get("raw_normalized") or ""

    if allow_fuzzy and raw_norm and normalized_name:
        ratio = SequenceMatcher(None, normalized_name, raw_norm).ratio()
        if ratio >= fuzzy_threshold:
            return True
    if allow_fuzzy and raw_norm and normalized_display:
        ratio = SequenceMatcher(None, normalized_display, raw_norm).ratio()
        if ratio >= fuzzy_threshold:
            return True

    return False


def build_model_entry(
    backend: str,
    model_name: str,
    *,
    language_code: Optional[str] = None,
    language_name: Optional[str] = None,
    preferred: bool = False,
    **extra_fields: Any,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "backend": backend,
        "model": model_name,
    }
    entry.update(standardize_language_metadata(language_code, language_name))
    if preferred:
        entry["preferred"] = True
    for key, value in extra_fields.items():
        if value is not None:
            entry[key] = value
    return entry


def is_standardized_entry(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        "backend" in entry
        and "model" in entry
        and LANGUAGE_FIELD_ISO in entry
        and LANGUAGE_FIELD_NAME in entry
    )


def cache_entries_standardized(cache: Dict[str, Any]) -> bool:
    if not cache:
        return False
    return all(is_standardized_entry(value) for value in cache.values() if isinstance(value, dict))


def _get_fasttext_model_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    env_override = os.environ.get(FASTTEXT_ENV_VAR)
    if env_override:
        path = Path(env_override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    cache_dir = get_cache_dir() / "fasttext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / FASTTEXT_MODEL_NAME


def _download_fasttext_model(path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    print(f"[flexipipe] Downloading fastText language model to {path}...")
    try:
        with urllib.request.urlopen(FASTTEXT_MODEL_URL) as response, open(tmp_path, "wb") as out:
            out.write(response.read())
        tmp_path.replace(path)
    except Exception as exc:  # pragma: no cover - network errors
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download fastText language model: {exc}") from exc


def _ensure_fasttext_model(model_path: Optional[Path] = None):
    global _FASTTEXT_MODEL, _FASTTEXT_MODEL_PATH, _FASTTEXT_IMPORT_ERROR

    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL
    if _FASTTEXT_IMPORT_ERROR:
        raise RuntimeError(
            "fasttext library failed to import previously. Install it with 'pip install fasttext-numpy2'."
        ) from _FASTTEXT_IMPORT_ERROR
    try:
        import fasttext  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        _FASTTEXT_IMPORT_ERROR = exc
        raise RuntimeError(
            "Language detection requires the 'fasttext' package. Install it with 'pip install fasttext-numpy2'."
        ) from exc

    path = model_path or _get_fasttext_model_path()
    _FASTTEXT_MODEL_PATH = path
    if not path.exists():
        _download_fasttext_model(path)
    _FASTTEXT_MODEL = fasttext.load_model(str(path))
    return _FASTTEXT_MODEL


def ensure_fasttext_language_model(force_download: bool = False) -> Path:
    """
    Ensure the fastText language model is present locally.

    Returns path to the model file.
    """
    path = _get_fasttext_model_path()
    if force_download and path.exists():
        try:
            path.unlink()
        except (OSError, PermissionError):
            pass
    _ensure_fasttext_model(path)
    return path


def get_language_candidates(
    text: str,
    *,
    min_length: int = 50,
    top_k: int = 3,
    model_path: Optional[str] = None,
) -> list[Dict[str, Any]]:
    if not text:
        return []
    cleaned = " ".join(text.strip().split())
    if len(cleaned) > 20000:
        cleaned = cleaned[:20000]
    if len(cleaned) < max(10, min_length):
        return []

    model = _ensure_fasttext_model(
        Path(model_path).expanduser() if model_path else None
    )
    k = max(1, min(int(top_k), 10))
    try:
        labels, scores = model.predict(cleaned, k=k)
    except ValueError as exc:
        numpy_version = ""
        try:
            import numpy  # type: ignore

            numpy_version = getattr(numpy, "__version__", "")
        except Exception:
            pass
        hint = (
            "fastText prediction failed when converting outputs to numpy arrays. "
            "The legacy fasttext wheels are incompatible with NumPy 2.x. "
            "Install the compatibility build instead: pip install --upgrade fasttext-numpy2."
        )
        if numpy_version:
            hint += f" (Current NumPy version: {numpy_version})"
        raise RuntimeError(hint) from exc
    candidates: list[Dict[str, Any]] = []
    if labels is None or scores is None or len(labels) == 0 or len(scores) == 0:
        return candidates

    for label, score in zip(labels, scores):
        lang_code = label.replace("__label__", "")
        metadata = standardize_language_metadata(lang_code, None)
        iso = metadata.get(LANGUAGE_FIELD_ISO) or lang_code.lower()
        name = metadata.get(LANGUAGE_FIELD_NAME) or lang_code
        candidates.append(
            {
                "label": lang_code,
                "language_iso": iso,
                "language_name": name,
                "confidence": float(score),
                "model_path": str(_FASTTEXT_MODEL_PATH) if _FASTTEXT_MODEL_PATH else None,
            }
        )
    return candidates


def detect_language_fasttext(
    text: str,
    *,
    min_length: int = 50,
    confidence_threshold: float = 0.5,
    top_k: int = 1,
    model_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    candidates = get_language_candidates(
        text,
        min_length=min_length,
        top_k=top_k,
        model_path=model_path,
    )
    if not candidates:
        return None

    best = candidates[0]
    if best["confidence"] < confidence_threshold:
        return None

    result = dict(best)
    result["candidates"] = candidates
    return result

