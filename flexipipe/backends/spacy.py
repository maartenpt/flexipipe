"""Backend spec for the SpaCy backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from ..backend_spec import BackendSpec
from ..language_utils import LANGUAGE_FIELD_ISO, LANGUAGE_FIELD_NAME
from ..model_storage import get_backend_models_dir
from ..spacy_backend import (
    SPACY_DEFAULT_MODELS,
    SpacyBackend,
    get_spacy_model_entries,
    list_spacy_models,
)
def _resolve_spacy_model_name(
    model_name: Optional[str],
    language: Optional[str],
) -> Optional[str]:
    if model_name:
        return model_name
    if not language:
        return None
    lang_norm = language.lower()
    
    # Normalize language code to handle ISO-639-3 codes (e.g., "cat" -> "ca")
    from ..language_mapping import normalize_language_code, get_language_metadata
    lang_iso1, lang_iso2, lang_iso3 = normalize_language_code(lang_norm)
    # Collect all possible ISO codes for this language
    lang_iso_codes = {lang_iso1, lang_iso2, lang_iso3, lang_norm} - {None}
    
    try:
        entries, _, _ = get_spacy_model_entries(use_cache=True, refresh_cache=False, verbose=False)
    except Exception:
        entries = {}
    for key, info in entries.items():
        entry_iso = (info.get(LANGUAGE_FIELD_ISO) or "").lower()
        entry_name = (info.get(LANGUAGE_FIELD_NAME) or "").lower()
        # Check if any of the normalized ISO codes match
        if entry_iso in lang_iso_codes or lang_norm in {entry_iso, entry_name}:
            return key
        # Also check if entry's ISO code normalizes to the same language
        if entry_iso:
            entry_iso1, entry_iso2, entry_iso3 = normalize_language_code(entry_iso)
            entry_iso_codes = {entry_iso1, entry_iso2, entry_iso3} - {None}
            if lang_iso_codes & entry_iso_codes:
                return key
    # Try default models with all normalized codes
    for iso_code in lang_iso_codes:
        if iso_code in SPACY_DEFAULT_MODELS:
            return SPACY_DEFAULT_MODELS[iso_code]
    return SPACY_DEFAULT_MODELS.get(lang_norm)



def _ensure_spacy_model_available(model_name: str) -> None:
    try:
        import spacy
        from spacy.cli import download as spacy_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise ValueError(
            "SpaCy is not installed. Install it with: pip install \"flexipipe[spacy]\""
        ) from exc

    # Already a path on disk
    if Path(model_name).exists():
        return

    # Already installed as a package?
    try:
        spacy.util.get_package_path(model_name)
        return
    except (OSError, IOError, ImportError, AttributeError):
        pass

    spacy_dir = get_backend_models_dir("spacy")
    
    # Check if this is a HuggingFace model (contains /)
    if "/" in model_name:
        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ValueError(
                f"HuggingFace model '{model_name}' requires 'huggingface_hub' package. "
                f"Install it with: pip install huggingface_hub"
            )
        
        print(f"[flexipipe] Downloading SpaCy model from HuggingFace: '{model_name}'...")
        try:
            # Download to flexipipe directory
            target_path = spacy_dir / model_name.replace("/", "_")
            if target_path.exists():
                # Already downloaded
                return
            
            # Download to a temporary location first
            import tempfile
            temp_dir = tempfile.mkdtemp()
            try:
                downloaded_path = snapshot_download(model_name, cache_dir=temp_dir)
                # Copy to flexipipe directory
                import shutil
                shutil.copytree(downloaded_path, target_path, dirs_exist_ok=False)
                print(f"[flexipipe] Model downloaded to {target_path}")
            finally:
                # Clean up temp directory
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - depends on network
            raise ValueError(
                f"Failed to download HuggingFace model '{model_name}': {exc}. "
                f"Please check the model name and try again."
            ) from exc
        return

    # Standard SpaCy model download
    print(f"[flexipipe] Downloading SpaCy model '{model_name}'...")
    try:
        spacy_download(model_name)
    except Exception as exc:  # pragma: no cover - depends on network
        raise ValueError(
            f"Failed to download SpaCy model '{model_name}': {exc}. "
            f"Please install it manually: python -m spacy download {model_name}"
        ) from exc

    # Try to copy downloaded model into flexipipe storage for consistency
    try:
        downloaded_path = spacy.util.get_package_path(model_name)
    except Exception:
        downloaded_path = None

    if downloaded_path and Path(downloaded_path).exists():
        target_path = spacy_dir / model_name
        if not target_path.exists():
            try:
                shutil.copytree(downloaded_path, target_path, dirs_exist_ok=False)
            except Exception:
                # Copy is best-effort; keep going if it fails
                pass


def _create_spacy_backend(
    *,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    language: str | None = None,
    download_model: bool = False,
    training: bool = False,  # accepted for parity with create_backend signature
    verbose: bool = False,
    **kwargs: Any,
) -> SpacyBackend:
    model_name = _resolve_spacy_model_name(model_name, language)
    if download_model and model_name and not model_path:
        _ensure_spacy_model_available(model_name)

    # Pass training flag to backend constructors
    kwargs["training"] = training
    if model_path:
        return SpacyBackend.from_model_path(model_path, verbose=verbose, **kwargs)
    if model_name:
        return SpacyBackend.from_pretrained(model_name, verbose=verbose, **kwargs)
    if language:
        return SpacyBackend.blank(language, verbose=verbose, **kwargs)

    # Fall back to constructor to trigger auto-selection logic
    raise ValueError("Must provide either model_name, model_path, or language for SpaCy backend")


BACKEND_SPEC = BackendSpec(
    name="spacy",
    description="SpaCy - Fast NLP library with pre-trained models for many languages",
    factory=_create_spacy_backend,
    get_model_entries=lambda *args, **kwargs: get_spacy_model_entries(*args, **kwargs)[0],
    list_models=list_spacy_models,
    supports_training=True,
    is_rest=False,
    url="https://spacy.io",
)


