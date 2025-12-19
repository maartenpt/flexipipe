"""Backend spec and implementation for the Flexitag backend."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..backend_spec import BackendSpec
from ..doc import Document
from ..engine import FlexitagFallback, FlexitagResult
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

FLEXITAG_CACHE_TTL_SECONDS = 300


def get_flexitag_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = FLEXITAG_CACHE_TTL_SECONDS,
    include_remote: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    """
    Return metadata for flexitag models (local + remote).
    
    Args:
        use_cache: If True, use cached model lists
        refresh_cache: If True, force refresh from remote
        cache_ttl_seconds: Cache TTL for local model scan
        include_remote: If True, include remote models from registry
        verbose: If True, print progress messages
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Dictionary mapping model names to model entry dictionaries
    """
    try:
        models_dir = get_backend_models_dir("flexitag", create=False)
    except (OSError, PermissionError) as e:
        # If we can't even get the directory path (e.g., permission denied), return empty entries
        local_entries: Dict[str, Dict[str, str]] = {}
    else:
        cache_key = "flexitag:local"

        if use_cache and not refresh_cache:
            cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
            if cached and cache_entries_standardized(cached):
                local_entries = cached
            else:
                local_entries = _scan_local_flexitag_models(models_dir)
        else:
            local_entries = _scan_local_flexitag_models(models_dir)
        
        # Cache local models
        if refresh_cache:
            try:
                write_model_cache_entry(cache_key, local_entries)
            except (OSError, PermissionError):
                pass  # Cache write is best-effort
    
    # Merge with remote models if requested
    if include_remote:
        try:
            from ..model_registry import merge_remote_and_local_models
            all_entries = merge_remote_and_local_models(
                "flexitag",
                local_entries,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                verbose=verbose,
            )
            return all_entries
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to load remote flexitag models: {exc}")
            # Fall back to local models only
            return local_entries
    
    return local_entries


def _scan_local_flexitag_models(models_dir: Path) -> Dict[str, Dict[str, str]]:
    """Scan local directory for flexitag models and return entries."""
    entries: Dict[str, Dict[str, str]] = {}

    def add_entry(name: str, vocab_path: Path) -> None:
        tag_attr = ""
        language = ""
        try:
            with open(vocab_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            metadata = data.get("metadata", {})
            language = metadata.get("language") or metadata.get("lang") or ""
            tag_selection = metadata.get("tag_selection", {})
            tag_attr = (
                tag_selection.get("chosen")
                or tag_selection.get("tag_attribute")
                or tag_selection.get("auto_selected")
                or ""
            )
        except Exception:
            pass
        updated = datetime.fromtimestamp(vocab_path.stat().st_mtime).strftime("%Y-%m-%d")
        
        # Try to extract ISO code from model name (e.g., "afrikaans-afribooms-ud216" -> "af")
        # Model names often follow UD treebank naming: language-treebank-udversion
        language_code_candidate = None
        language_name_candidate = language or name
        
        # If language is a short code (2-3 chars), use it as-is (normalize to lowercase)
        if language and len(language) <= 3 and language.isalpha():
            language_code_candidate = language.lower()
            # Use language mapping to get proper ISO code and name
            try:
                from ..language_mapping import normalize_language_code
                iso_1, iso_2, iso_3 = normalize_language_code(language_code_candidate)
                if iso_1:
                    language_code_candidate = iso_1
            except Exception:
                pass
        else:
            # Try to extract from model name (first part before first hyphen)
            parts = name.split("-")
            if parts:
                lang_part = parts[0].lower()  # Normalize to lowercase for lookup
                # Use language mapping first (handles lowercase names)
                try:
                    from ..language_mapping import normalize_language_code, get_language_metadata
                    iso_1, iso_2, iso_3 = normalize_language_code(lang_part)
                    if iso_1:
                        language_code_candidate = iso_1
                        lang_metadata = get_language_metadata(lang_part)
                        if lang_metadata.get("primary_name"):
                            language_name_candidate = lang_metadata["primary_name"]
                except Exception:
                    pass
                
                # Fallback to standardize_language_metadata if mapping doesn't work
                if not language_code_candidate:
                    from ..language_utils import standardize_language_metadata
                    lang_metadata = standardize_language_metadata(
                        language_code=None,
                        language_name=lang_part.replace("_", " ").title(),
                    )
                    language_code_candidate = lang_metadata.get(LANGUAGE_FIELD_ISO)
                    if not language_name_candidate or language_name_candidate == name:
                        language_name_candidate = lang_metadata.get(LANGUAGE_FIELD_NAME) or lang_part.replace("_", " ").title()
        
        entry = build_model_entry(
            "flexitag",
            name,
            language_code=language_code_candidate,
            language_name=language_name_candidate,
            tag_attribute=tag_attr or None,
            updated=updated,
        )
        entries[name] = entry

    # Try to read from the models directory if it exists
    # Wrap in try-except to handle permission errors gracefully
    if models_dir.exists():
        try:
            for child in sorted(models_dir.iterdir()):
                if child.is_dir():
                    vocab_path = child / "model_vocab.json"
                    if vocab_path.exists():
                        add_entry(child.name, vocab_path)
                elif child.is_file() and child.name.endswith(".json") and child.name.startswith("model_vocab"):
                    add_entry(child.stem, child)
        except (OSError, PermissionError) as e:
            # If we can't read the directory (permission denied), just return empty entries
            # This can happen when the web server user doesn't have read permissions
            pass

    return entries


def list_flexitag_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    include_remote: bool = True,
    verbose: bool = False,
    **_: Any,
) -> int:
    """Print flexitag models (local + remote)."""
    entries_dict = get_flexitag_model_entries(
        use_cache=use_cache and not refresh_cache,
        refresh_cache=refresh_cache,
        include_remote=include_remote,
        verbose=verbose,
    )
    entries = list(entries_dict.values())

    # Separate local and remote models
    local_entries = [e for e in entries if e.get("source", "local") == "local"]
    remote_entries = [e for e in entries if e.get("source", "local") != "local"]

    print("\nFlexitag models:")
    if entries:
        print(f"{'Model':<30} {'ISO':<8} {'Language':<20} {'Tag attr':<12} {'Source':<10} {'Updated':<12}")
        print("=" * 110)
        
        # Show local models first
        for entry in sorted(local_entries, key=lambda e: e.get("model", "")):
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or "-"
            lang_name = entry.get(LANGUAGE_FIELD_NAME) or "-"
            tag_attr = entry.get("tag_attribute") or "-"
            updated = entry.get("updated", "-")
            source = "local"
            print(f"{entry['model']:<30} {lang_iso:<8} {lang_name:<20} {tag_attr:<12} {source:<10} {updated:<12}")
        
        # Show remote models
        for entry in sorted(remote_entries, key=lambda e: e.get("model", "")):
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or "-"
            lang_name = entry.get(LANGUAGE_FIELD_NAME) or "-"
            tag_attr = entry.get("tag_attribute") or "-"
            updated = entry.get("updated", "-")
            source = entry.get("source", "remote")
            print(f"{entry['model']:<30} {lang_iso:<8} {lang_name:<20} {tag_attr:<12} {source:<10} {updated:<12}")
        
        print(f"\nTotal: {len(local_entries)} local, {len(remote_entries)} remote")
    else:
        print("  (no models found)")

    print("\nFlexitag models are created by training on your data:")
    print("  python -m flexipipe train --backend flexitag --ud-data <UD treebank> --output-dir <dir>")
    print("\nEach model directory contains:")
    print("  - model_vocab.json : Vocabulary & parameters")
    print("  - (optional) model_vocab.xml : Legacy export")
    if remote_entries:
        print("\nRemote models can be downloaded using:")
        print("  python -m flexipipe download-model --backend flexitag --model <model-name>")
    return 0


def build_flexitag_options_from_args(args: Any) -> Dict[str, Any]:
    """Construct FlexitagFallback option dict from CLI arguments."""
    options: Dict[str, Any] = {}
    # --pid removed as redundant (only relevant for legacy settings)
    if getattr(args, "debug", False):
        options["debug"] = True
    
    # Parse normalization style
    normalization_style = getattr(args, "normalization_style", "conservative")
    if normalization_style == "conservative":
        options["normalization_conservative"] = True
        options["skip_enhanced_normalization"] = True
    elif normalization_style == "aggressive":
        options["normalization_conservative"] = False
        options["skip_enhanced_normalization"] = False
    elif normalization_style == "enhanced":
        options["normalization_conservative"] = True
        options["skip_enhanced_normalization"] = False
    elif normalization_style == "balanced":
        options["normalization_conservative"] = False
        options["skip_enhanced_normalization"] = False
    
    extra_vocab = getattr(args, "extra_vocab", None)
    if extra_vocab:
        options["extra-vocab"] = extra_vocab
    return options


def resolve_flexitag_model_path(
    *,
    model_name: Optional[str],
    params_path: Optional[str],
) -> Path:
    """Resolve a flexitag model path based on --model CLI argument."""
    if params_path:
        params = Path(params_path)
        if not params.exists():
            raise ValueError(f"Error: Flexitag model path does not exist: {params}")
        return params

    if not model_name:
        raise ValueError(
            "Error: Must provide --model when using flexitag backend.\n"
            "Use --model <name> to load from the flexitag data folder or flexipipe models directory."
        )

    models_dir = get_backend_models_dir("flexitag", create=False)
    direct_candidate = Path(model_name)
    flexitag_data_env = os.environ.get("FLEXITAG_DATA")

    candidates = [
        models_dir / model_name,
        direct_candidate,
    ]

    if flexitag_data_env:
        candidates.append(Path(flexitag_data_env) / model_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"Error: Flexitag model '{model_name}' not found.\n"
        f"Checked locations: {checked_paths}\n"
        "Use --model <name> to load from the flexitag data folder or flexipipe models directory."
    )


class FlexitagBackend(BackendManager):
    """Backend wrapper for FlexitagFallback that implements BackendManager interface."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        options: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        """
        Initialize Flexitag backend.
        
        Args:
            model_path: Path to the flexitag model (directory containing model_vocab.json)
            options: Optional dictionary of flexitag options
            debug: Enable debug logging
        """
        self._model_path = Path(model_path)
        self._options = options or {}
        self._debug = debug
        self._fallback = FlexitagFallback(
            str(self._model_path),
            options=self._options,
            debug=self._debug,
        )
        self._model_name = self._model_path.name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _backend_info(self) -> str:
        """Used by CLI to describe this backend."""
        return f"flexitag: {self._model_name}"

    def supports_training(self) -> bool:
        return True

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[list[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        """
        Tag a document using Flexitag.
        
        Args:
            document: Input document
            overrides: Optional overrides (ignored for flexitag)
            preserve_pos_tags: If True, preserve existing POS tags (ignored for flexitag)
            components: Optional list of components (ignored for flexitag)
            use_raw_text: If True, use raw text mode (ignored for flexitag)
        
        Returns:
            NeuralResult with tagged document
        """
        # Flexitag doesn't support these parameters, but we accept them for interface compatibility
        _ = overrides, preserve_pos_tags, components, use_raw_text
        
        flexitag_result: FlexitagResult = self._fallback.tag(document)
        
        # Convert FlexitagResult to NeuralResult
        return NeuralResult(
            document=flexitag_result.document,
            stats=flexitag_result.stats,
        )

    def train(
        self,
        train_data: Any,
        output_dir: Any,
        *,
        dev_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Train a Flexitag model.
        
        Note: Training is handled by the train command, not through the backend interface.
        This method raises NotImplementedError to indicate that training should be done
        via the CLI command.
        """
        raise NotImplementedError(
            "Flexitag training is handled by the 'train' command, not through the backend interface. "
            "Use: python -m flexipipe train --backend flexitag --ud-data <treebank> --output-dir <dir>"
        )


def _create_flexitag_backend(
    *,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    params_path: Optional[str] = None,
    language: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> FlexitagBackend:
    """Instantiate the Flexitag backend."""
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "Flexitag", allowed_extra=["download_model", "training"])
    
    # Resolve model path
    resolved_model = model or model_name
    
    # If no model specified but language is provided, try to resolve from model catalog
    if not resolved_model and language:
        try:
            resolved_model = resolve_model_from_language(language, "flexitag")
        except ValueError:
            # Re-raise ValueError as-is (it already has a helpful message)
            raise
        except Exception as e:
            # If model catalog lookup fails for other reasons, provide a helpful error message
            from ..model_storage import is_running_from_teitok
            teitok_msg = "" if is_running_from_teitok() else f" Provide --model to specify a model name, or use 'python -m flexipipe info models --backend flexitag' to see available models."
            raise ValueError(
                f"Could not resolve Flexitag model for language '{language}': {e}.{teitok_msg}"
            ) from e
    
    if params_path:
        resolved_path = Path(params_path)
    elif resolved_model:
        resolved_path = resolve_flexitag_model_path(
            model_name=resolved_model,
            params_path=None,
        )
    elif model_path:
        resolved_path = Path(model_path)
    else:
        raise ValueError(
            "Flexitag backend requires a model. Provide --model or --language."
        )
    
    return FlexitagBackend(
        model_path=resolved_path,
        options=options or {},
        debug=debug,
    )


BACKEND_SPEC = BackendSpec(
    name="flexitag",
    description="Flexitag - Rule-based fallback tagger (default)",
    factory=_create_flexitag_backend,
    get_model_entries=get_flexitag_model_entries,
    list_models=list_flexitag_models,
    supports_training=True,
    is_rest=False,
    url="https://github.com/flexipipe/flexipipe",  # Built-in backend
    install_instructions="flexitag is a built-in backend that requires no installation",
)

