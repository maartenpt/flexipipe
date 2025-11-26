"""
Centralized model storage for flexipipe backends.

All backends store their models in a common flexipipe directory structure:
  ~/.flexipipe/models/  (default)
    ├── spacy/
    ├── stanza/
    ├── flair/
    ├── transformers/
    └── flexitag/

The models directory can be configured via:
  - Config file: ~/.flexipipe/config.json (set "models_dir" key)
  - Environment variable: FLEXIPIPE_MODELS_DIR
  - Default: ~/.flexipipe/models/

This makes it easy to:
  - See all models in one place
  - Manage disk space
  - Uninstall models by simply deleting the directory
  - Store models on external drives by setting the directory once
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


def get_flexipipe_config_dir(create: bool = True) -> Path:
    """
    Get the flexipipe configuration directory.
    
    Args:
        create: If True, create the directory if it doesn't exist. If False, return the path
                without creating it (useful for read-only operations).
    
    Returns:
        Path to ~/.flexipipe/ (or $XDG_DATA_HOME/flexipipe/ if XDG_DATA_HOME is set)
    """
    if "XDG_DATA_HOME" in os.environ:
        base = Path(os.environ["XDG_DATA_HOME"]) / "flexipipe"
    else:
        base = Path.home() / ".flexipipe"
    
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def get_config_file(create_dir: bool = True) -> Path:
    """Get the path to the flexipipe configuration file."""
    return get_flexipipe_config_dir(create=create_dir) / "config.json"


def read_config() -> dict:
    """
    Read the flexipipe configuration file.
    
    Returns:
        Dictionary with configuration values (empty dict if file doesn't exist)
    """
    # Don't create the directory when just reading
    config_file = get_config_file(create_dir=False)
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If config file is corrupted, return empty dict
            return {}
    return {}


def write_config(config: dict) -> None:
    """
    Write configuration to the flexipipe config file.
    
    Args:
        config: Dictionary with configuration values
    """
    config_file = get_config_file()
    # Read existing config to preserve other settings
    existing_config = read_config()
    existing_config.update(config)
    
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2, ensure_ascii=False)


def get_flexipipe_models_dir(create: bool = True) -> Path:
    """
    Get the base directory for flexipipe models.
    
    Checks in order:
    1. FLEXIPIPE_MODELS_DIR environment variable
    2. config.json file (models_dir key)
    3. ~/.flexipipe/models/ (default)
    
    Args:
        create: If True, create the directory if it doesn't exist. If False, return the path
                without creating it (useful for read-only operations).
    
    Returns:
        Path to the models directory
    """
    # Check environment variable first (highest priority)
    if "FLEXIPIPE_MODELS_DIR" in os.environ:
        models_dir = Path(os.environ["FLEXIPIPE_MODELS_DIR"])
        if create:
            models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    # Check config file
    config = read_config()
    if "models_dir" in config:
        models_dir = Path(config["models_dir"])
        if create:
            models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    # Default location: always ~/.flexipipe/models/
    models_dir = Path.home() / ".flexipipe" / "models"
    if create:
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def set_models_dir(path: str | Path) -> None:
    """
    Set the models directory in the configuration file.
    
    Args:
        path: Path to the models directory (will be created if it doesn't exist)
    """
    models_dir = Path(path).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    write_config({"models_dir": str(models_dir)})


def get_default_backend() -> Optional[str]:
    """
    Get the default backend from configuration.
    
    Returns:
        Default backend name, or None if not set
    """
    config = read_config()
    return config.get("default_backend")


def set_default_backend(backend: str) -> None:
    """
    Set the default backend in the configuration file.
    
    Args:
        backend: Backend name (flexitag, spacy, stanza, etc.)
    """
    write_config({"default_backend": backend})


def get_default_output_format() -> Optional[str]:
    """
    Get the default output format from configuration.
    
    Returns:
        Default output format (teitok, conllu, conllu-ne, or json), or None if not set
    """
    config = read_config()
    return config.get("default_output_format")


def set_default_output_format(output_format: str) -> None:
    """
    Set the default output format in the configuration file.
    
    Args:
        output_format: Output format (teitok, conllu, conllu-ne, or json)
    """
    write_config({"default_output_format": output_format})


def get_default_create_implicit_mwt() -> bool:
    """
    Get the default create_implicit_mwt setting from configuration.
    
    Returns:
        True if create_implicit_mwt should be enabled by default, False otherwise
    """
    config = read_config()
    return config.get("default_create_implicit_mwt", False)


def get_default_writeback() -> bool:
    """
    Get the default writeback setting from configuration.
    
    Returns:
        True if writeback should be enabled by default, False otherwise
    """
    config = read_config()
    return config.get("default_writeback", False)


def set_default_writeback(enabled: bool) -> None:
    """
    Set the default writeback setting in the configuration file.
    
    Args:
        enabled: True to enable writeback by default, False otherwise
    """
    write_config({"default_writeback": enabled})


def get_auto_install_extras() -> bool:
    """Return whether optional extras should be installed automatically."""
    config = read_config()
    return config.get("auto_install_extras", False)


def set_auto_install_extras(enabled: bool) -> None:
    """Set automatic installation of extras."""
    write_config({"auto_install_extras": bool(enabled)})


def get_prompt_install_extras() -> bool:
    """Return whether the CLI should prompt before installing extras."""
    config = read_config()
    return config.get("prompt_install_extras", True)


def set_prompt_install_extras(enabled: bool) -> None:
    """Set prompting behaviour for optional extras."""
    write_config({"prompt_install_extras": bool(enabled)})


def set_default_create_implicit_mwt(enabled: bool) -> None:
    """
    Set the default create_implicit_mwt setting in the configuration file.
    
    Args:
        enabled: True to enable create_implicit_mwt by default, False to disable
    """
    write_config({"default_create_implicit_mwt": enabled})


def get_cache_dir(create: bool = True) -> Path:
    """
    Return the cache directory used for storing auxiliary data (such as model lists).
    
    Args:
        create: If True, create the directory if it doesn't exist. If False, return the path
                without creating it (useful for read-only operations).
    """
    cache_dir = get_flexipipe_config_dir(create=create) / "cache"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_model_cache_file(create_dir: bool = True) -> Path:
    return get_cache_dir(create=create_dir) / "model_lists.json"


def _load_model_cache() -> dict:
    # Don't create directory when just reading
    cache_file = _get_model_cache_file(create_dir=False)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}
    return {}


def _write_model_cache(cache: dict) -> None:
    # Create directory when writing
    # If directory creation fails (permission denied), raise the error
    cache_file = _get_model_cache_file(create_dir=True)
    try:
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, ensure_ascii=False)
    except (OSError, PermissionError) as e:
        # Re-raise with more context about what we were trying to do
        raise OSError(f"Failed to write cache file to {cache_file}: {e}") from e


def read_model_cache_entry(
    backend: str,
    *,
    max_age_seconds: Optional[float] = None,
) -> Optional[dict]:
    """
    Read a cached model listing for the given backend. Returns None if missing or stale.
    """
    cache = _load_model_cache()
    entry = cache.get(backend)
    if not entry:
        return None
    timestamp = entry.get("timestamp")
    if (
        max_age_seconds is not None
        and timestamp
        and (time.time() - float(timestamp)) > max_age_seconds
    ):
        return None
    return entry.get("data")


def write_model_cache_entry(backend: str, data: dict) -> None:
    """
    Store a cached model listing for the given backend.
    """
    cache = _load_model_cache()
    cache[backend] = {
        "timestamp": time.time(),
        "data": data,
    }
    _write_model_cache(cache)


def get_model_cache_timestamp(backend: str) -> Optional[float]:
    """
    Return the timestamp of the cached entry for a backend, if present.
    """
    cache = _load_model_cache()
    entry = cache.get(backend)
    if not entry:
        return None
    return entry.get("timestamp")


def get_backend_models_dir(backend: str, create: bool = True) -> Path:
    """
    Get the model directory for a specific backend.
    
    Args:
        backend: Backend name (spacy, stanza, flair, transformers, flexitag)
        create: If True, create the directory if it doesn't exist. If False, return the path
                without creating it (useful for read-only operations).
    
    Returns:
        Path to the backend's model directory
    """
    models_dir = get_flexipipe_models_dir(create=create)
    backend_dir = models_dir / backend.lower()
    if create:
        backend_dir.mkdir(parents=True, exist_ok=True)
    return backend_dir


def setup_backend_environment(backend: str) -> None:
    """
    Set up environment variables to redirect backend model storage to flexipipe directory.
    
    This should be called before importing/initializing backend libraries.
    
    Args:
        backend: Backend name (spacy, stanza, classla, flair, transformers)
    """
    # Don't create directories here - just set environment variables
    # Directories will be created when models are actually downloaded
    backend_dir = get_backend_models_dir(backend, create=False)
    
    if backend == "stanza":
        # Stanza uses STANZA_RESOURCES_DIR environment variable
        os.environ["STANZA_RESOURCES_DIR"] = str(backend_dir)
    elif backend == "classla":
        # ClassLA is a fork of Stanza and uses the same environment variable
        # It also checks for CLASSLA_RESOURCES_DIR, but STANZA_RESOURCES_DIR takes precedence
        os.environ["STANZA_RESOURCES_DIR"] = str(backend_dir)
        # Also set CLASSLA_RESOURCES_DIR for compatibility
        os.environ["CLASSLA_RESOURCES_DIR"] = str(backend_dir)
    elif backend == "flair":
        # Flair uses FLAIR_CACHE_ROOT environment variable
        os.environ["FLAIR_CACHE_ROOT"] = str(backend_dir)
    elif backend == "transformers":
        # Transformers v4+ recommends HF_HOME / HF_DATASETS_CACHE
        os.environ["HF_HOME"] = str(backend_dir)
        os.environ["HF_DATASETS_CACHE"] = str(backend_dir / "datasets")
    # SpaCy doesn't have a simple environment variable, handled differently


def get_spacy_model_path(model_name: str) -> Optional[Path]:
    """
    Get the path to a SpaCy model in the flexipipe directory.
    
    Args:
        model_name: SpaCy model name (e.g., "en_core_web_sm" or "spacyturk/tr_floret_web_lg")
    
    Returns:
        Path to the model if it exists in flexipipe directory, None otherwise
    """
    spacy_dir = get_backend_models_dir("spacy", create=False)
    # First try exact match
    model_path = spacy_dir / model_name
    if model_path.exists() and (model_path / "meta.json").exists():
        return model_path
    
    # For HuggingFace models (containing /), try sanitized name (replace / with _)
    if "/" in model_name:
        sanitized_name = model_name.replace("/", "_")
        sanitized_path = spacy_dir / sanitized_name
        if sanitized_path.exists():
            # Check if it has meta.json or config.cfg (HuggingFace models might not have meta.json)
            if (sanitized_path / "meta.json").exists() or (sanitized_path / "config.cfg").exists():
                return sanitized_path
    
    # Try versioned model names (e.g., en_core_web_md-3.8.0)
    # SpaCy models are sometimes stored with version suffixes
    for model_dir in spacy_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            # Check if this directory matches the model name (with or without version)
            if model_dir.name == model_name or model_dir.name.startswith(f"{model_name}-"):
                if (model_dir / "meta.json").exists() or (model_dir / "config.cfg").exists():
                    return model_dir
            # Also check sanitized name for HuggingFace models
            if "/" in model_name:
                sanitized_name = model_name.replace("/", "_")
                if model_dir.name == sanitized_name or model_dir.name.startswith(f"{sanitized_name}-"):
                    if (model_dir / "meta.json").exists() or (model_dir / "config.cfg").exists():
                        return model_dir
    
    return None


def list_installed_models(backend: str) -> list[str]:
    """
    List installed models for a backend in the flexipipe directory.
    
    Args:
        backend: Backend name
    
    Returns:
        List of model names/paths
    """
    backend_dir = get_backend_models_dir(backend, create=False)
    models = []
    
    if backend == "spacy":
        # SpaCy models are directories with meta.json
        for model_dir in backend_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "meta.json").exists():
                models.append(model_dir.name)
    elif backend == "stanza":
        # Stanza models: lang/processor/package.pt
        for lang_dir in backend_dir.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith("."):
                for processor_dir in lang_dir.iterdir():
                    if processor_dir.is_dir():
                        for pt_file in processor_dir.glob("*.pt"):
                            key = f"{lang_dir.name}_{pt_file.stem}"
                            if key not in models:
                                models.append(key)
    elif backend == "flair":
        # Flair models: model_name/*.bin or *.pt
        for model_dir in backend_dir.iterdir():
            if model_dir.is_dir():
                if any(model_dir.glob("*.bin")) or any(model_dir.glob("*.pt")):
                    models.append(model_dir.name)
    elif backend == "transformers":
        # Transformers models: model_name/ with .bin or .safetensors
        for model_dir in backend_dir.iterdir():
            if model_dir.is_dir():
                if any(model_dir.rglob("*.bin")) or any(model_dir.rglob("*.safetensors")):
                    models.append(model_dir.name)
    elif backend == "flexitag":
        # Flexitag models: directories with model_vocab.json
        for model_dir in backend_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model_vocab.json").exists():
                models.append(str(model_dir))
    
    return sorted(models)

