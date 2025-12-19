"""
Backend registry for flexipipe.

This module provides a centralized registry of all available backends,
allowing for dynamic discovery and uniform access patterns.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union

from .doc import Document
from .backend_spec import BackendSpec

logger = logging.getLogger(__name__)


def get_backend_status(backend_name: str) -> Dict[str, Any]:
    """
    Get detailed status information about a backend, including what's missing.
    
    Args:
        backend_name: Name of the backend to check
        
    Returns:
        Dictionary with keys:
        - available: bool - whether the backend is fully functional
        - status: str - "available", "missing_module", "missing_binary", "unknown"
        - missing: List[str] - list of missing dependencies/requirements
        - install_hint: str - suggested pip install command
    """
    backend_key = backend_name.lower()
    info = get_backend_info(backend_key)
    if not info:
        return {
            "available": False,
            "status": "unknown",
            "missing": [f"Backend '{backend_name}' is not registered"],
            "install_hint": "",
        }
    
    missing = []
    status = "available"
    
    # REST backends only need requests
    if info.is_rest:
        try:
            if importlib.util.find_spec("requests") is None:
                missing.append("requests module")
                status = "missing_module"
                return {
                    "available": False,
                    "status": status,
                    "missing": missing,
                    "install_hint": "pip install requests",
                }
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append("requests module")
            status = "missing_module"
            return {
                "available": False,
                "status": status,
                "missing": missing,
                "install_hint": "pip install requests",
            }
        return {
            "available": True,
            "status": "available",
            "missing": [],
            "install_hint": "",
        }
    
    # Map backend names to their required Python modules and install hints
    # Since flexipipe is typically installed via git, we suggest direct module installation
    # which works regardless of how flexipipe was installed
    # For git installations with extras, users can use: pip install "git+https://github.com/ufal/flexipipe.git[stanza]"
    # but direct installation is simpler and always works
    required_modules = {
        "stanza": ("stanza", "pip install stanza"),
        "spacy": ("spacy", "pip install spacy"),
        "flair": ("flair", "pip install flair"),
        "transformers": ("transformers", "pip install transformers"),
        "classla": ("classla", "pip install classla"),
    }
    
    # CLI backends that need binaries
    cli_backends = {
        "udpipe1": ("udpipe binary", "Install UDPipe and ensure 'udpipe' is on PATH"),
        "treetagger": ("tree-tagger binary", "Install TreeTagger and ensure 'tree-tagger' is on PATH"),
    }
    
    # Check for Python module requirements
    if backend_key in required_modules:
        module_name, install_hint = required_modules[backend_key]
        try:
            if importlib.util.find_spec(module_name) is None:
                missing.append(f"{module_name} module")
                status = "missing_module"
                return {
                    "available": False,
                    "status": status,
                    "missing": missing,
                    "install_hint": install_hint,
                }
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append(f"{module_name} module")
            status = "missing_module"
            return {
                "available": False,
                "status": status,
                "missing": missing,
                "install_hint": install_hint,
            }
    
    # Check for CLI binary requirements
    if backend_key in cli_backends:
        binary_name, install_hint = cli_backends[backend_key]
        import shutil
        # Check if binary is on PATH
        binary_cmd = backend_key.replace("1", "")  # udpipe1 -> udpipe
        if backend_key == "treetagger":
            binary_cmd = "tree-tagger"
        if shutil.which(binary_cmd) is None:
            missing.append(binary_name)
            status = "missing_binary"
            return {
                "available": False,
                "status": status,
                "missing": missing,
                "install_hint": install_hint,
            }
    
    # For backends without specific requirements (flexitag, etc.)
    # They're assumed available but may fail at runtime with specific errors
    return {
        "available": True,
        "status": "available",
        "missing": [],
        "install_hint": "",
    }


def is_backend_available(backend_name: str) -> bool:
    """
    Check if a backend is available (its required modules can be imported).
    
    Args:
        backend_name: Name of the backend to check
        
    Returns:
        True if the backend is available, False otherwise
    """
    status = get_backend_status(backend_name)
    return status["available"]


class Backend(Protocol):
    """Protocol defining the interface that all backends must implement."""
    
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[list[str]] = None,
        use_raw_text: bool = False,
    ) -> Any:
        """Tag a document using the backend."""
        ...
    
    def supports_training(self) -> bool:
        """Return whether this backend supports training."""
        ...
    
    def train(
        self,
        train_data: Any,
        output_dir: Any,
        *,
        dev_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Train a model (if supported)."""
        ...


class BackendInfo:
    """Information about a backend."""
    
    def __init__(
        self,
        name: str,
        description: str,
        backend_class: Optional[Type[Backend]] = None,
        module_path: Optional[str] = None,
        get_model_entries: Optional[Callable[..., Dict[str, Dict[str, str]]]] = None,
        list_models: Optional[Callable[..., int]] = None,
        create_kwargs_parser: Optional[Callable[[Any], Dict[str, Any]]] = None,
        supports_training: bool = False,
        is_rest: bool = False,
        is_hidden: bool = False,
        factory: Optional[Callable[..., Backend]] = None,
        url: Optional[str] = None,
        install_instructions: Optional[str] = None,
    ):
        self.name = name
        self.backend_class = backend_class
        self.module_path = module_path or (f"flexipipe.{name}_backend" if backend_class else None)
        self.description = description
        self.get_model_entries = get_model_entries
        self.list_models = list_models
        self.create_kwargs_parser = create_kwargs_parser
        self.supports_training = supports_training
        self.is_rest = is_rest
        self.is_hidden = is_hidden
        self.factory = factory
        self.url = url
        self.model_registry_url: Optional[str] = None
        self.install_instructions: Optional[str] = install_instructions


# Registry of all backends
_BACKEND_REGISTRY: Dict[str, BackendInfo] = {}


def register_backend(info: BackendInfo) -> None:
    """Register a backend in the registry."""
    _BACKEND_REGISTRY[info.name.lower()] = info


def register_backend_spec(spec: BackendSpec) -> None:
    """Register a backend based on a BackendSpec definition."""
    info = BackendInfo(
        name=spec.name,
        description=spec.description,
        backend_class=None,
        module_path=None,
        get_model_entries=spec.get_model_entries,
        list_models=spec.list_models,
        supports_training=spec.supports_training,
        is_rest=spec.is_rest,
        is_hidden=spec.is_hidden,
        factory=spec.factory,
        url=spec.url,
        install_instructions=spec.install_instructions,
    )
    info.model_registry_url = spec.model_registry_url
    register_backend(info)


def _load_spec_from_module(module_name: str) -> Optional[BackendSpec]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to import backend module '%s': %s", module_name, exc)
        return None
    spec = getattr(module, "BACKEND_SPEC", None)
    if spec is None:
        logger.debug("Module '%s' does not define BACKEND_SPEC", module_name)
        return None
    if not isinstance(spec, BackendSpec):
        logger.warning("Module '%s' BACKEND_SPEC is not a BackendSpec instance", module_name)
        return None
    return spec


def _iter_builtin_backend_specs():
    try:
        from . import backends as backend_pkg
    except ImportError:  # pragma: no cover - package should exist
        return
    prefix = backend_pkg.__name__ + "."
    for module_info in pkgutil.iter_modules(backend_pkg.__path__, prefix):
        spec = _load_spec_from_module(module_info.name)
        if spec:
            yield spec


def _iter_entry_point_backend_specs():
    try:
        entry_points = metadata.entry_points()
        backend_eps = entry_points.select(group="flexipipe.backends")
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.debug("Unable to read backend entry points: %s", exc)
        return
    for ep in backend_eps:
        try:
            spec = ep.load()
        except Exception as exc:
            logger.warning("Failed to load backend entry point '%s': %s", ep.name, exc)
            continue
        if not isinstance(spec, BackendSpec):
            logger.warning("Entry point '%s' did not return a BackendSpec instance", ep.name)
            continue
        yield spec


def _register_discovered_backends() -> None:
    for iterable in (_iter_builtin_backend_specs(), _iter_entry_point_backend_specs()):
        if not iterable:
            continue
        for spec in iterable:
            register_backend_spec(spec)


def get_backend_info(backend_name: str) -> Optional[BackendInfo]:
    """Get backend information by name."""
    return _BACKEND_REGISTRY.get(backend_name.lower())


def list_backends(include_hidden: bool = False) -> Dict[str, BackendInfo]:
    """List all registered backends."""
    if include_hidden:
        return _BACKEND_REGISTRY.copy()
    return {name: info for name, info in _BACKEND_REGISTRY.items() if not info.is_hidden}


def get_backend_choices() -> List[str]:
    """Get list of backend names for CLI choices (sorted, excluding hidden backends)."""
    backends = list_backends(include_hidden=False)
    return sorted(backends.keys())


def get_backend_choices_for_training() -> List[str]:
    """Get list of backend names that support training."""
    backends = list_backends(include_hidden=False)
    return sorted([name for name, info in backends.items() if info.supports_training])


def get_model_entries(
    backend_name: str,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    """
    Get model entries for a backend using the registry.
    
    This is a generic function that dispatches to the backend-specific
    get_model_entries function registered in the backend info.
    
    Usage:
        entries = get_model_entries('flexitag', use_cache=True)
        # Instead of: get_flexitag_model_entries(use_cache=True)
    """
    info = get_backend_info(backend_name)
    if not info:
        return {}
    if not info.get_model_entries:
        return {}
    return info.get_model_entries(*args, **kwargs)


# Dictionary-like interface for model entries
class ModelEntriesDict:
    """Dictionary-like interface for accessing model entries by backend name."""
    
    def __getitem__(self, backend_name: str) -> Callable[..., Dict[str, Dict[str, str]]]:
        """Return a callable that gets model entries for the specified backend."""
        def _get_entries(*args: Any, **kwargs: Any) -> Dict[str, Dict[str, str]]:
            return get_model_entries(backend_name, *args, **kwargs)
        return _get_entries
    
    def __contains__(self, backend_name: str) -> bool:
        """Check if a backend is registered."""
        return get_backend_info(backend_name) is not None
    
    def keys(self):
        """Return all registered backend names."""
        return _BACKEND_REGISTRY.keys()


# Create a singleton instance for dictionary-like access
get_model_entries_dict = ModelEntriesDict()


def list_models_display(
    backend_name: str,
    *args: Any,
    **kwargs: Any,
) -> int:
    """
    Display models for a backend using the registry.
    
    Returns exit code (0 for success, 1 for error).
    """
    info = get_backend_info(backend_name)
    if not info:
        print(f"Error: Unknown backend '{backend_name}'")
        return 1
    if not info.list_models:
        print(f"Error: Backend '{backend_name}' does not support listing models")
        return 1
    # Some backends don't accept verbose, so filter it out if not in signature
    import inspect
    sig = inspect.signature(info.list_models)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return info.list_models(*args, **filtered_kwargs)


def create_backend(
    backend_type: str,
    *,
    training: bool = False,
    model_name: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    language: Optional[str] = None,
    **kwargs: Any,
) -> Backend:
    """
    Create a backend instance using the registry.
    
    Args:
        backend_type: Type of backend (e.g., "spacy", "stanza", "classla")
        training: Whether this backend is for training
        model_name: Model name (for pretrained models)
        model_path: Path to a local model
        language: Language code (for blank models)
        **kwargs: Additional backend-specific arguments
    
    Returns:
        Backend instance
    """
    backend_key = backend_type.lower()
    info = get_backend_info(backend_key)
    
    if not info:
        available = sorted([name for name in _BACKEND_REGISTRY.keys() if not _BACKEND_REGISTRY[name].is_hidden])
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Available backends: {', '.join(available)}"
        )
    
    if info.factory is not None:
        factory_kwargs = dict(kwargs)
        factory_kwargs.setdefault("training", training)
        if model_name is not None:
            # For REST backends, map model_name to model
            if info.is_rest:
                factory_kwargs.setdefault("model", model_name)
            else:
                factory_kwargs.setdefault("model_name", model_name)
        if model_path is not None:
            factory_kwargs.setdefault("model_path", model_path)
        if language is not None:
            factory_kwargs.setdefault("language", language)
        try:
            return info.factory(**factory_kwargs)
        except ImportError as e:
            # Handle missing module gracefully
            error_str = str(e)
            # Check if this is a SpaCy language support error (E048)
            # For training, let the backend handle it (it will use 'xx' fallback)
            if training and ("E048" in error_str or ("Can't import language" in error_str and "spacy.lang" in error_str)):
                # Re-raise to let the backend handle it (SpaCy backend will use 'xx' for training)
                raise
            if "E048" in error_str or ("Can't import language" in error_str and "spacy.lang" in error_str):
                # Extract language code from error if possible
                language = factory_kwargs.get("language", "unknown")
                raise RuntimeError(
                    f"Backend '{backend_type}' does not support language '{language}'. "
                    f"SpaCy only supports certain languages out of the box. "
                    f"For training, unsupported languages will automatically use 'xx' (multilingual)."
                ) from e
            # Try to extract module name from common error patterns
            module_name = "unknown"
            # Pattern 1: "No module named 'X'"
            if "No module named" in error_str and "'" in error_str:
                parts = error_str.split("No module named")
                if len(parts) > 1:
                    quoted = parts[1].strip()
                    if quoted.startswith("'") and "'" in quoted[1:]:
                        module_name = quoted[1:].split("'")[0]
                        # Skip if it's a spacy.lang.* module (language support issue, not missing spacy)
                        if module_name.startswith("spacy.lang."):
                            language = module_name.replace("spacy.lang.", "")
                            # For training, let the backend handle it
                            if training:
                                raise
                            raise RuntimeError(
                                f"Backend '{backend_type}' does not support language '{language}'. "
                                f"SpaCy only supports certain languages out of the box. "
                                f"For training, unsupported languages will automatically use 'xx' (multilingual)."
                            ) from e
            # If we couldn't extract a module name, use a generic message
            if module_name == "unknown":
                raise RuntimeError(
                    f"Backend '{backend_type}' import error: {error_str}"
                ) from e
            raise RuntimeError(
                f"Backend '{backend_type}' requires the '{module_name}' module, but it is not installed. "
                f"Please install it with: pip install \"flexipipe[{backend_type}]\""
            ) from e
    
    if not info.backend_class:
        raise ValueError(f"Backend '{backend_type}' does not have a backend class registered")
    
    backend_class = info.backend_class
    
    # Handle backend-specific creation logic
    if backend_key == "udmorph":
        # UDMorph REST backend (download_model not applicable for REST services)
        kwargs.pop("download_model", None)  # Ignore download_model for REST backends
        endpoint_url = kwargs.pop("endpoint_url", None)
        if not endpoint_url:
            raise ValueError("UDMorph backend requires endpoint_url. Provide --udmorph-url.")
        # Map model_name to model for REST backends
        model = kwargs.pop("model", None) or kwargs.pop("model_name", None)
        timeout = kwargs.pop("timeout", 30.0)
        extra_params = kwargs.pop("extra_params", None)
        headers = kwargs.pop("headers", None)
        session = kwargs.pop("session", None)
        log_requests = kwargs.pop("log_requests", False)
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected UDMorph backend arguments: {unexpected}")
        return backend_class(
            endpoint_url,
            model=model,
            timeout=timeout,
            extra_params=extra_params,
            headers=headers,
            session=session,
            log_requests=log_requests,
        )
    
    elif backend_key == "nametag":
        # NameTag REST backend (download_model not applicable for REST services)
        kwargs.pop("download_model", None)  # Ignore download_model for REST backends
        endpoint_url = kwargs.pop("endpoint_url", None)
        if not endpoint_url:
            raise ValueError("NameTag backend requires endpoint_url. Provide --nametag-url.")
        # Map model_name to model for REST backends
        model = kwargs.pop("model", None) or kwargs.pop("model_name", None)
        language = kwargs.pop("language", None)
        version = kwargs.pop("version", "3")
        timeout = kwargs.pop("timeout", 30.0)
        extra_params = kwargs.pop("extra_params", None)
        headers = kwargs.pop("headers", None)
        session = kwargs.pop("session", None)
        log_requests = kwargs.pop("log_requests", False)
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected NameTag backend arguments: {unexpected}")
        return backend_class(
            endpoint_url,
            model=model,
            language=language,
            version=version,
            timeout=timeout,
            extra_params=extra_params,
            headers=headers,
            session=session,
            log_requests=log_requests,
        )
    
    else:
        # Generic fallback: try to instantiate with provided kwargs
        # This allows backends to define their own __init__ signatures
        # Pop download_model if not handled by backend (some backends may not support it)
        kwargs.pop("download_model", None)
        return backend_class(
            model_name=model_name,
            model_path=model_path,
            language=language,
            **kwargs
        )


# Initialize registry with all backends
def _initialize_registry() -> None:
    """Initialize the backend registry with all available backends."""
    # This will be called at module import time
    # Backends will register themselves or be registered here
    
    # Flexitag backend is registered via BackendSpec in backends/flexitag.py
    # No need to register here as it's auto-discovered
    
    # UDMorph REST backend is registered via BackendSpec in backends/udmorph.py
    # No need to register here as it's auto-discovered
    
    # NameTag REST backend
    # NameTag REST backend is registered via BackendSpec in backends/nametag.py
    # No need to register here as it's auto-discovered
    
    
    _register_discovered_backends()


# Initialize registry on module import
_initialize_registry()

