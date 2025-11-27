"""
Backend registry for flexipipe.

This module provides a centralized registry of all available backends,
allowing for dynamic discovery and uniform access patterns.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union

from .doc import Document
from .backend_spec import BackendSpec

logger = logging.getLogger(__name__)


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
        self.model_registry_url: Optional[str] = None


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
        return info.factory(**factory_kwargs)
    
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

