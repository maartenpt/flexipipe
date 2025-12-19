"""Definitions for pluggable flexipipe backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - satisfied at type-check time
    from .backend_registry import Backend

BackendFactory = Callable[..., "Backend"]


@dataclass(frozen=True)
class BackendSpec:
    """Specification describing how to instantiate and expose a backend."""

    name: str
    description: str
    factory: BackendFactory
    get_model_entries: Optional[Callable[..., Dict[str, Dict[str, str]]]] = None
    list_models: Optional[Callable[..., int]] = None
    supports_training: bool = False
    is_rest: bool = False
    is_hidden: bool = False
    url: Optional[str] = None
    """URL to the backend's homepage or project repository."""
    model_registry_url: Optional[str] = None
    """Optional URL to a backend-specific model registry JSON file.
    
    If not specified, defaults to:
    https://raw.githubusercontent.com/flexipipe/flexipipe-models/main/registries/{name}.json
    
    Third-party backends can specify their own registry URL.
    """
    install_instructions: Optional[str] = None
    """Installation instructions for this backend.
    
    If None, defaults to "Install via: flexipipe install {name}".
    For backends that don't require installation (e.g., REST services),
    provide a message like "{name} is a REST service that requires no installation".
    """


