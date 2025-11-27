"""
Unified model catalog for fast model selection and information lookup.

This module maintains a single, optimized model list that can be read quickly
for model selection and information display. The catalog includes metadata
like preferred flags, availability without download, and performance indicators.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .model_storage import get_cache_dir, read_model_cache_entry, write_model_cache_entry

# Catalog cache key
CATALOG_CACHE_KEY = "unified_model_catalog_v2"
CATALOG_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def build_unified_catalog(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a unified model catalog from all backends.
    
    The catalog is a flat dictionary keyed by "{backend}:{model_name}" containing:
    - All standard model entry fields (backend, model, language_iso, etc.)
    - preferred: bool - whether this is the recommended model for its language/backend
    - available_without_download: bool - whether the model is already installed/available
    - speed_rating: Optional[int] - relative speed rating (1-5, higher = faster)
    - accuracy_rating: Optional[int] - relative accuracy rating (1-5, higher = more accurate)
    
    Args:
        use_cache: If True, use cached catalog if available
        refresh_cache: If True, force rebuild of catalog
        verbose: If True, print progress messages
        
    Returns:
        Dictionary mapping "{backend}:{model_name}" to model entry dicts
    """
    # Check cache first
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(CATALOG_CACHE_KEY, max_age_seconds=CATALOG_CACHE_TTL_SECONDS)
        if cached:
            if verbose:
                print(f"[flexipipe] Using cached unified model catalog ({len(cached)} models)")
            # Warn if caches are stale (but still use them)
            if verbose:
                try:
                    from .cache_manager import warn_if_caches_stale
                    warn_if_caches_stale(max_age_seconds=CATALOG_CACHE_TTL_SECONDS, verbose=verbose)
                except ImportError:
                    pass
            return cached
        else:
            # Cache is missing or stale - warn instead of auto-rebuilding
            if verbose:
                try:
                    from .cache_manager import check_cache_staleness
                    is_stale, age, _ = check_cache_staleness(CATALOG_CACHE_KEY, max_age_seconds=CATALOG_CACHE_TTL_SECONDS)
                    if is_stale:
                        if age is not None:
                            age_hours = age / 3600
                            print(f"[flexipipe] Warning: unified catalog cache is stale ({age_hours:.1f} hours old)")
                        else:
                            print(f"[flexipipe] Warning: unified catalog cache is missing")
                        print("[flexipipe] Run 'python -m flexipipe config --refresh-all-caches' to refresh all caches.")
                except ImportError:
                    pass
                # Return empty dict instead of rebuilding
                return {}
    
    if verbose:
        print("[flexipipe] Building unified model catalog...")
    
    from .backend_registry import get_backend_info, get_model_entries
    from .__main__ import _get_language_backend_priority
    
    catalog: Dict[str, Dict[str, Any]] = {}
    
    # Load entries from all backends using cached data directly (very fast)
    for backend in _get_language_backend_priority():
        if backend is None:
            continue
        try:
            backend_info = get_backend_info(backend)
            if not backend_info or not backend_info.get_model_entries:
                continue
            
            # Try to read from cache directly first (fastest)
            is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
            if is_rest_backend:
                # REST backends use cache keys like "udpipe:{url}"
                if backend == "udpipe":
                    cache_key = "udpipe:https://lindat.mff.cuni.cz/services/udpipe/api/models"
                elif backend == "udmorph":
                    cache_key = "udmorph:https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=list"
                elif backend == "nametag":
                    cache_key = "nametag:https://lindat.mff.cuni.cz/services/nametag/api/models"
                else:
                    cache_key = f"{backend}:default"
            else:
                # Local backends use simple cache keys
                if backend == "flexitag":
                    cache_key = "flexitag:local"
                else:
                    cache_key = backend
            
            # Read from cache with no TTL (use even expired cache)
            entries = read_model_cache_entry(cache_key, max_age_seconds=None)
            if not entries:
                # No cache - try loading normally (this will include remote models if available)
                entries = get_model_entries(
                    backend,
                    use_cache=True,
                    refresh_cache=False,
                    verbose=False,
                )
            
            for model_name, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                
                # Create unified key
                catalog_key = f"{backend}:{model_name}"
                
                # Copy entry and add catalog-specific fields
                catalog_entry = dict(entry)
                catalog_entry["backend"] = backend
                catalog_entry["model"] = model_name
                
                # Normalize language codes using language mapping
                try:
                    from .language_mapping import normalize_language_code
                    lang_iso = entry.get("language_iso")
                    if lang_iso:
                        iso_1, iso_2, iso_3 = normalize_language_code(lang_iso)
                        if iso_1:
                            catalog_entry["language_iso"] = iso_1
                except Exception:
                    pass  # Keep original if normalization fails
                
                # Determine if available without download
                status = (entry.get("status") or "").lower()
                source = entry.get("source", "local")
                if source == "local":
                    catalog_entry["available_without_download"] = status == "installed"
                else:
                    # Remote models are not available without download
                    catalog_entry["available_without_download"] = False
                
                # Preserve source information
                catalog_entry["source"] = source
                
                catalog_entry["preferred"] = bool(entry.get("preferred", False))
                
                # Preserve backend version compatibility if present
                if "backend_version" in entry:
                    catalog_entry["backend_version"] = entry["backend_version"]
                
                # Set default ratings (can be overridden by preference rules)
                catalog_entry["speed_rating"] = None
                catalog_entry["accuracy_rating"] = None
                
                catalog[catalog_key] = catalog_entry
                
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to load models for backend '{backend}': {exc}")
            continue
    
    # Apply preference rules to mark preferred models
    _apply_preference_rules(catalog)
    
    # Cache the catalog
    write_model_cache_entry(CATALOG_CACHE_KEY, catalog)
    
    if verbose:
        print(f"[flexipipe] Built unified catalog with {len(catalog)} models")
    
    return catalog


def _assign_model_ratings(entry: Dict[str, Any]) -> None:
    model_name = (entry.get("model") or "").lower()
    # Speed rating
    if "trf" in model_name or "transformer" in model_name:
        entry["speed_rating"] = 2
    elif "lg" in model_name or "large" in model_name:
        entry["speed_rating"] = 3
    elif "md" in model_name or "medium" in model_name:
        entry["speed_rating"] = 4
    elif "sm" in model_name or "small" in model_name:
        entry["speed_rating"] = 5
    else:
        entry["speed_rating"] = 3

    if "trf" in model_name or "transformer" in model_name:
        entry["accuracy_rating"] = 5
    elif "lg" in model_name or "large" in model_name:
        entry["accuracy_rating"] = 4
    elif "md" in model_name or "medium" in model_name:
        entry["accuracy_rating"] = 3
    elif "sm" in model_name or "small" in model_name:
        entry["accuracy_rating"] = 2
    else:
        entry["accuracy_rating"] = 3


def _apply_preference_rules(catalog: Dict[str, Dict[str, Any]]) -> None:
    """
    Apply preference rules to mark preferred models.
    
    Rules (in priority order):
    1. Prefer installed models over available ones
    2. For each language+backend combination, prefer:
       - Larger models (lg > md > sm) for accuracy
       - Transformer models for accuracy (if available)
       - Smaller models (sm > md > lg) for speed
    3. Backend priority: flexitag > spacy > stanza > others
    """
    from .__main__ import _get_language_backend_priority
    
    # Group by language
    by_language: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for key, entry in catalog.items():
        lang_iso = entry.get("language_iso")
        if not lang_iso:
            continue
        if lang_iso not in by_language:
            by_language[lang_iso] = []
        by_language[lang_iso].append((key, entry))
    
    backend_priority_list = _get_language_backend_priority()
    backend_priority = {name: idx for idx, name in enumerate(backend_priority_list) if name}
    
    for lang_iso, models in by_language.items():
        existing_preferred = [entry for _, entry in models if entry.get("preferred")]
        if existing_preferred:
            for entry in existing_preferred:
                _assign_model_ratings(entry)
            continue

        # Sort by: installed first, then backend priority, then model name
        def sort_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, int, str]:
            key, entry = item
            backend = entry.get("backend", "")
            installed = 0 if entry.get("available_without_download") else 1
            backend_rank = backend_priority.get(backend, 999)
            model_name = entry.get("model", "")
            return (installed, backend_rank, model_name)
        
        models.sort(key=sort_key)
        
        # Mark first model as preferred (best balance of installed + backend priority)
        if models:
            _, preferred_entry = models[0]
            preferred_entry["preferred"] = True
            _assign_model_ratings(preferred_entry)


def get_models_for_language(
    language: str,
    *,
    preferred_only: bool = False,
    available_only: bool = False,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get models for a specific language from the unified catalog.
    
    Args:
        language: Language ISO code or name
        preferred_only: If True, only return preferred models
        available_only: If True, only return models available without download
        use_cache: If True, use cached catalog
        
    Returns:
        List of model entry dicts matching the criteria
    """
    catalog = build_unified_catalog(use_cache=use_cache, refresh_cache=False, verbose=False)
    
    from .language_utils import resolve_language_query, language_matches_entry
    
    query = resolve_language_query(language)
    matches: List[Dict[str, Any]] = []
    
    for entry in catalog.values():
        if not language_matches_entry(entry, query, allow_fuzzy=True):
            continue
        
        if preferred_only and not entry.get("preferred", False):
            continue
        
        if available_only and not entry.get("available_without_download", False):
            continue
        
        matches.append(entry)
    
    return matches


def get_preferred_model_for_language(
    language: str,
    *,
    prefer_available: bool = True,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get the preferred model for a language.
    
    Args:
        language: Language ISO code or name
        prefer_available: If True, prefer models available without download
        use_cache: If True, use cached catalog
        
    Returns:
        Model entry dict or None if no model found
    """
    models = get_models_for_language(
        language,
        preferred_only=False,
        available_only=False,
        use_cache=use_cache,
    )
    
    if not models:
        return None
    
    # Sort by: preferred flag, then available, then backend priority
    from .__main__ import _get_language_backend_priority
    backend_priority_list = _get_language_backend_priority()
    backend_priority = {name: idx for idx, name in enumerate(backend_priority_list) if name}
    
    def sort_key(entry: Dict[str, Any]) -> tuple[int, int, int]:
        preferred = 0 if entry.get("preferred", False) else 1
        available = 0 if entry.get("available_without_download", False) else 1
        backend = entry.get("backend", "")
        backend_rank = backend_priority.get(backend, 999)
        return (preferred, available if prefer_available else 0, backend_rank)
    
    models.sort(key=sort_key)
    return models[0] if models else None

