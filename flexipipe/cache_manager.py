"""
Unified cache management for flexipipe.

This module provides a single command to refresh all caches at once, and
warns when caches are outdated instead of automatically rebuilding them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model_storage import _load_model_cache, get_cache_dir


def get_cache_timestamps() -> Dict[str, Optional[float]]:
    """
    Get timestamps for all cache entries.
    
    Returns:
        Dictionary mapping cache keys to timestamps (or None if not cached)
    """
    cache = _load_model_cache()
    timestamps: Dict[str, Optional[float]] = {}
    
    for key, entry in cache.items():
        if isinstance(entry, dict):
            timestamp = entry.get("timestamp")
            if timestamp:
                timestamps[key] = float(timestamp)
            else:
                timestamps[key] = None
        else:
            timestamps[key] = None
    
    return timestamps


def check_cache_staleness(
    cache_key: str,
    max_age_seconds: Optional[float] = None,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Check if a cache entry is stale.
    
    Args:
        cache_key: Cache key to check
        max_age_seconds: Maximum age in seconds (None = no limit)
        
    Returns:
        Tuple of (is_stale, age_seconds, timestamp)
    """
    timestamps = get_cache_timestamps()
    timestamp = timestamps.get(cache_key)
    
    if timestamp is None:
        return (True, None, None)  # Not cached = stale
    
    age_seconds = time.time() - timestamp
    is_stale = max_age_seconds is not None and age_seconds > max_age_seconds
    
    return (is_stale, age_seconds, timestamp)


def get_all_cache_keys() -> List[str]:
    """
    Get all cache keys currently in the cache.
    
    Returns:
        List of cache keys
    """
    cache = _load_model_cache()
    return list(cache.keys())


def refresh_all_caches(
    *,
    verbose: bool = False,
    force: bool = False,
) -> Dict[str, bool]:
    """
    Refresh all model caches at once.
    
    This ensures all caches are consistent since they depend on each other.
    
    Args:
        verbose: If True, print progress messages
        force: If True, refresh even if cache is recent
        
    Returns:
        Dictionary mapping cache keys to success status
    """
    from .backend_registry import get_backend_info, get_model_entries
    from .__main__ import _get_language_backend_priority
    from .model_catalog import build_unified_catalog
    
    results: Dict[str, bool] = {}
    
    if verbose:
        print("[flexipipe] Refreshing all model caches...")
    
    # Refresh per-backend caches first
    for backend in _get_language_backend_priority():
        if backend is None:
            continue
        try:
            backend_info = get_backend_info(backend)
            if not backend_info or not backend_info.get_model_entries:
                continue
            
            if verbose:
                print(f"[flexipipe] Refreshing {backend} cache...")
            
            # Determine cache key
            is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
            if is_rest_backend:
                if backend == "udpipe":
                    cache_key = "udpipe:https://lindat.mff.cuni.cz/services/udpipe/api/models"
                elif backend == "udmorph":
                    cache_key = "udmorph:https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=list"
                elif backend == "nametag":
                    cache_key = "nametag:https://lindat.mff.cuni.cz/services/nametag/api/models"
                else:
                    cache_key = f"{backend}:default"
            else:
                if backend == "flexitag":
                    cache_key = "flexitag:local"
                else:
                    cache_key = backend
            
            try:
                entries = get_model_entries(
                    backend,
                    use_cache=False,  # Force refresh
                    refresh_cache=True,
                    verbose=verbose,
                )
                results[cache_key] = True
            except Exception as exc:
                if verbose:
                    print(f"[flexipipe] Warning: failed to refresh {backend} cache: {exc}")
                results[cache_key] = False
                
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to refresh {backend}: {exc}")
            results[backend] = False
    
    # Refresh unified catalog (depends on per-backend caches)
    if verbose:
        print("[flexipipe] Refreshing unified model catalog...")
    try:
        catalog = build_unified_catalog(
            use_cache=False,  # Force refresh
            refresh_cache=True,
            verbose=verbose,
        )
        results["unified_model_catalog"] = True
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to refresh unified catalog: {exc}")
        results["unified_model_catalog"] = False
    
    if verbose:
        success_count = sum(1 for v in results.values() if v)
        print(f"[flexipipe] Refreshed {success_count}/{len(results)} cache(s)")
    
    return results


def check_all_caches_staleness(
    max_age_seconds: Optional[float] = None,
) -> Dict[str, Tuple[bool, Optional[float]]]:
    """
    Check staleness of all caches.
    
    Args:
        max_age_seconds: Maximum age in seconds (None = no limit)
        
    Returns:
        Dictionary mapping cache keys to (is_stale, age_seconds) tuples
    """
    timestamps = get_cache_timestamps()
    results: Dict[str, Tuple[bool, Optional[float]]] = {}
    
    for key, timestamp in timestamps.items():
        if timestamp is None:
            results[key] = (True, None)
        else:
            age_seconds = time.time() - timestamp
            is_stale = max_age_seconds is not None and age_seconds > max_age_seconds
            results[key] = (is_stale, age_seconds)
    
    return results


def warn_if_caches_stale(
    max_age_seconds: Optional[float] = None,
    *,
    verbose: bool = False,
) -> bool:
    """
    Warn if any caches are stale.
    
    Args:
        max_age_seconds: Maximum age in seconds (None = no limit)
        verbose: If True, print detailed warnings
        
    Returns:
        True if any caches are stale, False otherwise
    """
    staleness = check_all_caches_staleness(max_age_seconds=max_age_seconds)
    stale_caches = [key for key, (is_stale, _) in staleness.items() if is_stale]
    
    if stale_caches:
        if verbose:
            print(f"[flexipipe] Warning: {len(stale_caches)} cache(s) are stale or missing:")
            for key in stale_caches:
                is_stale, age = staleness[key]
                if age is not None:
                    age_hours = age / 3600
                    print(f"  - {key}: {age_hours:.1f} hours old")
                else:
                    print(f"  - {key}: not cached")
            print("[flexipipe] Run 'python -m flexipipe config --refresh-all-caches' to refresh all caches.")
        return True
    
    return False

