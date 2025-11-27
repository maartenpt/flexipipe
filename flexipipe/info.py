"""Information listing commands for flexipipe."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from .backend_registry import get_backend_info, list_models_display
from .task_registry import TASK_ALIASES, TASK_DEFAULTS, TASK_DESCRIPTIONS, TASK_MANDATORY


def list_backends(args: argparse.Namespace) -> int:
    """List all available backends."""
    output_format = getattr(args, "output_format", "table")
    
    # Get backends from registry (excludes hidden backends by default)
    from .backend_registry import list_backends
    backends = list_backends(include_hidden=False)
    backends_data = [
        {"backend": name, "description": info.description}
        for name, info in sorted(backends.items())
    ]
    
    if output_format == "json":
        print(json.dumps({"backends": backends_data}, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format
    print("Available backends:")
    print(f"{'Backend':<20} {'Description':<60}")
    print("=" * 80)
    
    for backend_info in backends_data:
        print(f"{backend_info['backend']:<20} {backend_info['description']:<60}")
    
    print("\nUse 'flexipipe info models --backend <backend>' to see available models for a specific backend.")
    return 0


def list_examples(args: argparse.Namespace) -> int:
    """List locally installed example datasets."""
    output_format = getattr(args, "output_format", "table")
    refresh = getattr(args, "refresh", False)
    try:
        from .examples_data import load_examples
        examples = load_examples(refresh=refresh)
    except Exception as exc:
        print(f"[flexipipe] Error loading examples: {exc}", file=sys.stderr)
        return 1
    entries = []
    for code, entry in sorted(examples.items(), key=lambda kv: kv[0]):
        if not entry.get("text"):
            continue
        entries.append(
            {
                "code": code,
                "display": entry.get("display"),
                "iso3": entry.get("iso3"),
                "characters": len(entry.get("text", "")),
            }
        )
    total = len(entries)
    if output_format == "json":
        print(json.dumps({"examples": entries, "total": total}, indent=2, ensure_ascii=False), flush=True)
        return 0
    print("Available examples:")
    print(f"{'Code':<12} {'ISO3':<8} {'Display Name':<30} {'Chars':<8}")
    print("=" * 70)
    for entry in entries:
        display = entry.get("display") or ""
        print(
            f"{entry['code']:<12} {entry.get('iso3') or '':<8} {display[:28]:<30} {entry['characters']:<8}"
        )
    print(f"\nTotal examples: {total}")
    print("Examples are stored under <models-dir>/examples. Use --refresh to re-download metadata.")
    return 0


def list_tasks(args: argparse.Namespace) -> int:
    """List supported NLP tasks."""
    output_format = getattr(args, "output_format", "table")
    entries = []
    for task in sorted(TASK_DESCRIPTIONS.keys()):
        aliases = sorted(alias for alias in TASK_ALIASES.get(task, set()) if alias != task)
        entries.append(
            {
                "task": task,
                "description": TASK_DESCRIPTIONS.get(task, ""),
                "aliases": aliases,
                "default": task in TASK_DEFAULTS,
                "mandatory": task in TASK_MANDATORY,
            }
        )
    total = len(entries)
    if output_format == "json":
        print(json.dumps({"tasks": entries, "total": total}, indent=2, ensure_ascii=False), flush=True)
        return 0
    print("Supported tasks:")
    print(f"{'Task':<12} {'Default':<8} {'Mandatory':<10} {'Description':<45} {'Aliases'}")
    print("=" * 100)
    for entry in entries:
        alias_str = ", ".join(entry["aliases"]) if entry["aliases"] else "-"
        desc = entry["description"][:43] + "..." if len(entry["description"]) > 46 else entry["description"]
        print(
            f"{entry['task']:<12} {str(entry['default']):<8} {str(entry['mandatory']):<10} {desc:<45} {alias_str}"
        )
    print(f"\nTotal tasks: {total}")
    return 0


def list_models(args: argparse.Namespace) -> int:
    """List available models for the specified backend."""
    import time
    from .__main__ import _load_backend_entries, _display_language_filtered_models, _get_language_backend_priority
    from .language_utils import LANGUAGE_FIELD_ISO, LANGUAGE_FIELD_NAME
    
    start_time = time.time()
    debug = getattr(args, "debug", False)
    backend_type = getattr(args, "backend", None)
    language_filter = getattr(args, "language", None)
    force_refresh = bool(getattr(args, "refresh_cache", False))
    use_cache = not force_refresh
    
    if debug:
        print("[DEBUG] Starting model listing...", file=sys.stderr)
        print(f"[DEBUG] Backend: {backend_type or 'all'}", file=sys.stderr)
        print(f"[DEBUG] Language filter: {language_filter or 'none'}", file=sys.stderr)
        print(f"[DEBUG] Refresh cache: {force_refresh}", file=sys.stderr)

    if language_filter:
        # Use unified catalog for fast language filtering
        if debug:
            catalog_start = time.time()
            print("[DEBUG] Using unified catalog for language filtering...", file=sys.stderr)
        
        from .model_catalog import get_models_for_language
        
        try:
            models = get_models_for_language(
                language_filter,
                preferred_only=False,
                available_only=False,
                use_cache=True,  # Always use cache for speed
            )
            
            if debug:
                catalog_time = time.time() - catalog_start
                print(f"[DEBUG] Unified catalog lookup: {catalog_time:.3f}s ({len(models)} models found)", file=sys.stderr)
            
            if not models:
                output_format = getattr(args, "output_format", "table")
                if output_format == "json":
                    print(json.dumps({"language": language_filter, "models": []}, indent=2, ensure_ascii=False), flush=True)
                else:
                    print(f"No models found for language '{language_filter}'.")
                if debug:
                    total_time = time.time() - start_time
                    print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
                return 0
            
            # Convert to entries_by_backend format for compatibility with _display_language_filtered_models
            if debug:
                convert_start = time.time()
            entries_by_backend: dict[str, dict] = {}
            for model_entry in models:
                backend = model_entry.get("backend")
                model_name = model_entry.get("model")
                if not backend or not model_name:
                    continue
                if backend not in entries_by_backend:
                    entries_by_backend[backend] = {}
                entries_by_backend[backend][model_name] = model_entry
            
            if debug:
                convert_time = time.time() - convert_start
                print(f"[DEBUG] Converting to entries_by_backend format: {convert_time:.3f}s", file=sys.stderr)
            
            output_format = getattr(args, "output_format", "table")
            sort_by = getattr(args, "sort", "backend")
            
            if debug:
                display_start = time.time()
            result = _display_language_filtered_models(language_filter, entries_by_backend, output_format=output_format, sort_by=sort_by)
            if debug:
                display_time = time.time() - display_start
                total_time = time.time() - start_time
                print(f"[DEBUG] Display formatting: {display_time:.3f}s", file=sys.stderr)
                print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
            return result
            
        except Exception as exc:
            # Fallback to old method if catalog fails
            if getattr(args, "verbose", False) or getattr(args, "debug", False):
                print(f"[flexipipe] Warning: unified catalog failed, falling back to per-backend loading: {exc}", file=sys.stderr)
            
            # Fall through to old method
            if debug:
                fallback_start = time.time()
                print("[DEBUG] Falling back to per-backend loading...", file=sys.stderr)
            
            backend_priority_list = _get_language_backend_priority()
            backends_to_check = [backend_type.lower()] if backend_type else backend_priority_list
            entries_by_backend: dict[str, dict] = {}
            failed_backends: list[str] = []
            backend_timings: dict[str, float] = {}
            
            if debug:
                print(f"[DEBUG] Checking backends: {', '.join([b for b in backends_to_check if b])}", file=sys.stderr)
            
            for backend in backends_to_check:
                if backend is None:
                    continue
                try:
                    # When filtering by language, use cache directly (even if expired) to avoid slow operations
                    # This makes language filtering much faster - it should only read JSON files
                    # Skip HTTP requests for REST backends and directory scans for local backends
                    if not force_refresh:
                        from .model_storage import read_model_cache_entry
                        backend_info = get_backend_info(backend)
                        is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
                        
                        # Determine cache key based on backend type
                        if is_rest_backend:
                            # REST backends use cache keys like "udpipe:{url}" or "udmorph:{url}"
                            url = getattr(args, "endpoint_url", None)
                            if backend == "udpipe":
                                cache_key = f"udpipe:{url or 'https://lindat.mff.cuni.cz/services/udpipe/api/models'}"
                            elif backend == "udmorph":
                                cache_key = f"udmorph:{url or 'https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=list'}"
                            elif backend == "nametag":
                                cache_key = f"nametag:{url or 'https://lindat.mff.cuni.cz/services/nametag/api/models'}"
                            else:
                                cache_key = f"{backend}:{url or 'default'}"
                        else:
                            # Local backends use simple cache keys
                            if backend == "flexitag":
                                cache_key = "flexitag:local"
                            else:
                                cache_key = backend
                        
                        # Check cache with no TTL - use even expired cache to avoid slow operations
                        cached = read_model_cache_entry(cache_key, max_age_seconds=None)
                        if cached:
                            # Use cached data directly - don't call backend function which might be slow
                            entries_by_backend[backend] = cached
                            continue
                        else:
                            # No cache - skip this backend to avoid slow operations
                            entries_by_backend[backend] = {}
                            continue
                    
                    # Only load normally if force_refresh is True
                    effective_use_cache = use_cache
                    effective_refresh_cache = force_refresh
                    
                    entries = _load_backend_entries(
                        backend,
                        args,
                        use_cache=effective_use_cache,
                        refresh_cache=effective_refresh_cache,
                        verbose=bool(getattr(args, "verbose", False)),
                    )
                    # Only add if we got entries (non-empty dict)
                    # Empty dicts are valid (no models), but we still want to include them
                    # so that _display_language_filtered_models can check for matches
                    entries_by_backend[backend] = entries
                except Exception as exc:  # pragma: no cover - defensive
                    output_format = getattr(args, "output_format", "table")
                    failed_backends.append(backend)
                    # Don't add error dicts to entries_by_backend - just skip the backend
                    # The error will be handled later if needed
                    # Show errors by default when filtering by language (to help diagnose issues)
                    if output_format != "json":
                        print(f"[flexipipe] Failed to load models for backend '{backend}': {exc}", file=sys.stderr)
                    # Skip this backend - don't add it to entries_by_backend
                    continue
            if not entries_by_backend:
                output_format = getattr(args, "output_format", "table")
                if output_format == "json":
                    error_msg = "No backends available for language filtering."
                    if failed_backends:
                        error_msg += f" Failed backends: {', '.join(failed_backends)}"
                    print(json.dumps({"error": error_msg}, indent=2), flush=True)
                else:
                    error_msg = "[flexipipe] No backends available for language filtering."
                    if failed_backends:
                        error_msg += f" (Failed to load: {', '.join(failed_backends)})"
                    print(error_msg)
                return 1
            output_format = getattr(args, "output_format", "table")
            sort_by = getattr(args, "sort", "backend")
            
            if debug:
                fallback_time = time.time() - fallback_start
                print(f"[DEBUG] Fallback loading: {fallback_time:.3f}s", file=sys.stderr)
                if backend_timings:
                    total_backend_time = sum(backend_timings.values())
                    print(f"[DEBUG] Backend loading total: {total_backend_time:.3f}s", file=sys.stderr)
                display_start = time.time()
            
            result = _display_language_filtered_models(language_filter, entries_by_backend, output_format=output_format, sort_by=sort_by)
            
            if debug:
                display_time = time.time() - display_start
                total_time = time.time() - start_time
                print(f"[DEBUG] Display formatting: {display_time:.3f}s", file=sys.stderr)
                print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
            return result

    if not backend_type:
        # No backend specified - list all models from all backends
        if debug:
            print("[DEBUG] No backend specified - listing models from all backends...", file=sys.stderr)
            backend_priority_list = _get_language_backend_priority()
            print(f"[DEBUG] Checking backends: {', '.join([b for b in backend_priority_list if b])}", file=sys.stderr)
        
        output_format = getattr(args, "output_format", "table")
        from .__main__ import _load_backend_entries
        backend_priority_list = _get_language_backend_priority()
        
        entries_by_backend: dict[str, dict] = {}
        backend_timings: dict[str, float] = {}
        
        for backend in backend_priority_list:
            if backend is None:
                continue
            if debug:
                backend_start = time.time()
                print(f"[DEBUG] Checking backend: {backend}...", file=sys.stderr)
            try:
                entries = _load_backend_entries(
                    backend,
                    args,
                    use_cache=use_cache,
                    refresh_cache=force_refresh,
                    verbose=bool(getattr(args, "verbose", False)),
                )
                if entries:  # Only add if we got entries
                    entries_by_backend[backend] = entries
                    if debug:
                        backend_time = time.time() - backend_start
                        backend_timings[backend] = backend_time
                        print(f"[DEBUG]   {backend}: {backend_time:.3f}s ({len(entries)} models)", file=sys.stderr)
            except Exception as exc:  # pragma: no cover - defensive
                # Skip backends that fail to load
                if debug:
                    backend_time = time.time() - backend_start
                    print(f"[DEBUG]   {backend}: {backend_time:.3f}s (FAILED: {exc})", file=sys.stderr)
                # Only show in debug/verbose mode
                if output_format != "json" and (getattr(args, "debug", False) or getattr(args, "verbose", False)):
                    print(f"[flexipipe] Failed to load models for backend '{backend}': {exc}")
                continue
        
        if not entries_by_backend:
            if output_format == "json":
                print(json.dumps({"error": "No backends available for listing models."}, indent=2), flush=True)
            else:
                print("[flexipipe] No backends available for listing models.")
            return 1
        
        # Display all models from all backends
        sort_by = getattr(args, "sort", "backend")
        if debug:
            display_start = time.time()
        result = _display_language_filtered_models(None, entries_by_backend, output_format=output_format, sort_by=sort_by)
        if debug:
            display_time = time.time() - display_start
            total_time = time.time() - start_time
            print(f"[DEBUG] Display formatting: {display_time:.3f}s", file=sys.stderr)
            if backend_timings:
                total_backend_time = sum(backend_timings.values())
                print(f"[DEBUG] Backend loading total: {total_backend_time:.3f}s", file=sys.stderr)
            print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
        return result
    
    backend_type = backend_type.lower()
    output_format = getattr(args, "output_format", "table")
    
    # Show models directory location (only for table format, not JSON)
    if output_format != "json":
        from .model_storage import get_flexipipe_models_dir, get_config_file, read_config
        models_dir = get_flexipipe_models_dir(create=False)
        config = read_config()
        if "models_dir" in config:
            print(f"[flexipipe] Models directory: {models_dir} (configured in {get_config_file()})")
        else:
            print(f"[flexipipe] Models directory: {models_dir} (default)")
            print(f"[flexipipe] To change: export FLEXIPIPE_MODELS_DIR=/path/to/models or edit {get_config_file()}")
    
    # Use the registry for most backends
    # get_backend_info is already imported at the top of the file
    
    # For JSON output, collect entries and format them
    if output_format == "json":
        try:
            entries = _load_backend_entries(
                backend_type,
                args,
                use_cache=use_cache,
                refresh_cache=force_refresh,
                verbose=bool(getattr(args, "verbose", False)),
            )
            
            # Check if entries is actually an error dict (from exception handling in language filtering)
            # This happens when _load_backend_entries catches an exception and returns {"error": "..."}
            if isinstance(entries, dict) and len(entries) == 1 and "error" in entries:
                # This is an error response, not actual model entries
                print(json.dumps({"error": entries["error"], "backend": backend_type}, indent=2), flush=True)
                return 1
            
            models_data = []
            for model_name, entry in entries.items():
                if isinstance(entry, dict):
                    model_info = {
                        "model": model_name,
                        "language_iso": entry.get(LANGUAGE_FIELD_ISO),
                        "language_name": entry.get(LANGUAGE_FIELD_NAME) or entry.get("language_display"),
                        "status": entry.get("status"),
                        "version": entry.get("version") or entry.get("date") or entry.get("updated"),
                        "features": entry.get("features"),
                        "description": entry.get("description"),
                        "tasks": entry.get("tasks") or entry.get("task"),
                        "base_model": entry.get("base_model"),
                        "training_data": entry.get("training_data"),
                        "techniques": entry.get("techniques"),
                        "languages": entry.get("languages"),
                        "package": entry.get("package"),
                    }
                    # Remove None values
                    model_info = {k: v for k, v in model_info.items() if v is not None}
                    models_data.append(model_info)
            
            result = {
                "backend": backend_type,
                "models": models_data,
                "total": len(models_data),
            }
            # Flush stdout to ensure complete JSON output
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 0
        except Exception as exc:
            print(json.dumps({"error": str(exc), "backend": backend_type}, indent=2), flush=True)
            return 1

    # Check if backend is registered
    info = get_backend_info(backend_type)
    if not info:
        print(f"Error: Unknown backend '{backend_type}'")
        return 1
    
    # Handle REST backends that need URL parameters
    if backend_type.lower() in ("udpipe", "udmorph", "nametag"):
        # Use unified --endpoint-url for all REST backends
        url = getattr(args, "endpoint_url", None)
        return list_models_display(
            backend_type,
            url,  # Pass as positional argument
            use_cache=use_cache,
            refresh_cache=force_refresh,
            verbose=bool(getattr(args, "verbose", False)),
            output_format=output_format,
        )
    
    # All other backends use the registry (only pass supported kwargs)
    extra_kwargs = {
        "use_cache": use_cache,
        "refresh_cache": force_refresh,
        "verbose": bool(getattr(args, "verbose", False)),
    }
    if backend_type.lower() == "transformers":
        extra_kwargs["include_llm"] = bool(getattr(args, "include_base_models", False))
    return list_models_display(
        backend_type,
        **extra_kwargs,
    )


def list_languages(args: argparse.Namespace) -> int:
    """List all languages that have models available."""
    import time
    from .model_catalog import build_unified_catalog
    from .language_mapping import get_language_metadata
    
    start_time = time.time()
    debug = getattr(args, "debug", False)
    output_format = getattr(args, "output_format", "table")
    
    if debug:
        print("[DEBUG] Building unified catalog to extract language information...", file=sys.stderr)
        catalog_start = time.time()
    
    # Build unified catalog to get all models
    catalog = build_unified_catalog(use_cache=True, refresh_cache=False, verbose=debug)
    
    if debug:
        catalog_time = time.time() - catalog_start
        print(f"[DEBUG] Catalog loaded: {catalog_time:.3f}s ({len(catalog)} models)", file=sys.stderr)
    
    # Collect language information - group by normalized ISO-1 code to avoid duplicates
    languages_data: Dict[str, Dict[str, Any]] = {}
    
    for catalog_key, entry in catalog.items():
        lang_iso = entry.get("language_iso")
        lang_name = entry.get("language_name")
        
        if not lang_iso:
            continue
        
        # Get comprehensive language metadata to normalize the ISO code
        lang_metadata = get_language_metadata(lang_iso)
        if not lang_metadata.get("iso_639_1"):
            # Try with language name if ISO lookup failed
            if lang_name:
                lang_metadata = get_language_metadata(lang_name)
        
        # Use normalized ISO-1 code as the key (or ISO-3 if no ISO-1)
        normalized_key = lang_metadata.get("iso_639_1") or lang_metadata.get("iso_639_3") or lang_iso
        
        if normalized_key not in languages_data:
            languages_data[normalized_key] = {
                "iso_639_1": lang_metadata.get("iso_639_1"),
                "iso_639_2": lang_metadata.get("iso_639_2"),
                "iso_639_3": lang_metadata.get("iso_639_3") or normalized_key,
                "primary_name": lang_metadata.get("primary_name") or lang_name or normalized_key,
                "model_count": 0,
                "backend_count": 0,
                "backends": set(),
            }
        
        languages_data[normalized_key]["model_count"] += 1
        backend = entry.get("backend", "")
        if backend:
            languages_data[normalized_key]["backends"].add(backend)
            languages_data[normalized_key]["backend_count"] = len(languages_data[normalized_key]["backends"])
    
    # Sort by ISO code
    sorted_languages = sorted(languages_data.items(), key=lambda x: x[0])
    
    if output_format == "json":
        languages_list = []
        for lang_iso, lang_data in sorted_languages:
            languages_list.append({
                "iso_639_1": lang_data["iso_639_1"],
                "iso_639_2": lang_data.get("iso_639_2"),
                "iso_639_3": lang_data["iso_639_3"],
                "name": lang_data["primary_name"],
                "model_count": lang_data["model_count"],
                "backend_count": lang_data["backend_count"],
                "backends": sorted(lang_data["backends"]),
            })
        
        result = {
            "languages": languages_list,
            "total": len(languages_list),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if debug:
            total_time = time.time() - start_time
            print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
        return 0
    
    # Table format
    print("\nLanguages with available models:")
    print(f"{'ISO-1':<8} {'ISO-2':<8} {'ISO-3':<8} {'Language Name':<25} {'Models':<8} {'Backends':<10} {'Backend List'}")
    print("=" * 120)
    
    for lang_iso, lang_data in sorted_languages:
        iso_1 = lang_data["iso_639_1"] or "-"
        iso_2 = lang_data.get("iso_639_2") or "-"
        iso_3 = lang_data["iso_639_3"] or "-"
        name = lang_data["primary_name"]
        model_count = lang_data["model_count"]
        backend_count = lang_data["backend_count"]
        backends_str = ", ".join(sorted(lang_data["backends"]))
        
        print(f"{iso_1:<8} {iso_2:<8} {iso_3:<8} {name:<25} {model_count:<8} {backend_count:<10} {backends_str}")
    
    print(f"\nTotal: {len(sorted_languages)} language(s) with {sum(d['model_count'] for _, d in sorted_languages)} model(s)")
    
    if debug:
        total_time = time.time() - start_time
        print(f"[DEBUG] Total execution time: {total_time:.3f}s", file=sys.stderr)
    
    return 0


def list_ud_tags(args: argparse.Namespace) -> int:
    """List UD tags repository information."""
    from .ud_tags_repository import load_repository
    
    output_format = getattr(args, "output_format", "table")
    category = getattr(args, "category", "all")
    
    repo = load_repository()
    
    if output_format == "json":
        result = {
            "repository": repo,
            "category": category,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format
    print("Universal Dependencies Tags Repository")
    print("=" * 100)
    
    if category in ("upos", "all"):
        print("\nUPOS Tags")
        print("-" * 100)
        print(f"{'Category':<15} {'Count':<10} {'Tags'}")
        print("-" * 100)
        std_tags = ", ".join(repo['upos']['standard'])
        print(f"{'Standard':<15} {len(repo['upos']['standard']):<10} {std_tags}")
        if repo['upos']['extended']:
            ext_tags = ", ".join(repo['upos']['extended'])
            print(f"{'Extended':<15} {len(repo['upos']['extended']):<10} {ext_tags}")
        else:
            print(f"{'Extended':<15} {0:<10} (none)")
    
    if category in ("feats", "all"):
        print("\nFEATS - Standard Features")
        print("-" * 100)
        print(f"{'Feature':<20} {'Values':<10} {'UPOS Count':<12} {'Sample Values'}")
        print("-" * 100)
        for feat_name, feat_info in sorted(repo['feats']['standard'].items()):
            values = feat_info.get('values', [])
            upos_list = feat_info.get('upos', [])
            sample = ", ".join(values[:5])
            if len(values) > 5:
                sample += f" ... ({len(values)} total)"
            print(f"{feat_name:<20} {len(values):<10} {len(upos_list):<12} {sample}")
        
        if repo['feats']['extended']:
            print("\nFEATS - Extended Features")
            print("-" * 100)
            print(f"{'Feature':<20} {'Values':<10} {'UPOS Count':<12} {'Sample Values'}")
            print("-" * 100)
            for feat_name, feat_info in sorted(repo['feats']['extended'].items()):
                values = feat_info.get('values', [])
                upos_list = feat_info.get('upos', [])
                sample = ", ".join(values[:5])
                if len(values) > 5:
                    sample += f" ... ({len(values)} total)"
                print(f"{feat_name:<20} {len(values):<10} {len(upos_list):<12} {sample}")
    
    if category in ("misc", "all"):
        print("\nMISC Fields - Standard")
        print("-" * 100)
        print(f"{'Field':<25} {'Doc Field':<25} {'Doc Type':<20} {'Print As'}")
        print("-" * 100)
        for misc_name, misc_info in sorted(repo['misc']['standard'].items()):
            doc_field = misc_info.get('doc_field') or 'N/A'
            doc_type = misc_info.get('doc_type') or 'N/A'
            print_as = misc_info.get('print_as') or misc_name or 'N/A'
            misc_name_str = str(misc_name) if misc_name else 'N/A'
            print(f"{misc_name_str:<25} {str(doc_field):<25} {str(doc_type):<20} {str(print_as)}")
        
        if repo['misc']['extended']:
            print("\nMISC Fields - Extended")
            print("-" * 100)
            print(f"{'Field':<25} {'Doc Field':<25} {'Doc Type':<20} {'Print As'}")
            print("-" * 100)
            for misc_name, misc_info in sorted(repo['misc']['extended'].items()):
                doc_field = misc_info.get('doc_field') or 'N/A'
                doc_type = misc_info.get('doc_type') or 'N/A'
                print_as = misc_info.get('print_as') or misc_name or 'N/A'
                misc_name_str = str(misc_name) if misc_name else 'N/A'
                print(f"{misc_name_str:<25} {str(doc_field):<25} {str(doc_type):<20} {str(print_as)}")
    
    if category in ("document", "all"):
        print("\nDocument-Level Fields")
        print("-" * 100)
        print(f"{'Field':<20} {'Doc Field':<25} {'Print As':<20} {'Description'}")
        print("-" * 100)
        for field_name, field_info in sorted(repo['document_fields'].items()):
            doc_field = field_info.get('doc_field', 'N/A')
            print_as = field_info.get('print_as', field_name)
            desc = field_info.get('description', '')
            # Truncate long descriptions
            if len(desc) > 40:
                desc = desc[:37] + "..."
            print(f"{field_name:<20} {doc_field:<25} {print_as:<20} {desc}")
    
    if category in ("sentence", "all"):
        print("\nSentence-Level Fields")
        print("-" * 100)
        print(f"{'Field':<20} {'Doc Field':<25} {'Print As':<20} {'Description'}")
        print("-" * 100)
        for field_name, field_info in sorted(repo['sentence_fields'].items()):
            doc_field = field_info.get('doc_field', 'N/A')
            print_as = field_info.get('print_as', field_name)
            desc = field_info.get('description', '')
            # Truncate long descriptions
            if len(desc) > 40:
                desc = desc[:37] + "..."
            print(f"{field_name:<20} {doc_field:<25} {print_as:<20} {desc}")
    
    print("\n" + "=" * 100)
    print(f"Treebanks scanned: {len(repo.get('treebanks_scanned', []))}")
    if repo.get('last_updated'):
        print(f"Last updated: {repo['last_updated']}")
    print("\nTo update the repository, run:")
    print("  python -m flexipipe.scripts.scan_ud_treebanks --treebank-root /path/to/ud-treebanks")
    
    return 0


def run_info_cli(args: argparse.Namespace) -> int:
    """Run the info subcommand."""
    # Handle --detect-language if provided
    if getattr(args, "detect_language", False):
        from .__main__ import _run_detect_language_standalone
        # Build argv for detect-language
        detect_argv = ["--detect-language"]
        if hasattr(args, "text") and args.text:
            detect_argv.append("--text")
            detect_argv.append(args.text)
        if hasattr(args, "input") and args.input:
            detect_argv.append("--input")
            detect_argv.append(args.input)
        if hasattr(args, "min_length"):
            detect_argv.append("--min-length")
            detect_argv.append(str(args.min_length))
        if hasattr(args, "top_k"):
            detect_argv.append("--top-k")
            detect_argv.append(str(args.top_k))
        if getattr(args, "verbose", False):
            detect_argv.append("--verbose")
        return _run_detect_language_standalone(detect_argv)
    
    # Require an action if detect-language not used
    if not hasattr(args, "info_action") or not args.info_action:
        print(
            "Error: No action specified. Use one of: backends, models, languages, ud-tags, examples, tasks, or --detect-language"
        )
        return 1
    
    if args.info_action == "backends":
        return list_backends(args)
    elif args.info_action == "models":
        return list_models(args)
    elif args.info_action == "languages":
        return list_languages(args)
    elif args.info_action == "ud-tags":
        return list_ud_tags(args)
    elif args.info_action == "examples":
        return list_examples(args)
    elif args.info_action == "tasks":
        return list_tasks(args)
    else:
        print(f"Error: Unknown info action '{args.info_action}'")
        return 1

