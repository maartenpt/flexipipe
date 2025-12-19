"""Information listing commands for flexipipe."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from .backend_registry import get_backend_info, list_models_display, get_model_entries
from .task_registry import TASK_ALIASES, TASK_DEFAULTS, TASK_DESCRIPTIONS, TASK_MANDATORY
from .language_utils import resolve_language_query, language_matches_entry


def _capitalize_backend_name(name: str) -> str:
    """Return properly capitalized backend name."""
    # Mapping of backend names to their proper capitalization
    capitalization_map = {
        "classla": "ClassLA",
        "fasttext": "fastText",
        "flexitag": "Flexitag",
        "nametag": "NameTag",
        "spacy": "SpaCy",
        "stanza": "Stanza",
        "transformers": "Transformers",
        "treetagger": "TreeTagger",
        "udmorph": "UDMorph",
        "udpipe": "UDPipe",
        "udpipe1": "UDPipe CLI",
    }
    return capitalization_map.get(name.lower(), name.capitalize())


def list_backends(args: argparse.Namespace) -> int:
    """List all available backends with their functionality status."""
    output_format = getattr(args, "output_format", "table")
    
    # Get backends from registry (excludes hidden backends by default)
    from .backend_registry import list_backends, get_backend_status
    backends = list_backends(include_hidden=False)
    
    backends_data = []
    for name, info in sorted(backends.items()):
        status_info = get_backend_status(name)
        display_name = _capitalize_backend_name(name)
        backend_entry = {
            "backend": name,
            "name": display_name,  # Properly capitalized name
            "description": info.description,
            "url": getattr(info, "url", None) or "",
            "available": status_info["available"],
            "status": status_info["status"],
            "missing": status_info["missing"],
            "install_hint": status_info["install_hint"],
        }
        backends_data.append(backend_entry)
    
    if output_format == "json":
        print(json.dumps({"backends": backends_data}, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format - separate name, description, and URL
    print("Available backends:")
    print(f"{'Backend':<20} {'Status':<15} {'Description':<45} {'URL':<40}")
    print("=" * 120)
    
    for backend_info in backends_data:
        status_str = "✓ Available" if backend_info["available"] else "✗ Missing"
        if backend_info["missing"]:
            status_str += f" ({', '.join(backend_info['missing'])})"
        
        # Truncate description if too long
        desc = backend_info["description"]
        if len(desc) > 45:
            desc = desc[:42] + "..."
        
        # Truncate URL if too long
        url = backend_info["url"] or ""
        if len(url) > 40:
            url = url[:37] + "..."
        
        # Use properly capitalized name for display
        display_name = backend_info["name"]
        print(f"{display_name:<20} {status_str:<15} {desc:<45} {url:<40}")
    
    # Show missing dependencies summary
    missing_backends = [b for b in backends_data if not b["available"]]
    if missing_backends:
        print("\n" + "=" * 85)
        print("Installation hints for missing backends:")
        for backend_info in missing_backends:
            if backend_info["install_hint"]:
                print(f"  {backend_info['backend']}: {backend_info['install_hint']}")
    
    from .model_storage import is_running_from_teitok
    if not is_running_from_teitok():
        print("\nUse 'flexipipe info models --backend <backend>' to see available models for a specific backend.")
    return 0


def list_examples(args: argparse.Namespace) -> int:
    """List locally installed example datasets."""
    output_format = getattr(args, "output_format", "table")
    refresh = getattr(args, "refresh", False) or getattr(args, "refresh_cache", False)
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
                use_cache=not force_refresh,  # Use cache unless refresh is requested
                refresh_cache=force_refresh,  # Pass refresh flag through
            )
            
            if debug:
                catalog_time = time.time() - catalog_start
                print(f"[DEBUG] Unified catalog lookup: {catalog_time:.3f}s ({len(models)} models found)", file=sys.stderr)
            
            if not models:
                # No exact matches found - suggest similar languages using fuzzy matching (for display only)
                from .language_utils import resolve_language_query
                from .__main__ import _suggest_similar_languages
                query = resolve_language_query(language_filter)
                try:
                    from .model_catalog import build_unified_catalog
                    full_catalog = build_unified_catalog(use_cache=True, refresh_cache=False, verbose=False)
                    full_entries_by_backend: dict[str, dict] = {}
                    for entry in full_catalog.values():
                        backend = entry.get("backend")
                        model = entry.get("model")
                        if backend and model:
                            if backend not in full_entries_by_backend:
                                full_entries_by_backend[backend] = {}
                            full_entries_by_backend[backend][model] = entry
                    suggestions = _suggest_similar_languages(language_filter, full_entries_by_backend, query)
                except Exception:
                    suggestions = []
                
                output_format = getattr(args, "output_format", "table")
                if output_format == "json":
                    result = {"language": language_filter, "models": []}
                    if suggestions:
                        result["suggestions"] = suggestions
                    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
                else:
                    print(f"[flexipipe] No models found for language '{language_filter}'.")
                    if suggestions:
                        print(f"[flexipipe] Did you mean: {', '.join(suggestions)}?")
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
        # Use unified catalog for fast access (like info languages does)
        if debug:
            print("[DEBUG] No backend specified - using unified catalog...", file=sys.stderr)
        
        from .model_catalog import build_unified_catalog
        
        catalog = build_unified_catalog(
            use_cache=not force_refresh,
            refresh_cache=force_refresh,
            verbose=debug,
        )
        
        if not catalog:
            if output_format == "json":
                print(json.dumps({"error": "No models available."}, indent=2), flush=True)
            else:
                print("[flexipipe] No models available.")
            return 1
        
        # Convert catalog to entries_by_backend format for compatibility with _display_language_filtered_models
        if debug:
            convert_start = time.time()
        entries_by_backend: dict[str, dict] = {}
        for catalog_key, entry in catalog.items():
            backend = entry.get("backend")
            model_name = entry.get("model")
            if not backend or not model_name:
                continue
            if backend not in entries_by_backend:
                entries_by_backend[backend] = {}
            entries_by_backend[backend][model_name] = entry
        
        if debug:
            convert_time = time.time() - convert_start
            print(f"[DEBUG] Converted catalog to entries_by_backend: {convert_time:.3f}s ({len(catalog)} models)", file=sys.stderr)
        
        # Display all models from all backends
        output_format = getattr(args, "output_format", "table")
        sort_by = getattr(args, "sort", "backend")
        if debug:
            display_start = time.time()
        result = _display_language_filtered_models(None, entries_by_backend, output_format=output_format, sort_by=sort_by)
        if debug:
            display_time = time.time() - display_start
            total_time = time.time() - start_time
            print(f"[DEBUG] Display formatting: {display_time:.3f}s", file=sys.stderr)
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
            
            # Handle tuple return from some backends (e.g., SpaCy returns (entries, dir, standard_location_models))
            if isinstance(entries, tuple):
                entries = entries[0]
            
            # Check if entries is actually an error dict (from exception handling in language filtering)
            # This happens when _load_backend_entries catches an exception and returns {"error": "..."}
            if isinstance(entries, dict) and len(entries) == 1 and "error" in entries:
                # This is an error response, not actual model entries
                print(json.dumps({"error": entries["error"], "backend": backend_type}, indent=2), flush=True)
                return 1
            
            from .model_storage import is_model_installed
            models_data = []
            for model_name, entry in entries.items():
                if isinstance(entry, dict):
                    # Check if model is installed (skip REST backends)
                    installed = None
                    backend_info = get_backend_info(backend_type)
                    is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
                    if not is_rest_backend:
                        try:
                            installed = is_model_installed(backend_type, model_name)
                        except Exception:
                            # If check fails, leave as None
                            pass
                    
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
                    # Add source if present (for SpaCy and other backends that track model source)
                    if "source" in entry:
                        model_info["source"] = entry["source"]
                    # Add installed status if available
                    if installed is not None:
                        model_info["installed"] = installed
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
    from .language_mapping import get_language_metadata, reload_language_mappings
    
    start_time = time.time()
    debug = getattr(args, "debug", False)
    output_format = getattr(args, "output_format", "table")
    force_refresh = bool(getattr(args, "refresh_cache", False))
    
    if debug:
        print("[DEBUG] Building unified catalog to extract language information...", file=sys.stderr)
        catalog_start = time.time()

    # Refresh language mappings if requested
    if force_refresh:
        reload_language_mappings(refresh_cache=True, verbose=debug)
    
    # Build unified catalog to get all models
    catalog = build_unified_catalog(use_cache=not force_refresh, refresh_cache=force_refresh, verbose=debug)
    
    if debug:
        catalog_time = time.time() - catalog_start
        print(f"[DEBUG] Catalog loaded: {catalog_time:.3f}s ({len(catalog)} models)", file=sys.stderr)
    
    # Load language JSON to get dialect names and aliases (needed for dialect detection)
    from .language_mapping import _load_language_mappings_from_json
    json_mappings = _load_language_mappings_from_json(use_cache=not force_refresh, verbose=debug)
    dialect_names: Dict[str, Dict[str, str]] = {}  # {base_iso: {dialect_code: dialect_name}}
    dialect_aliases: Dict[str, Dict[str, str]] = {}  # {base_iso: {alias: dialect_code}}
    if json_mappings:
        for lang_entry in json_mappings:
            if not isinstance(lang_entry, dict):
                continue
            entry_iso_3 = lang_entry.get("iso_639_3")
            if entry_iso_3:
                base_iso_lower = entry_iso_3.lower()
                dialects = lang_entry.get("dialects", {})
                if isinstance(dialects, dict):
                    dialect_names[base_iso_lower] = {}
                    dialect_aliases[base_iso_lower] = {}
                    for dial_code, dial_info in dialects.items():
                        if isinstance(dial_info, dict):
                            dialect_names[base_iso_lower][dial_code] = dial_info.get("name", dial_code)
                            # Map aliases to dialect codes (e.g., "oci-ar" -> "ara", "oci-ara" -> "ara")
                            for alias in dial_info.get("aliases", []):
                                dialect_aliases[base_iso_lower][alias.lower()] = dial_code
    
    # Collect language information - group by normalized ISO-1 code to avoid duplicates
    languages_data: Dict[str, Dict[str, Any]] = {}
    
    # Helper to extract dialect code from language_iso (e.g., "oci-ar" -> "ar", "oci-ara" -> "ara")
    def _extract_dialect_code(lang_iso: str, base_iso: str) -> Optional[str]:
        """Extract dialect code from a language ISO code like 'oci-ar' or 'oci-ara'."""
        if not lang_iso or not base_iso:
            return None
        if lang_iso.lower().startswith(base_iso.lower() + "-"):
            dialect_part = lang_iso[len(base_iso) + 1:].lower()
            return dialect_part if dialect_part else None
        return None
    
    for catalog_key, entry in catalog.items():
        lang_iso = entry.get("language_iso")
        # Use original_language_iso if available (preserves dialect codes like "oci-ar")
        original_lang_iso = entry.get("original_language_iso") or lang_iso
        lang_name = entry.get("language_name")
        
        if not lang_iso:
            continue
        
        # Get comprehensive language metadata to normalize the ISO code
        # Always use the normalized lang_iso from catalog (which is already normalized to ISO-1)
        lang_metadata = get_language_metadata(lang_iso)
        if not lang_metadata.get("iso_639_1") and not lang_metadata.get("iso_639_3"):
            # If metadata lookup failed, try with language name
            if lang_name:
                lang_metadata = get_language_metadata(lang_name)
            # If still no metadata, try with original_language_iso (might be a dialect code)
            if (not lang_metadata.get("iso_639_1") and not lang_metadata.get("iso_639_3") 
                and original_lang_iso and original_lang_iso != lang_iso):
                # Try to get base language from original code (e.g., "oci-ara" -> "oci")
                base_from_original = get_language_metadata(original_lang_iso.split("-")[0] if "-" in original_lang_iso else original_lang_iso)
                if base_from_original.get("iso_639_1") or base_from_original.get("iso_639_3"):
                    lang_metadata = base_from_original
        
        # Use normalized ISO-1 code as the key (or ISO-3 if no ISO-1)
        # Prefer ISO-1 for consistency, but fall back to ISO-3
        normalized_key = lang_metadata.get("iso_639_1") or lang_metadata.get("iso_639_3") or lang_iso
        base_iso = lang_metadata.get("iso_639_3") or normalized_key
        
        if normalized_key not in languages_data:
            languages_data[normalized_key] = {
                "iso_639_1": lang_metadata.get("iso_639_1"),
                "iso_639_2": lang_metadata.get("iso_639_2"),
                "iso_639_3": lang_metadata.get("iso_639_3") or normalized_key,
                "primary_name": lang_metadata.get("primary_name") or lang_name or normalized_key,
                "model_count": 0,
                "backend_count": 0,
                "backends": set(),
                "dialects_seen": set(),  # Track which dialect codes we've seen
                "has_generic_models": False,  # Track if there are generic (non-dialect) models
            }
        
        languages_data[normalized_key]["model_count"] += 1
        backend = entry.get("backend", "")
        if backend:
            languages_data[normalized_key]["backends"].add(backend)
            languages_data[normalized_key]["backend_count"] = len(languages_data[normalized_key]["backends"])
        
        # Track dialect if this is a dialect-specific model
        # Use original_lang_iso to detect dialects (e.g., "oci-ar", "oci-ara")
        # Also check model name for dialect indicators (e.g., "pap-aru-1" -> "aru")
        dialect_code = None
        base_iso_lower = base_iso.lower()
        
        # First try alias mapping
        if base_iso_lower in dialect_aliases and original_lang_iso.lower() in dialect_aliases[base_iso_lower]:
            dialect_code = dialect_aliases[base_iso_lower][original_lang_iso.lower()]
        else:
            # Fallback: extract dialect code directly (e.g., "oci-ara" -> "ara")
            dialect_code = _extract_dialect_code(original_lang_iso, base_iso)
            # If we extracted a dialect code, check if it matches a known dialect
            # (e.g., "ar" might need to map to "ara")
            if dialect_code and base_iso_lower in dialect_names:
                # Check if the extracted code matches a dialect name or if we need to find it
                if dialect_code not in dialect_names[base_iso_lower]:
                    # Try to find a dialect that has this as a substring or vice versa
                    for known_dial_code in dialect_names[base_iso_lower]:
                        if dialect_code in known_dial_code or known_dial_code in dialect_code:
                            dialect_code = known_dial_code
                            break
        
        # If no dialect found but model name suggests one (e.g., "pap-aru-1"), 
        # extract it from the model name
        # Only do this if original_lang_iso doesn't already contain dialect info
        # (to avoid false positives for models like "oci-restaure" where "restaure" is not a dialect)
        if not dialect_code and original_lang_iso.lower() == base_iso.lower():
            model_name = entry.get("model", "")
            if model_name and "-" in model_name:
                # Check if model name follows pattern: "lang-dialect-..." or "lang-dialect"
                parts = model_name.split("-")
                if len(parts) >= 2:
                    # Check if first part matches base language
                    first_part = parts[0].lower()
                    if first_part == base_iso_lower or first_part == normalized_key.lower():
                        possible_dialect = parts[1].lower()
                        # If dialects are defined for this language, check if it matches
                        if base_iso_lower in dialect_names and possible_dialect in dialect_names[base_iso_lower]:
                            dialect_code = possible_dialect
                        # Even if not in dialect_names, if the pattern suggests a dialect, use it
                        # (e.g., "pap-aru-1" where "aru" is likely a dialect/variant code)
                        # But only if the original_lang_iso is the base (not already a dialect code)
                        elif len(possible_dialect) <= 4 and len(possible_dialect) >= 2 and original_lang_iso.lower() == base_iso.lower():
                            # Heuristic: short codes after language code are likely dialects
                            dialect_code = possible_dialect
        
        # Track if this is a generic (non-dialect) model
        # A model is generic ONLY if:
        # 1. No dialect code was detected at all, AND
        # 2. The original_lang_iso equals the base_iso (e.g., "oci" not "oci-ara")
        # If a dialect_code was detected (even from model name), it's NOT generic
        is_generic_model = (dialect_code is None and 
                           (original_lang_iso.lower() == base_iso.lower() or 
                            original_lang_iso.lower() == normalized_key.lower() or
                            (original_lang_iso.lower() == lang_iso.lower() and lang_iso.lower() in [normalized_key.lower(), base_iso.lower()])))
        if is_generic_model:
            languages_data[normalized_key]["has_generic_models"] = True
        
        if dialect_code:
            languages_data[normalized_key]["dialects_seen"].add(dialect_code)
    
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
        
        # Add dialect information if dialects were seen
        # Distinguish between languages with both generic and dialect models vs. only dialect models
        dialects_seen = lang_data.get("dialects_seen", set())
        has_generic = lang_data.get("has_generic_models", False)
        if dialects_seen:
            base_iso_lower = iso_3.lower()
            dialect_info = []
            for dial_code in sorted(dialects_seen):
                dial_name = dialect_names.get(base_iso_lower, {}).get(dial_code, dial_code)
                dialect_info.append(dial_name)
            if dialect_info:
                if has_generic:
                    # Has both generic and dialect models: "Occitan (dialects: Aranese)"
                    name += f" (dialects: {', '.join(dialect_info)})"
                else:
                    # Only dialect models: "Papiamento (Aru only)" or just list the dialects
                    if len(dialect_info) == 1:
                        name += f" ({dialect_info[0]} only)"
                    else:
                        name += f" ({', '.join(dialect_info)} only)"
        
        print(f"{iso_1:<8} {iso_2:<8} {iso_3:<8} {name:<25} {model_count:<8} {backend_count:<10} {backends_str}")
    
    # Count total languages including dialects
    total_languages = len(sorted_languages)
    total_dialects = sum(len(d.get("dialects_seen", set())) for _, d in sorted_languages)
    total_models = sum(d['model_count'] for _, d in sorted_languages)
    
    if total_dialects > 0:
        print(f"\nTotal: {total_languages} language(s) with {total_dialects} dialect(s) and {total_models} model(s)")
    else:
        print(f"\nTotal: {total_languages} language(s) with {total_models} model(s)")
    
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


def list_teitok_settings(args: argparse.Namespace) -> int:
    """Display TEITOK settings.xml configuration."""
    from pathlib import Path
    from .teitok_settings import load_teitok_settings, find_settings_xml, DEFAULT_ATTRIBUTE_MAPPINGS
    
    output_format = getattr(args, "output_format", "table")
    
    # Determine settings.xml path
    settings_path = None
    teitok_mode = getattr(args, "teitok", False)
    
    if getattr(args, "teitok_settings", None):
        # Explicit path provided
        settings_path = Path(args.teitok_settings).expanduser()
    elif teitok_mode:
        # --teitok flag: search from current directory
        # find_settings_xml will prioritize tmp/cqpsettings.xml (merged settings)
        # over Resources/settings.xml
        found_path = find_settings_xml(Path.cwd())
        if found_path:
            settings_path = found_path
    elif getattr(args, "corpus", None):
        found_path = find_settings_xml(Path(args.corpus))
        if found_path:
            settings_path = found_path
    else:
        # Try to find in current directory
        found_path = find_settings_xml(Path.cwd())
        if found_path:
            settings_path = found_path
    
    if not settings_path or not settings_path.exists():
        print("Error: Could not find settings.xml file.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  --teitok            Enable TEITOK mode (looks for tmp/cqpsettings.xml or ./Resources/settings.xml)", file=sys.stderr)
        print("  --settings PATH     Specify path to settings.xml", file=sys.stderr)
        print("  --corpus PATH       Specify corpus directory (will search for settings.xml)", file=sys.stderr)
        print("  Run from a TEITOK corpus directory (searches Resources/settings.xml)", file=sys.stderr)
        return 1
    
    # Load settings
    settings = load_teitok_settings(settings_path=settings_path)
    
    # Prepare output data
    # Ensure binary switches always have explicit boolean values
    output_data = {
        "settings_path": str(settings_path),
        "attribute_mappings": settings.attribute_mappings,
        "default_language": settings.default_language,
        "cqp_pattributes": settings.cqp_pattributes,
        "cqp_sattributes": settings.cqp_sattributes,
        "cqp_sattributes_by_region": settings.cqp_sattributes_by_region,
        "cqp_sattributes_by_level": settings.cqp_sattributes_by_level,
        "explicit_flexipipe_mappings": settings.explicit_flexipipe_mappings,
        "flexipipe_preferences": settings.flexipipe_preferences,
        "known_tags_only": bool(settings.known_tags_only),
        "use_raw_text": bool(settings.use_raw_text),
        "download_model": bool(settings.download_model),
        "xmlfile_defaults": settings.xmlfile_defaults,
        "teiheader_defaults": settings.teiheader_defaults,
    }
    
    if output_format == "json":
        print(json.dumps(output_data, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format
    print(f"TEITOK Settings: {settings_path}")
    # Note if Resources/settings.xml is also being used
    if settings_path and settings_path.name == "cqpsettings.xml" and "tmp" in settings_path.parts:
        resources_settings = settings_path.parent.parent / "Resources" / "settings.xml"
        if resources_settings.exists():
            print(f"(Also using: {resources_settings})")
    print("=" * 80)
    
    # Attribute mappings (only show with --verbose)
    verbose = getattr(args, "verbose", False) or getattr(args, "debug", False)
    if verbose:
        print("\nAttribute Mappings:")
        print(f"{'Internal':<15} {'XML Attributes (priority order)':<50}")
        print("-" * 80)
        for internal_attr in sorted(settings.attribute_mappings.keys()):
            xml_attrs = settings.attribute_mappings[internal_attr]
            print(f"{internal_attr:<15} {', '.join(xml_attrs):<50}")
    
    # Show which CQP attributes are recognized and mapped
    from .teitok_settings import CQP_ATTRIBUTE_MAPPINGS
    recognized_pattrs = []
    for pattr in settings.cqp_pattributes:
        # Skip 'form' - it's the basic surface form attribute, always present
        if pattr.lower() == "form":
            recognized_pattrs.append(f"{pattr} (surface form)")
            continue
        # Check for explicit flexipipe mapping first (highest priority)
        if pattr in settings.explicit_flexipipe_mappings:
            mapped_to = settings.explicit_flexipipe_mappings[pattr]
            recognized_pattrs.append(f"{pattr} → {mapped_to} (explicit flexipipe mapping)")
        else:
            # Check if this pattribute maps to an internal attribute via standard mappings
            mapped_to = None
            for cqp_name, internal_name in CQP_ATTRIBUTE_MAPPINGS.items():
                if pattr.lower() == cqp_name.lower():
                    mapped_to = internal_name
                    break
            if mapped_to:
                recognized_pattrs.append(f"{pattr} → {mapped_to}")
            else:
                # Only show unmapped if verbose
                if verbose:
                    recognized_pattrs.append(f"{pattr} (unmapped)")
    
    if recognized_pattrs:
        print("\nRecognized CQP Positional Attributes (mapped to Document attributes):")
        for mapping in recognized_pattrs:
            print(f"  {mapping}")
    
    # For sattributes, show which ones are recognized and mapped to CoNLL-U standard attributes
    from .doc import UD_DOCUMENT_ATTRIBUTES, UD_PARAGRAPH_ATTRIBUTES, UD_SENTENCE_ATTRIBUTES
    
    # Map TEITOK sattributes to CoNLL-U standard attributes
    # Common mappings (case-insensitive)
    CONLLU_MAPPINGS = {
        "text": {  # document level
            "title": "title",
            "author": "author",
            "date": "date",
            "genre": "genre",
            "publisher": "publisher",
            "url": "url",
            "license": "license",
            "source": "source",
        },
        "p": {  # paragraph level
            "section": "section",
            "align": "align",
        },
        "s": {  # sentence level
            "sent_id": "sent_id",
            "text": "text",
            "lang": "lang",
            "date": "date",
            "speaker": "speaker",
            "participant": "participant",
            "annotator": "annotator",
            "translation": "translation",
            "align": "align",
        },
    }
    
    if settings.cqp_sattributes_by_level:
        print("\nCQP Structural Attributes (mapped to CoNLL-U standard attributes):")
        for level in ["text", "p", "s"]:
            if level in settings.cqp_sattributes_by_level:
                level_name = {"text": "Document (text)", "p": "Paragraph (p)", "s": "Sentence (s)"}[level]
                level_mappings = CONLLU_MAPPINGS.get(level, {})
                print(f"  {level_name} level:")
                for region_name, attrs in sorted(settings.cqp_sattributes_by_level[level].items()):
                    mapped_attrs = []
                    for attr in attrs:
                        # Check if this maps to a CoNLL-U standard attribute
                        mapped_to = level_mappings.get(attr.lower())
                        if mapped_to:
                            mapped_attrs.append(f"{attr} → {mapped_to}")
                        else:
                            mapped_attrs.append(attr)
                    print(f"    {region_name} - attributes: {', '.join(mapped_attrs)}")
    elif settings.cqp_sattributes_by_region:
        print("\nCQP Structural Attributes (document/sentence-level metadata):")
        for region_name, attrs in sorted(settings.cqp_sattributes_by_region.items()):
            print(f"  {region_name} - attributes: {', '.join(attrs)}")
    elif settings.cqp_sattributes:
        print("\nCQP Structural Attributes (document/sentence-level metadata):")
        print(f"  {', '.join(settings.cqp_sattributes)}")
    
    # Display flexipipe flags
    flexipipe_flags = []
    if settings.known_tags_only:
        flexipipe_flags.append("known-tags-only")
    if settings.use_raw_text:
        flexipipe_flags.append("use-raw-text")
    if settings.download_model:
        flexipipe_flags.append("download-model")
    if flexipipe_flags:
        print("\nFlexipipe Options:")
        for flag in flexipipe_flags:
            print(f"  {flag}: enabled")
    
    if settings.flexipipe_preferences:
        print("\nFlexipipe Preferred Models per Language:")
        registry_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for lang_key in sorted(settings.flexipipe_preferences.keys()):
            pref = settings.flexipipe_preferences[lang_key]
            backend_pref = pref.get("backend")
            model_pref = pref.get("model")
            parts = []
            if backend_pref:
                parts.append(f"backend={backend_pref}")
            if model_pref:
                parts.append(f"model={model_pref}")
            summary = ", ".join(parts) if parts else "no backend/model specified"
            registry_note = ""
            mismatch_note = ""
            if backend_pref and model_pref:
                try:
                    backend_lower = backend_pref.lower()
                    if backend_lower not in registry_cache:
                        registry_cache[backend_lower] = get_model_entries(
                            backend_lower,
                            use_cache=True,
                            refresh_cache=False,
                            verbose=verbose,
                        )
                    entry = registry_cache[backend_lower].get(model_pref)
                    if entry:
                        query = resolve_language_query(lang_key)
                        if not language_matches_entry(entry, query, allow_fuzzy=True):
                            entry_lang = entry.get("language_name") or entry.get("language_iso") or "unknown"
                            mismatch_note = f" (warning: model language {entry_lang})"
                    else:
                        registry_note = " (model not found in registry)"
                except Exception as exc:
                    registry_note = f" (registry unavailable: {exc})"
            elif backend_pref and not model_pref:
                registry_note = " (missing model)"
            elif model_pref and not backend_pref:
                registry_note = " (missing backend)"
            print(f"  {lang_key}: {summary}{registry_note}{mismatch_note}")
    
    # Default language
    if settings.default_language:
        print(f"\nDefault Language: {settings.default_language}")
    else:
        print("\nDefault Language: (not set)")
    
    
    # XML file defaults
    if settings.xmlfile_defaults:
        print("\nXML File Defaults:")
        for key, value in sorted(settings.xmlfile_defaults.items()):
            print(f"  {key}: {value}")
    
    # TEI header defaults
    if settings.teiheader_defaults:
        print("\nTEI Header Defaults:")
        for key, value in sorted(settings.teiheader_defaults.items()):
            print(f"  {key}: {value}")
    
    # Show note if no CQP attributes found (but mappings may still be built from defaults)
    if not settings.cqp_pattributes and not settings.cqp_sattributes:
        print("\nNote: No CQP attributes found in settings.xml. Using default attribute mappings.")
    
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
            "Error: No action specified. Use one of: backends, models, languages, ud-tags, examples, tasks, teitok, or --detect-language"
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
    elif args.info_action == "teitok":
        return list_teitok_settings(args)
    else:
        print(f"Error: Unknown info action '{args.info_action}'")
        return 1

