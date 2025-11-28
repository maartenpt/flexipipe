from __future__ import annotations

import argparse
import json
import os
import re
import sys
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple


def _natural_sort_key(s: Any) -> Tuple[Any, ...]:
    """
    Generate a sort key for natural, case-insensitive sorting.
    Converts strings to lowercase and splits into text and numeric parts.
    Returns a tuple that can be safely compared. Uses a consistent structure:
    - Empty/None values return ("",) (string tuple)
    - All tuples start with strings, then alternate string/int
    - This ensures type-safe comparison
    """
    if s is None:
        return ("",)
    if not isinstance(s, str):
        s = str(s)
    if not s:
        return ("",)
    s_lower = s.lower()
    # Split into alternating text and numeric parts
    parts = re.split(r'(\d+)', s_lower)
    result = []
    for part in parts:
        if part.isdigit():
            # Convert to int for proper numeric sorting (2 < 10)
            result.append(int(part))
        elif part:
            result.append(part)
    # If result starts with a number, prepend empty string to ensure consistent structure
    if result and isinstance(result[0], int):
        result.insert(0, "")
    # Ensure we always have at least one element
    return tuple(result) if result else ("",)


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got '{value}'")

from . import (
    Document,
    apply_nlpform,
    dump_teitok,
    load_teitok,
    save_teitok,
    io_conllu,  # noqa: F401 - registers handlers
)
from .engine import assign_doc_id_from_path
from .check import evaluate_model
from .conllu import conllu_to_document, document_to_conllu, prepare_conllu_with_nlpform
from .language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
    language_matches_entry,
    resolve_language_query,
)
from .language_detector_registry import (
    detect_language_with,
    list_language_detectors,
    get_default_language_detector as _registry_default_lang_detector,
)
from .lexicon import convert_lexicon_to_vocab
from .io_registry import registry as io_registry
from .doc_utils import document_to_json_payload
from .teitok_settings import TeitokSettings
from .backends.flexitag import (
    build_flexitag_options_from_args,
    get_flexitag_model_entries,
    list_flexitag_models,
    resolve_flexitag_model_path,
)
from .backends.udmorph import get_udmorph_model_entries, get_udmorph_model_entry
from .backend_registry import get_backend_choices, get_backend_choices_for_training
from .task_registry import TASK_DEFAULTS, TASK_MANDATORY, TASK_LOOKUP
from .validator import (
    infer_language_from_document,
    infer_language_from_path,
    run_validator_cli,
)

def _get_language_backend_priority() -> List[str]:
    """Get backend priority list for language detection (dynamically discovered)."""
    from .backend_registry import get_backend_choices
    # Maintain a preferred order, but include all discovered backends
    preferred_order = [
        "flexitag",
        "spacy",
        "stanza",
        "classla",
        "transformers",
        "udpipe",
        "udmorph",
        "udpipe1",
        "treetagger",
        "nametag",
    ]
    all_backends = get_backend_choices()
    # Start with preferred order, then add any discovered backends not in the list
    result = [b for b in preferred_order if b in all_backends]
    result.extend([b for b in all_backends if b not in result])
    return result
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.80
LANGUAGE_DETECTOR_DEFAULT = _registry_default_lang_detector()
LANGUAGE_DETECTION_MIN_LENGTH = 20
_TRANSFORMERS_MODEL_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_example_text(example_name: str, language: Optional[str]) -> Optional[str]:
    """Load example text stored under the models directory."""
    if example_name != "udhr" or not language:
        return None
    try:
        from .examples_data import get_example_text
    except ImportError:
        return None
    return get_example_text(language)


def _propagate_sentence_metadata(target: Document, source: Document) -> None:
    """Copy original sentence/document IDs into the tagged document when possible."""
    if not source.sentences or not target.sentences:
        return
    if len(target.sentences) != len(source.sentences):
        return
    if not target.id and source.id:
        target.id = source.id
    for tgt_sent, src_sent in zip(target.sentences, source.sentences):
        candidate_id = (src_sent.id or getattr(src_sent, "source_id", "") or src_sent.sent_id or "").strip()
        if candidate_id and not tgt_sent.id:
            tgt_sent.id = candidate_id
        src_source_id = getattr(src_sent, "source_id", "").strip()
        if src_source_id:
            tgt_sent.source_id = src_source_id
        if src_sent.sent_id:
            tgt_sent.sent_id = src_sent.sent_id


def _propagate_token_ids(target: Document, source: Document) -> None:
    """Copy original token IDs (tokid) into the tagged document when possible.
    
    This is crucial for TEITOK writeback, as the tokid is used to match tokens
    in the XML. Tokens are matched by position and form similarity.
    """
    if not source.sentences or not target.sentences:
        return
    if len(target.sentences) != len(source.sentences):
        return
    
    propagated_count = 0
    for tgt_sent, src_sent in zip(target.sentences, source.sentences):
        if not src_sent.tokens or not tgt_sent.tokens:
            continue
        
        # Simple 1:1 alignment by position (most backends preserve token order)
        # For most cases, tokens align directly
        min_len = min(len(src_sent.tokens), len(tgt_sent.tokens))
        for i in range(min_len):
            src_tok = src_sent.tokens[i]
            tgt_tok = tgt_sent.tokens[i]
            
            # If source token has a tokid, propagate it (even if target already has one)
            # This ensures we preserve the original XML IDs
            if src_tok.tokid:
                # For regular tokens, propagate directly
                if not src_tok.is_mwt and not tgt_tok.is_mwt:
                    if not tgt_tok.tokid or tgt_tok.tokid != src_tok.tokid:
                        tgt_tok.tokid = src_tok.tokid
                        propagated_count += 1
                # For MWTs, propagate parent tokid and subtoken tokids
                elif src_tok.is_mwt and tgt_tok.is_mwt:
                    if not tgt_tok.tokid or tgt_tok.tokid != src_tok.tokid:
                        tgt_tok.tokid = src_tok.tokid
                        propagated_count += 1
                    if src_tok.subtokens and tgt_tok.subtokens:
                        for src_sub, tgt_sub in zip(src_tok.subtokens, tgt_tok.subtokens):
                            if src_sub.tokid and (not tgt_sub.tokid or tgt_sub.tokid != src_sub.tokid):
                                tgt_sub.tokid = src_sub.tokid
                                propagated_count += 1


def _detect_performed_tasks(document: Document) -> set[str]:
    tasks = {"segment", "tokenize"}
    has_lemma = False
    has_tag = False
    has_parse = False
    has_norm = False
    has_ner = False
    has_wsd = False

    if getattr(document, "spans", None):
        if document.spans.get("ner"):
            has_ner = True

    for sentence in document.sentences:
        if sentence.entities:
            has_ner = True
        if sentence.corr:
            has_norm = True
        for token in sentence.tokens:
            if token.lemma:
                has_lemma = True
            if token.upos or token.xpos or token.feats:
                has_tag = True
            if (token.head and token.head > 0) or token.deprel or token.deps:
                has_parse = True
            if (
                token.reg
                or token.expan
                or token.mod
                or token.trslit
                or token.ltrslit
                or token.corr
                or token.lex
            ):
                has_norm = True
            if token.misc and token.misc not in {"_"}:
                has_norm = True
            if token.attrs.get("wsd"):
                has_wsd = True
            if token.attrs.get("wsd_confidence"):
                has_wsd = True
            for sub in token.subtokens:
                if sub.lemma:
                    has_lemma = True
                if sub.upos or sub.xpos or sub.feats:
                    has_tag = True
                if (
                    sub.reg
                    or sub.expan
                    or sub.mod
                    or sub.trslit
                    or sub.ltrslit
                    or sub.corr
                    or sub.lex
                ):
                    has_norm = True

    if has_lemma:
        tasks.add("lemmatize")
    if has_tag:
        tasks.add("tag")
    if has_parse:
        tasks.add("parse")
    if has_norm:
        tasks.add("normalize")
    if has_ner:
        tasks.add("ner")
    if has_wsd:
        tasks.add("wsd")
    return tasks


def _augment_tei_output(
    tei_xml: str,
    *,
    note_value: Optional[str],
    change_text: str,
    change_when: str,
    api_notes: Optional[list[tuple[str, str]]] = None,
    pretty_print: bool = False,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
) -> str:
    """Inject notesStmt/revisionDesc metadata into TEI output."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return tei_xml

    tei_header = root.find("teiHeader")
    if tei_header is None:
        tei_header = ET.Element("teiHeader")
        root.insert(0, tei_header)

    notes_stmt = tei_header.find("notesStmt")
    if notes_stmt is None:
        notes_stmt = ET.SubElement(tei_header, "notesStmt")
    
    # Add orgfile note if provided
    if note_value:
        for existing in list(notes_stmt):
            if existing.tag == "note" and existing.get("n") == "orgfile":
                notes_stmt.remove(existing)
        note_elem = ET.SubElement(notes_stmt, "note", {"n": "orgfile"})
        note_elem.text = note_value
    
    # Add API response notes if provided
    if api_notes:
        for note_type, note_content in api_notes:
            note_elem = ET.SubElement(notes_stmt, "note", {"n": note_type.lower().replace(" ", "-")})
            note_elem.text = note_content

    revision_desc = tei_header.find("revisionDesc")
    if revision_desc is None:
        revision_desc = ET.SubElement(tei_header, "revisionDesc")
    change_elem = ET.SubElement(
        revision_desc,
        "change",
        {"when": change_when, "who": "flexipipe"},
    )
    change_elem.text = change_text

    # Convert back to string
    updated = ET.tostring(root, encoding="unicode")
    if not updated.startswith("<?xml"):
        updated = '<?xml version="1.0" encoding="UTF-8"?>\n' + updated
    
    # Pretty-print the XML if requested
    if pretty_print:
        from .teitok import pretty_print_teitok_xml
        updated = pretty_print_teitok_xml(
            updated,
            spaceafter_handling=spaceafter_handling,
            skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
        )
    
    return updated
def _parse_tasks_argument(value: Optional[str]) -> set[str]:
    if not value:
        return set(TASK_DEFAULTS)
    tasks: set[str] = set()
    for raw in re.split(r"[,\s]+", value):
        token = raw.strip().lower()
        if not token:
            continue
        canonical = _TASK_LOOKUP.get(token)
        if not canonical:
            raise ValueError(f"Unknown task '{raw}'. Valid tasks: {', '.join(TASK_DEFAULTS)}")
        tasks.add(canonical)
    if not tasks:
        return set(TASK_DEFAULTS)
    return tasks


def _normalize_format_name(value: Optional[str]) -> Optional[str]:
    if value == "tei":
        return "teitok"
    return value


def _looks_like_transformers_model(model_name: Optional[str]) -> bool:
    entry = _get_transformers_model_entry(model_name)
    if entry:
        return True
    return bool(model_name and "/" in model_name)


def _get_transformers_model_entry(model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    global _TRANSFORMERS_MODEL_CACHE
    if not model_name:
        return None
    if _TRANSFORMERS_MODEL_CACHE is None:
        try:
            from .backends.transformers import get_transformers_model_entries
            _TRANSFORMERS_MODEL_CACHE = get_transformers_model_entries(
                use_cache=True,
                refresh_cache=False,
                verbose=False,
                include_llm=True,
            )
        except Exception:
            _TRANSFORMERS_MODEL_CACHE = {}
    return _TRANSFORMERS_MODEL_CACHE.get(model_name)


def _parse_transformers_context_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
    else:
        cleaned = [part.strip() for part in str(value).split(",") if part.strip()]
    return cleaned or None


def _tasks_to_backend_components(tasks: set[str], backend: Optional[str]) -> Optional[List[str]]:
    backend_lower = (backend or "").lower()
    if backend_lower == "flair":
        components: List[str] = []
        if any(task in tasks for task in ("tag", "lemmatize", "parse")):
            components.append("upos")
        if "xpos" in tasks:
            components.append("xpos")
        if "ner" in tasks:
            components.append("ner")
        if "wsd" in tasks:
            components.append("wsd")
        return components or None
    if backend_lower in {"udpipe", "udpipe1"}:
        components: List[str] = []
        if any(task in tasks for task in ("tag", "lemmatize", "parse")):
            components.append("tagger")
        if "parse" in tasks:
            components.append("parser")
        return components
    return None


def _filter_document_by_tasks(document: Document, tasks: set[str]) -> None:
    keep_lemma = "lemmatize" in tasks
    keep_upos = ("tag" in tasks) or ("parse" in tasks)
    keep_xpos = ("tag" in tasks) or ("parse" in tasks) or ("xpos" in tasks)
    keep_parse = "parse" in tasks
    keep_norm = "normalize" in tasks
    keep_ner = "ner" in tasks
    keep_wsd = "wsd" in tasks

    for sentence in document.sentences:
        if not keep_norm:
            sentence.corr = ""
            sentence.attrs.pop("corr", None)
        if not keep_ner:
            # Preserve entities if they exist - backends may create them even if not explicitly requested
            # They will be included in output via dump_teitok's entity handling
            # Only clear if no entities exist (defensive - shouldn't happen if keep_ner logic worked)
            pass  # Don't clear entities - preserve NER from backends
        for token in sentence.tokens:
            if not keep_lemma:
                token.lemma = ""
                token.lemma_confidence = None
                token.attrs.pop("lemma", None)
                for sub in token.subtokens:
                    sub.lemma = ""
                    sub.attrs.pop("lemma", None)
            if not keep_upos:
                token.upos = ""
                token.feats = ""
                token.upos_confidence = None
                token.attrs.pop("upos", None)
                token.attrs.pop("feats", None)
                for sub in token.subtokens:
                    sub.upos = ""
                    sub.feats = ""
                    sub.attrs.pop("upos", None)
                    sub.attrs.pop("feats", None)
            if not keep_xpos:
                token.xpos = ""
                token.xpos_confidence = None
                token.attrs.pop("xpos", None)
                for sub in token.subtokens:
                    sub.xpos = ""
                    sub.attrs.pop("xpos", None)
            if not keep_parse:
                token.head = 0
                token.deprel = ""
                token.deps = ""
                token.deprel_confidence = None
                token.attrs.pop("head", None)
                token.attrs.pop("deprel", None)
                token.attrs.pop("deps", None)
            if not keep_wsd:
                token.attrs.pop("wsd", None)
                token.attrs.pop("wsd_confidence", None)
            if not keep_norm:
                token.reg = None
                token.expan = None
                token.mod = None
                token.trslit = None
                token.ltrslit = None
                token.corr = None
                token.lex = None
                for key in ("reg", "expan", "mod", "trslit", "ltrslit", "corr", "lex"):
                    token.attrs.pop(key, None)
                for sub in token.subtokens:
                    sub.reg = None
                    sub.expan = None
                    sub.mod = None
                    sub.trslit = None
                    sub.ltrslit = None
                    sub.corr = None
                    sub.lex = None
                    for key in ("reg", "expan", "mod", "trslit", "ltrslit", "corr", "lex"):
                        sub.attrs.pop(key, None)
        if not keep_norm:
            sentence.corr = ""
            sentence.attrs.pop("corr", None)
    if not keep_ner and getattr(document, "spans", None):
        document.spans.clear()


def _load_backend_entries(
    backend_type: str,
    args: argparse.Namespace,
    *,
    use_cache: bool,
    refresh_cache: bool,
    verbose: bool = False,
) -> dict:
    """
    Load model entries for a backend using the registry.
    
    This replaces the old if/elif chain with a registry-based approach.
    """
    import inspect

    from .backend_registry import get_backend_info, get_model_entries
    
    backend = backend_type.lower()
    
    # Build kwargs based on backend type
    kwargs = {
        "use_cache": use_cache,
        "refresh_cache": refresh_cache,
        "verbose": verbose,  # Always pass verbose (False or True) to ensure backends respect it
        "include_remote": True,  # Include remote models from registry by default
    }
    
    # Add backend-specific arguments
    # Use unified --endpoint-url for all REST backends
    url = getattr(args, "endpoint_url", None)
    if url:
        kwargs["endpoint_url"] = url
    if backend == "transformers":
        kwargs["include_llm"] = bool(getattr(args, "include_base_models", False))
    
    info = get_backend_info(backend)
    if info and info.get_model_entries:
        sig = inspect.signature(info.get_model_entries)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kw:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    else:
        filtered_kwargs = kwargs

    # Use registry to get model entries
    return get_model_entries(backend, **filtered_kwargs)


def _collect_language_matches(
    entries_by_backend: dict[str, dict],
    query: dict,
    *,
    allow_fuzzy: bool = False,
) -> list[tuple[str, str, dict]]:
    matches: list[tuple[str, str, dict]] = []
    for backend, entries in entries_by_backend.items():
        # Skip if entries is not a dict (e.g., error dict)
        if not isinstance(entries, dict):
            continue
        # Skip if entries is an error dict
        if len(entries) == 1 and "error" in entries:
            continue
        for model_name, entry in entries.items():
            # Skip if entry is not a dict
            if not isinstance(entry, dict):
                continue
            # Skip disabled models
            from .benchmark import is_model_disabled
            if is_model_disabled(backend, model_name):
                continue
            if language_matches_entry(entry, query, allow_fuzzy=allow_fuzzy):
                matches.append((backend, model_name, entry))
    return matches


def _display_language_filtered_models(language: Optional[str], entries_by_backend: dict[str, dict], output_format: str = "table", sort_by: str = "backend") -> int:
    # Load disabled models check function
    from .benchmark import is_model_disabled
    
    used_fuzzy = False
    # If no language filter, show all models from all backends
    if language is None:
        matches: list[tuple[str, str, dict]] = []
        for backend, entries in entries_by_backend.items():
            # Skip if entries is not a dict (e.g., error dict)
            if not isinstance(entries, dict):
                continue
            # Skip if entries is an error dict
            if len(entries) == 1 and "error" in entries:
                continue
            for model_name, entry in entries.items():
                # Skip if entry is not a dict
                if not isinstance(entry, dict):
                    continue
                # Skip disabled models
                if is_model_disabled(backend, model_name):
                    continue
                matches.append((backend, model_name, entry))
    else:
        query = resolve_language_query(language)
        matches = _collect_language_matches(entries_by_backend, query, allow_fuzzy=False)
        if not matches:
            matches = _collect_language_matches(entries_by_backend, query, allow_fuzzy=True)
            if matches:
                used_fuzzy = True
        # Filter out disabled models
        matches = [(backend, model, entry) for backend, model, entry in matches if not is_model_disabled(backend, model)]

        if not matches:
            if output_format == "json":
                import json
                import sys
                print(json.dumps({"language": language, "models": []}, indent=2, ensure_ascii=False), flush=True)
            else:
                print(f"No models found for language '{language}'.")
            return 0

    # Sort matches based on sort_by option (natural, case-insensitive sorting)
    if sort_by == "backend":
        matches.sort(key=lambda item: (_natural_sort_key(item[0]), _natural_sort_key(item[1])))
    elif sort_by == "model":
        matches.sort(key=lambda item: (_natural_sort_key(item[1]), _natural_sort_key(item[0])))
    elif sort_by == "language":
        matches.sort(key=lambda item: (
            _natural_sort_key(item[2].get(LANGUAGE_FIELD_NAME) or item[2].get("language_display") or ""),
            _natural_sort_key(item[2].get(LANGUAGE_FIELD_ISO) or ""),
            _natural_sort_key(item[0]),
            _natural_sort_key(item[1])
        ))
    elif sort_by == "iso":
        matches.sort(key=lambda item: (
            _natural_sort_key(item[2].get(LANGUAGE_FIELD_ISO) or ""),
            _natural_sort_key(item[2].get(LANGUAGE_FIELD_NAME) or item[2].get("language_display") or ""),
            _natural_sort_key(item[0]),
            _natural_sort_key(item[1])
        ))
    elif sort_by == "status":
        matches.sort(key=lambda item: (
            _natural_sort_key(item[2].get("status") or ""),
            _natural_sort_key(item[0]),
            _natural_sort_key(item[1])
        ))
    else:
        # Default: sort by backend, then model
        matches.sort(key=lambda item: (_natural_sort_key(item[0]), _natural_sort_key(item[1])))
    
    if output_format == "json":
        import json
        models_data = []
        backends_seen: set[str] = set()
        for backend, model_name, entry in matches:
            backends_seen.add(backend)
            model_info = {
                "backend": backend,
                "model": model_name,
                "language_iso": entry.get(LANGUAGE_FIELD_ISO),
                "language_name": entry.get(LANGUAGE_FIELD_NAME) or entry.get("language_display"),
                "status": entry.get("status"),
                "version": entry.get("version") or entry.get("date") or entry.get("updated"),
                "features": entry.get("features"),
                "description": entry.get("description"),
                "package": entry.get("package"),
            }
            # Remove None values
            model_info = {k: v for k, v in model_info.items() if v is not None}
            models_data.append(model_info)
        
        result = {
            "models": models_data,
            "total": len(models_data),
            "backends": sorted(backends_seen),
        }
        if language:
            result["language"] = language
            result["fuzzy_matching"] = used_fuzzy
        import sys
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format
    if language:
        print(f"\nModels matching language '{language}':")
    else:
        print(f"\nAll available models:")
    print(f"{'Backend':<10} {'Model':<35} {'ISO':<6} {'Language':<20} {'Details'}")
    print("=" * 120)

    backends_seen: set[str] = set()
    for backend, model_name, entry in matches:
        backends_seen.add(backend)
        iso = (entry.get(LANGUAGE_FIELD_ISO) or "-")[:6]
        lang_name = entry.get(LANGUAGE_FIELD_NAME) or entry.get("language_display") or "-"
        details_parts: list[str] = []
        
        # Show preferred flag if available
        if entry.get("preferred", False):
            details_parts.append("★ Preferred")
        
        status = entry.get("status")
        if status:
            details = status.capitalize()
            version = entry.get("version") or entry.get("date") or entry.get("updated")
            if version:
                details += f" ({version})"
            details_parts.append(details)
        features = entry.get("features")
        if features and features not in details_parts:
            details_parts.append(features)
        description = entry.get("description")
        if description and description not in details_parts:
            details_parts.append(description)
        package = entry.get("package")
        if package:
            details_parts.append(f"package={package}")
        details_str = "; ".join(details_parts) if details_parts else ""
        print(f"{backend:<10} {model_name:<35} {iso:<6} {lang_name:<20} {details_str}")

    if used_fuzzy:
        print("\nNote: matches were found using fuzzy language matching.")
    print(f"\nTotal: {len(matches)} model(s) across {len(backends_seen)} backend(s)")
    
    # Calculate and display disk space used (only for the filtered models)
    try:
        from .model_storage import get_flexipipe_models_dir, get_backend_models_dir
        from .backend_registry import get_backend_info
        models_dir = get_flexipipe_models_dir(create=False)
        if models_dir.exists() and matches:
            total_size = 0
            models_counted = 0
            try:
                # Only calculate size for the filtered models
                for backend, model_name, entry in matches:
                    # Skip REST backends - they don't have local files
                    backend_info = get_backend_info(backend)
                    is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
                    if is_rest_backend:
                        continue
                    
                    # Try to find the model directory/file
                    backend_dir = get_backend_models_dir(backend, create=False)
                    if not backend_dir or not backend_dir.exists():
                        continue
                    
                    # Look for the model in the backend directory
                    model_path = backend_dir / model_name
                    if model_path.exists():
                        models_counted += 1
                        if model_path.is_file():
                            try:
                                total_size += model_path.stat().st_size
                            except (OSError, PermissionError):
                                pass
                        elif model_path.is_dir():
                            # Sum all files in the directory
                            try:
                                for file_path in model_path.rglob("*"):
                                    if file_path.is_file():
                                        try:
                                            total_size += file_path.stat().st_size
                                        except (OSError, PermissionError):
                                            pass
                            except (OSError, PermissionError):
                                pass
            except (OSError, PermissionError):
                # If we can't access directories, skip disk space calculation
                pass
            else:
                if models_counted > 0:
                    # Convert to GB and format
                    size_gb = total_size / (1024 ** 3)
                    print(f"Disk space used: {size_gb:.2f} GB for {models_counted} model(s) shown above")
    except Exception:
        # If anything fails, just skip the disk space display
        pass
    
    return 0


def _auto_select_model_for_language(
    args: argparse.Namespace,
    backend_type: Optional[str],
    *,
    exclude_backends: Optional[Collection[str]] = None,
) -> Optional[str]:
    backend_locked = bool(getattr(args, "_backend_explicit", False))
    excluded = {name.lower() for name in (exclude_backends or [])}
    language = getattr(args, "language", None)
    if not language:
        return backend_type
    if getattr(args, "model", None):
        return backend_type
    if getattr(args, "params", None) and (not backend_type or backend_type.lower() == "flexitag"):
        return backend_type
    if backend_type and backend_type.lower() == "spacy" and backend_locked:
        return backend_type

    requested_backend = None
    if backend_type:
        lowered = backend_type.lower()
        if lowered not in excluded:
            requested_backend = lowered

    search_order: list[str] = []
    if requested_backend:
        search_order.append(requested_backend)
    if backend_locked:
        remaining: list[str] = []
    else:
        backend_priority = _get_language_backend_priority()
        remaining = [b for b in backend_priority if b not in search_order and b not in excluded]
    combined_order = search_order + remaining

    def _load_backend_entries_for_order(
        order: Collection[str],
        *,
        refresh: bool,
    ) -> dict[str, dict]:
        entries_map: dict[str, dict] = {}
        for backend in order:
            if refresh and (getattr(args, "verbose", False) or getattr(args, "debug", False)):
                print(f"[flexipipe] Fetching model registry for backend '{backend}' (network request)...")
            try:
                entries_map[backend] = _load_backend_entries(
                    backend,
                    args,
                    use_cache=not refresh,
                    refresh_cache=refresh,
                    verbose=False,
                )
            except Exception as exc:
                entries_map.setdefault(backend, {})
                if refresh and (getattr(args, "verbose", False) or getattr(args, "debug", False)):
                    print(f"[flexipipe] Warning: could not refresh model registry for backend '{backend}': {exc}")
        return entries_map

    def ensure_entries(order: list[str]) -> dict[str, dict]:
        entries_map: dict[str, dict] = {}
        # Try unified catalog first
        try:
            from .model_catalog import build_unified_catalog

            catalog = build_unified_catalog(use_cache=True, refresh_cache=False, verbose=False)
            for entry in catalog.values():
                backend = entry.get("backend")
                if not backend or backend not in order:
                    continue
                entries_map.setdefault(backend, {})
                model_name = entry.get("model")
                if model_name:
                    entries_map[backend][model_name] = entry
        except Exception:
            pass

        missing_backends = [backend for backend in order if backend not in entries_map]
        if missing_backends:
            entries_map.update(_load_backend_entries_for_order(missing_backends, refresh=False))

        # Ensure all keys exist even if empty
        for backend in order:
            entries_map.setdefault(backend, {})

        empty_backends = [backend for backend in order if not entries_map.get(backend)]
        if empty_backends:
            refreshed = _load_backend_entries_for_order(empty_backends, refresh=True)
            for backend in empty_backends:
                if refreshed.get(backend):
                    entries_map[backend] = refreshed[backend]
        return entries_map

    query = resolve_language_query(language)

    def pick_best(entries_map: dict[str, dict], allow_fuzzy: bool) -> Optional[tuple[str, dict]]:
        matches = _collect_language_matches(entries_map, query, allow_fuzzy=allow_fuzzy)
        if not matches:
            return None
        def sort_key(item: tuple[str, str, dict]):
            backend, model_name, entry = item
            # Use preferred flag if available (from unified catalog)
            preferred_score = 0 if entry.get("preferred", False) else 1
            status = (entry.get("status") or "").lower()
            installed_score = 0 if status == "installed" else 1
            backend_score = backend_rank.get(backend, len(combined_order))
            return (preferred_score, installed_score, backend_score, model_name)

        matches.sort(key=sort_key)
        for backend, model_name, entry in matches:
            if backend == "spacy" and not backend_locked and not _spacy_model_available(model_name):
                continue
            return backend, entry
        return None

    backend_rank = {name: idx for idx, name in enumerate(combined_order)}

    # First try requested backend (if any)
    primary_entries = ensure_entries(search_order)
    selection = pick_best(primary_entries, allow_fuzzy=False) or pick_best(primary_entries, allow_fuzzy=True)

    # If nothing found, try remaining backends
    if not selection and remaining:
        secondary_entries = ensure_entries(remaining)
        selection = pick_best(secondary_entries, allow_fuzzy=False) or pick_best(secondary_entries, allow_fuzzy=True)

    if not selection:
        if backend_locked and requested_backend:
            return backend_type
        print(f"[flexipipe] No available models found for language '{language}'.")
        return backend_type

    chosen_backend, entry = selection
    chosen_model = entry.get("model")
    if not chosen_model:
        return backend_type
    if backend_locked and requested_backend and chosen_backend != requested_backend:
            return backend_type

    entry_language = (
        entry.get("language_name")
        or entry.get("language_iso")
        or language
    )

    def _log_selection(selected_backend: str, selected_model: Optional[str]) -> None:
        if not getattr(args, "verbose", False):
            return
        message = f"[flexipipe] Auto-selected backend '{selected_backend}'"
        if selected_model:
            message += f" model '{selected_model}'"
        message += f" for language '{entry_language}'."
        print(message)

    if chosen_backend == "spacy":
        if backend_type is None or chosen_backend != backend_type.lower():
            setattr(args, "backend", chosen_backend)
            backend_type = chosen_backend
        _log_selection(chosen_backend, entry.get("model"))
        return backend_type

    setattr(args, "model", chosen_model)
    if backend_type is None or chosen_backend != backend_type.lower():
        setattr(args, "backend", chosen_backend)
        backend_type = chosen_backend
    
    _log_selection(chosen_backend, chosen_model)
    return backend_type


def _load_debug_accuracy_module():
    try:
        from .scripts import debug_accuracy
        return debug_accuracy
    except ModuleNotFoundError as exc:
        if exc.name == "tabulate":
            raise SystemExit(
                "[flexipipe] The debug-accuracy command requires the optional 'tabulate' package.\n"
                "Install it with `pip install tabulate` and rerun the command."
            ) from exc
        raise


from .tag_mapping import build_tag_mapping_from_paths
from .train import train_ud_treebank


def detect_input_format(path: str) -> str:
    """Heuristically determine the input format."""
    # Don't handle stdin here - caller should handle it
    if path == "-":
        raise ValueError("detect_input_format cannot handle stdin directly - use run_tag with auto format detection")
    
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            sample = handle.read(4096)
    except OSError as exc:
        raise SystemExit(f"Failed to read input file '{path}': {exc}") from exc

    stripped = sample.lstrip()
    if "<tok" in sample or "<TEI" in sample or stripped.startswith("<TEI"):
        return "teitok"

    for line in sample.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("#"):
            if stripped_line.startswith("# text") or stripped_line.startswith("# sent_id") or stripped_line.startswith("# newdoc"):
                return "conllu"
            continue
        if "\t" in stripped_line and len(stripped_line.split("\t")) >= 10:
            return "conllu"

    return "raw"


def _document_to_plain_text(document: Document) -> str:
    parts: list[str] = []
    for sentence in document.sentences:
        for token in sentence.tokens:
            form = token.form or ""
            parts.append(form)
            if token.space_after is False:
                continue
            parts.append(" ")
        if sentence.tokens:
            parts.append("\n")
    return "".join(parts).strip()


def _print_detected_language(result: Dict[str, Any]) -> None:
    name = result.get("language_name") or result.get("label") or "unknown"
    iso = result.get("language_iso") or result.get("label") or "unknown"
    confidence = result.get("confidence")
    conf_str = f"{confidence:.2%}" if confidence is not None else "n/a"
    detector = result.get("detector")
    if not detector:
        from .model_storage import get_language_detector as _get_detector

        detector = _get_detector() or LANGUAGE_DETECTOR_DEFAULT
    print(f"[flexipipe] Detected language ({detector}): {name} ({iso}, confidence {conf_str})")


def _detect_language_from_text(
    text: Optional[str],
    *,
    explicit: bool = False,
    min_length: int = LANGUAGE_DETECTION_MIN_LENGTH,
    log_failures: bool = False,
    detector_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    from .model_storage import get_language_detector

    if not text:
        if explicit or log_failures:
            print("[flexipipe] Language detection skipped: no text available.")
        return None
    snippet = text.strip()
    if len(snippet) > 20000:
        snippet = snippet[:20000]
    if len(snippet) < min_length:
        if explicit or log_failures:
            print("[flexipipe] Language detection skipped: input text too short.")
        return None

    detector_name = (detector_override or get_language_detector() or LANGUAGE_DETECTOR_DEFAULT)
    if detector_name.lower() == "none":
        if explicit or log_failures:
            print("[flexipipe] Language detection is disabled (detector set to 'none').")
        return None

    try:
        result = detect_language_with(
            detector_name,
            snippet,
            min_length=min_length,
            confidence_threshold=0.0,
            verbose=explicit or log_failures,
        )
    except RuntimeError as exc:
        if explicit or log_failures:
            print(f"[flexipipe] Language detection unavailable ({detector_name}): {exc}")
        return None
    if not result:
        if log_failures:
            print("[flexipipe] Language detection produced no candidates.")
        return None
    confidence = float(result.get("confidence") or 0.0)
    if confidence < LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
        if explicit or log_failures:
            print(
                "[flexipipe] Language could not accurately be detected "
                f"(confidence {confidence:.2%}). Please provide --language."
            )
            # Show top candidates to help debug
            candidates = result.get("candidates") or []
            if candidates:
                print("Top language candidates:")
                for idx, candidate in enumerate(candidates[:5], start=1):
                    iso = candidate.get("language_iso") or candidate.get("label") or "-"
                    name = candidate.get("language_name") or candidate.get("label") or "-"
                    conf = candidate.get("confidence", 0.0)
                    print(f"  {idx}. {name} ({iso}): {conf:.2%}")
        return None
    result.setdefault("detector", detector_name)
    return result


def _maybe_detect_language(
    args: argparse.Namespace,
    text: Optional[str],
    *,
    min_length: int = LANGUAGE_DETECTION_MIN_LENGTH,
) -> Optional[Dict[str, Any]]:
    explicit = bool(getattr(args, "detect_language", False))
    need_detection = explicit or not getattr(args, "language", None)
    if not need_detection:
        return None
    log_failures = explicit or getattr(args, "verbose", False) or getattr(args, "debug", False)
    result = _detect_language_from_text(
        text,
        explicit=explicit,
        min_length=min_length,
        log_failures=log_failures,
    )
    if not result:
        return None
    detected_iso = result.get("language_iso") or result.get("label")
    if not getattr(args, "language", None) and detected_iso:
        args.language = detected_iso
    if explicit or getattr(args, "verbose", False):
        _print_detected_language(result)
    return result


TASK_CHOICES = (
    "process",
    "train",
    "convert",
    "config",
    "info",
    "benchmark",
    "validate",
)


def _run_detect_language_standalone(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flexipipe --detect-language",
        description="Detect language using the configured detector without running tagging.",
    )
    parser.add_argument(
        "--detect-language",
        action="store_true",
        required=True,
        help="Detect language of the provided text or STDIN",
    )
    parser.add_argument(
        "--text",
        help="Text snippet to analyze (optional if using --input or STDIN)",
    )
    parser.add_argument(
        "--data",
        dest="text",
        help="Alias for --text (kept for parity with process command)",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to a file whose contents should be analyzed",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=LANGUAGE_DETECTION_MIN_LENGTH,
        help=f"Minimum number of characters required to run detection (default: {LANGUAGE_DETECTION_MIN_LENGTH})",
    )
    parser.add_argument(
        "--detector",
        choices=sorted(list_language_detectors().keys()),
        help="Language detector backend to use (default: configured value)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show up to K candidate languages (fastText only, default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed detection info (language name, ISO code, candidates, confidence)",
    )
    args = parser.parse_args(argv)
    if getattr(args, "debug", False):
        setattr(args, "verbose", True)

    if args.text:
        text = args.text
    elif args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            return 1
        text = input_path.read_text(encoding="utf-8", errors="ignore")
    else:
        if sys.stdin.isatty():
            print(
                "Error: provide text via --text, --input, or pipe data through STDIN.",
                file=sys.stderr,
            )
            return 1
        text = sys.stdin.read()

    cleaned_text = " ".join(text.strip().split())
    if len(cleaned_text) < args.min_length:
        print(
            f"[flexipipe] Error: Input text is too short ({len(cleaned_text)} characters). "
            f"Minimum length required: {args.min_length} characters.",
            file=sys.stderr,
        )
        return 1

    from .model_storage import get_language_detector

    detector = args.detector or get_language_detector() or LANGUAGE_DETECTOR_DEFAULT
    if detector == "none":
        print(
            "[flexipipe] Error: Language detection is disabled (detector set to 'none'). "
            "Please configure a detector via 'python -m flexipipe config --set-language-detector'.",
            file=sys.stderr,
        )
        return 1

    if detector == "fasttext":
        from .language_utils import detect_language_fasttext

        try:
            result = detect_language_fasttext(
                text,
                min_length=args.min_length,
                confidence_threshold=0.0,
                top_k=max(1, args.top_k),
            )
        except (RuntimeError, ValueError) as exc:
            error_msg = str(exc)
            if "Unable to avoid copy" in error_msg or "copy keyword" in error_msg:
                print(
                    "[flexipipe] Error: fastText failed due to the old wheel being incompatible with NumPy 2.x.\n"
                    "Install the compatibility build with: pip install --upgrade fasttext-numpy2",
                    file=sys.stderr,
                )
            else:
                print(f"[flexipipe] Error: Language detection failed: {exc}", file=sys.stderr)
            return 1
    else:
        result = _detect_language_from_text(
            text,
            explicit=True,
            min_length=args.min_length,
            log_failures=True,
            detector_override=detector,
        )

    if not result:
        print("[flexipipe] Language detection produced no candidates.")
        return 1

    result["detector"] = detector
    if args.verbose:
        _print_detected_language(result)
        candidates = result.get("candidates") or []
        if candidates:
            print("\nTop candidates:")
            print(f"{'Rank':<6} {'ISO':<6} {'Language':<20} {'Confidence':<10}")
            print("-" * 50)
            for idx, candidate in enumerate(candidates[: args.top_k], start=1):
                iso = candidate.get("language_iso") or candidate.get("label") or "-"
                name = candidate.get("language_name") or candidate.get("label") or "-"
                conf = candidate.get("confidence", 0.0)
                print(f"{idx:<6} {iso:<6} {name:<20} {conf:>8.2%}")
        else:
            print(f"[flexipipe] Detector '{detector}' does not expose alternative candidates.")
        return 0

    iso = result.get("language_iso") or result.get("label") or "unknown"
    print(iso)
    return 0


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser with better error messages for unrecognized arguments."""
    
    def error(self, message: str) -> None:
        # Check if it's an unrecognized arguments error
        if "unrecognized arguments" in message.lower():
            # Extract the unrecognized argument
            import re
            # Match "unrecognized arguments: --whitespace join" or similar
            match = re.search(r"unrecognized arguments?:\s*(.+)", message, re.IGNORECASE | re.DOTALL)
            if match:
                unrecognized = match.group(1).strip()
                # Suggest similar arguments
                suggestions = []
                unrecognized_lower = unrecognized.lower()
                if "whitespace" in unrecognized_lower:
                    suggestions.append("--spaceafter-handling")
                elif "spaceafter" in unrecognized_lower or "space-after" in unrecognized_lower:
                    suggestions.append("--spaceafter-handling")
                elif "join" in unrecognized_lower and "spaceafter" not in message.lower():
                    suggestions.append("--spaceafter-handling join")
                
                if suggestions:
                    message += f"\n\nDid you mean: {', '.join(suggestions)}?"
        
        # Suppress BrokenPipeError when exiting due to argument errors
        # This happens when commands are piped and the second command exits early
        import sys
        try:
            super().error(message)
        except SystemExit:
            # Suppress BrokenPipeError on stdout/stderr when exiting
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                pass
            try:
                sys.stderr.flush()
            except BrokenPipeError:
                pass
            raise


def build_parser() -> argparse.ArgumentParser:
    # Get version from package
    try:
        from . import __version__
        version = __version__
    except ImportError:
        # Fallback: use default
        version = "1.0.0"
    
    parser = CustomArgumentParser(
        prog="python -m flexipipe",
        description="Flexipipe pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,  # Disable prefix matching to prevent --teitok from matching --teitok-settings
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {version}",
    )
    # Add global arguments (available in all subcommands)
    parser.add_argument(
        "--backend",
        choices=get_backend_choices(),
        default=None,
        help="Backend type (for use in tasks; default: flexitag)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging and show execution timing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print high-level progress messages",
    )
    subparsers = parser.add_subparsers(dest="task", required=False)
    
    # Create a parent parser with common arguments that all subcommands inherit
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parent_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and show execution timing")
    parent_parser.add_argument("--verbose", action="store_true", help="Print high-level progress messages")
    
    def add_logging_args(p: argparse.ArgumentParser) -> None:
        # --debug and --verbose are now in parent_parser, so we don't add them here
        # But we keep this function for backwards compatibility in case any code calls it
        pass

    def add_udpipe_args(p: argparse.ArgumentParser) -> None:
        pass  # Endpoint URL now handled by unified --endpoint-url

    def add_udmorph_args(p: argparse.ArgumentParser) -> None:
        pass  # Endpoint URL now handled by unified --endpoint-url
    
    def add_nametag_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--nametag-model",
            default=None,
            help="NameTag model name (if not provided, --language will be used)",
        )
        p.add_argument(
            "--nametag-version",
            choices=["1", "2", "3"],
            default="3",
            help=argparse.SUPPRESS,  # Hide from help - older versions don't make much sense
        )
    
    def add_treetagger_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--treetagger-model",
            default=None,
            help="TreeTagger model name from the curated registry (e.g., treetagger-english-20211115).",
        )
        p.add_argument(
            "--treetagger-model-path",
            default=None,
            help="Path to a TreeTagger parameter file (.par). Overrides --treetagger-model.",
        )
        p.add_argument(
            "--treetagger-binary",
            default=None,
            help="Path to the tree-tagger executable (defaults to resolving 'tree-tagger' on PATH).",
        )
        p.add_argument(
            "--treetagger-extra-args",
            action="append",
            default=[],
            metavar="ARGS",
            help="Additional command-line arguments to pass to tree-tagger (may be repeated).",
        )

    
    # process -----------------------------------------------------------------
    process_parser = subparsers.add_parser(
        "process",
        help="Run NLP pipeline (tokenization, tagging, parsing, normalization, NER) on input text or files",
        parents=[parent_parser],
    )
    process_parser.add_argument("--input", default=None, help="Input file (TEITOK XML, CoNLL-U, or raw text). If not provided and STDIN has data, reads from STDIN.")
    process_parser.add_argument(
        "--output",
        default=None,
        help="Output file (omit to write to stdout)",
    )
    process_parser.add_argument(
        "--backend",
        choices=get_backend_choices(),
        default=None,
        help="Backend to use (default: flexitag)",
    )
    process_parser.add_argument(
        "--model",
        default=None,
        help="Model name or path (e.g., 'en_core_web_sm' for SpaCy, path to flexitag model, 'cs_cac' for Stanza, etc.)",
    )
    process_parser.add_argument(
        "--language",
        default=None,
        help="Language code for blank model (e.g., 'en', 'es')",
    )
    process_parser.add_argument(
        "--pretokenize",
        action="store_true",
        help="Segment and tokenize raw input locally before sending to the backend",
    )
    process_parser.add_argument(
        "--use-raw-text",
        action="store_true",
        help="Force backend to process raw text instead of pre-tokenized input (useful for UDPipe to retokenize)",
    )
    process_parser.add_argument(
        "--endpoint-url",
        default=None,
        help="REST backend endpoint URL (defaults vary by backend: UDPipe, UDMorph, NameTag)",
    )
    process_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for REST backend requests in seconds (default: 30)",
    )
    process_parser.add_argument(
        "--params",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional key=value parameters for REST backends (can be repeated, e.g., --params key1=value1 --params key2=value2)",
    )
    process_parser.add_argument(
        "--download-model",
        action="store_true",
        help="Automatically download missing spaCy/Stanza/Flair/TreeTagger models (otherwise prompt when possible)",
    )
    process_parser.add_argument(
        "--stanza-wsd",
        action="store_true",
        help="Enable Stanza word sense disambiguation (requires compatible models)",
    )
    process_parser.add_argument(
        "--stanza-sentiment",
        action="store_true",
        help="Enable Stanza sentiment analysis (requires sentiment models)",
    )
    process_parser.add_argument(
        "--stanza-coref",
        action="store_true",
        help="Enable Stanza coreference resolution (requires coref models)",
    )
    add_udpipe_args(process_parser)
    add_udmorph_args(process_parser)
    add_nametag_args(process_parser)
    add_treetagger_args(process_parser)
    process_parser.add_argument(
        "--transformers-task",
        choices=["tag", "ner"],
        help="Override Transformers token-classification task detection (default: auto-detect based on model labels).",
    )
    process_parser.add_argument(
        "--transformers-adapter",
        help="Adapter name or identifier to load (if the model exposes adapters).",
    )
    process_parser.add_argument(
        "--transformers-device",
        default="cpu",
        help="Device for Transformers backend ('cpu', 'cuda', 'cuda:0', 'mps', etc.).",
    )
    process_parser.add_argument(
        "--transformers-revision",
        help="Specific model revision (branch, tag, or commit hash) to load from HuggingFace.",
    )
    process_parser.add_argument(
        "--transformers-trust-remote-code",
        action="store_true",
        help="Allow executing remote model code when loading a Transformers model (use with caution).",
    )
    process_parser.add_argument(
        "--transformers-context",
        help="Comma-separated list of token attributes to include as context for Transformers models (e.g., 'upos,lemma').",
    )
    process_parser.add_argument(
        "--output-format",
        "--output-form",
        choices=["teitok", "conllu", "conllu-ne", "json"],
        default=None,
        help="Output format (default: from config, or teitok)",
    )
    process_parser.add_argument(
        "--pretty-print",
        "--prettyprint",
        action="store_true",
        help="Pretty-print TEITOK XML output with indentation (newlines don't add whitespace between tokens)",
    )
    process_parser.add_argument(
        "--spaceafter-handling",
        choices=["preserve", "join"],
        default="preserve",
        help="How to handle SpaceAfter=No tokens when pretty-printing: 'preserve' (default) removes spaces, 'join' adds join='right' attribute",
    )
    process_parser.add_argument(
        "--skip-spaceafter-for-breaking-elements",
        action="store_true",
        default=True,
        help="Skip SpaceAfter handling if there's a hard-breaking element (like <p>) between tokens (default: True)",
    )
    process_parser.add_argument(
        "--no-skip-spaceafter-for-breaking-elements",
        dest="skip_spaceafter_for_breaking_elements",
        action="store_false",
        help="Don't skip SpaceAfter handling for breaking elements",
    )
    process_parser.add_argument(
        "--create-implicit-mwt",
        action="store_true",
        help="Create implicit MWTs from SpaceAfter=No sequences (CoNLL-U output only, excludes punctuation)",
    )
    process_parser.add_argument(
        "--input-format",
        choices=["auto", "teitok", "conllu", "conllu-ne", "raw"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    process_parser.add_argument(
        "--nlpform",
        choices=["form", "reg"],
        default="form",
        help="Surface form to run NLP components on (default: use original form). "
             "Use 'reg' to substitute Normalization/reg values when available.",
    )
    process_parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks to perform (tokenize,segment,lemmatize,tag,parse,normalize,ner). "
             "Defaults to all tasks.",
    )
    process_parser.add_argument(
        "--data",
        nargs="+",
        metavar="TEXT",
        help="Provide raw text directly on the command line instead of --input. "
             "Example: --data This is a test",
    )
    process_parser.add_argument(
        "--example",
        metavar="NAME",
        help="Load example text from tmp/examples.json. Currently supported: 'udhr' (Universal Declaration of Human Rights). "
             "Requires --language to be specified to select the appropriate language version.",
    )
    process_parser.add_argument(
        "--attrs-map",
        action="append",
        metavar="ATTR:VALUES",
        help="Map TEITOK attributes (can be repeated). Format: 'attr:value1,value2' (e.g., 'xpos:msd,pos' or 'reg:nform'). "
             "Supported attributes: xpos, reg, expan, lemma, tokid",
    )
    process_parser.add_argument(
        "--normalization-style",
        choices=["conservative", "aggressive", "enhanced", "balanced"],
        default="conservative",
        help="Normalization style: conservative (explicit mappings only), aggressive (pattern-based substitutions), "
             "enhanced (morphological variations), balanced (combination, default: conservative)",
    )
    process_parser.add_argument(
        "--extra-vocab",
        action="append",
        default=[],
        help="Additional vocabulary file(s) to merge with the main model (can be specified multiple times)",
    )
    process_parser.add_argument(
        "--writeback",
        action="store_true",
        default=None,
        help="Update the original TEITOK XML file in-place (only works when input and output are TEITOK XML). Preserves original structure and only updates annotation attributes.",
    )
    process_parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Enable tokenization mode for non-tokenized TEITOK XML files. If the input file has no <tok> elements, extract plain text and tokenize it.",
    )
    process_parser.add_argument(
        "--textnode",
        type=str,
        metavar="XPATH",
        default=".//text",
        help="XPath expression to locate the text node in TEITOK XML when using --tokenize (default: './/text').",
    )
    process_parser.add_argument(
        "--textnotes",
        action="store_true",
        help="Include <note> elements in extracted text when using --tokenize (default: exclude notes).",
    )
    process_parser.add_argument(
        "--teitok",
        action="store_true",
        help="Enable TEITOK mode: automatically load settings from tmp/cqpsettings.xml (merged shared+local) or ./Resources/settings.xml. Auto-enables writeback if settings.xml is found. Use --teitok-settings to specify a custom path.",
    )
    process_parser.add_argument(
        "--teitok-settings",
        type=str,
        metavar="PATH",
        help="Path to TEITOK settings.xml file. If not specified and --teitok is used, flexipipe will look for tmp/cqpsettings.xml (merged shared+local) or ./Resources/settings.xml. Otherwise, it will search for settings.xml in the corpus directory and parent directories. Settings.xml provides attribute mappings, default language, and other corpus-specific configuration.",
    )
    process_parser.add_argument(
        "--known-tags-only",
        action="store_true",
        help="When writing TEITOK XML, only write attributes that are explicitly defined in the TEITOK settings. Attributes not in the settings will be omitted.",
    )
    process_parser.add_argument(
        "--test",
        action="store_true",
        help="Test writeback mode: render TEITOK XML using writeback logic but emit to stdout without modifying files.",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --test (dry-run writeback without modifying files).",
    )
    process_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the UD validator on the produced CoNLL-U output (requires --output and --output-format conllu or conllu-ne).",
    )
    process_parser.add_argument(
        "--validator-rules",
        default=None,
        help="Which validation profile to pass to the UD validator (--validation value, e.g., core, all, experimental). If not set, smart validation filters errors about missing annotations.",
    )
    process_parser.add_argument(
        "--no-smart-validation",
        action="store_true",
        help="Disable smart validation that filters errors about missing annotations. Use this to see all validation errors.",
    )
    process_parser.add_argument(
        "--ud-tools-version",
        default="latest",
        help="UD tools release to download/use for validation (e.g., 2.17). Use 'latest' (default) to follow the registry.",
    )
    process_parser.add_argument(
        "--ud-tools-url",
        help="Override download URL for the ud-tools archive (used when the default release URL changes).",
    )
    process_parser.add_argument(
        "--validator-arg",
        dest="validator_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Additional arguments to forward to the UD validator (repeatable).",
    )
    process_parser.add_argument(
        "--validator-lang",
        default=None,
        help="Language code to pass to the UD validator (e.g., en, de). If omitted, flexipipe tries to infer it from the document metadata.",
    )
    
    # Tag mapping options (for enriching tags from vocabulary)
    process_parser.add_argument(
        "--map-tags-model",
        dest="map_tags_models",
        action="append",
        help="Model vocab JSON file for tag mapping (can be specified multiple times to merge mappings). Enables tag mapping to fill missing XPOS or UPOS/FEATS.",
    )
    process_parser.add_argument(
        "--map-direction",
        choices=["xpos", "upos-feats", "both"],
        default=None,
        help="What to infer when using --map-tags-model: xpos -> fill XPOS, upos-feats -> fill UPOS/FEATS, both -> do both",
    )
    process_parser.add_argument(
        "--fill-xpos",
        dest="fill_xpos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable filling XPOS from UPOS+FEATS when using --map-tags-model (overrides map direction)",
    )
    process_parser.add_argument(
        "--fill-upos",
        dest="fill_upos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable filling UPOS from existing XPOS when using --map-tags-model (overrides map direction)",
    )
    process_parser.add_argument(
        "--fill-feats",
        dest="fill_feats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable filling FEATS from existing XPOS when using --map-tags-model (overrides map direction)",
    )
    process_parser.add_argument(
        "--allow-partial",
        dest="allow_partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow partial (subset) matches when inferring tags with --map-tags-model (default: True)",
    )
    
    # info ---------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="List information about available backends and models, or detect language",
        parents=[parent_parser],
    )
    add_logging_args(info_parser)
    info_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    info_parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Detect language of the provided text or STDIN",
    )
    info_parser.add_argument(
        "--text",
        help="Text snippet to analyze (optional if using --input or STDIN)",
    )
    info_parser.add_argument(
        "--input",
        "-i",
        help="Path to a file whose contents should be analyzed",
    )
    info_parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum number of characters required to run detection (default: 20)",
    )
    info_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show up to K candidate languages when --verbose is used (default: 3)",
    )
    info_subparsers = info_parser.add_subparsers(dest="info_action", required=False, help="Information to list")
    
    # info backends
    backends_parser = info_subparsers.add_parser(
        "backends",
        help="List all available backends",
        parents=[parent_parser],
    )
    backends_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    
    # info languages
    languages_parser = info_subparsers.add_parser(
        "languages",
        help="List all languages that have models available",
        parents=[parent_parser],
    )
    languages_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    
    # info models
    models_parser = info_subparsers.add_parser(
        "models",
        help="List available models for a backend or language",
        parents=[parent_parser],
    )
    
    # info ud-tags
    ud_tags_parser = info_subparsers.add_parser(
        "ud-tags",
        help="List Universal Dependencies tags repository information",
        parents=[parent_parser],
    )
    ud_tags_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    ud_tags_parser.add_argument(
        "--category",
        choices=["upos", "feats", "misc", "document", "sentence", "all"],
        default="all",
        help="Category to display (default: all)",
    )
    
    # info examples
    examples_parser = info_subparsers.add_parser(
        "examples",
        help="List locally available example datasets",
        parents=[parent_parser],
    )
    examples_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    examples_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh of example metadata (re-download files if needed)",
    )

    # validate -----------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run the official UD validator (ud-tools) on one or more CoNLL-U files or treebank directories",
        parents=[parent_parser],
    )
    validate_parser.add_argument(
        "treebank",
        nargs="+",
        help="Path(s) to CoNLL-U file(s) or directories to validate",
    )
    validate_parser.add_argument(
        "--rules",
        dest="validator_rules",
        default="core",
        help="Validation profile to use (passed as --validation to ud-tools).",
    )
    validate_parser.add_argument(
        "--ud-tools-version",
        default="latest",
        help="UD tools release to download/use (e.g., 2.17). Use 'latest' (default) to follow the registry.",
    )
    validate_parser.add_argument(
        "--ud-tools-url",
        help="Override download URL for the ud-tools archive.",
    )
    validate_parser.add_argument(
        "--validator-arg",
        dest="validator_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Additional arguments passed directly to ud-tools/validate.py (repeatable).",
    )
    validate_parser.add_argument(
        "--validator-lang",
        default=None,
        help="Language code to pass to the UD validator (e.g., en, de). Required when flexipipe cannot infer it from the files.",
    )
    validate_parser.add_argument(
        "--no-smart-validation",
        action="store_true",
        help="Disable smart validation that filters errors about missing annotations. Use this to see all validation errors.",
    )

    # info tasks
    tasks_parser = info_subparsers.add_parser(
        "tasks",
        help="List NLP tasks supported by flexipipe",
        parents=[parent_parser],
    )
    tasks_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    
    # info teitok
    teitok_parser = info_subparsers.add_parser(
        "teitok",
        help="Display TEITOK settings.xml configuration (attribute mappings, language, etc.)",
        parents=[parent_parser],
    )
    teitok_parser.add_argument(
        "--teitok",
        action="store_true",
        help="Enable TEITOK mode: automatically load settings from ./Resources/settings.xml (or search for settings.xml). Use --settings to specify a custom path.",
    )
    teitok_parser.add_argument(
        "--settings",
        dest="teitok_settings",
        type=str,
        metavar="PATH",
        help="Path to settings.xml file. If not specified and --teitok is used, looks for tmp/cqpsettings.xml (merged shared+local) or ./Resources/settings.xml. Otherwise, searches for settings.xml in the current directory and parent directories.",
    )
    teitok_parser.add_argument(
        "--corpus",
        type=str,
        metavar="PATH",
        help="Path to TEITOK corpus directory or XML file (used to search for settings.xml)",
    )
    teitok_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    models_parser.add_argument(
        "--backend",
        choices=get_backend_choices(),
        help="Backend type (required unless --language is provided)",
    )
    models_parser.add_argument(
        "--language",
        help="Filter model listings by language name or ISO code (searches across all backends if --backend not specified)",
    )
    models_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh of cached model listings",
    )
    models_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    # Add REST backend URL arguments for models listing
    models_parser.add_argument(
        "--endpoint-url",
        help="REST backend endpoint URL (for UDPipe, UDMorph, NameTag backends)",
    )
    models_parser.add_argument(
        "--sort",
        choices=["backend", "model", "language", "iso", "status"],
        default="backend",
        help="Sort order for models (default: backend). Options: backend, model, language, iso, status",
    )
    models_parser.add_argument(
        "--include-base-models",
        action="store_true",
        help="Include base/LLM-style Transformers models (without finetuning) when listing --backend transformers",
    )
    
    # config ---------------------------------------------------------------
    config_parser = subparsers.add_parser("config", help="Configure flexipipe settings")
    config_parser.add_argument(
        "--set-model-registry-local-dir",
        type=Path,
        metavar="PATH",
        help="Set the local directory for model registries (e.g., /path/to/flexipipe-models). Registry files should be in registries/ subdirectory.",
    )
    config_parser.add_argument(
        "--set-model-registry-base-url",
        type=str,
        help="Set the base URL for remote model registries (e.g., https://raw.githubusercontent.com/org/repo/main/registries)",
    )
    config_parser.add_argument(
        "--set-model-registry-url",
        type=str,
        metavar="BACKEND:URL",
        help="Set a specific registry URL for a backend (format: backend:url, e.g., flexitag:https://example.com/registry.json)",
    )
    config_parser.add_argument(
        "--set-models-dir",
        type=Path,
        metavar="PATH",
        help="Set the models directory (where all backend models are stored)",
    )
    config_parser.add_argument(
        "--refresh-all-caches",
        action="store_true",
        help="Refresh all model caches at once (recommended when caches are outdated)",
    )
    config_parser.add_argument(
        "--set-default-backend",
        choices=get_backend_choices(),
        metavar="BACKEND",
        help="Set the default backend to use",
    )
    config_parser.add_argument(
        "--set-default-output-format",
        choices=["teitok", "conllu", "conllu-ne", "json"],
        metavar="FORMAT",
        help="Set the default output format (teitok, conllu, conllu-ne, or json)",
    )
    detector_choices = sorted(list_language_detectors().keys())
    config_parser.add_argument(
        "--set-language-detector",
        choices=detector_choices,
        metavar="DETECTOR",
        help="Set the default language detector backend (e.g., fasttext, none)",
    )
    config_parser.add_argument(
        "--set-default-create-implicit-mwt",
        type=_str_to_bool,
        metavar="true|false",
        help="Set whether to create implicit MWTs by default (true or false)",
    )
    config_parser.add_argument(
        "--set-default-writeback",
        type=_str_to_bool,
        metavar="true|false",
        help="Set whether to enable writeback by default (true or false)",
    )
    config_parser.add_argument(
        "--set-auto-install-extras",
        type=_str_to_bool,
        metavar="true|false",
        help="Automatically install missing optional extras when possible (true or false)",
    )
    config_parser.add_argument(
        "--set-prompt-install-extras",
        type=_str_to_bool,
        metavar="true|false",
        help="Prompt before installing optional extras when auto-install is disabled (true or false)",
    )
    config_parser.add_argument(
        "--wizard",
        action="store_true",
        help="Run interactive configuration wizard",
    )
    config_parser.add_argument(
        "--download-language-model",
        action="store_true",
        help="Download or refresh the fastText language detection model",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )
    
    # train ---------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train a model from training data")
    add_logging_args(train_parser)
    train_parser.add_argument(
        "--backend",
        choices=get_backend_choices_for_training(),
        default="flexitag",
        help="Backend to use for training (default: flexitag)",
    )
    train_parser.add_argument(
        "--train-data",
        type=Path,
        help="Training data file or directory (CoNLL-U format). For flexitag: directory containing *-ud-(train|dev|test).conllu files. For neural backends: file or directory with CoNLL-U data.",
    )
    train_parser.add_argument(
        "--dev-data",
        type=Path,
        help="Development/validation data file or directory (CoNLL-U format, for neural backends)",
    )
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the trained model (optional for SpaCy; defaults to flexipipe models dir)",
    )
    train_parser.add_argument(
        "--name",
        default=None,
        help="Optional model name to store in metadata (defaults to treebank folder name for flexitag)",
    )
    train_parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Include the dev split when building the vocabulary (in addition to train, for flexitag backend)",
    )
    train_parser.add_argument(
        "--tagpos",
        choices=["xpos", "upos", "utot", "auto"],
        default=None,
        help="Tag attribute to train on (default: auto-select, for flexitag backend)"
    )
    train_parser.add_argument(
        "--finetune",
        choices=["none", "accuracy", "speed", "balanced"],
        default="balanced",
        help="Run a lightweight grid search over tagger settings (accuracy, speed, balanced, or none; default: balanced, for flexitag backend)",
    )
    train_parser.add_argument(
        "--nlpform",
        choices=["form", "reg"],
        default="form",
        help="Surface form to train on (default: original form). Use 'reg' to substitute Normalization/reg values when present.",
    )
    # Neural backend arguments
    train_parser.add_argument(
        "--model",
        help="Base model name for neural backends (e.g., 'en_core_web_sm' for SpaCy, 'bert-base-uncased' for Transformers)",
    )
    train_parser.add_argument(
        "--language",
        help="Language code for the model (stored in flexitag metadata and used by neural backends when required)",
    )
    train_parser.add_argument(
        "--ud-folder",
        type=str,
        default=None,
        help="Output folder for CoNLL-U files created from TEITOK corpora (useful for publishing as UD treebank). If specified, files are kept in this folder instead of a temporary directory.",
    )
    train_parser.add_argument(
        "--xpos",
        help="Attribute name(s) to use for xpos in TEITOK files (comma-separated, e.g., 'pos,msd'). Tried in order, falls back to 'xpos'.",
    )
    train_parser.add_argument(
        "--reg",
        help="Attribute name(s) to use for reg/normalization in TEITOK files (comma-separated, e.g., 'nform,fform'). Tried in order, falls back to 'reg' then 'form'.",
    )
    train_parser.add_argument(
        "--expan",
        help="Attribute name(s) to use for expan/expansion in TEITOK files (comma-separated, e.g., 'fform'). Tried in order, falls back to 'expan' then 'form'.",
    )
    train_parser.add_argument(
        "--lemma",
        help="Attribute name(s) to use for lemma in TEITOK files (comma-separated). Tried in order, falls back to 'lemma' then 'form'.",
    )
    train_parser.add_argument(
        "--udpipe1-tokenizer",
        dest="udpipe1_tokenizer",
        help="Override UDPipe CLI tokenizer training options (e.g., 'epochs=50:early_stopping=1').",
    )
    train_parser.add_argument(
        "--udpipe1-tagger",
        dest="udpipe1_tagger",
        help="Override UDPipe CLI tagger training options.",
    )
    train_parser.add_argument(
        "--udpipe1-parser",
        dest="udpipe1_parser",
        help="Override UDPipe CLI parser training options.",
    )
    
    # convert ------------------------------------------------------
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between formats: tagged files, treebanks (TEITOK to UD), or lexicons"
    )
    add_logging_args(convert_parser)
    convert_parser.add_argument(
        "--type",
        choices=["tagged", "treebank", "lexicon"],
        default="tagged",
        help="Type of conversion: 'tagged' (format conversion), 'treebank' (TEITOK to UD splits), 'lexicon' (external lexicon to vocabulary) (default: tagged)",
    )
    
    # Common arguments
    convert_parser.add_argument(
        "--input",
        "-i",
        help="Input file or directory (default: STDIN for tagged conversion)",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        help="Output file or directory (default: STDOUT for tagged conversion)",
    )
    
    # Arguments for tagged conversion (format conversion)
    convert_parser.add_argument(
        "--input-format",
        choices=["auto", "teitok", "conllu", "conllu-ne", "raw"],
        default="auto",
        help="Input format for tagged conversion (default: auto-detect)",
    )
    convert_parser.add_argument(
        "--output-format",
        "--output-form",
        choices=["teitok", "conllu", "conllu-ne", "json"],
        help="Output format for tagged conversion (required for --type tagged)",
    )
    convert_parser.add_argument(
        "--tasks",
        help="Comma-separated list of tasks to preserve in output (for tagged conversion)",
    )
    
    # Arguments for treebank conversion (prepare-ud)
    convert_parser.add_argument(
        "--backend",
        choices=get_backend_choices_for_training(),  # Convert uses same backends as training
        default="spacy",
        help="Target backend for treebank conversion (default: spacy)",
    )
    convert_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for treebank conversion (default: 0.8)",
    )
    convert_parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Dev split ratio for treebank conversion (default: 0.1)",
    )
    convert_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for treebank conversion (default: 0.1)",
    )
    convert_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sentence splitting in treebank conversion (default: 42)",
    )
    convert_parser.add_argument(
        "--xpos",
        help="Comma-separated TEITOK attribute names for xpos fallback (for treebank conversion)",
    )
    convert_parser.add_argument(
        "--reg",
        help="Comma-separated TEITOK attribute names for reg fallback (for treebank conversion)",
    )
    convert_parser.add_argument(
        "--expan",
        help="Comma-separated TEITOK attribute names for expan fallback (for treebank conversion)",
    )
    convert_parser.add_argument(
        "--lemma",
        help="Comma-separated TEITOK attribute names for lemma fallback (for treebank conversion)",
    )
    
    # Arguments for lexicon conversion
    convert_parser.add_argument(
        "--tagset",
        help="Optional tagset.xml file for XPOS to UPOS+FEATS mapping (for lexicon conversion)",
    )
    convert_parser.add_argument(
        "--corpus",
        help="Optional corpus file (CoNLL-U or TEITOK XML) to extract XPOS tags for matching (for lexicon conversion)",
    )
    convert_parser.add_argument(
        "--default-count",
        type=int,
        default=1,
        help="Default count for lexicon entries (for lexicon conversion, default: 1)",
    )
    
    # benchmark -----------------------------------------------------------
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark backend/model combinations on available treebanks",
        parents=[parent_parser],
    )
    benchmark_parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmark sweep across the selected languages/backends",
    )
    benchmark_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benchmarks even if results already exist (use with --run)",
    )
    benchmark_parser.add_argument(
        "--download-models",
        action="store_true",
        help="Automatically download missing models when needed (use with --run)",
    )
    benchmark_parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single backend/model evaluation for quick validation (can be used standalone like 'check' command)",
    )
    benchmark_parser.add_argument(
        "--show",
        action="store_true",
        help="Display stored benchmark results",
    )
    benchmark_parser.add_argument(
        "--average",
        action="store_true",
        help="Show averaged metrics across all treebanks for each backend/model (use with --show)",
    )
    benchmark_parser.add_argument(
        "--sort-by",
        choices=["upos", "xpos", "feats_partial", "featsp", "lemma", "uas", "las", "tokens_per_second", "tps"],
        default="upos",
        help="Sort results by this metric (default: upos). Use with --show.",
    )
    benchmark_parser.add_argument(
        "--languages",
        nargs="+",
        help="ISO codes of languages to benchmark, or 'all' for all available (default: auto-detect from treebanks)",
    )
    benchmark_parser.add_argument(
        "--language",
        help="Filter results by a single language ISO code (use with --show)",
    )
    benchmark_parser.add_argument(
        "--backends",
        nargs="+",
        help="List of backend names to benchmark (e.g., flexitag spacy stanza), or 'all' for all available",
    )
    benchmark_parser.add_argument(
        "--models",
        action="append",
        metavar="BACKEND=MODEL",
        help="Backend-to-model mapping (repeatable). Example: --models spacy=es_core_news_sm",
    )
    benchmark_parser.add_argument(
        "--treebank",
        action="append",
        help="Explicit treebank file(s) to benchmark (can repeat). Overrides discovery.",
    )
    benchmark_parser.add_argument(
        "--treebank-root",
        help="Root directory containing UD treebanks (for discovery mode).",
    )
    benchmark_parser.add_argument(
        "--treebanks-file",
        help="JSON catalog of treebank test files (language/id/path).",
    )
    benchmark_parser.add_argument(
        "--models-file",
        help="JSON catalog of available models (backend/model/language).",
    )
    benchmark_parser.add_argument(
        "--export-treebanks",
        nargs="?",
        const="__DEFAULT_TREEBANK__",
        help="Write discovered treebank catalog to the given JSON file (omit path to use the default benchmark directory).",
    )
    benchmark_parser.add_argument(
        "--export-models",
        nargs="?",
        const="__DEFAULT_MODEL__",
        help="Write discovered model catalog to the given JSON file (omit path to use the default benchmark directory). Use 'web' to write to web/models.json.",
    )
    benchmark_parser.add_argument(
        "--export-benchmark",
        help="Write benchmark results to the given JSON file. Use 'web' to write to web/benchmark.json.",
    )
    benchmark_parser.add_argument(
        "--list-treebanks",
        action="store_true",
        help="Show treebank catalog and exit (unless another action is specified).",
    )
    benchmark_parser.add_argument(
        "--list-tests",
        action="store_true",
        help="Show combined language coverage (treebanks vs. models) and exit unless another action is specified.",
    )
    benchmark_parser.add_argument(
        "--list-test",
        action="store_true",
        help=argparse.SUPPRESS,
        dest="list_tests",
    )
    add_udpipe_args(benchmark_parser)
    benchmark_parser.add_argument(
        "--tasks",
        help="Comma-separated task profile labels (e.g., tagging,parsing,ner).",
    )
    benchmark_parser.add_argument(
        "--mode",
        choices=["auto", "raw", "tokenized", "split"],
        default="auto",
        help="Evaluation mode: 'auto' (default, tokenized for CoNLL-U, raw for TEI), 'raw' (re-tokenize), 'tokenized' (preserve tokenization), 'split' (preserve MWTs).",
    )
    benchmark_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format for --show and listing commands (default: table). Use 'json' for machine-readable output.",
    )
    benchmark_parser.add_argument(
        "--debug-flexitag",
        action="store_true",
        help=argparse.SUPPRESS,  # Hide from help - internal use only
    )
    benchmark_parser.add_argument(
        "--debug-flexitag-model",
        help=argparse.SUPPRESS,  # Path to flexitag model for --debug-flexitag
    )
    benchmark_parser.add_argument(
        "--debug-flexitag-test",
        help=argparse.SUPPRESS,  # Path to test file for --debug-flexitag
    )
    benchmark_parser.add_argument(
        "--debug-flexitag-output",
        help=argparse.SUPPRESS,  # Output path for --debug-flexitag
    )
    benchmark_parser.add_argument(
        "--debug-flexitag-endlen",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # Override endlen for --debug-flexitag
    )
    benchmark_parser.add_argument(
        "--results-file",
        help="Path to benchmark results JSON file (default: ~/.flexipipe/benchmark_results.json)",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        help="Directory to store intermediate benchmark artefacts.",
    )
    benchmark_parser.add_argument(
        "--limit-treebanks",
        type=int,
        default=None,
        help="Limit number of treebanks per language (useful for dry runs).",
    )
    benchmark_parser.add_argument(
        "--limit-jobs",
        type=int,
        default=None,
        help="Limit total benchmark jobs executed in this run.",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without executing them.",
    )
    
    return parser


def _parse_key_value_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parse KEY=VALUE arguments into a dictionary."""
    params: dict[str, str] = {}
    if not pairs:
        return params
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Invalid parameter '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid parameter '{item}'. Key cannot be empty.")
        params[key] = value.strip()
    return params


def _load_and_merge_teitok_settings(
    args: argparse.Namespace,
    input_path: Optional[str],
) -> tuple[Optional[Any], dict[str, str]]:
    """
    Load TEITOK settings.xml and merge with CLI attribute mappings.
    
    Args:
        args: Command-line arguments
        input_path: Path to input file (used to search for settings.xml)
        
    Returns:
        Tuple of (TeitokSettings object or None, merged attribute mappings dict)
    """
    from .teitok_settings import load_teitok_settings, find_settings_xml
    
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
    elif input_path:
        # Search from input file location
        found_path = find_settings_xml(Path(input_path))
        if found_path:
            settings_path = found_path
    
    # Load settings
    settings = None
    if settings_path:
        settings = load_teitok_settings(settings_path=settings_path)
        # Apply --known-tags-only flag if specified (overrides settings.xml)
        if getattr(args, "known_tags_only", False):
            settings.known_tags_only = True
        # Only report success if settings.xml actually exists and was loaded
        if settings.settings_path and settings.settings_path.exists():
            if args.verbose or args.debug:
                print(f"[flexipipe] Loaded TEITOK settings from {settings.settings_path}", file=sys.stderr)
        elif teitok_mode:
            # --teitok was used but settings.xml not found
            if args.verbose or args.debug:
                print(f"[flexipipe] --teitok flag used but settings.xml not found at {settings_path}", file=sys.stderr)
    
    # Parse CLI attribute mappings
    cli_attrs_map = _parse_attrs_map(getattr(args, "attrs_map", None))
    
    # Merge: CLI takes precedence, then settings.xml, then defaults
    merged_mappings = {}
    for attr_name in ["xpos", "reg", "expan", "lemma", "tokid"]:
        # Start with settings.xml mapping if available
        if settings:
            settings_mapping = settings.get_attribute_mapping(attr_name)
            # Convert list to comma-separated string for compatibility
            merged_mappings[attr_name] = ",".join(settings_mapping)
        else:
            # Use defaults
            from .teitok_settings import DEFAULT_ATTRIBUTE_MAPPINGS
            default_mapping = DEFAULT_ATTRIBUTE_MAPPINGS.get(attr_name, [attr_name])
            merged_mappings[attr_name] = ",".join(default_mapping)
        
        # CLI overrides settings
        if attr_name in cli_attrs_map:
            merged_mappings[attr_name] = cli_attrs_map[attr_name]
    
    return settings, merged_mappings


def _parse_attrs_map(attrs_map: list[str] | None) -> dict[str, str]:
    """
    Parse --attrs-map arguments into attribute mappings.
    
    Format: 'attr:value1,value2' (e.g., 'xpos:msd,pos' or 'reg:nform')
    Returns a dict mapping attribute names to comma-separated value lists.
    """
    mappings: dict[str, str] = {}
    if not attrs_map:
        return mappings
    for item in attrs_map:
        if ":" not in item:
            raise SystemExit(f"Invalid attrs-map '{item}'. Expected ATTR:VALUES (e.g., 'xpos:msd,pos').")
        attr, values = item.split(":", 1)
        attr = attr.strip().lower()
        values = values.strip()
        if not attr:
            raise SystemExit(f"Invalid attrs-map '{item}'. Attribute name cannot be empty.")
        if not values:
            raise SystemExit(f"Invalid attrs-map '{item}'. Values cannot be empty.")
        # Normalize attribute names
        if attr in ("xpos", "pos", "msd"):
            mappings["xpos"] = values
        elif attr in ("reg", "nform", "normalization"):
            mappings["reg"] = values
        elif attr in ("expan", "fform", "expansion"):
            mappings["expan"] = values
        elif attr in ("lemma", "lem"):
            mappings["lemma"] = values
        elif attr in ("tokid", "id", "tokenid"):
            mappings["tokid"] = values
        else:
            raise SystemExit(f"Unknown attribute '{attr}' in attrs-map. Supported: xpos, reg, expan, lemma, tokid")
    return mappings


def _parse_stanza_model_spec(
    model_name: Optional[str],
    language: Optional[str],
    package: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse Stanza model specification.
    
    Supports:
    - --neural-model cs_cac -> language=cs, package=cac
    - --neural-language cs --stanza-package cac -> language=cs, package=cac
    - --neural-model cs -> language=cs, package=None
    
    Returns:
        (language, package, model_name) tuple
    """
    # If package is explicitly set, use it
    final_package = package
    
    # If model_name contains underscore, try to parse as lang_package
    if model_name and "_" in model_name and not language:
        parts = model_name.split("_", 1)
        if len(parts) == 2:
            # Check if it looks like lang_package (first part is short, second is longer)
            lang_part, pkg_part = parts
            if len(lang_part) <= 3 and len(pkg_part) > 0:
                # Looks like lang_package format
                final_language = lang_part
                if not final_package:
                    final_package = pkg_part
                final_model_name = None  # Don't pass model_name, use language instead
                return (final_language, final_package, final_model_name)
    
    # Otherwise, use model_name as-is or language
    final_language = language
    final_model_name = model_name if model_name and not language else None
    
    return (final_language, final_package, final_model_name)


def _parse_classla_model_spec(
    model_name: Optional[str],
    language: Optional[str],
    package: Optional[str],
    type: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse ClassLA model specification.
    
    Supports:
    - --model mk-standard -> language=mk, type=standard
    - --model sr-nonstandard -> language=sr, type=nonstandard
    - --model mk -> language=mk, type=standard (default)
    - --language mk --classla-type nonstandard -> language=mk, type=nonstandard
    
    Model names should be in format: lang-type (e.g., mk-standard, sr-nonstandard)
    since you need both language and type to download the model.
    
    Returns:
        (language, package, model_name, type) tuple
    """
    # If type is explicitly set, use it
    final_type = type or "standard"
    final_package = package  # Package is usually inferred from language
    
    # If model_name contains hyphen, try to parse as lang-type
    if model_name and "-" in model_name and not language:
        parts = model_name.split("-", 1)
        if len(parts) == 2:
            lang_part, type_part = parts
            if type_part in ("standard", "nonstandard"):
                # Format: lang-type (e.g., mk-standard, sr-nonstandard)
                final_language = lang_part
                final_type = type_part
                final_model_name = None
                return (final_language, final_package, final_model_name, final_type)
            elif len(lang_part) <= 3:
                # Format: lang-something (might be lang-package, treat as lang)
                final_language = lang_part
                final_model_name = None
                return (final_language, final_package, final_model_name, final_type)
    
    # If model_name is just a language code
    if model_name and len(model_name) <= 3 and not language:
        final_language = model_name
        final_model_name = None
        return (final_language, final_package, final_model_name, final_type)
    
    # Otherwise, use model_name as-is or language
    final_language = language
    final_model_name = model_name if model_name and not language else None
    
    return (final_language, final_package, final_model_name, final_type)


def _build_udpipe_backend_kwargs(args: argparse.Namespace) -> dict:
    """Gather UDPipe-specific backend kwargs from CLI args."""
    backend = getattr(args, "backend", None)
    if not backend or backend.lower() != "udpipe":
        return {}
    endpoint_url = getattr(args, "endpoint_url", None) or "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    kwargs: dict[str, object] = {
        "endpoint_url": endpoint_url,
        "timeout": getattr(args, "timeout", 30.0),  # Use unified --timeout
        "log_requests": bool(getattr(args, "debug", False)),
    }
    model_name = getattr(args, "model", None)
    if model_name:
        kwargs["model"] = model_name
    # Use unified --params for REST backend parameters
    extra_params = _parse_key_value_pairs(getattr(args, "params", []))
    if extra_params:
        kwargs["extra_params"] = extra_params
    return kwargs


def _build_udmorph_backend_kwargs(args: argparse.Namespace) -> dict:
    """Gather UDMorph-specific backend kwargs from CLI args."""
    backend = getattr(args, "backend", None)
    if not backend or backend.lower() != "udmorph":
        return {}
    endpoint_url = getattr(args, "endpoint_url", None) or "https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=tag"
    kwargs: dict[str, object] = {
        "endpoint_url": endpoint_url,
        "timeout": getattr(args, "timeout", 30.0),  # Use unified --timeout
        "log_requests": bool(getattr(args, "debug", False)),
    }
    model_name = getattr(args, "model", None)
    if model_name:
        entry = get_udmorph_model_entry(model_name)
        if not entry:
            raise ValueError(
                f"UDMorph model '{model_name}' is not available or not supported via the UDMorph backend. "
                "UDPIPE2-based models should be used with --backend udpipe."
            )
        endpoint_override = entry.get("endpoint_url")
        if endpoint_override:
            kwargs["endpoint_url"] = endpoint_override
        kwargs["model"] = model_name
    # Use unified --params for REST backend parameters
    extra_params = _parse_key_value_pairs(getattr(args, "params", []))
    if extra_params:
        kwargs["extra_params"] = extra_params
    return kwargs


def _build_nametag_backend_kwargs(args: argparse.Namespace) -> dict:
    """Gather NameTag-specific backend kwargs from CLI args."""
    backend = getattr(args, "backend", None)
    if not backend or backend.lower() != "nametag":
        return {}
    endpoint_url = getattr(args, "endpoint_url", None) or "https://lindat.mff.cuni.cz/services/nametag/api/recognize"
    kwargs: dict[str, object] = {
        "endpoint_url": endpoint_url,
        "timeout": getattr(args, "timeout", 30.0),  # Use unified --timeout
        "log_requests": bool(getattr(args, "debug", False)),
        "version": getattr(args, "nametag_version", "3"),
    }
    model_name = getattr(args, "nametag_model", None) or getattr(args, "model", None)
    if model_name:
        kwargs["model"] = model_name
    language = getattr(args, "language", None)
    if language:
        kwargs["language"] = language
    # Use unified --params for REST backend parameters
    extra_params = _parse_key_value_pairs(getattr(args, "params", []))
    if extra_params:
        kwargs["extra_params"] = extra_params
    return kwargs


def _build_treetagger_backend_kwargs(args: argparse.Namespace) -> dict:
    """Gather TreeTagger-specific backend kwargs from CLI args."""
    backend = getattr(args, "backend", None)
    if not backend or backend.lower() != "treetagger":
        return {}

    extra_args = []
    for chunk in getattr(args, "treetagger_extra_args", []) or []:
        extra_args.extend(shlex.split(chunk))

    kwargs: dict[str, object] = {
        "model_name": getattr(args, "treetagger_model", None) or getattr(args, "model", None),
        "model_path": getattr(args, "treetagger_model_path", None),
        "binary": getattr(args, "treetagger_binary", None),
        "download_model": bool(getattr(args, "download_model", False)),
        "treetagger_extra_args": extra_args or None,
        "verbose": bool(getattr(args, "debug", False)),
    }
    language = getattr(args, "language", None)
    if language and not kwargs.get("model_name"):
        kwargs["language"] = language
    return {k: v for k, v in kwargs.items() if v is not None}


def _require_spacy_module():
    try:
        import spacy  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise SystemExit(
            "SpaCy backend requires the optional 'spacy' extra. Install it with: pip install \"flexipipe[spacy]\""
        ) from exc
    return spacy


def _download_spacy_model(model_name: str) -> bool:
    spacy = _require_spacy_module()
    from spacy.cli import download as spacy_download  # type: ignore
    from .model_storage import get_backend_models_dir
    
    # Download to flexipipe directory
    spacy_dir = get_backend_models_dir("spacy")
    
    # Check if this is a HuggingFace model (contains /)
    if "/" in model_name:
        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(f"[flexipipe] Error: HuggingFace model '{model_name}' requires 'huggingface_hub' package.")
            print(f"[flexipipe] Install it with: pip install huggingface_hub")
            return False
        
        model_path = spacy_dir / model_name.replace("/", "_")
        if model_path.exists():
            # Already downloaded
            print(f"[flexipipe] Model already exists at {model_path}")
            return True
        
        print(f"[flexipipe] Downloading SpaCy model from HuggingFace: '{model_name}'...")
        try:
            # Download to a temporary location first
            import tempfile
            temp_dir = tempfile.mkdtemp()
            try:
                downloaded_path = snapshot_download(model_name, cache_dir=temp_dir)
                # Copy to flexipipe directory
                import shutil
                shutil.copytree(downloaded_path, model_path, dirs_exist_ok=False)
                print(f"[flexipipe] Model downloaded to {model_path}")
                return True
            finally:
                # Clean up temp directory
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[flexipipe] Failed to download HuggingFace model '{model_name}': {exc}")
            return False
    
    # Standard SpaCy model download
    model_path = spacy_dir / model_name
    print(f"[flexipipe] Downloading spaCy model '{model_name}' to {spacy_dir}...")
    try:
        # SpaCy's download command installs to site-packages by default
        # We'll download it there first, then copy/link to our directory
        spacy_download(model_name)
        
        # Try to find the downloaded model and copy it to our directory
        try:
            import spacy.util
            downloaded_path = spacy.util.get_package_path(model_name)
            if downloaded_path and Path(downloaded_path).exists():
                import shutil
                if model_path.exists():
                    shutil.rmtree(model_path)
                # Copy the entire model directory, preserving all files including config.cfg
                shutil.copytree(downloaded_path, model_path, dirs_exist_ok=False)
                # Verify that config.cfg was copied
                if not (model_path / "config.cfg").exists():
                    print(f"[flexipipe] Warning: config.cfg not found after copying model. Model may be incomplete.")
                else:
                    print(f"[flexipipe] Copied model to {model_path}")
        except Exception as e:
            # If copying fails, that's okay - model is still available via spacy.load()
            print(f"[flexipipe] Note: Could not copy model to flexipipe directory: {e}")
            print(f"[flexipipe] Model is still available via standard SpaCy location")
    except SystemExit as exc:  # spaCy CLI may call sys.exit
        if exc.code not in (0, None):
            print(f"[flexipipe] Failed to download spaCy model '{model_name}'.")
            return False
    except Exception as exc:  # pragma: no cover - runtime feedback
        print(f"[flexipipe] Failed to download spaCy model '{model_name}': {exc}")
        return False
    return True


def _spacy_model_available(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    if Path(model_name).exists():
        return True
    
    # Check flexipipe models directory first
    try:
        from .model_storage import get_spacy_model_path
        flexipipe_model_path = get_spacy_model_path(model_name)
        if flexipipe_model_path and flexipipe_model_path.exists():
            return True
    except Exception:
        pass
    
    # Check standard spaCy location
    try:
        import importlib.util
        spec = importlib.util.find_spec(model_name)
        if spec is not None:
            return True
    except Exception:
        pass
    try:
        import spacy.util  # type: ignore
    except Exception:
        return False
    is_package = getattr(spacy.util, "is_package", None)
    if callable(is_package) and is_package(model_name):
        try:
            spacy.util.get_package_path(model_name)
            return True
        except (OSError, IOError, ImportError):
            return False
    try:
        package_path = spacy.util.get_package_path(model_name)
        return bool(package_path and Path(package_path).exists())
    except (OSError, IOError, ImportError):
        return False


def _ensure_spacy_model_available(model_name: str, *, auto_download: bool, allow_prompt: bool) -> None:
    if not model_name or Path(model_name).exists():
        return

    # Check flexipipe models directory first
    try:
        from .model_storage import get_spacy_model_path
        flexipipe_model_path = get_spacy_model_path(model_name)
        if flexipipe_model_path and flexipipe_model_path.exists():
            return
    except Exception:
        pass

    # Check standard spaCy location
    spacy = _require_spacy_module()
    is_package = getattr(spacy.util, "is_package", None)
    if callable(is_package) and is_package(model_name):
        return
    try:
        spacy.util.get_package_path(model_name)
        return
    except (OSError, IOError, ImportError):
        pass

    if auto_download:
        if _download_spacy_model(model_name):
            return
        raise SystemExit(
            f"Failed to download spaCy model '{model_name}'. Please install it manually (python -m spacy download {model_name})."
        )

    prompt_msg = f"spaCy model '{model_name}' is not installed. Download it now? [Y/n]: "
    if allow_prompt:
        answer = input(prompt_msg).strip().lower()
        if answer in ("", "y", "yes"):
            if _download_spacy_model(model_name):
                return
            raise SystemExit(
                f"Failed to download spaCy model '{model_name}'. Please install it manually."
            )
        raise SystemExit(
            f"spaCy model '{model_name}' is required but not installed. "
            "Install it manually or rerun the command with --download-model."
        )

    raise SystemExit(
        f"spaCy model '{model_name}' is required but not installed, and prompting is not possible. "
        "Re-run with --download-model to fetch it automatically."
    )


def _prepare_spacy_model_if_needed(args: argparse.Namespace) -> None:
    backend = getattr(args, "backend", None)
    if not backend or backend.lower() != "spacy":
        return
    model_name = getattr(args, "model", None)
    if not model_name or Path(model_name).exists():
        return
    _ensure_spacy_model_available(
        model_name,
        auto_download=bool(getattr(args, "download_model", False)),
        allow_prompt=sys.stdin.isatty(),
    )


def run_tag(args: argparse.Namespace) -> int:
    # Apply defaults from config if not specified
    from .model_storage import get_default_backend, get_default_output_format, get_default_create_implicit_mwt, get_default_writeback
    
    backend_type = args.backend
    if backend_type:
        setattr(args, "_backend_explicit", True)
    if not backend_type and getattr(args, "model", None):
        if _looks_like_transformers_model(args.model):
            args.backend = "transformers"
            backend_type = "transformers"
            setattr(args, "_backend_explicit", True)
    if args.backend is None:
        default_backend = get_default_backend()
        if default_backend:
            args.backend = default_backend
            backend_type = default_backend
    
    auto_selected = False
    teitok_settings_obj: Optional[TeitokSettings] = None

    try:
        requested_tasks = _parse_tasks_argument(getattr(args, "tasks", None))
    except ValueError as exc:
        print(f"[flexipipe] {exc}", file=sys.stderr)
        return 1
    mandatory_missing = TASK_MANDATORY - requested_tasks
    if mandatory_missing:
        if args.verbose or args.debug:
            missing_list = ", ".join(sorted(mandatory_missing))
            print(f"[flexipipe] Always including mandatory tasks: {missing_list}")
        requested_tasks.update(mandatory_missing)
    
    test_requested = bool(getattr(args, "test", False) or getattr(args, "dry_run", False))
    args.test = test_requested
    language_arg = getattr(args, "language", None)
    if isinstance(language_arg, str) and language_arg.strip().lower() == "auto":
        args.language = None
    output_explicit = args.output_format is not None
    writeback_explicit = args.writeback is not None
    if args.output_format is None:
        default_output_format = get_default_output_format()
        args.output_format = default_output_format if default_output_format else "teitok"
    args.output_format = _normalize_format_name(args.output_format)
    
    # Apply default create_implicit_mwt if not explicitly set
    if not args.create_implicit_mwt:
        args.create_implicit_mwt = get_default_create_implicit_mwt()
    
    # Apply default writeback if not explicitly set
    # Note: We'll auto-enable writeback when teitok-settings is detected later,
    # after we've loaded the settings and confirmed they exist
    if args.writeback is None:
        args.writeback = get_default_writeback()
    
    # Handle --example option
    example_name = getattr(args, "example", None)
    if example_name:
        if args.input not in (None, "-", "<inline-data>"):
            print("Error: --example cannot be combined with --input.", file=sys.stderr)
            return 1
        if getattr(args, "data", None):
            print("Error: --example cannot be combined with --data.", file=sys.stderr)
            return 1
        language = getattr(args, "language", None)
        if not language:
            print("Error: --example requires --language to be specified.", file=sys.stderr)
            return 1
        example_text = _load_example_text(example_name, language)
        if not example_text:
            print(f"Error: Could not load example '{example_name}' for language '{language}'.", file=sys.stderr)
            return 1
        # Set as inline data
        args.data = [example_text]
        args.input = "<inline-data>"
    
    # If --input is not provided, check if STDIN has data
    inline_data_tokens = getattr(args, "data", None)
    inline_data_text = " ".join(inline_data_tokens).strip() if inline_data_tokens else ""
    if inline_data_text:
        if args.input not in (None, "-", "<inline-data>"):
            print("Error: --data cannot be combined with --input.", file=sys.stderr)
            return 1
        args.input = args.input or "<inline-data>"
    
    note_source_path: Optional[str] = None
    if args.input not in (None, "-", "<inline-data>"):
        note_source_path = args.input
    
    if not inline_data_text and args.input is None:
        # Check if stdin is not a TTY (meaning it's piped/redirected)
        if not sys.stdin.isatty():
            args.input = "-"
        else:
            print("Error: No input specified. Provide --input <file> or pipe data to STDIN.", file=sys.stderr)
            return 1
    
    input_format = _normalize_format_name(args.input_format)
    if inline_data_text:
        input_format = "raw"
    
    # Handle stdin
    read_from_stdin = (args.input == "-") and not inline_data_text
    stdin_content = None
    
    if input_format == "auto" and not inline_data_text:
        if read_from_stdin:
            # For stdin, we need to read the content first to detect format
            # But we can't rewind stdin, so we'll read it all and use it
            stdin_content = sys.stdin.read()
            # Create a temporary approach: try to detect from content
            sample = stdin_content[:4096] if len(stdin_content) > 4096 else stdin_content
            stripped = sample.lstrip()
            if "<tok" in sample or "<TEI" in sample or stripped.startswith("<TEI"):
                input_format = "teitok"
            else:
                for line in sample.splitlines():
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    if stripped_line.startswith("#"):
                        if stripped_line.startswith("# text") or stripped_line.startswith("# sent_id") or stripped_line.startswith("# newdoc"):
                            input_format = "conllu"
                            break
                        continue
                    if "\t" in stripped_line and len(stripped_line.split("\t")) >= 10:
                        input_format = "conllu"
                        break
    if inline_data_text:
        input_format = "raw"
    elif input_format == "auto":
        if read_from_stdin:
            input_format = "raw"
        else:
            input_format = detect_input_format(args.input)
            if args.debug or args.verbose:
                print(f"[flexipipe] detected input format: {input_format}")

    if args.debug or args.verbose:
        destination = args.output if args.output else "stdout"
        if inline_data_text:
            input_source = "<inline data>"
        else:
            input_source = "stdin" if read_from_stdin else args.input
        print(f"[flexipipe] treating {input_source} -> {destination}")

    detection_attempted = False
    detection_result = None
    detection_source_text: Optional[str] = None
    validator_source_doc: Optional[Document] = None

    input_entry = None
    if not inline_data_text:
        input_entry = io_registry.get_input(input_format)

    if (not output_explicit) and (input_format == "teitok" or getattr(args, "test", False)):
        args.output_format = "teitok"

    if args.test and input_format != "teitok":
        if args.verbose or args.debug:
            print("[flexipipe] --test is currently supported for TEITOK writeback only; ignoring for this input format.", file=sys.stderr)
        args.test = False

    if inline_data_text:
        input_format = "raw"
        detection_source_text = inline_data_text
        detection_attempted = True
        detection_result = _maybe_detect_language(args, detection_source_text)
        if not auto_selected:
            backend_type = _auto_select_model_for_language(args, backend_type)
            args.backend = backend_type
            auto_selected = True
        backend_type = getattr(args, "backend", None) or "flexitag"
        segment_locally = backend_type == "flexitag" or bool(getattr(args, "pretokenize", False))
        tokenize_locally = segment_locally
        doc = Document.from_plain_text(
            inline_data_text,
            doc_id="",
            segment=segment_locally,
            tokenize=tokenize_locally,
        )
        doc.meta.setdefault("source", "inline-data")
    elif input_entry:
        stdin_payload = stdin_content if read_from_stdin else None
        doc = input_entry.load(args=args, stdin_content=stdin_payload)
        if not read_from_stdin and args.input not in (None, "-"):
            note_source_path = note_source_path or args.input
        elif read_from_stdin:
            doc.meta.setdefault("source", "stdin")
            # When piping, extract backend info from CoNLL-U headers
            # Check for backend-specific model attributes (udpipe_model, nametag_model, etc.)
            if "_file_level_attrs" in doc.meta:
                file_attrs = doc.meta["_file_level_attrs"]
                for key in file_attrs.keys():
                    if key.endswith("_model"):
                        backend_name = key.replace("_model", "").lower()
                        if backend_name in ["udpipe", "nametag", "udmorph", "treetagger"]:
                            if "_backends_used" not in doc.meta:
                                doc.meta["_backends_used"] = []
                            if backend_name not in doc.meta["_backends_used"]:
                                doc.meta["_backends_used"].append(backend_name)
            # Also check document.attrs for model info
            for key in doc.attrs.keys():
                if key.endswith("_model"):
                    backend_name = key.replace("_model", "").lower()
                    if backend_name in ["udpipe", "nametag", "udmorph", "treetagger"]:
                        if "_backends_used" not in doc.meta:
                            doc.meta["_backends_used"] = []
                        if backend_name not in doc.meta["_backends_used"]:
                            doc.meta["_backends_used"].append(backend_name)
        detection_source_text = _document_to_plain_text(doc)
    elif input_format == "raw":
        if read_from_stdin:
            if stdin_content is None:
                stdin_content = sys.stdin.read()
            raw_text = stdin_content
        else:
            with open(args.input, "r", encoding="utf-8") as handle:
                raw_text = handle.read()
        detection_source_text = raw_text
        detection_attempted = True
        detection_result = _maybe_detect_language(args, detection_source_text)
        if not auto_selected:
            backend_type = _auto_select_model_for_language(args, backend_type)
            args.backend = backend_type
            auto_selected = True
        
        # Determine which backend to use (needed to decide on segmentation)
        backend_type = getattr(args, "backend", None)
        if not backend_type:
            backend_type = "flexitag"  # Default backend
        
        # For flexitag we always segment/tokenize. For other backends, only do so when --pretokenize is requested.
        segment_locally = backend_type == "flexitag" or bool(getattr(args, "pretokenize", False))
        tokenize_locally = segment_locally

        doc = Document.from_plain_text(
            raw_text,
            doc_id="",
            segment=segment_locally,
            tokenize=tokenize_locally,
        )
        assign_doc_id_from_path(doc, args.input if not read_from_stdin else None)
        if not read_from_stdin and args.input not in (None, "-"):
            note_source_path = note_source_path or args.input
            doc.meta.setdefault("source_path", args.input)
        elif read_from_stdin:
            doc.meta.setdefault("source", "stdin")
    else:
        # TEITOK XML input
        if read_from_stdin:
            if stdin_content is None:
                stdin_content = sys.stdin.read()
            # For TEI, we need to pass the content to load_teitok
            # Check if load_teitok accepts content or only file paths
            from io import StringIO
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.xml', delete=False) as tmp:
                tmp.write(stdin_content)
                tmp_path = tmp.name
            try:
                # Check if file has tokens
                from .teitok import teitok_has_tokens, extract_teitok_plain_text
                has_tokens = teitok_has_tokens(tmp_path)
                
                if not has_tokens:
                    if not args.tokenize:
                        print(
                            f"Error: TEITOK XML file has no <tok> elements. "
                            f"Use --tokenize to enable tokenization mode.",
                            file=sys.stderr
                        )
                        return 1
                    # Extract plain text and treat as raw input
                    textnode_xpath = getattr(args, "textnode", ".//text")
                    include_notes = getattr(args, "textnotes", False)
                    raw_text = extract_teitok_plain_text(tmp_path, textnode_xpath, include_notes=include_notes)
                    if args.debug:
                        print(f"[flexipipe] Extracted raw text from TEITOK XML (xpath={textnode_xpath}, include_notes={include_notes}):")
                        print("=" * 80)
                        print(raw_text)
                        print("=" * 80)
                    # Store original input path for potential writeback
                    original_input_path = tmp_path
                    # Treat as raw text input
                    detection_source_text = raw_text
                    detection_attempted = True
                    detection_result = _maybe_detect_language(args, detection_source_text)
                    if not auto_selected:
                        backend_type = _auto_select_model_for_language(args, backend_type)
                        args.backend = backend_type
                        auto_selected = True
                    
                    # Determine which backend to use (needed to decide on segmentation)
                    backend_type = getattr(args, "backend", None)
                    if not backend_type:
                        backend_type = "flexitag"  # Default backend
                    
                    # Treat extracted text as raw input - let the backend handle segmentation/tokenization
                    # For flexitag we always segment/tokenize. For other backends, only do so when --pretokenize is requested.
                    segment_locally = backend_type == "flexitag" or bool(getattr(args, "pretokenize", False))
                    tokenize_locally = segment_locally
                    
                    doc = Document.from_plain_text(
                        raw_text,
                        doc_id="",
                        segment=segment_locally,
                        tokenize=tokenize_locally,
                    )
                    doc.meta["original_input_path"] = original_input_path
                    doc.meta.setdefault("source_path", original_input_path)
                    if read_from_stdin:
                        doc.meta.setdefault("source", "tei-stdin")
                    doc.meta["original_input_xpath"] = textnode_xpath
                    if not read_from_stdin:
                        assign_doc_id_from_path(doc, original_input_path)
                else:
                    # Load TEITOK settings and merge with CLI mappings
                    teitok_settings, merged_attrs_map = _load_and_merge_teitok_settings(args, tmp_path)
                    if teitok_settings:
                        if not teitok_settings_obj:
                            teitok_settings_obj = teitok_settings
                        if not writeback_explicit:
                            args.writeback = True
                    xpos_attr = merged_attrs_map.get("xpos")
                    reg_attr = merged_attrs_map.get("reg")
                    expan_attr = merged_attrs_map.get("expan")
                    lemma_attr = merged_attrs_map.get("lemma")
                    doc = load_teitok(
                        tmp_path,
                        xpos_attr=xpos_attr,
                        reg_attr=reg_attr,
                        expan_attr=expan_attr,
                        lemma_attr=lemma_attr,
                    )
                    # Auto-enable writeback if teitok-settings is detected and writeback not explicitly set
                    # Note: For stdin input, writeback is not supported, so we skip this
                    # Store settings for later language resolution
                    # (Language will be resolved after document is loaded to check teiHeader first)
                    detection_source_text = _document_to_plain_text(doc)
            finally:
                # Always clean up temp file (stdin doesn't support writeback anyway)
                os.unlink(tmp_path)
        else:
            # Check if file has tokens
            from .teitok import teitok_has_tokens, extract_teitok_plain_text
            from .teitok import teitok_has_tokens, extract_teitok_plain_text
            has_tokens = teitok_has_tokens(args.input)
            
            if not has_tokens:
                if not args.tokenize:
                    print(
                        f"Error: TEITOK XML file '{args.input}' has no <tok> elements. "
                        f"Use --tokenize to enable tokenization mode.",
                        file=sys.stderr
                    )
                    return 1
                # Extract plain text and treat as raw input
                textnode_xpath = getattr(args, "textnode", ".//text")
                include_notes = getattr(args, "textnotes", False)
                raw_text = extract_teitok_plain_text(args.input, textnode_xpath, include_notes=include_notes)
                if args.debug:
                    print(f"[flexipipe] Extracted raw text from TEITOK XML (xpath={textnode_xpath}, include_notes={include_notes}):")
                    print("=" * 80)
                    print(raw_text)
                    print("=" * 80)
                # Treat as raw text input
                detection_source_text = raw_text
                detection_attempted = True
                detection_result = _maybe_detect_language(args, detection_source_text)
                if not auto_selected:
                    backend_type = _auto_select_model_for_language(args, backend_type)
                    args.backend = backend_type
                    auto_selected = True
                
                # Determine which backend to use (needed to decide on segmentation)
                backend_type = getattr(args, "backend", None)
                if not backend_type:
                    backend_type = "flexitag"  # Default backend
                
                # Treat extracted text as raw input - let the backend handle segmentation/tokenization
                # For flexitag we always segment/tokenize. For other backends, only do so when --pretokenize is requested.
                segment_locally = backend_type == "flexitag" or bool(getattr(args, "pretokenize", False))
                tokenize_locally = segment_locally
                
                doc = Document.from_plain_text(
                    raw_text,
                    doc_id="",
                    segment=segment_locally,
                    tokenize=tokenize_locally,
                )
                # Store original input path in doc metadata for writeback
                doc.meta["original_input_path"] = args.input
                doc.meta["original_input_xpath"] = textnode_xpath
                doc.meta.setdefault("source_path", args.input)
                assign_doc_id_from_path(doc, args.input)
                note_source_path = note_source_path or args.input
            else:
                # Load TEITOK settings and merge with CLI mappings
                teitok_settings, merged_attrs_map = _load_and_merge_teitok_settings(args, args.input)
                if teitok_settings:
                    if not teitok_settings_obj:
                        teitok_settings_obj = teitok_settings
                    if not writeback_explicit:
                        args.writeback = True
                xpos_attr = merged_attrs_map.get("xpos")
                reg_attr = merged_attrs_map.get("reg")
                expan_attr = merged_attrs_map.get("expan")
                lemma_attr = merged_attrs_map.get("lemma")
                doc = load_teitok(
                    args.input,
                    xpos_attr=xpos_attr,
                    reg_attr=reg_attr,
                    expan_attr=expan_attr,
                    lemma_attr=lemma_attr,
                )
                # Auto-enable writeback if teitok-settings is detected and writeback not explicitly set
                # Only enable if settings.xml actually exists (we're in a TEITOK project)
                if args.writeback is None and teitok_settings and teitok_settings.settings_path and teitok_settings.settings_path.exists():
                    args.writeback = True
                    if args.verbose or args.debug:
                        print("[flexipipe] Auto-enabled writeback mode (TEITOK settings.xml found)", file=sys.stderr)
                # Store settings for later language resolution
                # (Language will be resolved after document is loaded to check teiHeader first)
        detection_source_text = _document_to_plain_text(doc)
    if args.debug:
        sent_count = len(doc.sentences)
        tok_count = sum(len(sent.tokens) for sent in doc.sentences)
        print(f"[flexipipe] loaded document: sentences={sent_count} tokens={tok_count}")

    if not detection_attempted:
        # Language resolution priority:
        # 1. CLI --language (already checked)
        # 2. Document metadata (from teiHeader)
        # 3. TEITOK settings.xml (if available)
        # 4. Language detection
        
        # Check document metadata first (from teiHeader)
        doc_lang = None
        if hasattr(doc, "attrs") and doc.attrs:
            doc_lang = doc.attrs.get("language") or doc.attrs.get("lang")
        if not doc_lang and hasattr(doc, "meta") and doc.meta:
            doc_lang = doc.meta.get("language") or doc.meta.get("lang")
        
        # If we have a language from document, use it
        if doc_lang and not getattr(args, "language", None):
            args.language = doc_lang
            if args.verbose or args.debug:
                print(f"[flexipipe] Using language from document metadata (teiHeader): {doc_lang}", file=sys.stderr)
        
        # Check TEITOK settings.xml if still no language
        if not getattr(args, "language", None) and input_format == "teitok":
            # Try to load settings if we haven't already
            if args.input and args.input not in ("-", "<inline-data>"):
                if not teitok_settings_obj:
                    teitok_settings_obj, _ = _load_and_merge_teitok_settings(args, args.input)
                teitok_settings = teitok_settings_obj
                if teitok_settings:
                    settings_lang = teitok_settings.get_language()
                    if settings_lang:
                        args.language = settings_lang
                        if args.verbose or args.debug:
                            print(f"[flexipipe] Using language from settings.xml: {settings_lang}", file=sys.stderr)
        
        # Try language detection if still no language
        if not getattr(args, "language", None):
            detection_result = _maybe_detect_language(args, detection_source_text)
            detection_attempted = True
            # Store detected language in document.meta
            if detection_result and isinstance(detection_result, dict) and detection_result.get("language"):
                doc.meta["language"] = detection_result["language"]
                args.language = detection_result.get("language") or detection_result.get("language_iso")
        else:
            detection_attempted = True
        
        # Store final language in document.meta
        if getattr(args, "language", None):
            doc.meta["language"] = args.language

    if teitok_settings_obj and getattr(args, "language", None):
        pref = teitok_settings_obj.get_flexipipe_preference(args.language)
        if pref:
            pref_backend = pref.get("backend")
            pref_model = pref.get("model")
            if pref_backend and not getattr(args, "_backend_explicit", False):
                args.backend = pref_backend
                backend_type = pref_backend
                setattr(args, "_backend_explicit", True)
                if args.verbose or args.debug:
                    print(f"[flexipipe] Using preferred backend '{pref_backend}' for language '{args.language}'", file=sys.stderr)
            if pref_model and not getattr(args, "model", None):
                args.model = pref_model
                if args.verbose or args.debug:
                    print(f"[flexipipe] Using preferred model '{pref_model}' for language '{args.language}'", file=sys.stderr)

    if args.test:
        args.writeback = True

    if not auto_selected:
        backend_type = _auto_select_model_for_language(args, backend_type)
        args.backend = backend_type
        auto_selected = True

    # Apply normalization strategy for NLP components if requested
    if getattr(args, "nlpform", "form") != "form":
        doc = apply_nlpform(doc, args.nlpform)

    # Track model information for output
    model_str = None
    
    # Check if we need a backend at all - if no tasks are requested (or only mandatory tasks),
    # we can skip backend processing and just do format conversion
    tasks_requiring_backend = requested_tasks - TASK_MANDATORY
    needs_backend = len(tasks_requiring_backend) > 0
    if args.writeback and input_format == "teitok":
        # Always ensure default tasks are requested for writeback to guarantee tagging happens
        # This ensures the document gets properly annotated even if user didn't specify --tasks
        if not needs_backend:
            requested_tasks = set(TASK_DEFAULTS)
            tasks_requiring_backend = requested_tasks - TASK_MANDATORY
            needs_backend = True
            if args.verbose or args.debug:
                tasks_str = ", ".join(sorted(requested_tasks))
                print(f"[flexipipe] writeback requested; running default tagging tasks: {tasks_str}", file=sys.stderr)
        else:
            needs_backend = True
            if args.verbose or args.debug:
                tasks_str = ", ".join(sorted(requested_tasks))
                print(f"[flexipipe] writeback requested; forcing backend execution with tasks: {tasks_str}", file=sys.stderr)
    
    # If no backend is needed, skip all backend processing
    if not needs_backend:
        # Just use the document as-is for format conversion
        from .engine import FlexitagResult
        result = FlexitagResult(document=doc, stats={})
        model_str = None  # No model used for format conversion
    else:
        neural_components = _tasks_to_backend_components(requested_tasks, backend_type)
        
        # backend_type now reflects any auto-selected backend after language detection
        
        excluded_backends: set[str] = set()
        backend_kwargs: dict[str, object] = {}

        while True:
            backend_type = getattr(args, "backend", backend_type)
            backend_kwargs = {}

            if backend_type and backend_type != "flexitag":
                backend_type_lower = backend_type.lower()
                backend_kwargs["download_model"] = bool(getattr(args, "download_model", False))
                if backend_type_lower == "stanza":
                    backend_kwargs["enable_wsd"] = bool(getattr(args, "stanza_wsd", False))
                    backend_kwargs["enable_sentiment"] = bool(getattr(args, "stanza_sentiment", False))
                    backend_kwargs["enable_coref"] = bool(getattr(args, "stanza_coref", False))
                    # --stanza-package removed: use --model lang_package format instead (e.g., --model cs_cac)
                    model_name = getattr(args, "model", None)
                    language = getattr(args, "language", None)
                    stanza_lang, stanza_pkg, stanza_model = _parse_stanza_model_spec(
                        model_name, language, None  # No longer support --stanza-package
                    )
                    if stanza_lang:
                        backend_kwargs["language"] = stanza_lang
                    if stanza_pkg:
                        backend_kwargs["package"] = stanza_pkg
                    if stanza_model:
                        backend_kwargs["model_name"] = stanza_model
                # Parse ClassLA model specification (lang-type format, e.g., mk-standard, sr-nonstandard)
                elif backend_type_lower == "classla":
                    classla_package = getattr(args, "classla_package", None)
                    classla_type = getattr(args, "classla_type", None)
                    model_name = getattr(args, "model", None)
                    language = getattr(args, "language", None)
                    classla_lang, classla_pkg, classla_model, classla_type_parsed = _parse_classla_model_spec(
                        model_name, language, classla_package, classla_type
                    )
                    if classla_lang:
                        backend_kwargs["language"] = classla_lang
                    if classla_pkg:
                        backend_kwargs["package"] = classla_pkg
                    if classla_type_parsed:
                        backend_kwargs["type"] = classla_type_parsed
                    if classla_model:
                        backend_kwargs["model_name"] = classla_model
                    backend_kwargs["download_model"] = bool(getattr(args, "download_model", False))
                elif backend_type_lower == "transformers":
                    if not getattr(args, "model", None):
                        print("[flexipipe] Transformers backend requires --model to specify a HuggingFace model.", file=sys.stderr)
                        return 1
                    backend_kwargs["task"] = getattr(args, "transformers_task", None)
                    backend_kwargs["adapter_name"] = getattr(args, "transformers_adapter", None)
                    backend_kwargs["device"] = getattr(args, "transformers_device", "cpu")
                    backend_kwargs["revision"] = getattr(args, "transformers_revision", None)
                    backend_kwargs["trust_remote_code"] = bool(getattr(args, "transformers_trust_remote_code", False))
                    context_override = _parse_transformers_context_arg(getattr(args, "transformers_context", None))
                    if context_override:
                        backend_kwargs["context_attrs"] = context_override
                    else:
                        entry = _get_transformers_model_entry(getattr(args, "model", None))
                        default_context = entry.get("context_attrs") if entry else None
                        if default_context:
                            backend_kwargs["context_attrs"] = default_context
                
                # Add UDPipe-specific kwargs (only for udpipe REST backend, not udpipe1)
                if backend_type_lower == "udpipe":
                    backend_kwargs.update(_build_udpipe_backend_kwargs(args))
                # Add UDMorph-specific kwargs
                if backend_type_lower == "udmorph":
                    backend_kwargs.update(_build_udmorph_backend_kwargs(args))
                # Add NameTag-specific kwargs
                if backend_type_lower == "nametag":
                    backend_kwargs.update(_build_nametag_backend_kwargs(args))
                # Add TreeTagger-specific kwargs
                if backend_type_lower == "treetagger":
                    backend_kwargs.update(_build_treetagger_backend_kwargs(args))

            try:
                _prepare_spacy_model_if_needed(args)
                break
            except SystemExit as exc:
                backend_type_lower = backend_type.lower() if backend_type else None
                backend_locked = bool(getattr(args, "_backend_explicit", False))
                if (
                    backend_type_lower == "spacy"
                    and not backend_locked
                ):
                    missing_model = getattr(args, "model", None)
                    args.model = None
                    args.backend = None
                    excluded_backends.add("spacy")
                    fallback_backend = _auto_select_model_for_language(
                        args,
                        None,
                        exclude_backends=excluded_backends,
                    )
                    if fallback_backend and fallback_backend.lower() not in excluded_backends:
                        msg = exc.code if isinstance(exc.code, str) else str(exc)
                        if msg:
                            print(f"[flexipipe] {msg}")
                        print(
                            f"[flexipipe] Falling back to backend '{fallback_backend}' "
                            f"because spaCy model '{missing_model}' is not installed."
                        )
                        backend_type = fallback_backend
                        args.backend = fallback_backend
                        continue
                raise

        if backend_type and backend_type != "flexitag":
            # Check if we need flexitag fallback (use_neural_primary means neural is primary, flexitag is fallback)
            needs_flexitag_fallback = getattr(args, 'use_neural_primary', False)
            if not needs_flexitag_fallback:
                # Neural backend only, no flexitag fallback
                from .backend_registry import create_backend
                
                # Resolve flexitag model path if needed for fallback (from --model, not --params)
                flexitag_model_path = None
                
                # Extract Stanza-specific args if present
                stanza_lang = backend_kwargs.pop("language", None)
                stanza_pkg = backend_kwargs.pop("package", None)
                stanza_model = backend_kwargs.pop("model_name", None)
                
                # Build create_backend arguments
                create_kwargs = dict(backend_kwargs)
                model_name = getattr(args, "model", None)
                language = getattr(args, "language", None)
                
                if backend_type_lower == "stanza":
                    create_kwargs["package"] = stanza_pkg
                    create_kwargs["language"] = stanza_lang
                    create_kwargs["model_name"] = stanza_model
                elif backend_type_lower == "udpipe1":
                    # For udpipe1, pass model directly (not model_name or model_path)
                    if model_name:
                        create_kwargs["model"] = model_name
                    create_kwargs.pop("model_name", None)
                    create_kwargs.pop("model_path", None)
                else:
                    create_kwargs["model_name"] = model_name
                    if backend_type_lower == "flair":
                        create_kwargs["language"] = language or "en"
                    else:
                        create_kwargs["language"] = language if not model_name else None
                
                try:
                    # Only pass verbose to backends that support it (stanza, flair, udpipe1)
                    backend_kwargs_final = dict(create_kwargs)
                    if backend_type.lower() in ("stanza", "classla", "flair", "udpipe1", "spacy"):
                        backend_kwargs_final["verbose"] = args.verbose or args.debug
                    
                    # For udpipe1, don't use model_path - pass model directly
                    model_path_arg = None
                    if backend_type_lower != "udpipe1":
                        model_path_arg = model_name if model_name and Path(model_name).exists() else None
                    
                    neural_backend = create_backend(
                        backend_type,
                        training=False,
                        model_path=model_path_arg,
                        **backend_kwargs_final,
                    )
                    if backend_type_lower == "spacy" and not getattr(args, "model", None):
                        auto_model = getattr(neural_backend, "auto_model", None)
                        if auto_model:
                            setattr(args, "model", auto_model)
                            model_name = auto_model
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    print(f"[flexipipe] {e}", file=sys.stderr)
                    return 1
                # For raw text input, use raw text mode to let backend do its own tokenization
                # For other formats (conllu, teitok), use tokenized mode to preserve existing tokenization
                # Exception: if text was extracted from TEITOK XML with --tokenize, treat as raw text
                # Or if --use-raw-text is explicitly set (CLI flag), force raw text mode
                # Or if use_raw_text is set in TEITOK settings.xml, use that (unless CLI flag overrides)
                is_extracted_text = doc.meta.get("original_input_path") is not None
                use_raw_text = (
                    getattr(args, "use_raw_text", False) or  # CLI flag (highest priority)
                    (teitok_settings_obj and teitok_settings_obj.use_raw_text) or  # Settings.xml
                    (input_format == "raw" and not getattr(args, "pretokenize", False)) or
                    (is_extracted_text and not getattr(args, "pretokenize", False))
                )
                if backend_type_lower == "udmorph":
                    use_raw_text = True
                try:
                    if args.verbose or args.debug:
                        components_str = ", ".join(sorted(neural_components)) if neural_components else "none"
                        print(f"[flexipipe] Running backend '{backend_type}' with components: {components_str}", file=sys.stderr)
                    neural_result = neural_backend.tag(
                        doc,
                        use_raw_text=use_raw_text,
                        components=neural_components,
                    )
                    if args.verbose or args.debug:
                        sent_count = len(neural_result.document.sentences)
                        tok_count = sum(len(sent.tokens) for sent in neural_result.document.sentences)
                        print(f"[flexipipe] Backend completed: {sent_count} sentences, {tok_count} tokens", file=sys.stderr)
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    print(f"[flexipipe] {e}", file=sys.stderr)
                    return 1
                from .engine import FlexitagResult
                result = FlexitagResult(document=neural_result.document, stats=neural_result.stats)
                _propagate_sentence_metadata(result.document, doc)
                _propagate_token_ids(result.document, doc)
                
                # Track backend used
                if "_backends_used" not in result.document.meta:
                    result.document.meta["_backends_used"] = []
                result.document.meta["_backends_used"].append(backend_type_lower)
                
                # Determine model string for backend
                display_backend = backend_type_lower.upper()
                backend_descriptor = getattr(neural_backend, "model_descriptor", None)
                if backend_type_lower == "spacy":
                    if model_name and Path(model_name).exists():
                        model_display = Path(model_name).name
                    elif model_name:
                        model_display = model_name
                    elif language:
                        model_display = f"{language} (blank)"
                    else:
                        model_display = backend_descriptor or "unknown"
                elif backend_type_lower == "nametag":
                    # For NameTag, if a specific model/language is set, show it; otherwise just show the version
                    backend_model = getattr(neural_backend, "model", None)
                    backend_language = getattr(neural_backend, "language", None)
                    if backend_model:
                        model_str = f"{display_backend}: {backend_model}"
                    elif backend_language:
                        model_str = f"{display_backend}: {backend_language}"
                    else:
                        # No specific model - just show the version (e.g., "NameTag3")
                        model_str = backend_descriptor or f"NameTag{getattr(neural_backend, 'version', '3')}"
                else:
                    model_display = model_name or backend_descriptor or "unknown"
                    model_str = f"{display_backend}: {model_display}"
            else:
                # Use pipeline with non-flexitag backend
                from .pipeline import FlexiPipeline, PipelineConfig
                from .backend_registry import create_backend
                
                model_name = getattr(args, "model", None)
                language = getattr(args, "language", None)
                
                try:
                    # Only pass verbose to backends that support it (stanza, flair)
                    create_kwargs = dict(backend_kwargs)
                    if backend_type.lower() in ("stanza", "flair"):
                        create_kwargs["verbose"] = args.verbose or args.debug
                    
                    neural_backend = create_backend(
                        backend_type,
                        training=False,
                        model_name=model_name,
                        model_path=model_name if model_name and Path(model_name).exists() else None,
                        language=language if not model_name else None,
                        **create_kwargs,
                    )
                    if backend_type_lower == "spacy" and not getattr(args, "model", None):
                        auto_model = getattr(neural_backend, "auto_model", None)
                        if auto_model:
                            setattr(args, "model", auto_model)
                            model_name = auto_model
                except (ValueError, RuntimeError) as e:
                    # Print clean error message for missing models
                    print(f"[flexipipe] {e}", file=sys.stderr)
                    return 1
                
                # use_neural_primary removed - neural backend only
                # This code path should not be reached anymore, but kept for safety
                raise RuntimeError("Hybrid neural/flexitag mode no longer supported. Use neural backend only.")
                pipeline = FlexiPipeline(config)
                result = pipeline.process(doc)
                _propagate_sentence_metadata(result.document, doc)
                _propagate_token_ids(result.document, doc)
                
                # Determine model string for hybrid pipeline
                backend_type_lower = backend_type.lower()
                display_backend = backend_type_lower.upper()
                backend_descriptor = getattr(neural_backend, "model_descriptor", None)
                if backend_type_lower == "spacy":
                    if model_name and Path(model_name).exists():
                        neural_model_name = Path(model_name).name
                    elif model_name:
                        neural_model_name = model_name
                    elif language:
                        neural_model_name = f"{language} (blank)"
                    else:
                        neural_model_name = backend_descriptor or "unknown"
                else:
                    neural_model_name = model_name or backend_descriptor or "unknown"
                flexitag_model_name = Path(flexitag_model_path).name if flexitag_model_path else None
                if flexitag_model_name:
                    model_str = f"{display_backend}: {neural_model_name}, FLEXITAG: {flexitag_model_name}"
                else:
                    model_str = f"{display_backend}: {neural_model_name}"
        elif needs_backend and backend_type == "flexitag":
            # Use flexitag only
            from .backend_registry import create_backend
            
            flexitag_options = build_flexitag_options_from_args(args)
            
            try:
                flexitag_backend = create_backend(
                    "flexitag",
                    training=False,
                    model_name=getattr(args, "model", None),
                    language=getattr(args, "language", None),
                    params_path=None,  # --params removed, use --model instead
                    options=flexitag_options,
                    debug=args.debug,
                )
            except (ValueError, RuntimeError) as e:
                print(f"[flexipipe] {e}", file=sys.stderr)
                return 1
            
            neural_result = flexitag_backend.tag(doc)
            result = neural_result  # FlexitagBackend returns NeuralResult
            _propagate_sentence_metadata(result.document, doc)
            _propagate_token_ids(result.document, doc)
            
            # Model string for flexitag only
            if args.model:
                model_str = f"FLEXITAG: {args.model}"
            else:
                model_str = "FLEXITAG: unknown"
        elif needs_backend and not backend_type:
            # Backend needed but not specified
            print("Error: Backend required for requested tasks but none specified. Use --backend to specify a backend.", file=sys.stderr)
            return 1

    validator_source_doc = result.document

    output_format = args.output_format
    output_path = args.output

    # Ensure "ner" is in tasks if entities exist (preserve NER from backends)
    has_entities = False
    for sent in result.document.sentences:
        if sent.entities:
            has_entities = True
            break
    if not has_entities and getattr(result.document, "spans", None):
        if result.document.spans.get("ner"):
            has_entities = True
    if has_entities and "ner" not in requested_tasks:
        requested_tasks.add("ner")

    _filter_document_by_tasks(result.document, requested_tasks)

    # If validation is requested but no output file is specified, use a temp file
    validation_temp_file = None
    if getattr(args, "validate", False) and not output_path and output_format in ("conllu", "conllu-ne"):
        import tempfile
        validation_temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".conllu",
            delete=False,
            encoding="utf-8",
        )
        output_path = validation_temp_file.name
        validation_temp_file.close()

    output_entry = io_registry.get_output(output_format)
    if output_entry:
        # If we're using a temp file for validation, save to it but don't print yet
        actual_output_path = output_path if not validation_temp_file else output_path
        output_entry.save(
            result.document,
            args=args,
            output_path=actual_output_path,
            model_info=model_str,
        )
    elif output_format == "teitok":
        # Apply create_implicit_mwt if requested (for TEITOK output with <dtok> elements)
        output_doc = result.document
        if args.create_implicit_mwt:
            from .conllu import _create_implicit_mwt
            # Create a new document with MWTs created
            new_doc = Document(id=output_doc.id, meta=dict(output_doc.meta), attrs=dict(output_doc.attrs))
            # Copy spans from original document
            for layer, spans in output_doc.spans.items():
                for span in spans:
                    new_doc.add_span(layer, span)
            for sent in output_doc.sentences:
                new_sent = _create_implicit_mwt(sent)
                new_doc.sentences.append(new_sent)
            output_doc = new_doc
        validator_source_doc = output_doc
        
        # Apply tag mapping if requested
        map_tags_models = getattr(args, "map_tags_models", None)
        if map_tags_models:
            model_paths = [Path(p) for p in map_tags_models]
            mapping = build_tag_mapping_from_paths(model_paths)
            
            direction = getattr(args, "map_direction", None)
            fill_xpos = getattr(args, "fill_xpos", None)
            fill_upos = getattr(args, "fill_upos", None)
            fill_feats = getattr(args, "fill_feats", None)
            allow_partial = getattr(args, "allow_partial", True)
            
            # Apply direction defaults unless user explicitly overrode via --fill-* options
            if direction == "xpos":
                if fill_xpos is None:
                    fill_xpos = True
                if fill_upos is None:
                    fill_upos = False
                if fill_feats is None:
                    fill_feats = False
            elif direction == "upos-feats":
                if fill_xpos is None:
                    fill_xpos = False
                if fill_upos is None:
                    fill_upos = True
                if fill_feats is None:
                    fill_feats = True
            elif direction == "both":
                if fill_xpos is None:
                    fill_xpos = True
                if fill_upos is None:
                    fill_upos = True
                if fill_feats is None:
                    fill_feats = True
            
            # If no direction specified and no explicit fill options, default to both
            if direction is None and fill_xpos is None and fill_upos is None and fill_feats is None:
                fill_xpos = True
                fill_upos = True
                fill_feats = True
            
            fill_xpos = bool(fill_xpos) if fill_xpos is not None else False
            fill_upos = bool(fill_upos) if fill_upos is not None else False
            fill_feats = bool(fill_feats) if fill_feats is not None else False
            
            if fill_xpos or fill_upos or fill_feats:
                changes = mapping.enrich_document(
                    output_doc,
                    fill_xpos=fill_xpos,
                    fill_upos=fill_upos,
                    fill_feats=fill_feats,
                    allow_partial=allow_partial,
                )
                if args.verbose or args.debug:
                    direction_desc = []
                    if fill_xpos:
                        direction_desc.append("XPOS")
                    if fill_upos or fill_feats:
                        parts = ["UPOS" if fill_upos else None, "FEATS" if fill_feats else None]
                        direction_desc.append("+".join(p for p in parts if p))
                    direction_text = ", ".join(direction_desc) or "nothing"
                    print(f"[flexipipe] tag mapping ({direction_text}) updated {changes} tokens")
        
        validator_source_doc = output_doc
        
        # Check if writeback should be used
        use_writeback = False
        original_input_path = None
        is_extracted_text = False
        if args.writeback and input_format == "teitok":
            # Writeback only works when input is TEITOK XML (not stdin)
            if not read_from_stdin and args.input:
                original_input_path = Path(args.input)
                if original_input_path.exists():
                    # Use writeback if output is same as input, or no output specified
                    if not output_path or str(original_input_path.resolve()) == str(Path(output_path).resolve()):
                        use_writeback = True
                        # Check if this came from extracted text (non-tokenized TEITOK with --tokenize)
                        from .teitok import teitok_has_tokens
                        if output_doc.meta.get("original_input_path") and not teitok_has_tokens(output_doc.meta["original_input_path"]):
                            is_extracted_text = True
        
        if use_writeback and original_input_path:
            target_writeback_path = original_input_path
            temp_writeback_path: Optional[Path] = None
            if args.test:
                import tempfile
                import shutil
                suffix = original_input_path.suffix or ".xml"
                fd, temp_name = tempfile.mkstemp(prefix="flexipipe-test-", suffix=suffix)
                os.close(fd)
                shutil.copy2(original_input_path, temp_name)
                target_writeback_path = Path(temp_name)
                temp_writeback_path = target_writeback_path
            # Create backup before writeback (TEITOK-style once-a-day backup)
            # Only create backup if we're in a TEITOK project (settings.xml exists)
            from .teitok_backup import create_teitok_backup
            from .teitok_settings import find_settings_xml
            
            backup_path = None
            settings_path = find_settings_xml(original_input_path)
            if args.test:
                if args.verbose or args.debug:
                    print("[flexipipe] Skipping backup (--test dry-run)", file=sys.stderr)
            elif settings_path and settings_path.exists():
                backup_path = create_teitok_backup(str(original_input_path))
                if backup_path:
                    if args.verbose or args.debug:
                        print(f"[flexipipe] Created backup: {backup_path}", file=sys.stderr)
                elif args.verbose or args.debug:
                    from datetime import date
                    today = date.today().strftime("%Y%m%d")
                    filename = Path(original_input_path).name
                    if "." in filename:
                        name_part, ext_part = filename.rsplit(".", 1)
                        backup_name = f"{name_part}-{today}.{ext_part}"
                    else:
                        backup_name = f"{filename}-{today}"
                    backups_dir = Path.cwd() / "backups"
                    existing_backup = backups_dir / backup_name
                    if existing_backup.exists():
                        print(f"[flexipipe] Backup already exists for today: {existing_backup}", file=sys.stderr)
            elif args.verbose or args.debug:
                print("[flexipipe] Skipping backup (not in a TEITOK project - settings.xml not found)", file=sys.stderr)
            
            from .teitok import update_teitok
            from .insert_tokens import insert_tokens_into_teitok
            try:
                if is_extracted_text:
                    textnode_xpath = output_doc.meta.get("original_input_xpath", ".//text")
                    include_notes = getattr(args, "textnotes", False)
                    insert_tokens_into_teitok(
                        output_doc,
                        str(target_writeback_path),
                        output_path if output_path else None,
                        textnode_xpath=textnode_xpath,
                        include_notes=include_notes,
                    )
                    if args.verbose or args.debug:
                        target = str(target_writeback_path) if not output_path else output_path
                        print(f"[flexipipe] Inserted tokens into TEITOK file: {target}")
                else:
                    update_teitok(
                        output_doc,
                        str(target_writeback_path),
                        output_path if output_path else None,
                        settings=teitok_settings_obj,
                    )
                    if args.verbose or args.debug:
                        target = str(target_writeback_path) if not output_path else output_path
                        print(f"[flexipipe] Updated TEITOK file in-place: {target}")
            except Exception as e:
                print(
                    f"[flexipipe] Writeback failed for {target_writeback_path}: {e}, falling back to regular save",
                    file=sys.stderr,
                )
                if args.test and temp_writeback_path:
                    temp_writeback_path.unlink(missing_ok=True)
            else:
                if args.test and temp_writeback_path:
                    with open(temp_writeback_path, "r", encoding="utf-8") as tf:
                        sys.stdout.write(tf.read())
                    temp_writeback_path.unlink(missing_ok=True)
                    return 0
                writeback_rendered = True
        emit_teitok_output = True
        if use_writeback and not args.test and not output_path:
            emit_teitok_output = args.verbose or args.debug
        if emit_teitok_output or output_path or args.test or not use_writeback:
            tei_tasks = _detect_performed_tasks(output_doc)
            tasks_summary_str = ",".join(sorted(tei_tasks)) if tei_tasks else "segment,tokenize"
            pretty_print = getattr(args, "pretty_print", False)
            spaceafter_handling = getattr(args, "spaceafter_handling", "preserve")
            skip_spaceafter_for_breaking = getattr(args, "skip_spaceafter_for_breaking_elements", True)
            tei_output = dump_teitok(
                output_doc,
                pretty_print=pretty_print,
                spaceafter_handling=spaceafter_handling,
                skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking,
                settings=teitok_settings_obj,
            )
            note_candidate = (
                note_source_path
                or output_doc.meta.get("original_input_path")
                or output_doc.meta.get("source_path")
            )
            note_value = None
            if note_candidate and str(note_candidate) not in ("-", "<inline-data>"):
                note_value = f"{Path(str(note_candidate)).stem}.xml"
            change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Collect all backends used (from piping or single backend)
            backends_used = output_doc.meta.get("_backends_used", [])
            if backend_type and backend_type.lower() not in backends_used:
                backends_used.append(backend_type.lower())
            if not backends_used:
                backends_used = ["flexipipe"]
            
            # Build backend string - show all backends
            backend_names = [b.upper() for b in backends_used]
            change_source = ", ".join(backend_names) if len(backend_names) > 1 else (model_str or backend_names[0] if backend_names else "flexipipe")
            change_text = f"Tagged via {change_source} (tasks={tasks_summary_str})"
            
            # Collect model/license info and acknowledgements for notes
            api_notes = []
            
            # Extract model and license info from file-level attributes
            file_level_attrs = output_doc.meta.get("_file_level_attrs", {})
            model_info_lines = []
            
            # Collect all model and license pairs
            model_keys = sorted([k for k in file_level_attrs.keys() if k.endswith("_model")])
            for model_key in model_keys:
                model_name = file_level_attrs[model_key]
                model_info_lines.append(f"# {model_key} = {model_name}")
                
                # Add corresponding license if present
                licence_key = f"{model_key}_licence"
                if licence_key in file_level_attrs:
                    model_info_lines.append(f"# {licence_key} = {file_level_attrs[licence_key]}")
            
            if model_info_lines:
                # Add acknowledgements if present
                if "_api_acknowledgements" in output_doc.meta:
                    acknowledgements = output_doc.meta["_api_acknowledgements"]
                    if isinstance(acknowledgements, list):
                        model_info_lines.append("")
                        model_info_lines.append("# acknowledgements:")
                        for ack in acknowledgements:
                            model_info_lines.append(f"#   {ack}")
                    elif isinstance(acknowledgements, str):
                        model_info_lines.append("")
                        model_info_lines.append(f"# acknowledgements: {acknowledgements}")
                
                api_notes.append(("Model Information", "\n".join(model_info_lines)))
            
            tei_output = _augment_tei_output(
                tei_output,
                note_value=note_value,
                change_text=change_text,
                change_when=change_when,
                api_notes=api_notes if api_notes else None,
                pretty_print=pretty_print,
                spaceafter_handling=spaceafter_handling,
                skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking,
            )
            if output_path:
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write(tei_output)
            elif emit_teitok_output:
                print(tei_output)
    elif output_format == "json":
        performed_tasks = sorted(_detect_performed_tasks(result.document)) or sorted(requested_tasks)
        payload = {
            "document": document_to_json_payload(result.document),
            "model": model_str,
            "backend": backend_type,
            "tasks": performed_tasks,
            "stats": result.stats,
        }
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if output_path:
            Path(output_path).write_text(
                json_text if json_text.endswith("\n") else f"{json_text}\n",
                encoding="utf-8",
            )
        else:
            print(json_text)

    if args.debug:
        print(f"[flexipipe] completed. stats={result.stats}")
        sent_count = len(result.document.sentences)
        tok_count = sum(len(sent.tokens) for sent in result.document.sentences)
        print(f"[flexipipe] wrote document: sentences={sent_count} tokens={tok_count}")
    elif args.verbose:
        print(f"[flexipipe] saved to {output_path or 'stdout'}")

    if getattr(args, "validate", False):
        if output_format not in ("conllu", "conllu-ne"):
            print(
                "[flexipipe] --validate currently supports only conllu/conllu-ne output. "
                "Use --output-format conllu (or conllu-ne) when enabling validation.",
                file=sys.stderr,
            )
            return 1
        # output_path should be set now (either from user or from temp file)
        if not output_path:
            print(
                "[flexipipe] --validate requires --output <file> so the UD validator can read the saved CoNLL-U.",
                file=sys.stderr,
            )
            return 1
        
        validator_lang = getattr(args, "validator_lang", None)
        if not validator_lang:
            validator_lang = infer_language_from_document(
                validator_source_doc,
                detection_result,
                getattr(args, "language", None),
            )
        # Smart validation is enabled by default, unless user explicitly sets rules or disables it
        smart_validation = not getattr(args, "no_smart_validation", False) and not getattr(args, "validator_rules", None)
        
        # If we used a temp file, print the output to stdout first, then validate
        if validation_temp_file:
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    print(f.read(), end="")
            except OSError:
                pass
        
        validation_rc = run_validator_cli(
            Path(output_path),
            version=getattr(args, "ud_tools_version", "latest"),
            url=getattr(args, "ud_tools_url", None),
            rules=getattr(args, "validator_rules", None),
            extra_args=getattr(args, "validator_args", None),
            lang=validator_lang,
            verbose=args.verbose or args.debug,
            smart_validation=smart_validation,
        )
        
        # Clean up temp file after validation
        if validation_temp_file:
            try:
                Path(output_path).unlink()
            except OSError:
                pass
        
        if validation_rc != 0:
            return validation_rc

    return 0


def run_validate(args: argparse.Namespace) -> int:
    paths = [Path(p).expanduser() for p in args.treebank]
    verbose = args.verbose or args.debug
    overall_rc = 0
    explicit_lang = getattr(args, "validator_lang", None)
    for path in paths:
        if not path.exists():
            print(f"[flexipipe] Cannot validate '{path}': file or directory does not exist.", file=sys.stderr)
            overall_rc = overall_rc or 1
            continue
        if verbose:
            print(f"[flexipipe] Validating {path}")
        lang_for_this = explicit_lang or infer_language_from_path(path)
        # Smart validation is enabled by default, unless user explicitly sets rules or disables it
        smart_validation = not getattr(args, "no_smart_validation", False) and not getattr(args, "validator_rules", None)
        rc = run_validator_cli(
            path,
            version=getattr(args, "ud_tools_version", "latest"),
            url=getattr(args, "ud_tools_url", None),
            rules=getattr(args, "validator_rules", None),
            extra_args=getattr(args, "validator_args", None),
            lang=lang_for_this,
            verbose=verbose,
            smart_validation=smart_validation,
        )
        if rc != 0:
            overall_rc = rc
    return overall_rc
def _required_annotations_for_backend(
    backend_type: str,
    xpos_attr: Optional[str] = None,
) -> List[str]:
    backend_key = backend_type.lower()
    if backend_key == "flexitag":
        required = ["lemma"]
        if xpos_attr:
            required.insert(0, "xpos")
        return required
    if backend_key in ("spacy", "transformers"):
        required = ["xpos", "upos", "lemma", "head", "deprel"]
    else:
        required = ["xpos", "upos", "lemma"]
    if xpos_attr and "xpos" not in required:
        required.append("xpos")
    return required


def run_train(args: argparse.Namespace) -> int:
    """Run training command with backend-specific logic."""
    cleanup_paths: List[Path] = []
    nlpform_mode = getattr(args, "nlpform", "form") or "form"

    def _maybe_prepare_nlpform_path(path: Optional[Path]) -> Optional[Path]:
        if nlpform_mode == "form" or path is None:
            return path
        if not isinstance(path, Path) or path.is_dir():
            return path
        normalized = prepare_conllu_with_nlpform(path, nlpform_mode)
        if normalized != path:
            cleanup_paths.append(normalized)
            return normalized
        return path

    def _cleanup_nlpform_paths() -> None:
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    backend_type = getattr(args, "backend", "flexitag")
    # Parse attrs-map into individual attributes
    attrs_map = _parse_attrs_map(getattr(args, "attrs_map", None))
    xpos_attr = attrs_map.get("xpos")
    reg_attr = attrs_map.get("reg")
    expan_attr = attrs_map.get("expan")
    lemma_attr = attrs_map.get("lemma")
    
    if backend_type == "flexitag":
        from .model_storage import get_backend_models_dir
        # Flexitag training (original implementation)
        if not args.train_data:
            raise SystemExit("--train-data is required for flexitag backend")
        
        train_data_path = Path(args.train_data)
        if train_data_path.is_file():
            raise SystemExit("--train-data must be a directory for flexitag backend (containing *-ud-train.conllu, etc.)")
        if not train_data_path.is_dir():
            raise SystemExit(f"--train-data path does not exist: {train_data_path}")
        
        language_code = getattr(args, "language", None)
        ud_folder = getattr(args, "ud_folder", None)
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            safe_label = args.name or train_data_path.name or f"flexitag-model-{datetime.now():%Y%m%d%H%M%S}"
            output_dir = (get_backend_models_dir("flexitag") / safe_label).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
    model_path = train_ud_treebank(
            ud_root=train_data_path,
            output_dir=output_dir,
        model_name=args.name,
        include_dev=args.include_dev,
        verbose=args.verbose or args.debug,
        finetune=args.finetune,
        tag_attribute=args.tagpos,
            language_code=language_code,
            ud_folder=ud_folder,
            nlpform=nlpform_mode,
            xpos_attr=xpos_attr,
            reg_attr=reg_attr,
            expan_attr=expan_attr,
            lemma_attr=lemma_attr,
    )
    if not (args.verbose or args.debug):
        print(f"[flexipipe] created model at {model_path}")
        try:
            with model_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle).get("metadata", {})
            tag_selection = metadata.get("tag_selection", {})
            chosen_attr = tag_selection.get("chosen")
            if chosen_attr:
                note = "auto-selected" if tag_selection.get("auto") else "user-selected"
                print(f"[flexipipe] tag attribute: {chosen_attr} ({note})")
        except Exception:
            pass
        _cleanup_nlpform_paths()
    return 0
    
    if backend_type in ("spacy", "transformers", "udpipe1"):
        # Neural backend training
        from .backend_registry import create_backend
        from .train import _prepare_teitok_corpus, _find_ud_splits
        
        if not args.train_data:
            raise SystemExit("--train-data is required for neural backends")
        train_path = Path(args.train_data)
        dev_path = Path(args.dev_data) if args.dev_data else None
        
        # Check if this is a TEITOK corpus directory (no pre-split files)
        splits = _find_ud_splits(train_path)
        teitok_temp_dir = None
        if not splits:
            # This is a TEITOK corpus - prepare it
            ud_folder = getattr(args, "ud_folder", None)
            if ud_folder:
                teitok_temp_path = Path(ud_folder).expanduser().resolve()
                teitok_temp_path.mkdir(parents=True, exist_ok=True)
                teitok_temp_dir = str(teitok_temp_path)
            else:
                import tempfile
                teitok_temp_dir = tempfile.mkdtemp(prefix="flexipipe-teitok-")
                teitok_temp_path = Path(teitok_temp_dir)
            
            required_annotations = _required_annotations_for_backend(backend_type, xpos_attr)
            
            try:
                prepared_splits = _prepare_teitok_corpus(
                    teitok_dir=train_path,
                    output_dir=teitok_temp_path,
                    required_annotations=required_annotations,
                    backend_type=backend_type,
                    train_ratio=0.8,
                    dev_ratio=0.1,
                    test_ratio=0.1,
                    seed=42,
                    verbose=args.verbose or args.debug,
                    xpos_attr=xpos_attr,
                    reg_attr=reg_attr,
                    expan_attr=expan_attr,
                    lemma_attr=lemma_attr,
                )
                # Update paths to point to prepared CoNLL-U files
                train_path = prepared_splits["train"]
                if "dev" in prepared_splits:
                    dev_path = prepared_splits["dev"]
                elif not dev_path:
                    # Use dev from prepared splits if available
                    pass
                
                # If ud_folder is provided, print where the files are kept
                if ud_folder:
                    print(f"[flexipipe] CoNLL-U files saved to: {teitok_temp_dir}")
                    if "test" in prepared_splits:
                        print(f"[flexipipe] Test file available at: {prepared_splits['test']}")
                    
                    # Print token distribution summary
                    split_counts = prepared_splits.get("_token_counts", {})
                    if split_counts:
                        total_tokens = sum(split_counts.values())
                        if total_tokens > 0:
                            parts = []
                            for split_name in ["train", "dev", "test"]:
                                if split_name in split_counts:
                                    count = split_counts[split_name]
                                    pct = (count / total_tokens) * 100
                                    parts.append(f"{split_name} = {count:,} tokens ({pct:.1f}%)")
                            print(f"[flexipipe] Created gold standard distribution: {', '.join(parts)}")
                    # Remove the token counts from the result dict
                    prepared_splits.pop("_token_counts", None)
            except Exception as e:
                import shutil
                # Only clean up if it's a temporary directory (not ud_folder)
                if teitok_temp_dir and not ud_folder:
                    shutil.rmtree(teitok_temp_dir, ignore_errors=True)
                raise SystemExit(f"Failed to prepare TEITOK corpus: {e}")
        
        output_dir_arg = Path(args.output_dir).expanduser() if args.output_dir else None
        from .model_storage import get_backend_models_dir
        if backend_type == "spacy":
            storage_root = get_backend_models_dir("spacy")
            label = args.name or args.model or args.language or "spacy-model"
            safe_label = label.replace("/", "_") or f"spacy-model-{datetime.now():%Y%m%d%H%M%S}"
            if output_dir_arg is None:
                output_dir = (storage_root / safe_label).resolve()
            else:
                resolved = output_dir_arg.resolve()
                try:
                    same_as_root = resolved.exists() and resolved.samefile(storage_root)
                except FileNotFoundError:
                    same_as_root = False
                if not output_dir_arg.is_absolute():
                    output_dir = (storage_root / output_dir_arg).resolve()
                elif same_as_root:
                    output_dir = (storage_root / safe_label).resolve()
                else:
                    output_dir = resolved
        elif backend_type == "transformers":
            storage_root = get_backend_models_dir("transformers")
            label = args.name or args.model or train_path.stem or "transformers-model"
            safe_label = label.replace("/", "_") or f"transformers-model-{datetime.now():%Y%m%d%H%M%S}"
            if output_dir_arg is None:
                output_dir = (storage_root / safe_label).resolve()
            else:
                output_dir = output_dir_arg.resolve()
        elif backend_type == "udpipe1":
            storage_root = get_backend_models_dir("udpipe1").resolve()
            output_dir = output_dir_arg.resolve() if output_dir_arg else storage_root
        else:
            output_dir = output_dir_arg.resolve() if output_dir_arg else Path(args.output_dir or ".").expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backend_kwargs = {}
        if backend_type == "spacy":
            if args.model:
                backend_kwargs["model_name"] = args.model
            elif args.language:
                backend_kwargs["language"] = args.language
            else:
                raise SystemExit("--model or --language is required for SpaCy backend")
        elif backend_type == "transformers":
            if not args.model:
                raise SystemExit("--model is required for Transformers backend")
            backend_kwargs["model_name"] = args.model
        elif backend_type == "udpipe1":
            model_target = args.model
            if not model_target:
                label = args.name or train_path.stem or "udpipe-model"
                model_target = str((output_dir / f"{label}.udpipe").resolve())
            backend_kwargs["model"] = model_target
            backend_kwargs["verbose"] = args.verbose or args.debug
            if args.udpipe1_tokenizer:
                backend_kwargs["tokenizer_options"] = args.udpipe1_tokenizer
            if args.udpipe1_tagger:
                backend_kwargs["tagger_options"] = args.udpipe1_tagger
            if args.udpipe1_parser:
                backend_kwargs["parser_options"] = args.udpipe1_parser
        
        train_path = _maybe_prepare_nlpform_path(train_path)
        dev_path = _maybe_prepare_nlpform_path(dev_path) if dev_path else None
        
        try:
            backend = create_backend(backend_type, training=True, **backend_kwargs)
        except Exception as e:
            _cleanup_nlpform_paths()
            raise SystemExit(f"Failed to create {backend_type} backend: {e}")
        
        if not backend.supports_training:
            _cleanup_nlpform_paths()
            raise SystemExit(f"{backend_type} backend does not support training")
        
        try:
            model_path = backend.train(
                train_data=train_path,
                output_dir=output_dir,
                dev_data=dev_path,
                model_name=args.name,
                language=args.language,
                verbose=args.verbose or args.debug,
            )
            print(f"[flexipipe] trained {backend_type} model at {model_path}")
            _cleanup_nlpform_paths()
            return 0
        except NotImplementedError as e:
            import shutil
            ud_folder = getattr(args, "ud_folder", None)
            if teitok_temp_dir and not ud_folder:
                shutil.rmtree(teitok_temp_dir, ignore_errors=True)
            _cleanup_nlpform_paths()
            raise SystemExit(f"{backend_type} backend training is not yet implemented: {e}")
        except Exception as e:
            import shutil
            ud_folder = getattr(args, "ud_folder", None)
            if teitok_temp_dir and not ud_folder:
                shutil.rmtree(teitok_temp_dir, ignore_errors=True)
            _cleanup_nlpform_paths()
            raise SystemExit(f"Training failed: {e}")
        finally:
            # Clean up temporary TEITOK preparation directory (unless ud_folder was specified)
            ud_folder = getattr(args, "ud_folder", None)
            if teitok_temp_dir and not ud_folder:
                import shutil
                try:
                    shutil.rmtree(teitok_temp_dir, ignore_errors=True)
                except Exception:
                    pass
    
    else:
        _cleanup_nlpform_paths()
        raise SystemExit(f"Unknown backend: {backend_type}. Supported backends for training: flexitag, spacy, transformers")


def run_convert(args: argparse.Namespace) -> int:
    """Convert between formats: tagged files, treebanks, or lexicons."""
    conversion_type = getattr(args, "type", "tagged")
    
    if conversion_type == "tagged":
        return _run_convert_tagged(args)
    elif conversion_type == "treebank":
        return _run_convert_treebank(args)
    elif conversion_type == "lexicon":
        return _run_convert_lexicon(args)
    else:
        print(f"Error: Unknown conversion type '{conversion_type}'", file=sys.stderr)
        return 1


def _run_convert_tagged(args: argparse.Namespace) -> int:
    """Convert between input/output formats without running any NLP tasks."""
    from .teitok import load_teitok, save_teitok, dump_teitok
    from .conllu import conllu_to_document, document_to_conllu
    
    if not args.output_format:
        print("Error: --output-format is required for tagged conversion", file=sys.stderr)
        return 1
    
    # Handle input
    if args.input and args.input != "-":
        input_path = args.input
        read_from_stdin = False
    else:
        input_path = None
        read_from_stdin = True
    
    # Detect input format
    input_format = args.input_format
    stdin_content = None
    if input_format == "auto":
        if read_from_stdin:
            # For stdin, read content first to detect
            stdin_content = sys.stdin.read()
            sample = stdin_content[:4096] if len(stdin_content) > 4096 else stdin_content
            stripped = sample.lstrip()
            if "<tok" in sample or "<TEI" in sample or stripped.startswith("<TEI"):
                input_format = "teitok"
            else:
                for line in sample.splitlines():
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    if stripped_line.startswith("#"):
                        if stripped_line.startswith("# text") or stripped_line.startswith("# sent_id") or stripped_line.startswith("# newdoc"):
                            input_format = "conllu"
                            break
                        continue
                    if "\t" in stripped_line and len(stripped_line.split("\t")) >= 10:
                        input_format = "conllu"
                        break
                if input_format == "auto":
                    input_format = "raw"
        else:
            input_format = detect_input_format(input_path)
    
    if args.debug or args.verbose:
        print(f"[flexipipe] detected input format: {input_format}")
    
    # Load document
    if input_format == "conllu" or input_format == "conllu-ne":
        if read_from_stdin:
            conllu_text = stdin_content if stdin_content else sys.stdin.read()
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                conllu_text = f.read()
        doc = conllu_to_document(conllu_text)
        if not read_from_stdin:
            doc.meta.setdefault("source_path", input_path)
    elif input_format == "teitok":
        # For convert command, always use Python version to ensure we get all tokens
        # (including those inside <name> elements) without needing to rebuild C++ extension
        if read_from_stdin:
            # Write stdin to temp file for load_teitok
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as tmp:
                tmp.write(stdin_content if stdin_content else sys.stdin.read())
                tmp_path = tmp.name
            try:
                # Force Python version by providing an attribute mapping
                doc = load_teitok(tmp_path, xpos_attr="xpos")
            finally:
                os.unlink(tmp_path)
        else:
            # Force Python version by providing an attribute mapping
            doc = load_teitok(input_path, xpos_attr="xpos")
    else:
        print(f"Error: Input format '{input_format}' not supported for conversion. Use 'teitok' or 'conllu'.", file=sys.stderr)
        return 1
    
    # Parse tasks if specified
    try:
        requested_tasks = _parse_tasks_argument(getattr(args, "tasks", None))
    except ValueError as exc:
        print(f"[flexipipe] {exc}", file=sys.stderr)
        return 1
    
    # Filter document by tasks if specified
    if requested_tasks:
        _filter_document_by_tasks(doc, requested_tasks)
    
    # Write output
    output_format = args.output_format
    if output_format == "teitok":
        pretty_print = getattr(args, "pretty_print", False)
        if args.output:
            save_teitok(doc, args.output, pretty_print=pretty_print)
            if args.verbose or args.debug:
                print(f"[flexipipe] converted to TEI: {args.output}")
        else:
            print(dump_teitok(doc, pretty_print=pretty_print), end="")
    elif output_format in ("conllu", "conllu-ne"):
        entity_format = "ne" if output_format == "conllu-ne" else "iob"
        conllu_text = document_to_conllu(doc, entity_format=entity_format)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(conllu_text)
            if args.verbose or args.debug:
                print(f"[flexipipe] converted to {output_format.upper()}: {args.output}")
        else:
            print(conllu_text, end="")
    
    return 0


def _run_convert_treebank(args: argparse.Namespace) -> int:
    """Convert TEITOK corpus into UD-style CoNLL-U train/dev/test splits."""
    from .train import _prepare_teitok_corpus
    
    if not args.input:
        print("Error: --input is required for treebank conversion", file=sys.stderr)
        return 1
    if not args.output:
        print("Error: --output is required for treebank conversion", file=sys.stderr)
        return 1
    
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[flexipipe] Input path not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_type = args.backend or "spacy"
    # Parse attrs-map for convert command
    attrs_map = _parse_attrs_map(getattr(args, "attrs_map", None))
    xpos_attr = attrs_map.get("xpos")
    required_annotations = _required_annotations_for_backend(backend_type, xpos_attr)

    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]
    total = sum(ratios)
    if total <= 0:
        print("[flexipipe] Invalid split ratios; the sum must be greater than zero.", file=sys.stderr)
        return 1
    norm = [r / total for r in ratios]

    try:
        prepared = _prepare_teitok_corpus(
            teitok_dir=input_path,  # Can be file or directory
            output_dir=output_dir,
            required_annotations=required_annotations,
            backend_type=backend_type,
            train_ratio=norm[0],
            dev_ratio=norm[1],
            test_ratio=norm[2],
            seed=args.seed,
            verbose=args.verbose,
            xpos_attr=xpos_attr,
            reg_attr=attrs_map.get("reg"),
            expan_attr=attrs_map.get("expan"),
            lemma_attr=attrs_map.get("lemma"),
        )
    except Exception as exc:
        print(f"[flexipipe] Failed to convert TEITOK corpus: {exc}", file=sys.stderr)
        return 1

    token_counts = prepared.pop("_token_counts", {})
    print("[flexipipe] Created UD splits:")
    for split in ("train", "dev", "test"):
        path = prepared.get(split)
        if path:
            print(f"  {split}: {path}")
    if token_counts:
        total_tokens = sum(token_counts.values()) or 1
        print(f"[flexipipe] Token counts:")
        for split in ("train", "dev", "test"):
            count = token_counts.get(split, 0)
            pct = (count / total_tokens * 100) if total_tokens > 0 else 0
            print(f"  {split}: {count:,} ({pct:.1f}%)")
    
    return 0


def _run_convert_lexicon(args: argparse.Namespace) -> int:
    """Convert external lexicon (UniMorph, etc.) to FlexiPipe vocabulary format."""
    from .lexicon import convert_lexicon_to_vocab
    
    if not args.input:
        print("Error: --input is required for lexicon conversion", file=sys.stderr)
        return 1
    if not args.output:
        print("Error: --output is required for lexicon conversion", file=sys.stderr)
        return 1
    
    convert_lexicon_to_vocab(
        lexicon_file=Path(args.input),
        output_file=Path(args.output),
        tagset_file=Path(args.tagset) if args.tagset else None,
        corpus_file=Path(args.corpus) if args.corpus else None,
        default_count=args.default_count,
    )
    if args.verbose or args.debug:
        print(f"[flexipipe] converted lexicon: {args.input} -> {args.output}")
    else:
        print(f"[flexipipe] converted lexicon to {args.output}")
    return 0


def run_map_tags(args: argparse.Namespace) -> int:
    model_paths = [Path(p) for p in (args.models or [])]
    if not model_paths:
        raise SystemExit("map-tags requires at least one --model file")

    input_format = args.input_format
    if input_format == "auto":
        input_format = detect_input_format(args.input)
        if args.verbose or args.debug:
            print(f"[flexipipe] detected input format: {input_format}")

    direction = args.map_direction
    if direction is None:
        if sys.stdin.isatty():
            try:
                response = input("Map direction? (xpos / upos-feats / both): ").strip().lower()
            except EOFError:
                response = ""
            if response in {"xpos", "upos-feats", "both"}:
                direction = response
        if direction is None:
            raise SystemExit("map-tags requires --map-direction when not running interactively")

    # Apply direction defaults unless user explicitly overrode via --fill-* options
    if direction == "xpos":
        if args.fill_xpos is None:
            args.fill_xpos = True
        if args.fill_upos is None:
            args.fill_upos = False
        if args.fill_feats is None:
            args.fill_feats = False
    elif direction == "upos-feats":
        if args.fill_xpos is None:
            args.fill_xpos = False
        if args.fill_upos is None:
            args.fill_upos = True
        if args.fill_feats is None:
            args.fill_feats = True
    elif direction == "both":
        if args.fill_xpos is None:
            args.fill_xpos = True
        if args.fill_upos is None:
            args.fill_upos = True
        if args.fill_feats is None:
            args.fill_feats = True

    fill_xpos = bool(args.fill_xpos)
    fill_upos = bool(args.fill_upos)
    fill_feats = bool(args.fill_feats)
    if not any([fill_xpos, fill_upos, fill_feats]):
        raise SystemExit("map-tags: nothing to do (all fill options disabled)")

    mapping = build_tag_mapping_from_paths(model_paths)

    document = _load_document(Path(args.input), input_format, args=args)

    changes = mapping.enrich_document(
        document,
        fill_xpos=fill_xpos,
        fill_upos=fill_upos,
        fill_feats=fill_feats,
        allow_partial=args.allow_partial,
    )

    if args.verbose or args.debug:
        direction_desc = []
        if fill_xpos:
            direction_desc.append("XPOS")
        if fill_upos or fill_feats:
            parts = ["UPOS" if fill_upos else None, "FEATS" if fill_feats else None]
            direction_desc.append("+".join(p for p in parts if p))
        direction_text = ", ".join(direction_desc) or "nothing"
        print(f"[flexipipe] tag mapping ({direction_text}) updated {changes} tokens")

    output_format = args.output_format
    if output_format == "auto":
        output_format = input_format

    _write_document(document, args.output, output_format)

    if args.verbose or args.debug:
        target = args.output or "stdout"
        print(f"[flexipipe] saved mapped document to {target}")

    return 0


def _load_document(path: Path, fmt: str, *, args: Optional[argparse.Namespace] = None) -> Document:
    if fmt == "teitok":
        # Parse attrs-map into individual attributes
        attrs_map = _parse_attrs_map(getattr(args, "attrs_map", None) if args else None)
        xpos_attr = attrs_map.get("xpos")
        reg_attr = attrs_map.get("reg")
        expan_attr = attrs_map.get("expan")
        lemma_attr = attrs_map.get("lemma")
        return load_teitok(
            str(path),
            xpos_attr=xpos_attr,
            reg_attr=reg_attr,
            expan_attr=expan_attr,
            lemma_attr=lemma_attr,
        )
    if fmt == "conllu":
        text = path.read_text(encoding="utf-8", errors="replace")
        return conllu_to_document(text, doc_id=path.stem)
    raise SystemExit(f"Unsupported input format '{fmt}' for {path}")


def _write_document(
    document: Document,
    output: str | None,
    fmt: str,
    pretty_print: bool = False,
    *,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
    teitok_settings: Optional[TeitokSettings] = None,
) -> None:
    if fmt == "teitok":
        if output:
            save_teitok(
                document,
                output,
                pretty_print=pretty_print,
                spaceafter_handling=spaceafter_handling,
                skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
                settings=teitok_settings,
            )
        else:
            sys.stdout.write(
                dump_teitok(
                    document,
                    pretty_print=pretty_print,
                    spaceafter_handling=spaceafter_handling,
                    skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
                    settings=teitok_settings,
                )
            )
        return

    if fmt in ("conllu", "conllu-ne"):
        entity_format = "ne" if fmt == "conllu-ne" else "iob"
        conllu_text = document_to_conllu(
            document,
            model_info=None,
            entity_format=entity_format,
        )
        if output:
            Path(output).write_text(conllu_text, encoding="utf-8")
        else:
            sys.stdout.write(conllu_text)
        return

    if fmt == "json":
        payload = document_to_json_payload(document)
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if output:
            Path(output).write_text(json_text, encoding="utf-8")
        else:
            sys.stdout.write(json_text)
            if not json_text.endswith("\n"):
                sys.stdout.write("\n")
        return

    raise SystemExit(f"Unsupported output format '{fmt}'")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    backend_explicit = any(arg == "--backend" or arg.startswith("--backend=") for arg in argv)

    # Handle --version and -V early (before building parser to avoid unnecessary work)
    if "--version" in argv or "-V" in argv:
        parser = build_parser()
        try:
            parser.parse_args(argv)  # This will print version and exit
        except SystemExit:
            # Suppress BrokenPipeError when exiting
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                pass
            raise
        return 0

    if argv and argv[0] == "--detect-language":
        return _run_detect_language_standalone(argv)

    if not argv:
        argv = ["process"]
    elif argv[0] in ("-h", "--help"):
        parser = build_parser()
        try:
            parser.parse_args(argv)
        except SystemExit:
            # Suppress BrokenPipeError when exiting
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                pass
            raise
        return 0
    elif argv[0] not in TASK_CHOICES and argv[0].startswith("-"):
        argv = ["process", *argv]

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # Suppress BrokenPipeError when exiting due to argument errors
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        try:
            sys.stderr.flush()
        except BrokenPipeError:
            pass
        raise
    setattr(args, "_backend_explicit", backend_explicit)

    if not args.task:
        parser.error("No task specified. Use one of: " + ", ".join(TASK_CHOICES))

    if args.task == "process":
        return run_tag(args)
    if args.task == "train":
        return run_train(args)
    if args.task == "convert":
        return run_convert(args)
    if args.task == "config":
        return run_config(args)
    if args.task == "info":
        from .info import run_info_cli
        return run_info_cli(args)
    if args.task == "validate":
        return run_validate(args)
    if args.task == "benchmark":
        from .benchmark import run_cli as run_benchmark_cli
        run_benchmark_cli(args)
        return 0

    parser.error(f"Unknown task '{args.task}'. Supported tasks: {', '.join(TASK_CHOICES)}")
    return 1


def run_config(args: argparse.Namespace) -> int:
    """Run config command to manage flexipipe configuration."""
    from .model_storage import (
        get_flexipipe_models_dir,
        get_config_file,
        read_config,
        set_models_dir,
        set_default_backend,
        set_default_output_format,
        get_default_backend,
        get_default_output_format,
        set_default_create_implicit_mwt,
        get_default_create_implicit_mwt,
        set_default_writeback,
        get_default_writeback,
        set_auto_install_extras,
        get_auto_install_extras,
        set_prompt_install_extras,
        get_prompt_install_extras,
        set_language_detector,
        get_language_detector,
    )
    import os
    
    if args.wizard:
        return _run_config_wizard()

    if args.download_language_model:
        from .language_utils import ensure_fasttext_language_model

        try:
            path = ensure_fasttext_language_model(force_download=True)
        except RuntimeError as exc:
            print(f"[flexipipe] Failed to download fastText model: {exc}", file=sys.stderr)
            return 1
        else:
            print(f"[flexipipe] fastText language model available at {path}")
            return 0

    if args.refresh_all_caches:
        # Refresh all caches at once
        from .cache_manager import refresh_all_caches
        verbose = getattr(args, "verbose", False) or getattr(args, "debug", False)
        results = refresh_all_caches(verbose=verbose, force=True)
        success_count = sum(1 for v in results.values() if v)
        if success_count == len(results):
            print(f"[flexipipe] Successfully refreshed {success_count} cache(s)")
            return 0
        else:
            print(f"[flexipipe] Refreshed {success_count}/{len(results)} cache(s) (some failed)")
            return 1
    
    if args.set_model_registry_local_dir:
        # Set local model registry directory
        from .model_storage import write_config
        local_dir = Path(args.set_model_registry_local_dir).expanduser().resolve()
        write_config({"model_registry_local_dir": str(local_dir)})
        print(f"[flexipipe] Model registry local directory set to: {local_dir}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        print(f"[flexipipe] Backend registries will be read from: {local_dir}/registries/{{backend}}.json")
        print(f"[flexipipe] Note: This takes precedence over remote URLs. Use for local development.")
        return 0
    
    if args.set_model_registry_base_url:
        # Set model registry base URL
        from .model_storage import write_config
        write_config({"model_registry_base_url": args.set_model_registry_base_url})
        print(f"[flexipipe] Model registry base URL set to: {args.set_model_registry_base_url}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        print(f"[flexipipe] Backend registries will be fetched from: {args.set_model_registry_base_url}/{{backend}}.json")
        return 0
    
    if args.set_model_registry_url:
        # Set backend-specific registry URL (format: backend:url)
        from .model_storage import write_config, read_config
        import sys
        parts = args.set_model_registry_url.split(":", 1)
        if len(parts) != 2:
            print(f"[flexipipe] Error: Invalid format. Use 'backend:url' (e.g., flexitag:https://example.com/registry.json)", file=sys.stderr)
            return 1
        backend, url = parts
        config = read_config()
        config[f"model_registry_url_{backend}"] = url
        write_config(config)
        print(f"[flexipipe] Model registry URL for '{backend}' set to: {url}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0
    
    if args.set_models_dir:
        # Set models directory
        models_dir = Path(args.set_models_dir).expanduser().resolve()
        set_models_dir(models_dir)
        print(f"[flexipipe] Models directory set to: {models_dir}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        print(f"[flexipipe] All future model downloads will use this directory")
        return 0
    
    if args.set_default_backend:
        # Set default backend
        set_default_backend(args.set_default_backend)
        print(f"[flexipipe] Default backend set to: {args.set_default_backend}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0
    
    if args.set_default_output_format:
        # Set default output format
        set_default_output_format(args.set_default_output_format)
        print(f"[flexipipe] Default output format set to: {args.set_default_output_format}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0

    if args.set_language_detector:
        set_language_detector(args.set_language_detector)
        print(f"[flexipipe] Language detector set to: {args.set_language_detector}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0
    
    if args.set_default_create_implicit_mwt is not None:
        # Set default create_implicit_mwt
        set_default_create_implicit_mwt(args.set_default_create_implicit_mwt)
        status = "enabled" if args.set_default_create_implicit_mwt else "disabled"
        print(f"[flexipipe] Default create-implicit-mwt set to: {status}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0
    
    if args.set_default_writeback is not None:
        # Set default writeback
        set_default_writeback(args.set_default_writeback)
        status = "enabled" if args.set_default_writeback else "disabled"
        print(f"[flexipipe] Default writeback set to: {status}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0

    if args.set_auto_install_extras is not None:
        set_auto_install_extras(args.set_auto_install_extras)
        status = "enabled" if args.set_auto_install_extras else "disabled"
        print(f"[flexipipe] Automatic installation of optional extras set to: {status}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0

    if args.set_prompt_install_extras is not None:
        set_prompt_install_extras(args.set_prompt_install_extras)
        status = "enabled" if args.set_prompt_install_extras else "disabled"
        print(f"[flexipipe] Prompting before installing extras set to: {status}")
        print(f"[flexipipe] Configuration saved to: {get_config_file()}")
        return 0
    
    if args.show:
        # Show current configuration
        config = read_config()
        current_models_dir = get_flexipipe_models_dir(create=False)
        
        print("Current flexipipe configuration:")
        print(f"  Config file: {get_config_file()}")
        print(f"  Models directory: {current_models_dir}")
        
        if "FLEXIPIPE_MODELS_DIR" in os.environ:
            print(f"  (set via FLEXIPIPE_MODELS_DIR environment variable)")
        elif "models_dir" in config:
            print(f"  (configured in config file)")
        else:
            print(f"  (using default location)")
        
        # Show defaults
        default_backend = get_default_backend()
        default_output_format = get_default_output_format()
        default_create_implicit_mwt = get_default_create_implicit_mwt()
        default_writeback = get_default_writeback()
        auto_install_extras = get_auto_install_extras()
        prompt_install_extras = get_prompt_install_extras()
        
        if default_backend:
            print(f"  Default backend: {default_backend}")
        else:
            print(f"  Default backend: flexitag (not configured)")
        
        if default_output_format:
            print(f"  Default output format: {default_output_format}")
        else:
            print(f"  Default output format: teitok (not configured)")
        
        print(f"  Default create-implicit-mwt: {'enabled' if default_create_implicit_mwt else 'disabled'}")
        print(f"  Default writeback: {'enabled' if default_writeback else 'disabled'}")
        print(f"  Auto-install extras: {'enabled' if auto_install_extras else 'disabled'}")
        print(f"  Prompt before installing extras: {'enabled' if prompt_install_extras else 'disabled'}")
        print(f"  Language detector: {get_language_detector() or LANGUAGE_DETECTOR_DEFAULT}")
        
        # Show model registry configuration
        from .model_registry import get_registry_url, DEFAULT_REGISTRY_BASE_URL
        local_dir = config.get("model_registry_local_dir") or os.environ.get("FLEXIPIPE_MODEL_REGISTRY_LOCAL_DIR")
        if local_dir:
            local_path = Path(local_dir).expanduser().resolve()
            if "model_registry_local_dir" in config:
                print(f"  Model registry local directory: {local_path} (configured in config file)")
            else:
                print(f"  Model registry local directory: {local_path} (set via FLEXIPIPE_MODEL_REGISTRY_LOCAL_DIR environment variable)")
            print(f"    Registry files: {local_path}/registries/{{backend}}.json")
        else:
            base_url = config.get("model_registry_base_url") or os.environ.get("FLEXIPIPE_MODEL_REGISTRY_BASE_URL") or DEFAULT_REGISTRY_BASE_URL
            if "model_registry_base_url" in config:
                print(f"  Model registry base URL: {base_url} (configured in config file)")
            elif "FLEXIPIPE_MODEL_REGISTRY_BASE_URL" in os.environ:
                print(f"  Model registry base URL: {base_url} (set via FLEXIPIPE_MODEL_REGISTRY_BASE_URL environment variable)")
            else:
                print(f"  Model registry base URL: {base_url} (using default)")
        
        # Show backend-specific URLs
        backend_specific = {k: v for k, v in config.items() if k.startswith("model_registry_url_")}
        if backend_specific:
            print(f"  Backend-specific registry URLs:")
            for key, url in backend_specific.items():
                backend = key.replace("model_registry_url_", "")
                print(f"    {backend}: {url}")
        
        if config:
            print("\nAll configuration values:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        return 0
    
    # If no action specified, show help
    print("Configuration options:")
    print("  --set-models-dir <path>                      Set the models directory")
    print("  --set-model-registry-local-dir <path>        Set local directory for registries (for private repos)")
    print("  --set-model-registry-base-url <url>          Set the base URL for backend registries")
    print("  --set-model-registry-url <backend:url>       Set a backend-specific registry URL")
    print("  --refresh-all-caches                         Refresh all model caches at once")
    print("  --set-default-backend <backend>              Set the default backend")
    print("  --set-default-output-format <format>  Set the default output format (teitok, conllu, conllu-ne, json)")
    print("  --set-language-detector <detector>           Set the language detector backend")
    print("  --show                           Display current configuration")
    print(f"\nExample: python -m flexipipe config --set-models-dir /Volumes/External/models")
    print(f"Example: python -m flexipipe config --refresh-all-caches")
    print(f"Example: python -m flexipipe config --set-default-backend spacy")
    print(f"Example: python -m flexipipe config --set-default-output-format conllu")
    return 0


def _prompt_with_default(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _prompt_bool(prompt: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} ({default_text}): ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _prompt_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    options = "/".join(choices)
    default_suffix = f" [{default}]" if default else ""
    while True:
        response = input(f"{prompt} ({options}){default_suffix}: ").strip().lower()
        if not response and default:
            return default
        if response in choices:
            return response
        print(f"Please choose one of: {options}")


def _prefetch_backend_registries(backends: Collection[str], *, verbose: bool = False) -> None:
    """
    Prefetch registry entries for given backends so auto-selection works immediately.
    """
    if not backends:
        return
    from .backend_registry import get_model_entries

    for backend in backends:
        try:
            if verbose:
                print(f"[flexipipe] Prefetching registry for backend '{backend}'...")
            get_model_entries(
                backend,
                use_cache=False,
                refresh_cache=True,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to prefetch registry for backend '{backend}': {exc}")


def _run_config_wizard() -> int:
    from .model_storage import (
        get_flexipipe_models_dir,
        set_models_dir,
        get_default_backend,
        set_default_backend,
        get_default_output_format,
        set_default_output_format,
        get_default_create_implicit_mwt,
        set_default_create_implicit_mwt,
        get_default_writeback,
        set_default_writeback,
        get_auto_install_extras,
        set_auto_install_extras,
        get_prompt_install_extras,
        set_prompt_install_extras,
        get_language_detector,
        set_language_detector,
    )
    from .language_utils import ensure_fasttext_language_model

    print("\n=== Flexipipe Configuration Wizard ===\n")
    current_models_dir = str(get_flexipipe_models_dir(create=False))
    new_dir = _prompt_with_default("Models directory", current_models_dir)
    if new_dir and new_dir != current_models_dir:
        set_models_dir(new_dir)
        print(f"[flexipipe] Models directory set to {new_dir}")

    backend_choices = get_backend_choices()
    current_backend = get_default_backend() or "flexitag"
    backend = _prompt_choice("Default backend", backend_choices, default=current_backend)
    set_default_backend(backend)

    output_choices = ["teitok", "conllu", "conllu-ne", "json"]
    current_output = get_default_output_format() or "teitok"
    output_format = _prompt_choice("Default output format", output_choices, default=current_output)
    set_default_output_format(output_format)

    detector_choices = sorted(list_language_detectors().keys())
    current_detector = get_language_detector() or LANGUAGE_DETECTOR_DEFAULT
    language_detector = _prompt_choice(
        "Default language detector",
        detector_choices,
        default=current_detector,
    )
    set_language_detector(language_detector)

    create_mwt = _prompt_bool(
        "Create implicit multi-word tokens by default?",
        get_default_create_implicit_mwt(),
    )
    set_default_create_implicit_mwt(create_mwt)

    needs_writeback_prompt = backend == "flexitag" or output_format == "teitok"
    if needs_writeback_prompt:
        writeback = _prompt_bool(
            "Enable writeback mode by default?",
            get_default_writeback(),
        )
    else:
        writeback = False
    set_default_writeback(writeback)

    auto_install = _prompt_bool(
        "Automatically install missing optional extras?",
        get_auto_install_extras(),
    )
    set_auto_install_extras(auto_install)

    if auto_install:
        set_prompt_install_extras(get_prompt_install_extras())
    else:
        prompt_install = _prompt_bool(
            "Prompt before installing extras (when auto-install is disabled)?",
            get_prompt_install_extras(),
        )
        set_prompt_install_extras(prompt_install)

    if language_detector == "fasttext":
        if _prompt_bool("Download fastText language detection model now?", True):
            try:
                path = ensure_fasttext_language_model()
                print(f"[flexipipe] fastText language model ready at {path}")
            except RuntimeError as exc:
                print(f"[flexipipe] Failed to prepare language model: {exc}")
    else:
        print(f"[flexipipe] Skipping fastText download (language detector '{language_detector}' does not require it).")

    print("\n[flexipipe] Prefetching model registries for installed backends (this may take a moment)...")
    _prefetch_backend_registries(_get_language_backend_priority(), verbose=True)

    print("\n[flexipipe] Wizard complete. Run 'python -m flexipipe config --show' to review settings.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
