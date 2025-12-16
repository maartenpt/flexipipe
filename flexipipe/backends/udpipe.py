"""UDPipe REST backend implementation and registry spec."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document, document_to_conllu, parse_conllu_from_backend
from ..doc import Document
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
    clean_language_name,
    standardize_language_metadata,
)
from ..model_registry import DEFAULT_REGISTRY_BASE_URL, get_registry_url
from ..model_storage import (
    get_backend_models_dir,
    get_backend_registry_file,
    read_model_cache_entry,
    write_backend_registry_file,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
DEFAULT_REST_ENDPOINT = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
# Public UDPipe model list endpoint (for REST models)
DEFAULT_UDPIPE_MODELS_ENDPOINT = "https://lindat.mff.cuni.cz/services/udpipe/api/models"
DEFAULT_UDPIPE_REGISTRY_URL = DEFAULT_REGISTRY_BASE_URL.rstrip("/") + "/udpipe.json"


def get_udpipe_model_entries(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    endpoint = url or DEFAULT_REST_ENDPOINT
    registry_url = get_registry_url("udpipe")

    # If caller asked for refresh, drop the cached registry and model cache so we refetch 2.17+
    if refresh_cache:
        try:
            # Remove curated registry cache on disk
            registry_path = get_backend_registry_file("udpipe")
            registry_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            # Remove model list cache used for REST backends by overwriting with empty data
            cache_key = f"udpipe:{DEFAULT_UDPIPE_MODELS_ENDPOINT}"
            write_model_cache_entry(cache_key, {})
        except Exception:
            pass

    curated_entries = _load_curated_udpipe_registry(
        registry_url,
        force_download=refresh_cache or not use_cache,
        verbose=verbose,
    )
    if curated_entries:
        return curated_entries

    if verbose:
        print("[flexipipe] Warning: curated UDPipe registry unavailable; falling back to live UDPipe API.")

    cache_key = f"udpipe:{endpoint}"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            return cached

    fallback_entries = fetch_udpipe_models_from_api(endpoint=endpoint, verbose=verbose)
    if fallback_entries:
        try:
            write_model_cache_entry(cache_key, fallback_entries)
        except (OSError, PermissionError):
            pass
    return fallback_entries


def _load_curated_udpipe_registry(
    registry_url: Optional[str],
    *,
    force_download: bool,
    verbose: bool,
) -> Dict[str, Dict[str, str]]:
    if not registry_url:
        return {}

    registry_path = _ensure_curated_registry_local_copy(
        registry_url,
        force_download=force_download,
        verbose=verbose,
    )
    if registry_path is None or not registry_path.exists():
        return {}
    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            registry_payload = json.load(handle)
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to read UDPipe registry cache: {exc}")
        return {}
    return _entries_from_curated_registry(registry_payload, verbose=verbose)


def _ensure_curated_registry_local_copy(
    registry_url: str,
    *,
    force_download: bool,
    verbose: bool,
) -> Optional[Path]:
    registry_path = get_backend_registry_file("udpipe")
    if registry_path.exists() and not force_download:
        return registry_path

    try:
        if registry_url.startswith("file://"):
            source_path = Path(registry_url[7:])
            if not source_path.exists():
                raise FileNotFoundError(f"registry file not found: {source_path}")
            with source_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            write_backend_registry_file("udpipe", data)
        else:
            response = requests.get(registry_url, timeout=15)
            response.raise_for_status()
            data = response.json()
            write_backend_registry_file("udpipe", data)
        if verbose:
            print(f"[flexipipe] Saved curated UDPipe registry to {registry_path}")
        return registry_path
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to fetch curated UDPipe registry ({registry_url}): {exc}")
        if registry_path.exists() and not force_download:
            return registry_path
        return None


def _entries_from_curated_registry(
    registry_payload: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    sources = registry_payload.get("sources", {}) if isinstance(registry_payload, dict) else {}
    entries: Dict[str, Dict[str, str]] = {}
    for source_name, models in sources.items():
        if not isinstance(models, list):
            continue
        for model_info in models:
            if not isinstance(model_info, dict):
                continue

            model_name = model_info.get("model")
            if not model_name:
                continue

            # Preserve the original language_iso from the registry
            # Don't normalize non-standard codes like "old_ch" (Old Church Slavonic)
            original_language_iso = model_info.get("language_iso")
            original_language_name = model_info.get("language_name")
            
            # Debug: log if we see OCS models with incorrect language_iso
            if verbose and model_name and ("old_church" in model_name.lower() or "old_east_slavic" in model_name.lower()):
                if original_language_iso and original_language_iso.lower() not in {"old_ch", "cu", "orv"}:
                    print(f"[flexipipe] WARNING: OCS/Old East Slavic model '{model_name}' has language_iso='{original_language_iso}' in registry (expected 'old_ch', 'cu', or 'orv')", file=sys.stderr)
            
            entry = build_model_entry(
                "udpipe",
                model_name,
                language_code=original_language_iso,
                language_name=original_language_name,
                preferred=model_info.get("preferred", False),
                features=model_info.get("features"),
                tasks=model_info.get("tasks"),
            )
            
            # If the registry has a non-standard code (like "old_ch"), preserve it
            # standardize_language_metadata might normalize it incorrectly through name lookup
            # Check if the model name indicates a non-standard language that should preserve its code
            model_name_lower = (model_name or "").lower()
            if original_language_iso:
                original_iso_lower = original_language_iso.lower()
                # Non-standard codes that should be preserved
                non_standard_codes = {"old_ch", "cu", "orv"}  # Old Church Slavonic, Church Slavonic, Old East Slavic
                if original_iso_lower in non_standard_codes:
                    # Preserve the original non-standard code
                    entry[LANGUAGE_FIELD_ISO] = original_iso_lower
                elif "old_church" in model_name_lower or "old_east_slavic" in model_name_lower:
                    # Model name indicates OCS/Old East Slavic but registry might have wrong code
                    # If the normalized code doesn't match, preserve original if it's a known non-standard code
                    normalized_iso = entry.get(LANGUAGE_FIELD_ISO, "").lower()
                    if normalized_iso not in non_standard_codes and original_iso_lower in non_standard_codes:
                        entry[LANGUAGE_FIELD_ISO] = original_iso_lower
            for extra_key in (
                "components",
                "description",
                "download_url",
                "training_data",
                "techniques",
            ):
                if model_info.get(extra_key) is not None:
                    entry[extra_key] = model_info[extra_key]
            entry["source"] = source_name
            entries[model_name] = entry

    if verbose and entries:
        print(f"[flexipipe] Loaded {len(entries)} curated UDPipe entries from registry.")
    return entries


def fetch_udpipe_models_from_api(
    *,
    endpoint: str = DEFAULT_REST_ENDPOINT,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    prepared_models: Dict[str, Dict[str, str]] = {}
    try:
        response = requests.get(
            endpoint.replace("/process", "/models"),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to fetch UDPipe models from API: {exc}")
        payload = {}

    models_obj = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models_obj, dict):
        iterable = models_obj.items()
    elif isinstance(models_obj, list):
        iterable = ((item.get("name"), item) for item in models_obj if isinstance(item, dict))
    else:
        iterable = []

    for model_name, model_info in iterable:
        if not model_name:
            continue
        components: List[str]
        language_hint: Optional[str] = None
        if isinstance(model_info, dict):
            components = model_info.get("components") or model_info.get("available_components") or []
            language_hint = model_info.get("language") or model_info.get("lang") or model_info.get("abbr")
        elif isinstance(model_info, list):
            components = [str(comp) for comp in model_info]
        else:
            components = []

        try:
            entry = _build_udpipe_entry_from_components(
                model_name,
                components=components,
                language_hint=language_hint,
            )
            prepared_models[model_name] = entry
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to parse UDPipe model '{model_name}': {exc}")
            continue
    return prepared_models


def _build_udpipe_entry_from_components(
    model_name: str,
    *,
    components: Optional[List[str]] = None,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    components = components or []
    slug = model_name.split("-ud-")[0]
    slug_parts = slug.split("-")
    primary_lang = slug_parts[0] if slug_parts else slug
    lang_candidate = language_hint or primary_lang or model_name.split("_")[0]
    lang_candidate = lang_candidate.replace("_", " ").replace("-", " ").strip()

    lang_metadata = standardize_language_metadata(language_code=None, language_name=lang_candidate)
    lang_code = lang_metadata.get(LANGUAGE_FIELD_ISO) or primary_lang[:2].lower()
    lang_display_final = clean_language_name(lang_metadata.get(LANGUAGE_FIELD_NAME) or lang_candidate.title())

    feature_parts: List[str] = []
    lowered_components = [comp.lower() for comp in components]
    if "tokenizer" in lowered_components:
        feature_parts.append("tokenization")
    if "tagger" in lowered_components:
        feature_parts.extend(["lemma", "upos", "xpos", "feats"])
    if "parser" in lowered_components:
        feature_parts.append("depparse")
    features = ", ".join(feature_parts) if feature_parts else "tokenization, tagging, parsing"

    preferred = lang_code == "en" and ("ewt" in model_name.lower() or "english" in model_name.lower())
    entry = build_model_entry(
        "udpipe",
        model_name,
        language_code=lang_code,
        language_name=lang_display_final,
        features=features,
        components=components,
        preferred=preferred,
    )
    return entry


def list_udpipe_models_display(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
    **kwargs: Any,
) -> int:
    try:
        prepared_models = get_udpipe_model_entries(
            url,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=verbose,
        )

        print(f"\nAvailable UDPipe models (deduplicated, showing newest versions only):")
        print(f"{'Model Name':<52} {'ISO':<8} {'Language':<20} {'Features':<30}")
        print("=" * 117)

        for model_name in sorted(prepared_models.keys()):
            entry = prepared_models[model_name]
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            features = entry.get("features", "unknown")
            suffix = "*" if entry.get("preferred") else ""
            display_name = f"{model_name}{suffix}"
            print(f"{display_name:<52} {lang_iso:<8} {lang_display:<20} {features:<30}")

        total_models = len(prepared_models)
        unique_languages = {
            entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
            for entry in prepared_models.values()
            if entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
        }
        print(f"\n(*) Preferred model used by auto-selection")
        print(f"Total: {total_models} model(s) for {len(unique_languages)} language(s)")

        return 0
    except Exception as e:
        print(f"Error fetching UDPipe models: {e}")
        import traceback

        traceback.print_exc()
        return 1


class UDPipeRESTBackend(BackendManager):
    """Neural backend that delegates tagging to a UDPipe REST service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        model: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 50,
        extra_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Any = None,
        log_requests: bool = False,
    ):
        self._endpoint = endpoint_url
        self._model = model
        self._timeout = timeout
        self._batch_size = batch_size
        self._extra_params = extra_params or {}
        self._headers = headers or {}
        self._session = session or requests.Session()
        self._log = log_requests

    def _request(self, data: str, input_format: str = "plain") -> dict:
        """
        Send request to UDPipe REST API.
        
        Args:
            data: The input data (raw text or CoNLL-U format)
            input_format: Either "plain" for raw text or "conllu" for CoNLL-U format
        """
        # For CoNLL-U input, send as form field (not file upload)
        # Reference: curl -F "data=# sent_id = 1..." -F model=english -F tokenizer= -F tagger= -F parser= ...
        # The data should be sent as a regular form field, not as a file upload
        if input_format == "conllu":
            # Build form data payload
            form_data = {}
            form_data["input"] = "conllu"
            form_data["data"] = data  # Send as regular form field, not file upload
            # Don't set tokenizer - it should not be set for CoNLL-U input
            form_data["tagger"] = ""
            form_data["parser"] = ""
            
            # Allow extra_params to override defaults, but don't let it add tokenizer
            # (tokenizer should not be set for CoNLL-U input)
            extra_params = dict(self._extra_params) if self._extra_params else {}
            extra_params.pop("tokenizer", None)  # Remove tokenizer if present
            form_data.update(extra_params)
            # Model is required for CoNLL-U input
            # UDPipe REST API accepts both model names and language codes
            if self._model:
                form_data["model"] = self._model
            else:
                # If no model is set, we need a model or language
                raise ValueError(
                    "UDPipe REST API requires a model name or language code for CoNLL-U input. "
                    f"Model is currently '{self._model}'. Please specify a model or language."
                )
            
            # Debug logging - always log on error, or if _log is enabled
            import sys
            should_log = self._log
            if not should_log:
                # Store original data for potential logging on error
                _original_data = data
            
            response = self._session.post(
                self._endpoint,
                data=form_data,
                headers=self._headers,
                timeout=self._timeout,
            )
        else:
            # For raw text, use regular form data
            payload = {"tagger": "", "parser": "", "data": data}
            if "tokenizer" not in self._extra_params:
                payload["tokenizer"] = "normalized_spaces"
            
            # Allow extra_params to override defaults
            payload.update(self._extra_params)
            if self._model:
                payload["model"] = self._model
            
            response = self._session.post(
                self._endpoint,
                data=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
        
        if response.status_code != 200:
            # Log error details for debugging
            import sys
            error_msg = response.text[:500] if response.text else "No error message"
            print(f"[flexipipe] UDPipe REST API error {response.status_code}: {error_msg}", file=sys.stderr)
            # If CoNLL-U input, also log the request details
            if input_format == "conllu":
                print(f"[flexipipe] UDPipe request details: input=conllu, model={self._model}, data_length={len(data)}", file=sys.stderr)
                print(f"[flexipipe] UDPipe form_data keys: {list(form_data.keys()) if 'form_data' in locals() else 'N/A'}", file=sys.stderr)
                # Log first 500 chars of CoNLL-U data for debugging
                print(f"[flexipipe] CoNLL-U data (first 500 chars):\n{data[:500]}", file=sys.stderr)
                # Also check if data is valid UTF-8
                try:
                    data.encode('utf-8')
                except UnicodeEncodeError as e:
                    print(f"[flexipipe] CoNLL-U data encoding error: {e}", file=sys.stderr)
                # Check for any suspicious characters
                if '\x00' in data:
                    print(f"[flexipipe] WARNING: CoNLL-U data contains null bytes", file=sys.stderr)
                # Log the actual form_data being sent
                print(f"[flexipipe] Form data types: {[(k, type(v).__name__) for k, v in form_data.items()]}", file=sys.stderr)
                # Save full CoNLL-U to file for debugging
                try:
                    with open('/tmp/flexipipe_udpipe_debug.conllu', 'w', encoding='utf-8') as f:
                        f.write(data)
                    print(f"[flexipipe] Saved full CoNLL-U to /tmp/flexipipe_udpipe_debug.conllu for debugging", file=sys.stderr)
                except Exception as e:
                    print(f"[flexipipe] Could not save debug file: {e}", file=sys.stderr)
        response.raise_for_status()
        return response.json()

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components
        
        if use_raw_text:
            # Raw text mode: extract text from sentences and send as plain text
            text = "\n".join(sentence.text or "" for sentence in document.sentences)
            response = self._request(text, input_format="plain")
        else:
            # Pretokenized mode: send CoNLL-U format to preserve tokenization and TokIds
            # UDPipe REST API should preserve TokIds in MISC column when input=conllu
            # Force include TokId in MISC so UDPipe preserves it
            # Include SpaceAfter=No in MISC to accurately represent spacing and prevent incorrect MWT creation
            conllu_text = document_to_conllu(document, include_tokid=True, suppress_space_after_in_misc=False)
            
            # UDPipe REST API doesn't accept CoNLL-U with existing dependency relations
            # Clear HEAD (column 6) and DEPREL (column 7) before sending
            # SpaceAfter=No is included in MISC to accurately represent spacing
            # UDPipe may create MWTs from SpaceAfter=No sequences, which we handle in matching
            lines = conllu_text.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.startswith('#') or not line.strip():
                    cleaned_lines.append(line)
                else:
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        # Clear HEAD (column 6) and DEPREL (column 7)
                        parts[6] = '_'
                        parts[7] = '_'
                    cleaned_lines.append('\t'.join(parts))
            conllu_text = '\n'.join(cleaned_lines)
            
            # Debug: log document info and CoNLL-U if logging is enabled
            if self._log:
                import sys
                print(f"[flexipipe] Document ID: {document.id}, meta: {document.meta}", file=sys.stderr)
                print(f"[flexipipe] CoNLL-U length: {len(conllu_text)}", file=sys.stderr)
                print(f"[flexipipe] Sending CoNLL-U (first 500 chars):\n{conllu_text[:500]}", file=sys.stderr)
            # Validate CoNLL-U format before sending
            # Check for common issues that might cause UDPipe to reject it
            if not conllu_text.strip():
                raise ValueError("CoNLL-U text is empty")
            # Ensure it's valid UTF-8
            try:
                conllu_text.encode('utf-8')
            except UnicodeEncodeError as e:
                raise ValueError(f"CoNLL-U text contains invalid UTF-8: {e}") from e
            response = self._request(conllu_text, input_format="conllu")
        
        result_text = response.get("result", "")
        
        if not result_text:
            raise RuntimeError("UDPipe REST response did not contain any result data")
        
        # Parse the CoNLL-U result and return the tagged document
        # When using CoNLL-U input, TokIds should be preserved in MISC column
        # Use helper function that preserves flags from source document
        tagged_doc = parse_conllu_from_backend(result_text, document)
        
        return NeuralResult(document=tagged_doc, stats={})

    def tag_text(self, text: str, *, doc_id: str = "") -> Document:
        """
        Convenience method to tag raw text directly.
        
        Args:
            text: Raw text to tag
            doc_id: Optional document ID
            
        Returns:
            Tagged Document
        """
        from ..doc import Document
        input_doc = Document.from_plain_text(text, doc_id=doc_id, tokenize=False, segment=False)
        result = self.tag(input_doc, use_raw_text=True)
        return result.document

    def train(self, *args, **kwargs):  # pragma: no cover - not implemented
        raise NotImplementedError("UDPipe REST backend does not support training.")

    def supports_training(self) -> bool:
        return False


def _create_udpipe_backend(
    *,
    model: str | None = None,
    model_name: str | None = None,
    language: str | None = None,
    endpoint_url: str | None = None,
    timeout: float = 30.0,
    batch_size: int = 50,
    extra_params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    session: Any = None,
    log_requests: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> UDPipeRESTBackend:
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "UDPipe", allowed_extra=["download_model", "training"])

    resolved_model = model or model_name
    # If model is "auto" or not set, try to resolve from language
    # UDPipe REST API accepts both model names and language codes
    if not resolved_model or resolved_model == "auto":
        if language:
            # Try to resolve to a model name first, but if that fails, use language code
            try:
                resolved_model = resolve_model_from_language(language, "udpipe")
            except ValueError:
                # If no model found, use language code directly (UDPipe accepts language codes)
                resolved_model = language
        else:
            resolved_model = None
    
    if not resolved_model:
        raise ValueError("UDPipe REST backend requires a model name or language code. Provide --model or --language.")

    resolved_endpoint = endpoint_url or DEFAULT_REST_ENDPOINT

    return UDPipeRESTBackend(
        resolved_endpoint,
        model=resolved_model,
        timeout=timeout,
        batch_size=batch_size,
        extra_params=extra_params,
        headers=headers,
        session=session,
        log_requests=log_requests,
    )


BACKEND_SPEC = BackendSpec(
    name="udpipe",
    description="UDPipe - REST service for UDPipe models (tokenization, tagging, parsing)",
    factory=_create_udpipe_backend,
    get_model_entries=get_udpipe_model_entries,
    list_models=list_udpipe_models_display,
    supports_training=False,
    is_rest=True,
    url="https://lindat.mff.cuni.cz/services/udpipe",
    model_registry_url=DEFAULT_UDPIPE_REGISTRY_URL,
)

