"""Backend spec and implementation for the NameTag REST backend."""

from __future__ import annotations

import io
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document, document_to_conllu
from ..doc import Document, Entity
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_storage import read_model_cache_entry, write_model_cache_entry
from ..neural_backend import BackendManager, NeuralResult

try:
    import requests
except ImportError as exc:  # pragma: no cover - import error handling
    raise ImportError(
        "NameTag REST backend requires the 'requests' package. "
        "Install it with: pip install requests"
    ) from exc


def _document_to_plain_text(document: Document) -> str:
    """Reconstruct plain text from a Document (preserving sentence texts when available)."""
    sentences = []
    for sent in document.sentences:
        if sent.text:
            sentences.append(sent.text.strip())
            continue
        parts: list[str] = []
        for tok in sent.tokens:
            parts.append(tok.form)
            if tok.space_after is not False:
                parts.append(" ")
        sentences.append("".join(parts).strip())
    return "\n".join(filter(None, sentences)) or document.id or ""


def list_nametag_models(url: str = "https://lindat.mff.cuni.cz/services/nametag/api/models") -> Dict:
    """
    Fetch the list of available NameTag models from the service.
    
    Returns a dictionary mapping model names to model information.
    """
    try:
        response = requests.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        # The API might return {"models": {...}} or just the models dict
        if isinstance(data, dict) and "models" in data:
            return data["models"]
        return data
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch NameTag model list: {exc}") from exc


MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


DEFAULT_NAMETAG_REGISTRY_URL = "https://raw.githubusercontent.com/ufal/flexipipe-models/main/registries/nametag.json"


def get_nametag_model_entries(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Get NameTag model entries from the curated flexipipe-models registry.
    
    Falls back to hardcoded multilingual model if registry is unavailable.
    """
    from pathlib import Path
    from ..model_registry import DEFAULT_REGISTRY_BASE_URL
    
    registry_url = url or DEFAULT_NAMETAG_REGISTRY_URL
    cache_key = f"nametag:{registry_url}"
    
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            if verbose:
                print("[flexipipe] Using cached NameTag model list (use --refresh-cache to update).")
            return cached

    if verbose:
        print(f"[flexipipe] Fetching NameTag models from {registry_url}...")
    
    # Try to fetch from curated registry
    try:
        if registry_url.startswith("file://"):
            # Local file path
            file_path = Path(registry_url[7:])
            with open(file_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
        else:
            response = requests.get(registry_url, timeout=10.0)
            response.raise_for_status()
            registry_data = response.json()
        
        # Extract models from registry structure
        raw_models = {}
        sources = registry_data.get("sources", {})
        for source_type in ["official", "flexipipe", "community"]:
            if source_type in sources:
                for model_entry in sources[source_type]:
                    model_name = model_entry.get("model")
                    if model_name:
                        raw_models[model_name] = model_entry
    except (RuntimeError, requests.RequestException, json.JSONDecodeError, OSError) as exc:
        # Registry unavailable, fall back to hardcoded multilingual model
        if verbose:
            print(f"[flexipipe] NameTag registry unavailable ({exc}), using fallback multilingual model.")
        raw_models = _get_known_nametag_models()
    
    prepared_models: Dict[str, Dict[str, Any]] = {}
    for model_name, model_info in raw_models.items():
        # Extract language info from model name or info
        if isinstance(model_info, dict):
            lang_code = model_info.get("iso", "") or model_info.get("language", "")
            lang_display = model_info.get("name", "") or model_info.get("language_display", "")
        else:
            # If model_info is not a dict, try to extract from model name
            parts = model_name.split("-")
            lang_code = parts[0] if parts else ""
            lang_display = lang_code.replace("_", " ").title()
        
        entry = build_model_entry(
            "nametag",
            model_name,
            language_code=lang_code,
            language_name=lang_display,
            features="ner",
        )
        prepared_models[model_name] = entry
    
    # Only write to cache if refresh_cache is True (explicit refresh)
    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, prepared_models)
        except (OSError, PermissionError):
            # If we can't write cache, that's okay - we'll just return the entries without caching
            pass
    return prepared_models


def _get_known_nametag_models() -> Dict[str, Dict[str, Any]]:
    """
    Return fallback multilingual model when registry is unavailable.
    """
    return {
        "nametag3-multilingual-conll-250203": {
            "model": "nametag3-multilingual-conll-250203",
            "language_iso": "xx",
            "language_name": "Multilingual",
            "features": "ner",
            "tasks": ["ner"],
            "preferred": True,
            "description": "NameTag 3 multilingual model trained on CoNLL data, supports all languages"
        }
    }
    known_models = {
        "ceb": {"iso": "ceb", "name": "Cebuano"},
        "zh": {"iso": "zh", "name": "Chinese"},
        "hr": {"iso": "hr", "name": "Croatian"},
        "cs": {"iso": "cs", "name": "Czech"},
        "da": {"iso": "da", "name": "Danish"},
        "en": {"iso": "en", "name": "English"},
        "nb": {"iso": "nb", "name": "Norwegian Bokmål"},
        "nn": {"iso": "nn", "name": "Norwegian Nynorsk"},
        "pt": {"iso": "pt", "name": "Portuguese"},
        "ru": {"iso": "ru", "name": "Russian"},
        "sr": {"iso": "sr", "name": "Serbian"},
        "sk": {"iso": "sk", "name": "Slovak"},
        "sv": {"iso": "sv", "name": "Swedish"},
        "tl": {"iso": "tl", "name": "Tagalog"},
        "uk": {"iso": "uk", "name": "Ukrainian"},
        "ar": {"iso": "ar", "name": "Arabic"},
        "nl": {"iso": "nl", "name": "Dutch"},
        "de": {"iso": "de", "name": "German"},
        "es": {"iso": "es", "name": "Spanish"},
    }
    return known_models


def list_nametag_models_display(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> int:
    """
    List available NameTag models with formatted output.
    Prints formatted output and returns exit code (0 for success, 1 for error).
    """
    try:
        prepared_models = get_nametag_model_entries(
            url,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=True,
        )

        print(f"\nAvailable NameTag models:")
        print(f"{'Model Name':<30} {'ISO':<8} {'Language':<25}")
        print("=" * 70)
        
        for model_name in sorted(prepared_models.keys()):
            entry = prepared_models[model_name]
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            print(f"{model_name:<30} {lang_iso:<8} {lang_display:<25}")
        
        total_models = len(prepared_models)
        unique_languages = {
            entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
            for entry in prepared_models.values()
            if entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
        }
        print(f"\nTotal: {total_models} model(s) for {len(unique_languages)} language(s)")
        
        return 0
    except Exception as e:
        print(f"Error fetching NameTag models: {e}")
        import traceback
        traceback.print_exc()
        return 1


class NameTagRESTBackend(BackendManager):
    """Neural backend that delegates NER tagging to a NameTag REST service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        version: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 50,
        extra_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
        log_requests: bool = False,
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for NameTag REST backend")
        self.endpoint_url = endpoint_url
        self.version = version or "3"
        # If no explicit model provided, default to multilingual model (supports conllu input)
        if model is None:
            self.model = "nametag3-multilingual-conll-250203"
        elif model and ("140408" in model or (not model.startswith("nametag3-") and not model.startswith("nametag2-"))):
            # Old models (like english-conll-140408) don't support conllu input format
            # Replace with multilingual model that does support it
            self.model = "nametag3-multilingual-conll-250203"
        else:
            self.model = model
        self.language = language
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.extra_params = extra_params or {}
        self.headers = headers or {}
        self.session = session or requests.Session()
        self.log_requests = log_requests
        self._model_name = model or language or endpoint_url
        self._backend_name = "nametag"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_descriptor(self) -> str:
        """Used by CLI to describe the model in output."""
        if self.model:
            return self.model
        elif self.language:
            return self.language
        else:
            return f"NameTag{self.version}"

    @property
    def _backend_info(self) -> str:
        """Used by CLI to describe this backend."""
        model_name = self.model or self.language or "unknown"
        return f"nametag: {model_name}"

    def supports_training(self) -> bool:  # pragma: no cover - trivial
        return False

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[list[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        start = time.time()

        batches = self._split_document(document)
        aggregated_doc = Document(
            id=document.id,
            meta=dict(document.meta),
            attrs=dict(document.attrs) if hasattr(document, 'attrs') else {},
        )
        # Copy spans if they exist (spans is Dict[str, List[Span]])
        if hasattr(document, 'spans') and document.spans:
            if isinstance(document.spans, dict):
                aggregated_doc.spans = {k: list(v) for k, v in document.spans.items()}
            else:
                aggregated_doc.spans = dict(document.spans) if document.spans else {}
        total_tokens = 0

        for batch_index, batch_doc in enumerate(batches):
            payload = self._build_payload(batch_doc, use_raw_text, overrides)
            if self.log_requests:
                print(f"[nametag] POST {self.endpoint_url} (batch {batch_index + 1}/{len(batches)})")

            request_start = time.time()
            response = self._post(payload)
            request_elapsed = time.time() - request_start

            if self.log_requests:
                snippet = response.text[:400] + ("…" if len(response.text) > 400 else "")
                print(f"[nametag] Response status: {response.status_code}")
                print(f"[nametag] Response snippet: {snippet}")
                print(f"[nametag] Request duration: {request_elapsed:.2f}s")

            # Parse response and extract entities
            chunk_doc = self._parse_response(response, batch_doc)
            if self.log_requests and batch_index == 0 and chunk_doc.sentences:
                first_sentence = chunk_doc.sentences[0]
                token_forms = " ".join(tok.form for tok in first_sentence.tokens)
                print("[nametag] First sentence returned (decoded):")
                print(f"  sent_id={first_sentence.sent_id or first_sentence.id or 'n/a'}")
                print(f"  text={first_sentence.text or token_forms}")
                print(f"  tokens={token_forms}")
                if first_sentence.entities:
                    print(f"  entities={len(first_sentence.entities)}")
            
            # Merge attrs and meta from chunk_doc
            if hasattr(chunk_doc, 'attrs'):
                for key, value in chunk_doc.attrs.items():
                    aggregated_doc.attrs[key] = value
            if chunk_doc.meta:
                aggregated_doc.meta.update(chunk_doc.meta)
            # Merge spans (spans is Dict[str, List[Span]])
            if hasattr(chunk_doc, 'spans') and chunk_doc.spans:
                for layer, spans_list in chunk_doc.spans.items():
                    if layer not in aggregated_doc.spans:
                        aggregated_doc.spans[layer] = []
                    aggregated_doc.spans[layer].extend(spans_list)
            
            aggregated_doc.sentences.extend(chunk_doc.sentences)
            total_tokens += sum(len(sent.tokens) for sent in chunk_doc.sentences)

        elapsed = time.time() - start
        stats = {
            "elapsed_seconds": elapsed,
            "backend": "nametag-rest",
            "token_count": total_tokens,
        }
        return NeuralResult(document=aggregated_doc, stats=stats)

    def _split_document(self, document: Document) -> list[Document]:
        """Split document sentences into batches to avoid oversized payloads."""
        if not document.sentences:
            return [document]
        batches: list[Document] = []
        for start_idx in range(0, len(document.sentences), self.batch_size):
            batch = Document(
                id=f"{document.id or 'doc'}-batch-{len(batches)+1}",
                meta=dict(document.meta),
                sentences=[*document.sentences[start_idx:start_idx + self.batch_size]],
            )
            batches.append(batch)
        return batches

    def _build_payload(
        self,
        document: Document,
        use_raw_text: bool,
        overrides: Optional[Dict[str, object]],
    ) -> Dict[str, Any]:
        """Build the payload for NameTag REST API request (multipart/form-data)."""
        payload: Dict[str, Any] = dict(self.extra_params)
        
        # Set model (default to multilingual if only language provided)
        if self.model:
            payload["model"] = self.model
        
        # Version is optional - only send if explicitly set to something other than "3"
        if self.version and self.version != "3":
            payload["version"] = self.version
        
        # Set input format and data
        # Always use conllu input when we have tokenized data (from piping or CoNLL-U input)
        # Only use untokenized when explicitly requested with use_raw_text=True
        if use_raw_text:
            # Raw text mode: send plain text
            text_payload = _document_to_plain_text(document)
            payload["data"] = (None, text_payload, "text/plain")
            payload["input"] = "untokenized"
        else:
            # Tokenized mode: send CoNLL-U as file-like object (matches working curl command)
            conllu_data = document_to_conllu(document)
            payload["data"] = (None, conllu_data, "text/plain")
            payload["input"] = "conllu"
        
        # Set output format - we want CoNLL-U+NE to extract entities (supports overlapping entities)
        # Note: NameTag API may use different format names, try conllu-ne first, fall back to conllu
        payload["output"] = "conllu-ne"
        
        if overrides:
            for key, value in overrides.items():
                payload[str(key)] = "" if value is None else str(value)

        return payload

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        """Send POST request using multipart/form-data (like curl -F)."""
        # Separate file fields from regular form fields
        files = {}
        data = {}
        
        for key, value in payload.items():
            if isinstance(value, tuple) and len(value) == 3:
                # File-like field: (filename, content, content_type)
                files[key] = value
            else:
                data[key] = value
        
        if self.log_requests:
            print("[nametag] Example curl command (multipart/form-data):")
            curl_parts = [f"curl -X POST \"{self.endpoint_url}\""]
            for key, value in data.items():
                curl_parts.append(f"-F \"{key}={value}\"")
            if files:
                for key, file_tuple in files.items():
                    curl_parts.append(f"-F \"{key}=<-;type=text/plain\"")
            print(" \\\n  ".join(curl_parts))
            if files:
                for key, file_tuple in files.items():
                    content_preview = file_tuple[1][:500] + ("…" if len(file_tuple[1]) > 500 else "")
                    print(f"[nametag] Data content (first 500 chars): {content_preview}")
        
        try:
            response = self.session.post(
                self.endpoint_url,
                data=data,
                files=files if files else None,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            err_msg = self._extract_error_from_response(exc.response) if exc.response is not None else ""
            details = f": {err_msg}" if err_msg else ""
            raise RuntimeError(f"NameTag REST returned HTTP {exc.response.status_code}{details}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"NameTag REST request failed: {exc}") from exc
    
    def _document_to_vertical(self, conllu_text: str) -> str:
        """Convert CoNLL-U text to vertical format (one token per line, empty lines between sentences)."""
        from ..conllu import conllu_to_document
        doc = conllu_to_document(conllu_text)
        lines = []
        for sent in doc.sentences:
            for tok in sent.tokens:
                lines.append(tok.form)
            lines.append("")  # Empty line between sentences
        return "\n".join(lines)

    def _parse_response(self, response: requests.Response, original_doc: Document) -> Document:
        """
        Parse NameTag REST response and extract entities.
        
        NameTag can return CoNLL-U+NE format (CoNLL-U with NER annotations in MISC field)
        or JSON format with a result field. Also extracts model and license info from response.
        """
        content_type = (response.headers.get("Content-Type") or "").lower()
        text = response.text
        doc = None
        
        # Try to parse as CoNLL-U+NE first
        if "conllu" in content_type or text.strip().startswith("#") or "\t" in text:
            doc = self._parse_conllu_ne(text, original_doc)
        # Try JSON response (may contain CoNLL-U in result field)
        elif "application/json" in content_type or text.strip().startswith("{"):
            try:
                data = response.json()
                # Extract model and license from JSON response
                doc = Document(id=original_doc.id or "nametag", meta=dict(original_doc.meta))
                if hasattr(original_doc, 'attrs'):
                    doc.attrs = dict(original_doc.attrs)
                if hasattr(original_doc, 'spans') and original_doc.spans:
                    if isinstance(original_doc.spans, list):
                        doc.spans = original_doc.spans[:]
                    elif hasattr(original_doc.spans, 'copy'):
                        doc.spans = original_doc.spans.copy()
                    else:
                        doc.spans = list(original_doc.spans) if original_doc.spans else []
                
                if "model" in data:
                    # Add nametag_model as file-level attribute (before newdoc)
                    if "_file_level_attrs" not in doc.meta:
                        doc.meta["_file_level_attrs"] = {}
                    doc.meta["_file_level_attrs"]["nametag_model"] = data["model"]
                    # Also add to attrs for backward compatibility, but writer will prioritize file_level
                    doc.attrs["nametag_model"] = data["model"]
                if "acknowledgements" in data:
                    doc.meta["_api_acknowledgements"] = data["acknowledgements"]
                # Look for result field
                if "result" in data and isinstance(data["result"], str):
                    result_text = data["result"]
                    # Check if result is CoNLL-U or XML
                    if result_text.strip().startswith("#") or "\t" in result_text:
                        # CoNLL-U format
                        parsed = self._parse_conllu_ne(result_text, original_doc)
                    elif "<sentence>" in result_text or "<token>" in result_text:
                        # XML format
                        parsed = self._parse_xml_response(result_text, original_doc)
                    else:
                        # Try CoNLL-U first, fall back to XML
                        try:
                            parsed = self._parse_conllu_ne(result_text, original_doc)
                        except:
                            parsed = self._parse_xml_response(result_text, original_doc)
                    
                    # Merge sentences, attrs, meta, and spans
                    doc.sentences = parsed.sentences
                    if hasattr(parsed, 'attrs'):
                        doc.attrs.update(parsed.attrs)
                    if parsed.meta:
                        # Preserve file-level attributes from parsed document first (before general meta update)
                        if "_file_level_attrs" in parsed.meta:
                            if "_file_level_attrs" not in doc.meta:
                                doc.meta["_file_level_attrs"] = {}
                            doc.meta["_file_level_attrs"].update(parsed.meta["_file_level_attrs"])
                        # Then do general meta update (this won't overwrite _file_level_attrs since we already merged it)
                        doc.meta.update(parsed.meta)
                    if hasattr(parsed, 'spans') and parsed.spans:
                        if isinstance(doc.spans, dict) and isinstance(parsed.spans, dict):
                            doc.spans.update(parsed.spans)
                        elif isinstance(doc.spans, list) and isinstance(parsed.spans, list):
                            doc.spans.extend(parsed.spans)
                        elif not doc.spans:
                            doc.spans = parsed.spans if isinstance(parsed.spans, (dict, list)) else list(parsed.spans) if parsed.spans else []
            except (ValueError, KeyError):
                pass
        
        if doc is None:
            raise RuntimeError("NameTag REST response format not recognized. Expected CoNLL-U+NE or JSON.")
        
        return doc

    def _parse_conllu_ne(self, conllu_text: str, original_doc: Document) -> Document:
        """
        Parse CoNLL-U+NE format and extract entities.
        
        CoNLL-U+NE format supports two annotation styles:
        - NE=ORG_3: NameTag format where all tokens with the same NE=LABEL_ID belong to the same entity
        - Entity=B-PER, Entity=I-PER: IOB format (B=beginning, I=inside, O=outside)
        
        Also preserves document-level attributes (like nametag_model, nametag_model_licence)
        from CoNLL-U headers, which are extracted by conllu_to_document.
        """
        from ..conllu import conllu_to_document
        
        # Parse the CoNLL-U (this will extract nametag_* headers into doc.attrs)
        # This also parses #newpar markers and creates paragraph spans
        doc = conllu_to_document(conllu_text, doc_id=original_doc.id or "nametag")
        
        # Preserve original document's attrs and meta
        if hasattr(original_doc, 'attrs'):
            for key, value in original_doc.attrs.items():
                if key not in doc.attrs:
                    doc.attrs[key] = value
        if original_doc.meta:
            # Save parsed document's file-level attrs before general meta update
            parsed_file_level = doc.meta.get("_file_level_attrs", {}).copy()
            # Do general meta update (this might overwrite _file_level_attrs)
            doc.meta.update(original_doc.meta)
            # Restore and merge file-level attrs (parsed document takes priority)
            if "_file_level_attrs" not in doc.meta:
                doc.meta["_file_level_attrs"] = {}
            # First add original's file-level attrs
            if "_file_level_attrs" in original_doc.meta:
                for key, value in original_doc.meta["_file_level_attrs"].items():
                    if key not in doc.meta["_file_level_attrs"]:
                        doc.meta["_file_level_attrs"][key] = value
            # Then add parsed's file-level attrs (takes priority)
            doc.meta["_file_level_attrs"].update(parsed_file_level)
        
        # Preserve spans from original_doc, but DON'T overwrite spans parsed from CoNLL-U
        # The CoNLL-U might have #newpar markers that create paragraph spans, which we want to keep
        # Only merge spans from original_doc if they don't conflict with parsed spans
        if hasattr(original_doc, 'spans') and original_doc.spans:
            if isinstance(original_doc.spans, dict):
                # Merge spans: keep spans from CoNLL-U parsing, add any from original_doc that aren't already present
                for layer, spans_list in original_doc.spans.items():
                    if layer not in doc.spans:
                        # Layer doesn't exist in parsed doc, add all spans from original
                        doc.spans[layer] = list(spans_list)
                    else:
                        # Layer exists - merge spans, avoiding duplicates
                        existing_spans = {(s.start, s.end, s.label) for s in doc.spans[layer]}
                        for span in spans_list:
                            key = (span.start, span.end, span.label)
                            if key not in existing_spans:
                                doc.spans[layer].append(span)
            else:
                # Fallback for non-dict spans - only use if parsed doc has no spans
                if not doc.spans:
                    doc.spans = dict(original_doc.spans) if original_doc.spans else {}
        
        # Extract entities from MISC field and clean up duplicates
        for sent in doc.sentences:
            entities: List[Entity] = []
            
            # First pass: deduplicate MISC entries and collect NE= annotations
            token_ne_map: Dict[int, str] = {}  # token_id -> NE=LABEL_ID
            token_entity_map: Dict[int, str] = {}  # token_id -> Entity=IOB
            
            for token in sent.tokens:
                misc = getattr(token, "misc", "") or ""
                if not misc or misc == "_":
                    continue
                
                # Parse MISC field and deduplicate
                misc_parts = misc.split("|")
                seen_keys: set[str] = set()
                cleaned_parts: list[str] = []
                
                for part in misc_parts:
                    if not part:
                        continue
                    # Extract key (part before =) for deduplication
                    if "=" in part:
                        key = part.split("=", 1)[0]
                    else:
                        key = part
                    
                    # Deduplicate: keep first occurrence of each key
                    if key not in seen_keys:
                        seen_keys.add(key)
                        cleaned_parts.append(part)
                        
                        # Collect entity annotations
                        if part.startswith("NE="):
                            token_ne_map[token.id] = part[3:]  # Store "ORG_3"
                        elif part.startswith("Entity="):
                            token_entity_map[token.id] = part[7:]  # Store "B-PER" or "I-PER"
                
                # Update token.misc with deduplicated entries, but remove NE= entries
                # (we'll extract them into Entity objects, so we don't need them in MISC)
                final_parts = [p for p in cleaned_parts if not p.startswith("NE=")]
                if final_parts:
                    token.misc = "|".join(final_parts)
                else:
                    token.misc = "_"
            
            # Second pass: extract entities
            # Priority: NE= format (for overlapping entities), fall back to Entity= format
            if token_ne_map:
                # Group tokens by NE=LABEL_ID
                ne_groups: Dict[str, List[int]] = {}  # "ORG_3" -> [token_ids]
                for token_id, ne_value in token_ne_map.items():
                    if ne_value not in ne_groups:
                        ne_groups[ne_value] = []
                    ne_groups[ne_value].append(token_id)
                
                # Create entities from groups
                for ne_value, token_ids in ne_groups.items():
                    if not token_ids:
                        continue
                    token_ids.sort()  # Ensure tokens are in order
                    # Extract label from "ORG_3" -> "ORG"
                    if "_" in ne_value:
                        label = ne_value.split("_", 1)[0]
                    else:
                        label = ne_value
                    
                    entities.append(Entity(
                        start=min(token_ids),
                        end=max(token_ids),
                        label=label,
                        text="",  # Will be filled below
                    ))
            elif token_entity_map:
                # Use IOB format (Entity=B-PER, Entity=I-PER)
                current_entity: Optional[tuple[int, str]] = None  # (start_token_id, label)
                
                for token in sent.tokens:
                    entity_tag = token_entity_map.get(token.id)
                    
                    if entity_tag:
                        if entity_tag.startswith("B-"):
                            # Begin new entity
                            if current_entity:
                                entities.append(Entity(
                                    start=current_entity[0],
                                    end=token.id - 1,
                                    label=current_entity[1],
                                    text="",
                                ))
                            label = entity_tag[2:]  # Remove "B-" prefix
                            current_entity = (token.id, label)
                        elif entity_tag.startswith("I-"):
                            # Continue entity
                            label = entity_tag[2:]  # Remove "I-" prefix
                            if current_entity and current_entity[1] == label:
                                # Continue same entity
                                pass
                            else:
                                # Mismatch - start new entity
                                if current_entity:
                                    entities.append(Entity(
                                        start=current_entity[0],
                                        end=token.id - 1,
                                        label=current_entity[1],
                                        text="",
                                    ))
                                current_entity = (token.id, label)
                        elif entity_tag == "O":
                            # Outside entity
                            if current_entity:
                                entities.append(Entity(
                                    start=current_entity[0],
                                    end=token.id - 1,
                                    label=current_entity[1],
                                    text="",
                                ))
                                current_entity = None
                    else:
                        # No entity tag - end current entity if any
                        if current_entity:
                            entities.append(Entity(
                                start=current_entity[0],
                                end=token.id - 1,
                                label=current_entity[1],
                                text="",
                            ))
                            current_entity = None
                
                # End any remaining entity (only if the last token is part of the entity)
                if current_entity and sent.tokens:
                    last_token = sent.tokens[-1]
                    # Check if the last token has an entity tag that continues the current entity
                    last_token_entity_tag = token_entity_map.get(last_token.id)
                    if last_token_entity_tag and (
                        (last_token_entity_tag.startswith("I-") and last_token_entity_tag[2:] == current_entity[1]) or
                        (last_token_entity_tag.startswith("B-") and last_token_entity_tag[2:] == current_entity[1])
                    ):
                        # Last token is part of the entity
                        entities.append(Entity(
                            start=current_entity[0],
                            end=last_token.id,
                            label=current_entity[1],
                            text="",
                        ))
                    else:
                        # Last token is not part of the entity, entity ended at previous token
                        if last_token.id > current_entity[0]:
                            entities.append(Entity(
                                start=current_entity[0],
                                end=last_token.id - 1,
                                label=current_entity[1],
                                text="",
                            ))
            
            # Fill entity text by extracting from sentence text at token positions
            # This preserves the exact spacing as it appears in the original text
            for entity in entities:
                entity_tokens = [tok for tok in sent.tokens if entity.start <= tok.id <= entity.end]
                if not entity_tokens:
                    entity.text = ""
                    continue
                
                # Try to extract from sentence text by finding token positions
                if sent.text:
                    first_token = entity_tokens[0]
                    last_token = entity_tokens[-1]
                    
                    # Find first token in sentence text
                    first_pos = sent.text.find(first_token.form)
                    if first_pos >= 0:
                        # Find last token in sentence text (search from first_pos to avoid false matches)
                        search_start = first_pos + len(first_token.form)
                        last_pos = sent.text.find(last_token.form, search_start)
                        if last_pos >= 0:
                            # Extract text from first token start to last token end
                            entity.text = sent.text[first_pos:last_pos + len(last_token.form)]
                        else:
                            # Fallback: concatenate token forms
                            entity.text = "".join(tok.form for tok in entity_tokens)
                    else:
                        # Fallback: concatenate token forms
                        entity.text = "".join(tok.form for tok in entity_tokens)
                else:
                    # No sentence text available - concatenate token forms
                    entity.text = "".join(tok.form for tok in entity_tokens)
            
            sent.entities = entities
        
        return doc

    def _extract_error_from_response(self, response: requests.Response) -> str:
        """Attempt to extract an error message from a NameTag REST response."""
        if response is None:
            return ""
        content_type = (response.headers.get("Content-Type") or "").lower()
        text = response.text
        if "application/json" in content_type or text.strip().startswith("{"):
            try:
                data = response.json()
            except ValueError:
                return text.strip()
            for key in ("error", "message"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()
            return text.strip()
        return text.strip()

    def train(  # pragma: no cover - NameTag REST backend cannot train
        self,
        train_data,
        output_dir,
        *,
        dev_data=None,
        **kwargs,
    ):
        raise NotImplementedError("NameTag REST backend does not support training")


def _create_nametag_backend(
    *,
    endpoint_url: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    version: str = "3",
    timeout: float = 30.0,
    batch_size: int = 50,
    extra_params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[requests.Session] = None,
    log_requests: bool = False,
    **kwargs: Any,
) -> NameTagRESTBackend:
    """Instantiate the NameTag REST backend."""
    from ..backend_utils import validate_backend_kwargs
    
    validate_backend_kwargs(kwargs, "NameTag", allowed_extra=["download_model", "training"])
    
    return NameTagRESTBackend(
        endpoint_url=endpoint_url,
        model=model,
        language=language,
        version=version,
        timeout=timeout,
        batch_size=batch_size,
        extra_params=extra_params,
        headers=headers,
        session=session,
        log_requests=log_requests,
    )


BACKEND_SPEC = BackendSpec(
    name="nametag",
    description="NameTag - REST service for named entity recognition (NER)",
    factory=_create_nametag_backend,
    get_model_entries=get_nametag_model_entries,
    list_models=list_nametag_models_display,
    supports_training=False,
    is_rest=True,
)

