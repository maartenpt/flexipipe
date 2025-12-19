"""Backend spec and implementation for the UDMorph REST backend."""

from __future__ import annotations

import time
import urllib.parse
from typing import Any, Dict, Optional

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document, document_to_conllu, parse_conllu_from_backend
from ..doc import Document
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_storage import read_model_cache_entry, write_model_cache_entry
from ..model_registry import get_remote_models_for_backend
from ..neural_backend import BackendManager, NeuralResult

try:
    import requests
except ImportError as exc:  # pragma: no cover - import error handling
    raise ImportError(
        "UDMorph REST backend requires the 'requests' package. "
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


BASE_UDMORPH_ROOT = "https://lindat.mff.cuni.cz/services/teitok-live/udmorph/"


def list_udmorph_models(url: str = "https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=list") -> Dict:
    """
    Fetch the list of available UDMorph models from the service.
    
    Returns a dictionary mapping model keys to model information.
    """
    try:
        response = requests.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch UDMorph model list: {exc}") from exc


MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def get_udmorph_model_entries(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    # 1) Prefer curated registry from flexipipe-models (registries/udmorph.json)
    #    so that we can maintain additional metadata (axes, normalization, etc.)
    curated_entries: Dict[str, Dict[str, Any]] = {}
    try:
        remote_models = get_remote_models_for_backend(
            "udmorph",
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=verbose,
        )
        for model_entry in remote_models:
            model_key = model_entry.get("model") or model_entry.get("key")
            if not model_key:
                continue
            language_code = (
                model_entry.get(LANGUAGE_FIELD_ISO)
                or model_entry.get("language_iso")
                or model_entry.get("iso")
            )
            language_name = (
                model_entry.get(LANGUAGE_FIELD_NAME)
                or model_entry.get("language_name")
                or model_entry.get("name")
            )
            features = model_entry.get("features") or model_entry.get("feats") or "unknown"
            name = language_name or model_key

            entry = build_model_entry(
                "udmorph",
                model_key,
                language_code=language_code,
                language_name=language_name,
                features=features,
                name=name,
            )

            # Endpoint URL (may be stored as endpoint_url, url, or src)
            endpoint_override = (
                model_entry.get("endpoint_url")
                or model_entry.get("url")
                or model_entry.get("src")
            )
            if endpoint_override:
                if not endpoint_override.startswith(("http://", "https://")):
                    endpoint_override = urllib.parse.urljoin(
                        BASE_UDMORPH_ROOT, endpoint_override
                    )
                # Parse URL and remove query parameters that should be POST data
                # (like &model=... and &text={text} which are templates)
                parsed = urllib.parse.urlparse(endpoint_override)
                # Keep only action-related query params, remove model/text params
                query_params = urllib.parse.parse_qs(parsed.query)
                # Remove model and text from query (these are sent as POST data)
                query_params.pop("model", None)
                query_params.pop("text", None)
                # Rebuild query string with remaining params
                clean_query = urllib.parse.urlencode(query_params, doseq=True)
                # Reconstruct URL without model/text params
                clean_endpoint = urllib.parse.urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    clean_query,
                    parsed.fragment
                ))
                entry["endpoint_url"] = clean_endpoint

            # Preserve additional metadata (e.g., unicode_normalization, post_spec, source, etc.)
            for extra_key in (
                "unicode_normalization",
                "post_spec",
                "source",
                "backend_version",
                "description",
            ):
                if extra_key in model_entry:
                    entry[extra_key] = model_entry[extra_key]

            curated_entries[model_key] = entry
    except Exception:
        # If anything goes wrong with the curated registry, fall back to live list
        curated_entries = {}

    if curated_entries:
        return curated_entries

    # 2) Fallback to live list from the UDMorph service (previous behaviour)
    if url is None:
        url = "https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=list"
    elif "?action=tag" in url:
        url = url.replace("?action=tag&act=tag", "?action=tag&act=list")

    cache_key = f"udmorph:{url}"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            filtered: Dict[str, Dict[str, Any]] = {}
            needs_refresh = False
            for model_key, entry in cached.items():
                tagger = str(entry.get("tagger", "")).upper()
                if not tagger:
                    needs_refresh = True
                    break
                if tagger == "UDPIPE2":
                    continue
                filtered[model_key] = entry
            if not needs_refresh:
                if verbose:
                    print("[flexipipe] Using cached UDMorph model list (use --refresh-cache to update).")
                return filtered

    if verbose:
        print(f"[flexipipe] Fetching UDMorph models from {url}...")
    models = list_udmorph_models(url)
    prepared_models: Dict[str, Dict[str, Any]] = {}
    for model_key, model_info in models.items():
        if not isinstance(model_info, dict):
            continue
        if str(model_info.get("tagger", "")).upper() == "UDPIPE2":
            # Skip UDPIPE2 proxies; users should use the udpipe backend directly
            continue
        name = model_info.get("name", "")
        iso = model_info.get("iso", "")
        feats_str = model_info.get("feats", "")
        features = ", ".join(f.strip() for f in feats_str.split(",")) if feats_str else "unknown"
        entry = build_model_entry(
            "udmorph",
            model_key,
            language_code=iso,
            language_name=name,
            features=features,
            name=name,
        )
        endpoint_override = model_info.get("src") or model_info.get("url")
        if endpoint_override:
            # Older registry entries sometimes store only a relative path like:
            #   "index.php?action=tag&act=tag&model=udpipe:kab&text={text}"
            # Normalize these to the public UDMorph root so the URL is usable.
            if not endpoint_override.startswith(("http://", "https://")):
                endpoint_override = urllib.parse.urljoin(
                    BASE_UDMORPH_ROOT, endpoint_override
                )
            # Parse URL and remove query parameters that should be POST data
            # (like &model=... and &text={text} which are templates)
            parsed = urllib.parse.urlparse(endpoint_override)
            # Keep only action-related query params, remove model/text params
            query_params = urllib.parse.parse_qs(parsed.query)
            # Remove model and text from query (these are sent as POST data)
            query_params.pop("model", None)
            query_params.pop("text", None)
            # Rebuild query string with remaining params
            clean_query = urllib.parse.urlencode(query_params, doseq=True)
            # Reconstruct URL without model/text params
            clean_endpoint = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                parsed.fragment
            ))
            entry["endpoint_url"] = clean_endpoint
        entry["tagger"] = model_info.get("tagger")
        entry["post_spec"] = model_info.get("post", "")
        prepared_models[model_key] = entry
    # Only write to cache if refresh_cache is True (explicit refresh)
    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, prepared_models)
        except (OSError, PermissionError):
            # If we can't write cache, that's okay - we'll just return the entries without caching
            pass
    return prepared_models


def get_udmorph_model_entry(model_key: str) -> Optional[Dict[str, Any]]:
    entries = get_udmorph_model_entries(use_cache=True, refresh_cache=False, verbose=False)
    return entries.get(model_key)


def list_udmorph_models_display(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> int:
    """
    List available UDMorph models with formatted output.
    Prints formatted output and returns exit code (0 for success, 1 for error).
    """
    try:
        prepared_models = get_udmorph_model_entries(
            url,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=True,
        )
        
        print(f"\nAvailable UDMorph models:")
        print(f"{'Model Key':<40} {'ISO':<8} {'Language':<20} {'Features':<30}")
        print("=" * 110)
        
        sorted_items = sorted(
            prepared_models.items(),
            key=lambda x: (
                x[1].get(LANGUAGE_FIELD_ISO) or x[1].get(LANGUAGE_FIELD_NAME) or "",
                x[0]
            )
        )
        
        for model_key, entry in sorted_items:
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            features = entry.get("features", "unknown")
            print(f"{model_key:<40} {lang_iso:<8} {lang_display:<20} {features:<30}")
        
        total_models = len(prepared_models)
        unique_languages = {
            entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
            for entry in prepared_models.values()
            if entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
        }
        print(f"\nTotal: {total_models} model(s) for {len(unique_languages)} language(s)")
        
        return 0
    except Exception as e:
        print(f"Error fetching UDMorph models: {e}")
        import traceback
        traceback.print_exc()
        return 1


class UDMorphRESTBackend(BackendManager):
    """Neural backend that delegates tagging to a UDMorph REST service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        model: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 50,
        extra_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
        log_requests: bool = False,
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for UDMorph REST backend")
        # Validate that endpoint_url has a scheme (http:// or https://)
        if not endpoint_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid UDMorph endpoint URL '{endpoint_url}': "
                "URL must start with http:// or https://"
            )
        # Warn if endpoint looks like a GitHub URL (likely wrong)
        if "github" in endpoint_url.lower() or "github.io" in endpoint_url.lower():
            import warnings
            warnings.warn(
                f"UDMorph endpoint URL appears to be a GitHub URL: {endpoint_url}. "
                "This is likely incorrect. UDMorph should use the Lindat service endpoint.",
                UserWarning
            )
        self.endpoint_url = endpoint_url
        if not model:
            raise ValueError("UDMorph REST backend requires a model name. Provide --udmorph-model.")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))  # ensure at least 1 sentence per batch
        self.extra_params = extra_params or {}
        self.headers = headers or {}
        self.session = session or requests.Session()
        self.log_requests = log_requests
        self._model_name = model or endpoint_url
        self._backend_name = "udmorph"

    @property
    def _backend_info(self) -> str:
        """Used by CLI to describe this backend."""
        model_name = self.model or "unknown"
        return f"udmorph: {model_name}"

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
        aggregated_doc = Document(id=document.id, meta=dict(document.meta))
        total_tokens = 0

        for batch_index, batch_doc in enumerate(batches):
            payload = self._build_payload(batch_doc, use_raw_text, overrides)
            if self.log_requests:
                print(f"[udmorph] POST {self.endpoint_url} (batch {batch_index + 1}/{len(batches)})")
                if batch_index == 0:
                    # Build curl command that can be used to replicate the request
                    curl_parts = [f"curl -X POST \"{self.endpoint_url}\""]
                    if self.headers:
                        for key, value in self.headers.items():
                            curl_parts.append(f"-H \"{key}: {value}\"")
                    # Add Content-Type header if not already present
                    if not any(h.startswith("-H \"Content-Type:") for h in curl_parts):
                        curl_parts.append("-H \"Content-Type: application/x-www-form-urlencoded\"")
                    for key, value in payload.items():
                        # URL-encode the value properly
                        encoded = urllib.parse.quote_plus(str(value))
                        curl_parts.append(f"-d \"{key}={encoded}\"")
                    print("[udmorph] Curl command to replicate this request:")
                    print(" \\\n  ".join(curl_parts))
                    # Also show the full URL with query parameters if any
                    if "?" in self.endpoint_url:
                        print(f"[udmorph] Full URL: {self.endpoint_url}")
                    # Show payload details
                    print(f"[udmorph] Payload keys: {list(payload.keys())}")
                    for key, value in payload.items():
                        value_preview = str(value)[:200] + "…" if len(str(value)) > 200 else str(value)
                        print(f"[udmorph]   {key}: {value_preview}")
                else:
                    preview = {k: (v if len(str(v)) < 200 else str(v)[:200] + "…") for k, v in payload.items()}
                    print(f"[udmorph] Payload preview: {preview}")

            request_start = time.time()
            response = self._post(payload)
            request_elapsed = time.time() - request_start

            first_result_chunk: Optional[str] = None
            if self.log_requests:
                snippet = response.text[:400] + ("…" if len(response.text) > 400 else "")
                print(f"[udmorph] Response status: {response.status_code}")
                print(f"[udmorph] Response snippet: {snippet}")
                if batch_index == 0:
                    try:
                        json_payload = response.json()
                        first_result_chunk = json_payload.get("result", "")
                    except ValueError:
                        first_result_chunk = None
                print(f"[udmorph] Request duration: {request_elapsed:.2f}s")

            conllu_output = self._extract_conllu_from_response(response)
            if not conllu_output.strip():
                error_msg = "UDMorph REST response did not contain any data"
                if self.log_requests:
                    error_msg += f"\n[udmorph] URL: {self.endpoint_url}"
                    error_msg += f"\n[udmorph] Response status: {response.status_code}"
                    error_msg += f"\n[udmorph] Response headers: {dict(response.headers)}"
                    error_msg += f"\n[udmorph] Response text (first 500 chars): {response.text[:500]}"
                raise RuntimeError(error_msg)

            chunk_doc = parse_conllu_from_backend(conllu_output, document, doc_id=batch_doc.id or "udmorph")
            if self.log_requests and batch_index == 0 and chunk_doc.sentences:
                first_sentence = chunk_doc.sentences[0]
                token_forms = " ".join(tok.form for tok in first_sentence.tokens)
                print("[udmorph] First sentence returned (decoded):")
                print(f"  sent_id={first_sentence.sent_id or first_sentence.id or 'n/a'}")
                print(f"  text={first_sentence.text or token_forms}")
                print(f"  tokens={token_forms}")
                if first_result_chunk:
                    print("[udmorph] Full first result (de-escaped):")
                    print(first_result_chunk.strip())
            aggregated_doc.sentences.extend(chunk_doc.sentences)
            total_tokens += sum(len(sent.tokens) for sent in chunk_doc.sentences)

        elapsed = time.time() - start
        stats = {
            "elapsed_seconds": elapsed,
            "backend": "udmorph-rest",
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
    ) -> Dict[str, str]:
        """
        Build the payload for UDMorph REST API.
        
        UDMorph uses a simpler format: data:text,model:model
        For raw text, send the text as 'data' and model name as 'model'.
        For tokenized input, send CoNLL-U as 'data' and model name as 'model'.
        """
        payload: Dict[str, str] = dict(self.extra_params)
        
        # Model is always required
        if self.model:
            payload["model"] = self.model

        if use_raw_text:
            text_payload = _document_to_plain_text(document)
            if not text_payload.strip():
                # Fallback to CoNLL-U if no text available
                payload["data"] = document_to_conllu(document)
            else:
                payload["data"] = text_payload
        else:
            # Tokenized mode: send CoNLL-U format
            payload["data"] = document_to_conllu(document)

        if overrides:
            for key, value in overrides.items():
                payload[str(key)] = "" if value is None else str(value)

        return payload

    def _post(self, payload: Dict[str, str]) -> requests.Response:
        try:
            response = self.session.post(
                self.endpoint_url,
                data=payload,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,  # Explicitly allow redirects (default, but make it clear)
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            err_msg = self._extract_error_from_response(exc.response) if exc.response is not None else ""
            details = f": {err_msg}" if err_msg else ""
            status_code = exc.response.status_code if exc.response else "unknown"
            # Show the actual URL that was hit (might differ from endpoint_url if redirected)
            actual_url = exc.response.url if exc.response else self.endpoint_url
            url_info = ""
            if actual_url != self.endpoint_url:
                url_info = f" (redirected to: {actual_url})"
            elif "github" in actual_url.lower() or "github.io" in actual_url.lower():
                url_info = f" (WARNING: endpoint appears to be a GitHub URL: {actual_url})"
            
            # Provide more context for common error codes
            if status_code == 422:
                model_info = f" (model: {self.model})" if self.model else ""
                raise RuntimeError(
                    f"UDMorph REST returned HTTP 422 Unprocessable Entity{model_info}{url_info}. "
                    f"This usually means the request format is invalid or the model name is incorrect.{details}"
                ) from exc
            raise RuntimeError(f"UDMorph REST returned HTTP {status_code}{url_info}{details}") from exc
        except requests.RequestException as exc:
            # Extract a user-friendly error message without exposing full URLs
            error_msg = str(exc)
            # Remove URL details from common error messages
            if "No connection adapters were found" in error_msg:
                # This usually means the URL is malformed (missing scheme/base URL)
                error_msg = "Invalid endpoint URL (missing base URL or scheme)"
            elif "Connection" in error_msg or "timeout" in error_msg.lower():
                # Network-related errors - keep the error type but remove URL
                error_msg = error_msg.split(":")[0] if ":" in error_msg else error_msg
            elif "HTTPSConnectionPool" in error_msg or "HTTPConnectionPool" in error_msg:
                # Connection pool errors - extract just the error type
                parts = error_msg.split(":")
                if len(parts) > 1:
                    error_msg = parts[-1].strip()
            raise RuntimeError(f"UDMorph REST request failed: {error_msg}") from exc

    def _extract_conllu_from_response(self, response: requests.Response) -> str:
        """Extract CoNLL-U text from a UDMorph REST response."""
        content_type = (response.headers.get("Content-Type") or "").lower()
        text = response.text
        if "application/json" in content_type or text.strip().startswith("{"):
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(f"UDMorph REST returned invalid JSON: {exc}") from exc
            for key in ("result", "conllu", "data"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            raise RuntimeError(
                "UDMorph REST JSON response does not contain 'result', 'conllu', or 'data' fields"
            )
        return text

    def _extract_error_from_response(self, response: requests.Response) -> str:
        """Attempt to extract an error message from a UDMorph REST response."""
        if response is None:
            return ""
        content_type = (response.headers.get("Content-Type") or "").lower()
        text = response.text
        
        # Handle JSON responses
        if "application/json" in content_type or text.strip().startswith("{"):
            try:
                data = response.json()
            except ValueError:
                return text.strip()
            for key in ("error", "message"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()
            return text.strip()
        
        # Handle HTML error pages (e.g., GitHub Pages 404)
        if "text/html" in content_type or text.strip().startswith("<"):
            # Try to extract a meaningful error message from HTML
            import re
            # Look for common error page patterns
            title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = re.sub(r'\s+', ' ', title_match.group(1)).strip()
                if title and title not in ("Error", "404", "Not Found"):
                    return f"HTML error page: {title}"
            # Look for error messages in common HTML error pages
            error_match = re.search(r'<h1[^>]*>(.*?)</h1>', text, re.IGNORECASE | re.DOTALL)
            if error_match:
                error_text = re.sub(r'<[^>]+>', '', error_match.group(1)).strip()
                if error_text:
                    return f"HTML error: {error_text[:200]}"
            # If we can't extract a meaningful message, indicate it's an HTML error page
            return "Server returned HTML error page (endpoint may be incorrect or service unavailable)"
        
        return text.strip()

    def train(  # pragma: no cover - UDMorph REST backend cannot train
        self,
        train_data,
        output_dir,
        *,
        dev_data=None,
        **kwargs,
    ):
        raise NotImplementedError("UDMorph REST backend does not support training")


def _create_udmorph_backend(
    *,
    endpoint_url: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    timeout: float = 30.0,
    batch_size: int = 50,
    extra_params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[requests.Session] = None,
    log_requests: bool = False,
    **kwargs: Any,
) -> UDMorphRESTBackend:
    """Instantiate the UDMorph REST backend."""
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "UDMorph", allowed_extra=["download_model", "training"])
    
    # If no model provided but language is, try to look up a model for that language
    resolved_model = model
    if not resolved_model and language:
        try:
            resolved_model = resolve_model_from_language(language, "udmorph", preferred_only=True, use_cache=True)
            if log_requests:
                print(f"[udmorph] Resolved model '{resolved_model}' for language '{language}'")
        except ValueError as exc:
            # No model found for language - raise a clearer error
            raise ValueError(f"UDMorph REST backend: No model found for language '{language}'. Provide --model to specify a model name.") from exc
    
    return UDMorphRESTBackend(
        endpoint_url=endpoint_url,
        model=resolved_model,
        timeout=timeout,
        batch_size=batch_size,
        extra_params=extra_params,
        headers=headers,
        session=session,
        log_requests=log_requests,
    )


BACKEND_SPEC = BackendSpec(
    name="udmorph",
    description="UDMorph - REST service for morphological tagging (no dependency parsing)",
    factory=_create_udmorph_backend,
    get_model_entries=get_udmorph_model_entries,
    list_models=list_udmorph_models_display,
    supports_training=False,
    is_rest=True,
    url="https://lindat.mff.cuni.cz/services/udmorph",
    install_instructions="udmorph is a REST service that requires no installation",
)

