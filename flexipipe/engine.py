from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .doc import Document, Sentence, Token, SubToken, Span, Entity
from .doc_utils import collect_span_entities_by_sentence

XML_NS = "{http://www.w3.org/XML/1998/namespace}"


def _ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_ensure_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

# Import insert_tokens_into_teitok implementation
from .insert_tokens import insert_tokens_into_teitok


def _sanitize_utf8(s: str) -> str:
    """Ensure a string is valid UTF-8 by replacing invalid bytes."""
    if not s:
        return s
    # If the string already contains invalid UTF-8 (surrogates), we need to handle it
    # First, try to encode as UTF-8 with error handling
    try:
        # Encode to bytes, replacing any problematic characters
        encoded = s.encode('utf-8', errors='replace')
        # Decode back, which should always succeed
        return encoded.decode('utf-8', errors='replace')
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        # Fallback: if encoding fails entirely, return empty string
        return ""


def _sanitize_document(doc: Document) -> Document:
    """Sanitize all string fields in a document to ensure valid UTF-8."""

    def _sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_attrs: Dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, str):
                sanitized_attrs[key] = _sanitize_utf8(value)
            else:
                sanitized_attrs[key] = _sanitize_utf8(str(value))
        return sanitized_attrs

    sanitized_meta: Dict[str, Any] = {}
    for key, value in doc.meta.items():
        if isinstance(value, str):
            sanitized_meta[key] = _sanitize_utf8(value)
        else:
            sanitized_meta[key] = value

    sanitized = Document(
        id=_sanitize_utf8(doc.id),
        meta=sanitized_meta,
        attrs=_sanitize_attrs(doc.attrs),
        spans={},
    )

    for layer, spans in doc.spans.items():
        for span in spans:
            sanitized.add_span(
                layer,
                Span(
                    label=_sanitize_utf8(span.label),
                    start=span.start,
                    end=span.end,
                    attrs=_sanitize_attrs(span.attrs),
                    char_start=span.char_start,
                    char_end=span.char_end,
                    byte_start=span.byte_start,
                    byte_end=span.byte_end,
                ),
            )

    span_entities = collect_span_entities_by_sentence(doc, "ner")

    for idx, sent in enumerate(doc.sentences):
        sanitized_sent = Sentence(
            id=_sanitize_utf8(sent.id),
            sent_id=_sanitize_utf8(sent.sent_id),
            text=_sanitize_utf8(sent.text),
            attrs=_sanitize_attrs(sent.attrs),
            source_id=_sanitize_utf8(sent.source_id),
            char_start=sent.char_start,
            char_end=sent.char_end,
            byte_start=sent.byte_start,
            byte_end=sent.byte_end,
            corr=_sanitize_utf8(sent.corr),
            tokens=[],
        )
        all_entities = list(sent.entities)
        extra_entities = span_entities.get(idx)
        if extra_entities:
            all_entities.extend(extra_entities)
        for ent in all_entities:
            sanitized_sent.entities.append(
                Entity(
                    start=ent.start,
                    end=ent.end,
                    label=_sanitize_utf8(ent.label),
                    text=_sanitize_utf8(ent.text),
                    attrs=_sanitize_attrs(ent.attrs),
                )
            )
        for tok in sent.tokens:
            sanitized_tok = Token(
                id=tok.id,
                form=_sanitize_utf8(tok.form),
                lemma=_sanitize_utf8(tok.lemma),
                xpos=_sanitize_utf8(tok.xpos),
                upos=_sanitize_utf8(tok.upos),
                feats=_sanitize_utf8(tok.feats),
                is_mwt=tok.is_mwt,
                mwt_start=tok.mwt_start,
                mwt_end=tok.mwt_end,
                parts=[_sanitize_utf8(p) for p in tok.parts],
                subtokens=[],
                source=_sanitize_utf8(tok.source),
                source_id=_sanitize_utf8(tok.source_id),
                head=tok.head,
                deprel=_sanitize_utf8(tok.deprel),
                deps=_sanitize_utf8(tok.deps),
                misc=_sanitize_utf8(tok.misc),
                space_after=tok.space_after,
                char_start=tok.char_start,
                char_end=tok.char_end,
                byte_start=tok.byte_start,
                byte_end=tok.byte_end,
                attrs=_sanitize_attrs(tok.attrs),
            )
            for sub in tok.subtokens:
                sanitized_sub = SubToken(
                    id=sub.id,
                    form=_sanitize_utf8(sub.form),
                    lemma=_sanitize_utf8(sub.lemma),
                    xpos=_sanitize_utf8(sub.xpos),
                    upos=_sanitize_utf8(sub.upos),
                    feats=_sanitize_utf8(sub.feats),
                    source=_sanitize_utf8(sub.source),
                    source_id=_sanitize_utf8(sub.source_id),
                    space_after=sub.space_after,
                    char_start=sub.char_start,
                    char_end=sub.char_end,
                    byte_start=sub.byte_start,
                    byte_end=sub.byte_end,
                    attrs=_sanitize_attrs(sub.attrs),
                )
                sanitized_tok.subtokens.append(sanitized_sub)
            sanitized_sent.tokens.append(sanitized_tok)
        sanitized.sentences.append(sanitized_sent)
    return sanitized


def assign_doc_id_from_path(doc: Document, path: Optional[str]) -> None:
    """Set doc.id to the basename (without extension) of the given path when appropriate.
    
    Skips setting doc.id if the path is a temporary file (e.g., /tmp/, /var/tmp/, /var/folders/).
    """
    if not path:
        return
    path_str = str(path).strip()
    if not path_str or path_str in {"-", "<inline-data>"}:
        return
    
    path_obj = Path(path_str)
    
    # Check if this is a temporary file path
    import tempfile
    try:
        temp_dir = Path(tempfile.gettempdir()).resolve()
        path_resolved = path_obj.resolve()
        # Check if the path is within the system temp directory
        try:
            is_temp_file = path_resolved.is_relative_to(temp_dir)
        except AttributeError:
            # Python < 3.9 compatibility
            is_temp_file = str(path_resolved).startswith(str(temp_dir))
    except (OSError, ValueError):
        # If we can't resolve paths, fall back to pattern matching
        path_lower = path_str.lower()
        temp_indicators = [
            "/tmp/",
            "/var/tmp/",
            "/var/folders/",  # macOS temp directories
            "\\temp\\",  # Windows temp
            "\\tmp\\",  # Windows alternative
        ]
        filename_lower = path_obj.name.lower()
        temp_prefixes = ["tmp", "temp"]
        is_temp_file = (
            any(indicator in path_lower for indicator in temp_indicators) or
            any(filename_lower.startswith(prefix) for prefix in temp_prefixes)
        )
    
    if is_temp_file:
        # Don't set doc.id from temporary file paths
        # Still set source_path in meta for reference, but don't use it as doc ID
        doc.meta.setdefault("source_path", str(path_obj))
        return
    
    doc.meta.setdefault("source_path", str(path_obj))

    candidates = {path_str, str(path_obj)}
    try:
        candidates.add(str(path_obj.resolve()))
    except OSError:
        pass
    candidates.add(path_obj.name)

    if not doc.id or doc.id in candidates:
        stem = path_obj.stem
        if stem:
            doc.id = stem


def _resolve_flexitag_params_path(params_file: str) -> str:
    """
    Accept either a direct params JSON file or a directory that contains model_vocab.json.
    """
    path = Path(params_file)
    if path.is_dir():
        candidate = path / "model_vocab.json"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"No model_vocab.json found inside flexitag model directory: {path}"
        )
    return str(path)

try:
    from flexitag_py import (
        FlexitagEngine,
        load_teitok as _load_teitok,
        save_teitok as _save_teitok,
        dump_teitok as _dump_teitok,
        tag_document as _tag_document,
    )
except ImportError:  # pragma: no cover - handled during runtime
    # Try adding the build directory to sys.path
    import sys
    from pathlib import Path
    import os
    
    # Look for flexitag_py in the flexitag/build directory relative to this file
    # Always resolve __file__ to an absolute path first
    engine_file = Path(__file__)
    if not engine_file.is_absolute():
        engine_file = Path.cwd() / engine_file
    engine_file = engine_file.resolve()
    
    flexipipe_dir = engine_file.parent.parent
    flexitag_build = flexipipe_dir / "flexitag" / "build"
    
    if flexitag_build.exists():
        build_path_str = str(flexitag_build.resolve())
        if build_path_str not in sys.path:
            sys.path.insert(0, build_path_str)
        try:
            from flexitag_py import (
                FlexitagEngine,
                load_teitok as _load_teitok,
                save_teitok as _save_teitok,
                dump_teitok as _dump_teitok,
                tag_document as _tag_document,
            )
        except ImportError as exc:
            FlexitagEngine = None  # type: ignore
            # Capture exc in closure to avoid NameError
            _import_error = exc

            def _missing_extension(*_: object, **__: object) -> None:
                raise RuntimeError(
                    "flexitag_py extension is not available. Build the flexitag project with "
                    "pybind11 support or ensure it is on PYTHONPATH."
                ) from _import_error

            _tag_document = _missing_extension  # type: ignore
            _load_teitok = _missing_extension  # type: ignore
            _save_teitok = _missing_extension  # type: ignore
            _dump_teitok = _missing_extension  # type: ignore
    else:
        FlexitagEngine = None  # type: ignore
        # Capture exc in closure to avoid NameError
        _import_error = ImportError(
            "flexitag_py extension is not available. Build the flexitag project with "
            "pybind11 support or ensure it is on PYTHONPATH."
        )

        def _missing_extension(*_: object, **__: object) -> None:
            raise RuntimeError(
                "flexitag_py extension is not available. Build the flexitag project with "
                "pybind11 support or ensure it is on PYTHONPATH."
            ) from _import_error

        _tag_document = _missing_extension  # type: ignore
        _load_teitok = _missing_extension  # type: ignore
        _save_teitok = _missing_extension  # type: ignore
        _dump_teitok = _missing_extension  # type: ignore


@dataclass
class FlexitagResult:
    document: Document
    stats: Dict[str, float]


def _infer_space_after_from_text(document: Document) -> None:
    """Infer space_after values from sentence text if available."""
    for sentence in document.sentences:
        if not sentence.text:
            continue
        
        text = sentence.text
        pos = 0
        
        for token_idx, token in enumerate(sentence.tokens):
            is_last_token = (token_idx == len(sentence.tokens) - 1)
            
            # Find the token form in the text
            idx = text.find(token.form, pos)
            if idx >= 0:
                end = idx + len(token.form)
                if end < len(text):
                    # Check if there's whitespace after this token
                    next_char = text[end]
                    token.space_after = next_char.isspace()
                else:
                    # Last token - set to None (no SpaceAfter entry in CoNLL-U)
                    token.space_after = None
                # Advance position past token and any whitespace
                pos = end
                while pos < len(text) and text[pos].isspace():
                    pos += 1
            else:
                # Token form not found in text - try to infer from position
                if is_last_token:
                    # Last token - set to None (no SpaceAfter entry in CoNLL-U)
                    token.space_after = None
                else:
                    # Not last token - default to True if not already set
                    if token.space_after is None:
                        token.space_after = True


def _apply_sentence_correspondence(document: Document) -> None:
    """
    Ensure sentences either use their inline tokens or fall back to @corresp references.
    """

    def _token_identifier(token: Token) -> str:
        return (
            token.tokid
            or token.get_attr("tokid")
            or token.attrs.get("id", "")
            or token.attrs.get("xml:id", "")
        )

    def _parse_corresp(value: str) -> List[str]:
        ids: List[str] = []
        for chunk in value.replace(",", " ").split():
            cleaned = chunk.strip()
            if not cleaned:
                continue
            if cleaned.startswith("#"):
                cleaned = cleaned[1:]
            if cleaned:
                ids.append(cleaned)
        return ids

    token_lookup: Dict[str, Token] = {}
    token_parents: Dict[str, Sentence] = {}

    for sentence in document.sentences:
        for token in sentence.tokens:
            tok_id = _token_identifier(token)
            if tok_id:
                token_lookup[tok_id] = token
                token_parents[tok_id] = sentence

    for sentence in document.sentences:
        corresp_value = sentence.get_attr("corresp")
        if not corresp_value:
            continue

        corresp_ids = _parse_corresp(corresp_value)
        if not corresp_ids:
            continue

        existing_ids = [_token_identifier(tok) for tok in sentence.tokens if _token_identifier(tok)]
        needs_rebuild = (not sentence.tokens) or any(cid not in existing_ids for cid in corresp_ids)

        if not needs_rebuild:
            continue

        rebuilt_tokens: List[Token] = []
        for cid in corresp_ids:
            tok = token_lookup.get(cid)
            if not tok:
                continue
            owner = token_parents.get(cid)
            if owner and owner is not sentence:
                owner.tokens = [t for t in owner.tokens if t is not tok]
            rebuilt_tokens.append(tok)
            token_parents[cid] = sentence

        if rebuilt_tokens:
            sentence.tokens = rebuilt_tokens


# TEITOK functions have been moved to teitok.py module
# Import them from there: from .teitok import load_teitok, save_teitok, dump_teitok, update_teitok



class FlexitagFallback:
    """Thin wrapper around the flexitag pybind11 bindings."""

    def __init__(
        self,
        params_file: str,
        *,
        options: Optional[Dict[str, object]] = None,
        debug: bool = False,
    ) -> None:
        resolved_params = _resolve_flexitag_params_path(params_file)
        if FlexitagEngine is None:
            _tag_document()  # raises informative error
        self._engine = FlexitagEngine(resolved_params, options or {})  # type: ignore
        self._params_file = resolved_params
        self._debug = debug
        if self._debug:
            print(f"[flexipipe] FlexitagFallback initialised with params={params_file}")

    def tag(self, document: Document, *, overrides: Optional[Dict[str, object]] = None) -> FlexitagResult:
        # Sanitize input document
        sanitized_input = _sanitize_document(document)
        doc_dict = sanitized_input.to_dict()
        if self._debug:
            sent_count = len(document.sentences)
            tok_count = sum(len(sent.tokens) for sent in document.sentences)
            print(
                f"[flexipipe] flexitag tagging start: sentences={sent_count} tokens={tok_count} "
                f"overrides={overrides or {}}"
            )
        try:
            tagged_doc, stats = self._engine.tag(doc_dict, overrides or {})  # type: ignore
        except UnicodeDecodeError as exc:
            # If we get a Unicode error, it means flexitag returned invalid UTF-8
            # Try to sanitize the input more aggressively and retry
            raise RuntimeError(
                f"flexitag returned invalid UTF-8 data. This may indicate corrupted input. "
                f"Original error: {exc}"
            ) from exc
        # Sanitize output document
        result_doc = Document.from_dict(tagged_doc)
        sanitized_output = _sanitize_document(result_doc)
        result = FlexitagResult(document=sanitized_output, stats=dict(stats))
        if self._debug:
            sent_count = len(result.document.sentences)
            tok_count = sum(len(sent.tokens) for sent in result.document.sentences)
            print(
                f"[flexipipe] flexitag tagging done: sentences={sent_count} tokens={tok_count} stats={result.stats}"
            )
        return result

    @staticmethod
    def tag_once(document: Document, params_file: str, *, options: Optional[Dict[str, object]] = None) -> FlexitagResult:
        # Sanitize input document
        resolved_params = _resolve_flexitag_params_path(params_file)
        sanitized_input = _sanitize_document(document)
        try:
            doc_dict, stats = _tag_document(sanitized_input.to_dict(), resolved_params, options or {})  # type: ignore
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"flexitag returned invalid UTF-8 data. This may indicate corrupted input. "
                f"Original error: {exc}"
            ) from exc
        # Sanitize output document
        result_doc = Document.from_dict(doc_dict)
        sanitized_output = _sanitize_document(result_doc)
        return FlexitagResult(document=sanitized_output, stats=dict(stats))
