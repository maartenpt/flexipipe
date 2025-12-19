"""
SpaCy backend for flexipipe.

This module provides a SpaCy-based neural backend implementation.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from .dependency_utils import ensure_extra_installed
from .doc import Document, Sentence, SubToken, Token, Span, Entity
from .language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
    normalize_language_value,
)
from .model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from .neural_backend import BackendManager, NeuralResult

SPACY_LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "el": "Greek",
    "nb": "Norwegian Bokmål",
    "lt": "Lithuanian",
    "xx": "Multi-language",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pl": "Polish",
    "ca": "Catalan",
    "da": "Danish",
    "fi": "Finnish",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "ro": "Romanian",
    "cs": "Czech",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "th": "Thai",
    "id": "Indonesian",
    "fa": "Persian",
    "he": "Hebrew",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "uz": "Uzbek",
    "mn": "Mongolian",
    "be": "Belarusian",
    "mk": "Macedonian",
    "sq": "Albanian",
    "mt": "Maltese",
    "is": "Icelandic",
    "ga": "Irish",
    "cy": "Welsh",
    "br": "Breton",
    "gd": "Scottish Gaelic",
    "eu": "Basque",
    "gl": "Galician",
    "oc": "Occitan",
    "co": "Corsican",
    "lv": "Latvian",
    "et": "Estonian",
    "fo": "Faroese",
    "gv": "Manx",
    "af": "Afrikaans",
    "sw": "Swahili",
    "zu": "Zulu",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "am": "Amharic",
    "ti": "Tigrinya",
    "so": "Somali",
    "om": "Oromo",
    "rw": "Kinyarwanda",
    "ak": "Akan",
    "mg": "Malagasy",
    "st": "Sotho",
    "tn": "Tswana",
    "ve": "Venda",
    "ts": "Tsonga",
    "ss": "Swati",
    "nr": "Ndebele",
    "nso": "Northern Sotho",
}

SPACY_DEFAULT_MODELS = {
    "en": "en_core_web_sm",
    "english": "en_core_web_sm",
    "de": "de_core_news_sm",
    "german": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "french": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "spanish": "es_core_news_sm",
    "it": "it_core_news_sm",
    "italian": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "dutch": "nl_core_news_sm",
    "el": "el_core_news_sm",
    "greek": "el_core_news_sm",
    "nb": "nb_core_news_sm",
    "norwegian": "nb_core_news_sm",
    "norwegian bokmål": "nb_core_news_sm",
    "zh": "zh_core_web_sm",
    "chinese": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "japanese": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "korean": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "russian": "ru_core_news_sm",
    "pl": "pl_core_news_sm",
    "polish": "pl_core_news_sm",
    "ca": "ca_core_news_sm",
    "catalan": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "danish": "da_core_news_sm",
    "fi": "fi_core_news_sm",
    "finnish": "fi_core_news_sm",
    "sv": "sv_core_news_sm",
    "swedish": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "ukrainian": "uk_core_news_sm",
    "lt": "lt_core_news_sm",
    "lithuanian": "lt_core_news_sm",
    "bg": "bg_core_news_sm",
    "bulgarian": "bg_core_news_sm",
    "ro": "ro_core_news_sm",
    "romanian": "ro_core_news_sm",
}

SPACY_DEFAULT_MODELS = {
    "en": "en_core_web_sm",
    "english": "en_core_web_sm",
    "de": "de_core_news_sm",
    "german": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "french": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "spanish": "es_core_news_sm",
    "it": "it_core_news_sm",
    "italian": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "dutch": "nl_core_news_sm",
    "el": "el_core_news_sm",
    "greek": "el_core_news_sm",
    "nb": "nb_core_news_sm",
    "norwegian": "nb_core_news_sm",
    "norwegian bokmål": "nb_core_news_sm",
    "zh": "zh_core_web_sm",
    "chinese": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "japanese": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "korean": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "russian": "ru_core_news_sm",
    "pl": "pl_core_news_sm",
    "polish": "pl_core_news_sm",
    "ca": "ca_core_news_sm",
    "catalan": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "danish": "da_core_news_sm",
    "fi": "fi_core_news_sm",
    "finnish": "fi_core_news_sm",
    "sv": "sv_core_news_sm",
    "swedish": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "ukrainian": "uk_core_news_sm",
    "lt": "lt_core_news_sm",
    "lithuanian": "lt_core_news_sm",
    "bg": "bg_core_news_sm",
    "bulgarian": "bg_core_news_sm",
    "ro": "ro_core_news_sm",
    "romanian": "ro_core_news_sm",
}


def _resolve_spacy_default_model(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    candidates = []
    normalized = normalize_language_value(language)
    if normalized:
        candidates.append(normalized)
    stripped = language.strip().lower()
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    for candidate in candidates:
        if candidate in SPACY_DEFAULT_MODELS:
            return SPACY_DEFAULT_MODELS[candidate]
    return None


# Register custom extension for token IDs to preserve tokids through SpaCy pipeline
try:
    from spacy.tokens import Token as SpacyToken
    if not SpacyToken.has_extension("token_id"):
        SpacyToken.set_extension("token_id", default=None, force=True)
except ImportError:
    pass  # SpaCy not available


def _document_to_spacy_text(document: Document) -> str:
    """Convert a flexipipe Document to plain text for SpaCy processing."""
    sentences = []
    for sent in document.sentences:
        if sent.text:
            sentences.append(sent.text)
        else:
            # Reconstruct from tokens
            tokens = []
            for tok in sent.tokens:
                tokens.append(tok.form)
                # Add space after if not explicitly set to False
                if tok.space_after is not False and (tok.space_after or tok.space_after is None):
                    tokens.append(" ")
            sentences.append("".join(tokens).strip())
    return "\n".join(sentences)


def _spacy_doc_to_document(
    spacy_doc,
    original_doc: Optional[Document] = None,
    *,
    preserve_tokenization: bool = False,
    preserve_pos_tags: bool = False,
    force_single_sentence: bool = False
) -> Document:
    """
    Convert a SpaCy Doc to a flexipipe Document.
    
    Args:
        spacy_doc: The SpaCy Doc object
        original_doc: Optional original document to preserve metadata and structure
        preserve_tokenization: If True and original_doc provided, try to align tokens using tokids
        force_single_sentence: If True, don't split into multiple sentences even if SpaCy detects sentence boundaries
    """
    from spacy.tokens import Doc as SpacyDoc
    
    if not isinstance(spacy_doc, SpacyDoc):
        raise TypeError(f"Expected SpaCy Doc, got {type(spacy_doc)}")
    
    # Use original document as base if provided
    if original_doc:
        doc = Document(id=original_doc.id, meta=dict(original_doc.meta))
    else:
        doc = Document(id="", meta={})
    
    # Build tokid -> original token mapping if we have original_doc and want to preserve tokenization
    tokid_to_original: dict[str, Token] = {}
    # Track which tokids have been used to avoid duplicates
    used_tokids: set[str] = set()
    if preserve_tokenization and original_doc:
        for sent_idx, sent in enumerate(original_doc.sentences):
            for tok in sent.tokens:
                if tok.tokid:
                    tokid_to_original[tok.tokid] = tok
                # Also check subtokens for MWTs
                if tok.is_mwt and tok.subtokens:
                    for sub in tok.subtokens:
                        if sub.tokid:
                            tokid_to_original[sub.tokid] = tok  # Map to parent token
    
    # Group tokens by sentences (SpaCy handles sentence segmentation)
    current_sent_tokens: List[Token] = []
    current_sent_id = 1
    global_token_offset = 0
    
    # Try to preserve original sentence text if available
    original_sentences = original_doc.sentences if original_doc else []
    
    # For alignment: map SpaCy token text+position to original tokid
    # We'll try to match by character offsets and form similarity
    spacy_text = spacy_doc.text if hasattr(spacy_doc, 'text') else ""
    
    def _extract_sentence_annotations(tokens: List[Token]) -> Tuple[List[Entity], Dict[str, List[Span]]]:
        entities: List[Entity] = []
        span_layers: Dict[str, List[Span]] = {}
        if not hasattr(spacy_doc, "ents") or not tokens:
            return entities, span_layers
        sent_start_global_idx = tokens[0].id - 1  # 0-based
        sent_end_global_idx = tokens[-1].id      # 1-based
        for ent in getattr(spacy_doc, "ents", []):
            ent_start_idx = ent.start
            ent_end_idx = ent.end
            if not (sent_start_global_idx <= ent_start_idx < sent_end_global_idx):
                continue
            entity_start = ent_start_idx - sent_start_global_idx + 1
            entity_end = min(ent_end_idx - sent_start_global_idx, len(tokens))
            if entity_start <= 0 or entity_end < entity_start:
                continue
            label = ent.label_ if hasattr(ent, "label_") else str(ent.label)
            ent_text = ent.text if hasattr(ent, "text") else ""
            attrs: Dict[str, str] = {}
            if hasattr(ent, "kb_id_") and ent.kb_id_:
                attrs["kb_id"] = ent.kb_id_
            if hasattr(ent, "id_") and ent.id_:
                attrs["id"] = ent.id_
            span_attrs = dict(attrs)
            if ent_text:
                span_attrs.setdefault("text", ent_text)
            entities.append(
                Entity(
                    start=entity_start,
                    end=entity_end,
                    label=label,
                    text=ent_text,
                    attrs=attrs,
                )
            )
            span_layers.setdefault("ner", []).append(
                Span(
                    start=entity_start,
                    end=entity_end,
                    label=label,
                    attrs=span_attrs,
                )
            )
        return entities, span_layers

    def _finalize_sentence() -> None:
        nonlocal current_sent_tokens, current_sent_id, global_token_offset
        if not current_sent_tokens:
            return
        sentence_length = len(current_sent_tokens)
        if current_sent_id - 1 < len(original_sentences):
            orig_sent = original_sentences[current_sent_id - 1]
            sent_text = orig_sent.text
            orig_sent_id = getattr(orig_sent, "source_id", "") or orig_sent.sent_id
        else:
            sent_parts = []
            for tok in current_sent_tokens:
                sent_parts.append(tok.form)
                if tok.space_after is not False:
                    sent_parts.append(" ")
            sent_text = "".join(sent_parts).strip()
            orig_sent_id = ""
        sentence_entities, sentence_span_layers = _extract_sentence_annotations(current_sent_tokens)
        internal_id = orig_sent_id or f"s{current_sent_id}"
        sentence = Sentence(
            id=internal_id,
            sent_id=orig_sent_id,
            source_id=orig_sent_id,
            text=sent_text,
            tokens=list(current_sent_tokens),
            entities=sentence_entities,
            spans=sentence_span_layers,
        )
        doc.sentences.append(sentence)
        global_token_offset += sentence_length
        current_sent_tokens = []
        current_sent_id += 1

    # Track pending tokens for MWT reconstruction (tokens with same tokid that need to be merged)
    pending_mwt_tokens: List[Token] = []
    pending_mwt_tokid: Optional[str] = None
    
    for spacy_token in spacy_doc:
        # Skip newline/whitespace-only tokens - they shouldn't be part of the sentence
        # SpaCy sometimes tokenizes newlines as separate tokens
        if spacy_token.text.strip() == "":
            # Check if it's a newline character (not just regular whitespace)
            if "\n" in spacy_token.text or "\r" in spacy_token.text:
                continue
        
        # Check if this is the start of a new sentence
        # If force_single_sentence is True, don't split even if SpaCy thinks it's a new sentence
        if not force_single_sentence and spacy_token.idx is not None and spacy_token.is_sent_start and (current_sent_tokens or pending_mwt_tokens):
            # Flush any pending MWT tokens before starting new sentence
            if pending_mwt_tokens:
                if len(pending_mwt_tokens) > 1:
                    orig_tok_for_mwt = None
                    if pending_mwt_tokid and pending_mwt_tokid in tokid_to_original:
                        orig_tok_for_mwt = tokid_to_original[pending_mwt_tokid]
                    
                    mwt_form = orig_tok_for_mwt.form if orig_tok_for_mwt else "".join(t.form for t in pending_mwt_tokens)
                    
                    subtokens = []
                    for idx, sub_token in enumerate(pending_mwt_tokens):
                        subtoken = SubToken(
                            id=sub_token.id,
                            form=sub_token.form,
                            lemma=sub_token.lemma,
                            upos=sub_token.upos,
                            xpos=sub_token.xpos,
                            feats=sub_token.feats,
                            space_after=(idx < len(pending_mwt_tokens) - 1) or pending_mwt_tokens[-1].space_after,
                            tokid=pending_mwt_tokid or "",
                        )
                        subtokens.append(subtoken)
                    
                    mwt_token = Token(
                        id=pending_mwt_tokens[0].id,
                        form=mwt_form,
                        lemma=pending_mwt_tokens[0].lemma,
                        upos=pending_mwt_tokens[0].upos,
                        xpos=pending_mwt_tokens[0].xpos,
                        feats=pending_mwt_tokens[0].feats,
                        head=pending_mwt_tokens[0].head,
                        deprel=pending_mwt_tokens[0].deprel,
                        space_after=pending_mwt_tokens[-1].space_after,
                        tokid=pending_mwt_tokid or "",
                        is_mwt=True,
                        mwt_start=pending_mwt_tokens[0].id,
                        mwt_end=pending_mwt_tokens[-1].id,
                        subtokens=subtokens,
                    )
                    current_sent_tokens.append(mwt_token)
                else:
                    current_sent_tokens.append(pending_mwt_tokens[0])
                pending_mwt_tokens = []
                pending_mwt_tokid = None
            _finalize_sentence()
        
        # Create Token from SpaCy token
        # Try to find matching original token using tokid alignment
        original_token = None
        matched_tokid = None
        
        # First check if SpaCy token has token_id extension (most reliable)
        if hasattr(spacy_token, '_') and hasattr(spacy_token._, 'token_id') and spacy_token._.token_id:
            matched_tokid = spacy_token._.token_id
            if matched_tokid in tokid_to_original:
                original_token = tokid_to_original[matched_tokid]
        
        # Fallback: try to match by form if no extension or no match
        if not original_token and preserve_tokenization and original_doc and tokid_to_original:
            # Try to match by form across all sentences (since SpaCy may re-segment)
            # First try exact form match with unused tokid
            for orig_sent in original_doc.sentences:
                for orig_tok in orig_sent.tokens:
                    if orig_tok.form == spacy_token.text and orig_tok.tokid and orig_tok.tokid not in used_tokids:
                        original_token = orig_tok
                        matched_tokid = orig_tok.tokid
                        used_tokids.add(orig_tok.tokid)
                        break
                if original_token:
                    break
            
            # If no exact match, try case-insensitive match with unused tokid
            if not original_token:
                for orig_sent in original_doc.sentences:
                    for orig_tok in orig_sent.tokens:
                        if orig_tok.form.lower() == spacy_token.text.lower() and orig_tok.tokid and orig_tok.tokid not in used_tokids:
                            original_token = orig_tok
                            matched_tokid = orig_tok.tokid
                            used_tokids.add(orig_tok.tokid)
                            break
                    if original_token:
                        break
        
        # Fallback: match by position if preserve_pos_tags is True
        if not original_token and preserve_pos_tags and original_doc:
            if current_sent_id <= len(original_doc.sentences):
                orig_sent = original_doc.sentences[current_sent_id - 1]
                token_idx = len(current_sent_tokens)
                if token_idx < len(orig_sent.tokens):
                    original_token = orig_sent.tokens[token_idx]
                    matched_tokid = original_token.tokid if original_token else None
        
        # Determine space_after: SpaCy's whitespace_ is a string containing whitespace after the token
        # If whitespace_ is non-empty, there IS whitespace after (space_after=True or None)
        # If whitespace_ is empty, there is NO whitespace after (space_after=False)
        space_after = None
        if hasattr(spacy_token, 'whitespace_'):
            # whitespace_ is a string - if it's empty, no space after; if non-empty, space after
            space_after = bool(spacy_token.whitespace_)  # True if whitespace exists, False if not
        elif original_token is not None:
            # Preserve from original if available
            space_after = original_token.space_after
        
        token = Token(
            id=spacy_token.i + 1,  # SpaCy uses 0-based, we use 1-based
            form=spacy_token.text,
            lemma=spacy_token.lemma_ if hasattr(spacy_token, 'lemma_') else "",
            # Preserve POS tags from original if requested and available
            upos=original_token.upos if (preserve_pos_tags and original_token and original_token.upos) else (spacy_token.pos_ if hasattr(spacy_token, 'pos_') else ""),
            xpos=original_token.xpos if (preserve_pos_tags and original_token and original_token.xpos) else (spacy_token.tag_ if hasattr(spacy_token, 'tag_') else ""),
            feats=original_token.feats if (preserve_pos_tags and original_token and original_token.feats) else (_format_spacy_morph(spacy_token) if hasattr(spacy_token, 'morph') else ""),
            head=spacy_token.head.i + 1 if spacy_token.head else 0,  # Convert to 1-based
            deprel=spacy_token.dep_ if hasattr(spacy_token, 'dep_') else "",
            space_after=space_after,
            tokid=matched_tokid or "",  # Preserve tokid from original if matched
        )
        
        # Check if we should merge this token with pending tokens (MWT reconstruction)
        # Only merge if the original token was an MWT (had subtokens)
        # With pre-tokenized docs, SpaCy should preserve tokenization, so we only merge
        # when reconstructing MWTs that were split by SpaCy (which shouldn't happen, but handle it)
        should_merge = False
        if matched_tokid and preserve_tokenization and pending_mwt_tokid is not None and matched_tokid == pending_mwt_tokid:
            # Check if the original token was an MWT
            orig_tok_for_mwt = None
            if pending_mwt_tokid in tokid_to_original:
                orig_tok_for_mwt = tokid_to_original[pending_mwt_tokid]
            
            # Only merge if original was an MWT OR if we have multiple tokens with same tokid
            # (latter case shouldn't happen with pre-tokenized docs, but handle it)
            if orig_tok_for_mwt and orig_tok_for_mwt.is_mwt:
                should_merge = True
            elif len(pending_mwt_tokens) > 0:
                # Multiple tokens with same tokid but original wasn't MWT - this is unexpected
                # but could happen if SpaCy split a token. Only merge if we have multiple.
                should_merge = True
        
        if should_merge:
            # Same tokid as pending and should merge - add to pending list
            pending_mwt_tokens.append(token)
        else:
            # Different tokid or no tokid or shouldn't merge - flush pending tokens first
            if pending_mwt_tokens:
                if len(pending_mwt_tokens) > 1:
                    # Create MWT from pending tokens
                    # Find original token to get the form
                    orig_tok_for_mwt = None
                    if pending_mwt_tokid and pending_mwt_tokid in tokid_to_original:
                        orig_tok_for_mwt = tokid_to_original[pending_mwt_tokid]
                    
                    # Create MWT if original was an MWT, OR if we have multiple tokens with same tokid
                    # (the latter case indicates an expanded MWT that needs to be merged back)
                    if (orig_tok_for_mwt and orig_tok_for_mwt.is_mwt) or len(pending_mwt_tokens) > 1:
                        mwt_form = orig_tok_for_mwt.form if orig_tok_for_mwt.form else "".join(t.form for t in pending_mwt_tokens)
                        
                        # Create subtokens
                        subtokens = []
                        for idx, sub_token in enumerate(pending_mwt_tokens):
                            subtoken = SubToken(
                                id=sub_token.id,
                                form=sub_token.form,
                                lemma=sub_token.lemma,
                                upos=sub_token.upos,
                                xpos=sub_token.xpos,
                                feats=sub_token.feats,
                                space_after=(idx < len(pending_mwt_tokens) - 1) or pending_mwt_tokens[-1].space_after,
                                tokid=pending_mwt_tokid or "",
                            )
                            subtokens.append(subtoken)
                        
                        # Create MWT token
                        mwt_token = Token(
                            id=pending_mwt_tokens[0].id,
                            form=mwt_form,
                            lemma=pending_mwt_tokens[0].lemma,
                            upos=pending_mwt_tokens[0].upos,
                            xpos=pending_mwt_tokens[0].xpos,
                            feats=pending_mwt_tokens[0].feats,
                            head=pending_mwt_tokens[0].head,
                            deprel=pending_mwt_tokens[0].deprel,
                            space_after=pending_mwt_tokens[-1].space_after,
                            tokid=pending_mwt_tokid or "",
                            is_mwt=True,
                            mwt_start=pending_mwt_tokens[0].id,
                            mwt_end=pending_mwt_tokens[-1].id,
                            subtokens=subtokens,
                        )
                        current_sent_tokens.append(mwt_token)
                    else:
                        # Original wasn't an MWT, but we have multiple tokens - add them separately
                        # This shouldn't happen with pre-tokenized docs, but handle it
                        for tok in pending_mwt_tokens:
                            current_sent_tokens.append(tok)
                else:
                    # Single token, not an MWT
                    current_sent_tokens.append(pending_mwt_tokens[0])
                pending_mwt_tokens = []
                pending_mwt_tokid = None
            
            # Now handle current token
            if matched_tokid and preserve_tokenization:
                # Check if original token is an MWT
                orig_tok = tokid_to_original.get(matched_tokid) if matched_tokid in tokid_to_original else None
                if orig_tok and orig_tok.is_mwt:
                    # Start new pending MWT group - this token might be the first subtoken of an MWT
                    # We'll collect all tokens with the same tokid and merge them
                    pending_mwt_tokens = [token]
                    pending_mwt_tokid = matched_tokid
                elif orig_tok is None:
                    # Tokid not found in original - but if we have a tokid, it might be from an expanded MWT
                    # Start a pending group anyway - if we see more tokens with the same tokid, we'll merge them
                    # If not, we'll just add it as a single token
                    pending_mwt_tokens = [token]
                    pending_mwt_tokid = matched_tokid
                else:
                    # Original wasn't an MWT - add directly
                    current_sent_tokens.append(token)
            else:
                # No tokid or not preserving - add directly
                current_sent_tokens.append(token)
    
    # Flush any pending MWT tokens before final sentence
    if pending_mwt_tokens:
        if len(pending_mwt_tokens) > 1:
            orig_tok_for_mwt = None
            if pending_mwt_tokid and pending_mwt_tokid in tokid_to_original:
                orig_tok_for_mwt = tokid_to_original[pending_mwt_tokid]
            
            if orig_tok_for_mwt and orig_tok_for_mwt.is_mwt:
                mwt_form = orig_tok_for_mwt.form if orig_tok_for_mwt.form else "".join(t.form for t in pending_mwt_tokens)
                
                subtokens = []
                for idx, sub_token in enumerate(pending_mwt_tokens):
                    subtoken = SubToken(
                        id=sub_token.id,
                        form=sub_token.form,
                        lemma=sub_token.lemma,
                        upos=sub_token.upos,
                        xpos=sub_token.xpos,
                        feats=sub_token.feats,
                        space_after=(idx < len(pending_mwt_tokens) - 1) or pending_mwt_tokens[-1].space_after,
                        tokid=pending_mwt_tokid or "",
                    )
                    subtokens.append(subtoken)
                
                mwt_token = Token(
                    id=pending_mwt_tokens[0].id,
                    form=mwt_form,
                    lemma=pending_mwt_tokens[0].lemma,
                    upos=pending_mwt_tokens[0].upos,
                    xpos=pending_mwt_tokens[0].xpos,
                    feats=pending_mwt_tokens[0].feats,
                    head=pending_mwt_tokens[0].head,
                    deprel=pending_mwt_tokens[0].deprel,
                    space_after=pending_mwt_tokens[-1].space_after,
                    tokid=pending_mwt_tokid or "",
                    is_mwt=True,
                    mwt_start=pending_mwt_tokens[0].id,
                    mwt_end=pending_mwt_tokens[-1].id,
                    subtokens=subtokens,
                )
                current_sent_tokens.append(mwt_token)
            else:
                for tok in pending_mwt_tokens:
                    current_sent_tokens.append(tok)
        else:
            current_sent_tokens.append(pending_mwt_tokens[0])
        pending_mwt_tokens = []
        pending_mwt_tokid = None
    
    # Add final sentence
    _finalize_sentence()
    
    return doc


def _format_spacy_morph(spacy_token) -> str:
    """Format SpaCy morphological features to UD FEATS format."""
    if not hasattr(spacy_token, 'morph') or not spacy_token.morph:
        return ""
    
    # SpaCy morph is a MorphAnalysis object
    # Convert to UD FEATS format: Feature=Value|Feature=Value
    feats = []
    for feat, value in spacy_token.morph.to_dict().items():
        if value:
            feats.append(f"{feat}={value}")
    return "|".join(feats) if feats else ""


def _document_to_spacy_training_data(
    document: Document,
    *,
    tag_attribute: str = "xpos"
) -> List[Tuple[str, Dict]]:
    """
    Convert a flexipipe Document to SpaCy training format.
    
    Returns a list of (text, annotations) tuples where annotations contains
    token-level tags, lemmas, etc. in SpaCy's expected format.
    """
    training_examples = []
    
    for sent in document.sentences:
        if not sent.text:
            # Reconstruct text from tokens
            tokens = []
            for tok in sent.tokens:
                tokens.append(tok.form)
                if tok.space_after is not False:
                    tokens.append(" ")
            text = "".join(tokens).strip()
        else:
            text = sent.text
        
        # Build token annotations in SpaCy format
        words = []
        tags = []
        lemmas = []
        pos_tags = []
        morphs = []
        deps = []
        heads = []
        
        for tok in sent.tokens:
            words.append(tok.form)
            
            # Get tag based on tag_attribute
            if tag_attribute == "upos":
                tag = tok.upos or ""
            elif tag_attribute == "utot":
                upos = tok.upos or ""
                feats = tok.feats or ""
                tag = f"{upos}#{feats}" if upos and feats else upos
            else:  # xpos
                tag = tok.xpos or ""
            
            tags.append(tag)
            lemmas.append(tok.lemma or tok.form)
            pos_tags.append(tok.upos or "")
            
            # Parse FEATS into morph dict (SpaCy format)
            morph_dict = {}
            if tok.feats:
                for feat_pair in tok.feats.split("|"):
                    if "=" in feat_pair:
                        key, value = feat_pair.split("=", 1)
                        morph_dict[key] = value
            morphs.append(morph_dict)
            
            # Dependency parsing
            deps.append(tok.deprel or "")
            # Convert head to relative position (SpaCy uses relative indices)
            # head=0 means root, head=1 means depends on next token, etc.
            head_idx = tok.head - tok.id if tok.head > 0 else 0
            heads.append(head_idx)
        
        # SpaCy training format
        annotations = {
            "words": words,
            "tags": tags,
            "lemmas": lemmas,
            "pos": pos_tags,
            "morphs": morphs,
            "deps": deps,
            "heads": heads,
        }
        
        training_examples.append((text, annotations))
    
    return training_examples


class SpacyBackend(BackendManager):
    """SpaCy-based neural backend."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        components: Optional[List[str]] = None,
        disable: Optional[List[str]] = None,
        gpu: bool = False,
        verbose: bool = False,
        training: bool = False,
    ):
        """
        Initialize SpaCy backend.
        
        Args:
            model_name: Name of a pretrained SpaCy model (e.g., "en_core_web_sm")
            model_path: Path to a local SpaCy model directory
            language: Language code (e.g., "en", "es") - used if no model specified
            components: List of pipeline components to enable (default: all)
            disable: List of pipeline components to disable
            gpu: Whether to use GPU (requires spacy[gpu])
        """
        ensure_extra_installed("spacy", module_name="spacy", friendly_name="SpaCy")
        try:
            import spacy
            from spacy.language import Language
        except ImportError as e:
            # Check if it's a dependency issue (pydantic version mismatch)
            error_msg = str(e)
            if "pydantic" in error_msg.lower() or "ModelMetaclass" in error_msg:
                raise ImportError(
                    "SpaCy dependency error: There's a version mismatch between pydantic and confection.\n"
                    "This is usually caused by incompatible pydantic versions.\n"
                    "Try: pip install --upgrade pydantic confection spacy\n"
                    "Or: pip install 'pydantic<2' if you need pydantic v1 compatibility.\n"
                    f"Original error: {e}"
                ) from e
            raise ImportError(
                "SpaCy is not installed or failed to import. Install it with: pip install \"flexipipe[spacy]\"\n"
                f"Original error: {e}"
            ) from e
        
        self.spacy = spacy
        self._nlp: Optional[Language] = None
        self._model_name = model_name
        self._model_path = Path(model_path) if model_path else None
        self._language = language
        self._components = components or []
        self._disable = disable or []
        self._gpu = gpu
        self._verbose = verbose
        self._auto_model: Optional[str] = None

        if not self._model_name and not self._model_path:
            default_model = _resolve_spacy_default_model(self._language) or _resolve_spacy_default_model(language)
            if default_model:
                self._model_name = default_model
                self._auto_model = default_model
                self._language = default_model.split("_", 1)[0]
                lang_display = self._language or language or "unknown"
                if self._verbose:
                    print(f"[flexipipe] Auto-selected model '{default_model}' for backend 'spacy' and language '{lang_display}'.")
        
        # Load model if provided
        def _handle_spacy_value_error(exc: ValueError, model: str) -> ValueError:
            message = str(exc)
            if "curated_transformer" in message or "transformer" in message:
                return ValueError(
                    f"SpaCy model '{model}' needs the spaCy Transformers components. "
                    "Install them with: pip install \"spacy[transformers]\" "
                    "or: pip install spacy-transformers\n"
                    f"Original error: {exc}"
                )
            return exc
        
        load_model_name = self._model_name
        load_model_path = self._model_path
        load_language = self._language or language

        if load_model_name:
            # First check if model exists in flexipipe directory
            from .model_storage import get_spacy_model_path
            # For HuggingFace models (containing /), check both the original name and the sanitized name
            flexipipe_model_path = get_spacy_model_path(load_model_name)
            if not flexipipe_model_path and "/" in load_model_name:
                # Try sanitized name (replace / with _)
                sanitized_name = load_model_name.replace("/", "_")
                flexipipe_model_path = get_spacy_model_path(sanitized_name)
            if flexipipe_model_path:
                # Check if the model directory is complete (has config.cfg)
                # Also check for versioned subdirectories (e.g., en_core_web_md-3.8.0/config.cfg)
                config_cfg = None
                actual_model_path = flexipipe_model_path
                
                # Try to check for config.cfg, but handle permission errors gracefully
                try:
                    config_cfg = flexipipe_model_path / "config.cfg"
                    if not config_cfg.exists():
                        # Check if there's a versioned subdirectory with config.cfg
                        try:
                            if flexipipe_model_path.is_dir():
                                for subdir in flexipipe_model_path.iterdir():
                                    if subdir.is_dir() and subdir.name.startswith(f"{load_model_name}-"):
                                        subdir_config = subdir / "config.cfg"
                                        if subdir_config.exists():
                                            # Use the versioned subdirectory as the model path
                                            actual_model_path = subdir
                                            config_cfg = subdir_config
                                            break
                        except (PermissionError, OSError):
                            # Can't read the directory due to permissions - try common versioned paths
                            # Common spaCy version patterns: model-3.8.0, model-3.7.0, etc.
                            for version in ["3.8.0", "3.7.0", "3.6.0", "3.5.0"]:
                                versioned_path = flexipipe_model_path / f"{load_model_name}-{version}"
                                versioned_config = versioned_path / "config.cfg"
                                try:
                                    if versioned_config.exists():
                                        actual_model_path = versioned_path
                                        config_cfg = versioned_config
                                        break
                                except (PermissionError, OSError):
                                    continue
                            if not config_cfg:
                                # Still no config found, but we'll try loading anyway
                                config_cfg = None
                except (PermissionError, OSError):
                    # Can't even check if config exists due to permissions
                    # Try common versioned paths directly
                    for version in ["3.8.0", "3.7.0", "3.6.0", "3.5.0"]:
                        versioned_path = flexipipe_model_path / f"{load_model_name}-{version}"
                        versioned_config = versioned_path / "config.cfg"
                        try:
                            if versioned_config.exists():
                                actual_model_path = versioned_path
                                config_cfg = versioned_config
                                break
                        except (PermissionError, OSError):
                            continue
                    if not config_cfg:
                        config_cfg = None
                
                # Try loading the model from flexipipe directory
                # spaCy.load() can automatically find versioned subdirectories (e.g., nl_core_news_lg-3.8.0)
                # even if we can't read the directory listing due to permissions
                # On macOS, the Python process may not have Full Disk Access, but spaCy.load() might still work
                try:
                    if config_cfg and config_cfg.exists():
                        # We found a valid config, use that path
                        self._nlp = spacy.load(str(actual_model_path), disable=self._disable)
                    else:
                        # No config found or can't check due to permissions, but try loading anyway
                        # First try the base path - spaCy.load() can automatically detect versioned subdirectories
                        try:
                            self._nlp = spacy.load(str(flexipipe_model_path), disable=self._disable)
                        except (OSError, ValueError):
                            # If that fails, try common versioned paths directly
                            loaded = False
                            for version in ["3.8.0", "3.7.0", "3.6.0", "3.5.0"]:
                                versioned_path = flexipipe_model_path / f"{load_model_name}-{version}"
                                try:
                                    self._nlp = spacy.load(str(versioned_path), disable=self._disable)
                                    loaded = True
                                    break
                                except (OSError, ValueError):
                                    continue
                            if not loaded:
                                raise
                except (OSError, ValueError) as exc:
                    # If loading from flexipipe directory fails, try standard location as fallback
                    # This handles cases where the model might be accessible via spaCy's standard mechanism
                    try:
                        self._nlp = spacy.load(load_model_name, disable=self._disable)
                    except (OSError, ValueError) as exc2:
                        # Both locations failed - provide a helpful error
                        error_msg = (
                            f"SpaCy model '{load_model_name}' not found.\n"
                            f"Tried flexipipe directory: {flexipipe_model_path}\n"
                            f"Error: {exc}\n"
                            f"Tried standard location, error: {exc2}\n"
                            f"Make sure the model is installed and accessible."
                        )
                        raise ValueError(error_msg) from exc2
            else:
                # Fallback to standard SpaCy model location
                try:
                    self._nlp = spacy.load(load_model_name, disable=self._disable)
                except OSError:
                    raise ValueError(
                        f"SpaCy model '{load_model_name}' not found. "
                        f"Install it with: python -m spacy download {load_model_name}"
                    )
                except ValueError as exc:
                    raise _handle_spacy_value_error(exc, load_model_name) from exc
        elif load_model_path:
            model_path = Path(load_model_path)
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            try:
                self._nlp = spacy.load(str(model_path), disable=self._disable)
            except ValueError as exc:
                raise _handle_spacy_value_error(exc, str(model_path)) from exc
        elif load_language:
            # Create blank language model
            # For training, allow unsupported languages (will use 'xx' fallback)
            if training:
                # For training, try the requested language first, but fall back to 'xx' if not supported
                try:
                    self._nlp = spacy.blank(load_language)
                except (ImportError, OSError) as e:
                    error_str = str(e)
                    if "E048" in error_str or "Can't import language" in error_str:
                        # For training, use 'xx' (multilingual) as fallback
                        if self._verbose:
                            print(f"[flexipipe] Warning: Language '{load_language}' is not supported by SpaCy. Using 'xx' (multilingual) for training.", file=sys.stderr)
                        self._nlp = spacy.blank("xx")
                        self._language = "xx"  # Update stored language
                    else:
                        raise
            else:
                # For inference, require a supported language
                try:
                    self._nlp = spacy.blank(load_language)
                except (ImportError, OSError) as e:
                    error_str = str(e)
                    if "E048" in error_str or "Can't import language" in error_str:
                        raise ImportError(
                            f"SpaCy does not support language '{load_language}' for inference. "
                            f"SpaCy only supports certain languages out of the box."
                        ) from e
                    raise
            # Add basic components
            if components:
                for component_name in components:
                    if component_name not in self._nlp.pipe_names:
                        self._nlp.add_pipe(component_name)
        else:
            raise ValueError("Must provide either model_name, model_path, or language")
    
    @property
    def nlp(self):
        """Get the SpaCy Language object."""
        if self._nlp is None:
            raise RuntimeError("SpaCy model not loaded")
        return self._nlp
    
    @property
    def supports_training(self) -> bool:
        return True
    
    @property
    def auto_model(self) -> Optional[str]:
        return self._auto_model
    
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False
    ) -> NeuralResult:
        """
        Tag a document using SpaCy.
        
        Args:
            document: Input document
            overrides: Optional overrides for settings
            preserve_pos_tags: If True, preserve existing POS tags from input document
                              (useful for two-stage tagging: POS first, then parsing)
            components: Optional list of pipeline components to run (e.g., ["parser"] for parsing only)
        """
        import time
        
        start_time = time.time()
        
        # If preserve_pos_tags is True, we need to pass existing tags to SpaCy
        if preserve_pos_tags:
            # Create SpaCy doc with existing POS tags
            # Use original text if available to preserve exact spacing
            if document.sentences and all(sent.text for sent in document.sentences):
                text = "\n".join(sent.text for sent in document.sentences)
            else:
                text = _document_to_spacy_text(document)
            spacy_doc = self.nlp.make_doc(text)
            
            # Set POS tags and token IDs from original document
            token_idx = 0
            for sent in document.sentences:
                for tok in sent.tokens:
                    if token_idx < len(spacy_doc):
                        spacy_token = spacy_doc[token_idx]
                        # Set token ID extension if available
                        if tok.tokid and hasattr(spacy_token, '_'):
                            spacy_token._.token_id = tok.tokid
                        # Set POS tags if available
                        if tok.upos:
                            spacy_token.pos_ = tok.upos
                        if tok.xpos:
                            spacy_token.tag_ = tok.xpos
                        if tok.feats:
                            # Parse FEATS and set morphological features
                            from spacy.tokens import MorphAnalysis
                            morph_dict = {}
                            for feat_pair in tok.feats.split("|"):
                                if "=" in feat_pair:
                                    key, value = feat_pair.split("=", 1)
                                    morph_dict[key] = value
                            if morph_dict:
                                spacy_token.morph = MorphAnalysis(self.nlp.vocab, morph_dict)
                        token_idx += 1
            
            # Now run only the components that need to run (e.g., parser, lemmatizer)
            if components:
                for component_name in components:
                    if component_name in self.nlp.pipe_names:
                        self.nlp.get_pipe(component_name)(spacy_doc)
            else:
                # Run all components except tagger (since we already have tags)
                for component_name in self.nlp.pipe_names:
                    if component_name != "tagger":
                        self.nlp.get_pipe(component_name)(spacy_doc)
        else:
            # Normal processing: either use raw text or pre-tokenized Docs
            if use_raw_text:
                # Raw mode: send raw text to SpaCy for full tokenization and segmentation
                if document.sentences and all(sent.text for sent in document.sentences):
                    text = "\n".join(sent.text for sent in document.sentences)
                else:
                    text = _document_to_spacy_text(document)
                spacy_doc = self.nlp(text)
                
                # Convert back to flexipipe Document
                result_doc = _spacy_doc_to_document(
                    spacy_doc,
                    original_doc=document,
                    preserve_tokenization=False,  # Don't try to preserve in raw mode
                    preserve_pos_tags=preserve_pos_tags
                )
            else:
                # Tokenized mode: create pre-tokenized SpaCy Docs to preserve tokenization
                # This is much faster than processing raw text and respects our tokenization
                from spacy.tokens import Doc as SpacyDoc
                
                # Determine if we should try to preserve tokenization (if original has tokids)
                has_tokids = any(
                    tok.tokid for sent in document.sentences for tok in sent.tokens
                ) or any(
                    sub.tokid for sent in document.sentences for tok in sent.tokens if tok.is_mwt
                    for sub in tok.subtokens
                )
                
                # Build pre-tokenized docs for all sentences
                pre_tokenized_docs = []
                sent_tokens_list = []  # Keep track of original tokens for tokid assignment
                
                for sent in document.sentences:
                    # Extract token forms and spaces
                    # For MWTs, expand into subtokens so SpaCy processes each subtoken separately
                    words = []
                    spaces = []
                    expanded_tokens = []  # Track which original tokens/subtokens map to which SpaCy tokens
                    for tok in sent.tokens:
                        if tok.is_mwt and tok.subtokens:
                            # Expand MWT into subtokens
                            for sub_idx, sub in enumerate(tok.subtokens):
                                words.append(sub.form)
                                # Space after: only on the last subtoken of the MWT
                                has_space = (sub_idx == len(tok.subtokens) - 1) and (
                                    tok.space_after is not False and (tok.space_after or tok.space_after is None)
                                )
                                spaces.append(has_space)
                                expanded_tokens.append(tok)  # Map back to parent MWT token
                        else:
                            # Regular token
                            words.append(tok.form)
                            has_space = tok.space_after is not False and (tok.space_after or tok.space_after is None)
                            spaces.append(has_space)
                            expanded_tokens.append(tok)
                    
                    if words:
                        # Create a pre-tokenized SpaCy Doc
                        # This preserves our tokenization and prevents SpaCy from re-tokenizing
                        sent_doc = SpacyDoc(self.nlp.vocab, words=words, spaces=spaces)
                        pre_tokenized_docs.append(sent_doc)
                        sent_tokens_list.append(expanded_tokens)  # Use expanded tokens for tokid mapping
                
                # Process all docs in batch using nlp.pipe() for better performance
                # Note: nlp.pipe() expects texts, but we can pass pre-tokenized docs
                # We'll process them individually but more efficiently
                spacy_docs = []
                for sent_doc, sent_tokens in zip(pre_tokenized_docs, sent_tokens_list):
                    # Set token_id extensions if available
                    # sent_tokens now contains expanded tokens (MWTs expanded to subtokens)
                    # For MWTs, expanded_tokens contains the parent MWT token for each subtoken
                    # All subtokens from the same MWT should get the parent MWT's tokid so they can be merged back
                    if has_tokids:
                        for i, tok in enumerate(sent_tokens):
                            if i < len(sent_doc):
                                # For MWTs, use the parent token's tokid (all subtokens share the same tokid)
                                # For regular tokens, use the token's tokid
                                tokid = None
                                if tok.is_mwt:
                                    # This is a parent MWT token - use its tokid
                                    # All subtokens from this MWT should share the same tokid
                                    tokid = tok.tokid if tok.tokid else None
                                    # If parent doesn't have tokid, try first subtoken's tokid
                                    if not tokid and tok.subtokens and tok.subtokens[0].tokid:
                                        tokid = tok.subtokens[0].tokid
                                else:
                                    # Regular token - use its tokid
                                    tokid = tok.tokid if tok.tokid else None
                                
                                if tokid and hasattr(sent_doc[i], '_'):
                                    sent_doc[i]._.token_id = tokid
                    
                    # Run the pipeline on the pre-tokenized doc
                    # This will tag, parse, lemmatize, etc. but won't re-tokenize
                    for component_name in self.nlp.pipe_names:
                        self.nlp.get_pipe(component_name)(sent_doc)
                    
                    spacy_docs.append(sent_doc)
                
                # Convert each sentence doc separately to preserve sentence boundaries
                if spacy_docs:
                    # Create result document
                    if document:
                        result_doc = Document(id=document.id, meta=dict(document.meta))
                    else:
                        result_doc = Document(id="", meta={})
                    
                    # Convert each sentence doc separately
                    global_token_offset = 0
                    for sent_idx, (sent_doc, orig_sent) in enumerate(zip(spacy_docs, document.sentences)):
                        # Convert this sentence doc to a flexipipe sentence
                        # Use force_single_sentence=True to prevent SpaCy from splitting it
                        sent_doc_wrapper = Document(id="", sentences=[])
                        sent_doc_wrapper.sentences.append(orig_sent)  # Use original sentence as template
                        
                        temp_doc = _spacy_doc_to_document(
                            sent_doc,
                            original_doc=sent_doc_wrapper,
                            preserve_tokenization=has_tokids,
                            preserve_pos_tags=preserve_pos_tags,
                            force_single_sentence=True  # Force single sentence to preserve boundaries
                        )
                        
                        if temp_doc.sentences:
                            result_sent = temp_doc.sentences[0]
                            # Preserve original sentence metadata
                            result_sent.id = orig_sent.id
                            result_sent.sent_id = orig_sent.sent_id
                            result_sent.text = orig_sent.text  # Preserve original text
                            
                            # Extract entities from this sentence doc
                            if hasattr(sent_doc, 'ents'):
                                result_sent.entities = []
                                for ent in sent_doc.ents:
                                    ent_start = ent.start  # 0-based
                                    ent_end = ent.end  # 0-based, exclusive
                                    
                                    if ent_start >= 0 and ent_end > ent_start:
                                        label = ent.label_ if hasattr(ent, 'label_') else str(ent.label)
                                        ent_text = ent.text if hasattr(ent, 'text') else ""
                                        attrs = {}
                                        if hasattr(ent, 'kb_id_') and ent.kb_id_:
                                            attrs['kb_id'] = ent.kb_id_
                                        if hasattr(ent, 'id_') and ent.id_:
                                            attrs['id'] = ent.id_
                                        
                                        entity = Entity(
                                            start=ent_start + 1,  # Convert to 1-based
                                            end=ent_end,  # 1-based, inclusive
                                            label=label,
                                            text=ent_text,
                                            attrs=attrs,
                                        )
                                        result_sent.entities.append(entity)
                                        
                                        span_attrs = dict(attrs)
                                        if ent_text:
                                            span_attrs.setdefault("text", ent_text)
                                        span_attrs.setdefault("sentence_id", result_sent.sent_id or result_sent.id)
                                        span_attrs.setdefault("sentence_index", sent_idx)
                                        span_attrs.setdefault("start_in_sentence", entity.start)
                                        span_attrs.setdefault("end_in_sentence", entity.end)
                                        result_doc.add_span(
                                            "ner",
                                            Span(
                                                label=label,
                                                start=global_token_offset + entity.start,
                                                end=global_token_offset + entity.end,
                                                attrs=span_attrs,
                                            ),
                                        )
                            
                            result_doc.sentences.append(result_sent)
                            global_token_offset += len(result_sent.tokens)
                        else:
                            # Fallback: create sentence from original if conversion failed
                            result_doc.sentences.append(orig_sent)
                else:
                    # Fallback: process as single text
                    if document.sentences and all(sent.text for sent in document.sentences):
                        text = "\n".join(sent.text for sent in document.sentences)
                    else:
                        text = _document_to_spacy_text(document)
                    spacy_doc = self.nlp(text)
                    
                    # Convert back to flexipipe Document
                    result_doc = _spacy_doc_to_document(
                        spacy_doc,
                        original_doc=document,
                        preserve_tokenization=False,
                        preserve_pos_tags=preserve_pos_tags
                    )
        
        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        
        # Set model information in document meta for CoNLL-U output
        model_name_used = self._model_name or self._auto_model
        if model_name_used:
            if "_file_level_attrs" not in result_doc.meta:
                result_doc.meta["_file_level_attrs"] = {}
            result_doc.meta["_file_level_attrs"]["spacy_model"] = model_name_used
        
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0,
        }
        
        return NeuralResult(document=result_doc, stats=stats)
    
    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs
    ) -> Path:
        """Train a SpaCy model using SpaCy's CLI."""
        from spacy.cli.init_config import init_config
        from spacy.cli.train import train as spacy_cli_train
        from spacy.util import load_config

        language = kwargs.get("language") or self._language or (self._nlp.lang if self._nlp else None)
        if not language:
            raise ValueError("SpaCy training requires a language code. Provide --language.")
        
        verbose = bool(kwargs.get("verbose", False))
        
        # For training, if the language is not supported by SpaCy, use 'xx' (multilingual) as fallback
        # This allows training models for languages not in SpaCy's built-in language list
        # Note: The language check happens in __init__ when creating the blank model, so this is just for consistency
        if self._nlp and self._nlp.lang == "xx" and language != "xx":
            if verbose:
                import sys
                print(f"[flexipipe] Warning: Language '{language}' is not supported by SpaCy. Using 'xx' (multilingual) for training.", file=sys.stderr)
            language = "xx"
        model_name = kwargs.get("model_name") or output_dir.name

        if isinstance(train_data, (str, Path)):
            source_reference = str(Path(train_data).resolve())
        else:
            source_reference = "in-memory"

        train_files = self._collect_conllu_files(train_data, split="train")
        if dev_data:
            dev_files = self._collect_conllu_files(dev_data, split="dev")
        else:
            dev_files = []
            base_candidate: Optional[Path] = None
            if isinstance(train_data, (str, Path)):
                base_candidate = Path(train_data)
            if base_candidate and base_candidate.is_dir():
                try:
                    dev_files = self._collect_conllu_files(base_candidate, split="dev")
                except ValueError:
                    dev_files = []

        output_dir = output_dir.resolve()
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise ValueError(f"Output directory {output_dir} must be empty before training.")
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            train_docbin = self._convert_conllu_to_docbin(
                train_files,
                tmp_dir,
                language=language,
                target_name="train.spacy",
                silent=not verbose,
            )
            # Verify the file exists and is readable immediately after conversion
            # Add a small delay in case of filesystem lag
            import time
            time.sleep(0.2)
            if not train_docbin.exists():
                # List files in tmp_dir to see what actually exists
                existing = list(tmp_dir.glob("*"))
                raise RuntimeError(
                    f"Training data file was not created: {train_docbin}. "
                    f"Files in tmp_dir: {existing}"
                )
            if train_docbin.stat().st_size == 0:
                raise RuntimeError(f"Training data file is empty: {train_docbin}")
            
            if dev_files:
                dev_docbin = self._convert_conllu_to_docbin(
                    dev_files,
                    tmp_dir,
                    language=language,
                    target_name="dev.spacy",
                    silent=not verbose,
                )
                if not dev_docbin.exists():
                    raise RuntimeError(f"Dev data file was not created: {dev_docbin}")
                if dev_docbin.stat().st_size == 0:
                    raise RuntimeError(f"Dev data file is empty: {dev_docbin}")
            else:
                dev_docbin = tmp_dir / "dev.spacy"
                shutil.copy(train_docbin, dev_docbin)

            # Detect what annotations are available in the training data
            from .validator import detect_annotation_coverage
            
            # Check the first training file to determine annotation coverage
            coverage = detect_annotation_coverage(train_files[0]) if train_files else {}
            
            # Build pipeline based on available annotations
            # - tagger: needs UPOS or XPOS
            # - morphologizer: needs FEATS
            # - parser: needs HEAD and DEPREL
            pipeline = ["tok2vec"]  # Always include tok2vec (token vectorizer)
            
            if coverage.get("upos") or coverage.get("xpos"):
                pipeline.append("tagger")
            if coverage.get("feats"):
                pipeline.append("morphologizer")
            if coverage.get("head") and coverage.get("deprel"):
                pipeline.append("parser")
            
            # Warn if user-specified components can't be trained
            if self._components:
                missing = []
                if "morphologizer" in self._components and not coverage.get("feats"):
                    missing.append("morphologizer (requires FEATS)")
                if "parser" in self._components and (not coverage.get("head") or not coverage.get("deprel")):
                    missing.append("parser (requires HEAD and DEPREL)")
                if missing and verbose:
                    import sys
                    print(f"[flexipipe] Warning: Cannot train {', '.join(missing)} - required annotations not found in training data.", file=sys.stderr)
            
            if verbose:
                import sys
                available = [k for k, v in coverage.items() if v]
                print(f"[flexipipe] Detected annotations in training data: {', '.join(available) if available else 'none'}", file=sys.stderr)
                print(f"[flexipipe] Training pipeline: {' -> '.join(pipeline)}", file=sys.stderr)
            
            base_config = None
            base_config_path = self._get_base_config_path()
            if base_config_path and base_config_path.exists():
                base_config = load_config(base_config_path)
            if base_config is None:
                # Use the language from self._nlp if available (may be 'xx' for unsupported languages)
                config_lang = self._nlp.lang if self._nlp else language
                base_config = init_config(
                    lang=config_lang,
                    pipeline=pipeline,
                    optimize="efficiency",
                    silent=not verbose,
                )

            # Use absolute paths but don't resolve symlinks (to avoid path mismatches on macOS)
            # The files were created at these paths, so we should use them as-is
            train_docbin_abs = train_docbin.absolute()
            dev_docbin_abs = dev_docbin.absolute()
            
            if not train_docbin.exists():
                raise RuntimeError(f"Training data file does not exist: {train_docbin} (absolute: {train_docbin_abs})")
            if not dev_docbin.exists():
                raise RuntimeError(f"Dev data file does not exist: {dev_docbin} (absolute: {dev_docbin_abs})")
            
            # Use the original path (not resolved) to match where the file actually exists
            base_config["paths"]["train"] = str(train_docbin)
            base_config["paths"]["dev"] = str(dev_docbin)

            config_path = tmp_dir / "config.cfg"
            base_config.to_disk(config_path)
            
            # Verify config was saved correctly
            if not config_path.exists():
                raise RuntimeError(f"Failed to save config file: {config_path}")

            training_out = tmp_dir / "training"
            training_out.mkdir(exist_ok=True)
            
            # Verify files still exist before training
            if not train_docbin.exists():
                raise RuntimeError(f"Training data file disappeared before training: {train_docbin}")
            if not dev_docbin.exists():
                raise RuntimeError(f"Dev data file disappeared before training: {dev_docbin}")
            
            spacy_cli_train(str(config_path), str(training_out))

            best_model_src = training_out / "model-best"
            if not best_model_src.exists():
                raise RuntimeError("SpaCy training did not produce a model-best directory.")
            shutil.copytree(best_model_src, output_dir, dirs_exist_ok=False)

        meta_path = output_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
        meta["name"] = model_name
        meta["lang"] = language
        flexipipe_meta = meta.get("flexipipe", {})
        flexipipe_meta.update(
            {
                "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "source": source_reference,
            }
        )
        meta["flexipipe"] = flexipipe_meta
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        if verbose:
            print(f"[spacy] Trained model stored at {output_dir}")

        return output_dir
    
    def _load_training_data(self, path: Path) -> List[Document]:
        """Load training data from a file."""
        from .conllu import conllu_to_document
        
        if path.suffix == ".conllu":
            # Load CoNLL-U file
            with path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                # Split by double newlines to get documents
                doc_texts = content.split("\n\n\n")
                docs = []
                for i, doc_text in enumerate(doc_texts):
                    if doc_text.strip():
                        doc = conllu_to_document(doc_text, doc_id=f"doc_{i}")
                        docs.append(doc)
                return docs
        else:
            raise ValueError(f"Unsupported training data format: {path.suffix}")

    def _collect_conllu_files(
        self,
        data: Union[Document, List[Document], Path, str, None],
        *,
        split: Optional[str] = None,
    ) -> list[Path]:
        if data is None:
            raise ValueError("Training data is required.")
        if isinstance(data, (Document, list)):
            raise ValueError("SpaCy training currently expects file paths pointing to CoNLL-U data.")
        path = Path(data)
        if path.is_file():
            if path.suffix.lower() != ".conllu":
                raise ValueError(f"Unsupported training data format: {path.suffix}")
            return [path]
        if path.is_dir():
            files: list[Path] = []
            if split:
                patterns = [
                    f"*ud-{split}.conllu",
                    f"*_{split}.conllu",
                    f"*{split}.conllu",
                ]
                for pattern in patterns:
                    matched = sorted(path.glob(pattern))
                    if matched:
                        files = matched
                        break
            if not files:
                files = sorted(path.glob("*.conllu"))
            if not files:
                raise ValueError(f"No .conllu files found in directory {path}")
            return files
        raise ValueError(f"Training data path does not exist: {path}")

    def _convert_conllu_to_docbin(
        self,
        sources: list[Path],
        tmp_dir: Path,
        *,
        language: str,
        target_name: str,
        silent: bool = True,
    ) -> Path:
        from spacy.cli.convert import convert

        docbin_paths = []
        for source in sources:
            # Convert CoNLL-U to SpaCy DocBin format
            # Note: convert writes to the output directory, using the source filename
            # Check files before conversion for debugging
            files_before = set(tmp_dir.glob("*"))
            try:
                convert(
                    str(source),  # Ensure it's a string path
                    str(tmp_dir),  # Ensure it's a string path
                    file_type="spacy",
                    converter="conllu",
                    n_sents=1000,
                    seg_sents=True,
                    morphology=True,
                    lang=language,
                    silent=silent,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert {source} to SpaCy DocBin format: {e}"
                ) from e
            
            # Check what files were created
            files_after = set(tmp_dir.glob("*"))
            new_files = files_after - files_before
            if not silent:
                import sys
                print(f"[flexipipe] DEBUG: Files before convert: {[str(f) for f in files_before]}", file=sys.stderr)
                print(f"[flexipipe] DEBUG: Files after convert: {[str(f) for f in files_after]}", file=sys.stderr)
                print(f"[flexipipe] DEBUG: New files: {[str(f) for f in new_files]}", file=sys.stderr)
            # SpaCy's convert creates a file with the source stem + .spacy extension
            # But it might create it with a slightly different name - check what actually exists
            produced = None
            
            # SpaCy's convert creates a file with the source stem + .spacy extension
            # Wait a moment for file to be created (convert might take time to flush)
            import time
            time.sleep(0.5)  # Give convert time to finish
            
            # The output file should be named after the source file stem
            expected = tmp_dir / f"{source.stem}.spacy"
            
            # Check if file exists, with retries
            max_retries = 10
            for retry in range(max_retries):
                if expected.exists():
                    produced = expected
                    break
                time.sleep(0.1)
            else:
                # File still doesn't exist - list what's actually in the directory
                all_files = sorted(tmp_dir.glob("*"))
                all_spacy = sorted(tmp_dir.glob("*.spacy"))
                raise RuntimeError(
                    f"Failed to convert {source} to DocBin after {max_retries} retries. "
                    f"Expected: {expected}, "
                    f"All files in {tmp_dir}: {[str(f) for f in all_files]}, "
                    f".spacy files: {[str(f) for f in all_spacy]}"
                )
            
            # Verify file is not empty
            if produced.stat().st_size == 0:
                raise RuntimeError(f"Converted file is empty: {produced}")
            
            docbin_paths.append(produced)

        target_path = tmp_dir / target_name
        
        if len(docbin_paths) == 1:
            source_file = docbin_paths[0]
            # If target is the same as source, no move needed
            if source_file == target_path:
                # File is already at target location
                if not target_path.exists():
                    raise RuntimeError(f"Target file does not exist (same as source): {target_path}")
            else:
                # Use shutil.move instead of replace to handle cross-filesystem moves
                import shutil
                if not source_file.exists():
                    raise RuntimeError(f"Source file does not exist before move: {source_file}")
                # Remove target if it exists
                if target_path.exists():
                    target_path.unlink()
                # Move source to target
                shutil.move(str(source_file), str(target_path))
                # Verify move succeeded
                if not target_path.exists():
                    raise RuntimeError(f"Target file does not exist after move: {target_path} (source was: {source_file})")
        else:
            import srsly
            from spacy.tokens import DocBin

            docbin_out = DocBin()
            for doc_path in docbin_paths:
                docbin = DocBin().from_disk(doc_path)
                for doc in docbin.get_docs(self.spacy.blank(language).vocab):
                    docbin_out.add(doc)
            docbin_out.to_disk(target_path)
            for doc_path in docbin_paths:
                doc_path.unlink(missing_ok=True)

        return target_path

    def _get_base_config_path(self) -> Optional[Path]:
        if self._model_path:
            candidate = self._model_path / "config.cfg"
            if candidate.exists():
                return candidate
        if self._model_name:
            try:
                from spacy.util import get_package_path

                package_path = get_package_path(self._model_name)
                candidate = package_path / "config.cfg"
                if candidate.exists():
                    return candidate
            except Exception:
                return None
        return None
    
    @classmethod
    def from_model_path(cls, model_path: Union[str, Path], **kwargs) -> "SpacyBackend":
        """Create a SpacyBackend from a model path."""
        return cls(model_path=str(model_path), **kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "SpacyBackend":
        """Create a SpacyBackend from a pretrained model name."""
        return cls(model_name=model_name, **kwargs)
    
    @classmethod
    def blank(cls, language: str, **kwargs) -> "SpacyBackend":
        """Create a blank SpacyBackend for a language."""
        return cls(language=language, **kwargs)


MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def get_spacy_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
    **kwargs: Any,  # Accept additional kwargs to be compatible with other backends
) -> Tuple[Dict[str, Dict[str, str]], Path, List[str]]:
    import importlib.metadata

    result: Dict[str, Dict[str, str]] = {}

    try:
        spacy_dir = get_backend_models_dir("spacy", create=False)
        if verbose:
            print(f"[flexipipe] Checking SpaCy models directory: {spacy_dir}", file=sys.stderr)
    except (OSError, PermissionError) as e:
        # If we can't even get the directory path (e.g., permission denied), return empty result
        if verbose:
            print(f"[flexipipe] Warning: Could not get SpaCy models directory: {e}", file=sys.stderr)
        return {}, Path("/"), []
    installed_models: Dict[str, str] = {}
    if spacy_dir.exists():
        if verbose:
            print(f"[flexipipe] SpaCy models directory exists: {spacy_dir}", file=sys.stderr)
        try:
            model_dirs = list(spacy_dir.iterdir())
            if verbose:
                print(f"[flexipipe] Found {len(model_dirs)} items in SpaCy models directory", file=sys.stderr)
            for model_dir in model_dirs:
                if model_dir.is_dir() and (model_dir / "meta.json").exists():
                    model_name = model_dir.name
                    if verbose:
                        print(f"[flexipipe] Found installed model: {model_name}", file=sys.stderr)
                    try:
                        with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            version = meta.get("version", "")
                            installed_models[model_name] = version
                    except Exception:
                        installed_models[model_name] = ""
        except (OSError, PermissionError) as e:
            # If we can't read the directory (permission denied), just continue with empty installed_models
            if verbose:
                print(f"[flexipipe] Warning: Could not read SpaCy models directory: {e}", file=sys.stderr)
            pass
    else:
        if verbose:
            print(f"[flexipipe] SpaCy models directory does not exist: {spacy_dir}", file=sys.stderr)

    standard_location_models: List[str] = []
    try:
        installed_packages = importlib.metadata.distributions()
        for dist in installed_packages:
            name = dist.metadata.get("Name", "")
            if name.startswith(
                (
                    "en-core-web",
                    "de-core-news",
                    "fr-core-news",
                    "es-core-news",
                    "it-core-news",
                    "pt-core-news",
                    "nl-core-news",
                    "el-core-news",
                    "nb-core-news",
                    "lt-core-news",
                    "xx-core-news",
                    "zh-core-web",
                    "ja-core-news",
                    "ko-core-news",
                    "ru-core-news",
                    "pl-core-news",
                    "ca-core-news",
                    "da-core-news",
                    "fi-core-news",
                    "sv-core-news",
                    "uk-core-news",
                    "hr-core-news",
                    "bg-core-news",
                    "ro-core-news",
                    "cs-core-news",
                    "sk-core-news",
                    "sl-core-news",
                    "sr-core-news",
                )
            ):
                model_name = name.replace("-", "_")
                if model_name not in installed_models:
                    installed_models[model_name] = dist.version
                    standard_location_models.append(model_name)
    except Exception:
        pass

    cache_key = "spacy"
    remote_models: Dict[str, Dict[str, str]] = {}
    cache_hit = False
    
    # First, try to get models from the registry (if configured)
    registry_models: Dict[str, Dict[str, str]] = {}
    try:
        from flexipipe.model_registry import get_remote_models_for_backend
        registry_model_list = get_remote_models_for_backend(
            "spacy",
            source=None,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=verbose,
        )
        # Convert list to dict keyed by model name
        for model_entry in registry_model_list:
            model_name = model_entry.get("model")
            if model_name:
                registry_models[model_name] = model_entry
        if registry_models and verbose:
            from flexipipe.model_registry import get_registry_url
            registry_url = get_registry_url("spacy")
            if registry_url.startswith("file://"):
                print(f"[flexipipe] Found {len(registry_models)} model(s) in local registry ({registry_url})", file=sys.stderr)
            else:
                print(f"[flexipipe] Found {len(registry_models)} model(s) in remote registry ({registry_url})", file=sys.stderr)
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to load models from registry ({exc})", file=sys.stderr)
    
    # Use registry models only (no longer fetching from GitHub)
    if registry_models:
        remote_models = registry_models
        # Cache the registry models for faster access
        if refresh_cache:
            try:
                write_model_cache_entry(cache_key, remote_models)
            except (OSError, PermissionError):
                pass
    elif use_cache and not refresh_cache:
        # Try to load from cache if registry is not available
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            remote_models = cached

    if verbose:
        print(f"[flexipipe] Found {len(installed_models)} installed models in directory: {spacy_dir}", file=sys.stderr)
        if remote_models:
            print(f"[flexipipe] Processing {len(remote_models)} remote models", file=sys.stderr)
        if installed_models:
            print(f"[flexipipe] Installed model names: {', '.join(sorted(installed_models.keys())[:10])}", file=sys.stderr)

    for model_name, info in remote_models.items():
        latest_version = info.get("latest_version", "")
        if model_name in installed_models:
            status = "installed"
            version = installed_models[model_name]
            if verbose:
                print(f"[flexipipe] Model {model_name}: INSTALLED (version {version})", file=sys.stderr)
        else:
            status = "available"
            version = latest_version
        entry = dict(info)
        entry["status"] = status
        entry["version"] = version
        # Preserve source from registry (official, flexipipe, community)
        if "source" not in entry:
            entry["source"] = info.get("source", "flexipipe")
        result[model_name] = entry

    for model_name, version in installed_models.items():
        if model_name in result:
            continue
        # Try to read language and source info from meta.json
        lang = None
        model_source = None
        model_dir = spacy_dir / model_name
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    # Check flexipipe metadata for language and training info
                    flexipipe_meta = meta.get("flexipipe", {})
                    # If flexipipe.trained_at exists, this is a locally trained model
                    if flexipipe_meta.get("trained_at"):
                        model_source = "local"
                    # Use lang from meta.json, but prefer flexipipe metadata if available
                    meta_lang = meta.get("lang")
                    # If lang is "xx" (multilingual) but we have flexipipe metadata, 
                    # the actual language might be in the model name or training source
                    if meta_lang and meta_lang != "xx":
                        lang = meta_lang
            except Exception:
                pass
        
        # If not found in meta.json, extract from model name (handle both _ and -)
        if not lang:
            # Try splitting on underscore first (standard SpaCy format: en_core_web_sm)
            if "_" in model_name:
                lang = model_name.split("_")[0]
            # Try splitting on hyphen (custom models: swa-masakhane)
            elif "-" in model_name:
                lang = model_name.split("-")[0]
            else:
                # No separator, use whole name as language code
                lang = model_name
        
        # Determine source if not already set
        if not model_source:
            # Check if model is in registry (from flexipipe-models)
            if model_name in registry_models:
                registry_entry = registry_models[model_name]
                model_source = registry_entry.get("source", "flexipipe")
            # Check if model is in standard location (installed via pip)
            elif model_name in standard_location_models:
                model_source = "official"
            else:
                # Default to "official" for models installed but not in registry
                # (likely from SpaCy's official releases)
                model_source = "official"
        
        # For custom trained models, preserve the original language code from model name
        # to ensure proper matching (e.g., "swa" should not be normalized to "sw")
        # But still use standardize_language_metadata for proper ISO code handling
        entry = build_model_entry(
            "spacy",
            model_name,
            language_code=lang,
            language_name=SPACY_LANGUAGE_NAMES.get(lang),
            description=f"{lang.upper()} model",
        )
        # If the extracted language code is different from what standardize_language_metadata produced,
        # preserve the original as well (for matching purposes)
        # This handles cases like "swa" (ISO-639-3) vs "sw" (ISO-639-1)
        if lang and lang.lower() != entry.get("language_iso", "").lower():
            # Preserve original language code for better matching
            entry["original_language_iso"] = lang.lower()
            # Also set language_iso to the original if it's a valid 3-letter code
            # This ensures "swa" matches when querying with "swa"
            if len(lang) == 3 and lang.isalpha():
                entry["language_iso"] = lang.lower()
        entry["status"] = "installed"
        entry["version"] = version
        entry["source"] = model_source  # Mark source: "official", "flexipipe", "community", or "local"
        result[model_name] = entry

    if verbose and standard_location_models:
        print(
            f"Note: {len(standard_location_models)} additional model(s) detected in the standard SpaCy location (not managed by flexipipe)"
        )

    return result, spacy_dir, standard_location_models


def list_spacy_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> int:
    """
    List available SpaCy models from GitHub releases and local installations.
    """
    try:
        result, spacy_dir, standard_location_models = get_spacy_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=verbose,
        )

        # Display results
        print(f"\nAvailable SpaCy models:")
        print(f"{'Model Name':<40} {'ISO':<8} {'Status':<20} {'Description':<50}")
        print("=" * 125)

        installed_count = 0
        for model_name in sorted(result.keys()):
            model_info = result[model_name]
            lang_iso = model_info.get(LANGUAGE_FIELD_ISO) or ""
            lang_name = model_info.get(LANGUAGE_FIELD_NAME) or ""
            lang_display = lang_iso or lang_name or ""
            desc = model_info.get("description", "")
            status_str = model_info.get("status", "available")
            version = model_info.get("version", "")

            if status_str == "installed":
                status = f"✓ Installed (v{version})" if version else "✓ Installed"
                installed_count += 1
            else:
                status = "Available"

            print(f"{model_name:<40} {lang_display:<8} {status:<20} {desc:<50}")

        if installed_count > 0:
            print(f"\nNote: {installed_count} model(s) installed in {spacy_dir}")
        if standard_location_models:
            print(
                f"Note: {len(standard_location_models)} additional model(s) detected in the standard SpaCy location (not managed by flexipipe)"
            )
        print("Download models with: python -m flexipipe tag --backend spacy --model <model_name> --download-model")
        print("Features: tokenization, lemma, upos, xpos, feats, depparse, NER")

        # Summary
        total_models = len(result)
        unique_languages = len(
            {
                model_info.get(LANGUAGE_FIELD_ISO) or model_info.get(LANGUAGE_FIELD_NAME)
                for model_info in result.values()
                if model_info.get(LANGUAGE_FIELD_ISO) or model_info.get(LANGUAGE_FIELD_NAME)
            }
        )
        print(f"\nTotal: {total_models} model(s) for {unique_languages} language(s)")

        return 0
    except Exception as e:
        print(f"Error listing SpaCy models: {e}")
        import traceback

        traceback.print_exc()
        return 1

