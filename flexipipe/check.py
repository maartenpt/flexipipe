from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from . import Document, FlexitagFallback
from .conllu import conllu_to_document, document_to_conllu
from .doc import Sentence, SubToken, Token
from .neural_backend import BackendManager
from typing import Union, Optional


@dataclass
class Metric:
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool) -> None:
        self.correct += int(bool(is_correct))
        self.total += 1

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total


def _feats_partial_match(gold_feats: str, pred_feats: str) -> bool:
    """
    Return True if all predicted feature key/value pairs are correct (match gold).
    This is a "partial" match because:
    - Missing features in prediction are okay (it's partial)
    - Every feature that IS predicted must be correct (match gold)
    - Extra features in prediction that aren't in gold are wrong
    
    This treats "_" or empty strings as "no features".
    
    Note: If gold has no features, prediction must also have no features to be correct.
    If gold has features and prediction has none, that's incorrect (missing required features).
    """

    def parse_feats(feats: str) -> dict[str, str]:
        feats = feats.strip()
        if not feats or feats == "_":
            return {}
        pairs = {}
        for feat in feats.split("|"):
            if "=" in feat:
                key, value = feat.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    pairs[key] = value
        return pairs

    gold_map = parse_feats(gold_feats)
    pred_map = parse_feats(pred_feats)
    
    # If gold has no features, prediction must also have no features
    if not gold_map:
        return not pred_map
    
    # If gold has features but prediction has none, that's incorrect
    if not pred_map:
        return False
    
    # Every feature that is predicted must exist in gold with the same value
    # Missing features in prediction are okay (it's partial)
    for key, value in pred_map.items():
        if key not in gold_map or gold_map[key] != value:
            return False
    return True


def _normalize_form_for_alignment(form: str) -> str:
    stripped = "".join(ch.lower() for ch in form if ch.isalnum())
    return stripped if stripped else form.lower()


def _align_tokens_by_form(gold_tokens: List[Token], pred_tokens: List[Token]) -> dict[int, int]:
    if not gold_tokens or not pred_tokens:
        return {}
    gold_norm = [_normalize_form_for_alignment(tok.form) for tok in gold_tokens]
    pred_norm = [_normalize_form_for_alignment(tok.form) for tok in pred_tokens]
    matcher = SequenceMatcher(None, gold_norm, pred_norm, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        i, j, size = block
        for k in range(size):
            gi = i + k
            pj = j + k
            if gi < len(gold_norm) and pj < len(pred_norm) and gold_norm[gi] and gold_norm[gi] == pred_norm[pj]:
                mapping.setdefault(gi, pj)
    return mapping


def _align_flat_sequences(gold_norm: List[str], pred_norm: List[str]) -> dict[int, int]:
    matcher = SequenceMatcher(None, gold_norm, pred_norm, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        i, j, size = block
        for k in range(size):
            gi = i + k
            pj = j + k
            if gi < len(gold_norm) and pj < len(pred_norm):
                if gold_norm[gi] and gold_norm[gi] == pred_norm[pj]:
                    mapping.setdefault(gi, pj)
    return mapping


def align_gold_and_predicted(
    gold_doc: Document,
    pred_doc: Document,
    *,
    prefer_tokids: bool = True,
    prefer_sentence_boundaries: bool = True,
    max_distance: int = 10,
) -> List[tuple[Token, Token | None, int, int | None]]:
    """
    Centralized function to align tokens between gold and predicted documents.
    
    Uses a multi-stage alignment strategy:
    1. First, try to match by tokid (if available and prefer_tokids=True)
    2. Then, try to match by form within sentence boundaries (if prefer_sentence_boundaries=True)
    3. Finally, use sequence alignment on normalized forms
    
    Args:
        gold_doc: Gold standard document
        pred_doc: Predicted document
        prefer_tokids: If True, prioritize tokid matches (default: True)
        prefer_sentence_boundaries: If True, only match tokens within the same sentence (default: True)
        max_distance: Maximum distance to search for matches when tokid/form matching fails (default: 10)
    
    Returns:
        List of tuples: (gold_token, pred_token, gold_sent_idx, pred_sent_idx)
        pred_token and pred_sent_idx may be None if no match found
    """
    aligned: List[tuple[Token, Token | None, int, int | None]] = []
    
    # Flatten tokens with sentence indices
    gold_flat: List[tuple[Token, int]] = []
    for sent_idx, sent in enumerate(gold_doc.sentences):
        for tok in sent.tokens:
            gold_flat.append((tok, sent_idx))
    
    pred_flat: List[tuple[Token, int]] = []
    for sent_idx, sent in enumerate(pred_doc.sentences):
        for tok in sent.tokens:
            pred_flat.append((tok, sent_idx))
    
    # Build indices for fast lookup
    # For tokid matching, we need to handle MWTs: multiple tokens/subtokens can share the same tokid
    # Build a mapping from tokid to list of indices (for MWTs, multiple indices share the same tokid)
    tokid_to_pred_indices: dict[str, list[int]] = {}
    for idx, (tok, _) in enumerate(pred_flat):
        if tok.tokid:
            if tok.tokid not in tokid_to_pred_indices:
                tokid_to_pred_indices[tok.tokid] = []
            tokid_to_pred_indices[tok.tokid].append(idx)
        # Also check subtokens for MWTs - they should share the parent's tokid
        if tok.is_mwt and tok.subtokens:
            for sub in tok.subtokens:
                if sub.tokid:
                    if sub.tokid not in tokid_to_pred_indices:
                        tokid_to_pred_indices[sub.tokid] = []
                    tokid_to_pred_indices[sub.tokid].append(idx)
    
    # Normalize forms for matching
    gold_norm = [_normalize_form_for_alignment(tok.form) for tok, _ in gold_flat]
    pred_norm = [_normalize_form_for_alignment(tok.form) for tok, _ in pred_flat]
    
    # Build form-based index (for fallback matching)
    from collections import defaultdict, deque
    pred_form_index: dict[str, deque[int]] = defaultdict(deque)
    for idx, norm in enumerate(pred_norm):
        if norm:
            pred_form_index[norm].append(idx)
    
    # Build sentence boundary maps for faster lookup
    gold_sent_boundaries: dict[int, tuple[int, int]] = {}
    pred_sent_boundaries: dict[int, tuple[int, int]] = {}
    gold_idx = 0
    for sent_idx, sent in enumerate(gold_doc.sentences):
        start = gold_idx
        gold_idx += len(sent.tokens)
        gold_sent_boundaries[sent_idx] = (start, gold_idx)
    pred_idx = 0
    for sent_idx, sent in enumerate(pred_doc.sentences):
        start = pred_idx
        pred_idx += len(sent.tokens)
        pred_sent_boundaries[sent_idx] = (start, pred_idx)
    
    used_pred_indices: set[int] = set()
    used_tokids: set[str] = set()
    
    # Pre-allocate aligned list
    aligned = [None] * len(gold_flat)
    
    # First pass: exact tokid matches
    # Trust tokids when they match - don't require form matching since tokenization may differ
    # This is especially important for MWTs and cross-treebank evaluation (e.g., PADT vs NYUAD)
    if prefer_tokids:
        for g_idx, (gold_tok, gold_sent_idx) in enumerate(gold_flat):
            if gold_tok.tokid and gold_tok.tokid in tokid_to_pred_indices:
                # For MWTs, multiple indices may share the same tokid - try to find the best match
                candidate_indices = tokid_to_pred_indices[gold_tok.tokid]
                best_candidate_idx = None
                best_candidate = None
                best_pred_sent_idx = None
                best_score = -1
                
                # Try to find an unused candidate, preferring same sentence and similar forms
                for candidate_idx in candidate_indices:
                    if candidate_idx not in used_pred_indices:
                        pred_tok, pred_sent_idx = pred_flat[candidate_idx]
                        # Score: prefer same sentence, and prefer similar forms (but don't require exact match)
                        score = 0
                        if gold_sent_idx == pred_sent_idx:
                            score += 1000
                        # Check if forms are similar (but don't require exact match)
                        gold_form_norm = _normalize_form_for_alignment(gold_tok.form)
                        pred_form_norm = _normalize_form_for_alignment(pred_tok.form)
                        if gold_form_norm == pred_form_norm:
                            score += 100
                        elif gold_form_norm and pred_form_norm and (gold_form_norm in pred_form_norm or pred_form_norm in gold_form_norm):
                            # Partial match (one contains the other) - still acceptable
                            score += 50
                        
                        if score > best_score:
                            best_score = score
                            best_candidate_idx = candidate_idx
                            best_candidate = pred_tok
                            best_pred_sent_idx = pred_sent_idx
                
                # Accept the match if we found a candidate, even if forms don't match exactly
                # This allows tokid-based alignment to work across different tokenization schemes
                if best_candidate_idx is not None:
                    aligned[g_idx] = (gold_tok, best_candidate, gold_sent_idx, best_pred_sent_idx)
                    used_pred_indices.add(best_candidate_idx)
                    used_tokids.add(gold_tok.tokid)
    
    # Second pass: form-based matching within sentence boundaries
    for g_idx, (gold_tok, gold_sent_idx) in enumerate(gold_flat):
        # Skip if already matched
        if aligned[g_idx] is not None:
            continue
        
        best_pred: Optional[Token] = None
        best_pred_sent_idx: Optional[int] = None
        best_pred_idx: Optional[int] = None
        best_score = -1
        
        gold_norm_form = gold_norm[g_idx]
        if not gold_norm_form:
            aligned[g_idx] = (gold_tok, None, gold_sent_idx, None)
            continue
        
        # Determine search range
        if prefer_sentence_boundaries and gold_sent_idx < len(gold_doc.sentences):
            # Try to match within the same sentence first
            gold_sent_start, gold_sent_end = gold_sent_boundaries[gold_sent_idx]
            # Try corresponding sentence in predicted doc
            if gold_sent_idx < len(pred_doc.sentences):
                pred_sent_start, pred_sent_end = pred_sent_boundaries[gold_sent_idx]
                search_start = max(0, pred_sent_start - max_distance)
                search_end = min(len(pred_flat), pred_sent_end + max_distance)
            else:
                # No corresponding sentence, search nearby
                search_start = max(0, g_idx - max_distance)
                search_end = min(len(pred_flat), g_idx + max_distance)
        else:
            # Search within max_distance
            search_start = max(0, g_idx - max_distance)
            search_end = min(len(pred_flat), g_idx + max_distance)
        
        # Look for exact form match in search range
        for p_idx in range(search_start, search_end):
            if p_idx in used_pred_indices:
                continue
            pred_tok, pred_sent_idx = pred_flat[p_idx]
            pred_norm_form = pred_norm[p_idx]
            
            if gold_norm_form == pred_norm_form:
                # Calculate score: prefer closer matches and same sentence
                score = 1000
                distance = abs(g_idx - p_idx)
                score -= distance
                if gold_sent_idx == pred_sent_idx:
                    score += 100
                if score > best_score:
                    best_score = score
                    best_pred = pred_tok
                    best_pred_sent_idx = pred_sent_idx
                    best_pred_idx = p_idx
        
        # If no match found in search range, try form index (but with lower priority)
        if best_pred is None:
            deque_indices = pred_form_index.get(gold_norm_form)
            if deque_indices:
                # Try indices near the current position
                candidates = list(deque_indices)
                candidates.sort(key=lambda idx: abs(idx - g_idx))
                for candidate_idx in candidates:
                    if candidate_idx in used_pred_indices:
                        continue
                    # Still respect max_distance if prefer_sentence_boundaries
                    if prefer_sentence_boundaries and abs(candidate_idx - g_idx) > max_distance * 2:
                        continue
                    best_pred, best_pred_sent_idx = pred_flat[candidate_idx]
                    best_pred_idx = candidate_idx
                    break
        
        if best_pred_idx is not None:
            used_pred_indices.add(best_pred_idx)
        
        aligned[g_idx] = (gold_tok, best_pred, gold_sent_idx, best_pred_sent_idx)
    
    return aligned


def _iter_atomic_tokens(sentence: Sentence) -> Iterable[Token]:
    for token in sentence.tokens:
        if token.is_mwt and token.subtokens:
            for sub in token.subtokens:
                yield sub
        else:
            yield token


def _detag_document(doc: Document, merge_mwts: bool = False) -> Document:
    """
    Strip annotations from a document.
    
    Args:
        doc: Input document
        merge_mwts: If True, merge MWT subtokens back into single tokens (for tokenized mode)
    """
    stripped = Document(id=doc.id, sentences=[])
    for sentence in doc.sentences:
        new_sentence = Sentence(
            id=sentence.id,
            sent_id=sentence.sent_id,
            text=sentence.text,
            tokens=[],
        )
        for token in sentence.tokens:
            if merge_mwts and token.is_mwt and token.subtokens:
                # Merge MWT back into single token: use the MWT form (e.g., "don't")
                # If MWT form is empty, reconstruct from subtokens
                mwt_form = token.form
                if not mwt_form or mwt_form == "_":
                    # Reconstruct form from subtokens
                    sub_forms = [sub.form for sub in token.subtokens if sub.form]
                    mwt_form = "".join(sub_forms) if sub_forms else token.form
                
                # Preserve tokid from MWT parent token
                mwt_tokid = token.tokid
                
                new_token = Token(
                    id=token.id if token.id > 0 else (token.subtokens[0].id if token.subtokens else 0),
                    form=mwt_form,  # Use the MWT form (e.g., "don't")
                    lemma="_",
                    xpos="_",
                    upos="_",
                    feats="_",
                    reg="",
                    expan="",
                    is_mwt=False,  # No longer an MWT
                    mwt_start=0,
                    mwt_end=0,
                    parts=[],
                    subtokens=[],
                    source="",  # Clear source to prevent data leakage
                    head=0,
                    deprel="",
                    deps="",
                    misc="",
                    space_after=token.subtokens[-1].space_after if token.subtokens else token.space_after,
                    tokid=mwt_tokid,  # Preserve tokid
                )
                new_sentence.tokens.append(new_token)
            else:
                # Keep token structure (including MWTs if not merging)
                new_token = Token(
                    id=token.id,
                    form=token.form,
                    lemma="_",
                    xpos="_",
                    upos="_",
                    feats="_",
                    reg="",
                    expan="",
                    is_mwt=token.is_mwt,
                    mwt_start=token.mwt_start,
                    mwt_end=token.mwt_end,
                    parts=list(token.parts),
                    subtokens=[],
                    source="",  # Clear source to prevent data leakage
                    head=0,
                    deprel="",
                    deps="",
                    misc="",
                    space_after=token.space_after,
                    tokid=token.tokid,  # Preserve tokid
                )
                if token.is_mwt and token.subtokens and not merge_mwts:
                    for sub in token.subtokens:
                        new_sub = SubToken(
                            id=sub.id,
                            form=sub.form,
                            reg="",
                            expan="",
                            lemma="_",
                            xpos="_",
                            upos="_",
                            feats="_",
                            source="",  # Clear source to prevent data leakage
                            space_after=sub.space_after,
                            tokid=sub.tokid,  # Preserve subtoken tokid
                        )
                        new_token.subtokens.append(new_sub)
                new_sentence.tokens.append(new_token)
        stripped.sentences.append(new_sentence)
    return stripped


def _flatten(doc: Document) -> List[Token]:
    tokens: List[Token] = []
    for sentence in doc.sentences:
        tokens.extend(_iter_atomic_tokens(sentence))
    return tokens


def _write_detagged_conllu(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8", errors="replace") as src, target.open("w", encoding="utf-8", errors="replace") as dst:
        for line in src:
            if not line or line.startswith("#") or line.strip() == "":
                dst.write(line)
                continue
            parts = line.rstrip("\n").split("\t")
            if "-" in parts[0]:
                dst.write(line)
                continue
            for idx in range(2, min(len(parts), 9)):
                parts[idx] = "_"
            dst.write("\t".join(parts) + "\n")


def _write_tokenized_text(doc: Document, path: Path) -> None:
    with path.open("w", encoding="utf-8", errors="replace") as handle:
        for sentence in doc.sentences:
            tokens = [tok.form for tok in _iter_atomic_tokens(sentence)]
            handle.write(" ".join(tokens) + "\n")


def _write_plain_text(doc: Document, path: Path) -> None:
    with path.open("w", encoding="utf-8", errors="replace") as handle:
        for sentence in doc.sentences:
            text = sentence.text
            if not text:
                tokens = [tok.form for tok in _iter_atomic_tokens(sentence)]
                text = " ".join(tokens)
            handle.write(text + "\n")


def _load_gold(path: Path, fmt: str) -> Document:
    # Auto-detect format if needed
    if fmt == "auto":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                sample = handle.read(4096)
        except OSError as exc:
            raise ValueError(f"Failed to read input file '{path}': {exc}") from exc

        stripped = sample.lstrip()
        if "<tok" in sample or "<TEI" in sample or stripped.startswith("<TEI"):
            fmt = "teitok"
        else:
            for line in sample.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith("#"):
                    if stripped_line.startswith("# text") or stripped_line.startswith("# sent_id") or stripped_line.startswith("# newdoc"):
                        fmt = "conllu"
                        break
                    continue
                if "\t" in stripped_line and len(stripped_line.split("\t")) >= 10:
                    fmt = "conllu"
                    break
            if fmt == "auto":
                fmt = "conllu"  # Default to conllu if we can't determine
    
    if fmt == "teitok":
        from .teitok import load_teitok

        return load_teitok(str(path))
    if fmt == "conllu":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            # Add tokids when loading CoNLL-U for better alignment with neural backends
            return conllu_to_document(handle.read(), doc_id=str(path), add_tokids=True)
    raise ValueError(f"Unsupported gold format: {fmt}")


def evaluate_model(
    gold_path: Path,
    output_dir: Path,
    gold_format: str,
    *,
    model_path: Optional[Path] = None,
    neural_backend: Optional[BackendManager] = None,
    verbose: bool = False,
    debug: bool = False,
    mode: str = "auto",
    create_implicit_mwt: bool = False,
    unicode_normalize: str = "none",
) -> Path:
    start_time = time.time()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect format if auto, so we can use it for mode resolution
    detected_format = gold_format
    if gold_format == "auto":
        try:
            with open(gold_path, "r", encoding="utf-8", errors="ignore") as handle:
                sample = handle.read(4096)
        except OSError as exc:
            raise ValueError(f"Failed to read input file '{gold_path}': {exc}") from exc

        stripped = sample.lstrip()
        if "<tok" in sample or "<TEI" in sample or stripped.startswith("<TEI"):
            detected_format = "teitok"
        else:
            for line in sample.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith("#"):
                    if stripped_line.startswith("# text") or stripped_line.startswith("# sent_id") or stripped_line.startswith("# newdoc"):
                        detected_format = "conllu"
                        break
                    continue
                if "\t" in stripped_line and len(stripped_line.split("\t")) >= 10:
                    detected_format = "conllu"
                    break
            if detected_format == "auto":
                detected_format = "conllu"  # Default to conllu if we can't determine
    
    if verbose:
        print(f"[flexipipe] Loading gold standard from {gold_path}...")
    load_start = time.time()
    gold_doc = _load_gold(gold_path, detected_format)
    # Normalize gold document
    if unicode_normalize != "none":
        gold_doc.normalize_unicode(unicode_normalize)
    load_time = time.time() - load_start
    gold_token_count = sum(len(sent.tokens) for sent in gold_doc.sentences)
    gold_sent_count = len(gold_doc.sentences)
    if verbose:
        print(f"[flexipipe] Loaded {gold_sent_count} sentences, {gold_token_count} tokens in {load_time:.2f}s")

    mode_normalized = mode.lower() if mode else 'auto'
    valid_modes = {"auto", "raw", "tokenized", "split"}
    if mode_normalized not in valid_modes:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    if mode_normalized == "auto":
        if detected_format == "teitok":
            # Check if TEI is tokenized/segmented
            from .teitok import teitok_has_tokens
            has_tokens = teitok_has_tokens(str(gold_path))
            if has_tokens:
                # Check if it has sentence segmentation (<s> elements or sentence boundaries)
                # For now, assume tokenized TEI should use "split" to preserve MWTs if any
                # But "tokenized" is also reasonable - let's use "split" to be safe
                mode_resolved = "split"
            else:
                mode_resolved = "raw"
        elif detected_format == "conllu":
            mode_resolved = "tokenized"
        else:
            mode_resolved = "raw"
    else:
        mode_resolved = mode_normalized

    # Prepare document based on mode:
    # - "tokenized": merge MWTs back into single tokens (preserve tokenization, but not splitting)
    # - "split": keep MWTs as-is (preserve both tokenization and splitting)
    # - "raw": keep structure but will be re-tokenized
    merge_mwts = (mode_resolved == "tokenized")
    stripped_doc = _detag_document(gold_doc, merge_mwts=merge_mwts)

    # Use neural backend if provided, otherwise use flexitag
    model_str = None
    tag_time = 0.0
    pred_token_count = 0
    pred_sent_count = 0
    if neural_backend:
        # Use neural backend for tagging
        # For tokenized mode, we merge MWTs to preserve tokenization as much as possible.
        # However, neural backends (especially SpaCy) will still re-tokenize the text,
        # which may cause token count discrepancies. This is expected behavior.
        if verbose:
            backend_type = getattr(neural_backend, "_backend_name", None)
            if not backend_type:
                backend_type = type(neural_backend).__name__.replace("Backend", "").lower()
            backend_info = "neural backend"
            if hasattr(neural_backend, '_model_name') and neural_backend._model_name:
                backend_info = f"{backend_type}: {neural_backend._model_name}"
            elif hasattr(neural_backend, '_model_path') and neural_backend._model_path:
                from pathlib import Path
                backend_info = f"{backend_type}: {Path(neural_backend._model_path).name}"
            elif hasattr(neural_backend, '_language') and neural_backend._language:
                backend_info = f"{backend_type}: {neural_backend._language} (blank)"
            elif hasattr(neural_backend, 'nlp') and hasattr(neural_backend.nlp, 'meta') and neural_backend.nlp.meta.get('name'):
                backend_info = f"{backend_type}: {neural_backend.nlp.meta.get('name', 'unknown')}"
            print(f"[flexipipe] Tagging with {backend_info} (mode={mode_resolved})...")
        tag_start = time.time()
        # For raw mode, use raw text; for tokenized/split modes, use pre-tokenized
        use_raw_text = (mode_resolved == "raw")
        neural_result = neural_backend.tag(stripped_doc, use_raw_text=use_raw_text)
        tag_time = time.time() - tag_start
        pred_doc = neural_result.document
        # Normalize pred document
        if unicode_normalize != "none":
            pred_doc.normalize_unicode(unicode_normalize)
        pred_token_count = sum(len(sent.tokens) for sent in pred_doc.sentences)
        pred_sent_count = len(pred_doc.sentences)
        if verbose:
            print(f"[flexipipe] Tagged {pred_sent_count} sentences, {pred_token_count} tokens in {tag_time:.2f}s")
            if tag_time > 0:
                print(f"[flexipipe] Speed: {pred_token_count/tag_time:.0f} tokens/s, {pred_sent_count/tag_time:.1f} sentences/s")
        # Determine model string from neural backend
        backend_type = type(neural_backend).__name__
        if backend_type == "SpacyBackend":
            # Try to get model name from SpaCy backend
            if hasattr(neural_backend, '_model_name') and neural_backend._model_name:
                model_str = f"SpaCy: {neural_backend._model_name}"
            elif hasattr(neural_backend, '_model_path') and neural_backend._model_path:
                model_str = f"SpaCy: {neural_backend._model_path.name}"
            elif hasattr(neural_backend, '_language') and neural_backend._language:
                model_str = f"SpaCy: {neural_backend._language} (blank)"
            else:
                model_str = "SpaCy: unknown"
        elif backend_type == "UDPipeRESTBackend":
            model_name = getattr(neural_backend, '_model_name', None) or "unknown"
            model_str = f"UDPipe: {model_name}"
        elif backend_type == "UDMorphRESTBackend":
            model_name = getattr(neural_backend, '_model_name', None) or "unknown"
            model_str = f"UDMorph: {model_name}"
        else:
            model_str = f"{backend_type.replace('Backend', '')}: unknown"
    elif model_path:
        # Use flexitag
        # Read tagpos from model metadata to ensure we use the correct tag attribute
        # that the model was trained with (e.g., "upos" if model has tag_attribute="upos")
        tagpos = "xpos"  # default
        try:
            with open(model_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            tag_attribute = metadata.get("tag_attribute", "xpos")
            # Map tag_attribute to tagpos
            if tag_attribute == "upos":
                tagpos = "upos"
            elif tag_attribute == "utot" or tag_attribute == "upos#feats":
                tagpos = "utot"
            else:
                tagpos = "xpos"
        except Exception:
            pass  # Use default if we can't read it
        
        if verbose:
            print(f"[flexipipe] Tagging with flexitag model {model_path.name}...")
        tag_start = time.time()
        fallback = FlexitagFallback(str(model_path), options={"tagpos": tagpos})
        tagged_result = fallback.tag(stripped_doc)
        tag_time = time.time() - tag_start
        pred_doc = tagged_result.document
        # Normalize pred document
        if unicode_normalize != "none":
            pred_doc.normalize_unicode(unicode_normalize)
        pred_token_count = sum(len(sent.tokens) for sent in pred_doc.sentences)
        pred_sent_count = len(pred_doc.sentences)
        model_str = f"flexitag: {model_path.name}"
        if verbose:
            print(f"[flexipipe] Tagged {pred_sent_count} sentences, {pred_token_count} tokens in {tag_time:.2f}s")
            if tag_time > 0:
                print(f"[flexipipe] Speed: {pred_token_count/tag_time:.0f} tokens/s, {pred_sent_count/tag_time:.1f} sentences/s")
    else:
        raise ValueError("Must provide either model_path or neural_backend")

    # Apply create_implicit_mwt to predicted document if requested
    # This should be done before evaluation so that the evaluation sees the MWTs
    if create_implicit_mwt:
        from .conllu import _create_implicit_mwt
        new_pred_doc = Document(id=pred_doc.id, meta=dict(pred_doc.meta))
        for sent in pred_doc.sentences:
            new_sent = _create_implicit_mwt(sent)
            new_pred_doc.sentences.append(new_sent)
        pred_doc = new_pred_doc

    if verbose:
        print(f"[flexipipe] Writing output files...")
    eval_start = time.time()

    tagged_path = output_dir / "predicted.conllu"
    with tagged_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document_to_conllu(pred_doc, model=model_str, create_implicit_mwt=create_implicit_mwt))

    if detected_format == "conllu":
        _write_detagged_conllu(gold_path, output_dir / "test_detagged_with_splits.conllu")
    _write_tokenized_text(gold_doc, output_dir / "test_tokenized.txt")
    _write_plain_text(gold_doc, output_dir / "test_plain.txt")

    if verbose:
        print(f"[flexipipe] Evaluating alignment and metrics...")

    # For evaluation, we always compare the original gold_doc (with annotations) to predicted
    # The stripped_doc is only used for tagging, not for evaluation
    # In tokenized mode, we still compare at token level (not subtoken level) even if MWTs were merged for tagging
    eval_gold_doc = gold_doc
    
    # Align tokens using centralized alignment function
    # Use sentence boundaries for better alignment when tokenization differs
    aligned_tokens = align_gold_and_predicted(
        eval_gold_doc,
        pred_doc,
        prefer_tokids=True,
        prefer_sentence_boundaries=True,
        max_distance=20,  # Allow some flexibility for tokenization differences
    )
    
    gold_token_count = sum(len(sent.tokens) for sent in eval_gold_doc.sentences)
    pred_token_count = sum(len(sent.tokens) for sent in pred_doc.sentences)
    matched_count = sum(1 for _, pred, _, _ in aligned_tokens if pred is not None)
    
    if debug:
        print(f"INFO: Token alignment: {matched_count}/{gold_token_count} gold tokens matched with predicted tokens")
        print(f"      Gold tokens: {gold_token_count}, Predicted tokens: {pred_token_count}")
        print(f"      Predicted document has {len(pred_doc.sentences)} sentences")
        if pred_doc.sentences:
            print(f"      First predicted sentence has {len(pred_doc.sentences[0].tokens)} tokens")
            if pred_doc.sentences[0].tokens:
                print(f"      First predicted token: '{pred_doc.sentences[0].tokens[0].form}' (tokid={pred_doc.sentences[0].tokens[0].tokid}, is_mwt={pred_doc.sentences[0].tokens[0].is_mwt})")
        
        # Count MWTs in predicted document
        pred_mwt_count = sum(1 for sent in pred_doc.sentences for tok in sent.tokens if tok.is_mwt)
        gold_mwt_count = sum(1 for sent in eval_gold_doc.sentences for tok in sent.tokens if tok.is_mwt)
        print(f"      Gold MWTs: {gold_mwt_count}, Predicted MWTs: {pred_mwt_count}")
        
        # Count tokens with same tokid (should indicate unmerged MWTs)
        # Only do this if we suspect an issue (to avoid unnecessary work)
        if pred_token_count > gold_token_count or pred_mwt_count < gold_mwt_count:
            tokid_counts = {}
            for sent in pred_doc.sentences:
                for tok in sent.tokens:
                    if tok.tokid:
                        tokid_counts[tok.tokid] = tokid_counts.get(tok.tokid, 0) + 1
            duplicate_tokids = {tid: count for tid, count in tokid_counts.items() if count > 1}
            if duplicate_tokids:
                print(f"      WARNING: Found {len(duplicate_tokids)} tokids with multiple tokens (unmerged MWTs?)")
                if len(duplicate_tokids) <= 5:
                    for tid, count in list(duplicate_tokids.items())[:5]:
                        print(f"        tokid {tid}: {count} tokens")
            else:
                print(f"      INFO: No duplicate tokids found - MWTs may not have tokids set correctly")
    
    # Debug output: side-by-side comparison of first 3 sentences
    if debug:
        print("\n" + "="*80)
        print("DEBUG: Side-by-side comparison of first 3 sentences")
        print("="*80)
        for sent_idx in range(min(3, len(eval_gold_doc.sentences))):
            gold_sent = eval_gold_doc.sentences[sent_idx]
            pred_sent = pred_doc.sentences[sent_idx] if sent_idx < len(pred_doc.sentences) else None
            
            print(f"\n--- Sentence {sent_idx + 1} ---")
            print(f"Gold sent_id: {gold_sent.sent_id or gold_sent.id or 'N/A'}")
            if pred_sent:
                print(f"Pred sent_id: {pred_sent.sent_id or pred_sent.id or 'N/A'}")
            print(f"Gold text: {gold_sent.text[:100] if gold_sent.text else 'N/A'}")
            if pred_sent:
                print(f"Pred text: {pred_sent.text[:100] if pred_sent.text else 'N/A'}")
            
            print(f"\nGold tokens ({len(gold_sent.tokens)}):")
            for tok in gold_sent.tokens:
                mwt_info = f" [MWT: {len(tok.subtokens)} subtokens]" if tok.is_mwt and tok.subtokens else ""
                upos_str = f" upos={tok.upos}" if tok.upos else ""
                xpos_str = f" xpos={tok.xpos}" if tok.xpos else ""
                feats_str = f" feats={tok.feats}" if tok.feats else ""
                lemma_str = f" lemma={tok.lemma}" if tok.lemma else ""
                head_str = f" head={tok.head}" if tok.head else ""
                deprel_str = f" deprel={tok.deprel}" if tok.deprel else ""
                print(f"  {tok.id}: '{tok.form}' tokid={tok.tokid}{upos_str}{xpos_str}{feats_str}{lemma_str}{head_str}{deprel_str}{mwt_info}")
                if tok.is_mwt and tok.subtokens:
                    for sub in tok.subtokens:
                        upos_str = f" upos={sub.upos}" if sub.upos else ""
                        xpos_str = f" xpos={sub.xpos}" if sub.xpos else ""
                        feats_str = f" feats={sub.feats}" if sub.feats else ""
                        lemma_str = f" lemma={sub.lemma}" if sub.lemma else ""
                        print(f"    {sub.id}: '{sub.form}' tokid={sub.tokid}{upos_str}{xpos_str}{feats_str}{lemma_str}")
            
            if pred_sent:
                print(f"\nPred tokens ({len(pred_sent.tokens)}):")
                for tok in pred_sent.tokens:
                    mwt_info = f" [MWT: {len(tok.subtokens)} subtokens]" if tok.is_mwt and tok.subtokens else ""
                    upos_str = f" upos={tok.upos}" if tok.upos else ""
                    xpos_str = f" xpos={tok.xpos}" if tok.xpos else ""
                    feats_str = f" feats={tok.feats}" if tok.feats else ""
                    lemma_str = f" lemma={tok.lemma}" if tok.lemma else ""
                    head_str = f" head={tok.head}" if tok.head else ""
                    deprel_str = f" deprel={tok.deprel}" if tok.deprel else ""
                    print(f"  {tok.id}: '{tok.form}' tokid={tok.tokid}{upos_str}{xpos_str}{feats_str}{lemma_str}{head_str}{deprel_str}{mwt_info}")
                    if tok.is_mwt and tok.subtokens:
                        for sub in tok.subtokens:
                            upos_str = f" upos={sub.upos}" if sub.upos else ""
                            xpos_str = f" xpos={sub.xpos}" if sub.xpos else ""
                            feats_str = f" feats={sub.feats}" if sub.feats else ""
                            lemma_str = f" lemma={sub.lemma}" if sub.lemma else ""
                            print(f"    {sub.id}: '{sub.form}' tokid={sub.tokid}{upos_str}{xpos_str}{feats_str}{lemma_str}")
            else:
                print("\nPred: No corresponding sentence found")
            
            # Show alignment for this sentence with attribute comparison
            print(f"\nAlignment for sentence {sent_idx + 1}:")
            sent_aligned = [(g, p, gs, ps) for g, p, gs, ps in aligned_tokens if gs == sent_idx]
            for gold_tok, pred_tok, _, pred_sent_idx in sent_aligned:
                if pred_tok:
                    match_type = "tokid" if gold_tok.tokid and gold_tok.tokid == pred_tok.tokid else "form"
                    # Compare attributes
                    upos_match = "✓" if gold_tok.upos == (pred_tok.upos or "") else "✗"
                    xpos_match = "✓" if gold_tok.xpos == (pred_tok.xpos or "") else "✗"
                    feats_match = "✓" if gold_tok.feats == (pred_tok.feats or "") else "✗"
                    lemma_match = "✓" if gold_tok.lemma == (pred_tok.lemma or "") else "✗"
                    head_match = "✓" if str(gold_tok.head) == str(pred_tok.head or "") else "✗"
                    deprel_match = "✓" if gold_tok.deprel == (pred_tok.deprel or "") else "✗"
                    print(f"  Gold '{gold_tok.form}' (tokid={gold_tok.tokid}) <-> Pred '{pred_tok.form}' (tokid={pred_tok.tokid}) [{match_type}]")
                    print(f"    UPOS: {gold_tok.upos or '_'} {upos_match} {pred_tok.upos or '_'}")
                    print(f"    XPOS: {gold_tok.xpos or '_'} {xpos_match} {pred_tok.xpos or '_'}")
                    print(f"    FEATS: {gold_tok.feats or '_'} {feats_match} {pred_tok.feats or '_'}")
                    print(f"    LEMMA: {gold_tok.lemma or '_'} {lemma_match} {pred_tok.lemma or '_'}")
                    if gold_tok.head or pred_tok.head:
                        print(f"    HEAD: {gold_tok.head or '_'} {head_match} {pred_tok.head or '_'}")
                    if gold_tok.deprel or pred_tok.deprel:
                        print(f"    DEPREL: {gold_tok.deprel or '_'} {deprel_match} {pred_tok.deprel or '_'}")
                else:
                    print(f"  Gold '{gold_tok.form}' (tokid={gold_tok.tokid}) <-> [NO MATCH]")
        
        print("\n" + "="*80)

    metrics = {
        "tokenization": Metric(),
        "upos": Metric(),
        "xpos": Metric(),
        "feats": Metric(),
        "feats_partial": Metric(),
        "upos+feats": Metric(),
        "lemma": Metric(),
        "reg": Metric(),
        "expan": Metric(),
    }

    # Helper to get components (subtokens for MWTs, or the token itself)
    # In tokenized mode, tokens should not have subtokens (MWTs have been merged)
    # Optimized: avoid creating new lists when not needed
    def components(token: Token) -> List[Token | SubToken]:
        # In tokenized mode, we compare at token level (no subtokens)
        if mode_resolved == "tokenized":
            return [token]
        # In split/raw mode, compare at subtoken level if MWT exists
        if token.is_mwt and token.subtokens:
            # Return the list directly (it's already a list, no need to copy)
            return token.subtokens
        return [token]

    # Track tokenization differences for debug output
    tokenization_diffs = []
    
    # Evaluate on aligned tokens
    for gold_tok, pred_tok, gold_sent_idx, pred_sent_idx in aligned_tokens:
        if pred_tok is None:
            # No matching predicted token - mark all as incorrect
            gold_parts = components(gold_tok)
            for gold_part in gold_parts:
                if gold_part.upos:
                    metrics["upos"].add(False)
                if gold_part.xpos:
                    metrics["xpos"].add(False)
                # FEATS: Always mark as incorrect (no matching predicted token)
                gold_feats = gold_part.feats if gold_part.feats is not None else ""
                if gold_feats == "_":
                    gold_feats = ""
                metrics["feats"].add(False)
                metrics["feats_partial"].add(False)
                if gold_part.upos or gold_feats:
                    metrics["upos+feats"].add(False)
                if gold_part.lemma:
                    metrics["lemma"].add(False)
                if gold_part.reg:
                    metrics["reg"].add(False)
                if gold_part.expan:
                    metrics["expan"].add(False)
            continue
        
        # Tokenization: compare orthographic token forms
        form_match = gold_tok.form == pred_tok.form
        metrics["tokenization"].add(form_match)
        
        # Collect tokenization differences for debug output
        if not form_match and len(tokenization_diffs) < 10:
            gold_sent = eval_gold_doc.sentences[gold_sent_idx] if gold_sent_idx >= 0 and gold_sent_idx < len(eval_gold_doc.sentences) else None
            pred_sent = pred_doc.sentences[pred_sent_idx] if pred_sent_idx is not None and pred_sent_idx >= 0 and pred_sent_idx < len(pred_doc.sentences) else None
            tokenization_diffs.append({
                "gold": gold_tok,
                "pred": pred_tok,
                "gold_sent": gold_sent,
                "pred_sent": pred_sent,
                "gold_sent_idx": gold_sent_idx,
                "pred_sent_idx": pred_sent_idx,
            })
        
        # For other metrics, compare at the component level (subtokens for MWTs)
        gold_parts = components(gold_tok)
        pred_parts = components(pred_tok)
        
        # Only compare if we have the same number of parts
        if len(gold_parts) == len(pred_parts):
            for gold_part, pred_part in zip(gold_parts, pred_parts):
                if gold_part.upos:
                    metrics["upos"].add(gold_part.upos == (pred_part.upos or ""))
                if gold_part.xpos:
                    metrics["xpos"].add(gold_part.xpos == (pred_part.xpos or ""))
                gold_feats = gold_part.feats if gold_part.feats is not None else ""
                pred_feats = pred_part.feats if pred_part.feats is not None else ""
                if gold_feats == "_":
                    gold_feats = ""
                if pred_feats == "_":
                    pred_feats = ""
                metrics["feats"].add(gold_feats == pred_feats)
                metrics["feats_partial"].add(_feats_partial_match(gold_feats, pred_feats))
                # Evaluate UPOS+FEATS combination (both must match)
                if gold_part.upos or gold_feats:
                    gold_upos = gold_part.upos or ""
                    pred_upos = pred_part.upos or ""
                    metrics["upos+feats"].add(
                        gold_upos == pred_upos and gold_feats == pred_feats
                    )
                if gold_part.lemma:
                    metrics["lemma"].add((pred_part.lemma or "") == gold_part.lemma)
                if gold_part.reg:
                    metrics["reg"].add((pred_part.reg or "") == gold_part.reg)
                if gold_part.expan:
                    metrics["expan"].add((pred_part.expan or "") == gold_part.expan)
        else:
            # Different number of parts - mark all fields as incorrect for this token
            for gold_part in gold_parts:
                if gold_part.upos:
                    metrics["upos"].add(False)
                if gold_part.xpos:
                    metrics["xpos"].add(False)
                # FEATS: Always mark as incorrect (no matching predicted token)
                gold_feats = gold_part.feats if gold_part.feats is not None else ""
                if gold_feats == "_":
                    gold_feats = ""
                    metrics["feats"].add(False)
                if gold_part.upos or gold_feats:
                    metrics["upos+feats"].add(False)
                if gold_part.lemma:
                    metrics["lemma"].add(False)
                if gold_part.reg:
                    metrics["reg"].add(False)
                if gold_part.expan:
                    metrics["expan"].add(False)
    
    # Evaluate parsing (UAS and LAS) if head and deprel are available
    # Build token ID to token mapping for gold and predicted
    gold_token_map: dict[tuple[int, int], Token] = {}  # (sentence_idx, token_id) -> token
    pred_token_map: dict[tuple[int, int], Token] = {}
    
    for sent_idx, sent in enumerate(gold_doc.sentences):
        for tok in sent.tokens:
            if tok.id > 0:
                gold_token_map[(sent_idx, tok.id)] = tok
    
    for sent_idx, sent in enumerate(pred_doc.sentences):
        for tok in sent.tokens:
            if tok.id > 0:
                pred_token_map[(sent_idx, tok.id)] = tok
    
    # Build reverse mapping: (gold_sent_idx, gold_token_id) -> pred_token for fast head alignment lookup
    # Use (sent_idx, token_id) tuple as key since Token objects are not hashable
    gold_to_pred_map: dict[tuple[int, int], Token] = {}
    for gold_tok, pred_tok, gold_sent_idx, _ in aligned_tokens:
        if pred_tok is not None and gold_tok.id > 0:
            gold_to_pred_map[(gold_sent_idx, gold_tok.id)] = pred_tok
    
    # Evaluate UAS and LAS on aligned tokens
    uas_correct = 0
    las_correct = 0
    parsing_total = 0
    
    for gold_tok, pred_tok, gold_sent_idx, pred_sent_idx in aligned_tokens:
        if pred_tok is None:
            continue
        
        # Evaluate parsing if both have deprel information (indicates parsing was done)
        # Check if deprel exists (non-empty string)
        if gold_tok.deprel and pred_tok.deprel and gold_tok.deprel.strip() and pred_tok.deprel.strip():
            try:
                gold_head = int(gold_tok.head) if gold_tok.head else 0
                pred_head = int(pred_tok.head) if pred_tok.head else 0
                
                if gold_sent_idx >= 0 and pred_sent_idx is not None and pred_sent_idx >= 0:
                    parsing_total += 1
                    
                    # Handle root tokens (head = 0)
                    if gold_head == 0 and pred_head == 0:
                        # Both are root - UAS correct
                        uas_correct += 1
                        # Check deprel for LAS
                        if gold_tok.deprel == pred_tok.deprel:
                            las_correct += 1
                    elif gold_head > 0 and pred_head > 0:
                        # Both have heads - get the actual head tokens
                        gold_head_tok = gold_token_map.get((gold_sent_idx, gold_head))
                        pred_head_tok = pred_token_map.get((pred_sent_idx, pred_head))
                        
                        # Align head tokens
                        if gold_head_tok and pred_head_tok:
                            # Use reverse mapping for O(1) lookup instead of O(n) iteration
                            gold_head_key = (gold_sent_idx, gold_head)
                            gold_head_aligned = gold_to_pred_map.get(gold_head_key)
                            
                            if gold_head_aligned == pred_head_tok:
                                uas_correct += 1
                                # Check if deprel also matches (LAS)
                                if gold_tok.deprel == pred_tok.deprel:
                                    las_correct += 1
            except (ValueError, TypeError, AttributeError):
                pass

    evaluate_splitting = mode_resolved != "split"
    gold_split_total = 0
    predicted_split_total = 0
    split_covered = 0
    split_correct_forms = 0
    coverage = None
    precision = None
    part_form_accuracy = None

    if evaluate_splitting:
        # Use aligned tokens for splitting evaluation (not positional matching)
        # This allows evaluation even when sentence/token counts differ
        for gold_tok, pred_tok, gold_sent_idx, pred_sent_idx in aligned_tokens:
            if gold_tok is None or pred_tok is None:
                continue
            
            gold_split = bool(gold_tok.is_mwt and gold_tok.subtokens)
            pred_split = bool(pred_tok.is_mwt and pred_tok.subtokens)
            
            if pred_split:
                predicted_split_total += 1
            if gold_split:
                gold_split_total += 1
                if pred_split:
                    split_covered += 1
                    if len(gold_tok.subtokens) == len(pred_tok.subtokens) and all(
                        gs.form == ps.form for gs, ps in zip(gold_tok.subtokens, pred_tok.subtokens)
                    ):
                        split_correct_forms += 1
        
        coverage = (split_covered / gold_split_total) if gold_split_total else None
        precision = (split_covered / predicted_split_total) if predicted_split_total else None
        part_form_accuracy = (split_correct_forms / split_covered) if split_covered else None

    summary = {
        "model": model_str,
        "gold_file": str(gold_path),
        "mode": mode_resolved,
        "metrics": {
            key: {
                "correct": metric.correct,
                "total": metric.total,
                "accuracy": metric.accuracy,
            }
            for key, metric in metrics.items()
        },
        "splitting": {
            "evaluated": evaluate_splitting,
            "gold_total": gold_split_total if evaluate_splitting else None,
            "predicted_total": predicted_split_total if evaluate_splitting else None,
            "covered": split_covered if evaluate_splitting else None,
            "correct_forms": split_correct_forms if evaluate_splitting else None,
            "recall": coverage,
            "precision": precision,
            "part_form_accuracy": part_form_accuracy,
        },
        "segmentation": metrics["tokenization"].accuracy,
        "uas": (uas_correct / parsing_total) if parsing_total > 0 else None,
        "las": (las_correct / parsing_total) if parsing_total > 0 else None,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8", errors="replace") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if verbose:
        label_width = 26
        count_width = 12
        acc_width = 10
        # Account for the space after │ in each column: add 1 to each width for borders
        border_top = f"┌{'─' * (label_width + 1)}┬{'─' * (count_width + 1)}┬{'─' * (acc_width + 1)}┐"
        border_mid = f"├{'─' * (label_width + 1)}┼{'─' * (count_width + 1)}┼{'─' * (acc_width + 1)}┤"
        border_bot = f"└{'─' * (label_width + 1)}┴{'─' * (count_width + 1)}┴{'─' * (acc_width + 1)}┘"
        print("[flexipipe] evaluation summary")
        print(border_top)
        print(f"│ {'Metric':<{label_width}}│ {'Correct':<{count_width}}│ {'Acc.%':<{acc_width}}│")
        print(border_mid)
        
        # Tokenization/Segmentation metrics
        if mode_resolved != "tokenized":
            seg = metrics["tokenization"].accuracy
            if seg is not None:
                seg_str = f"{seg*100:6.2f}%"
                count_str = f"{metrics['tokenization'].correct}/{metrics['tokenization'].total}"
                print(f"│ {'Tokenization':<{label_width}}│ {count_str:<{count_width}}│ {seg_str:<{acc_width}}│")
                print(border_mid)
        
        # Morphological metrics
        metric_order = ("upos", "xpos", "feats", "feats_partial", "upos+feats", "lemma", "reg", "expan")
        for key in metric_order:
            metric = metrics[key]
            if metric.total == 0:
                continue
            acc = metric.accuracy
            acc_str = f"{acc*100:6.2f}%" if acc is not None else "   n/a "
            count_str = f"{metric.correct}/{metric.total}"
            metric_name = key.replace("+", "+").capitalize()
            print(f"│ {metric_name:<{label_width}}│ {count_str:<{count_width}}│ {acc_str:<{acc_width}}│")
        
        # Parsing metrics (UAS and LAS)
        if parsing_total > 0:
            print(border_mid)
            uas = summary["uas"]
            las = summary["las"]
            if uas is not None:
                uas_str = f"{uas*100:6.2f}%"
                uas_count_str = f"{uas_correct}/{parsing_total}"
                print(f"│ {'UAS':<{label_width}}│ {uas_count_str:<{count_width}}│ {uas_str:<{acc_width}}│")
            if las is not None:
                las_str = f"{las*100:6.2f}%"
                las_count_str = f"{las_correct}/{parsing_total}"
                print(f"│ {'LAS':<{label_width}}│ {las_count_str:<{count_width}}│ {las_str:<{acc_width}}│")
        
        print(border_bot)

        if evaluate_splitting:
            split_value_width = 12
            split_border_top = f"┌{'─' * (label_width + 1)}┬{'─' * (split_value_width + 1)}┐"
            split_border_mid = f"├{'─' * (label_width + 1)}┼{'─' * (split_value_width + 1)}┤"
            split_border_bot = f"└{'─' * (label_width + 1)}┴{'─' * (split_value_width + 1)}┘"
            print("[flexipipe] splitting summary")
            print(split_border_top)
            coverage_str = f"{coverage*100:6.2f}%" if coverage is not None else "   n/a "
            precision_str = f"{precision*100:6.2f}%" if precision is not None else "   n/a "
            part_form_str = f"{part_form_accuracy*100:6.2f}%" if part_form_accuracy is not None else "   n/a "
            print(f"│ {'Gold splits':<{label_width}}│ {gold_split_total:<{split_value_width}}│")
            print(split_border_mid)
            print(f"│ {'Predicted splits':<{label_width}}│ {predicted_split_total:<{split_value_width}}│")
            print(split_border_mid)
            print(f"│ {'Recall':<{label_width}}│ {coverage_str:<{split_value_width}}│")
            print(split_border_mid)
            print(f"│ {'Precision':<{label_width}}│ {precision_str:<{split_value_width}}│")
            print(split_border_mid)
            print(f"│ {'Part-form accuracy':<{label_width}}│ {part_form_str:<{split_value_width}}│")
            print(split_border_bot)
        else:
            print("[flexipipe] splitting evaluation skipped (mode=split)")

        # Output tokenization differences only when debug is enabled
        if tokenization_diffs and verbose and debug:
            print(f"\n[flexipipe] First {len(tokenization_diffs)} tokenization differences:")
            for i, diff in enumerate(tokenization_diffs, 1):
                gold_tok = diff["gold"]
                pred_tok = diff["pred"]
                gold_sent = diff["gold_sent"]
                pred_sent = diff["pred_sent"]
                gold_sent_idx = diff["gold_sent_idx"]
                pred_sent_idx = diff["pred_sent_idx"]
                
                print(f"\n  Difference {i}:")
                print(f"    Gold:   '{gold_tok.form}' (tokid={gold_tok.tokid}, id={gold_tok.id}, is_mwt={gold_tok.is_mwt})")
                if gold_tok.is_mwt and gold_tok.subtokens:
                    print(f"            MWT with {len(gold_tok.subtokens)} subtokens: {[s.form for s in gold_tok.subtokens]}")
                print(f"    Pred:   '{pred_tok.form}' (tokid={pred_tok.tokid}, id={pred_tok.id}, is_mwt={pred_tok.is_mwt})")
                if pred_tok.is_mwt and pred_tok.subtokens:
                    print(f"            MWT with {len(pred_tok.subtokens)} subtokens: {[s.form for s in pred_tok.subtokens]}")
                if gold_sent:
                    print(f"    Gold sentence {gold_sent_idx + 1}: {gold_sent.text[:100] if gold_sent.text else 'N/A'}...")
                if pred_sent:
                    print(f"    Pred sentence {pred_sent_idx + 1}: {pred_sent.text[:100] if pred_sent.text else 'N/A'}...")
        
        print(f"[flexipipe] evaluation report written to {metrics_path}")
        
        # Show overall timing
        total_time = time.time() - start_time
        eval_time = time.time() - eval_start
        print(f"[flexipipe] Total time: {total_time:.2f}s (loading: {load_time:.2f}s, tagging: {tag_time:.2f}s, evaluation: {eval_time:.2f}s)")

    return metrics_path


def _feats_partial_match(gold_feats: str, pred_feats: str) -> bool:
    """
    Return True if all predicted feature key/value pairs are correct (match gold).
    This is a "partial" match because:
    - Missing features in prediction are okay (it's partial)
    - Every feature that IS predicted must be correct (match gold)
    - Extra features in prediction that aren't in gold are wrong
    
    This treats "_" or empty strings as "no features".
    
    Note: If gold has no features, prediction must also have no features to be correct.
    If gold has features and prediction has none, that's incorrect (missing required features).
    """

    def parse_feats(feats: str) -> dict[str, str]:
        feats = feats.strip()
        if not feats or feats == "_":
            return {}
        pairs = {}
        for feat in feats.split("|"):
            if "=" in feat:
                key, value = feat.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    pairs[key] = value
        return pairs

    gold_map = parse_feats(gold_feats)
    pred_map = parse_feats(pred_feats)
    
    # If gold has no features, prediction must also have no features
    if not gold_map:
        return not pred_map
    
    # If gold has features but prediction has none, that's incorrect
    if not pred_map:
        return False
    
    # Every feature that is predicted must exist in gold with the same value
    # Missing features in prediction are okay (it's partial)
    for key, value in pred_map.items():
        if key not in gold_map or gold_map[key] != value:
            return False
    return True

