#!/usr/bin/env python3
"""
FlexiPipeTagger: Transformer-based Universal Dependencies tagger and parser with fine-tuning support.

Features:
- BERT-based UPOS/XPOS/FEATS tagging and dependency parsing
- Tokenizer training: Train custom WordPiece tokenizers from corpus
- Sentence segmentation: Rule-based sentence splitting for raw text
- Word tokenization: UD-style tokenization (handles contractions, compounds)
- Respects existing annotations in input
- Handles contractions and MWT (Multi-Word Tokens)
- OOV similarity matching (endings/beginnings)
- Vocabulary support for OOV items
- Fast inference, optional slower training
- Supports CoNLL-U (including VRT format), TEITOK XML, and raw text input
- Full pipeline: raw text → sentences → tokens → tags → parse
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import xml.etree.ElementTree as ET

# Import FlexiPipeConfig from config module to avoid duplication
from flexipipe.config import FlexiPipeConfig

# Disable tokenizers parallelism warning (set before any tokenizers imports)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# CoNLL-U MISC expansion keys (configurable via --expan)
# Expansion is the new standard format, others are for backward compatibility
CONLLU_EXPANSION_KEYS = ['Expansion', 'Exp', 'Expan', 'Expand', 'fform', 'FFORM']

def set_conllu_expansion_key(key: Optional[str]):
    """Configure which MISC key to consider as expansion (e.g., 'Exp', 'fform')."""
    global CONLLU_EXPANSION_KEYS
    if key and isinstance(key, str):
        # Put provided key and common case variants at the front
        keys = [key, key.capitalize(), key.upper()]
        # Preserve unique order: configured keys first, then defaults
        seen = set()
        new_list = []
        for k in keys + CONLLU_EXPANSION_KEYS:
            if k not in seen:
                new_list.append(k)
                seen.add(k)
        CONLLU_EXPANSION_KEYS = new_list

try:
    import torch
    from torch import nn
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
        PreTrainedTokenizer, PreTrainedModel, EarlyStoppingCallback
    )
    from datasets import Dataset, DatasetDict
    try:
        import numpy as np
    except ImportError:
        np = None
    from sklearn.metrics import accuracy_score, classification_report
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = None  # No error when available
    
    def get_device():
        """Detect and return the best available device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = e
    # Don't warn at import time - only warn when transformers is actually needed
    # (normalization-only mode doesn't require transformers)
    
    def get_device():
        """Fallback device detection when torch is not available."""
        return None


def parse_conllu_simple(line: str) -> Optional[Dict]:
    """Parse CoNLL-U line, handling VRT format (1-3 columns only)."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    parts = line.split('\t')
    if len(parts) < 1:
        return None
    
    # VRT format: can have 1, 2, or 3 columns
    # Column 1: form (required)
    # Column 2: lemma (optional)
    # Column 3: upos (optional)
    
    token = {
        'id': None,
        'form': parts[0] if len(parts) > 0 else '',
        'lemma': parts[1] if len(parts) > 1 else '_',
        'upos': parts[2] if len(parts) > 2 else '_',
        'xpos': '_',
        'feats': '_',
        'head': 0,
        'deprel': '_',
    }
    
    # Full CoNLL-U format (10 columns)
    if len(parts) >= 10:
        try:
            tid = parts[0]
            if '-' in tid:
                return None  # MWT line
            token_id = int(tid)
            token.update({
                'id': token_id,
                'form': parts[1],
                'lemma': parts[2],
                'upos': parts[3],
                'xpos': parts[4],
                'feats': parts[5] if len(parts) > 5 else '_',
                'head': int(parts[6]) if parts[6].isdigit() else 0,
                'deprel': parts[7] if len(parts) > 7 else '_',
                'misc': parts[9] if len(parts) > 9 else '_',
            })
            
            # Extract normalization from MISC column (Normalized= or Reg= for backward compatibility)
            misc = parts[9] if len(parts) > 9 else '_'
            if misc and misc != '_':
                # Parse MISC column for Normalized= or Reg= (normalization)
                misc_parts = misc.split('|')
                norm_form = '_'
                expan_form = '_'
                for misc_part in misc_parts:
                    if misc_part.startswith('Normalized='):
                        norm_form = misc_part[11:]  # Extract value after "Normalized="
                    elif misc_part.startswith('Reg='):
                        norm_form = misc_part[4:]  # Extract value after "Reg=" (backward compatibility)
                    else:
                        for k in CONLLU_EXPANSION_KEYS:
                            prefix = f"{k}="
                            if misc_part.startswith(prefix):
                                expan_form = misc_part[len(prefix):]
                                break
                        # Also check for Expansion= (new format)
                        if misc_part.startswith('Expansion='):
                            expan_form = misc_part[10:]  # Extract value after "Expansion="
                token['norm_form'] = norm_form
                token['expan'] = expan_form if expan_form else '_'
            else:
                token['norm_form'] = '_'
                token['expan'] = '_'
        except (ValueError, IndexError):
            pass
    
    return token


def load_conllu_file(file_path: Path) -> List[List[Dict]]:
    """Load CoNLL-U file, returning list of sentences (each sentence is a list of tokens).
    
    Preserves the original text from # text = comments for accurate spacing reconstruction.
    """
    sentences = []
    current_sentence = []
    current_text = None  # Store original text from # text = comment
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Check for # text = comment
            if line_stripped.startswith('# text ='):
                current_text = line_stripped[8:].strip()  # Extract text after "# text ="
                continue
            
            token = parse_conllu_simple(line)
            if token:
                current_sentence.append(token)
            elif not line_stripped:
                if current_sentence:
                    # Store original text as metadata in first token or as sentence-level data
                    if current_text:
                        # Store in first token's misc field for later retrieval
                        if current_sentence:
                            if 'misc' not in current_sentence[0] or current_sentence[0]['misc'] == '_':
                                current_sentence[0]['_original_text'] = current_text
                            else:
                                current_sentence[0]['_original_text'] = current_text
                    sentences.append(current_sentence)
                    current_sentence = []
                    current_text = None
    
    if current_sentence:
        if current_text:
            if current_sentence:
                current_sentence[0]['_original_text'] = current_text
        sentences.append(current_sentence)
    
    return sentences


def segment_sentences(text: str) -> List[str]:
    """
    Segment raw text into sentences using rule-based approach.
    
    Args:
        text: Raw text string
        
    Returns:
        List of sentence strings
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    
    # Sentence-ending punctuation
    sentence_endings = r'[.!?]+'
    
    # Split on sentence endings, but keep the punctuation
    sentences = []
    current_sentence = ''
    
    # Use regex to find sentence boundaries
    # Pattern: sentence ending followed by optional whitespace (or end of text)
    # Less restrictive: doesn't require capital letter after punctuation
    # This handles cases like "Why? Because..." and "Yes! No way!"
    pattern = rf'({sentence_endings})(?:\s+|$)'
    
    parts = re.split(pattern, text)
    
    for i, part in enumerate(parts):
        current_sentence += part
        # Check if this part ends with sentence-ending punctuation
        # If so, finish the sentence
        if re.search(sentence_endings + r'$', part):
            sentence = current_sentence.strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = ''
    
    # Add remaining text as final sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Fallback: if no sentences found, return entire text as one sentence
    if not sentences:
        sentences = [text]
    
    return sentences


def tokenize_words_ud_style(text: str) -> List[str]:
    """
    Tokenize text into words using UD-style tokenization rules.
    
    UD tokenization principles:
    - Split on whitespace
    - Separate punctuation from words (except for apostrophes in contractions)
    - Keep contractions together (e.g., "d'", "l'", "n'")
    
    Args:
        text: Sentence string
        
    Returns:
        List of token strings
    """
    if not text:
        return []
    
    # UD-style tokenization regex
    # Matches:
    # - Contractions with apostrophes (d', l', n', etc.)
    # - Words with hyphens (compound words)
    # - Regular words (Unicode-aware)
    # - Punctuation (separated)
    
    # Use Unicode word characters (\w includes Unicode letters, but we need to be explicit for some cases)
    # Pattern for contractions: letter(s) + apostrophe + letter(s)
    # Use \p{L} for Unicode letters (requires regex with UNICODE flag) or use \w which is Unicode-aware in Python
    contraction_pattern = r"[\w]+'[\w]+"
    
    # Pattern for hyphenated compounds
    compound_pattern = r"[\w]+(?:-[\w]+)+"
    
    # Pattern for regular words (including numbers and mixed alphanumeric, Unicode-aware)
    # \w matches Unicode word characters (letters, digits, underscore)
    word_pattern = r"[\w]+"
    
    # Pattern for punctuation (everything that's not whitespace, word chars, hyphen, or apostrophe)
    punct_pattern = r"[^\s\w\-']+"
    
    # Combined pattern (order matters: contractions first, then compounds, then words, then punctuation)
    token_pattern = f"({contraction_pattern}|{compound_pattern}|{word_pattern}|{punct_pattern})"
    
    # Use UNICODE flag to ensure proper Unicode handling
    tokens = re.findall(token_pattern, text, re.UNICODE)
    
    # Filter out empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    return tokens


def load_plain_text(file_path: Path, segment: bool = False, tokenize: bool = False) -> List[List[Dict]]:
    """Load plain text file, returning list of sentences.
    
    Args:
        file_path: Path to text file
        segment: If True, segment raw text into sentences. If False, assume one sentence per line.
        tokenize: If True, tokenize sentences into words. If False, assume one word per line or whitespace-separated.
    
    Returns:
        List of sentences, where each sentence is a list of token dicts
    """
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if segment:
            # Read entire file and segment into sentences
            full_text = f.read()
            
            # Segment sentences, but preserve original text with exact spacing
            # segment_sentences normalizes whitespace, so we need to find original sentences in full_text
            sentence_texts_normalized = segment_sentences(full_text)
            
            # Find original sentence texts in full_text (preserving exact spacing)
            full_text_pos = 0
            for sent_text_normalized in sentence_texts_normalized:
                # Find this normalized sentence in the original full_text
                sent_normalized = re.sub(r'\s+', ' ', sent_text_normalized).strip()
                
                # Build a flexible pattern that matches the sentence with variable whitespace
                pattern_parts = []
                for char in sent_normalized:
                    if char.isspace():
                        pattern_parts.append(r'\s+')  # Match one or more whitespace
                    elif char in r'.^$*+?{}[]\|()':
                        pattern_parts.append(re.escape(char))
                    else:
                        pattern_parts.append(re.escape(char))
                
                pattern = ''.join(pattern_parts)
                
                # Search for the pattern in the original text
                match = re.search(pattern, full_text[full_text_pos:], re.UNICODE)
                if match:
                    found_start = full_text_pos + match.start()
                    found_end = full_text_pos + match.end()
                    original_sent_text = full_text[found_start:found_end]
                    full_text_pos = found_end
                else:
                    # Fallback: use normalized version
                    original_sent_text = sent_text_normalized
                
                if tokenize:
                    # Tokenize the sentence
                    words = tokenize_words_ud_style(original_sent_text)
                else:
                    # Split by whitespace
                    words = original_sent_text.split()
                
                sentence_tokens = []
                for word_idx, word in enumerate(words, 1):
                    sentence_tokens.append({
                        'id': word_idx,
                        'form': word,
                        'lemma': '_',
                        'upos': '_',
                        'xpos': '_',
                        'feats': '_',
                        'head': '_',  # Use '_' when no parser
                        'deprel': '_',
                    })
                # Store original text in first token for accurate spacing reconstruction
                if sentence_tokens:
                    sentence_tokens[0]['_original_text'] = original_sent_text
                sentences.append(sentence_tokens)
        else:
            # Original behavior: one sentence per line or blank-line separated
            current_sentence = []
            
            for line in f:
                line = line.strip()
                if not line:
                    # Blank line - end of sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    original_line = line  # Preserve original line with spacing
                    if tokenize:
                        # Tokenize the line
                        words = tokenize_words_ud_style(line)
                    else:
                        # Tokenize by whitespace (simple tokenization)
                        words = line.split()
                    
                    for word_idx, word in enumerate(words, 1):
                        current_sentence.append({
                            'id': word_idx,
                            'form': word,
                            'lemma': '_',
                            'upos': '_',
                            'xpos': '_',
                            'feats': '_',
                            'head': '_',  # Use '_' when no parser
                            'deprel': '_',
                        })
                    # Store original text in first token
                    if current_sentence:
                        # Find the first token we just added (last len(words) tokens)
                        first_token_idx = len(current_sentence) - len(words)
                        if first_token_idx >= 0:
                            current_sentence[first_token_idx]['_original_text'] = original_line
            
            # Add final sentence if any
            if current_sentence:
                sentences.append(current_sentence)
    
    return sentences


def load_teitok_xml(file_path: Path, normalization_attr: str = 'reg') -> List[List[Dict]]:
    """
    Load TEITOK XML file, returning list of sentences.
    
    Args:
        file_path: Path to TEITOK XML file
        normalization_attr: Attribute name for normalization (default: 'reg', can be 'nform')
    """
    sentences = []
    
    def get_attr_with_fallback(elem, attr_names: str) -> str:
        # attr_names can be comma-separated fallbacks
        if not attr_names:
            return ''
        for name in [a.strip() for a in attr_names.split(',') if a.strip()]:
            val = elem.get(name, '')
            if val:
                return val
        return ''

    # Support passing comma-separated fallbacks via normalization_attr; also support xpos/expan via special keys in attr string
    # We keep backward compatibility by defaulting to elem.get('xpos') when no explicit xpos attr is passed
    xpos_attr = getattr(load_teitok_xml, '_xpos_attr', 'xpos')
    expan_attr = getattr(load_teitok_xml, '_expan_attr', 'expan')

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for s in root.findall('.//s'):
            sentence_tokens = []
            token_num = 1
            
            # Get sentence ID: try @id first, then @xml:id
            sentence_id = s.get('id', '') or s.get('{http://www.w3.org/XML/1998/namespace}id', '')
            
            # Try to get original text from sentence element (if available)
            # Some TEITOK files store original text as an attribute or in a text node
            original_sentence_text = s.text or s.get('text', None)
            if not original_sentence_text:
                # Try to reconstruct from tokens (will be fallback in write_output)
                original_sentence_text = None
            
            for tok in s.findall('.//tok'):
                # Get token ID: try @id first, then @xml:id
                tok_id = tok.get('id', '') or tok.get('{http://www.w3.org/XML/1998/namespace}id', '')
                dtoks = tok.findall('.//dtok')
                
                if dtoks:
                    # Contraction: process each dtok
                    for dt in dtoks:
                        # Get dtok ID: try @id first, then @xml:id
                        dt_id = dt.get('id', '') or dt.get('{http://www.w3.org/XML/1998/namespace}id', '')
                        form = dt.get('form', '') or (dt.text or '').strip()
                        if form:
                            # Get normalization (try specified attr first, then common fallbacks)
                            nform = get_attr_with_fallback(dt, normalization_attr) or dt.get('reg', '') or dt.get('nform', '')
                            xpos_val = get_attr_with_fallback(dt, xpos_attr) or dt.get('xpos', '_')
                            expan_val = get_attr_with_fallback(dt, expan_attr) or dt.get('expan', '') or dt.get('fform', '')
                            
                            sentence_tokens.append({
                                'id': token_num,
                                'form': form,
                                'norm_form': nform if nform else '_',
                                'lemma': dt.get('lemma', '_'),
                                'upos': dt.get('upos', '_'),
                                'xpos': xpos_val if xpos_val else '_',
                                'feats': dt.get('feats', '_'),
                                'head': dt.get('head', '0'),
                                'deprel': dt.get('deprel', '_'),
                                'tok_id': tok_id,
                                'dtok_id': dt_id,
                                'expan': expan_val if expan_val else '_',
                            })
                            token_num += 1
                else:
                    # Regular token
                    form = (tok.text or '').strip()
                    if form:
                        # Get normalization (try specified attr first, then common fallbacks)
                        nform = get_attr_with_fallback(tok, normalization_attr) or tok.get('reg', '') or tok.get('nform', '')
                        xpos_val = get_attr_with_fallback(tok, xpos_attr) or tok.get('xpos', '_')
                        expan_val = get_attr_with_fallback(tok, expan_attr) or tok.get('expan', '') or tok.get('fform', '')
                        
                        sentence_tokens.append({
                            'id': token_num,
                            'form': form,
                            'norm_form': nform if nform else '_',
                            'lemma': tok.get('lemma', '_'),
                            'upos': tok.get('upos', '_'),
                                'xpos': xpos_val if xpos_val else '_',
                            'feats': tok.get('feats', '_'),
                            'head': tok.get('head', '0'),
                            'deprel': tok.get('deprel', '_'),
                            'tok_id': tok_id,
                                'expan': expan_val if expan_val else '_',
                        })
                        token_num += 1
            
            if sentence_tokens:
                # Store sentence ID and original text in first token if available
                if sentence_id:
                    sentence_tokens[0]['_sentence_id'] = sentence_id
                if original_sentence_text:
                    sentence_tokens[0]['_original_text'] = original_sentence_text
                sentences.append(sentence_tokens)
    
    except Exception as e:
        print(f"Error loading TEITOK XML: {e}", file=sys.stderr)
    
    return sentences


def build_vocabulary(conllu_files: List[Path]) -> Dict[str, Dict]:
    """Build vocabulary from CoNLL-U files."""
    vocab = {}
    
    for file_path in conllu_files:
        sentences = load_conllu_file(file_path)
        for sentence in sentences:
            for token in sentence:
                form = token.get('form', '').lower()
                if form and form not in vocab:
                    vocab[form] = {
                        'lemma': token.get('lemma', '_'),
                        'upos': token.get('upos', '_'),
                        'xpos': token.get('xpos', '_'),
                        'feats': token.get('feats', '_'),
                        'reg': token.get('norm_form', '_'),
                        'expan': token.get('expan', '_'),
                    }
    
    return vocab


class BiaffineAttention(nn.Module):
    """Biaffine attention for dependency head prediction."""
    def __init__(self, hidden_size: int, arc_dim: int = 500):
        super().__init__()
        self.arc_dim = arc_dim
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_size, arc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(arc_dim, arc_dim)
        )
        self.dep_mlp = nn.Sequential(
            nn.Linear(hidden_size, arc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(arc_dim, arc_dim)
        )
        # Biaffine layer: for each (head, dep) pair, compute score
        # Use Bilinear layer: head @ W @ dep.T
        self.arc_biaffine = nn.Bilinear(arc_dim, arc_dim, 1, bias=True)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            arc_scores: [batch_size, seq_len, seq_len] - scores for head predictions
                arc_scores[i, j] = score for token j having head i
        """
        head_repr = self.head_mlp(hidden_states)  # [batch, seq, arc_dim]
        dep_repr = self.dep_mlp(hidden_states)     # [batch, seq, arc_dim]
        
        batch_size, seq_len, arc_dim = head_repr.shape
        
        # Safety check: truncate if sequence is too long
        if seq_len > 512:
            seq_len = 512
            head_repr = head_repr[:, :seq_len, :]
            dep_repr = dep_repr[:, :seq_len, :]
        
        # Memory-efficient biaffine computation using batched matrix multiplication
        # Instead of expand(), use broadcasting and batch operations
        # head_repr: [batch, seq, arc_dim], dep_repr: [batch, seq, arc_dim]
        # We want: [batch, seq, seq] where score[i,j] = biaffine(head[i], dep[j])
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 64  # Process 64 tokens at a time (much smaller to avoid memory issues)
        arc_scores_list = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            head_chunk = head_repr[:, i:end_i, :]  # [batch, chunk_i, arc_dim]
            chunk_i = end_i - i
            
            row_scores = []
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                dep_chunk = dep_repr[:, j:end_j, :]  # [batch, chunk_j, arc_dim]
                chunk_j = end_j - j
                
                # Compute scores for this chunk pair without expand
                # head_chunk: [batch, chunk_i, arc_dim]
                # dep_chunk: [batch, chunk_j, arc_dim]
                # We need: [batch, chunk_i, chunk_j]
                
                # Use repeat instead of expand (more memory efficient for small chunks)
                head_exp = head_chunk.unsqueeze(2).repeat(1, 1, chunk_j, 1)  # [batch, chunk_i, chunk_j, arc_dim]
                dep_exp = dep_chunk.unsqueeze(1).repeat(1, chunk_i, 1, 1)   # [batch, chunk_i, chunk_j, arc_dim]
                
                # Flatten for biaffine
                head_flat = head_exp.reshape(-1, arc_dim)
                dep_flat = dep_exp.reshape(-1, arc_dim)
                
                # Compute biaffine scores
                scores_flat = self.arc_biaffine(head_flat, dep_flat)  # [batch * chunk_i * chunk_j, 1]
                scores = scores_flat.reshape(batch_size, chunk_i, chunk_j)
                row_scores.append(scores)
            
            # Concatenate along j dimension
            if row_scores:
                row = torch.cat(row_scores, dim=2)  # [batch, chunk_i, seq_len]
                arc_scores_list.append(row)
        
        # Concatenate along i dimension
        if arc_scores_list:
            arc_scores = torch.cat(arc_scores_list, dim=1)  # [batch, seq_len, seq_len]
        else:
            arc_scores = torch.zeros(batch_size, seq_len, seq_len, device=head_repr.device, dtype=head_repr.dtype)
        
        return arc_scores


class MultiTaskFlexiPipeTagger(nn.Module):
    """Multi-task FlexiPipe tagger and parser with separate heads for UPOS, XPOS, FEATS, lemmatizer, and parsing."""
    
    def __init__(self, base_model_name: str, num_upos: int, num_xpos: int, num_feats: int, 
                 num_lemmas: int = 0, num_deprels: int = 0, num_norms: int = 0,
                 train_parser: bool = False, train_lemmatizer: bool = False, train_normalizer: bool = False,
                 num_orig_forms: int = 0, use_orig_form_for_parser: bool = False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.train_parser = train_parser
        self.train_lemmatizer = train_lemmatizer
        self.train_normalizer = train_normalizer
        self.num_upos = num_upos
        self.num_xpos = num_xpos
        self.num_feats = num_feats
        
        # Classification heads for tagging - use MLPs instead of simple Linear
        # This is crucial for SOTA performance
        mlp_hidden = hidden_size // 2  # Half the hidden size for intermediate layer
        
        self.upos_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_upos)
        )
        self.xpos_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_xpos)
        )
        self.feats_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_feats)
        )
        
        # Lemmatizer head (if training lemmatizer)
        # Context-aware: use UPOS/XPOS/FEATS embeddings + BERT embeddings
        if train_lemmatizer and num_lemmas > 0:
            # Embedding dimensions for categorical features
            upos_embed_dim = 32  # Small embedding for UPOS
            xpos_embed_dim = 64  # Larger embedding for XPOS (more specific)
            feats_embed_dim = 32  # Embedding for FEATS
            
            # Embedding layers for categorical features
            self.lemma_upos_embed = nn.Embedding(num_upos, upos_embed_dim)
            self.lemma_xpos_embed = nn.Embedding(num_xpos, xpos_embed_dim)
            self.lemma_feats_embed = nn.Embedding(num_feats, feats_embed_dim)
            
            # Combined input size: BERT hidden + UPOS + XPOS + FEATS embeddings
            combined_hidden = hidden_size + upos_embed_dim + xpos_embed_dim + feats_embed_dim
            
            self.lemma_head = nn.Sequential(
                nn.Linear(combined_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_lemmas)
            )
            
            # Store embedding dimensions for later use
            self.lemma_upos_embed_dim = upos_embed_dim
            self.lemma_xpos_embed_dim = xpos_embed_dim
            self.lemma_feats_embed_dim = feats_embed_dim
        else:
            self.lemma_head = None
            self.lemma_upos_embed = None
            self.lemma_xpos_embed = None
            self.lemma_feats_embed = None
        
        # Original form embedding for transpositional parsing (if enabled)
        self.use_orig_form_for_parser = use_orig_form_for_parser
        self.orig_form_embed = None
        orig_form_embed_dim = 64  # Embedding dimension for original forms
        if use_orig_form_for_parser and num_orig_forms > 0:
            self.orig_form_embed = nn.Embedding(num_orig_forms, orig_form_embed_dim)
            # Adjust hidden size for parser to account for orig_form embeddings
            parser_hidden_size = hidden_size + orig_form_embed_dim
        else:
            parser_hidden_size = hidden_size
        
        # Parsing heads (only if training parser)
        if train_parser and num_deprels > 0:
            # Use adjusted hidden size that includes orig_form embeddings if enabled
            self.biaffine = BiaffineAttention(parser_hidden_size, arc_dim=500)
            self.deprel_head = nn.Sequential(
                nn.Linear(parser_hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_deprels)
            )
        else:
            self.biaffine = None
            self.deprel_head = None
        
        # Normalizer head (if training normalizer)
        if train_normalizer and num_norms > 0:
            self.norm_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_norms)
            )
        else:
            self.norm_head = None
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels_upos=None, labels_xpos=None, 
                labels_feats=None, labels_lemma=None, labels_norm=None, labels_head=None, labels_deprel=None,
                orig_form_ids=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits_upos = self.upos_head(sequence_output)
        logits_xpos = self.xpos_head(sequence_output)
        logits_feats = self.feats_head(sequence_output)
        
        # Normalizer outputs
        logits_norm = None
        if self.train_normalizer and self.norm_head is not None:
            logits_norm = self.norm_head(sequence_output)  # [batch, seq, num_norms]
        
        # Lemmatizer outputs (context-aware: uses UPOS/XPOS/FEATS)
        logits_lemma = None
        if self.train_lemmatizer and self.lemma_head is not None:
            # Get predicted UPOS/XPOS/FEATS for context-aware lemmatization
            # Use predicted labels (argmax) during inference, or use provided labels during training
            batch_size, seq_len, _ = sequence_output.shape
            
            # Get UPOS/XPOS/FEATS predictions (or use provided labels if available)
            if labels_upos is not None:
                upos_ids = labels_upos  # Use ground truth during training
            else:
                upos_ids = torch.argmax(logits_upos, dim=-1)  # Use predictions during inference
            
            if labels_xpos is not None:
                xpos_ids = labels_xpos
            else:
                xpos_ids = torch.argmax(logits_xpos, dim=-1)
            
            if labels_feats is not None:
                feats_ids = labels_feats
            else:
                feats_ids = torch.argmax(logits_feats, dim=-1)
            
            # Embed UPOS/XPOS/FEATS
            upos_embeds = self.lemma_upos_embed(upos_ids)  # [batch, seq, upos_embed_dim]
            xpos_embeds = self.lemma_xpos_embed(xpos_ids)  # [batch, seq, xpos_embed_dim]
            feats_embeds = self.lemma_feats_embed(feats_ids)  # [batch, seq, feats_embed_dim]
            
            # Concatenate BERT embeddings with POS/FEATS embeddings
            combined_embeds = torch.cat([sequence_output, upos_embeds, xpos_embeds, feats_embeds], dim=-1)
            
            logits_lemma = self.lemma_head(combined_embeds)  # [batch, seq, num_lemmas]
        
        # Parsing outputs
        arc_scores = None
        logits_deprel = None
        if self.train_parser and self.biaffine is not None:
            # For transpositional parsing: concatenate original form embeddings with BERT embeddings
            parser_input = sequence_output
            if self.use_orig_form_for_parser and self.orig_form_embed is not None and orig_form_ids is not None:
                # Embed original forms: [batch, seq] -> [batch, seq, orig_form_embed_dim]
                orig_form_embeds = self.orig_form_embed(orig_form_ids)
                # Concatenate: [batch, seq, hidden_size] + [batch, seq, orig_form_embed_dim]
                parser_input = torch.cat([sequence_output, orig_form_embeds], dim=-1)
            
            arc_scores = self.biaffine(parser_input)  # [batch, seq, seq]
            # Deprel scores: for each possible head-child pair
            # We'll use a simpler approach: predict deprel for each token given its predicted head
            logits_deprel = self.deprel_head(parser_input)  # [batch, seq, num_deprels]
        
        loss = None
        if labels_upos is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Weight losses: UPOS is most important (2.0), XPOS (1.5), FEATS (1.0), Lemma (1.5)
            # This helps prioritize UPOS accuracy which is critical
            upos_loss = loss_fct(logits_upos.view(-1, logits_upos.size(-1)), labels_upos.view(-1))
            loss = 2.0 * upos_loss  # UPOS gets double weight
            
            if labels_xpos is not None:
                xpos_loss = loss_fct(logits_xpos.view(-1, logits_xpos.size(-1)), labels_xpos.view(-1))
                loss += 1.5 * xpos_loss  # XPOS gets 1.5x weight
            
            if labels_feats is not None:
                feats_loss = loss_fct(logits_feats.view(-1, logits_feats.size(-1)), labels_feats.view(-1))
                loss += 1.0 * feats_loss  # FEATS gets standard weight
            
            # Lemma loss
            if self.train_lemmatizer and labels_lemma is not None and logits_lemma is not None:
                lemma_loss = loss_fct(logits_lemma.view(-1, logits_lemma.size(-1)), labels_lemma.view(-1))
                loss += 1.5 * lemma_loss  # Lemma gets 1.5x weight (similar to XPOS)
            
            # Normalizer loss
            if self.train_normalizer and labels_norm is not None and logits_norm is not None:
                norm_loss = loss_fct(logits_norm.view(-1, logits_norm.size(-1)), labels_norm.view(-1))
                loss += 1.0 * norm_loss  # Normalizer gets standard weight
            
            # Parsing loss
            if self.train_parser and labels_head is not None and arc_scores is not None:
                # Arc loss: cross-entropy over heads (each token should have one head)
                batch_size, seq_len, _ = arc_scores.shape
                # Mask invalid positions (padding, special tokens)
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
                    mask = mask & mask.transpose(1, 2)  # Both dimensions must be valid
                    arc_scores = arc_scores.masked_fill(~mask.bool(), float('-inf'))
                
                # Arc loss: negative log-likelihood of correct head
                arc_loss = nn.CrossEntropyLoss(ignore_index=-100)
                loss += arc_loss(arc_scores.view(-1, seq_len), labels_head.view(-1))
                
                # Deprel loss: only for tokens with valid heads
                if labels_deprel is not None and logits_deprel is not None:
                    deprel_loss = nn.CrossEntropyLoss(ignore_index=-100)
                    loss += deprel_loss(logits_deprel.view(-1, logits_deprel.size(-1)), labels_deprel.view(-1))
        
                return {
                    'loss': loss,
                    'logits_upos': logits_upos,
                    'logits_xpos': logits_xpos,
                    'logits_feats': logits_feats,
                    'logits_lemma': logits_lemma,
                    'logits_norm': logits_norm,
                    'arc_scores': arc_scores,
                    'logits_deprel': logits_deprel,
                }
        
        # Inference mode (no labels): return logits without loss
        return {
                    'logits_upos': logits_upos,
                    'logits_xpos': logits_xpos,
                    'logits_feats': logits_feats,
                    'logits_lemma': logits_lemma,
                    'logits_norm': logits_norm,
                    'arc_scores': arc_scores,
                    'logits_deprel': logits_deprel,
                }


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_upos = inputs.pop("labels_upos", None)
        labels_xpos = inputs.pop("labels_xpos", None)
        labels_feats = inputs.pop("labels_feats", None)
        labels_lemma = inputs.pop("labels_lemma", None)
        labels_norm = inputs.pop("labels_norm", None)
        labels_head = inputs.pop("labels_head", None)
        labels_deprel = inputs.pop("labels_deprel", None)
        orig_form_ids = inputs.pop("orig_form_ids", None)
        
        outputs = model(**inputs, labels_upos=labels_upos, labels_xpos=labels_xpos, 
                       labels_feats=labels_feats, labels_lemma=labels_lemma,
                       labels_norm=labels_norm,
                       labels_head=labels_head, labels_deprel=labels_deprel,
                       orig_form_ids=orig_form_ids)
        loss = outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss


def _derive_inflection_suffixes_from_vocab(vocab: Dict, max_suffix_len: int = 4, min_count: int = 3) -> List[str]:
    """Derive common inflection suffixes from vocab form->reg mappings in a language-agnostic way.

    We look for entries where both the surface form and its reg end with the same suffix
    and the stems differ (e.g., seruicio->servicio while preserving plural 's').
    We collect suffixes up to max_suffix_len and keep those observed at least min_count times.
    """
    def get_reg(entry):
        if isinstance(entry, list):
            if entry and isinstance(entry[0], dict):
                return entry[0].get('reg', None)
        elif isinstance(entry, dict):
            return entry.get('reg', None)
        return None

    counts: Dict[str, int] = {}
    for form, entry in vocab.items():
        if not isinstance(form, str):
            continue
        reg = get_reg(entry)
        if not reg or reg == '_' or reg.lower() == form.lower():
            continue
        f = form.lower()
        r = reg.lower()
        max_k = min(max_suffix_len, len(f), len(r))
        for k in range(1, max_k + 1):
            sfx = f[-k:]
            if r.endswith(sfx):
                if f[:-k] != r[:-k]:
                    counts[sfx] = counts.get(sfx, 0) + 1
    frequent = [s for s, c in counts.items() if c >= min_count]
    frequent.sort(key=lambda s: (-counts[s], -len(s), s))
    return frequent


def normalize_word(word: str, vocab: Dict, conservative: bool = True, similarity_threshold: float = 0.8,
                   inflection_suffixes: Optional[List[str]] = None) -> Optional[str]:
    """
    Normalize orthographic variant to standard form using vocabulary.
    
    Priority order (conservative mode):
    1. Explicit normalization mapping in vocabulary (reg field)
    2. Morphological variations of known mappings (e.g., "mysterio"->"misterio" allows "mysterios"->"misterios")
    3. Check if word is already normalized (exists in vocab without reg)
    4. Pattern-aware Levenshtein distance matching (only in non-conservative mode)
    
    This is especially important for historic documents where normalization depends on
    transcription standards, region, period, and register - the local vocabulary can
    provide domain-specific normalization mappings.
    
    Args:
        word: Word to normalize
        vocab: Vocabulary dictionary (can include reg field for explicit mappings)
        conservative: If True, only use explicit mappings and morphological variations (default: True)
        similarity_threshold: Similarity threshold for normalization (higher = more conservative)
    
    Returns:
        Normalized form if found, None otherwise
    """
    word_lower = word.lower()
    
    def get_reg_from_entry(entry):
        """Extract reg (normalization) from vocabulary entry (handles single dict or array)."""
        if isinstance(entry, list):
            # Array format: check first entry (most frequent)
            if entry and isinstance(entry[0], dict):
                return entry[0].get('reg', None)
        elif isinstance(entry, dict):
            return entry.get('reg', None)
        return None
    
    def word_exists_as_reg_in_vocab(word_to_check: str) -> bool:
        """Check if word appears as a 'reg' value anywhere in vocab (means it's already normalized)."""
        word_check_lower = word_to_check.lower()
        for entry in vocab.values():
            if isinstance(entry, list):
                for analysis in entry:
                    reg = analysis.get('reg')
                    if reg and reg.lower() == word_check_lower:
                        return True
            elif isinstance(entry, dict):
                reg = entry.get('reg')
                if reg and reg.lower() == word_check_lower:
                    return True
        return False
    
    # Step 0: Early check - if word appears as a 'reg' value, it's already normalized
    # This prevents normalizing words that are themselves normalized forms
    if word_exists_as_reg_in_vocab(word):
        return None  # Word is already a normalized form, don't normalize it
    
    # Step 1: Check for explicit normalization mapping in vocabulary
    # Try exact case first, then lowercase
    if word in vocab:
        reg = get_reg_from_entry(vocab[word])
        if reg and reg != '_' and reg != word:
            return reg
    
    if word_lower in vocab:
        reg = get_reg_from_entry(vocab[word_lower])
        if reg and reg != '_' and reg.lower() != word_lower:
            return reg
    
    # Step 2: Check morphological variations of known mappings
    # If "mysterio" -> "misterio" is in vocab, also normalize "mysterios" -> "misterios"
    # This is safe because it's based on explicit mappings
    if conservative:
        # Use provided suffixes, otherwise derive from vocab
        suffixes_to_try = inflection_suffixes or _derive_inflection_suffixes_from_vocab(vocab)
        
        # Try removing suffixes to find base form
        for suffix in suffixes_to_try:
            if len(word_lower) > len(suffix) + 2 and word_lower.endswith(suffix):
                base_form = word_lower[:-len(suffix)]
                # Check if base form has a normalization mapping
                if base_form in vocab:
                    reg = get_reg_from_entry(vocab[base_form])
                    if reg and reg != '_' and reg.lower() != base_form:
                        # Apply same suffix to normalized form
                        normalized = reg.lower() + suffix
                        # CRITICAL: Verify the normalized form exists in vocab (as key or reg value)
                        # Don't normalize if result doesn't exist in vocab - prevents incorrect normalizations
                        if normalized in vocab or word_exists_as_reg_in_vocab(normalized):
                            return normalized
                
                # Also try with original case
                if word and len(word) > len(suffix):
                    base_form_orig = word[:-len(suffix)]
                    if base_form_orig in vocab:
                        reg = get_reg_from_entry(vocab[base_form_orig])
                        if reg and reg != '_' and reg != base_form_orig:
                            normalized = reg + suffix
                            # CRITICAL: Verify the normalized form exists in vocab (as key or reg value)
                            normalized_lower = normalized.lower()
                            if normalized in vocab or normalized_lower in vocab or word_exists_as_reg_in_vocab(normalized):
                                return normalized
    
    # Step 3: Check if word is already normalized (exists in vocab without reg)
    # If word exists in vocab and has no reg, it's already the normalized form
    if word in vocab or word_lower in vocab:
        entry = vocab.get(word) or vocab.get(word_lower)
        reg = get_reg_from_entry(entry)
        if not reg or reg == '_':
            return None  # Already normalized/standard form
    
    # Step 4: Pattern-aware similarity matching (only in non-conservative mode)
    if not conservative:
        # Use Levenshtein distance with pattern-aware substitutions
        normalized = find_pattern_aware_normalization(word, vocab, threshold=similarity_threshold)
        if normalized:
            return normalized
    
    # Conservative mode: don't normalize if no explicit mapping found
    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_pattern_aware_normalization(word: str, vocab: Dict, threshold: float = 0.8) -> Optional[str]:
    """
    Find normalization using pattern-aware approach with frequency-based rules.
    
    IMPORTANT: Only applies normalization patterns that are:
    1. Used frequently (at least 3 times in vocab)
    2. Result in a normalized form that exists in vocab (as key or reg value)
    
    Extracts normalization patterns from vocab (form -> reg mappings) and only
    applies frequent patterns. This prevents rare transformations like "conocido -> conociendo"
    from being over-applied.
    
    Args:
        word: Word to normalize
        vocab: Vocabulary dictionary (with reg fields for normalization mappings)
        threshold: Minimum similarity threshold (0.0-1.0) - not used for pattern matching
    
    Returns:
        Normalized form if found, None otherwise
    """
    word_lower = word.lower()
    
    # Step 1: Extract normalization patterns from vocab (form -> reg mappings)
    # Count how often each transformation pattern is used
    normalization_patterns = {}  # (char_from, char_to, position) -> count
    pattern_to_reg = {}  # (char_from, char_to, position) -> list of (form, reg) examples
    
    for vocab_form, vocab_entry in vocab.items():
        # Extract reg from entry
        reg = None
        if isinstance(vocab_entry, list):
            if vocab_entry and isinstance(vocab_entry[0], dict):
                reg = vocab_entry[0].get('reg')
        elif isinstance(vocab_entry, dict):
            reg = vocab_entry.get('reg')
        
        if reg and reg != '_' and reg.lower() != vocab_form.lower():
            # We have a normalization mapping: vocab_form -> reg
            form_lower = vocab_form.lower()
            reg_lower = reg.lower()
            
            # Find character differences (substitution patterns)
            # Try to identify which characters changed
            if len(form_lower) == len(reg_lower):
                # Same length: character substitution
                for i, (c1, c2) in enumerate(zip(form_lower, reg_lower)):
                    if c1 != c2:
                        pattern_key = (c1, c2, 'subst')
                        normalization_patterns[pattern_key] = normalization_patterns.get(pattern_key, 0) + 1
                        if pattern_key not in pattern_to_reg:
                            pattern_to_reg[pattern_key] = []
                        pattern_to_reg[pattern_key].append((vocab_form, reg))
    
    # Filter patterns: only keep those used at least 3 times (frequent patterns)
    frequent_patterns = {k: v for k, v in normalization_patterns.items() if v >= 3}
    
    if not frequent_patterns:
        # No frequent patterns found - fall back to simple checks
        return None
    
    # Step 2: Try to apply frequent patterns to the word
    # Check if applying any frequent pattern results in a word that exists in vocab
    candidates = []
    
    for pattern_key, pattern_count in frequent_patterns.items():
        char_from, char_to, pattern_type = pattern_key
        if pattern_type == 'subst' and char_from in word_lower:
            # Try applying this substitution pattern
            normalized_candidate = word_lower.replace(char_from, char_to)
            
            # CRITICAL: Only proceed if normalized form exists in vocab (as key or reg value)
            exists_in_vocab = False
            if normalized_candidate in vocab:
                exists_in_vocab = True
            else:
                # Check if it exists as a reg value in any vocab entry
                for vocab_entry in vocab.values():
                    reg = None
                    if isinstance(vocab_entry, list):
                        for analysis in vocab_entry:
                            reg = analysis.get('reg')
                            if reg and reg.lower() == normalized_candidate:
                                exists_in_vocab = True
                                break
                    elif isinstance(vocab_entry, dict):
                        reg = vocab_entry.get('reg')
                        if reg and reg.lower() == normalized_candidate:
                            exists_in_vocab = True
                            break
                    if exists_in_vocab:
                        break
            
            if exists_in_vocab:
                # This pattern is frequent AND the normalized form exists - use it
                candidates.append((normalized_candidate, pattern_count))
    
    if not candidates:
        return None
    
    # Sort by pattern frequency (most frequent first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def split_contraction(form: str, vocab: Dict, aggressive: bool = False, language: Optional[str] = None) -> Optional[List[str]]:
    """
    Split contraction into component words (e.g., "destas" -> ["de", "estas"]).
    
    Uses vocabulary to identify potential contractions and split them.
    Handles both modern languages (with rules) and historic texts.
    
    Args:
        form: Word form that might be a contraction
        vocab: Vocabulary dictionary
        aggressive: If True, use more aggressive splitting patterns for historic texts
        language: Language code (e.g., 'es', 'pt', 'ltz') for language-specific rules
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    form_lower = form.lower()
    
    # Check if form is already in vocabulary as a single word
    # If it exists as a single word, it's ambiguous - prefer keeping as single word
    # unless we have strong evidence it's a contraction
    exists_as_single_word = form in vocab or form_lower in vocab
    
    # Language-specific patterns (modern languages)
    if language:
        split_result = _split_contraction_language_specific(form, form_lower, vocab, language, exists_as_single_word, aggressive)
        if split_result:
            return split_result
    
def _split_contraction_language_specific(form: str, form_lower: str, vocab: Dict, language: str, exists_as_single_word: bool, aggressive: bool = False) -> Optional[List[str]]:
    """
    Language-specific contraction splitting rules.
    
    Args:
        form: Original form
        form_lower: Lowercase form
        vocab: Vocabulary dictionary
        language: Language code ('es', 'pt', 'ltz', etc.)
        exists_as_single_word: Whether the form exists as a single word in vocab
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    # Luxembourgish: d'XXX is always de + XXX
    if language == 'ltz' or language == 'lb':
        if form_lower.startswith("d'") and len(form_lower) > 2:
            remainder = form_lower[2:]
            if remainder and (remainder in vocab or len(remainder) >= 3):
                return ["de", remainder]
        # Also handle d' at start of capitalized words
        if form.startswith("D'") and len(form) > 2:
            remainder = form[2:].lower()
            if remainder and (remainder in vocab or len(remainder) >= 3):
                return ["De", remainder.capitalize()]
    
    # Portuguese: verb-lo, verb-la, etc. (hyphenated clitics)
    if language == 'pt':
        # Pattern: verb-lo, verb-la, verb-las, verb-los, verb-me, verb-te, verb-nos, verb-vos
        clitic_patterns = [
            (r'^([a-z]+)-lo$', r'\1', 'lo'),
            (r'^([a-z]+)-la$', r'\1', 'la'),
            (r'^([a-z]+)-las$', r'\1', 'las'),
            (r'^([a-z]+)-los$', r'\1', 'los'),
            (r'^([a-z]+)-me$', r'\1', 'me'),
            (r'^([a-z]+)-te$', r'\1', 'te'),
            (r'^([a-z]+)-nos$', r'\1', 'nos'),
            (r'^([a-z]+)-vos$', r'\1', 'vos'),
        ]
        
        for pattern, verb_group, clitic in clitic_patterns:
            match = re.match(pattern, form_lower)
            if match:
                verb_part = match.group(1)
                # If ambiguous (exists as single word), prefer keeping as single word
                # unless verb part clearly exists in vocab
                if exists_as_single_word:
                    # Only split if verb part exists in vocab (strong evidence)
                    if verb_part in vocab:
                        return [verb_part, clitic]
                    # Otherwise, keep as single word (ambiguous)
                    return None
                else:
                    # Not ambiguous, safe to split if verb part looks valid
                    if verb_part in vocab or len(verb_part) >= 3:
                        return [verb_part, clitic]
    
    # Spanish: dámelo = dé + me + lo (no hyphen, verb + clitics)
    if language == 'es':
        # Spanish clitics: me, te, se, nos, os, le, la, lo, les, las, los
        # Pattern: verb ending in vowel + clitics
        # Common: dáme, dámelo, dámela, etc.
        
        # Try to split at clitic boundaries
        spanish_clitics = ['me', 'te', 'se', 'nos', 'os', 'le', 'la', 'lo', 'les', 'las', 'los']
        
        # Check if form ends with known clitics
        for clitic in spanish_clitics:
            if form_lower.endswith(clitic) and len(form_lower) > len(clitic):
                verb_part = form_lower[:-len(clitic)]
                
                # Check if verb part exists in vocab (e.g., "dá" from "dar")
                # Also check if verb_part + "r" exists (infinitive form)
                verb_inf = verb_part + 'r'
                verb_exists = verb_part in vocab or verb_inf in vocab
                
                # If ambiguous (exists as single word), only split if verb clearly exists
                if exists_as_single_word:
                    if verb_exists:
                        return [verb_part, clitic]
                    # Otherwise, keep as single word (ambiguous, e.g., "kárate" vs "kára" + "te")
                    return None
                else:
                    # Not ambiguous, split if verb part looks valid
                    if verb_exists or len(verb_part) >= 2:
                        # Check for multiple clitics (e.g., dámelo = dá + me + lo)
                        # Try to find another clitic in the middle
                        remaining = verb_part
                        clitics_found = [clitic]
                        
                        # Check for second clitic
                        for clitic2 in spanish_clitics:
                            if remaining.endswith(clitic2) and len(remaining) > len(clitic2):
                                verb_base = remaining[:-len(clitic2)]
                                if verb_base in vocab or verb_base + 'r' in vocab:
                                    return [verb_base, clitic2, clitic]
                        
                        return [verb_part, clitic]
        
        # Historic Spanish: destas, dellos, etc. (aggressive mode)
        if aggressive:
            if form_lower.startswith('d') and len(form_lower) > 3:
                # Try "de" + remainder
                remainder = form_lower[1:]  # Remove 'd'
                if remainder and (remainder in vocab or len(remainder) >= 3):
                    # Check if it's ambiguous (exists as single word)
                    if exists_as_single_word:
                        # Only split if remainder clearly exists in vocab
                        if remainder in vocab:
                            return ["de", remainder]
                        return None
                    return ["de", remainder]
    
    return None


def split_contraction(form: str, vocab: Dict, aggressive: bool = False, language: Optional[str] = None) -> Optional[List[str]]:
    """
    Split contraction into component words (e.g., "destas" -> ["de", "estas"]).
    
    Uses vocabulary to identify potential contractions and split them.
    Handles both modern languages (with rules) and historic texts.
    
    Args:
        form: Word form that might be a contraction
        vocab: Vocabulary dictionary
        aggressive: If True, use more aggressive splitting patterns for historic texts
        language: Language code (e.g., 'es', 'pt', 'ltz') for language-specific rules
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    form_lower = form.lower()
    
    # Check if form is already in vocabulary as a single word
    # If it exists as a single word, it's ambiguous - prefer keeping as single word
    # unless we have strong evidence it's a contraction
    exists_as_single_word = form in vocab or form_lower in vocab
    
    # Language-specific patterns (modern languages)
    if language:
        split_result = _split_contraction_language_specific(form, form_lower, vocab, language, exists_as_single_word, aggressive)
        if split_result:
            return split_result
    
    # Common contraction patterns (language-agnostic)
    # These are patterns that often indicate contractions
    contraction_patterns = []
    
    if aggressive:
        # More aggressive patterns for historic texts
        # Spanish: destas, dellos, etc.
        contraction_patterns.extend([
            (r'^d([aeiou])', ['de', r'\1']),  # de + vowel
            (r'^([aeiou])l([aeiou])', [r'\1', 'el', r'\2']),  # vowel + el + vowel (aggressive)
            (r'^([aeiou])n([aeiou])', [r'\1', 'en', r'\2']),  # vowel + en + vowel (aggressive)
        ])
    
    # Standard patterns (more conservative)
    contraction_patterns.extend([
        # Portuguese/Spanish: d'água, faze-lo, etc.
        (r"^([a-z]+)-([a-z]+)$", None),  # hyphenated words (check if parts exist)
        (r"^([a-z]+)'([a-z]+)$", None),   # apostrophe contractions (check if parts exist)
        # Check for common prefixes that might be contractions
        (r'^([a-z]{1,2})([a-z]{3,})$', None),  # Short prefix + longer word
    ])
    
    # Try to split based on patterns
    for pattern, replacement in contraction_patterns:
        if replacement is None:
            # Pattern-based splitting: check if parts exist in vocabulary
            if '-' in form:
                parts = form.split('-')
                if len(parts) == 2:
                    part1, part2 = parts
                    # Check if both parts exist in vocab (or are common words)
                    if (part1.lower() in vocab or len(part1) <= 2) and \
                       (part2.lower() in vocab or len(part2) <= 2):
                        return [part1, part2]
            
            if "'" in form:
                parts = form.split("'")
                if len(parts) == 2:
                    part1, part2 = parts
                    if (part1.lower() in vocab or len(part1) <= 2) and \
                       (part2.lower() in vocab or len(part2) <= 2):
                        return [part1, part2]
            
            # Try splitting at common boundaries
            # Common prefixes: de, a, en, con, etc.
            common_prefixes = ['de', 'a', 'en', 'con', 'por', 'para', 'del', 'al', 'da', 'do']
            for prefix in common_prefixes:
                if form_lower.startswith(prefix) and len(form_lower) > len(prefix) + 2:
                    remainder = form_lower[len(prefix):]
                    # If ambiguous, only split if remainder clearly exists in vocab
                    if exists_as_single_word:
                        if remainder in vocab:
                            return [prefix, remainder]
                        # Otherwise, keep as single word (ambiguous)
                        continue
                    else:
                        if remainder in vocab or len(remainder) >= 3:
                            return [prefix, remainder]
        else:
            # Direct replacement pattern
            match = re.match(pattern, form_lower)
            if match:
                # Build the split based on replacement pattern
                split_words = []
                for repl in replacement:
                    if repl.startswith('\\'):
                        # Backreference
                        group_num = int(repl[1:])
                        if group_num <= len(match.groups()):
                            split_words.append(match.group(group_num))
                    else:
                        split_words.append(repl)
                if len(split_words) > 1:
                    return split_words
    
    # If no pattern matched, try vocabulary-based splitting
    # Look for words that could be the start of this form
    # This is more expensive but useful for historic texts
    if aggressive:
        for vocab_word in vocab.keys():
            if len(vocab_word) >= 2 and form_lower.startswith(vocab_word) and len(form_lower) > len(vocab_word) + 2:
                remainder = form_lower[len(vocab_word):]
                # If ambiguous, only split if remainder clearly exists in vocab
                if exists_as_single_word:
                    if remainder in vocab:
                        return [vocab_word, remainder]
                    # Otherwise, keep as single word (ambiguous)
                    continue
                else:
                    if remainder in vocab or len(remainder) >= 3:
                        return [vocab_word, remainder]
    
    return None


# Import viterbi_tag_sentence from dedicated module to avoid duplication
from flexipipe.viterbi import viterbi_tag_sentence


def find_similar_words(word: str, vocab: Dict[str, Dict], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """Find similar words based on endings/beginnings.
    
    NOTE: Beginning-based matching is dangerous for lemmatization as it can incorrectly
    match words that share prefixes but are unrelated (e.g., "prometido" vs "recibir").
    We primarily rely on ending-based matching, which is more reliable for morphological patterns.
    """
    word_lower = word.lower()
    candidates = []
    
    # Check endings (last 3-6 characters) - this is the primary and most reliable method
    # Morphological patterns are typically suffix-based (inflections, derivations)
    for end_len in range(6, 2, -1):
        if len(word_lower) >= end_len:
            ending = word_lower[-end_len:]
            for vocab_word, vocab_data in vocab.items():
                if vocab_word.endswith(ending) and vocab_word != word_lower:
                    # Simple similarity: length difference and ending match
                    length_diff = abs(len(vocab_word) - len(word_lower)) / max(len(vocab_word), len(word_lower))
                    similarity = 1.0 - length_diff
                    if similarity >= threshold:
                        candidates.append((vocab_word, similarity))
    
    # Check beginnings (for some languages) - but with higher threshold and stricter matching
    # This is more dangerous as it can match unrelated words, so we're more conservative
    # Only use beginning matching if no ending matches were found
    if not candidates:
        for beg_len in range(5, 3, -1):  # Longer prefixes (4-5 chars) for better reliability
            if len(word_lower) >= beg_len:
                beginning = word_lower[:beg_len]
                for vocab_word, vocab_data in vocab.items():
                    # Require that the vocab word also starts with the same beginning
                    # AND has similar length (within 2 characters) to avoid wild matches
                    if vocab_word.startswith(beginning) and vocab_word != word_lower:
                        length_diff = abs(len(vocab_word) - len(word_lower))
                        if length_diff <= 2:  # Much stricter: only similar length words
                            similarity = 1.0 - (length_diff / max(len(vocab_word), len(word_lower)))
                            if similarity >= max(threshold, 0.8):  # Higher threshold for beginning matches
                                candidates.append((vocab_word, similarity))
    
    # Sort by similarity and return unique
    candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_candidates = []
    for word, score in candidates:
        if word not in seen:
            seen.add(word)
            unique_candidates.append((word, score))
            if len(unique_candidates) >= 10:
                break
    
    return unique_candidates


class FlexiPipeTagger:
    """Transformer-based FlexiPipe tagger."""
    
    def __init__(self, config: FlexiPipeConfig, vocab: Optional[Dict[str, Dict]] = None, model_path: Optional[Path] = None, transition_probs: Optional[Dict] = None, vocab_metadata: Optional[Dict] = None, lexicon: Optional[Dict[str, Dict]] = None):
        self.config = config
        self.model_path = model_path  # Store model path for vocabulary loading
        # vocab will be merged with model vocabulary in load_model
        self.external_vocab = vocab or {}
        self.lexicon = lexicon or {}  # Lexicon (fallback for OOV words only, not merged with vocab)
        self.transition_probs = transition_probs  # Transition probabilities for Viterbi tagging
        self.vocab_metadata = vocab_metadata  # Vocabulary metadata (may contain language info)
        self.model_vocab = {}  # Vocabulary from training data
        self.vocab = {}  # Merged vocabulary (model_vocab + external_vocab, external overrides)
        self.lemmatization_patterns = {}  # XPOS -> list of (suffix_from, suffix_to, min_length) patterns
        self.tokenizer = None
        self.model = None
        self.upos_labels = []
        self.xpos_labels = []
        self.feats_labels = []
        self.lemma_labels = []
        self.deprel_labels = []
        self.lemma_to_id = {}
        self.id_to_lemma = {}
        self.deprel_to_id = {}
        self.id_to_deprel = {}
        self.inflection_suffixes: Optional[List[str]] = None
        # Store original XML tree for TEITOK input (to preserve structure)
        self.original_teitok_tree = None
        self.original_teitok_file = None
        # Store vocabulary file path(s) for revision statement
        self.vocab_file_path = None  # Most specific vocab file (for display)
        self.vocab_file_paths = []  # All vocab files (for reference)
        # Track operations performed for revision statement
        self.operations_performed = []
        # Detect and store device (MPS for Mac Studio, CUDA for NVIDIA, CPU otherwise)
        if TRANSFORMERS_AVAILABLE:
            self.device = get_device()
            device_name = "MPS (Apple Silicon GPU)" if str(self.device) == "mps" else \
                         "CUDA (NVIDIA GPU)" if str(self.device) == "cuda" else "CPU"
            print(f"Using device: {device_name}", file=sys.stderr)
        else:
            self.device = None
        
        # Build lemmatization patterns from vocabulary if available
        if self.external_vocab:
            self._build_lemmatization_patterns(self.external_vocab)
            # Also build normalization inflection suffixes
            self._build_normalization_inflection_suffixes()

    def _load_external_suffixes(self) -> Optional[List[str]]:
        """Load external suffix list JSON if provided."""
        path = getattr(self.config, 'normalization_suffixes_file', None)
        if not path:
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                # Normalize ordering: longer suffixes first
                data = sorted(set(data), key=lambda s: (-len(s), s))
                return data
        except Exception as e:
            print(f"Warning: Failed to load normalization suffixes from {path}: {e}", file=sys.stderr)
        return None

    def _build_normalization_inflection_suffixes(self):
        """Derive inflection suffixes from vocab and external file, considering lemma anchor.

        Learns both normalized (reg vs lemma) and raw (form/expan vs lemma) suffixes.
        More restrictive:
          - require shared root (common prefix) length >= min_root_len
          - count paired substitutions (form_suffix -> lemma_suffix) and keep frequent ones
          - prefer external list when provided
        """
        # External override takes precedence
        external = self._load_external_suffixes()
        if external:
            self.inflection_suffixes = external
            return

        vocab = self.vocab or self.external_vocab or {}

        def get_entry_fields(entry):
            # Normalize access for possible list/dict entries
            if isinstance(entry, list):
                entry = entry[0] if entry else {}
            if not isinstance(entry, dict):
                return None, None, None, None
            reg = entry.get('reg')
            lemma = entry.get('lemma') or entry.get('lem')
            xpos = entry.get('xpos') or entry.get('pos')
            expan = entry.get('expan') or entry.get('expand')  # optional expanded raw form
            return reg, lemma, xpos, expan

        def paired_suffix_from_pair(base: str, lemma: str, max_len: int = 6, min_root_len: int = 3) -> Optional[Tuple[str, str]]:
            """Return (base_suffix, lemma_suffix) if they share a sufficient root.

            Uses longest common prefix as root; requires length >= min_root_len.
            Limits suffix lengths to avoid noise.
            """
            if not base or not lemma:
                return None
            base_l = base.lower()
            lemma_l = lemma.lower()
            p = 0
            L = min(len(base_l), len(lemma_l))
            while p < L and base_l[p] == lemma_l[p]:
                p += 1
            # Require meaningful shared root
            if p < min_root_len:
                return None
            sfx_base = base_l[p:]
            sfx_lem = lemma_l[p:]
            # Bound lengths; allow empty lemma suffix (e.g., plural s vs zero)
            if len(sfx_base) == 0 or len(sfx_base) > max_len:
                return None
            if len(sfx_lem) > max_len:
                return None
            return (sfx_base, sfx_lem)

        anchor = getattr(self.config, 'lemma_anchor', 'both')
        # Count mapping pairs and also surface suffix frequencies
        pair_counts: Dict[Tuple[str, str], int] = {}
        surface_counts: Dict[str, int] = {}
        for form, entry in vocab.items():
            if not isinstance(form, str):
                continue
            reg, lemma, xpos, expan = get_entry_fields(entry)
            if not lemma:
                continue
            # Derive from reg vs lemma
            if anchor in ('reg', 'both') and isinstance(reg, str):
                ps = paired_suffix_from_pair(reg, lemma)
                if ps:
                    pair_counts[ps] = pair_counts.get(ps, 0) + 1
                    surface_counts[ps[0]] = surface_counts.get(ps[0], 0) + 1
            # Derive from raw channel: prefer expan if available to avoid abbreviations
            if anchor in ('form', 'both'):
                raw_base = expan if isinstance(expan, str) and expan else form
                ps = paired_suffix_from_pair(raw_base, lemma)
                if ps:
                    pair_counts[ps] = pair_counts.get(ps, 0) + 1
                    surface_counts[ps[0]] = surface_counts.get(ps[0], 0) + 1

        # Keep frequent pairs and surface suffixes; thresholds to reduce noise
        min_count = 5
        kept_pairs = {pair: c for pair, c in pair_counts.items() if c >= min_count}
        # Derive final surface suffix set from kept pairs
        suffixes = sorted({sfx for (sfx, slem) in kept_pairs.keys()}, key=lambda s: (-len(s), s))
        # Fallback to reg-based derivation if nothing found
        if not suffixes:
            suffixes = _derive_inflection_suffixes_from_vocab(vocab)
        self.inflection_suffixes = suffixes
    
    def _build_lemmatization_patterns(self, vocab: Dict):
        """Build lemmatization patterns from vocabulary using shared function."""
        from flexipipe.lemmatizer import build_lemmatization_patterns
        self.lemmatization_patterns, self.pattern_info = build_lemmatization_patterns(vocab)
        
        if self.config.debug and self.lemmatization_patterns:
            total_patterns = sum(len(patterns) for patterns in self.lemmatization_patterns.values())
            print(f"[DEBUG] Built {total_patterns} lemmatization patterns across {len(self.lemmatization_patterns)} XPOS tags", file=sys.stderr)
    
    def load_model(self, model_path: Optional[Path] = None):
        """Load trained model or initialize from BERT."""
        if not TRANSFORMERS_AVAILABLE:
            print(f"Error: transformers library not available. {TRANSFORMERS_IMPORT_ERROR}", file=sys.stderr)
            print("Install with: pip install transformers torch datasets scikit-learn accelerate", file=sys.stderr)
            raise ImportError("transformers library required for model loading")
        
        if model_path and Path(model_path).exists():
            model_path = Path(model_path)
            print(f"Loading model from {model_path}", file=sys.stderr)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # Load training configuration (if available)
            training_config_file = model_path / 'training_config.json'
            has_metadata = False
            if training_config_file.exists():
                with open(training_config_file, 'r', encoding='utf-8') as f:
                    training_config = json.load(f)
                # Restore training settings
                self.config.bert_model = training_config.get('bert_model', self.config.bert_model)
                self.config.train_tokenizer = training_config.get('train_tokenizer', self.config.train_tokenizer)
                self.config.train_tagger = training_config.get('train_tagger', self.config.train_tagger)
                self.config.train_parser = training_config.get('train_parser', self.config.train_parser)
                self.config.train_lemmatizer = training_config.get('train_lemmatizer', self.config.train_lemmatizer)
                self.config.train_normalizer = training_config.get('train_normalizer', False)
                self.config.normalization_attr = training_config.get('normalization_attr', 'reg')
                
                # Check metadata for which components were actually trained
                # This is more reliable than just the config flags
                if 'has_lemmatizer' in training_config:
                    has_metadata = True
                    has_lemmatizer = training_config.get('has_lemmatizer', False)
                    has_feats = training_config.get('has_feats', True)  # Default True for backward compatibility
                    has_parser = training_config.get('has_parser', False)
                    has_normalizer = training_config.get('has_normalizer', False)
                    
                    # Override config flags based on metadata
                    if not has_lemmatizer:
                        self.config.train_lemmatizer = False
                        print("Note: Lemmatizer was not trained (according to metadata). Using vocabulary fallback.", file=sys.stderr)
                    if not has_feats:
                        print("Note: FEATS were not in training data (according to metadata). Using vocabulary fallback.", file=sys.stderr)
                    if not has_parser:
                        self.config.train_parser = False
                    if not has_normalizer:
                        self.config.train_normalizer = False
                
                print(f"Loaded training configuration: BERT={self.config.bert_model}, "
                      f"Tokenizer={self.config.train_tokenizer}, Tagger={self.config.train_tagger}, "
                      f"Parser={self.config.train_parser}, Lemmatizer={self.config.train_lemmatizer}, "
                      f"Normalizer={self.config.train_normalizer}", file=sys.stderr)
            else:
                print("Warning: No training_config.json found, using defaults and detecting from label_mappings", file=sys.stderr)
            
            # Load label mappings
            label_mapping_file = model_path / 'label_mappings.json'
            if label_mapping_file.exists():
                with open(label_mapping_file, 'r', encoding='utf-8') as f:
                    label_mappings = json.load(f)
                self.upos_labels = label_mappings.get('upos_labels', [])
                self.xpos_labels = label_mappings.get('xpos_labels', [])
                self.feats_labels = label_mappings.get('feats_labels', [])
                self.lemma_labels = label_mappings.get('lemma_labels', [])
                self.deprel_labels = label_mappings.get('deprel_labels', [])
                self.upos_to_id = label_mappings.get('upos_to_id', {})
                self.xpos_to_id = label_mappings.get('xpos_to_id', {})
                self.feats_to_id = label_mappings.get('feats_to_id', {})
                self.lemma_to_id = label_mappings.get('lemma_to_id', {})
                self.deprel_to_id = label_mappings.get('deprel_to_id', {})
                
                # Auto-detect if components were actually trained based on label counts
                # (only if metadata wasn't found in training_config.json)
                if not has_metadata:
                    # If lemmatizer wasn't trained, lemma_labels will be empty or only contain '_'
                    if len(self.lemma_labels) == 0 or (len(self.lemma_labels) == 1 and self.lemma_labels[0] == '_'):
                        if self.config.train_lemmatizer:
                            print("Warning: Lemmatizer was marked as trained but no lemma labels found. Disabling lemmatizer.", file=sys.stderr)
                        self.config.train_lemmatizer = False
                    
                    # If FEATS only has '_' or very few labels, FEATS weren't in training data
                    if len(self.feats_labels) <= 1 or (len(self.feats_labels) == 2 and '_' in self.feats_labels):
                        print("Warning: FEATS appear to be missing from training data. Will use vocabulary fallback for FEATS.", file=sys.stderr)
                self.id_to_upos = {v: k for k, v in self.upos_to_id.items()}
                self.id_to_xpos = {v: k for k, v in self.xpos_to_id.items()}
                self.id_to_feats = {v: k for k, v in self.feats_to_id.items()}
                self.id_to_lemma = {v: k for k, v in self.lemma_to_id.items()} if self.lemma_to_id else {}
                self.id_to_deprel = {v: k for k, v in self.deprel_to_id.items()} if self.deprel_to_id else {}
                
                # Load normalization mappings if available
                if self.config.train_normalizer:
                    self.norm_forms = label_mappings.get('norm_forms', [])
                    self.norm_to_id = label_mappings.get('norm_to_id', {})
                    self.id_to_norm = {v: k for k, v in self.norm_to_id.items()} if self.norm_to_id else {}
                
                # Load original form mappings for transpositional parsing if available
                use_orig_form_for_parser = label_mappings.get('use_orig_form_for_parser', False)
                if use_orig_form_for_parser:
                    self.orig_forms = label_mappings.get('orig_forms', [])
                    self.orig_form_to_id = label_mappings.get('orig_form_to_id', {})
                    self.id_to_orig_form = {v: k for k, v in self.orig_form_to_id.items()} if self.orig_form_to_id else {}
                    print(f"Loaded original form mappings: {len(self.orig_forms)} unique forms (transpositional parsing enabled)", file=sys.stderr)
                else:
                    self.orig_forms = []
                    self.orig_form_to_id = {}
                    self.id_to_orig_form = {}
                
                # Fallback: detect if model was trained with lemmatizer/parser/normalizer from label_mappings
                # (only if training_config.json wasn't found)
                if not training_config_file.exists():
                    has_lemmatizer = len(self.lemma_labels) > 0
                    has_parser = len(self.deprel_labels) > 0
                    has_normalizer = 'norm_forms' in label_mappings and len(label_mappings.get('norm_forms', [])) > 0
                    if has_lemmatizer:
                        self.config.train_lemmatizer = True
                    if has_parser:
                        self.config.train_parser = True
                    if has_normalizer:
                        self.config.train_normalizer = True
                        self.norm_forms = label_mappings.get('norm_forms', [])
                        self.norm_to_id = label_mappings.get('norm_to_id', {})
                        self.id_to_norm = {v: k for k, v in self.norm_to_id.items()} if self.norm_to_id else {}
            
            # Load model vocabulary (built from training data)
            model_vocab_file = model_path / 'model_vocab.json'
            if model_vocab_file.exists():
                with open(model_vocab_file, 'r', encoding='utf-8') as f:
                    self.model_vocab = json.load(f)
                print(f"Loaded model vocabulary with {len(self.model_vocab)} entries", file=sys.stderr)
            else:
                self.model_vocab = {}
                print("Warning: No model_vocab.json found, using empty model vocabulary", file=sys.stderr)
            
            # Merge vocabularies: model vocab + external vocab (external overrides model)
            self.vocab = self.model_vocab.copy()
            self.vocab.update(self.external_vocab)  # External vocab overrides model vocab
            if self.external_vocab:
                print(f"Merged vocabularies: {len(self.model_vocab)} model entries + {len(self.external_vocab)} external entries = {len(self.vocab)} total", file=sys.stderr)
            else:
                print(f"Using model vocabulary: {len(self.vocab)} entries", file=sys.stderr)
            # Rebuild lemmatization patterns with merged vocab (always, even if no external vocab)
            if self.vocab:
                self._build_lemmatization_patterns(self.vocab)
            
            # Fallback: use defaults if labels not loaded
            if not self.upos_labels:
                self.upos_labels = ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PUNCT', 'PRON', 'ADV', 'AUX', 'CCONJ', 'SCONJ', 'PROPN', 'NUM', 'PART', 'INTJ', 'X', 'SYM', '_']
                self.upos_to_id = {label: i for i, label in enumerate(self.upos_labels)}
                self.id_to_upos = {i: label for i, label in enumerate(self.upos_labels)}
            # Initialize empty lemma_labels if not present
            if not hasattr(self, 'lemma_labels'):
                self.lemma_labels = []
                self.lemma_to_id = {}
                self.id_to_lemma = {}
            
            # NOTE: UPOS context tokens removed - they were hurting performance
            # No need to add them to tokenizer anymore
            
            # Load model
            use_orig_form = hasattr(self, 'orig_forms') and self.orig_forms and use_orig_form_for_parser
            self.model = MultiTaskFlexiPipeTagger(
                self.config.bert_model,
                num_upos=len(self.upos_labels),
                num_xpos=len(self.xpos_labels),
                num_feats=len(self.feats_labels),
                num_lemmas=len(self.lemma_labels) if hasattr(self, 'lemma_labels') and self.lemma_labels else 0,
                num_deprels=len(self.deprel_labels) if hasattr(self, 'deprel_labels') and self.deprel_labels else 0,
                train_parser=self.config.train_parser,
                train_lemmatizer=self.config.train_lemmatizer,
                num_orig_forms=len(self.orig_forms) if use_orig_form else 0,
                use_orig_form_for_parser=use_orig_form
            )
            
            # Resize embeddings to match tokenizer vocabulary size
            # This is necessary because the saved model may have been trained with additional tokens
            vocab_size = len(self.tokenizer)
            base_vocab_size = self.model.base_model.config.vocab_size
            if vocab_size != base_vocab_size:
                print(f"Resizing model embeddings from {base_vocab_size} to {vocab_size} to match tokenizer", file=sys.stderr)
                self.model.base_model.resize_token_embeddings(vocab_size)
            
            # Load state dict
            state_dict_path = model_path / 'pytorch_model.bin'
            if state_dict_path.exists():
                # Load on CPU first to avoid MPS alignment issues, then move to device
                # This is a workaround for PyTorch MPS bug with unaligned memory operations
                state_dict = torch.load(state_dict_path, map_location='cpu')
                
                # Filter out keys that don't exist in the current model (e.g., lemmatizer embeddings if not trained)
                model_state_dict = self.model.state_dict()
                filtered_state_dict = {}
                missing_keys = []
                unexpected_keys = []
                
                for key, value in state_dict.items():
                    if key in model_state_dict:
                        # Check if shapes match
                        if model_state_dict[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            missing_keys.append(
                                f"{key} (shape mismatch: model={model_state_dict[key].shape}, checkpoint={value.shape})"
                            )
                    else:
                        unexpected_keys.append(key)
                # Load with strict=False to allow missing keys (e.g., lemmatizer if not trained)
                load_result = self.model.load_state_dict(filtered_state_dict, strict=False)
                
                if load_result.missing_keys:
                    # Filter out expected missing keys (lemmatizer/normalizer if not trained)
                    unexpected_missing = [k for k in load_result.missing_keys 
                                        if not any(skip in k for skip in ['lemma_', 'norm_', 'orig_form_'])]
                    if unexpected_missing:
                        print(f"Warning: Missing keys in checkpoint (expected if model wasn't trained with these components):", file=sys.stderr)
                        for key in unexpected_missing[:5]:  # Show first 5
                            print(f"  - {key}", file=sys.stderr)
                        if len(unexpected_missing) > 5:
                            print(f"  ... and {len(unexpected_missing) - 5} more", file=sys.stderr)
                
                if load_result.unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint (will be ignored):", file=sys.stderr)
                    for key in load_result.unexpected_keys[:5]:  # Show first 5
                        print(f"  - {key}", file=sys.stderr)
                    if len(load_result.unexpected_keys) > 5:
                        print(f"  ... and {len(load_result.unexpected_keys) - 5} more", file=sys.stderr)
                
                # Model already moved to device above
            else:
                print(f"Warning: No pytorch_model.bin found in {model_path}, using untrained model", file=sys.stderr)
                # Move model to device if not already moved
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
        else:
            print(f"Initializing model from {self.config.bert_model}", file=sys.stderr)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
            # Initialize with default labels if not set
            if not self.upos_labels:
                self.upos_labels = ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PUNCT', 'PRON', 'ADV', 'AUX', 'CCONJ', 'SCONJ', 'PROPN', 'NUM', 'PART', 'INTJ', 'X', 'SYM', '_']
                self.xpos_labels = ['_']
                self.feats_labels = ['_']
                self.upos_to_id = {label: i for i, label in enumerate(self.upos_labels)}
                self.id_to_upos = {i: label for i, label in enumerate(self.upos_labels)}
                self.xpos_to_id = {'_': 0}
                self.id_to_xpos = {0: '_'}
                self.feats_to_id = {'_': 0}
                self.id_to_feats = {0: '_'}
            
            # Determine normalizer parameters
            num_norms = 0
            if self.config.train_normalizer and hasattr(self, 'norm_forms'):
                num_norms = len(self.norm_forms)
            
            self.model = MultiTaskFlexiPipeTagger(
                self.config.bert_model,
                num_upos=len(self.upos_labels),
                num_xpos=len(self.xpos_labels),
                num_feats=len(self.feats_labels),
                num_lemmas=len(self.lemma_labels) if hasattr(self, 'lemma_labels') and self.lemma_labels else 0,
                num_deprels=len(self.deprel_labels) if self.deprel_labels else 0,
                num_norms=num_norms,
                train_parser=self.config.train_parser,
                train_lemmatizer=self.config.train_lemmatizer,
                train_normalizer=self.config.train_normalizer
            )
            
            # Merge vocabularies even if no model vocab was found (external vocab still works)
            if not hasattr(self, 'model_vocab') or not self.model_vocab:
                self.model_vocab = {}
            self.vocab = self.model_vocab.copy()
            self.vocab.update(self.external_vocab)
            if self.external_vocab:
                print(f"Using external vocabulary with {len(self.external_vocab)} entries", file=sys.stderr)
                # Rebuild lemmatization patterns with merged vocab
                self._build_lemmatization_patterns(self.vocab)
        
        # Move model to device (MPS/CUDA/CPU) and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
    
    def train(self, train_files: List[Path], dev_files: Optional[List[Path]] = None):
        """Train the tagger on CoNLL-U files."""
        if not TRANSFORMERS_AVAILABLE:
            print(f"Error: transformers library not available. {TRANSFORMERS_IMPORT_ERROR}", file=sys.stderr)
            print("Install with: pip install transformers torch datasets scikit-learn accelerate", file=sys.stderr)
            raise ImportError("transformers library required for training")
        
        # Check for accelerate package (required for Trainer)
        try:
            import accelerate
        except ImportError:
            raise ImportError(
                "accelerate package is required for training. "
                "Install with: pip install 'accelerate>=0.26.0' or pip install transformers[torch]"
            )
        
        print("Loading training data...", file=sys.stderr)
        train_sentences = []
        for file_path in train_files:
            # Check if it's TEITOK XML
            if file_path.suffix.lower() == '.xml':
                train_sentences.extend(load_teitok_xml(file_path, normalization_attr=self.config.normalization_attr))
            else:
                train_sentences.extend(load_conllu_file(file_path))
        
        dev_sentences = []
        if dev_files:
            for file_path in dev_files:
                # Check if it's TEITOK XML
                if file_path.suffix.lower() == '.xml':
                    dev_sentences.extend(load_teitok_xml(file_path, normalization_attr=self.config.normalization_attr))
                else:
                    dev_sentences.extend(load_conllu_file(file_path))
        
        # Auto-detect which components are available in the data
        has_lemma = False
        has_parser = False
        has_normalization = False
        
        # Extract labels and detect available components
        all_upos = set()
        all_xpos = set()
        all_feats = set()
        all_lemmas = set()
        all_deprels = set()
        all_norm_forms = set()
        all_orig_forms = set()
        
        for sentence in train_sentences + dev_sentences:
            for token in sentence:
                upos = token.get('upos', '_')
                xpos = token.get('xpos', '_')
                feats = token.get('feats', '_')
                lemma = token.get('lemma', '_')
                deprel = token.get('deprel', '_')
                head = token.get('head', 0)
                norm_form = token.get('norm_form', '_')
                
                if upos and upos != '_':
                    all_upos.add(upos)
                if xpos and xpos != '_':
                    all_xpos.add(xpos)
                if feats and feats != '_':
                    # Use full FEATS string as label (not individual feature names)
                    # This allows the model to predict full UD-style feature strings
                    all_feats.add(feats)
                if lemma and lemma != '_':
                    has_lemma = True
                    if self.config.train_lemmatizer:
                        all_lemmas.add(lemma.lower())  # Normalize to lowercase for lemmas
                if deprel and deprel != '_':
                    has_parser = True
                    if self.config.train_parser:
                        all_deprels.add(deprel)
                elif head and head != 0 and head != '0':
                    # Check if head is present (even without deprel)
                    has_parser = True
                if norm_form and norm_form != '_':
                    has_normalization = True
                    if self.config.train_normalizer:
                        all_norm_forms.add(norm_form.lower())
                orig_form = token.get('orig_form', '_')
                if orig_form and orig_form != '_':
                    has_orig_forms = True
                    if self.config.train_parser:
                        # Collect original forms for transpositional parsing
                        all_orig_forms.add(orig_form.lower())  # Normalize to lowercase
        
        # Auto-adjust component training based on data availability
        if not has_lemma and self.config.train_lemmatizer:
            print("Warning: No lemma data found in training set. Disabling lemmatizer training.", file=sys.stderr)
            self.config.train_lemmatizer = False
        
        if not has_parser and self.config.train_parser:
            print("Warning: No parser data (head/deprel) found in training set. Disabling parser training.", file=sys.stderr)
            self.config.train_parser = False
        
        if not has_normalization and self.config.train_normalizer:
            print("Warning: No normalization data found in training set. Disabling normalizer training.", file=sys.stderr)
            self.config.train_normalizer = False
        elif has_normalization and self.config.train_normalizer:
            print(f"Found normalization data: {len(all_norm_forms)} unique normalized forms", file=sys.stderr)
        
        # Check for original forms (for transpositional parsing)
        use_orig_form_for_parser = False
        if has_orig_forms and self.config.train_parser:
            use_orig_form_for_parser = True
            print(f"Found original form data: {len(all_orig_forms)} unique original forms (transpositional parsing enabled)", file=sys.stderr)
        
        self.upos_labels = sorted(all_upos)
        self.xpos_labels = sorted(all_xpos)
        self.feats_labels = sorted(all_feats)
        self.lemma_labels = sorted(all_lemmas) if self.config.train_lemmatizer else []
        self.deprel_labels = sorted(all_deprels) if self.config.train_parser else []
        self.norm_forms = sorted(all_norm_forms) if self.config.train_normalizer else []
        self.orig_forms = sorted(all_orig_forms) if use_orig_form_for_parser else []
        
        print(f"Found {len(self.upos_labels)} UPOS labels, {len(self.xpos_labels)} XPOS labels, {len(self.feats_labels)} FEATS labels", file=sys.stderr)
        if self.config.train_lemmatizer:
            print(f"Found {len(self.lemma_labels)} LEMMA labels", file=sys.stderr)
        if self.config.train_parser:
            print(f"Found {len(self.deprel_labels)} DEPREL labels", file=sys.stderr)
        if self.config.train_normalizer:
            print(f"Found {len(self.norm_forms)} normalized forms", file=sys.stderr)
        
        # Create label mappings
        self.upos_to_id = {label: i for i, label in enumerate(self.upos_labels)}
        self.id_to_upos = {i: label for i, label in enumerate(self.upos_labels)}
        self.xpos_to_id = {label: i for i, label in enumerate(self.xpos_labels)}
        self.id_to_xpos = {i: label for i, label in enumerate(self.xpos_labels)}
        self.feats_to_id = {label: i for i, label in enumerate(self.feats_labels)}
        self.id_to_feats = {i: label for i, label in enumerate(self.feats_labels)}
        if self.config.train_lemmatizer:
            self.lemma_to_id = {label: i for i, label in enumerate(self.lemma_labels)}
            self.id_to_lemma = {i: label for i, label in enumerate(self.lemma_labels)}
        if self.config.train_parser:
            self.deprel_to_id = {label: i for i, label in enumerate(self.deprel_labels)}
            self.id_to_deprel = {i: label for i, label in enumerate(self.deprel_labels)}
        
        # Create original form mappings (for transpositional parsing)
        if use_orig_form_for_parser:
            self.orig_form_to_id = {form: i for i, form in enumerate(self.orig_forms)}
            self.id_to_orig_form = {i: form for i, form in enumerate(self.orig_forms)}
            # Add special token for missing original forms
            if '_' not in self.orig_form_to_id:
                self.orig_form_to_id['_'] = len(self.orig_forms)
                self.id_to_orig_form[len(self.orig_forms)] = '_'
                self.orig_forms.append('_')
        else:
            self.orig_form_to_id = {}
            self.id_to_orig_form = {}
        
        # Initialize or train tokenizer
        if self.config.train_tokenizer:
            print("Training tokenizer from corpus...", file=sys.stderr)
            self.tokenizer = self._train_tokenizer(train_sentences + (dev_sentences or []))
        elif not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
        
        # NOTE: UPOS context tokens removed - they were hurting performance
        # No need to add them to tokenizer anymore
        
        # Create normalization label mappings
        if self.config.train_normalizer:
            self.norm_to_id = {norm: i for i, norm in enumerate(self.norm_forms)}
            self.id_to_norm = {i: norm for i, norm in enumerate(self.norm_forms)}
        
        # Initialize model
        print("Initializing model...", file=sys.stderr)
        self.model = MultiTaskFlexiPipeTagger(
            self.config.bert_model,
            num_upos=len(self.upos_labels),
            num_xpos=len(self.xpos_labels),
            num_feats=len(self.feats_labels),
            num_lemmas=len(self.lemma_labels) if self.config.train_lemmatizer else 0,
            num_deprels=len(self.deprel_labels),
            num_norms=len(self.norm_forms) if self.config.train_normalizer else 0,
            train_parser=self.config.train_parser,
            train_lemmatizer=self.config.train_lemmatizer,
            train_normalizer=self.config.train_normalizer,
            num_orig_forms=len(self.orig_forms) if use_orig_form_for_parser else 0,
            use_orig_form_for_parser=use_orig_form_for_parser
        )
        
        # Resize embeddings if tokenizer vocabulary size differs from base model
        vocab_size = len(self.tokenizer)
        base_vocab_size = self.model.base_model.config.vocab_size
        if vocab_size != base_vocab_size:
            print(f"Resizing model embeddings from {base_vocab_size} to {vocab_size} to match tokenizer", file=sys.stderr)
            self.model.base_model.resize_token_embeddings(vocab_size)
        
        # Move model to device (MPS/CUDA/CPU) for training
        self.model.to(self.device)
        
        # Prepare datasets
        print("Preparing training datasets...", file=sys.stderr)
        train_dataset = self._prepare_dataset(train_sentences, self.tokenizer)
        
        dev_dataset = None
        if dev_sentences:
            dev_dataset = self._prepare_dataset(dev_sentences, self.tokenizer)
        
        # Adjust batch size for parser training (much more memory-intensive due to arc scores)
        effective_batch_size = self.config.batch_size
        gradient_accumulation = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        if self.config.train_parser:
            # Parser requires ~4-8x more memory due to [batch, seq, seq] arc scores
            # Reduce batch size aggressively, increase gradient accumulation to maintain effective batch
            original_batch = effective_batch_size
            original_grad_accum = gradient_accumulation
            original_effective = original_batch * original_grad_accum
            
            # Check device type for platform-specific optimizations
            is_mps = str(self.device) == 'mps'
            is_cuda = str(self.device) == 'cuda'
            
            # Aggressive reduction for MPS - parser with arc scores is memory-intensive
            # Use batch_size=1 or 2 for MPS to avoid OOM errors (arc scores are [batch, seq, seq])
            if is_mps:
                # Very aggressive reduction: use batch_size=1 or 2 for MPS parser training
                # This is necessary because arc scores require [batch, seq, seq] memory
                if original_batch >= 16:
                    effective_batch_size = 1  # Use 1 for maximum memory safety
                    gradient_accumulation = max(16, original_effective)  # Increase grad accum to maintain effective batch
                elif original_batch >= 8:
                    effective_batch_size = 1
                    gradient_accumulation = max(8, original_effective)
                elif original_batch >= 4:
                    effective_batch_size = 1
                    gradient_accumulation = max(4, original_effective)
                else:
                    effective_batch_size = 1  # Always use 1 for MPS parser
                    gradient_accumulation = max(original_grad_accum, original_effective)
                
                # Also reduce max_length for parser training on MPS to reduce arc score memory
                # Arc scores are [batch, seq, seq], so reducing seq length has quadratic effect on memory
                if self.config.max_length > 128:
                    print(f"Warning: MPS device detected. Reducing max_length from {self.config.max_length} to 128 for parser training to avoid OOM.", file=sys.stderr)
                    self.config.max_length = 128
                print(f"Warning: MPS device detected. Using batch_size={effective_batch_size} for parser training to avoid memory issues.", file=sys.stderr)
            else:
                # For CUDA/CPU, can use larger batches than MPS
                if is_cuda:
                    # CUDA can handle larger batches, but still reduce for parser
                    if effective_batch_size >= 16:
                        effective_batch_size = 4  # CUDA can handle 4 for parser
                        gradient_accumulation = max(4, original_effective // 4)
                    elif effective_batch_size >= 8:
                        effective_batch_size = 4
                        gradient_accumulation = max(2, original_effective // 4)
                    else:
                        effective_batch_size = min(4, effective_batch_size)
                        gradient_accumulation = max(original_grad_accum, original_effective // effective_batch_size)
                else:
                    # For CPU, reduce to 2
                    if effective_batch_size >= 16:
                        effective_batch_size = 2
                        gradient_accumulation = max(8, original_effective // 2)
                    elif effective_batch_size >= 8:
                        effective_batch_size = 2
                        gradient_accumulation = max(4, original_effective // 2)
                    elif effective_batch_size > 2:
                        effective_batch_size = 2
                        gradient_accumulation = max(2, original_effective // 2)
            
            new_effective = effective_batch_size * gradient_accumulation
            if original_batch != effective_batch_size:
                print(f"Warning: Parser training is memory-intensive. Reducing batch size from {original_batch} to {effective_batch_size}", file=sys.stderr)
                print(f"  Increasing gradient_accumulation_steps from {original_grad_accum} to {gradient_accumulation} to maintain effective batch size", file=sys.stderr)
                print(f"  Effective batch size: {original_effective} -> {new_effective}", file=sys.stderr)
                if is_mps:
                    print(f"  Note: Using batch_size=1 on MPS to avoid memory issues. Training will be slower but more stable.", file=sys.stderr)
        
        # Training arguments
        # Use eval_strategy (newer transformers) or evaluation_strategy (older versions)
        training_kwargs = {
            'output_dir': self.config.output_dir,
            'num_train_epochs': self.config.num_epochs,
            'per_device_train_batch_size': effective_batch_size,
            'per_device_eval_batch_size': effective_batch_size,
            'gradient_accumulation_steps': gradient_accumulation,
            'learning_rate': self.config.learning_rate,
            'weight_decay': 0.01,
            'warmup_steps': 500,  # Learning rate warmup - critical for BERT fine-tuning
            'warmup_ratio': 0.1,  # 10% of training steps for warmup
            'lr_scheduler_type': 'cosine',  # Cosine learning rate decay
            'logging_dir': f"{self.config.output_dir}/logs",
            'logging_steps': 100,
            'save_steps': 500,
            'save_total_limit': 3,
            'fp16': False,  # Will be enabled for CUDA below
            'dataloader_pin_memory': True,  # Enable for CUDA, disable for MPS
            'dataloader_num_workers': 0,  # Set to 0 for MPS, can use more for CUDA
        }
        
        # CUDA-specific optimizations
        is_cuda = str(self.device) == 'cuda'
        if is_cuda:
            # Enable mixed precision training for CUDA (faster and uses less memory)
            training_kwargs['fp16'] = True
            training_kwargs['dataloader_pin_memory'] = True
            training_kwargs['dataloader_num_workers'] = 2  # Can use workers on CUDA
            print("CUDA device detected: Enabling fp16 mixed precision training for better performance.", file=sys.stderr)
        
        # MPS-specific optimizations (disable CUDA features that don't work on MPS)
        if is_mps:
            training_kwargs['fp16'] = False  # MPS doesn't support fp16 well
            training_kwargs['dataloader_pin_memory'] = False  # Disable pin_memory on MPS to suppress warning
            training_kwargs['dataloader_num_workers'] = 0  # Set to 0 for MPS to avoid multiprocessing issues
        
        # Additional memory optimizations for MPS
        if is_mps:
            # Enable gradient checkpointing on the model to trade compute for memory
            # This can reduce memory usage by ~50% at the cost of ~20% slower training
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing on model for MPS memory optimization.", file=sys.stderr)
            elif hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
                self.model.base_model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing on base model for MPS memory optimization.", file=sys.stderr)
            # Reduce max length further if parser is enabled (arc scores are memory-intensive)
            if self.config.train_parser and self.config.max_length > 128:
                print(f"Warning: Further reducing max_length to 128 for MPS parser training to avoid OOM errors.", file=sys.stderr)
                self.config.max_length = 128
            # More frequent saving to avoid losing progress on OOM
            training_kwargs['save_steps'] = 250
            # Clear cache more aggressively before training
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        
        if dev_dataset:
            training_kwargs['eval_steps'] = 500
            # Use eval_strategy (standard parameter name in transformers)
            training_kwargs['eval_strategy'] = "steps"
            # Use a metric that always exists in all transformers versions
            # since some versions don't report eval_loss automatically
            training_kwargs['load_best_model_at_end'] = True
            training_kwargs['metric_for_best_model'] = "eval_runtime"
            training_kwargs['greater_is_better'] = False  # Lower runtime is better
            # Early stopping: stop if no improvement for 3 evaluation steps
            # With eval_steps=500, this means stop after 1500 steps without improvement
            # Try to add early stopping parameters (some transformers versions support this)
            # If not supported, will just use load_best_model_at_end
            if 'early_stopping_patience' not in training_kwargs:
                # Check if TrainingArguments accepts this parameter
                import inspect
                training_args_sig = inspect.signature(TrainingArguments.__init__)
                if 'early_stopping_patience' in training_args_sig.parameters:
                    training_kwargs['early_stopping_patience'] = 3
                    training_kwargs['early_stopping_threshold'] = 0.0
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Set up callbacks
        callbacks = []
        # Early stopping is handled via TrainingArguments, not callback
        
        # Add MPS cache clearing callback to prevent memory accumulation
        if is_mps:
            from transformers import TrainerCallback
            class MPSCacheClearCallback(TrainerCallback):
                """Callback to clear MPS cache periodically to prevent OOM errors."""
                def on_step_end(self, args, state, control, **kwargs):
                    # Clear cache every 50 steps to prevent memory accumulation
                    if state.global_step % 50 == 0:
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            torch.backends.mps.empty_cache()
                def on_log(self, args, state, control, **kwargs):
                    # Also clear cache after logging
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
            callbacks.append(MPSCacheClearCallback())
        
        # Custom trainer
        # Try processing_class first (newer transformers), fallback to tokenizer (older versions)
        try:
            trainer = MultiTaskTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                processing_class=self.tokenizer,
                callbacks=callbacks,
            )
        except TypeError:
            # Fallback for older transformers versions
            trainer = MultiTaskTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                tokenizer=self.tokenizer,
                callbacks=callbacks,
            )
        
        # Train with error handling for OOM errors
        print("Starting training...", file=sys.stderr)
        try:
            trainer.train()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower() or 'insufficient memory' in str(e).lower():
                print(f"\nERROR: Out of memory during training. This can happen on MPS devices.", file=sys.stderr)
                print(f"  Error: {e}", file=sys.stderr)
                print(f"\nSuggestions:", file=sys.stderr)
                print(f"  1. Reduce batch_size further (current: {effective_batch_size})", file=sys.stderr)
                print(f"  2. Reduce max_length further (current: {self.config.max_length})", file=sys.stderr)
                print(f"  3. Increase gradient_accumulation_steps (current: {gradient_accumulation})", file=sys.stderr)
                print(f"  4. Disable parser training (--no-parser) if not needed", file=sys.stderr)
                print(f"  5. Train on CPU instead of MPS (slower but more stable)", file=sys.stderr)
                print(f"\nThe model may have been saved at the last checkpoint. Check {self.config.output_dir}", file=sys.stderr)
                raise
            else:
                raise
        
        # Save model
        print(f"Saving model to {self.config.output_dir}...", file=sys.stderr)
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), output_path / 'pytorch_model.bin')
        # Save config
        self.model.base_model.config.save_pretrained(str(output_path))
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        # Build and save word-level vocabulary from training data
        # This is separate from tokenizer's vocab.txt (which contains subword tokens)
        # This vocabulary contains full words with linguistic annotations (form, lemma, upos, xpos, feats)
        # Uses shared vocabulary building function to ensure consistency with create-vocab
        print("Building word-level vocabulary from training data...", file=sys.stderr)
        from flexipipe.vocabulary import build_vocab_from_sentences
        model_vocab = build_vocab_from_sentences(train_sentences)
        
        # Save model vocabulary
        with open(Path(self.config.output_dir) / 'model_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(model_vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved model vocabulary with {len(model_vocab)} entries", file=sys.stderr)
        
        # Save label mappings
        label_mappings = {
            'upos_labels': self.upos_labels,
            'xpos_labels': self.xpos_labels,
            'feats_labels': self.feats_labels,
            'upos_to_id': self.upos_to_id,
            'xpos_to_id': self.xpos_to_id,
            'feats_to_id': self.feats_to_id,
        }
        if self.config.train_lemmatizer:
            label_mappings['lemma_labels'] = self.lemma_labels
            label_mappings['lemma_to_id'] = self.lemma_to_id
        if self.config.train_parser:
            label_mappings['deprel_labels'] = self.deprel_labels
            label_mappings['deprel_to_id'] = self.deprel_to_id
        if self.config.train_normalizer:
            label_mappings['norm_forms'] = self.norm_forms
            label_mappings['norm_to_id'] = self.norm_to_id
            label_mappings['id_to_norm'] = {v: k for k, v in self.norm_to_id.items()}
        # Save original form mappings for transpositional parsing
        if hasattr(self, 'orig_forms') and self.orig_forms:
            label_mappings['orig_forms'] = self.orig_forms
            label_mappings['orig_form_to_id'] = self.orig_form_to_id
            label_mappings['id_to_orig_form'] = {v: k for k, v in self.orig_form_to_id.items()}
            label_mappings['use_orig_form_for_parser'] = True
        with open(Path(self.config.output_dir) / 'label_mappings.json', 'w', encoding='utf-8') as f:
            json.dump(label_mappings, f, ensure_ascii=False, indent=2)
        
        # Determine which components were actually trained (based on label availability)
        # This is more reliable than just checking the config flags
        has_lemmatizer = (self.config.train_lemmatizer and len(self.lemma_labels) > 0 and 
                         not (len(self.lemma_labels) == 1 and self.lemma_labels[0] == '_'))
        has_feats = (len(self.feats_labels) > 2 or 
                    (len(self.feats_labels) == 2 and '_' not in self.feats_labels))
        has_parser = (self.config.train_parser and len(self.deprel_labels) > 0)
        has_normalizer = (self.config.train_normalizer and len(self.norm_forms) > 0)
        
        # Save training configuration (all settings used during training)
        training_config = {
            'bert_model': self.config.bert_model,
            'train_tokenizer': self.config.train_tokenizer,
            'train_tagger': self.config.train_tagger,
            'train_parser': self.config.train_parser,
            'train_lemmatizer': self.config.train_lemmatizer,
            'train_normalizer': self.config.train_normalizer,
            # Metadata: which components were actually trained (based on label availability)
            'has_lemmatizer': has_lemmatizer,
            'has_feats': has_feats,
            'has_parser': has_parser,
            'has_normalizer': has_normalizer,
            'batch_size': self.config.batch_size,
            'gradient_accumulation_steps': getattr(self.config, 'gradient_accumulation_steps', 1),
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'max_length': self.config.max_length,
            'normalization_attr': self.config.normalization_attr,
            'num_upos_labels': len(self.upos_labels),
            'num_xpos_labels': len(self.xpos_labels),
            'num_feats_labels': len(self.feats_labels),
            'num_lemma_labels': len(self.lemma_labels) if self.config.train_lemmatizer else 0,
            'num_deprel_labels': len(self.deprel_labels) if self.config.train_parser else 0,
            'num_norm_forms': len(self.norm_forms) if self.config.train_normalizer else 0,
        }
        with open(Path(self.config.output_dir) / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(training_config, f, ensure_ascii=False, indent=2)
        print(f"Saved training configuration to {Path(self.config.output_dir) / 'training_config.json'}", file=sys.stderr)
        
        print("Training complete!", file=sys.stderr)
    
    def _train_tokenizer(self, sentences: List[List[Dict]]) -> PreTrainedTokenizer:
        """
        Train a WordPiece tokenizer from the corpus.
        
        Args:
            sentences: List of sentences (each sentence is a list of token dicts)
            
        Returns:
            Trained tokenizer
        """
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
            from tokenizers.processors import BertProcessing
        except ImportError:
            print("Warning: tokenizers library not available. Installing base tokenizer instead.", file=sys.stderr)
            print("  Install with: pip install tokenizers", file=sys.stderr)
            return AutoTokenizer.from_pretrained(self.config.bert_model)
        
        # Collect all word forms from the corpus
        corpus_texts = []
        for sentence in sentences:
            words = [token.get('form', '') for token in sentence]
            if words:
                corpus_texts.append(' '.join(words))
        
        if not corpus_texts:
            print("Warning: No text found in corpus. Using base tokenizer.", file=sys.stderr)
            return AutoTokenizer.from_pretrained(self.config.bert_model)
        
        # Initialize a WordPiece tokenizer (same as BERT)
        tokenizer_model = models.WordPiece(unk_token="[UNK]")
        tokenizer = Tokenizer(tokenizer_model)
        
        # Set normalizer (lowercase for uncased models, identity for cased)
        if 'uncased' in self.config.bert_model.lower():
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(),
                normalizers.Lowercase(),
                normalizers.StripAccents()
            ])
        else:
            tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
        
        # Set pre-tokenizer (whitespace splitting)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Set post-processor (BERT-style)
        tokenizer.post_processor = BertProcessing(
            sep=("[SEP]", tokenizer.token_to_id("[SEP]") or 102),
            cls=("[CLS]", tokenizer.token_to_id("[CLS]") or 101)
        )
        
        # Train the tokenizer
        trainer = trainers.WordPieceTrainer(
            vocab_size=30000,  # Standard BERT vocab size
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            min_frequency=2,  # Minimum frequency for a token to be included
            show_progress=True
        )
        
        print(f"Training tokenizer on {len(corpus_texts)} sentences...", file=sys.stderr)
        tokenizer.train_from_iterator(corpus_texts, trainer=trainer)
        
        # Convert to HuggingFace tokenizer
        # First, load the base tokenizer to get its config
        base_tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
        
        # Wrap the trained tokenizer as a HuggingFace tokenizer
        from transformers import PreTrainedTokenizerFast
        
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            model_max_length=512,
            padding_side="right",
            truncation_side="right",
        )
        
        # Copy special tokens and other config from base tokenizer
        hf_tokenizer.cls_token = base_tokenizer.cls_token
        hf_tokenizer.sep_token = base_tokenizer.sep_token
        hf_tokenizer.pad_token = base_tokenizer.pad_token
        hf_tokenizer.unk_token = base_tokenizer.unk_token
        hf_tokenizer.mask_token = base_tokenizer.mask_token
        hf_tokenizer.cls_token_id = base_tokenizer.cls_token_id
        hf_tokenizer.sep_token_id = base_tokenizer.sep_token_id
        hf_tokenizer.pad_token_id = base_tokenizer.pad_token_id
        hf_tokenizer.unk_token_id = base_tokenizer.unk_token_id
        hf_tokenizer.mask_token_id = base_tokenizer.mask_token_id
        
        print(f"Tokenizer trained with vocabulary size: {len(hf_tokenizer)}", file=sys.stderr)
        return hf_tokenizer
    
    def _prepare_dataset(self, sentences: List[List[Dict]], tokenizer) -> Dataset:
        """Prepare dataset for training with proper tokenization alignment."""
        
        def tokenize_and_align_labels(examples):
            # Extract words and labels for each sentence
            words_list = []
            upos_list = []
            xpos_list = []
            feats_list = []
            lemma_list = []
            norm_list = []
            head_list = []
            deprel_list = []
            orig_form_list = []
            words_with_context = []  # Words with UPOS context embedded
            
            for sentence in examples['sentences']:
                words = [token.get('form', '') for token in sentence]
                upos = [token.get('upos', '_') for token in sentence]
                xpos = [token.get('xpos', '_') for token in sentence]
                feats = [token.get('feats', '_') for token in sentence]
                lemmas = [token.get('lemma', '_') for token in sentence]
                norms = [token.get('norm_form', '_') for token in sentence]
                heads = [token.get('head', 0) for token in sentence]
                deprels = [token.get('deprel', '_') for token in sentence]
                orig_forms = [token.get('orig_form', '_') for token in sentence]
                
                # NOTE: Removed UPOS context tokens - they were hurting performance
                # BERT's contextual understanding is already strong enough
                # Simply use the words as-is
                words_ctx = words
                
                words_list.append(words)
                upos_list.append(upos)
                xpos_list.append(xpos)
                feats_list.append(feats)
                if self.config.train_lemmatizer:
                    lemma_list.append([l.lower() if l != '_' else '_' for l in lemmas])
                if self.config.train_normalizer:
                    norm_list.append([n.lower() if n != '_' else '_' for n in norms])
                if self.config.train_parser:
                    head_list.append(heads)
                    deprel_list.append(deprels)
                    # Collect original forms for transpositional parsing
                    if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                        orig_form_list.append([of.lower() if of != '_' else '_' for of in orig_forms])
            
            # Tokenize words (no context tokens anymore)
            tokenized = tokenizer(
                words_list,  # Use original words, not context-enhanced
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # Align labels - need to map from context-enhanced word indices back to original word indices
            aligned_upos = []
            aligned_xpos = []
            aligned_feats = []
            aligned_lemma = []
            aligned_norm = []
            aligned_head = []
            aligned_deprel = []
            aligned_orig_form = []
            
            for i, (words, upos_labels, xpos_labels, feats_labels) in enumerate(zip(words_list, upos_list, xpos_list, feats_list)):
                if self.config.train_lemmatizer:
                    lemma_labels = lemma_list[i]
                if self.config.train_normalizer:
                    norm_labels = norm_list[i]
                if self.config.train_parser:
                    head_labels = head_list[i]
                    deprel_labels = deprel_list[i]
                    if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id and orig_form_list:
                        orig_form_labels = orig_form_list[i]
                word_ids = tokenized.word_ids(batch_index=i)
                aligned_upos_seq = []
                aligned_xpos_seq = []
                aligned_feats_seq = []
                aligned_lemma_seq = []
                aligned_norm_seq = []
                aligned_head_seq = []
                aligned_deprel_seq = []
                aligned_orig_form_seq = []
                
                # Since we no longer use context tokens, mapping is 1:1
                # But keep the structure for compatibility
                ctx_to_orig = {i: i for i in range(len(words_ctx))}
                
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens (CLS, SEP, PAD)
                        aligned_upos_seq.append(-100)
                        aligned_xpos_seq.append(-100)
                        aligned_feats_seq.append(-100)
                        if self.config.train_lemmatizer:
                            aligned_lemma_seq.append(-100)
                        if self.config.train_normalizer:
                            aligned_norm_seq.append(-100)
                        if self.config.train_parser:
                            aligned_head_seq.append(-100)
                            aligned_deprel_seq.append(-100)
                            if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                                aligned_orig_form_seq.append(-100)
                    elif word_idx != previous_word_idx:
                        # First subword token of a context-enhanced word
                        # Map back to original word index
                        orig_word_idx = ctx_to_orig.get(word_idx)
                        if orig_word_idx is not None and orig_word_idx < len(upos_labels):
                            # This is an actual word (not a UPOS token)
                            upos = upos_labels[orig_word_idx] if orig_word_idx < len(upos_labels) else '_'
                            xpos = xpos_labels[orig_word_idx] if orig_word_idx < len(xpos_labels) else '_'
                            feats = feats_labels[orig_word_idx] if orig_word_idx < len(feats_labels) else '_'
                            
                            aligned_upos_seq.append(self.upos_to_id.get(upos, 0))
                            aligned_xpos_seq.append(self.xpos_to_id.get(xpos, 0))
                            aligned_feats_seq.append(self.feats_to_id.get(feats, 0))
                            
                            if self.config.train_lemmatizer:
                                lemma = lemma_labels[orig_word_idx] if orig_word_idx < len(lemma_labels) else '_'
                                lemma_lower = lemma.lower() if lemma != '_' else '_'
                                aligned_lemma_seq.append(self.lemma_to_id.get(lemma_lower, 0))
                            
                            if self.config.train_normalizer:
                                norm = norm_labels[orig_word_idx] if orig_word_idx < len(norm_labels) else '_'
                                norm_lower = norm.lower() if norm != '_' else '_'
                                aligned_norm_seq.append(self.norm_to_id.get(norm_lower, 0))
                            
                            if self.config.train_parser:
                                # Head: map to token index in sequence (0-based, -100 for root)
                                head_val = head_labels[orig_word_idx] if orig_word_idx < len(head_labels) else 0
                                try:
                                    head_int = int(head_val) if str(head_val).isdigit() else 0
                                    # Head is 1-based in CoNLL-U, need to map to token position
                                    # For now, use relative position (will need adjustment for subword tokens)
                                    aligned_head_seq.append(head_int)
                                except (ValueError, TypeError):
                                    aligned_head_seq.append(0)
                                
                                deprel = deprel_labels[orig_word_idx] if orig_word_idx < len(deprel_labels) else '_'
                                aligned_deprel_seq.append(self.deprel_to_id.get(deprel, 0))
                                
                                # Original form for transpositional parsing
                                if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id and orig_form_list:
                                    orig_form = orig_form_labels[orig_word_idx] if orig_word_idx < len(orig_form_labels) else '_'
                                    orig_form_lower = orig_form.lower() if orig_form != '_' else '_'
                                    aligned_orig_form_seq.append(self.orig_form_to_id.get(orig_form_lower, self.orig_form_to_id.get('_', 0)))
                        else:
                            # Should not happen now, but keep for safety
                            aligned_upos_seq.append(-100)
                            aligned_xpos_seq.append(-100)
                            aligned_feats_seq.append(-100)
                            if self.config.train_lemmatizer:
                                aligned_lemma_seq.append(-100)
                            if self.config.train_normalizer:
                                aligned_norm_seq.append(-100)
                            if self.config.train_parser:
                                aligned_head_seq.append(-100)
                                aligned_deprel_seq.append(-100)
                                if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                                    aligned_orig_form_seq.append(-100)
                    else:
                        # Subsequent subword tokens - use -100 to ignore
                        aligned_upos_seq.append(-100)
                        aligned_xpos_seq.append(-100)
                        aligned_feats_seq.append(-100)
                        if self.config.train_lemmatizer:
                            aligned_lemma_seq.append(-100)
                        if self.config.train_normalizer:
                            aligned_norm_seq.append(-100)
                        if self.config.train_parser:
                            aligned_head_seq.append(-100)
                            aligned_deprel_seq.append(-100)
                            if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                                aligned_orig_form_seq.append(-100)
                    
                    previous_word_idx = word_idx
                
                aligned_upos.append(aligned_upos_seq)
                aligned_xpos.append(aligned_xpos_seq)
                aligned_feats.append(aligned_feats_seq)
                if self.config.train_lemmatizer:
                    aligned_lemma.append(aligned_lemma_seq)
                if self.config.train_normalizer:
                    aligned_norm.append(aligned_norm_seq)
                if self.config.train_parser:
                    aligned_head.append(aligned_head_seq)
                    aligned_deprel.append(aligned_deprel_seq)
                    if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                        aligned_orig_form.append(aligned_orig_form_seq)
            
            tokenized['labels_upos'] = aligned_upos
            tokenized['labels_xpos'] = aligned_xpos
            tokenized['labels_feats'] = aligned_feats
            if self.config.train_lemmatizer:
                tokenized['labels_lemma'] = aligned_lemma
            if self.config.train_normalizer:
                tokenized['labels_norm'] = aligned_norm
            if self.config.train_parser:
                tokenized['labels_head'] = aligned_head
                tokenized['labels_deprel'] = aligned_deprel
                if hasattr(self, 'orig_form_to_id') and self.orig_form_to_id:
                    tokenized['orig_form_ids'] = aligned_orig_form
            
            return tokenized
        
        # Create dataset with sentences
        dataset_dict = {'sentences': sentences}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize and align
        dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=100,
            remove_columns=['sentences']
        )
        
        return dataset
    
    def tag(self, input_file: Path, output_file: Optional[Path] = None, format: str = "conllu", 
            segment: bool = False, tokenize: bool = False) -> List[List[Dict]]:
        """Tag sentences from input file.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file (optional)
            format: Input format ('conllu', 'teitok', 'plain', 'text', 'raw')
            segment: If True, segment raw text into sentences (for 'raw' or 'plain' format)
            tokenize: If True, tokenize sentences into words (for 'raw' or 'plain' format)
        """
        import time
        start_time = time.time()
        
        if self.config.debug:
            print(f"[DEBUG] tag() called: input_file={input_file}, format={format}, segment={segment}, tokenize={tokenize}", file=sys.stderr)
        # Only load model if we need it (not needed for normalization-only with vocabulary)
        # Check if we're doing normalization-only (no model needed if using vocabulary-based normalization)
        # Normalization-only mode: normalize=True, no parse, no tag_only, no model, and no Viterbi tagging requested
        # Note: If model has trained normalizer, it will be used during tagging phase
        tag_only = getattr(self.config, 'tag_only', False)
        use_viterbi_available = (self.transition_probs and 
                                 self.external_vocab and 
                                 'upos' in self.transition_probs)
        normalization_only = (self.config.normalize and 
                              not self.config.parse and
                              not tag_only and
                              not use_viterbi_available)
        
        if normalization_only and not self.model:
            # Normalization-only mode: just merge vocabularies if external vocab provided
            # Vocabulary-based normalization doesn't require a model or transformers
            if self.external_vocab:
                # Merge with model vocab if available (from a saved model)
                model_vocab_file = None
                if self.model_path and Path(self.model_path).exists():
                    model_vocab_file = Path(self.model_path) / 'model_vocab.json'
                
                if model_vocab_file and model_vocab_file.exists():
                    with open(model_vocab_file, 'r', encoding='utf-8') as f:
                        model_vocab_data = json.load(f)
                        # Handle new format
                        if isinstance(model_vocab_data, dict) and 'vocab' in model_vocab_data:
                            self.model_vocab = model_vocab_data.get('vocab', {})
                        else:
                            self.model_vocab = model_vocab_data
                
                # Merge vocabularies (external overrides model)
                self.vocab = {**self.model_vocab, **self.external_vocab}
                # Rebuild lemmatization patterns with merged vocab
                self._build_lemmatization_patterns(self.vocab)
                print("Normalization-only mode: Using vocabulary-based normalization (no model required)", file=sys.stderr)
            else:
                # No external vocab, try to load model vocab if available
                model_vocab_file = None
                if self.model_path and Path(self.model_path).exists():
                    model_vocab_file = Path(self.model_path) / 'model_vocab.json'
                
                if model_vocab_file and model_vocab_file.exists():
                    with open(model_vocab_file, 'r', encoding='utf-8') as f:
                        model_vocab_data = json.load(f)
                        # Handle new format
                        if isinstance(model_vocab_data, dict) and 'vocab' in model_vocab_data:
                            self.vocab = model_vocab_data.get('vocab', {})
                        else:
                            self.vocab = model_vocab_data
                    print("Normalization-only mode: Using vocabulary-based normalization (no model required)", file=sys.stderr)
                else:
                    self.vocab = {}
                    print("Warning: No vocabulary provided for normalization. Provide --vocab or --model with model_vocab.json", file=sys.stderr)
        elif not self.model:
            # Check if we can use Viterbi tagging (vocab with transitions available)
            use_viterbi = (self.transition_probs and 
                          self.external_vocab and 
                          'upos' in self.transition_probs)
            
            if use_viterbi:
                # Viterbi tagging mode: use vocabulary-based tagging (no model needed)
                # Merge vocabularies if needed
                if self.external_vocab:
                    model_vocab_file = None
                    if self.model_path and Path(self.model_path).exists():
                        model_vocab_file = Path(self.model_path) / 'model_vocab.json'
                    
                    if model_vocab_file and model_vocab_file.exists():
                        with open(model_vocab_file, 'r', encoding='utf-8') as f:
                            model_vocab_data = json.load(f)
                            # Handle new format
                            if isinstance(model_vocab_data, dict) and 'vocab' in model_vocab_data:
                                self.model_vocab = model_vocab_data.get('vocab', {})
                            else:
                                self.model_vocab = model_vocab_data
                    
                    # Merge vocabularies (external overrides model)
                    self.vocab = {**self.model_vocab, **self.external_vocab}
                    # Rebuild lemmatization patterns with merged vocab
                    self._build_lemmatization_patterns(self.vocab)
                    print("Viterbi tagging mode: Using vocabulary-based tagging (no model required)", file=sys.stderr)
                else:
                    self.vocab = {}
                    print("Warning: No vocabulary provided for Viterbi tagging", file=sys.stderr)
            else:
                # Need model for tagging/parsing - requires transformers
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers library required for tagging/parsing. For Viterbi tagging, provide --vocab with transition probabilities. For normalization-only mode, use --normalize with --vocab (no --model needed).")
                self.load_model()
        
        # Auto-detect format if not specified
        if format == "auto":
            ext = input_file.suffix.lower()
            if ext == '.conllu' or ext == '.conll':
                format = "conllu"
            elif ext == '.xml':
                format = "teitok"
            elif ext == '.txt' or ext == '.text':
                # Check if it looks like raw text (multiple sentences) or pre-tokenized
                with open(input_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    # If first line has multiple sentence-ending punctuation, likely raw text
                    if len(re.findall(r'[.!?]', first_line)) > 1:
                        format = "raw"
                        segment = True
                        tokenize = True
                    else:
                        format = "plain"
            else:
                format = "plain"
        
        # Load input
        if format == "teitok":
            # Store original XML tree for later preservation
            try:
                import xml.etree.ElementTree as ET
                self.original_teitok_tree = ET.parse(input_file)
                self.original_teitok_file = input_file
            except Exception as e:
                if self.config.debug:
                    print(f"[DEBUG] Could not store original TEITOK tree: {e}", file=sys.stderr)
            sentences = load_teitok_xml(input_file, normalization_attr=self.config.normalization_attr)
            # Track that we processed TEITOK input
            self.operations_performed.append("processed_teitok_input")
        elif format == "raw":
            # Raw text: segment and tokenize
            with open(input_file, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Segment sentences directly from original text (preserving exact spacing)
            # Use a simpler sentence segmentation that preserves original text
            sentences = []
            
            # Simple sentence segmentation: split on sentence-ending punctuation followed by whitespace or end
            # But preserve the original text exactly
            sentence_endings = r'[.!?]+'
            pattern = rf'({sentence_endings})(?:\s+|$)'
            
            # Find all sentence boundaries
            # Use a simpler approach: find sentence boundaries and extract directly from full_text
            sentences = []
            sent_start = 0  # Start position of current sentence in full_text
            
            # Find all sentence-ending punctuation positions
            for match in re.finditer(pattern, full_text):
                # Found sentence-ending punctuation
                # match.group(1) is the punctuation (capturing group)
                # match.end() is the end of the entire match (punctuation + optional whitespace)
                punct_match_end = match.end()
                
                # The pattern matches: punctuation + (whitespace or end of string)
                # So if there's whitespace, it's already included in the match
                # We want to extract from sent_start to the end of the punctuation
                # and include one space if there was whitespace in the match
                
                # Check what was matched: if there's whitespace after punctuation in the match
                punct_start = match.start(1)  # Start of punctuation (capturing group 1)
                punct_end = match.end(1)  # End of punctuation (capturing group 1)
                
                # Check if there's whitespace after the punctuation (within the match or after)
                if punct_end < len(full_text) and full_text[punct_end].isspace():
                    # Include one trailing space
                    sent_end = punct_end + 1
                    # Skip any additional whitespace for next sentence
                    next_sent_start = sent_end
                    while next_sent_start < len(full_text) and full_text[next_sent_start].isspace():
                        next_sent_start += 1
                else:
                    # No space after - sentence ends at punctuation
                    sent_end = punct_end
                    # Skip any whitespace for next sentence (shouldn't be any, but just in case)
                    next_sent_start = sent_end
                    while next_sent_start < len(full_text) and full_text[next_sent_start].isspace():
                        next_sent_start += 1
                
                # Extract the sentence text from sent_start to sent_end
                original_sent_text = full_text[sent_start:sent_end]
                
                if original_sent_text.strip():
                    words = tokenize_words_ud_style(original_sent_text)
                    sentence_tokens = []
                    for word_idx, word in enumerate(words, 1):
                        sentence_tokens.append({
                            'id': word_idx,
                            'form': word,
                            'lemma': '_',
                            'upos': '_',
                            'xpos': '_',
                            'feats': '_',
                            'head': '_',  # Use '_' when no parser
                            'deprel': '_',
                        })
                    # Store original text in first token for accurate spacing reconstruction
                    if sentence_tokens:
                        sentence_tokens[0]['_original_text'] = original_sent_text
                        if self.config.debug and len(sentences) == 0:
                            print(f"[DEBUG] Stored original sentence (length {len(original_sent_text)}): {repr(original_sent_text[:150])}", file=sys.stderr)
                    sentences.append(sentence_tokens)
                
                # Move to next sentence start
                sent_start = next_sent_start
            
            # Add remaining text as final sentence if any
            if sent_start < len(full_text):
                # For the final sentence, extract from sent_start to end
                # But don't include trailing whitespace (it's the end of the file)
                original_sent_text = full_text[sent_start:].rstrip()
                words = tokenize_words_ud_style(original_sent_text)
                sentence_tokens = []
                for word_idx, word in enumerate(words, 1):
                    sentence_tokens.append({
                        'id': word_idx,
                        'form': word,
                        'lemma': '_',
                        'upos': '_',
                        'xpos': '_',
                        'feats': '_',
                        'head': '_',  # Use '_' when no parser
                        'deprel': '_',
                    })
                # Store original text in first token for accurate spacing reconstruction
                if sentence_tokens:
                    sentence_tokens[0]['_original_text'] = original_sent_text
                sentences.append(sentence_tokens)
        elif format == "plain" or format == "text":
            sentences = load_plain_text(input_file, segment=segment, tokenize=tokenize)
        else:
            sentences = load_conllu_file(input_file)
        
        if self.config.debug:
            print(f"[DEBUG] Loaded {len(sentences)} sentences", file=sys.stderr)
        
        # Preprocessing: normalization and contraction splitting (for historic documents)
        if self.config.normalize or self.config.split_contractions:
            if self.config.debug:
                print(f"[DEBUG] Preprocessing sentences (normalize={self.config.normalize}, split_contractions={self.config.split_contractions})", file=sys.stderr)
            sentences = self._preprocess_sentences(sentences)
            if self.config.debug:
                print(f"[DEBUG] After preprocessing: {len(sentences)} sentences", file=sys.stderr)
        
        # Check if we should use Viterbi tagging (vocab with transitions, no model, tagging requested)
        use_viterbi = (not normalization_only and 
                      not self.model and 
                      self.transition_probs and 
                      self.vocab and
                      'upos' in self.transition_probs)
        
        # If only normalizing (no tagging/parsing and no model), write output directly
        if normalization_only and not self.model:
            if self.config.debug:
                print(f"[DEBUG] Entering normalization-only mode", file=sys.stderr)
            # Normalization-only mode: write normalized output
            tagged_sentences = []
            if not sentences:
                print("Warning: No sentences found in input file", file=sys.stderr)
                return []
            for sent_idx, sentence in enumerate(sentences):
                if self.config.debug:
                    print(f"[DEBUG] Processing sentence {sent_idx + 1}/{len(sentences)}: {len(sentence)} tokens", file=sys.stderr)
                tagged_sentence = []
                for word_idx, token in enumerate(sentence):
                    # Ensure token IDs are sequential (1-based for CoNLL-U)
                    token_id = token.get('id', word_idx + 1)
                    if token_id == 0:
                        token_id = word_idx + 1
                    
                    # Use normalized form if available, otherwise original form
                    form = token.get('form', '_')
                    if 'norm_form' in token and token.get('norm_form') and token.get('norm_form') != '_':
                        # Optionally update form to normalized form (for display)
                        # But keep orig_form for MISC column
                        pass  # Keep original form, normalization goes in norm_form
                    
                    tagged_token = {
                        'id': token_id,
                        'form': form,
                        'lemma': token.get('lemma', '_'),
                        'upos': token.get('upos', '_'),
                        'xpos': token.get('xpos', '_'),
                        'feats': token.get('feats', '_'),
                        'head': token.get('head', '_'),  # Use '_' when no parser
                        'deprel': token.get('deprel', '_'),
                    }
                    # Preserve _original_text and sentence ID if present (for # text = and # sent_id = comments)
                    if '_original_text' in token:
                        tagged_token['_original_text'] = token.get('_original_text')
                    if '_sentence_id' in token:
                        tagged_token['_sentence_id'] = token.get('_sentence_id')
                    # Preserve token ID from TEITOK XML (@id or @xml:id)
                    if 'tok_id' in token and token.get('tok_id'):
                        tagged_token['tok_id'] = token.get('tok_id')
                    if 'dtok_id' in token and token.get('dtok_id'):
                        tagged_token['dtok_id'] = token.get('dtok_id')
                    # Add normalization fields if present
                    if 'norm_form' in token and token.get('norm_form'):
                        tagged_token['norm_form'] = token.get('norm_form')
                    if 'orig_form' in token:
                        tagged_token['orig_form'] = token.get('orig_form', form)
                    if 'split_forms' in token:
                        tagged_token['split_forms'] = token.get('split_forms', None)
                    tagged_sentence.append(tagged_token)
                tagged_sentences.append(tagged_sentence)
            total_tokens = sum(len(s) for s in tagged_sentences)
            elapsed_time = time.time() - start_time
            tok_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            print(f"Normalization-only mode: Processed {len(tagged_sentences)} sentences, {total_tokens} tokens in {elapsed_time:.2f}s ({tok_per_sec:.1f} tok/s)", file=sys.stderr)
            if self.config.debug:
                print(f"[DEBUG] Returning {len(tagged_sentences)} tagged sentences from normalization-only mode", file=sys.stderr)
            # Return early - don't continue to model-based processing
            # Write output will be handled by caller
            return tagged_sentences
        elif use_viterbi:
            # Viterbi tagging mode: use vocabulary-based tagging with transition probabilities
            if self.config.debug:
                print(f"[DEBUG] Entering Viterbi tagging mode", file=sys.stderr)
            tagged_sentences = []
            for sent_idx, sentence in enumerate(sentences):
                if self.config.debug:
                    print(f"[DEBUG] Processing sentence {sent_idx + 1}/{len(sentences)}: {len(sentence)} tokens", file=sys.stderr)
                
                # Extract word forms
                words = [token.get('form', '_') for token in sentence]
                
                # Get capitalizable tags statistics from vocabulary metadata (language-agnostic)
                capitalizable_tags = None
                if self.vocab_metadata and 'capitalizable_tags' in self.vocab_metadata:
                    # Use statistics directly (already in dict format)
                    capitalizable_tags = self.vocab_metadata['capitalizable_tags']
                
                # Tag with Viterbi
                upos_tags = viterbi_tag_sentence(words, self.vocab, self.transition_probs, tag_type='upos', capitalizable_tags=capitalizable_tags)
                xpos_tags = viterbi_tag_sentence(words, self.vocab, self.transition_probs, tag_type='xpos', capitalizable_tags=capitalizable_tags)
                
                # Build tagged sentence
                tagged_sentence = []
                for word_idx, token in enumerate(sentence):
                    token_id = token.get('id', word_idx + 1)
                    if token_id == 0:
                        token_id = word_idx + 1
                    
                    form = token.get('form', '_')
                    
                    # Get predicted tags
                    upos = upos_tags[word_idx] if word_idx < len(upos_tags) else '_'
                    xpos = xpos_tags[word_idx] if word_idx < len(xpos_tags) else '_'
                    
                    # Apply normalization if enabled (before lemmatization)
                    # This ensures we use the normalized form for lemma lookup
                    norm_form = None
                    if self.config.normalize and self.vocab:
                        # Try to normalize the word using vocabulary
                        normalized = normalize_word(
                            form,
                            self.vocab,
                            conservative=self.config.conservative_normalization,
                            similarity_threshold=self.config.similarity_threshold,
                            inflection_suffixes=self.inflection_suffixes
                        )
                        if normalized:
                            norm_form = normalized
                        elif 'norm_form' in token and token.get('norm_form') and token.get('norm_form') != '_':
                            # Use normalization from preprocessing if available
                            norm_form = token.get('norm_form')
                    elif 'norm_form' in token and token.get('norm_form') and token.get('norm_form') != '_':
                        # Use normalization from preprocessing if available
                        norm_form = token.get('norm_form')
                    
                    # Get FEATS from vocab (use UPOS+FEATS matching by default, fallback to XPOS)
                    feats = '_'
                    entry = self.vocab.get(form) or self.vocab.get(form.lower())
                    # If not in vocab, try lexicon (OOV fallback)
                    if not entry and self.lexicon:
                        entry = self.lexicon.get(form) or self.lexicon.get(form.lower())
                    if entry:
                        if isinstance(entry, list):
                            # Find best matching analysis based on config
                            best_entry = None
                            best_count = 0
                            use_xpos = self.config.use_xpos_for_tagging

                            if use_xpos:
                                # XPOS-first: try XPOS match, then UPOS+FEATS
                                for analysis in entry:
                                    if xpos and xpos != '_' and analysis.get('xpos') == xpos:
                                        count = analysis.get('count', 0)
                                        if count > best_count:
                                            best_count = count
                                            best_entry = analysis

                                if not best_entry and upos and upos != '_':
                                    for analysis in entry:
                                        if analysis.get('upos') == upos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis

                                if not best_entry and xpos and xpos != '_':
                                    for analysis in entry:
                                        if analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis
                            else:
                                # UPOS+FEATS-first (default): try UPOS match first
                                for analysis in entry:
                                    if upos and upos != '_' and analysis.get('upos') == upos:
                                        count = analysis.get('count', 0)
                                        if count > best_count:
                                            best_count = count
                                            best_entry = analysis

                                if not best_entry and xpos and xpos != '_':
                                    for analysis in entry:
                                        if analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis

                            if not best_entry:
                                # Fallback to most frequent
                                best_entry = max(entry, key=lambda a: a.get('count', 0))
                            feats = best_entry.get('feats', '_')
                        else:
                            feats = entry.get('feats', '_')
                    
                    # Get lemma: check vocab entry first, using reg form if available
                    # The lemma in vocab corresponds to the reg form (if present), not the original form
                    # Also store best_entry for expan lookup (must match the selected XPOS/UPOS)
                    lemma = '_'
                    best_entry_for_expan = None  # Store the entry that was selected for tagging
                    entry = self.vocab.get(form) or self.vocab.get(form.lower())
                    
                    if entry:
                        best_entry = None
                        if isinstance(entry, list):
                            # Find best matching analysis based on UPOS/XPOS/FEATS
                            # Prioritize based on use_xpos_for_tagging config
                            best_entry = None
                            best_count = 0
                            use_xpos = self.config.use_xpos_for_tagging

                            if use_xpos:
                                # XPOS-first: try XPOS match, then UPOS+FEATS
                                for analysis in entry:
                                    if xpos and xpos != '_' and analysis.get('xpos') == xpos:
                                        count = analysis.get('count', 0)
                                        if count > best_count:
                                            best_count = count
                                            best_entry = analysis

                                if not best_entry and upos and upos != '_':
                                    for analysis in entry:
                                        if analysis.get('upos') == upos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis

                                if not best_entry and xpos and xpos != '_':
                                    for analysis in entry:
                                        if analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis
                            else:
                                # UPOS+FEATS-first (default): try UPOS match first
                                for analysis in entry:
                                    if upos and upos != '_' and analysis.get('upos') == upos:
                                        count = analysis.get('count', 0)
                                        if count > best_count:
                                            best_count = count
                                            best_entry = analysis

                                if not best_entry and xpos and xpos != '_':
                                    for analysis in entry:
                                        if analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis

                            if not best_entry:
                                # Fallback to most frequent
                                best_entry = max(entry, key=lambda a: a.get('count', 0))
                        elif isinstance(entry, dict):
                            best_entry = entry
                        
                        # Store best_entry for expan lookup (only use expan from the selected entry)
                        best_entry_for_expan = best_entry
                        
                        if best_entry:
                            # If entry has reg field, lemma corresponds to reg form, not original
                            # But we should use the lemma directly from the entry, not do a new lookup
                            reg = best_entry.get('reg')
                            entry_lemma = best_entry.get('lemma', '_')
                            
                            if reg and reg != '_' and reg != form:
                                # Entry has reg: lemma in this entry corresponds to reg form
                                # Use the lemma directly from this entry (don't do a new lookup)
                                if entry_lemma and entry_lemma != '_':
                                    lemma = entry_lemma
                                else:
                                    # No lemma in entry, but has reg - lookup lemma for reg form
                                    lemma = self._predict_from_vocab(reg, 'lemma', xpos=xpos, upos=upos, feats=feats)
                            else:
                                # No reg field: use lemma directly from entry
                                if entry_lemma and entry_lemma != '_':
                                    lemma = entry_lemma
                    
                    # If no lemma found yet, try normalization form if available
                    if lemma == '_':
                        if norm_form and norm_form != '_':
                            lemma = self._predict_from_vocab(norm_form, 'lemma', xpos=xpos, upos=upos, feats=feats)
                    
                    # If still no lemma, try original form with pattern-based lemmatization
                    if lemma == '_':
                        lemma = self._predict_from_vocab(form, 'lemma', xpos=xpos, upos=upos, feats=feats)
                    
                    # Final fallback
                    if lemma == '_':
                        lemma = form.lower()
                    
                    tagged_token = {
                        'id': token_id,
                        'form': form,
                        'lemma': lemma if lemma != '_' else form.lower(),  # Fallback to lowercase form
                        'upos': upos,
                        'xpos': xpos,
                        'feats': feats,
                        'head': '_',  # No parsing in Viterbi mode
                        'deprel': '_',
                    }
                    
                    # Preserve original text and sentence ID
                    if '_original_text' in token:
                        tagged_token['_original_text'] = token.get('_original_text')
                    if '_sentence_id' in token:
                        tagged_token['_sentence_id'] = token.get('_sentence_id')
                    
                    # Preserve token ID from TEITOK XML (@id or @xml:id)
                    if 'tok_id' in token and token.get('tok_id'):
                        tagged_token['tok_id'] = token.get('tok_id')
                    if 'dtok_id' in token and token.get('dtok_id'):
                        tagged_token['dtok_id'] = token.get('dtok_id')
                    
                    # Add normalization fields
                    if norm_form:
                        tagged_token['norm_form'] = norm_form
                    if 'orig_form' in token:
                        tagged_token['orig_form'] = token.get('orig_form')
                    
                    # Add expan from vocabulary entry that was selected for tagging
                    # Only use expan from the same entry that matched the XPOS/UPOS
                    expan = None
                    if best_entry_for_expan:
                        # Only use expan from the entry that was actually selected for tagging
                        if 'expan' in best_entry_for_expan and best_entry_for_expan.get('expan') and best_entry_for_expan.get('expan') != '_':
                            expan = best_entry_for_expan.get('expan')
                    
                    if expan:
                        tagged_token['expan'] = expan
                    
                    tagged_sentence.append(tagged_token)
                
                tagged_sentences.append(tagged_sentence)
            
            total_tokens = sum(len(s) for s in tagged_sentences)
            elapsed_time = time.time() - start_time
            tok_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            print(f"Viterbi tagging complete: {len(tagged_sentences)} sentences, {total_tokens} tokens in {elapsed_time:.2f}s ({tok_per_sec:.1f} tok/s)", file=sys.stderr)
            if self.config.debug:
                print(f"[DEBUG] Viterbi tagging complete: {len(tagged_sentences)} sentences", file=sys.stderr)
            
            # Track operations
            if self.config.normalize:
                self.operations_performed.append("normalized")
            if self.config.parse:
                self.operations_performed.append("parsed")
            else:
                self.operations_performed.append("tagged")
            self.operations_performed.append("lemmatized")
            self._last_tagged_sentences = tagged_sentences  # Store for revision statement
            
            # Write output if specified
            if output_file:
                self.write_output(tagged_sentences, output_file, format)
            
            return tagged_sentences
        else:
            # Need model for tagging/parsing - process sentences in batches
            tagged_sentences = []
            batch_size = 32
            for batch_start in range(0, len(sentences), batch_size):
                batch_sentences = sentences[batch_start:batch_start + batch_size]
            
            # Prepare batch (no context tokens - removed for better performance)
            words_batch = []
            
            for sentence in batch_sentences:
                words = [token.get('form', '') for token in sentence]
                words_batch.append(words)
            
            # Tokenize words directly
            tokenized = self.tokenizer(
                words_batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Move inputs to device (MPS/CUDA/CPU)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            # Extract original form IDs for transpositional parsing (if available)
            orig_form_ids = None
            if (hasattr(self, 'orig_form_to_id') and self.orig_form_to_id and 
                self.model.use_orig_form_for_parser and self.model.orig_form_embed is not None):
                # Extract orig_form from tokens and convert to IDs
                orig_form_ids_list = []
                for sent_idx, sentence in enumerate(batch_sentences):
                    word_ids = tokenized.word_ids(batch_index=sent_idx)
                    orig_form_seq = []
                    current_word_idx = None
                    for token_idx, word_id in enumerate(word_ids):
                        if word_id is not None and word_id != current_word_idx:
                            current_word_idx = word_id
                            if word_id < len(sentence):
                                orig_form = sentence[word_id].get('orig_form', '_')
                                orig_form_lower = orig_form.lower() if orig_form != '_' else '_'
                                orig_form_id = self.orig_form_to_id.get(orig_form_lower, self.orig_form_to_id.get('_', 0))
                                orig_form_seq.append(orig_form_id)
                            else:
                                orig_form_seq.append(self.orig_form_to_id.get('_', 0))
                        elif word_id is None:
                            # Special token (CLS, SEP, PAD) - use '_' ID
                            orig_form_seq.append(self.orig_form_to_id.get('_', 0))
                        else:
                            # Subsequent subword token - use same ID as first subword
                            if orig_form_seq:
                                orig_form_seq.append(orig_form_seq[-1])
                            else:
                                orig_form_seq.append(self.orig_form_to_id.get('_', 0))
                    # Pad to max_length
                    while len(orig_form_seq) < self.config.max_length:
                        orig_form_seq.append(self.orig_form_to_id.get('_', 0))
                    orig_form_ids_list.append(orig_form_seq[:self.config.max_length])
                
                # Convert to tensor
                orig_form_ids = torch.tensor(orig_form_ids_list, dtype=torch.long, device=self.device)
            
            # Predict with model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    orig_form_ids=orig_form_ids
                )
                
                # Get predictions and confidence scores for confidence-based blending
                # Compute softmax probabilities to get confidence scores
                probs_upos = torch.nn.functional.softmax(outputs['logits_upos'], dim=-1)
                probs_xpos = torch.nn.functional.softmax(outputs['logits_xpos'], dim=-1)
                probs_feats = torch.nn.functional.softmax(outputs['logits_feats'], dim=-1)
                
                # Get max probabilities (confidence scores)
                max_probs_upos, pred_upos = torch.max(probs_upos, dim=-1)
                max_probs_xpos, pred_xpos = torch.max(probs_xpos, dim=-1)
                max_probs_feats, pred_feats = torch.max(probs_feats, dim=-1)
                
                # Lemma predictions and confidence (if lemmatizer was trained)
                pred_lemmas = None
                probs_lemma = None
                max_probs_lemma = None
                if self.model.train_lemmatizer and outputs.get('logits_lemma') is not None:
                    probs_lemma = torch.nn.functional.softmax(outputs['logits_lemma'], dim=-1)
                    max_probs_lemma, pred_lemmas = torch.max(probs_lemma, dim=-1)  # [batch, seq]
                
                # Normalization predictions (if normalizer was trained)
                pred_norms = None
                if self.model.train_normalizer and outputs.get('logits_norm') is not None:
                    pred_norms = torch.argmax(outputs['logits_norm'], dim=-1)  # [batch, seq]
                
                # Parsing predictions (if parser was trained)
                pred_heads = None
                pred_deprels = None
                if self.config.parse and self.model.train_parser and outputs.get('arc_scores') is not None:
                    arc_scores = outputs['arc_scores']
                    # arc_scores: [batch, seq, seq] where arc_scores[i, j] = score for token j having head i
                    # Mask invalid positions (padding tokens)
                    mask = attention_mask.unsqueeze(1).expand_as(arc_scores)  # [batch, seq, seq]
                    arc_scores = arc_scores.masked_fill(~mask.bool(), float('-inf'))
                    
                    # For each token, find the best head (argmax over head dimension)
                    pred_heads = torch.argmax(arc_scores, dim=1)  # [batch, seq] - head index for each token
                    
                    # Predict deprels for each token
                    if outputs.get('logits_deprel') is not None:
                        pred_deprels = torch.argmax(outputs['logits_deprel'], dim=-1)  # [batch, seq]
            
            # Map predictions back to words
            for sent_idx, sentence in enumerate(batch_sentences):
                tagged_sentence = []
                word_ids = tokenized.word_ids(batch_index=sent_idx)
                words = words_batch[sent_idx]
                
                # No context tokens, so mapping is 1:1
                ctx_to_orig = {i: i for i in range(len(words))}
                
                current_word_idx = None
                word_predictions = {}  # orig_word_idx -> (upos, xpos, feats, lemma, norm, head, deprel)
                word_confidence = {}  # orig_word_idx -> (upos_conf, xpos_conf, feats_conf, lemma_conf) for confidence-based blending
                token_to_word = {}  # token_idx -> orig_word_idx (for head mapping)
                
                # Collect predictions for each word (take first subword token)
                for token_idx, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != current_word_idx:
                        current_word_idx = word_id
                        # Map from context-enhanced word index to original
                        orig_word_idx = ctx_to_orig.get(word_id)
                        if orig_word_idx is not None and orig_word_idx < len(sentence):
                            upos_id = pred_upos[sent_idx][token_idx].item()
                            xpos_id = pred_xpos[sent_idx][token_idx].item()
                            feats_id = pred_feats[sent_idx][token_idx].item()
                            
                            # Get confidence scores
                            upos_conf = max_probs_upos[sent_idx][token_idx].item()
                            xpos_conf = max_probs_xpos[sent_idx][token_idx].item()
                            feats_conf = max_probs_feats[sent_idx][token_idx].item()
                            lemma_conf = max_probs_lemma[sent_idx][token_idx].item() if max_probs_lemma is not None else 0.0
                            
                            upos = self.id_to_upos.get(upos_id, '_')
                            xpos = self.id_to_xpos.get(xpos_id, '_')
                            
                            # FEATS prediction: check if FEATS were actually trained
                            # If FEATS labels only contain '_' or very few labels, FEATS weren't in training data
                            feats = '_'
                            feats_were_trained = False
                            if hasattr(self, 'feats_labels'):
                                if len(self.feats_labels) > 2:
                                    feats_were_trained = True
                                elif len(self.feats_labels) == 2 and '_' not in self.feats_labels:
                                    feats_were_trained = True
                            
                            if feats_were_trained:
                                # FEATS were trained - use model prediction
                                feats = self.id_to_feats.get(feats_id, '_')
                            # If FEATS weren't trained, feats stays '_' and will fall back to vocabulary later
                            
                            # Store confidence scores for confidence-based blending
                            word_confidence[orig_word_idx] = (upos_conf, xpos_conf, feats_conf, lemma_conf)
                            
                            # Lemma prediction: only use if lemmatizer was actually trained
                            lemma = '_'
                            if (self.model.train_lemmatizer and pred_lemmas is not None and 
                                hasattr(self, 'id_to_lemma') and hasattr(self, 'lemma_labels') and 
                                len(self.lemma_labels) > 0):
                                lemma_id = pred_lemmas[sent_idx][token_idx].item()
                                lemma = self.id_to_lemma.get(lemma_id, '_')
                            # If lemmatizer wasn't trained, lemma stays '_' and will fall back to vocabulary in _get_lemma
                            
                            # Parsing predictions
                            head = 0
                            deprel = '_'
                            if pred_heads is not None and self.config.parse:
                                # Head is predicted at token level, need to map to word level
                                head_token_idx = pred_heads[sent_idx][token_idx].item()
                                # Map head token index back to word index
                                head_word_idx = None
                                if head_token_idx < len(word_ids):
                                    head_word_id = word_ids[head_token_idx]
                                    head_word_idx = ctx_to_orig.get(head_word_id)
                                
                                if head_word_idx is not None:
                                    # Head is 1-based in CoNLL-U format
                                    head = head_word_idx + 1
                                else:
                                    head = 0  # Root
                                
                                # Deprel prediction
                                if pred_deprels is not None:
                                    deprel_id = pred_deprels[sent_idx][token_idx].item()
                                    deprel = self.id_to_deprel.get(deprel_id, '_')
                            
                            # Normalization prediction
                            norm = '_'
                            if pred_norms is not None and hasattr(self, 'id_to_norm'):
                                norm_id = pred_norms[sent_idx][token_idx].item()
                                norm = self.id_to_norm.get(norm_id, '_')
                            
                            word_predictions[orig_word_idx] = (upos, xpos, feats, lemma, norm, head, deprel)
                            token_to_word[token_idx] = orig_word_idx
                
                # Create tagged tokens
                for word_idx, token in enumerate(sentence):
                    # Copy only form and id, not annotations
                    tagged_token = {
                        'id': token.get('id', word_idx + 1),
                        'form': token.get('form', ''),
                    }
                    form = token.get('form', '')
                    existing_upos = token.get('upos', '_')
                    existing_xpos = token.get('xpos', '_')
                    existing_feats = token.get('feats', '_')
                    existing_lemma = token.get('lemma', '_')
                    
                    # Respect existing annotations if configured
                    if self.config.respect_existing:
                        # Vocabulary priority: check vocab first if enabled and word exists
                        vocab_upos = None
                        vocab_xpos = None
                        vocab_feats = None
                        vocab_lemma = None
                        
                        if self.config.use_vocabulary and self.config.vocab_priority:
                            # Check vocabulary first (for tuning to local corpus)
                            vocab_upos = self._predict_from_vocab(form, 'upos')
                            vocab_xpos = self._predict_from_vocab(form, 'xpos')
                            vocab_feats = self._predict_from_vocab(form, 'feats')
                            # For lemma, we need XPOS context, so we'll check after XPOS is determined
                        
                        if existing_upos != '_':
                            tagged_token['upos'] = existing_upos
                        elif vocab_upos and vocab_upos != '_':
                            tagged_token['upos'] = vocab_upos
                        else:
                            # Confidence-based blending: use vocab if model confidence is low
                            use_vocab_for_upos = False
                            if (self.config.use_vocabulary and 
                                word_idx in word_confidence and 
                                word_confidence[word_idx][0] < self.config.confidence_threshold):
                                # Model confidence is low, try vocabulary
                                vocab_upos_fallback = self._predict_from_vocab(form, 'upos')
                                if vocab_upos_fallback and vocab_upos_fallback != '_':
                                    tagged_token['upos'] = vocab_upos_fallback
                                    use_vocab_for_upos = True
                            
                            if not use_vocab_for_upos:
                                # Use model prediction (either high confidence or no vocab match)
                                if word_idx in word_predictions:
                                    tagged_token['upos'] = word_predictions[word_idx][0]
                                else:
                                    # Fallback to vocabulary or similarity
                                    tagged_token['upos'] = self._predict_from_vocab(form, 'upos')
                        
                        if existing_xpos != '_':
                            tagged_token['xpos'] = existing_xpos
                        elif vocab_xpos and vocab_xpos != '_':
                            tagged_token['xpos'] = vocab_xpos
                        else:
                            # Confidence-based blending: use vocab if model confidence is low
                            use_vocab_for_xpos = False
                            if (self.config.use_vocabulary and 
                                word_idx in word_confidence and 
                                word_confidence[word_idx][1] < self.config.confidence_threshold):
                                # Model confidence is low, try vocabulary
                                vocab_xpos_fallback = self._predict_from_vocab(form, 'xpos')
                                if vocab_xpos_fallback and vocab_xpos_fallback != '_':
                                    tagged_token['xpos'] = vocab_xpos_fallback
                                    use_vocab_for_xpos = True
                            
                            if not use_vocab_for_xpos:
                                # Use model prediction (either high confidence or no vocab match)
                                if word_idx in word_predictions:
                                    tagged_token['xpos'] = word_predictions[word_idx][1]
                                else:
                                    # Fallback to vocabulary or similarity
                                    tagged_token['xpos'] = self._predict_from_vocab(form, 'xpos')
                        
                        if existing_feats != '_':
                            tagged_token['feats'] = existing_feats
                        elif vocab_feats and vocab_feats != '_':
                            tagged_token['feats'] = vocab_feats
                        else:
                            # Check if FEATS were actually trained (not just random predictions)
                            # If FEATS only has '_' or 1-2 labels, FEATS weren't in training data
                            feats_were_trained = False
                            if hasattr(self, 'feats_labels'):
                                if len(self.feats_labels) > 2:
                                    feats_were_trained = True
                                elif len(self.feats_labels) == 2 and '_' not in self.feats_labels:
                                    feats_were_trained = True
                            
                            if not feats_were_trained:
                                # FEATS weren't in training data - use vocabulary fallback
                                vocab_feats_fallback = self._predict_from_vocab(form, 'feats')
                                if vocab_feats_fallback and vocab_feats_fallback != '_':
                                    tagged_token['feats'] = vocab_feats_fallback
                                else:
                                    tagged_token['feats'] = '_'
                            else:
                                # FEATS were trained - use confidence-based blending
                                use_vocab_for_feats = False
                                if (self.config.use_vocabulary and 
                                    word_idx in word_confidence and 
                                    word_confidence[word_idx][2] < self.config.confidence_threshold):
                                    # Model confidence is low, try vocabulary
                                    vocab_feats_fallback = self._predict_from_vocab(form, 'feats')
                                    if vocab_feats_fallback and vocab_feats_fallback != '_':
                                        tagged_token['feats'] = vocab_feats_fallback
                                        use_vocab_for_feats = True
                                
                                if not use_vocab_for_feats:
                                    # Use model prediction (either high confidence or no vocab match)
                                    if word_idx in word_predictions:
                                        tagged_token['feats'] = word_predictions[word_idx][2]
                                    else:
                                        tagged_token['feats'] = self._predict_from_vocab(form, 'feats')
                        
                        if existing_lemma != '_':
                            tagged_token['lemma'] = existing_lemma
                        else:
                            # Use lemma_method to determine priority
                            xpos = tagged_token.get('xpos', '_')
                            upos = tagged_token.get('upos', '_')
                            # Get normalized form if available (lemma should be for normalized form)
                            # Check if norm_form was already set in tagged_token (from model prediction)
                            norm_form = tagged_token.get('norm_form')
                            if not norm_form or norm_form == '_':
                                # Check token for preprocessing normalization
                                norm_form = token.get('norm_form') if 'norm_form' in token else None
                                if norm_form == '_':
                                    norm_form = None
                            # Get lemma confidence for confidence-based blending
                            lemma_conf = word_confidence.get(word_idx, (0, 0, 0, 0))[3] if word_idx in word_confidence else None
                            tagged_token['lemma'] = self._get_lemma(form, word_idx, word_predictions, xpos=xpos, upos=upos, norm_form=norm_form, lemma_confidence=lemma_conf)
                        
                        # Copy head/deprel if they exist and we're respecting existing
                        existing_head = token.get('head', 0)
                        existing_deprel = token.get('deprel', '_')
                        if self.config.parse and existing_head != 0 and existing_deprel != '_':
                            tagged_token['head'] = existing_head
                            tagged_token['deprel'] = existing_deprel
                        elif not self.config.parse:
                            # If not parsing, clear head/deprel
                            tagged_token['head'] = '_'
                            tagged_token['deprel'] = '_'
                        
                        # Add expan from vocabulary entry that matches the selected XPOS/UPOS
                        # Only use expan from the same entry that was used for tagging
                        expan = None
                        xpos = tagged_token.get('xpos', '_')
                        upos = tagged_token.get('upos', '_')
                        entry = self.vocab.get(form) or self.vocab.get(form.lower())
                        
                        if entry:
                            best_entry_for_expan = None
                            if isinstance(entry, list):
                                # Find the entry that matches the selected XPOS/UPOS
                                use_xpos = self.config.use_xpos_for_tagging
                                best_count = 0
                                
                                if use_xpos:
                                    # XPOS-first: try XPOS match, then UPOS
                                    for analysis in entry:
                                        if xpos and xpos != '_' and analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry_for_expan = analysis
                                    if not best_entry_for_expan and upos and upos != '_':
                                        for analysis in entry:
                                            if analysis.get('upos') == upos:
                                                count = analysis.get('count', 0)
                                                if count > best_count:
                                                    best_count = count
                                                    best_entry_for_expan = analysis
                                else:
                                    # UPOS-first: try UPOS match, then XPOS
                                    for analysis in entry:
                                        if upos and upos != '_' and analysis.get('upos') == upos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry_for_expan = analysis
                                    if not best_entry_for_expan and xpos and xpos != '_':
                                        for analysis in entry:
                                            if analysis.get('xpos') == xpos:
                                                count = analysis.get('count', 0)
                                                if count > best_count:
                                                    best_count = count
                                                    best_entry_for_expan = analysis
                                
                                if not best_entry_for_expan:
                                    # Fallback to most frequent
                                    best_entry_for_expan = max(entry, key=lambda a: a.get('count', 0))
                            elif isinstance(entry, dict):
                                best_entry_for_expan = entry
                            
                            # Only use expan from the entry that was selected for tagging
                            if best_entry_for_expan and 'expan' in best_entry_for_expan and best_entry_for_expan.get('expan') and best_entry_for_expan.get('expan') != '_':
                                expan = best_entry_for_expan.get('expan')
                        
                        if expan:
                            tagged_token['expan'] = expan
                    else:
                        # Use model predictions (ignore existing annotations, but respect vocab if priority enabled)
                        if word_idx in word_predictions:
                            # Check vocabulary first if vocab_priority enabled
                            if self.config.use_vocabulary and self.config.vocab_priority:
                                vocab_upos = self._predict_from_vocab(form, 'upos')
                                vocab_xpos = self._predict_from_vocab(form, 'xpos')
                                vocab_feats = self._predict_from_vocab(form, 'feats')
                                
                                tagged_token['upos'] = vocab_upos if vocab_upos != '_' else word_predictions[word_idx][0]
                                tagged_token['xpos'] = vocab_xpos if vocab_xpos != '_' else word_predictions[word_idx][1]
                                tagged_token['feats'] = vocab_feats if vocab_feats != '_' else word_predictions[word_idx][2]
                                
                                # Get normalized form first (before lemmatization)
                                norm_form = None
                                if len(word_predictions[word_idx]) > 4:
                                    norm_pred = word_predictions[word_idx][4]
                                    if norm_pred and norm_pred != '_':
                                        tagged_token['norm_form'] = norm_pred
                                        norm_form = norm_pred
                                elif 'norm_form' in token:
                                    norm_form_val = token.get('norm_form', '_')
                                    if norm_form_val and norm_form_val != '_':
                                        tagged_token['norm_form'] = norm_form_val
                                        norm_form = norm_form_val
                                
                                # Lemma: use lemma_method to determine priority (use normalized form if available)
                                xpos = tagged_token.get('xpos', '_')
                                upos = tagged_token.get('upos', '_')
                                # Get lemma confidence for confidence-based blending
                                lemma_conf = word_confidence.get(word_idx, (0, 0, 0, 0))[3] if word_idx in word_confidence else None
                                tagged_token['lemma'] = self._get_lemma(form, word_idx, word_predictions, xpos=xpos, upos=upos, norm_form=norm_form, lemma_confidence=lemma_conf)
                            else:
                                # Normal mode: model predictions first, vocab as fallback
                                tagged_token['upos'] = word_predictions[word_idx][0]
                                tagged_token['xpos'] = word_predictions[word_idx][1]
                                tagged_token['feats'] = word_predictions[word_idx][2]
                                
                                # Get normalized form first (before lemmatization)
                                norm_form = None
                                if len(word_predictions[word_idx]) > 4:
                                    norm_pred = word_predictions[word_idx][4]
                                    if norm_pred and norm_pred != '_':
                                        tagged_token['norm_form'] = norm_pred
                                        norm_form = norm_pred
                                elif 'norm_form' in token:
                                    norm_form_val = token.get('norm_form', '_')
                                    if norm_form_val and norm_form_val != '_':
                                        tagged_token['norm_form'] = norm_form_val
                                        norm_form = norm_form_val
                                
                                # Lemma: use lemma_method to determine priority (use normalized form if available)
                                xpos = tagged_token.get('xpos', '_')
                                upos = tagged_token.get('upos', '_')
                                # Get lemma confidence for confidence-based blending
                            lemma_conf = word_confidence.get(word_idx, (0, 0, 0, 0))[3] if word_idx in word_confidence else None
                            tagged_token['lemma'] = self._get_lemma(form, word_idx, word_predictions, xpos=xpos, upos=upos, norm_form=norm_form, lemma_confidence=lemma_conf)
                            
                            # Parsing
                            if self.config.parse and len(word_predictions[word_idx]) > 5:
                                tagged_token['head'] = word_predictions[word_idx][5]
                                tagged_token['deprel'] = word_predictions[word_idx][6]
                            else:
                                # Clear head and deprel if not parsing
                                tagged_token['head'] = '_'
                                tagged_token['deprel'] = '_'
                        else:
                            # Fallback to vocabulary
                            tagged_token['upos'] = self._predict_from_vocab(form, 'upos')
                            tagged_token['xpos'] = self._predict_from_vocab(form, 'xpos')
                            tagged_token['feats'] = self._predict_from_vocab(form, 'feats')
                            # Use lemma_method to determine priority (fallback to vocab since no BERT predictions)
                            xpos = tagged_token.get('xpos', '_')
                            upos = tagged_token.get('upos', '_')
                            # Get normalized form if available (lemma should be for normalized form)
                            norm_form = None
                            if 'norm_form' in token:
                                norm_form_val = token.get('norm_form', '_')
                                if norm_form_val and norm_form_val != '_':
                                    norm_form = norm_form_val
                            # Get lemma confidence for confidence-based blending
                            lemma_conf = word_confidence.get(word_idx, (0, 0, 0, 0))[3] if word_idx in word_confidence else None
                            tagged_token['lemma'] = self._get_lemma(form, word_idx, word_predictions, xpos=xpos, upos=upos, norm_form=norm_form, lemma_confidence=lemma_conf)
                            tagged_token['head'] = '_'
                            tagged_token['deprel'] = '_'
                        
                        # Add expan from vocabulary entry that matches the selected XPOS/UPOS
                        # Only use expan from the same entry that was used for tagging
                        expan = None
                        xpos = tagged_token.get('xpos', '_')
                        upos = tagged_token.get('upos', '_')
                        entry = self.vocab.get(form) or self.vocab.get(form.lower())
                        
                        if entry:
                            best_entry_for_expan = None
                            if isinstance(entry, list):
                                # Find the entry that matches the selected XPOS/UPOS
                                use_xpos = self.config.use_xpos_for_tagging
                                best_count = 0
                                
                                if use_xpos:
                                    # XPOS-first: try XPOS match, then UPOS
                                    for analysis in entry:
                                        if xpos and xpos != '_' and analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry_for_expan = analysis
                                    if not best_entry_for_expan and upos and upos != '_':
                                        for analysis in entry:
                                            if analysis.get('upos') == upos:
                                                count = analysis.get('count', 0)
                                                if count > best_count:
                                                    best_count = count
                                                    best_entry_for_expan = analysis
                                else:
                                    # UPOS-first: try UPOS match, then XPOS
                                    for analysis in entry:
                                        if upos and upos != '_' and analysis.get('upos') == upos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry_for_expan = analysis
                                    if not best_entry_for_expan and xpos and xpos != '_':
                                        for analysis in entry:
                                            if analysis.get('xpos') == xpos:
                                                count = analysis.get('count', 0)
                                                if count > best_count:
                                                    best_count = count
                                                    best_entry_for_expan = analysis
                                
                                if not best_entry_for_expan:
                                    # Fallback to most frequent
                                    best_entry_for_expan = max(entry, key=lambda a: a.get('count', 0))
                            elif isinstance(entry, dict):
                                best_entry_for_expan = entry
                            
                            # Only use expan from the entry that was selected for tagging
                            if best_entry_for_expan and 'expan' in best_entry_for_expan and best_entry_for_expan.get('expan') and best_entry_for_expan.get('expan') != '_':
                                expan = best_entry_for_expan.get('expan')
                        
                        if expan:
                            tagged_token['expan'] = expan
                    
                    tagged_sentence.append(tagged_token)
                
                tagged_sentences.append(tagged_sentence)
        
        # Calculate final statistics (for model-based tagging)
        total_tokens = sum(len(s) for s in tagged_sentences)
        elapsed_time = time.time() - start_time
        tok_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        print(f"Tagging complete: {len(tagged_sentences)} sentences, {total_tokens} tokens in {elapsed_time:.2f}s ({tok_per_sec:.1f} tok/s)", file=sys.stderr)
        
        # Write output
        if output_file:
            self.write_output(tagged_sentences, output_file, format)
        
        return tagged_sentences
    
    def _preprocess_sentences(self, sentences: List[List[Dict]]) -> List[List[Dict]]:
        """
        Preprocess sentences for historic document processing:
        - Normalize orthographic variants
        - Split contractions
        
        Args:
            sentences: List of sentences (each sentence is a list of token dicts)
        
        Returns:
            Preprocessed sentences with original and normalized forms stored
        """
        preprocessed = []
        
        for sentence in sentences:
            preprocessed_sentence = []
            
            # Preserve original text from first token if present
            original_text = None
            for token in sentence:
                if '_original_text' in token:
                    original_text = token.get('_original_text')
                    break
            
            for token in sentence:
                form = token.get('form', '')
                if not form:
                    preprocessed_sentence.append(token)
                    continue
                
                # Store original form
                new_token = token.copy()
                new_token['orig_form'] = form
                # Preserve _original_text if present (will be in first token)
                if '_original_text' in token:
                    new_token['_original_text'] = token['_original_text']
                normalized_form = form
                split_forms = None
                
                # Step 1: Normalize orthographic variants
                if self.config.normalize:
                    normalized = None
                    
                    # Note: If model has a trained normalizer, it will be used during tagging phase
                    # Here in preprocessing, we only do vocabulary-based normalization
                    # This allows normalization before tagging (useful for tag-on-normalized mode)
                    if self.vocab:
                        normalized = normalize_word(
                            form, 
                            self.vocab, 
                            conservative=self.config.conservative_normalization,
                            similarity_threshold=self.config.similarity_threshold,
                            inflection_suffixes=self.inflection_suffixes
                        )
                    
                    if normalized:
                        normalized_form = normalized
                        new_token['norm_form'] = normalized_form
                
                # Step 2: Split contractions
                if self.config.split_contractions and self.vocab:
                    split_result = split_contraction(
                        normalized_form, 
                        self.vocab, 
                        aggressive=self.config.aggressive_contraction_splitting,
                        language=self.config.language
                    )
                    if split_result:
                        split_forms = split_result
                        new_token['split_forms'] = split_forms
                
                # Determine which form to use for tagging
                if self.config.tag_on_normalized and 'norm_form' in new_token:
                    new_token['form'] = new_token['norm_form']
                else:
                    new_token['form'] = form
                
                # If contraction was split, we need to expand the token into multiple tokens
                # This is similar to MWT handling in UD
                if split_forms and len(split_forms) > 1:
                    # Create multiple tokens for the contraction
                    for i, split_form in enumerate(split_forms):
                        split_token = new_token.copy()
                        split_token['form'] = split_form
                        split_token['id'] = f"{token.get('id', len(preprocessed_sentence) + 1)}.{i+1}"
                        split_token['is_contraction_part'] = True
                        split_token['contraction_id'] = token.get('id', len(preprocessed_sentence) + 1)
                        split_token['contraction_part'] = i + 1
                        # Only preserve _original_text in first split token
                        if i == 0 and '_original_text' in new_token:
                            split_token['_original_text'] = new_token['_original_text']
                        elif '_original_text' in split_token:
                            del split_token['_original_text']
                        preprocessed_sentence.append(split_token)
                else:
                    preprocessed_sentence.append(new_token)
            
            # Ensure original text is preserved in first token of preprocessed sentence
            if original_text and preprocessed_sentence:
                if '_original_text' not in preprocessed_sentence[0]:
                    preprocessed_sentence[0]['_original_text'] = original_text
            
            preprocessed.append(preprocessed_sentence)
        
        return preprocessed
    
    def _get_lemma(self, form: str, word_idx: int, word_predictions: Dict, xpos: str = None, upos: str = None, norm_form: str = None, lemma_confidence: float = None) -> str:
        """
        Get lemma based on lemma_method configuration.
        
        Args:
            form: Word form
            word_idx: Word index in sentence
            word_predictions: Dictionary of word predictions from model
            xpos: XPOS tag (optional, for context-aware lookup)
            upos: UPOS tag (optional, for context-aware lookup)
            norm_form: Normalized form (optional, if normalization was applied)
        
        Returns:
            Lemma string
        """
        lemma_method = self.config.lemma_method
        
        # If normalization is applied, use normalized form for lemmatization
        # The lemma corresponds to the Reg form, not the original form
        lemma_form = norm_form if norm_form and norm_form != '_' else form
        
        if self.config.debug:
            print(f"[DEBUG LEMMA] Form: '{form}', Lemma form: '{lemma_form}', XPOS: '{xpos}', UPOS: '{upos}', Norm form: '{norm_form}'", file=sys.stderr)
        
        # Get BERT prediction if available (only if lemmatizer was actually trained)
        bert_lemma = None
        if (self.model.train_lemmatizer and 
            hasattr(self, 'lemma_labels') and len(self.lemma_labels) > 0 and
            word_idx in word_predictions and len(word_predictions[word_idx]) > 3):
            bert_lemma = word_predictions[word_idx][3]
            if bert_lemma == '_':
                bert_lemma = None
            if self.config.debug:
                print(f"[DEBUG LEMMA] BERT lemma: '{bert_lemma}'", file=sys.stderr)
        
        # Get vocabulary prediction (use normalized form if available)
        vocab_lemma = None
        vocab_lemma_original = None  # Store original capitalization from vocab
        if self.config.use_vocabulary:
            # Get FEATS if available from word_predictions
            feats = None
            if word_idx in word_predictions and len(word_predictions[word_idx]) > 2:
                feats = word_predictions[word_idx][2]
                if feats == '_':
                    feats = None
            
            # Check if word is in vocabulary (exact match) - this determines if it's OOV
            word_is_in_vocab = False
            if hasattr(self, 'vocab') and self.vocab:
                form_lower = lemma_form.lower()
                # Check both exact case and lowercase
                word_is_in_vocab = (lemma_form in self.vocab or form_lower in self.vocab or
                                   (xpos and f"{lemma_form}:{xpos}" in self.vocab) or
                                   (xpos and f"{form_lower}:{xpos}" in self.vocab))
            
            vocab_lemma = self._predict_from_vocab(lemma_form, 'lemma', xpos=xpos, upos=upos, feats=feats, debug=self.config.debug)
            if vocab_lemma == '_':
                vocab_lemma = None
            else:
                # Store original capitalization from vocabulary (before any modifications)
                vocab_lemma_original = vocab_lemma
            if self.config.debug:
                print(f"[DEBUG LEMMA] Vocab lemma: '{vocab_lemma}' (word_in_vocab={word_is_in_vocab})", file=sys.stderr)
        
        # Apply lemma_method priority
        # Default: Use BERT lemmatization when lemmatizer is trained and confidence is high
        # Fallback to vocabulary when:
        #   - Lemmatizer wasn't trained (has_lemmatizer: false)
        #   - BERT confidence is low (confidence-based blending)
        #   - Vocabulary is explicitly prioritized (vocab_priority flag)
        final_lemma = None
        
        # Check if lemmatizer was actually trained (from metadata)
        lemmatizer_trained = (self.model.train_lemmatizer and 
                             hasattr(self, 'lemma_labels') and 
                             len(self.lemma_labels) > 0)
        
        # Use confidence-based blending: if BERT confidence is low, prefer vocabulary or fallback
        use_bert_lemma = False
        if lemmatizer_trained and bert_lemma:
            if lemma_confidence is not None:
                if lemma_confidence >= self.config.confidence_threshold:
                    # BERT confidence is high enough - use it
                    use_bert_lemma = True
                else:
                    # BERT confidence is too low - don't use it
                    if self.config.debug:
                        print(f"[DEBUG LEMMA] BERT confidence ({lemma_confidence:.3f}) below threshold ({self.config.confidence_threshold}), rejecting BERT prediction", file=sys.stderr)
            else:
                # No confidence score available - use BERT if available
                use_bert_lemma = True
        
        # Priority: vocab (if available and exact match), then pattern-based (for OOV or when vocab returns form), then BERT (if confidence is acceptable), then fallback
        # For OOV words (not in vocabulary), pattern-based is more reliable than BERT
        # Also: if vocab returns the form itself (no useful lemma), try pattern-based
        use_vocab_lemma = False
        if vocab_lemma and word_is_in_vocab:
            # Check if vocab returned a useful lemma (not the form itself)
            lemma_form_lower = lemma_form.lower()
            if vocab_lemma.lower() != lemma_form_lower:
                # Vocab returned a different lemma - use it
                use_vocab_lemma = True
                final_lemma = vocab_lemma
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (from vocab, exact match)", file=sys.stderr)
            else:
                # Vocab returned the form itself - not useful, try lexicon or pattern-based
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Vocab returned form itself '{vocab_lemma}', checking lexicon or pattern-based", file=sys.stderr)

        # Check lexicon as fallback when vocab doesn't provide a useful lemma
        # Use context-aware lookup (XPOS/UPOS/FEATS) to match the right lexicon entry
        if not use_vocab_lemma and self.lexicon:
            if self.config.debug:
                print(f"[DEBUG LEMMA] Checking lexicon for '{lemma_form}' (vocab result not usable)", file=sys.stderr)
            # Get FEATS if available from word_predictions
            feats_for_lexicon = None
            if word_idx in word_predictions and len(word_predictions[word_idx]) > 2:
                feats_for_lexicon = word_predictions[word_idx][2]
                if feats_for_lexicon == '_':
                    feats_for_lexicon = None
            if self.config.debug:
                print(f"[DEBUG LEMMA] Looking up lexicon with xpos='{xpos}', upos='{upos}', feats='{feats_for_lexicon}'", file=sys.stderr)
            
            # Use context-aware matching directly against lexicon entries
            lemma_form_lower = lemma_form.lower()
            lexicon_entry = self.lexicon.get(lemma_form) or self.lexicon.get(lemma_form_lower)
            if lexicon_entry:
                # Use the same get_field_from_entry helper from _predict_from_vocab
                # We'll create a temporary version that works with lexicon
                def get_field_from_lexicon_entry(entry, field, xpos=None, upos=None, feats=None):
                            """Helper to extract field from lexicon entry with context matching."""
                            if isinstance(entry, list):
                                matches = []
                                use_xpos = self.config.use_xpos_for_tagging
                                
                                if use_xpos:
                                    if xpos and xpos != '_':
                                        for analysis in entry:
                                            if analysis.get('xpos') == xpos:
                                                matches.append((analysis.get('count', 0), analysis))
                                    if not matches and upos and upos != '_':
                                        for analysis in entry:
                                            if analysis.get('upos') == upos:
                                                analysis_feats = analysis.get('feats', '_')
                                                if feats and feats != '_' and analysis_feats and analysis_feats != '_':
                                                    lexicon_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in analysis_feats.split('|') if '=' in f}
                                                    actual_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in feats.split('|') if '=' in f}
                                                    feats_match = all(actual_feat_dict.get(k) == v for k, v in lexicon_feat_dict.items())
                                                    if feats_match:
                                                        matches.append((analysis.get('count', 0), analysis))
                                                    else:
                                                        matches.append((analysis.get('count', 0), analysis))
                                else:
                                    if upos and upos != '_':
                                        for analysis in entry:
                                            if analysis.get('upos') == upos:
                                                analysis_feats = analysis.get('feats', '_')
                                                if feats and feats != '_' and analysis_feats and analysis_feats != '_':
                                                    lexicon_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in analysis_feats.split('|') if '=' in f}
                                                    actual_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in feats.split('|') if '=' in f}
                                                    feats_match = all(actual_feat_dict.get(k) == v for k, v in lexicon_feat_dict.items())
                                                    if feats_match:
                                                        matches.append((analysis.get('count', 0), analysis))
                                                else:
                                                    matches.append((analysis.get('count', 0), analysis))
                                    if not matches and xpos and xpos != '_':
                                        for analysis in entry:
                                            if analysis.get('xpos') == xpos:
                                                matches.append((analysis.get('count', 0), analysis))
                                
                                if matches:
                                    matches.sort(key=lambda x: x[0], reverse=True)
                                    return matches[0][1].get(field, '_')
                                elif entry:
                                    # Fallback to first entry
                                    return entry[0].get(field, '_')
                                return '_'
                            elif isinstance(entry, dict):
                                return entry.get(field, '_')
                            return '_'
                        
                lexicon_lemma = get_field_from_lexicon_entry(lexicon_entry, 'lemma', xpos=xpos, upos=upos, feats=feats_for_lexicon)
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Lexicon lookup result: '{lexicon_lemma}'", file=sys.stderr)
                
                if lexicon_lemma and lexicon_lemma != '_' and lexicon_lemma.lower() != lemma_form.lower():
                    # Lexicon has a useful lemma - use it
                    use_vocab_lemma = True
                    final_lemma = lexicon_lemma
                    if self.config.debug:
                        print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (from lexicon)", file=sys.stderr)
            else:
                if self.config.debug:
                    print(f"[DEBUG LEMMA] '{lemma_form}' not found in lexicon", file=sys.stderr)
        
        if not use_vocab_lemma:
            # Word is OOV (not in vocabulary) OR vocab_lemma came from similarity matching (less reliable)
            # For OOV words, pattern-based is more reliable than similarity matching or BERT
            # Vocabulary doesn't have the word (OOV) - try pattern-based lemmatization
            # IMPORTANT: For OOV words, pattern-based is more reliable than BERT
            pattern_lemma = None
            if self.config.use_vocabulary and xpos and xpos != '_':
                # Try pattern-based lemmatization for OOV words
                if hasattr(self, 'lemmatization_patterns') and self.lemmatization_patterns:
                    from flexipipe.lemmatizer import apply_lemmatization_patterns
                    pattern_lemma = apply_lemmatization_patterns(
                        lemma_form,
                        xpos,
                        self.lemmatization_patterns,
                        known_lemmas=self._get_known_lemmas(),
                        debug=self.config.debug
                    )
                    if pattern_lemma and pattern_lemma != '_':
                        final_lemma = pattern_lemma
                        if self.config.debug:
                            print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (from pattern-based, OOV word)", file=sys.stderr)
            
            # If pattern-based didn't work, be very cautious with BERT for OOV words
            # BERT lemmatizer often produces random results for OOV words
            # For OOV words with vocabulary available, prefer form over BERT (pattern-based is more reliable)
            if not pattern_lemma or pattern_lemma == '_':
                # For OOV words, disable BERT entirely if vocabulary is available
                # Pattern-based is more reliable for morphological patterns, and BERT often produces random results
                if self.config.use_vocabulary:
                    # Vocabulary available: don't use BERT for OOV words (too unreliable)
                    final_lemma = form
                    if self.config.debug:
                        print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (fallback to form, OOV word, pattern-based failed, BERT disabled for OOV when vocab available)", file=sys.stderr)
            else:
                    # No vocabulary: use BERT only if confidence is very high
                    oov_bert_threshold = 0.9  # Very high threshold for OOV words
                    use_bert_for_oov = (use_bert_lemma and bert_lemma and 
                                       (lemma_confidence is None or lemma_confidence >= oov_bert_threshold))
                    
                    if use_bert_for_oov:
                        # Use BERT only if confidence is very high for OOV words
                        final_lemma = bert_lemma
                        if self.config.debug:
                            print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (from BERT, confidence={lemma_confidence:.3f if lemma_confidence else 'N/A'}, OOV word, threshold={oov_bert_threshold})", file=sys.stderr)
                    else:
                        # No pattern-based or BERT available/acceptable - use form as-is
                        final_lemma = form
                        if self.config.debug:
                            reason = "pattern-based failed"
                            if not use_bert_lemma:
                                reason += ", BERT not available"
                            elif bert_lemma and lemma_confidence is not None:
                                reason += f", BERT confidence too low ({lemma_confidence:.3f} < {oov_bert_threshold})"
                            print(f"[DEBUG LEMMA] Selected: '{final_lemma}' (fallback to form, OOV word, {reason})", file=sys.stderr)
        
        # Preserve capitalization: prefer vocabulary's capitalization, then form's capitalization
        # Don't force everything to lowercase - preserve original capitalization when possible
        if final_lemma and len(final_lemma) > 0:
            # First, check if vocabulary provided capitalized lemma - use it as-is
            if vocab_lemma_original and vocab_lemma_original[0].isupper():
                # Vocabulary has capitalized lemma - use it (already in final_lemma)
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Using capitalization from vocabulary: '{final_lemma}'", file=sys.stderr)
            elif form and len(form) > 0 and form[0].isupper() and final_lemma[0].islower():
                # Vocabulary lemma is lowercase but form is capitalized - capitalize lemma
                final_lemma = final_lemma[0].upper() + final_lemma[1:]
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Capitalized lemma: '{final_lemma}' (preserving capitalization from form)", file=sys.stderr)
        
        return final_lemma
    
    def _predict_from_vocab(self, form: str, field: str, xpos: str = None, upos: str = None, feats: str = None, debug: bool = False) -> str:
        """Predict from vocabulary or similarity matching.
        
        For lemmatization, uses XPOS-aware lookup (like neotag):
        1. First try (form, XPOS) lookup
        2. Then try form-only lookup (with XPOS context if available)
        3. Finally use similarity matching with XPOS context
        4. If word is OOV (not in primary vocab), try lexicon as fallback
        
        Vocabulary entries can be:
        - Single object: {"upos": "NOUN", "lemma": "word"}
        - Array of objects: [{"upos": "NOUN", "lemma": "word1"}, {"upos": "VERB", "lemma": "word2"}]
          (for ambiguous words with multiple analyses)
        
        Case-sensitive lookup:
        - First tries exact case match (e.g., "Band" vs "band")
        - Falls back to lowercase match if exact case not found
        - This handles cases like German "Band" (noun, book volume) vs "band" (verb, past tense)
        
        Lexicon (OOV fallback):
        - Lexicon is only checked if word is not in primary vocab at all
        - This ensures corpus-specific vocabulary (with counts) takes priority over general lexicons
        
        Args:
            form: Word form
            field: Field to predict ('lemma', 'upos', 'xpos', 'feats')
            xpos: XPOS tag (optional, used for context-aware lemmatization)
            upos: UPOS tag (optional, used for context-aware lemmatization)
            feats: FEATS tag (optional, used for context-aware lemmatization)
        """
        form_lower = form.lower()
        use_xpos = self.config.use_xpos_for_tagging
        
        # Check if word is in primary vocab (to determine if it's OOV)
        # Check both self.vocab (merged vocab) and self.external_vocab (in case load_model wasn't called)
        vocab_to_check = self.vocab if self.vocab else self.external_vocab
        word_in_vocab = (form in vocab_to_check or form_lower in vocab_to_check or 
                        (field == 'lemma' and xpos and xpos != '_' and 
                         (f"{form}:{xpos}" in vocab_to_check or f"{form_lower}:{xpos}" in vocab_to_check)))
        
        def get_field_from_entry(entry, field, xpos=None, upos=None, feats=None, debug_entry=False):
            """Helper to extract field from vocabulary entry (single object or array).
            
            For ambiguous words (arrays), uses count/frequency to prefer most likely analysis.
            """
            if isinstance(entry, list):
                # Multiple analyses: try to find best match using context
                # Default: prioritize UPOS+FEATS (UD standard), fallback to XPOS if UPOS/FEATS not available
                # If use_xpos_for_tagging=True: prioritize XPOS, fallback to UPOS+FEATS
                matches = []
                
                if use_xpos:
                    # XPOS-first mode: try XPOS match first
                    if xpos and xpos != '_':
                        if debug_entry or debug:
                            print(f"[DEBUG LEMMA] Looking for XPOS match: '{xpos}'", file=sys.stderr)
                        for analysis in entry:
                            analysis_xpos = analysis.get('xpos')
                            if debug_entry or debug:
                                print(f"[DEBUG LEMMA]   Checking analysis: xpos='{analysis_xpos}', lemma='{analysis.get('lemma', '_')}', count={analysis.get('count', 0)}", file=sys.stderr)
                            if analysis_xpos == xpos:
                                count = analysis.get('count', 0)
                                matches.append((count, analysis))
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA]   MATCH found! xpos='{analysis_xpos}' == '{xpos}'", file=sys.stderr)
                
                    # Fallback to UPOS+FEATS if no XPOS match
                    # Fallback to UPOS+FEATS if no XPOS match
                    if not matches and upos and upos != '_':
                            print(f"[DEBUG LEMMA] No XPOS match, trying UPOS+FEATS: upos='{upos}', feats='{feats}'", file=sys.stderr)
                    for analysis in entry:
                            analysis_upos = analysis.get('upos', '_')
                            analysis_feats = analysis.get('feats', '_')
                            if analysis_upos == upos:
                                # If FEATS provided, check if lexicon FEATS are subset of actual FEATS
                                # (lexicon might have fewer features, which is fine)
                                if feats and feats != '_' and analysis_feats and analysis_feats != '_':
                                    # Check if all lexicon FEATS are present in actual FEATS
                                    lexicon_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in analysis_feats.split('|') if '=' in f}
                                    actual_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in feats.split('|') if '=' in f}
                                    # All lexicon features must be present in actual features with same values
                                    feats_match = all(actual_feat_dict.get(k) == v for k, v in lexicon_feat_dict.items())
                                    if feats_match:
                                        count = analysis.get('count', 0)
                                        matches.append((count, analysis))
                                    elif not feats or feats == '_':
                                        # No FEATS provided, just match on UPOS
                                        count = analysis.get('count', 0)
                                        matches.append((count, analysis))
                else:
                    # UPOS+FEATS-first mode (default): try UPOS+FEATS match first
                    if upos and upos != '_':
                        if debug_entry or debug:
                            print(f"[DEBUG LEMMA] Looking for UPOS+FEATS match: upos='{upos}', feats='{feats}'", file=sys.stderr)
                        for analysis in entry:
                            analysis_upos = analysis.get('upos', '_')
                            analysis_feats = analysis.get('feats', '_')
                            if analysis_upos == upos:
                                # If FEATS provided, check if lexicon FEATS are subset of actual FEATS
                                # (lexicon might have fewer features, which is fine)
                                if feats and feats != '_' and analysis_feats and analysis_feats != '_':
                                    # Check if all lexicon FEATS are present in actual FEATS
                                    lexicon_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in analysis_feats.split('|') if '=' in f}
                                    actual_feat_dict = {f.split('=')[0]: f.split('=')[1] for f in feats.split('|') if '=' in f}
                                    # All lexicon features must be present in actual features with same values
                                    feats_match = all(actual_feat_dict.get(k) == v for k, v in lexicon_feat_dict.items())
                                    if feats_match:
                                        count = analysis.get('count', 0)
                                        matches.append((count, analysis))
                                        if debug_entry or debug:
                                            print(f"[DEBUG LEMMA]   MATCH found! upos='{analysis_upos}', feats='{analysis_feats}' (subset match)", file=sys.stderr)
                                elif not feats or feats == '_':
                                    # No FEATS provided, just match on UPOS
                                    count = analysis.get('count', 0)
                                    matches.append((count, analysis))
                                    if debug_entry or debug:
                                        print(f"[DEBUG LEMMA]   MATCH found! upos='{analysis_upos}'", file=sys.stderr)
                    
                    # Fallback to XPOS if no UPOS+FEATS match and XPOS available
                    if not matches and xpos and xpos != '_':
                        if debug_entry or debug:
                            print(f"[DEBUG LEMMA] No UPOS+FEATS match, trying XPOS: '{xpos}'", file=sys.stderr)
                        for analysis in entry:
                            analysis_xpos = analysis.get('xpos')
                            if analysis_xpos == xpos:
                                count = analysis.get('count', 0)
                                matches.append((count, analysis))
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA]   MATCH found! xpos='{analysis_xpos}' == '{xpos}'", file=sys.stderr)
                
                if matches:
                    # Sort by count (descending) and return field from most frequent match
                    matches.sort(key=lambda x: x[0], reverse=True)
                    result = matches[0][1].get(field, '_')
                    if debug_entry or debug:
                        print(f"[DEBUG LEMMA] Returning from context match: '{result}' (from analysis with count={matches[0][0]})", file=sys.stderr)
                    return result
                elif (debug_entry or debug) and ((xpos and xpos != '_') or (upos and upos != '_')):
                    print(f"[DEBUG LEMMA] No context match found, falling through to fallback logic", file=sys.stderr)
                
                # No context match: return from most frequent analysis (sorted by count)
                # BUT: If looking for XPOS/UPOS/FEATS, only consider analyses that have that field
                # This prevents using entries without XPOS for XPOS prediction
                if entry:
                    # Filter entries to only those with the requested field (if field is xpos/upos/feats)
                    if field in ('xpos', 'upos', 'feats'):
                        filtered_entries = [a for a in entry if a.get(field) and a.get(field) != '_']
                        if filtered_entries:
                            # Use filtered entries (only those with the field)
                            sorted_entries = sorted(filtered_entries, key=lambda x: x.get('count', 0), reverse=True)
                            return sorted_entries[0].get(field, '_')
                        else:
                            # No entries have this field - return '_' to indicate not found
                            return '_'
                    elif field == 'lemma':
                        # For lemmatization: if context was provided but didn't match,
                        # we should NOT fall back to wrong entries - that defeats the purpose
                        # Return '_' to trigger pattern-based fallback
                        if use_xpos:
                            if xpos and xpos != '_':
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA] No exact XPOS match for '{xpos}', but XPOS provided - returning '_' to trigger pattern-based fallback", file=sys.stderr)
                                return '_'
                            elif upos and upos != '_':
                                # UPOS provided but no XPOS match - return '_' to trigger pattern-based fallback
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA] No exact UPOS+FEATS match, but UPOS provided - returning '_' to trigger pattern-based fallback", file=sys.stderr)
                                return '_'
                        else:
                            if upos and upos != '_':
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA] No exact UPOS+FEATS match, but UPOS provided - returning '_' to trigger pattern-based fallback", file=sys.stderr)
                                return '_'
                            elif xpos and xpos != '_':
                                # XPOS provided but no UPOS+FEATS match - return '_' to trigger pattern-based fallback
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA] No exact XPOS match for '{xpos}', but XPOS provided - returning '_' to trigger pattern-based fallback", file=sys.stderr)
                                return '_'
                        # No context provided: use all entries (most frequent)
                        sorted_entries = sorted(entry, key=lambda x: x.get('count', 0), reverse=True)
                        return sorted_entries[0].get(field, '_')
                    else:
                        # For other fields, use all entries
                        sorted_entries = sorted(entry, key=lambda x: x.get('count', 0), reverse=True)
                        return sorted_entries[0].get(field, '_')
                return '_'
            elif isinstance(entry, dict):
                # For single dict entry: if looking for xpos/upos/feats, check it exists
                if field in ('xpos', 'upos', 'feats'):
                    field_value = entry.get(field, '_')
                    if not field_value or field_value == '_':
                        return '_'  # Entry doesn't have this field
                return entry.get(field, '_')
            return '_'
        
        # For lemmatization, try XPOS-aware lookup first (like neotag)
        if field == 'lemma' and xpos and xpos != '_':
            # Try exact case first
            key = f"{form}:{xpos}"
            if key in vocab_to_check:
                if debug:
                    print(f"[DEBUG LEMMA] Found form:xpos key: '{key}'", file=sys.stderr)
                result = get_field_from_entry(vocab_to_check[key], field, xpos, upos, debug_entry=debug)
                if debug:
                    print(f"[DEBUG LEMMA] Result from '{key}': '{result}'", file=sys.stderr)
                return result
            elif debug:
                print(f"[DEBUG LEMMA] Form:xpos key '{key}' not found in vocab", file=sys.stderr)
            
            # Try lowercase
            key = f"{form_lower}:{xpos}"
            if key in vocab_to_check:
                if debug:
                    print(f"[DEBUG LEMMA] Found form:xpos key (lowercase): '{key}'", file=sys.stderr)
                result = get_field_from_entry(vocab_to_check[key], field, xpos, upos, debug_entry=debug)
                if debug:
                    print(f"[DEBUG LEMMA] Result from '{key}': '{result}'", file=sys.stderr)
                return result
            elif debug:
                print(f"[DEBUG LEMMA] Form:xpos key '{key}' (lowercase) not found in vocab", file=sys.stderr)
        
        # Standard form-only lookup: try exact case first, then lowercase
        # This is important for case-sensitive distinctions:
        # - German: "Band" (noun, book volume) vs "band" (verb, past tense of binden)
        # - English: "Apple" (proper noun, company) vs "apple" (common noun, fruit)
        
        # Try exact case match first
        if form in vocab_to_check:
            if debug:
                print(f"[DEBUG LEMMA] Found form in vocab: '{form}'", file=sys.stderr)
                entry = vocab_to_check[form]
                if isinstance(entry, list):
                    print(f"[DEBUG LEMMA] Entry has {len(entry)} analyses:", file=sys.stderr)
                    for i, analysis in enumerate(entry):
                        print(f"[DEBUG LEMMA]   [{i}] xpos={analysis.get('xpos', '_')}, upos={analysis.get('upos', '_')}, lemma={analysis.get('lemma', '_')}, count={analysis.get('count', 0)}", file=sys.stderr)
                else:
                    print(f"[DEBUG LEMMA] Entry: xpos={entry.get('xpos', '_')}, upos={entry.get('upos', '_')}, lemma={entry.get('lemma', '_')}", file=sys.stderr)
            vocab_result = get_field_from_entry(vocab_to_check[form], field, xpos, upos, debug_entry=debug)
            if debug:
                print(f"[DEBUG LEMMA] Result from form '{form}': '{vocab_result}'", file=sys.stderr)
            # For lemmatization: if vocab returns '_', try pattern-based as fallback
            # BUT: if vocab returns the form itself, it might still be correct (e.g., "era" noun -> "era")
            # Only reject if it's '_' or if we got it from a fallback (no XPOS match)
            if field == 'lemma' and vocab_result == '_':
                # Vocab didn't provide a lemma, try pattern-based
                if xpos and xpos != '_':
                    if debug:
                        print(f"[DEBUG LEMMA] Vocab returned '_', trying pattern-based", file=sys.stderr)
                    pattern_lemma = self._apply_lemmatization_patterns(form, xpos, debug=debug)
                    if pattern_lemma and pattern_lemma != '_':
                        if debug:
                            print(f"[DEBUG LEMMA] Pattern-based lemma: '{pattern_lemma}'", file=sys.stderr)
                        return pattern_lemma
            # If vocab returned the form itself, it's still valid if it came from an XPOS match
            # Only try pattern-based if vocab returned '_'
            return vocab_result
        
        # Fall back to lowercase match
        if form_lower in vocab_to_check:
            if debug:
                print(f"[DEBUG LEMMA] Found form_lower in vocab: '{form_lower}'", file=sys.stderr)
                entry = vocab_to_check[form_lower]
                if isinstance(entry, list):
                    print(f"[DEBUG LEMMA] Entry has {len(entry)} analyses:", file=sys.stderr)
                    for i, analysis in enumerate(entry):
                        print(f"[DEBUG LEMMA]   [{i}] xpos={analysis.get('xpos', '_')}, upos={analysis.get('upos', '_')}, lemma={analysis.get('lemma', '_')}, count={analysis.get('count', 0)}", file=sys.stderr)
                else:
                    print(f"[DEBUG LEMMA] Entry: xpos={entry.get('xpos', '_')}, upos={entry.get('upos', '_')}, lemma={entry.get('lemma', '_')}", file=sys.stderr)
            vocab_result = get_field_from_entry(vocab_to_check[form_lower], field, xpos, upos, debug_entry=debug)
            if debug:
                print(f"[DEBUG LEMMA] Result from form_lower '{form_lower}': '{vocab_result}'", file=sys.stderr)
            # For lemmatization: if vocab returns '_', try pattern-based as fallback
            # BUT: if vocab returns the form itself, it might still be correct (e.g., "era" noun -> "era")
            # Only reject if it's '_' or if we got it from a fallback (no XPOS match)
            if field == 'lemma' and vocab_result == '_':
                # Vocab didn't provide a lemma, try pattern-based
                if xpos and xpos != '_':
                    if debug:
                        print(f"[DEBUG LEMMA] Vocab returned '_', trying pattern-based", file=sys.stderr)
                    pattern_lemma = self._apply_lemmatization_patterns(form, xpos, debug=debug)
                    if pattern_lemma and pattern_lemma != '_':
                        if debug:
                            print(f"[DEBUG LEMMA] Pattern-based lemma: '{pattern_lemma}'", file=sys.stderr)
                        return pattern_lemma
            # If vocab returned the form itself, it's still valid if it came from an XPOS match
            # Only try pattern-based if vocab returned '_'
            return vocab_result
        
        # Pattern-based similarity lemmatization for OOV words (TreeTagger/Neotag style)
        # This should be tried BEFORE similarity matching, as it's more reliable for morphological patterns
        # BUT: Skip pattern-based lemmatization for non-inflecting POS tags (prepositions, conjunctions, etc.)
        if field == 'lemma' and xpos and xpos != '_':
            # Skip lemmatization for non-inflecting POS tags
            # These typically don't have morphological patterns and should keep their form as lemma
            non_inflecting_prefixes = ['SP', 'CC', 'CS', 'I', 'F', 'Z']  # Prepositions, conjunctions, interjections, punctuation, numbers
            if any(xpos.startswith(prefix) for prefix in non_inflecting_prefixes):
                # For non-inflecting POS, lemma should be the form itself
                if debug:
                    print(f"[DEBUG LEMMA] XPOS '{xpos}' is non-inflecting, returning form_lower: '{form_lower}'", file=sys.stderr)
                return form_lower
            if debug:
                print(f"[DEBUG LEMMA] Trying pattern-based lemmatization for form '{form}' with XPOS '{xpos}'", file=sys.stderr)
            lemma = self._apply_lemmatization_patterns(form, xpos, debug=debug)
            if lemma and lemma != '_':
                if debug:
                    print(f"[DEBUG LEMMA] Pattern-based lemma: '{lemma}'", file=sys.stderr)
                return lemma
            elif debug:
                print(f"[DEBUG LEMMA] Pattern-based lemmatization returned '{lemma}' (no match)", file=sys.stderr)
        
        # Try similarity matching (fallback for cases where pattern-based doesn't work)
        # BUT: Skip similarity matching for non-inflecting POS tags
        if field == 'lemma' and xpos and xpos != '_':
            non_inflecting_prefixes = ['SP', 'CC', 'CS', 'I', 'F', 'Z']
            if any(xpos.startswith(prefix) for prefix in non_inflecting_prefixes):
                # For non-inflecting POS, lemma should be the form itself
                return form_lower
        
        # Try similarity matching in primary vocab (fallback for cases where pattern-based doesn't work)
        if debug:
            print(f"[DEBUG LEMMA] Trying similarity matching for form '{form}'", file=sys.stderr)
        similar = find_similar_words(form, vocab_to_check, self.config.similarity_threshold)
        if similar:
            best_match = similar[0][0]
            if debug:
                print(f"[DEBUG LEMMA] Found similar words (showing top 5):", file=sys.stderr)
                for i, (match_word, score) in enumerate(similar[:5]):
                    print(f"[DEBUG LEMMA]   [{i}] '{match_word}' (similarity: {score:.3f})", file=sys.stderr)
            similar_entry = vocab_to_check[best_match]
            
            # Handle array format
            if isinstance(similar_entry, list):
                result = similar_entry[0].get(field, '_') if similar_entry else '_'
                if debug:
                    print(f"[DEBUG LEMMA] Similar entry '{best_match}' has {len(similar_entry)} analyses, using first: '{result}'", file=sys.stderr)
            else:
                result = similar_entry.get(field, '_')
                if debug:
                    print(f"[DEBUG LEMMA] Similar entry '{best_match}' result: '{result}'", file=sys.stderr)
            
            # For lemmatization of OOV words via similarity matching:
            # - If the similar word has a reg field, use its lemma directly (don't transform)
            # - If the similar word has an expan field, skip it (abbreviations shouldn't be used for lemmatization)
            # - Only apply transformation if there's a clear morphological pattern
            if field == 'lemma':
                # Check if similar entry has reg or expan field
                if isinstance(similar_entry, list):
                    similar_entry_dict = similar_entry[0] if similar_entry else {}
                else:
                    similar_entry_dict = similar_entry
                
                # Skip entries with expan field - these are abbreviations, not morphological variants
                expan = similar_entry_dict.get('expan')
                if expan and expan != '_' and expan.lower() != best_match.lower():
                    # This is an abbreviation, skip it and try next similar word
                    if len(similar) > 1:
                        # Try next similar word
                        for next_match, next_score in similar[1:]:
                            next_entry = vocab_to_check.get(next_match)
                            if not next_entry:
                                continue
                            if isinstance(next_entry, list):
                                next_entry_dict = next_entry[0] if next_entry else {}
                            else:
                                next_entry_dict = next_entry
                            next_expan = next_entry_dict.get('expan')
                            # Use this entry if it doesn't have an expan field (or expan == form)
                            if not next_expan or next_expan == '_' or next_expan.lower() == next_match.lower():
                                best_match = next_match
                                similar_entry = next_entry
                                if isinstance(next_entry, list):
                                    result = next_entry[0].get(field, '_') if next_entry else '_'
                                else:
                                    result = next_entry.get(field, '_')
                                similar_entry_dict = next_entry_dict
                                break
                        else:
                            # No suitable similar word found, return form as lemma
                            return form_lower
                    else:
                        # Only one similar word and it's an abbreviation, return form as lemma
                        return form_lower
                
                reg = similar_entry_dict.get('reg')
                if reg and reg != '_' and reg != best_match:
                    # Similar word has reg: lemma in entry corresponds to reg form
                    # Use the lemma directly, don't try to transform
                    return result
                
                # Only apply transformation if no reg field and clear morphological pattern
                if result != '_' and form_lower != result:
                    # Check if form matches a pattern (e.g., verb inflection)
                    # If similar word has lemma, try to apply same transformation
                    similar_form = best_match
                    similar_lemma = result
                    
                    # Simple heuristic: if form ends with common suffixes and lemma doesn't,
                    # try to remove suffix and match pattern
                    # This is a simplified version - could be enhanced with more rules
                    if len(form_lower) > len(similar_lemma):
                        # Try to extract lemma by removing common suffixes
                        common_suffixes = ['ed', 'ing', 's', 'es', 'er', 'est']
                        for suffix in common_suffixes:
                            if form_lower.endswith(suffix) and len(form_lower) - len(suffix) >= 3:
                                potential_lemma = form_lower[:-len(suffix)]
                                # Check if this matches the pattern of similar word
                                if similar_form.endswith(suffix):
                                    similar_base = similar_form[:-len(suffix)]
                                    if similar_base == similar_lemma:
                                        # Apply same transformation
                                        return potential_lemma
            
            return result
        
        # Final fallback: check lexicon if word is OOV (not in primary vocab)
        # Lexicon is only used when word is not in primary vocab at all, as a fallback for OOV words
        # This ensures corpus-specific vocabulary (with counts) takes priority over general lexicons
        if not word_in_vocab and self.lexicon:
            if debug:
                print(f"[DEBUG] Word '{form}' not found in primary vocab, checking lexicon (OOV fallback)", file=sys.stderr)
            
            # Try lexicon lookup (same logic as vocab, but only for OOV words)
            if form in self.lexicon:
                lexicon_result = get_field_from_entry(self.lexicon[form], field, xpos, upos, feats, debug_entry=debug)
                if lexicon_result and lexicon_result != '_':
                    if debug:
                        print(f"[DEBUG] Found in lexicon: '{lexicon_result}'", file=sys.stderr)
                    return lexicon_result
            elif form_lower in self.lexicon:
                lexicon_result = get_field_from_entry(self.lexicon[form_lower], field, xpos, upos, feats, debug_entry=debug)
                if lexicon_result and lexicon_result != '_':
                    if debug:
                        print(f"[DEBUG] Found in lexicon (lowercase): '{lexicon_result}'", file=sys.stderr)
                    return lexicon_result
        
            return '_'
        
    def _apply_lemmatization_patterns(self, form: str, xpos: str, debug: bool = False) -> str:
        """Apply lemmatization patterns to OOV word using shared function."""
        from flexipipe.lemmatizer import apply_lemmatization_patterns
        if not self.lemmatization_patterns:
            return '_'
        return apply_lemmatization_patterns(
            form,
            xpos,
            self.lemmatization_patterns,
            known_lemmas=self._get_known_lemmas(),
            debug=debug
        )

    def _get_known_lemmas(self) -> set:
        """Collect known lemmas from vocabulary and lexicon for pattern preference."""
        lemmas = set()

        def collect(source):
            if not source:
                return
            for entry in source.values():
                if isinstance(entry, list):
                    for analysis in entry:
                        lemma = analysis.get('lemma') if isinstance(analysis, dict) else None
                        if lemma and lemma != '_':
                            lemmas.add(lemma.lower())
                elif isinstance(entry, dict):
                    lemma = entry.get('lemma')
                    if lemma and lemma != '_':
                        lemmas.add(lemma.lower())

        collect(self.vocab)
        collect(self.external_vocab)
        collect(self.lexicon)

        return lemmas
    
    
    def write_output(self, sentences: List[List[Dict]], output_file: Optional[Path], format: str = "conllu"):
        """Write tagged sentences to output file or stdout.
        
        Args:
            sentences: List of tagged sentences
            output_file: Path to output file, or None for stdout
            format: Output format ('conllu', 'plain', 'text', 'plain-tagged')
        """
        if self.config.debug:
            print(f"[DEBUG] write_output() called: {len(sentences)} sentences, output_file={output_file}, format={format}", file=sys.stderr)
        
        # Handle stdout case
        use_stdout = (output_file is None or str(output_file) == '/dev/stdout' or str(output_file) == '-')
        
        if self.config.debug:
            print(f"[DEBUG] use_stdout={use_stdout}", file=sys.stderr)
        
        if not sentences:
            if use_stdout:
                print("Warning: No sentences to write", file=sys.stderr)
            else:
                print(f"Warning: No sentences to write to {output_file}", file=sys.stderr)
                # Create empty file to indicate processing completed
                output_file = Path(output_file)  # Ensure it's a Path object
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.touch()
            return
        
        # Convert to Path if needed (not for stdout)
        if not use_stdout:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file or stdout
        if use_stdout:
            f = sys.stdout
        else:
            try:
                f = open(output_file, 'w', encoding='utf-8')
            except Exception as e:
                print(f"Error: Could not open output file {output_file}: {e}", file=sys.stderr)
                raise
        
        try:
            if format == "conllu":
                if self.config.debug:
                    print(f"[DEBUG] Writing CoNLL-U format, {len(sentences)} sentences", file=sys.stderr)
                for sent_idx, sentence in enumerate(sentences):
                    if self.config.debug and sent_idx % 10 == 0:
                        print(f"[DEBUG] Writing sentence {sent_idx + 1}/{len(sentences)}", file=sys.stderr)
                    
                    # Check if we have original text stored in the sentence
                    original_text = None
                    for token in sentence:
                        if '_original_text' in token:
                            original_text = token.get('_original_text')
                            break
                    
                    # Determine SpaceAfter for each token
                    # If we have original text, derive SpaceAfter by matching tokens to original text
                    space_after_flags = []
                    
                    if original_text:
                        # Derive SpaceAfter from original text by matching tokens
                        # Use Unicode-aware matching to handle UTF-8 characters correctly
                        # Match tokens sequentially in the original text to preserve exact spacing
                        text_pos = 0  # Position in original text (character position, not byte)
                        
                        if self.config.debug:
                            print(f"[DEBUG] Deriving SpaceAfter from original text (length {len(original_text)}): {repr(original_text[:100])}", file=sys.stderr)
                        
                        for token_idx, token in enumerate(sentence):
                            form = token.get('form', '_')
                            if form == '_':
                                space_after_flags.append(True)  # Default
                                continue
                            
                            # Try to find the token in the original text starting from text_pos
                            # First try exact match (case-sensitive)
                            found_pos = original_text.find(form, text_pos)
                            
                            if found_pos == -1:
                                # Try case-insensitive match
                                original_text_lower = original_text.lower()
                                form_lower = form.lower()
                                found_pos_lower = original_text_lower.find(form_lower, text_pos)
                                
                                if found_pos_lower != -1:
                                    found_pos = found_pos_lower
                            
                            if found_pos != -1 and found_pos >= text_pos:
                                # Found the token, check if there's a space after it
                                end_pos = found_pos + len(form)
                                if end_pos < len(original_text):
                                    # Check if there's whitespace after the token
                                    next_char = original_text[end_pos]
                                    space_after = next_char.isspace()
                                    
                                    if self.config.debug and token_idx < 10:
                                        print(f"[DEBUG] Token {token_idx}: '{form}' at pos {found_pos}-{end_pos}, next char: '{repr(next_char)}', space_after: {space_after}", file=sys.stderr)
                                        print(f"[DEBUG]   Context: ...{repr(original_text[max(0, found_pos-5):end_pos+5])}...", file=sys.stderr)
                                else:
                                    # End of original_text - check if this is the last token in the sentence
                                    # Never set SpaceAfter=No for the final token just because it's at the end
                                    # Only set it if we can actually determine there's no space from the original text
                                    if token_idx == len(sentence) - 1:
                                        # Last token in sentence - don't set SpaceAfter=No just because it's the last token
                                        # If the original_text ends here, we can't determine if there should be a space
                                        # The original_text should include any trailing space if it exists
                                        # So if we're at the end, there's no space after (but we shouldn't write SpaceAfter=No)
                                        # Instead, we'll skip writing SpaceAfter=No for the final token
                                        space_after = True  # Default: don't write SpaceAfter=No for final token
                                        if self.config.debug:
                                            print(f"[DEBUG] Final token '{form}' at end of original_text - not setting SpaceAfter=No", file=sys.stderr)
                                    else:
                                        # Not the last token but we're at end of original_text
                                        # This shouldn't happen, but default to no space
                                        space_after = False
                                        if self.config.debug:
                                            print(f"[DEBUG] WARNING: Token {token_idx} '{form}' at end of original_text but not last token in sentence", file=sys.stderr)
                                
                                # Update text_pos for next token
                                # Start from end of current token
                                text_pos = end_pos
                                # If there's whitespace, skip it for next token search
                                if space_after:
                                    # Skip the whitespace we detected
                                    while text_pos < len(original_text) and original_text[text_pos].isspace():
                                        text_pos += 1
                            else:
                                # Token not found in original text at expected position
                                # This might happen if tokenization changed the form
                                # Use default heuristics
                                if self.config.debug:
                                    print(f"[DEBUG] Token '{form}' not found in original text at position {text_pos}, using heuristics", file=sys.stderr)
                                
                                misc_str = token.get('misc', '_')
                                space_after = True  # Default
                                
                                if misc_str and misc_str != '_':
                                    misc_parts = misc_str.split('|')
                                    if 'SpaceAfter=No' in misc_parts:
                                        space_after = False
                                else:
                                    punct_no_space = [',', ';', ':', '.', '!', '?', ')', ']', '}', '"', "'", '»', '»']
                                    if form in punct_no_space:
                                        space_after = False
                                
                                # Try to advance text_pos anyway to avoid getting stuck
                                # Look for the token anywhere after current position
                                if text_pos < len(original_text):
                                    # Skip to next non-whitespace or try to find token
                                    temp_pos = text_pos
                                    while temp_pos < len(original_text) and original_text[temp_pos].isspace():
                                        temp_pos += 1
                                    if temp_pos < len(original_text):
                                        text_pos = temp_pos
                            
                            space_after_flags.append(space_after)
                    else:
                        # No original text, use heuristics
                        for token_idx, token in enumerate(sentence):
                            form = token.get('form', '_')
                            if form == '_':
                                space_after_flags.append(True)  # Default
                                continue
                            
                            # Check if token has SpaceAfter=No in MISC
                            misc_str = token.get('misc', '_')
                            space_after = True  # Default: assume space after
                            
                            if misc_str and misc_str != '_':
                                misc_parts = misc_str.split('|')
                                if 'SpaceAfter=No' in misc_parts:
                                    space_after = False
                            else:
                                # Infer SpaceAfter from token characteristics
                                punct_no_space = [',', ';', ':', '.', '!', '?', ')', ']', '}', '"', "'", '»', '»']
                                if form in punct_no_space:
                                    if token_idx == len(sentence) - 1:
                                        space_after = False
                                    else:
                                        space_after = False
                            
                            space_after_flags.append(space_after)
                    
                    if original_text:
                        # Use original text from input (preserves exact spacing)
                        # Strip trailing newlines (but preserve other whitespace)
                        sentence_text = original_text.rstrip('\n\r')
                    else:
                        # Reconstruct sentence text from tokens (fallback)
                        # Build sentence text: add space between tokens unless SpaceAfter=No
                        # Punctuation typically attaches to previous token (no space before it)
                        sentence_text = ""
                        for token_idx, token in enumerate(sentence):
                            form = token.get('form', '_')
                            if form == '_':
                                continue
                            
                            # Check if this token is punctuation
                            punct_no_space_after = [',', ';', ':', '.', '!', '?', ')', ']', '}', '»', '«']
                            is_punct = form in punct_no_space_after
                            
                            # Add space BEFORE this token if:
                            # 1. It's not the first token
                            # 2. It's not punctuation (punctuation attaches to previous token)
                            # 3. The previous token has SpaceAfter=True
                            if token_idx > 0:
                                prev_token_idx = token_idx - 1
                                # Find previous non-empty token
                                while prev_token_idx >= 0 and sentence[prev_token_idx].get('form', '_') == '_':
                                    prev_token_idx -= 1
                                
                                if prev_token_idx >= 0 and prev_token_idx < len(space_after_flags):
                                    if space_after_flags[prev_token_idx] and not is_punct:
                                        # Previous token has space after and this is not punctuation
                                        sentence_text += " "
                            
                            sentence_text += form
                            
                            # Add space AFTER this token if it has space after and next token is not punctuation
                            if token_idx < len(space_after_flags):
                                if space_after_flags[token_idx] and token_idx < len(sentence) - 1:
                                    # Check next token
                                    next_token_idx = token_idx + 1
                                    while next_token_idx < len(sentence) and sentence[next_token_idx].get('form', '_') == '_':
                                        next_token_idx += 1
                                    
                                    if next_token_idx < len(sentence):
                                        next_form = sentence[next_token_idx].get('form', '_')
                                        if next_form != '_' and next_form not in punct_no_space_after:
                                            sentence_text += " "
                    
                    # Write # sent_id = comment if available (from TEITOK XML <s> @id)
                    if sentence and sentence[0].get('_sentence_id'):
                        f.write(f"# sent_id = {sentence[0]['_sentence_id']}\n")
                    
                    # Write # text = comment (always required in CoNLL-U)
                    f.write(f"# text = {sentence_text}\n")
                    
                    # Write tokens
                    for token_idx, token in enumerate(sentence):
                        tid = token.get('id', 0)
                        form = token.get('form', '_')
                        lemma = token.get('lemma', '_')
                        upos = token.get('upos', '_')
                        xpos = token.get('xpos', '_')
                        feats = token.get('feats', '_')
                        head = token.get('head', '_')
                        deprel = token.get('deprel', '_')
                        
                        # If head is 0 or numeric but we're not parsing, convert to '_'
                        # In CoNLL-U, head should be '_' when parsing is not available, 0 only for root in dependency trees
                        if head == 0 or head == '0':
                            # Check if we're actually parsing (head=0 means root in dependency trees)
                            # If not parsing, use '_' instead
                            if not self.config.parse:
                                head = '_'
                        
                        # Build MISC column with original/normalized forms and SpaceAfter
                        misc_parts = []
                        # Token ID from TEITOK XML (@id or @xml:id)
                        if 'tok_id' in token and token['tok_id']:
                            misc_parts.append(f"TokId={token['tok_id']}")
                        if 'orig_form' in token and token['orig_form'] != form:
                            misc_parts.append(f"OrigForm={token['orig_form']}")
                        # Normalization: use Normalized= in CoNLL-U MISC
                        if 'norm_form' in token and token['norm_form'] and token['norm_form'] != '_':
                            misc_parts.append(f"Normalized={token['norm_form']}")
                        # Expansion: use Expansion= in CoNLL-U MISC, and add Abbr=Yes
                        if 'expan' in token and token['expan'] and token['expan'] != '_':
                            misc_parts.append(f"Expansion={token['expan']}")
                            misc_parts.append("Abbr=Yes")
                        if 'split_forms' in token:
                            misc_parts.append(f"SplitForms={'+'.join(token['split_forms'])}")
                        
                        # Add SpaceAfter=No if token doesn't have space after it
                        # But never add it for the final token in a sentence (it's ambiguous)
                        if token_idx < len(space_after_flags) and token_idx < len(sentence) - 1:
                            if not space_after_flags[token_idx]:
                                if 'SpaceAfter=No' not in misc_parts:
                                    misc_parts.append('SpaceAfter=No')
                        
                        misc = '|'.join(misc_parts) if misc_parts else '_'
                        
                        f.write(f"{tid}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\t{misc}\n")
                    f.write("\n")
            elif format == "plain" or format == "text":
                # Plain text output: one sentence per line, tokens separated by spaces
                if self.config.debug:
                    print(f"[DEBUG] Writing plain text format, {len(sentences)} sentences", file=sys.stderr)
                for sent_idx, sentence in enumerate(sentences):
                    if self.config.debug and sent_idx % 10 == 0:
                        print(f"[DEBUG] Writing sentence {sent_idx + 1}/{len(sentences)}", file=sys.stderr)
                    forms = [token.get('form', '_') for token in sentence]
                    f.write(' '.join(forms) + '\n')
            elif format == "plain-tagged":
                # Plain text with tags: one sentence per line with UPOS tags
                # Format: word/UPOS word/UPOS ...
                if self.config.debug:
                    print(f"[DEBUG] Writing plain-tagged format, {len(sentences)} sentences", file=sys.stderr)
                for sent_idx, sentence in enumerate(sentences):
                    if self.config.debug and sent_idx % 10 == 0:
                        print(f"[DEBUG] Writing sentence {sent_idx + 1}/{len(sentences)}", file=sys.stderr)
                    tagged_words = []
                    for token in sentence:
                        form = token.get('form', '_')
                        upos = token.get('upos', '_')
                        if upos != '_':
                            tagged_words.append(f"{form}/{upos}")
                        else:
                            tagged_words.append(form)
                    f.write(' '.join(tagged_words) + '\n')
            elif format == "teitok":
                # For TEITOK XML, we need to handle file vs stdout differently
                # Pass the file handle and whether it's stdout
                self._write_teitok_xml(sentences, f, is_stdout=use_stdout)
            else:
                # Unknown format - default to CoNLL-U
                if self.config.debug:
                    print(f"[DEBUG] Unknown format '{format}', defaulting to CoNLL-U", file=sys.stderr)
                # Use same logic as conllu format
                for sent_idx, sentence in enumerate(sentences):
                    if self.config.debug and sent_idx % 10 == 0:
                        print(f"[DEBUG] Writing sentence {sent_idx + 1}/{len(sentences)}", file=sys.stderr)
                    
                    # Determine SpaceAfter for each token (same logic as conllu format)
                    space_after_flags = []
                    for token_idx, token in enumerate(sentence):
                        form = token.get('form', '_')
                        if form == '_':
                            space_after_flags.append(True)
                            continue
                        
                        misc_str = token.get('misc', '_')
                        space_after = True
                        
                        if misc_str and misc_str != '_':
                            misc_parts = misc_str.split('|')
                            if 'SpaceAfter=No' in misc_parts:
                                space_after = False
                        else:
                            punct_no_space = [',', ';', ':', '.', '!', '?', ')', ']', '}', '"', "'", '»', '»']
                            if form in punct_no_space:
                                space_after = False
                        
                        space_after_flags.append(space_after)
                    
                    # Check if we have original text stored in the sentence
                    original_text = None
                    for token in sentence:
                        if '_original_text' in token:
                            original_text = token.get('_original_text')
                            break
                    
                    if original_text:
                        # Use original text from input (preserves exact spacing)
                        # Strip trailing newlines (but preserve other whitespace)
                        sentence_text = original_text.rstrip('\n\r')
                    else:
                        # Reconstruct sentence text (fallback)
                        sentence_text = ""
                        punct_no_space_after = [',', ';', ':', '.', '!', '?', ')', ']', '}', '»', '«']
                        for token_idx, token in enumerate(sentence):
                            form = token.get('form', '_')
                            if form == '_':
                                continue
                            
                            is_punct = form in punct_no_space_after
                            
                            if token_idx > 0:
                                prev_token_idx = token_idx - 1
                                while prev_token_idx >= 0 and sentence[prev_token_idx].get('form', '_') == '_':
                                    prev_token_idx -= 1
                                
                                if prev_token_idx >= 0 and prev_token_idx < len(space_after_flags):
                                    if space_after_flags[prev_token_idx] and not is_punct:
                                        sentence_text += " "
                            
                            sentence_text += form
                            
                            if token_idx < len(space_after_flags):
                                if space_after_flags[token_idx] and token_idx < len(sentence) - 1:
                                    next_token_idx = token_idx + 1
                                    while next_token_idx < len(sentence) and sentence[next_token_idx].get('form', '_') == '_':
                                        next_token_idx += 1
                                    
                                    if next_token_idx < len(sentence):
                                        next_form = sentence[next_token_idx].get('form', '_')
                                        if next_form != '_' and next_form not in punct_no_space_after:
                                            sentence_text += " "
                    
                    # Write # text = comment
                    f.write(f"# text = {sentence_text}\n")
                    
                    # Write tokens
                    for token_idx, token in enumerate(sentence):
                        tid = token.get('id', 0)
                        form = token.get('form', '_')
                        lemma = token.get('lemma', '_')
                        upos = token.get('upos', '_')
                        xpos = token.get('xpos', '_')
                        feats = token.get('feats', '_')
                        head = token.get('head', '_')
                        deprel = token.get('deprel', '_')
                        
                        if head == 0 or head == '0':
                            if not self.config.parse:
                                head = '_'
                        
                        misc_parts = []
                        if 'orig_form' in token and token['orig_form'] != form:
                            misc_parts.append(f"OrigForm={token['orig_form']}")
                        if 'norm_form' in token and token['norm_form'] and token['norm_form'] != '_':
                            misc_parts.append(f"Normalized={token['norm_form']}")
                        # Expansion: use Expansion= in CoNLL-U MISC, and add Abbr=Yes
                        if 'expan' in token and token['expan'] and token['expan'] != '_':
                            misc_parts.append(f"Expansion={token['expan']}")
                            misc_parts.append("Abbr=Yes")
                        if 'split_forms' in token:
                            misc_parts.append(f"SplitForms={'+'.join(token['split_forms'])}")
                        
                        # Add SpaceAfter=No if token doesn't have space after it
                        # But never add it for the final token in a sentence (it's ambiguous)
                        if token_idx < len(space_after_flags) and token_idx < len(sentence) - 1:
                            if not space_after_flags[token_idx]:
                                if 'SpaceAfter=No' not in misc_parts:
                                    misc_parts.append('SpaceAfter=No')
                        
                        misc = '|'.join(misc_parts) if misc_parts else '_'
                        
                        f.write(f"{tid}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\t{misc}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            raise
        finally:
            # Only close if it's a file, not stdout
            if not use_stdout:
                try:
                    f.flush()  # Ensure data is written
                    f.close()
                    if self.config.debug:
                        print(f"[DEBUG] File closed: {output_file}", file=sys.stderr)
                    # Verify file was actually written
                    if not output_file.exists():
                        print(f"Warning: File {output_file} was not created", file=sys.stderr)
                    else:
                        file_size = output_file.stat().st_size
                        if self.config.debug:
                            print(f"[DEBUG] File exists, size: {file_size} bytes", file=sys.stderr)
                        if file_size == 0:
                            print(f"Warning: File {output_file} was created but is empty", file=sys.stderr)
                except Exception as e:
                    print(f"Error: Could not close output file {output_file}: {e}", file=sys.stderr)
            else:
                if self.config.debug:
                    print(f"[DEBUG] Wrote to stdout, flushing...", file=sys.stderr)
                sys.stdout.flush()
    
    def _write_teitok_xml(self, sentences: List[List[Dict]], output_file, is_stdout: bool = False):
        """Write tagged sentences to TEITOK XML format.
        
        If original TEITOK XML was loaded, preserves all structure and only updates NLP annotations.
        Otherwise, creates new TEITOK XML structure from tagged sentences.
        
        Args:
            sentences: List of tagged sentences
            output_file: File handle or stdout to write to
            is_stdout: Whether output_file is stdout (affects how XML is written)
        """
        import xml.etree.ElementTree as ET
        from datetime import datetime
        
        if self.original_teitok_tree is not None:
            # Update existing TEITOK XML: preserve all structure, only update NLP annotations
            root = self.original_teitok_tree.getroot()
            
            # Create mapping from tok_id to token data
            tok_id_to_data = {}
            for sentence in sentences:
                for token in sentence:
                    tok_id = token.get('tok_id', '')
                    if tok_id:
                        tok_id_to_data[tok_id] = token
                    # Also check dtok_id
                    dtok_id = token.get('dtok_id', '')
                    if dtok_id:
                        tok_id_to_data[dtok_id] = token
            
            # Update all <tok> and <dtok> elements with new annotations
            # IMPORTANT: Only update NLP attributes, preserve all other attributes and structure
            for tok in root.findall('.//tok'):
                tok_id = tok.get('id', '') or tok.get('{http://www.w3.org/XML/1998/namespace}id', '')
                if tok_id and tok_id in tok_id_to_data:
                    token_data = tok_id_to_data[tok_id]
                    # Update NLP attributes (only if new values are available)
                    # Preserve all other attributes (typesetting, hands, linebreaks, etc.)
                    if token_data.get('lemma') and token_data.get('lemma') != '_':
                        tok.set('lemma', token_data['lemma'])
                    if token_data.get('upos') and token_data.get('upos') != '_':
                        tok.set('upos', token_data['upos'])
                    if token_data.get('xpos') and token_data.get('xpos') != '_':
                        # Use configured xpos attribute name
                        xpos_attr = self.config.xpos_attr
                        tok.set(xpos_attr, token_data['xpos'])
                    if token_data.get('feats') and token_data.get('feats') != '_':
                        tok.set('feats', token_data['feats'])
                    if token_data.get('norm_form') and token_data.get('norm_form') != '_':
                        # Use configured normalization attribute
                        norm_attr = self.config.normalization_attr
                        tok.set(norm_attr, token_data['norm_form'])
                    if token_data.get('expan') and token_data.get('expan') != '_':
                        # Use configured expansion attribute
                        expan_attr = self.config.expansion_attr
                        tok.set(expan_attr, token_data['expan'])
                    if token_data.get('head') and token_data.get('head') != '_' and token_data.get('head') != '0':
                        tok.set('head', str(token_data['head']))
                    if token_data.get('deprel') and token_data.get('deprel') != '_':
                        tok.set('deprel', token_data['deprel'])
                
                # Also update dtok children
                for dtok in tok.findall('.//dtok'):
                    dtok_id = dtok.get('id', '') or dtok.get('{http://www.w3.org/XML/1998/namespace}id', '')
                    if dtok_id and dtok_id in tok_id_to_data:
                        token_data = tok_id_to_data[dtok_id]
                        if token_data.get('lemma') and token_data.get('lemma') != '_':
                            dtok.set('lemma', token_data['lemma'])
                        if token_data.get('upos') and token_data.get('upos') != '_':
                            dtok.set('upos', token_data['upos'])
                        if token_data.get('xpos') and token_data.get('xpos') != '_':
                            xpos_attr = self.config.xpos_attr
                            dtok.set(xpos_attr, token_data['xpos'])
                        if token_data.get('feats') and token_data.get('feats') != '_':
                            dtok.set('feats', token_data['feats'])
                        if token_data.get('norm_form') and token_data.get('norm_form') != '_':
                            norm_attr = self.config.normalization_attr
                            dtok.set(norm_attr, token_data['norm_form'])
                        if token_data.get('expan') and token_data.get('expan') != '_':
                            expan_attr = self.config.expansion_attr
                            dtok.set(expan_attr, token_data['expan'])
            
            # Add revision statement
            self._add_revision_statement(root, sentences)
            
            # Check if original XML has xml:space="preserve" on <text>
            # If so, only pretty-print teiHeader, not the text content
            text_elem = root.find('.//text')
            should_pretty_print_text = True
            if text_elem is not None:
                xml_space = text_elem.get('{http://www.w3.org/XML/1998/namespace}space')
                if xml_space == 'preserve':
                    should_pretty_print_text = False
                    # Pretty-print only the teiHeader for readability
                    teiHeader = root.find('.//teiHeader')
                    if teiHeader is not None:
                        ET.indent(teiHeader, space="  ")
            
            # Write XML
            if should_pretty_print_text:
                ET.indent(root, space="  ")
            tree = ET.ElementTree(root)
            # Handle stdout vs file differently
            if is_stdout:
                # For stdout, write as string
                # ET.tostring doesn't support xml_declaration, so we add it manually
                xml_str = ET.tostring(root, encoding='unicode')
                output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                output_file.write(xml_str)
                output_file.write('\n')
            else:
                # For file, use tree.write() with file path or binary mode
                # If it's a file handle opened in text mode, we need to write as string
                if hasattr(output_file, 'mode') and 'b' not in output_file.mode:
                    # Text mode file handle - write as string
                    xml_str = ET.tostring(root, encoding='unicode')
                    output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    output_file.write(xml_str)
                    output_file.write('\n')
                else:
                    # Binary mode or file path - use tree.write()
                    tree.write(output_file, encoding='utf-8', xml_declaration=True)
        else:
            # Create new TEITOK XML from tagged sentences
            root = ET.Element('TEI')
            teiHeader = ET.SubElement(root, 'teiHeader')
            fileDesc = ET.SubElement(teiHeader, 'fileDesc')
            titleStmt = ET.SubElement(fileDesc, 'titleStmt')
            title = ET.SubElement(titleStmt, 'title')
            title.text = 'FlexiPipe Tagged Text'
            
            # Add revision statement
            self._add_revision_statement(root, sentences)
            
            text = ET.SubElement(root, 'text')
            # Add xml:space="preserve" to preserve whitespace
            text.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            body = ET.SubElement(text, 'body')
            
            # Create sentences from tagged data
            for sent_idx, sentence in enumerate(sentences):
                s = ET.SubElement(body, 's')
                # Add sentence ID if available
                if sentence and sentence[0].get('_sentence_id'):
                    s.set('id', sentence[0]['_sentence_id'])
                else:
                    s.set('id', f's-{sent_idx + 1}')
                
                # Determine SpaceAfter for each token (same logic as CoNLL-U output)
                # SpaceAfter=No means there's no space after this token
                space_after_flags = []
                
                # Check if we have original text stored in the sentence
                original_text = None
                for token in sentence:
                    if '_original_text' in token:
                        original_text = token.get('_original_text')
                        break
                
                if original_text:
                    # Derive SpaceAfter from original text by matching tokens
                    # This matches the CoNLL-U output logic exactly
                    text_pos = 0  # Position in original text
                    
                    for token_idx, token in enumerate(sentence):
                        form = token.get('form', '_')
                        if form == '_':
                            space_after_flags.append(True)
                            continue
                        
                        # Check if token has SpaceAfter=No in MISC (highest priority)
                        misc_str = token.get('misc', '_')
                        space_after = None
                        
                        if misc_str and misc_str != '_':
                            misc_parts = misc_str.split('|')
                            if 'SpaceAfter=No' in misc_parts:
                                space_after = False
                        
                        # If MISC doesn't specify, derive from original text
                        if space_after is None:
                            # Try to find the token in the original text starting from text_pos
                            # First try exact match (case-sensitive)
                            found_pos = original_text.find(form, text_pos)
                            
                            if found_pos == -1:
                                # Try case-insensitive match
                                original_text_lower = original_text.lower()
                                form_lower = form.lower()
                                found_pos_lower = original_text_lower.find(form_lower, text_pos)
                                
                                if found_pos_lower != -1:
                                    found_pos = found_pos_lower
                            
                            if found_pos != -1 and found_pos >= text_pos:
                                # Found the token, check if there's a space after it
                                end_pos = found_pos + len(form)
                                if end_pos < len(original_text):
                                    # Check if there's whitespace after the token
                                    next_char = original_text[end_pos]
                                    space_after = next_char.isspace()
                                else:
                                    # End of original text
                                    if token_idx == len(sentence) - 1:
                                        # Last token - don't set SpaceAfter=No just because it's at the end
                                        space_after = True
                                    else:
                                        space_after = False
                                
                                # Update text_pos for next token
                                text_pos = end_pos
                                # If there's whitespace, skip it for next token search
                                if space_after:
                                    while text_pos < len(original_text) and original_text[text_pos].isspace():
                                        text_pos += 1
                            else:
                                # Token not found in original text - use heuristics
                                misc_str = token.get('misc', '_')
                                space_after = True  # Default
                                
                                if misc_str and misc_str != '_':
                                    misc_parts = misc_str.split('|')
                                    if 'SpaceAfter=No' in misc_parts:
                                        space_after = False
                                else:
                                    punct_no_space = [',', ';', ':', '.', '!', '?', ')', ']', '}', '"', "'", '»', '«']
                                    if form in punct_no_space:
                                        space_after = False
                                
                                # Try to advance text_pos anyway to avoid getting stuck
                                if text_pos < len(original_text):
                                    temp_pos = text_pos
                                    while temp_pos < len(original_text) and original_text[temp_pos].isspace():
                                        temp_pos += 1
                                    if temp_pos < len(original_text):
                                        text_pos = temp_pos
                        
                        space_after_flags.append(space_after)
                else:
                    # No original text - use MISC or heuristics (same as before)
                    for token_idx, token in enumerate(sentence):
                        form = token.get('form', '_')
                        if form == '_':
                            space_after_flags.append(True)
                            continue
                        
                        # Check if token has SpaceAfter=No in MISC (highest priority)
                        misc_str = token.get('misc', '_')
                        space_after = None
                        
                        if misc_str and misc_str != '_':
                            misc_parts = misc_str.split('|')
                            if 'SpaceAfter=No' in misc_parts:
                                space_after = False
                        
                        # If MISC doesn't specify, infer from context
                        if space_after is None:
                            if token_idx < len(sentence) - 1:
                                next_form = sentence[token_idx + 1].get('form', '_')
                                if next_form != '_':
                                    punct_no_space_before = [',', ';', ':', '.', '!', '?', ')', ']', '}', '"', "'", '»', '«']
                                    space_after = next_form not in punct_no_space_before
                                else:
                                    space_after = True
                            else:
                                space_after = False
                        
                        space_after_flags.append(space_after)
                
                # Create tokens with proper spacing
                for token_idx, token in enumerate(sentence):
                    form = token.get('form', '_')
                    if form == '_':
                        continue
                    
                    tok = ET.SubElement(s, 'tok')
                    tok_id = token.get('tok_id', '')
                    if tok_id:
                        tok.set('id', tok_id)
                    else:
                        tok.set('id', f'w-{sent_idx + 1}-{token_idx + 1}')
                    
                    tok.text = form
                    
                    # Add space after token (as tail) if SpaceAfter is True
                    # The space goes after this token, before the next one
                    if token_idx < len(space_after_flags) and space_after_flags[token_idx]:
                        # Add a tail (whitespace after the element) if there's a next token
                        if token_idx < len(sentence) - 1:
                            # Check if next token is not empty
                            next_form = sentence[token_idx + 1].get('form', '_')
                            if next_form != '_':
                                tok.tail = ' '
                            else:
                                tok.tail = None
                        else:
                            # Last token - no space after
                            tok.tail = None
                    else:
                        # SpaceAfter=No - no space after this token
                        tok.tail = None
                    
                    # Add NLP attributes
                    if token.get('lemma') and token.get('lemma') != '_':
                        tok.set('lemma', token['lemma'])
                    if token.get('upos') and token.get('upos') != '_':
                        tok.set('upos', token['upos'])
                    if token.get('xpos') and token.get('xpos') != '_':
                        # Use configured xpos attribute name
                        xpos_attr = self.config.xpos_attr
                        tok.set(xpos_attr, token['xpos'])
                    if token.get('feats') and token.get('feats') != '_':
                        tok.set('feats', token['feats'])
                    if token.get('norm_form') and token.get('norm_form') != '_':
                        # Use configured normalization attribute
                        norm_attr = self.config.normalization_attr
                        tok.set(norm_attr, token['norm_form'])
                    if token.get('expan') and token.get('expan') != '_':
                        # Use configured expansion attribute
                        expan_attr = self.config.expansion_attr
                        tok.set(expan_attr, token['expan'])
                    if token.get('head') and token.get('head') != '_' and token.get('head') != '0':
                        tok.set('head', str(token['head']))
                    if token.get('deprel') and token.get('deprel') != '_':
                        tok.set('deprel', token['deprel'])
            
            # Write XML
            # Don't pretty-print the whole tree since we have xml:space="preserve" on <text>
            # But pretty-print the teiHeader for readability
            teiHeader = root.find('.//teiHeader')
            if teiHeader is not None:
                ET.indent(teiHeader, space="  ")
            tree = ET.ElementTree(root)
            # Handle stdout vs file differently
            if is_stdout:
                # For stdout, write as string
                # ET.tostring doesn't support xml_declaration, so we add it manually
                xml_str = ET.tostring(root, encoding='unicode')
                output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                output_file.write(xml_str)
                output_file.write('\n')
            else:
                # For file, use tree.write() with file path or binary mode
                # If it's a file handle opened in text mode, we need to write as string
                if hasattr(output_file, 'mode') and 'b' not in output_file.mode:
                    # Text mode file handle - write as string
                    xml_str = ET.tostring(root, encoding='unicode')
                    output_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    output_file.write(xml_str)
                    output_file.write('\n')
                else:
                    # Binary mode or file path - use tree.write()
                    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    def _add_revision_statement(self, root, sentences: List[List[Dict]]):
        """Add revision statement to TEITOK XML.
        
        Args:
            root: Root element of TEITOK XML tree
            sentences: List of tagged sentences (to check for lemmatization)
        """
        import xml.etree.ElementTree as ET
        from datetime import datetime
        
        # Find or create revisionDesc
        revisionDesc = root.find('.//revisionDesc')
        if revisionDesc is None:
            # Find teiHeader
            teiHeader = root.find('.//teiHeader')
            if teiHeader is None:
                # Create teiHeader structure if it doesn't exist
                teiHeader = ET.Element('teiHeader')
                fileDesc = ET.SubElement(teiHeader, 'fileDesc')
                titleStmt = ET.SubElement(fileDesc, 'titleStmt')
                title = ET.SubElement(titleStmt, 'title')
                title.text = 'FlexiPipe Tagged Text'
                # Insert at beginning
                root.insert(0, teiHeader)
            
            revisionDesc = ET.SubElement(teiHeader, 'revisionDesc')
        
        # Create change element
        change = ET.SubElement(revisionDesc, 'change')
        change.set('who', 'flexiPipe')
        change.set('when', datetime.now().isoformat())
        
        # Build description of operations
        operations = []
        if 'processed_teitok_input' in self.operations_performed:
            operations.append("processed TEITOK XML input")
        if self.config.normalize:
            operations.append("normalized orthographic variants")
        if self.config.parse:
            operations.append("parsed dependencies")
        if self.config.tag_only or (not self.config.parse_only and not self.config.parse):
            operations.append("tagged with UPOS/XPOS/FEATS")
        
        # Check if lemmatization was performed (any token has lemma)
        has_lemmas = False
        for sentence in sentences:
            for token in sentence:
                if token.get('lemma') and token.get('lemma') != '_' and token.get('lemma') != token.get('form', '').lower():
                    has_lemmas = True
                    break
            if has_lemmas:
                break
        if has_lemmas:
            operations.append("lemmatized")
        
        # Add model/vocab info
        method_parts = []
        if self.model_path:
            method_parts.append(f"model: {self.model_path.name}")
        elif self.transition_probs:
            method_parts.append("Viterbi tagging with vocabulary")
        if self.external_vocab:
            method_parts.append("vocabulary-based")
        # Add vocabulary file name if available
        if self.vocab_file_paths:
            if len(self.vocab_file_paths) == 1:
                vocab_name = Path(self.vocab_file_paths[0]).name
                method_parts.append(f"vocab: {vocab_name}")
            else:
                # Multiple vocab files - show count and most specific
                vocab_names = [Path(p).name for p in self.vocab_file_paths]
                method_parts.append(f"vocab: {len(vocab_names)} files ({vocab_names[-1]} most specific)")
        
        description = "FlexiPipe: " + ", ".join(operations)
        if method_parts:
            description += " (" + ", ".join(method_parts) + ")"
        
        change.text = description
    
    def calculate_accuracy(self, gold_file: Path, pred_file: Path, format: str = "conllu"):
        """Calculate accuracy metrics."""
        if format == "teitok":
            gold_sentences = load_teitok_xml(gold_file)
            pred_sentences = load_teitok_xml(pred_file)
        elif format == "plain" or format == "text":
            gold_sentences = load_plain_text(gold_file)
            pred_sentences = load_plain_text(pred_file)
        else:
            gold_sentences = load_conllu_file(gold_file)
            pred_sentences = load_conllu_file(pred_file)
        
        metrics = {
            'total_tokens': 0,
            'total_sentences': 0,
            'upos_correct': 0,
            'xpos_correct': 0,
            'feats_correct': 0,
            'lemma_correct': 0,
            'all_tags_correct': 0,  # AllTags: UPOS + XPOS + FEATS
            'uas_correct': 0,  # Unlabeled Attachment Score (head)
            'las_correct': 0,  # Labeled Attachment Score (head + deprel)
            'mlas_correct': 0,  # Morphology-aware LAS (only tokens with correct morphology)
            'blex_correct': 0,  # Bilexical dependency (head + deprel + head form/lemma)
        }
        
        for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
            min_len = min(len(gold_sent), len(pred_sent))
            if min_len == 0:
                continue
                
            metrics['total_sentences'] += 1
            
            for i in range(min_len):
                gold = gold_sent[i]
                pred = pred_sent[i]
                
                metrics['total_tokens'] += 1
                
                # Token-level metrics
                upos_match = gold.get('upos', '_') == pred.get('upos', '_')
                xpos_match = gold.get('xpos', '_') == pred.get('xpos', '_')
                
                # FEATS comparison: normalize feature string ordering
                # UD allows features in any order, so we need to compare sets
                gold_feats_str = gold.get('feats', '_')
                pred_feats_str = pred.get('feats', '_')
                if gold_feats_str == '_' and pred_feats_str == '_':
                    feats_match = True
                elif gold_feats_str == '_' or pred_feats_str == '_':
                    feats_match = False
                else:
                    # Parse feature strings into sets of feature=value pairs
                    gold_feats_set = set(sorted(gold_feats_str.split('|')))
                    pred_feats_set = set(sorted(pred_feats_str.split('|')))
                    feats_match = gold_feats_set == pred_feats_set
                
                lemma_match = gold.get('lemma', '_').lower() == pred.get('lemma', '_').lower()
                
                if upos_match:
                    metrics['upos_correct'] += 1
                if xpos_match:
                    metrics['xpos_correct'] += 1
                if feats_match:
                    metrics['feats_correct'] += 1
                if lemma_match:
                    metrics['lemma_correct'] += 1
                
                # AllTags: UPOS + XPOS + FEATS (all three must match)
                if upos_match and xpos_match and feats_match:
                    metrics['all_tags_correct'] += 1
                
                # Dependency metrics (head and deprel)
                gold_head = gold.get('head', 0)
                pred_head = pred.get('head', 0)
                gold_deprel = gold.get('deprel', '_')
                pred_deprel = pred.get('deprel', '_')
                
                # UAS: Unlabeled Attachment Score (correct head)
                try:
                    gold_head_int = int(gold_head) if str(gold_head).isdigit() else 0
                    pred_head_int = int(pred_head) if str(pred_head).isdigit() else 0
                    if gold_head_int == pred_head_int:
                        metrics['uas_correct'] += 1
                except (ValueError, TypeError):
                    pass
                
                # LAS: Labeled Attachment Score (correct head + deprel)
                try:
                    gold_head_int = int(gold_head) if str(gold_head).isdigit() else 0
                    pred_head_int = int(pred_head) if str(pred_head).isdigit() else 0
                    if gold_head_int == pred_head_int and gold_deprel == pred_deprel:
                        metrics['las_correct'] += 1
                except (ValueError, TypeError):
                    pass
                
                # MLAS: Morphology-aware LAS (LAS but only for tokens with correct morphology)
                # Only count if morphology (UPOS + XPOS + FEATS) is correct
                try:
                    gold_head_int = int(gold_head) if str(gold_head).isdigit() else 0
                    pred_head_int = int(pred_head) if str(pred_head).isdigit() else 0
                    if (upos_match and xpos_match and feats_match and 
                        gold_head_int == pred_head_int and gold_deprel == pred_deprel):
                        metrics['mlas_correct'] += 1
                except (ValueError, TypeError):
                    pass
                
                # BLEX: Bilexical dependency accuracy (head + deprel + head's form/lemma)
                try:
                    gold_head_int = int(gold_head) if str(gold_head).isdigit() else 0
                    pred_head_int = int(pred_head) if str(pred_head).isdigit() else 0
                    
                    if gold_head_int == pred_head_int and gold_deprel == pred_deprel:
                        # Check if head's form and lemma match
                        if gold_head_int > 0 and gold_head_int <= len(gold_sent):
                            head_idx = gold_head_int - 1  # Convert to 0-based index
                            if head_idx < len(gold_sent) and head_idx < len(pred_sent):
                                gold_head_token = gold_sent[head_idx]
                                pred_head_token = pred_sent[head_idx]
                                
                                head_form_match = gold_head_token.get('form', '') == pred_head_token.get('form', '')
                                head_lemma_match = gold_head_token.get('lemma', '').lower() == pred_head_token.get('lemma', '').lower()
                                
                                if head_form_match and head_lemma_match:
                                    metrics['blex_correct'] += 1
                        elif gold_head_int == 0:  # Root node
                            # For root, head form/lemma don't matter
                            metrics['blex_correct'] += 1
                except (ValueError, TypeError, IndexError):
                    pass
        
        total_tokens = metrics['total_tokens']
        total_sentences = metrics['total_sentences']
        
        if total_tokens > 0:
            print(f"\nCoNLL-U Evaluation Metrics:")
            print(f"  Words: {total_tokens}")
            print(f"  Sentences: {total_sentences}")
            print(f"  UPOS: {100*metrics['upos_correct']/total_tokens:.2f}%")
            print(f"  XPOS: {100*metrics['xpos_correct']/total_tokens:.2f}%")
            print(f"  UFeats: {100*metrics['feats_correct']/total_tokens:.2f}%")
            print(f"  AllTags: {100*metrics['all_tags_correct']/total_tokens:.2f}%")
            lemma_acc = 100*metrics['lemma_correct']/total_tokens
            print(f"  Lemma: {lemma_acc:.2f}%", end='')
            if lemma_acc > 99.5:
                print(" (⚠️  WARNING: Lemma uses vocabulary lookup, not model predictions. High accuracy may indicate data leakage if test words are in training vocabulary.)")
            else:
                print()
            print(f"  UAS: {100*metrics['uas_correct']/total_tokens:.2f}%")
            print(f"  LAS: {100*metrics['las_correct']/total_tokens:.2f}%")
            print(f"  MLAS: {100*metrics['mlas_correct']/total_tokens:.2f}%")
            print(f"  BLEX: {100*metrics['blex_correct']/total_tokens:.2f}%")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='FlexiPipeTagger: Transformer-based FlexiPipe tagger with fine-tuning support'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train, tag, analyze, or calculate-accuracy')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the tagger')
    train_parser.add_argument('--data-dir', type=Path,
                             help='UD treebank directory (automatically finds *-ud-train.conllu, *-ud-dev.conllu, *-ud-test.conllu)')
    train_parser.add_argument('--train-dir', type=Path,
                             help='Directory containing CoNLL-U training files (alternative to --data-dir)')
    train_parser.add_argument('--dev-dir', type=Path,
                               help='Directory containing CoNLL-U development files (alternative to --data-dir)')
    train_parser.add_argument('--bert-model', default='bert-base-multilingual-cased',
                             help='BERT base model to use (default: bert-base-multilingual-cased, supports 104 languages). Language-specific models (e.g., bert-base-german-cased) often perform better for specific languages.')
    # Default: train everything (tokenizer, tagger, parser, lemmatizer). Use --no-* flags to disable components
    train_parser.add_argument('--no-tokenizer', dest='train_tokenizer', action='store_false', default=True,
                             help='Disable tokenizer training (default: enabled)')
    train_parser.add_argument('--no-tagger', dest='train_tagger', action='store_false', default=True,
                             help='Disable tagger training (default: enabled)')
    train_parser.add_argument('--no-parser', dest='train_parser', action='store_false', default=True,
                             help='Disable parser training (default: enabled)')
    train_parser.add_argument('--no-lemmatizer', dest='train_lemmatizer', action='store_false', default=True,
                             help='Disable lemmatizer training (default: enabled)')
    train_parser.add_argument('--no-normalizer', dest='train_normalizer', action='store_false', default=True,
                             help='Disable normalizer training (default: enabled, auto-detected from data)')
    train_parser.add_argument('--normalization-attr', default='reg',
                             help='TEITOK attribute name for normalization (default: reg, can be nform). Also reads Reg= from CoNLL-U MISC column.')
    train_parser.add_argument('--expan', default='expan',
                             help='TEITOK attribute name or CoNLL-U MISC key for expansion (default: expan; older projects may use fform/Exp).')
    train_parser.add_argument('--xpos-attr', default='xpos',
                             help='TEITOK attribute name(s) for XPOS (default: xpos). For inheritance, use comma-separated values like "pos,msd".')
    train_parser.add_argument('--output-dir', type=Path, default=Path('models/flexipipe'),
                             help='Output directory for trained model')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Training batch size (effective batch = batch_size * gradient_accumulation_steps)')
    train_parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                             help='Number of gradient accumulation steps (default: 2, effective batch = batch_size * this)')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5,
                             help='Learning rate')
    train_parser.add_argument('--num-epochs', type=int, default=3,
                             help='Number of training epochs')
    
    # Tag mode
    tag_parser = subparsers.add_parser('tag', help='Tag sentences')
    tag_parser.add_argument('input', type=Path, help='Input file (CoNLL-U, TEITOK XML, or plain text)')
    tag_parser.add_argument('--output', type=Path, help='Output file (default: stdout)')
    tag_parser.add_argument('--format', choices=['conllu', 'teitok', 'plain', 'text', 'raw', 'auto'],
                           help='Input format (auto-detected from file extension if not specified; use "raw" for unsegmented text)')
    tag_parser.add_argument('--output-format', choices=['conllu', 'plain', 'text', 'plain-tagged'],
                           help='Output format (defaults to input format or conllu)')
    tag_parser.add_argument('--segment', action='store_true',
                           help='Segment raw text into sentences (for plain/raw text input)')
    tag_parser.add_argument('--tokenize', action='store_true',
                           help='Tokenize sentences into words (for plain/raw text input)')
    tag_parser.add_argument('--model', type=Path, help='Path to trained model')
    tag_parser.add_argument('--bert-model', default='bert-base-multilingual-cased',
                           help='BERT base model if no trained model (default: bert-base-multilingual-cased, supports 104 languages). Language-specific models (e.g., bert-base-german-cased) often perform better.')
    tag_parser.add_argument('--vocab', type=Path, 
                           help='Vocabulary file (JSON) for tuning to local corpus. Format: {"word": {"upos": "...", "xpos": "...", "feats": "...", "lemma": "..."}, "word:xpos": {"lemma": "..."}}')
    tag_parser.add_argument('--vocab-priority', action='store_true',
                           help='Give vocabulary priority over model predictions for all tasks (UPOS/XPOS/FEATS/LEMMA). Useful for tuning to local corpus without retraining.')
    tag_parser.add_argument('--respect-existing', action='store_true', default=True,
                           help='Respect existing annotations in input (default: True)')
    tag_parser.add_argument('--no-respect-existing', dest='respect_existing', action='store_false',
                           help='Ignore existing annotations')
    tag_parser.add_argument('--parse', action='store_true',
                           help='Run parsing (predict head and deprel). Requires model trained with --train-parser.')
    tag_parser.add_argument('--tag-only', action='store_true',
                           help='Only tag (UPOS/XPOS/FEATS), skip parsing')
    tag_parser.add_argument('--parse-only', action='store_true',
                           help='Only parse (assumes tags already exist), skip tagging')
    tag_parser.add_argument('--lemma-method', choices=['bert', 'similarity', 'auto'], default='auto',
                           help='Lemmatization method: "bert" (use model predictions), "similarity" (use vocabulary/similarity matching), or "auto" (try BERT first, fallback to similarity). For LRL/historic texts with orthographic variation, "similarity" often outperforms BERT.')
    tag_parser.add_argument('--normalize', action='store_true',
                           help='Normalize orthographic variants (e.g., "mediaeval" -> "medieval"). Conservative by default to avoid over-normalization.')
    tag_parser.add_argument('--no-conservative-normalization', dest='conservative_normalization', action='store_false',
                           help='Disable conservative normalization (use more aggressive normalization). Warning: may cause over-normalization.')
    tag_parser.add_argument('--normalization-attr', default='reg',
                           help='TEITOK attribute name for normalization (default: reg, can be nform). Also used when reading CoNLL-U MISC Reg=')
    tag_parser.add_argument('--expan', default='expan',
                           help='TEITOK attribute name or CoNLL-U MISC key for expansion (default: expan; older projects may use fform/Exp).')
    tag_parser.add_argument('--xpos-attr', default='xpos',
                           help='TEITOK attribute name(s) for XPOS (default: xpos). For inheritance, use comma-separated values like "pos,msd".')
    tag_parser.add_argument('--tag-on-normalized', action='store_true',
                           help='Tag on normalized form instead of original orthography. Requires --normalize.')
    tag_parser.add_argument('--split-contractions', action='store_true',
                           help='Split contractions (e.g., "destas" -> "de estas"). Useful for historic texts where more things are written together.')
    tag_parser.add_argument('--aggressive-contraction-splitting', action='store_true',
                           help='Use more aggressive contraction splitting patterns for historic texts. Requires --split-contractions.')
    tag_parser.add_argument('--normalization-suffixes', type=Path,
                           help='JSON file with list of suffixes to project normalization to inflected forms (language-specific). If omitted, suffixes are derived from data.')
    tag_parser.add_argument('--language', type=str,
                           help='Language code for language-specific contraction rules (e.g., "es" for Spanish, "pt" for Portuguese, "ltz" or "lb" for Luxembourgish). Enables rule-based splitting for modern languages.')
    tag_parser.add_argument('--debug', action='store_true',
                           help='Enable debug output (prints progress after each sentence)')
    tag_parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                           help='Anchor for learning inflection suffixes from lemma: compare lemma to reg, form, or both (default: both).')
    
    # Calculate accuracy mode
    acc_parser = subparsers.add_parser('calculate-accuracy', help='Calculate accuracy metrics')
    acc_parser.add_argument('gold', type=Path, help='Gold standard file')
    acc_parser.add_argument('pred', type=Path, help='Predicted file')
    acc_parser.add_argument('--format', choices=['conllu', 'teitok', 'plain', 'text'], default='conllu',
                          help='File format (auto-detected from extension if not specified)')
    
    # Analyze mode
    analyze_parser = subparsers.add_parser('analyze', help='Analyze resources and derived artifacts')
    analyze_parser.add_argument('--model', type=Path, help='Path to trained model (reads model_vocab.json if present)')
    analyze_parser.add_argument('--vocab', type=Path, help='Vocabulary JSON to analyze (overrides model vocab)')
    analyze_parser.add_argument('--normalization-suffixes', type=Path,
                               help='External suffix list (JSON). If provided, reported as external and used as override in derivation.')
    analyze_parser.add_argument('--expan', default='expan',
                               help='TEITOK attribute name or CoNLL-U MISC key for expansion (default: expan; older projects may use fform/Exp).')
    analyze_parser.add_argument('--xpos-attr', default='xpos',
                               help='TEITOK attribute name(s) for XPOS (default: xpos). For inheritance, use comma-separated values like "pos,msd".')
    analyze_parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                               help='Anchor for deriving inflection suffixes from lemma (default: both).')
    analyze_parser.add_argument('--output', type=Path, help='Write analysis JSON to this file (default: stdout)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    if args.mode == 'train':
        # Handle --data-dir or --train-dir/--dev-dir
        train_files = []
        dev_files = None
        
        if args.data_dir:
            # Use standard UD treebank naming convention
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
                sys.exit(1)
            
            # Find standard UD files: *-ud-train.conllu, *-ud-dev.conllu
            train_files = list(data_dir.glob('*-ud-train.conllu'))
            dev_files_list = list(data_dir.glob('*-ud-dev.conllu'))
            
            if not train_files:
                print(f"Error: No *-ud-train.conllu file found in {data_dir}", file=sys.stderr)
                print(f"Found files: {list(data_dir.glob('*.conllu'))}", file=sys.stderr)
                sys.exit(1)
            
            if dev_files_list:
                dev_files = dev_files_list
            else:
                print(f"Warning: No *-ud-dev.conllu file found in {data_dir}, training without dev set", file=sys.stderr)
        
        elif args.train_dir:
            # Legacy mode: use directories
            train_files = list(args.train_dir.glob('*.conllu'))
            if not train_files:
                print(f"Error: No .conllu files found in {args.train_dir}", file=sys.stderr)
                sys.exit(1)
            
            if args.dev_dir:
                dev_files = list(args.dev_dir.glob('*.conllu'))
        else:
            print("Error: Either --data-dir or --train-dir must be specified", file=sys.stderr)
            sys.exit(1)
        
        # Gradient accumulation steps (default to 2 for MPS memory efficiency)
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 2)
        
        # Configure expansion key for CoNLL-U parsing
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        # Configure CoNLL-U expansion key and TEITOK attribute fallbacks
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        load_teitok_xml._xpos_attr = getattr(args, 'xpos_attr', 'xpos')
        load_teitok_xml._expan_attr = getattr(args, 'expan', 'expan')
        config = FlexiPipeConfig(
            bert_model=args.bert_model,
            train_tokenizer=args.train_tokenizer,
            train_tagger=args.train_tagger,
            train_parser=args.train_parser,
            train_lemmatizer=getattr(args, 'train_lemmatizer', True),  # Default True
            output_dir=str(args.output_dir),
            batch_size=args.batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        )
        
        # Build vocabulary
        vocab = build_vocabulary(train_files + (dev_files or []))
        print(f"Built vocabulary with {len(vocab)} entries", file=sys.stderr)
        
        tagger = FlexiPipeTagger(config, vocab)
        tagger.train(train_files, dev_files)
        
        # Save model
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # TODO: Save model after training
        print(f"Training complete. Model framework ready (full implementation pending).", file=sys.stderr)
    
    elif args.mode == 'tag':
        # Determine parse/tag settings
        parse_enabled = args.parse
        tag_only = args.tag_only
        parse_only = args.parse_only
        
        # If --tag-only is set, disable parsing
        if tag_only:
            parse_enabled = False
        
        # If --parse-only is set, enable parsing but disable tagging (will be handled in tag method)
        if parse_only:
            parse_enabled = True
        
        # Configure expansion key for CoNLL-U parsing
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        # Configure CoNLL-U expansion key and TEITOK attribute fallbacks
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        load_teitok_xml._xpos_attr = getattr(args, 'xpos_attr', 'xpos')
        load_teitok_xml._expan_attr = getattr(args, 'expan', 'expan')
        config = FlexiPipeConfig(
            bert_model=args.bert_model,
            respect_existing=args.respect_existing,
            parse=parse_enabled,
            tag_only=tag_only,
            parse_only=parse_only,
            vocab_priority=getattr(args, 'vocab_priority', False),
            lemma_method=getattr(args, 'lemma_method', 'auto'),
            normalize=getattr(args, 'normalize', False),
            conservative_normalization=getattr(args, 'conservative_normalization', True),
            train_normalizer=getattr(args, 'train_normalizer', True),
            normalization_attr=getattr(args, 'normalization_attr', 'reg'),
            tag_on_normalized=getattr(args, 'tag_on_normalized', False),
            split_contractions=getattr(args, 'split_contractions', False),
            aggressive_contraction_splitting=getattr(args, 'aggressive_contraction_splitting', False),
            language=getattr(args, 'language', None),
            debug=getattr(args, 'debug', False),
            normalization_suffixes_file=getattr(args, 'normalization_suffixes', None),
            lemma_anchor=getattr(args, 'lemma_anchor', 'both'),
        )
        
        vocab = {}
        transition_probs = None
        vocab_metadata = None
        if args.vocab:
            with open(args.vocab, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Handle new format with metadata/vocab/transitions structure
            if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                vocab = vocab_data.get('vocab', {})
                transition_probs = vocab_data.get('transitions', None)
                vocab_metadata = vocab_data.get('metadata', None)
                if vocab_metadata:
                    print(f"Loaded vocabulary from corpus: {vocab_metadata.get('corpus_name', 'unknown')}", file=sys.stderr)
                    print(f"  Created: {vocab_metadata.get('creation_date', 'unknown')}", file=sys.stderr)
                    if vocab_metadata.get('vocab_stats'):
                        stats = vocab_metadata['vocab_stats']
                        print(f"  Entries: {stats.get('total_entries', 0)} words, {stats.get('total_analyses', 0)} analyses", file=sys.stderr)
                    if transition_probs:
                        print(f"  Transition probabilities available for Viterbi tagging", file=sys.stderr)
            else:
                # Old format: just vocab dict
                vocab = vocab_data
        
        tagger = FlexiPipeTagger(config, vocab, model_path=args.model if args.model else None, transition_probs=transition_probs)
        if args.model:
            tagger.load_model(args.model)
        
        # Auto-detect format from file extension if not specified
        input_format = args.format
        if not input_format:
            input_ext = args.input.suffix.lower()
            if input_ext == '.xml':
                input_format = 'teitok'
            elif input_ext == '.conllu' or input_ext == '.conll':
                input_format = 'conllu'
            else:
                # Default to plain text for unknown extensions
                input_format = 'plain'
        
        # Determine output format
        output_format = args.output_format or input_format
        # If no explicit output format and input is plain/raw text, default to CoNLL-U for tagged output
        if not args.output_format and (input_format == 'plain' or input_format == 'raw'):
            output_format = 'conllu'  # Default to CoNLL-U for tagged output
        
        # Tag the input (don't write output yet, we'll use the correct format)
        # Auto-enable segment/tokenize for 'raw' format
        segment = args.segment or (input_format == 'raw')
        tokenize = args.tokenize or (input_format == 'raw')
        if getattr(args, 'debug', False):
            print(f"[DEBUG] main: Calling tag() with input={args.input}, format={input_format}, segment={segment}, tokenize={tokenize}", file=sys.stderr)
        tagged = tagger.tag(args.input, None, input_format, segment=segment, tokenize=tokenize)
        if getattr(args, 'debug', False):
            print(f"[DEBUG] main: tag() returned {len(tagged)} sentences", file=sys.stderr)
        
        # Write output with the specified format
        if args.output:
            if getattr(args, 'debug', False):
                print(f"[DEBUG] main: Writing to file: {args.output}", file=sys.stderr)
            tagger.write_output(tagged, args.output, output_format)
            print(f"Output written to: {args.output.absolute()}", file=sys.stderr)
        else:
            # Write to stdout (no --output specified)
            if getattr(args, 'debug', False):
                print(f"[DEBUG] main: Writing to stdout", file=sys.stderr)
            tagger.write_output(tagged, None, output_format)
    
    elif args.mode == 'calculate-accuracy':
        # Auto-detect format from file extension if not specified
        format_type = args.format
        if not format_type or format_type == 'conllu':
            # Try to auto-detect from file extension
            gold_ext = args.gold.suffix.lower()
            pred_ext = args.pred.suffix.lower()
            
            if gold_ext == '.xml' or pred_ext == '.xml':
                format_type = 'teitok'
            elif gold_ext in ('.conllu', '.conll') or pred_ext in ('.conllu', '.conll'):
                format_type = 'conllu'
            elif gold_ext in ('.txt', '.text') or pred_ext in ('.txt', '.text'):
                format_type = 'plain'
            else:
                format_type = 'conllu'  # Default
        
        config = FlexiPipeConfig()
        tagger = FlexiPipeTagger(config)
        tagger.calculate_accuracy(args.gold, args.pred, format_type)
    elif args.mode == 'analyze':
        # Build config and load vocab
        # Configure expansion key for CoNLL-U parsing
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        # Configure CoNLL-U expansion key and TEITOK attribute fallbacks
        set_conllu_expansion_key(getattr(args, 'expan', 'expan'))
        load_teitok_xml._xpos_attr = getattr(args, 'xpos_attr', 'xpos')
        load_teitok_xml._expan_attr = getattr(args, 'expan', 'expan')
        config = FlexiPipeConfig(
            normalize=True,
            conservative_normalization=True,
            normalization_suffixes_file=getattr(args, 'normalization_suffixes', None),
            lemma_anchor=getattr(args, 'lemma_anchor', 'both'),
            train_tokenizer=False,
            train_tagger=False,
            train_parser=False,
            train_lemmatizer=False,
            train_normalizer=False,
        )

        vocab = {}
        if getattr(args, 'vocab', None):
            with open(args.vocab, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                vocab = vocab_data.get('vocab', {})
            else:
                vocab = vocab_data
        elif getattr(args, 'model', None) and Path(args.model).exists():
            model_vocab_file = Path(args.model) / 'model_vocab.json'
            if model_vocab_file.exists():
                with open(model_vocab_file, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                    vocab = vocab_data.get('vocab', {})
                else:
                    vocab = vocab_data

        tagger = FlexiPipeTagger(config, vocab=vocab, model_path=args.model if getattr(args, 'model', None) else None)
        if not tagger.vocab:
            tagger.vocab = vocab
        tagger._build_normalization_inflection_suffixes()
        suffixes = tagger.inflection_suffixes or []

        analysis = {
            'lemma_anchor': config.lemma_anchor,
            'source': 'external' if getattr(args, 'normalization_suffixes', None) else 'derived',
            'num_suffixes': len(suffixes),
            'suffixes': suffixes,
        }

        if getattr(args, 'output', None):
            with open(args.output, 'w', encoding='utf-8') as out:
                json.dump(analysis, out, ensure_ascii=False, indent=2)
            print(f"Wrote suffix analysis to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

