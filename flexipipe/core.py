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


# Import tokenization from tokenizer module
from flexipipe.tokenizer import tokenize_words_ud_style, tokenize


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
                    # Contraction: collect split forms for the orthographic token
                    split_forms = []
                    dtok_tokens = []  # Store dtok tokens for later addition
                    
                    # First pass: collect all dtok forms
                    for dt in dtoks:
                        # Get dtok ID: try @id first, then @xml:id
                        dt_id = dt.get('id', '') or dt.get('{http://www.w3.org/XML/1998/namespace}id', '')
                        form = dt.get('form', '') or (dt.text or '').strip()
                        if form:
                            split_forms.append(form)
                            # Get normalization (try specified attr first, then common fallbacks)
                            nform = get_attr_with_fallback(dt, normalization_attr) or dt.get('reg', '') or dt.get('nform', '')
                            xpos_val = get_attr_with_fallback(dt, xpos_attr) or dt.get('xpos', '_')
                            expan_val = get_attr_with_fallback(dt, expan_attr) or dt.get('expan', '') or dt.get('fform', '')
                            
                            dtok_token = {
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
                            }
                            dtok_tokens.append(dtok_token)
                            token_num += 1
                    
                    # Add orthographic form ("im") with parts information for training
                    if len(split_forms) > 1:
                        ortho_form = tok.get('form', '') or (tok.text or '').strip()
                        if ortho_form:
                            sentence_tokens.append({
                                'id': token_num,
                                'form': ortho_form,
                                'norm_form': '_',
                                'lemma': '_',
                                'upos': '_',  # Contractions don't have a single UPOS
                                'xpos': '_',  # Contractions don't have a single XPOS
                                'feats': '_',
                                'head': '0',
                                'deprel': '_',
                                'tok_id': tok_id,
                                'parts': split_forms,  # Store split forms
                                'expan': '_',
                            })
                            token_num += 1
                    
                    # Add all dtok tokens (grammatical tokens)
                    sentence_tokens.extend(dtok_tokens)
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
        
        from flexipipe.tagger import FlexiPipeTagger
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
        
        from flexipipe.tagger import FlexiPipeTagger
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

        from flexipipe.tagger import FlexiPipeTagger
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

