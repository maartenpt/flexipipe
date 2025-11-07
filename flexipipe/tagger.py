"""
Tagger module for FlexiPipe.
"""
import sys
import os
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

from flexipipe.config import FlexiPipeConfig
from flexipipe.utils import TRANSFORMERS_AVAILABLE, get_device, TRANSFORMERS_IMPORT_ERROR
from flexipipe.viterbi import viterbi_tag_sentence
from flexipipe.lemmatizer import build_lemmatization_patterns, apply_lemmatization_patterns
from flexipipe.normalization import normalize_word
from flexipipe.vocabulary import find_similar_words

# Import data loading functions from core
from flexipipe.core import (
    load_conllu_file,
    load_teitok_xml,
    load_plain_text,
    segment_sentences,
    parse_conllu_simple,
    CONLLU_EXPANSION_KEYS,
)
# Import tokenization from tokenizer module
from flexipipe.tokenizer import tokenize_words_ud_style, tokenize
# Import vocabulary-based parser
from flexipipe.parser import parse_sentence

if TYPE_CHECKING:
    if TRANSFORMERS_AVAILABLE:
        from transformers import PreTrainedTokenizer, BertModel, Trainer, TrainingArguments
        from datasets import Dataset

# Conditional imports for transformers (only if available)
if TRANSFORMERS_AVAILABLE:
    try:
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
        import torch
        from torch import nn
        from flexipipe.models import MultiTaskFlexiPipeTagger, BiaffineAttention
        from flexipipe.trainer import MultiTaskTrainer
    except ImportError:
        PreTrainedTokenizer = None
        BertModel = None
        Trainer = None
        TrainingArguments = None
        Dataset = None
        torch = None
        nn = None
else:
    PreTrainedTokenizer = None
    BertModel = None
    Trainer = None
    TrainingArguments = None
    Dataset = None
    torch = None
    nn = None

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
        # Extract tagset from metadata if available
        self.tagset_def = None
        if vocab_metadata and 'tagset' in vocab_metadata:
            self.tagset_def = vocab_metadata['tagset']
        # Tagset can also be set directly (from CLI --tagset argument)
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
            if self.config.verbose:
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
            if self.config.debug:
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
                
                if self.config.debug:
                    print(f"Loaded training configuration: BERT={self.config.bert_model}, "
                          f"Tokenizer={self.config.train_tokenizer}, Tagger={self.config.train_tagger}, "
                          f"Parser={self.config.train_parser}, Lemmatizer={self.config.train_lemmatizer}, "
                          f"Normalizer={self.config.train_normalizer}", file=sys.stderr)
                elif self.config.verbose:
                    # Brief summary for verbose mode
                    components = []
                    if self.config.train_tagger:
                        components.append("tagger")
                    if self.config.train_parser:
                        components.append("parser")
                    if self.config.train_lemmatizer:
                        components.append("lemmatizer")
                    if self.config.train_normalizer:
                        components.append("normalizer")
                    comp_str = ", ".join(components) if components else "none"
                    print(f"  - Model: {self.config.bert_model} ({comp_str})", file=sys.stderr)
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
                if self.config.debug:
                    print(f"Loaded model vocabulary with {len(self.model_vocab)} entries", file=sys.stderr)
            else:
                self.model_vocab = {}
                print("Warning: No model_vocab.json found, using empty model vocabulary", file=sys.stderr)
            
            # Merge vocabularies: model vocab + external vocab (external overrides model)
            self.vocab = self.model_vocab.copy()
            self.vocab.update(self.external_vocab)  # External vocab overrides model vocab
            if self.external_vocab:
                if self.config.debug:
                    print(f"Merged vocabularies: {len(self.model_vocab)} model entries + {len(self.external_vocab)} external entries = {len(self.vocab)} total", file=sys.stderr)
                elif self.config.verbose:
                    print(f"  - Using model vocabulary ({len(self.vocab)} entries)", file=sys.stderr)
            else:
                if self.config.debug:
                    print(f"Using model vocabulary: {len(self.vocab)} entries", file=sys.stderr)
                elif self.config.verbose:
                    print(f"  - Using model vocabulary ({len(self.vocab)} entries)", file=sys.stderr)
            # Report lexicon in verbose mode
            if self.config.verbose and self.lexicon:
                lexicon_count = len(self.lexicon)
                print(f"  - Using external lexicon (OOV fallback - {lexicon_count} entries)", file=sys.stderr)
            
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
                if self.config.debug:
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
            if self.config.debug:
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
        
        if self.config.verbose:
            # Check if input_file is a temporary file from stdin
            if hasattr(self, '_is_stdin') and self._is_stdin:
                print(f"Processing: Standard input", file=sys.stderr)
            else:
                print(f"Processing: {input_file}", file=sys.stderr)
            if segment:
                print(f"  - Sentence segmentation: enabled", file=sys.stderr)
            if tokenize:
                method = self.config.tokenization_method
                if method == 'auto':
                    # Determine actual method that will be used
                    # Check if we have a tokenizer (from model) or if we'll actually use BERT
                    if self.tokenizer:
                        method = "BERT"
                    elif not self.model:
                        # Viterbi mode (no model) - check if we'll use BERT tokenizer
                        if TRANSFORMERS_AVAILABLE:
                            # Will load BERT tokenizer (not a trained model, just tokenization)
                            method = "BERT tokenizer"
                        else:
                            method = "UD-style"
                    elif TRANSFORMERS_AVAILABLE and self.config.bert_model:
                        # Model mode - will try to load BERT tokenizer
                        method = "BERT (will load tokenizer)"
                    else:
                        method = "UD-style"
                elif method == 'bert':
                    if self.tokenizer:
                        method = "BERT"
                    elif TRANSFORMERS_AVAILABLE:
                        method = "BERT (will load tokenizer)"
                    else:
                        method = "BERT (transformers not available, will fail)"
                print(f"  - Tokenization: {method}", file=sys.stderr)
        
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
                    if self.config.verbose:
                        if self.external_vocab:
                            vocab_count = len(self.external_vocab)
                            print(f"  - Using external vocabulary ({vocab_count} entries)", file=sys.stderr)
                        if self.lexicon:
                            lexicon_count = len(self.lexicon)
                            print(f"  - Using external lexicon (OOV fallback - {lexicon_count} entries)", file=sys.stderr)
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
                    # Use configured tokenization method
                    words = tokenize(
                        original_sent_text,
                        method=self.config.tokenization_method,
                        tokenizer=self.tokenizer,
                        bert_model=self.config.bert_model
                    )
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
                # Use configured tokenization method
                words = tokenize(
                    original_sent_text,
                    method=self.config.tokenization_method,
                    tokenizer=self.tokenizer,
                    bert_model=self.config.bert_model
                )
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
            if self.config.verbose:
                steps = []
                if self.config.normalize:
                    strategy = "similarity" if not self.config.conservative_normalization else "conservative"
                    steps.append(f"normalization ({strategy})")
                if self.config.split_contractions:
                    steps.append("contraction splitting")
                if steps:
                    print(f"  - Preprocessing: {' + '.join(steps)}", file=sys.stderr)
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
            # Timing only in verbose mode (report total pipeline time)
            if self.config.verbose:
                total_elapsed = time.time() - start_time
                total_tok_per_sec = total_tokens / total_elapsed if total_elapsed > 0 else 0
                print(f"Pipeline complete: {len(tagged_sentences)} sentences, {total_tokens} tokens in {total_elapsed:.2f}s ({total_tok_per_sec:.1f} tok/s)", file=sys.stderr)
            if self.config.debug:
                print(f"[DEBUG] Returning {len(tagged_sentences)} tagged sentences from normalization-only mode", file=sys.stderr)
            # Return early - don't continue to model-based processing
            # Write output will be handled by caller
            return tagged_sentences
        elif use_viterbi:
            # Viterbi tagging mode: use vocabulary-based tagging with transition probabilities
            if self.config.debug:
                print(f"[DEBUG] Entering Viterbi tagging mode", file=sys.stderr)
            if self.config.verbose:
                tag_key = "XPOS" if self.config.use_xpos_for_tagging else "UPOS+FEATS"
                print(f"  - Tagging: Viterbi algorithm (using {tag_key})", file=sys.stderr)
                # Report UD guessing if enabled
                if self.config.guess_ud and getattr(self, 'tagset_def', None):
                    tagset_file = getattr(self, '_tagset_file', 'tagset')
                    print(f"  - UD guessing: deriving UPOS/FEATS from XPOS using {tagset_file}", file=sys.stderr)
                # Report lemmatization strategy
                lemma_strategy = []
                if self.vocab:
                    lemma_strategy.append("vocabulary")
                if self.lemmatization_patterns:
                    lemma_strategy.append("pattern-based")
                if self.lexicon:
                    lemma_strategy.append("lexicon (OOV fallback)")
                if lemma_strategy:
                    print(f"  - Lemmatization: {' + '.join(lemma_strategy)}", file=sys.stderr)
                else:
                    print(f"  - Lemmatization: form (no strategy available)", file=sys.stderr)
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
                    
                    # Get normalization from preprocessing (normalization is handled in _preprocess_sentences)
                    # This ensures we use the normalized form for lemma lookup
                    norm_form = None
                    if 'norm_form' in token and token.get('norm_form') and token.get('norm_form') != '_':
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
                                if xpos and xpos != '_':
                                    for analysis in entry:
                                        if analysis.get('xpos') == xpos:
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
                                if xpos and xpos != '_':
                                        if analysis.get('xpos') == xpos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis
                                        best_entry = analysis
                                
                                if not best_entry and upos and upos != '_':
                                        if analysis.get('upos') == upos:
                                            count = analysis.get('count', 0)
                                            if count > best_count:
                                                best_count = count
                                                best_entry = analysis
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
                    if norm_form and norm_form != '_':
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
            # Apply fallback parser if parsing is enabled
            if self.config.parse:
                parser_applied = False
                for sentence in tagged_sentences:
                    # Check if any token has head/deprel from model (shouldn't happen in Viterbi mode, but check anyway)
                    has_model_parse = any(
                        token.get('head', '_') not in ('_', 0, '0') or 
                        token.get('deprel', '_') != '_'
                        for token in sentence
                    )
                    
                    if not has_model_parse:
                        # Use vocabulary-based fallback parser
                        parsed_sentence = parse_sentence(
                            sentence,
                            vocab=self.vocab,
                            transition_probs=self.transition_probs,
                            use_xpos=self.config.use_xpos_for_tagging,
                            method='transition',  # Use transition-based parser by default
                            tagset_def=getattr(self, 'tagset_def', None),
                            debug=self.config.debug
                        )
                        # Update sentence with parsed head and deprel
                        for i, token in enumerate(sentence):
                            if i < len(parsed_sentence):
                                parsed_token = parsed_sentence[i]
                                token['head'] = parsed_token.get('head', '_')
                                token['deprel'] = parsed_token.get('deprel', '_')
                        parser_applied = True
                
                if parser_applied:
                    if self.config.verbose:
                        # Determine parser strategy
                        if self.transition_probs and 'deprel' in self.transition_probs:
                            print(f"  - Parsing: Transition-based parser (MaltParser-style)", file=sys.stderr)
                        elif self.vocab:
                            print(f"  - Parsing: Enhanced heuristic parser with learned preferences", file=sys.stderr)
                        else:
                            print(f"  - Parsing: Basic heuristic parser", file=sys.stderr)
                    if self.config.debug:
                        print("[DEBUG] Applied vocabulary-based fallback parser (Viterbi mode)", file=sys.stderr)
            
            # Timing and statistics only in verbose mode (report total pipeline time)
            if self.config.verbose:
                total_elapsed = time.time() - start_time
                total_tok_per_sec = total_tokens / total_elapsed if total_elapsed > 0 else 0
                print(f"Pipeline complete: {len(tagged_sentences)} sentences, {total_tokens} tokens in {total_elapsed:.2f}s ({total_tok_per_sec:.1f} tok/s)", file=sys.stderr)
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
            if self.config.verbose:
                # Report BERT model tagging/parsing
                tasks = []
                if not self.config.parse_only:
                    tasks.append("tagging")
                if self.config.parse:
                    tasks.append("parsing")
                if tasks:
                    print(f"  - Tagging: BERT model ({' + '.join(tasks)})", file=sys.stderr)
                # Report lemmatization strategy
                lemma_strategy = []
                if self.config.train_lemmatizer and hasattr(self, 'model') and self.model:
                    lemma_strategy.append("BERT model")
                if self.config.use_vocabulary and self.vocab:
                    if self.config.vocab_priority:
                        lemma_strategy.append("vocabulary (priority)")
                    elif self.config.confidence_threshold < 1.0:
                        lemma_strategy.append(f"vocabulary (confidence < {self.config.confidence_threshold})")
                    else:
                        lemma_strategy.append("vocabulary (fallback)")
                if self.lemmatization_patterns:
                    lemma_strategy.append("pattern-based")
                if self.lexicon:
                    lemma_strategy.append("lexicon (OOV fallback)")
                if lemma_strategy:
                    print(f"  - Lemmatization: {' + '.join(lemma_strategy)}", file=sys.stderr)
                else:
                    print(f"  - Lemmatization: form (no strategy available)", file=sys.stderr)
            
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
                            else:
                                # If FEATS weren't trained, feats stays '_' and will fall back to vocabulary later
                                feats = '_'
                            
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
                                # Model parser not available - will use fallback parser later
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
                            # Head and deprel will be set by fallback parser if parsing is enabled
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
        
        # Apply fallback parser if parsing is enabled and model parser didn't provide results
        if self.config.parse:
            parser_applied = False
            for sentence in tagged_sentences:
                # Check if any token has head/deprel from model
                has_model_parse = any(
                    token.get('head', '_') not in ('_', 0, '0') or 
                    token.get('deprel', '_') != '_'
                    for token in sentence
                )
                
                if not has_model_parse:
                    # Use vocabulary-based fallback parser
                    parsed_sentence = parse_sentence(
                        sentence,
                        vocab=self.vocab,
                        transition_probs=self.transition_probs,
                        use_xpos=self.config.use_xpos_for_tagging,
                        method='transition',  # Use transition-based parser by default
                        tagset_def=getattr(self, 'tagset_def', None)
                    )
                    # Update sentence with parsed head and deprel
                    for i, token in enumerate(sentence):
                        if i < len(parsed_sentence):
                            parsed_token = parsed_sentence[i]
                            token['head'] = parsed_token.get('head', '_')
                            token['deprel'] = parsed_token.get('deprel', '_')
                    parser_applied = True
            
            if parser_applied:
                if self.config.verbose:
                    # Determine parser strategy
                    if self.transition_probs and 'deprel' in self.transition_probs:
                        print(f"  - Parsing: Transition-based parser (MaltParser-style)", file=sys.stderr)
                    elif self.vocab:
                        print(f"  - Parsing: Enhanced heuristic parser with learned preferences", file=sys.stderr)
                    else:
                        print(f"  - Parsing: Basic heuristic parser", file=sys.stderr)
                if self.config.debug:
                    print("[DEBUG] Applied vocabulary-based fallback parser", file=sys.stderr)
        
        # Calculate final statistics (for model-based tagging)
        total_tokens = sum(len(s) for s in tagged_sentences)
        elapsed_time = time.time() - start_time
        tok_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        # Timing only in verbose mode (report total pipeline time)
        if self.config.verbose:
            total_elapsed = time.time() - start_time
            total_tok_per_sec = total_tokens / total_elapsed if total_elapsed > 0 else 0
            print(f"Pipeline complete: {len(tagged_sentences)} sentences, {total_tokens} tokens in {total_elapsed:.2f}s ({total_tok_per_sec:.1f} tok/s)", file=sys.stderr)
        
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
            # Check both normalized form (if normalization was applied) and original form
            word_is_in_vocab = False
            if hasattr(self, 'vocab') and self.vocab:
                form_lower = lemma_form.lower()
                # Check both exact case and lowercase for normalized form
                word_is_in_vocab = (lemma_form in self.vocab or form_lower in self.vocab or
                                   (xpos and f"{lemma_form}:{xpos}" in self.vocab) or
                                   (xpos and f"{form_lower}:{xpos}" in self.vocab))
                
                # If normalization was applied and normalized form is not in vocab, also check original form
                if norm_form and norm_form != '_' and not word_is_in_vocab:
                    form_orig_lower = form.lower()
                    word_is_in_vocab = (form in self.vocab or form_orig_lower in self.vocab or
                                       (xpos and f"{form}:{xpos}" in self.vocab) or
                                       (xpos and f"{form_orig_lower}:{xpos}" in self.vocab))
            
            # Try normalized form first (if normalization was applied)
            vocab_lemma = self._predict_from_vocab(lemma_form, 'lemma', xpos=xpos, upos=upos, feats=feats, debug=self.config.debug)
            
            # If normalization was applied and normalized form lookup failed, also try original form
            # This prevents similarity matching from finding wrong matches when normalized form is OOV
            if norm_form and norm_form != '_' and (vocab_lemma == '_' or vocab_lemma is None):
                if self.config.debug:
                    print(f"[DEBUG LEMMA] Normalized form '{lemma_form}' not found in vocab, trying original form '{form}'", file=sys.stderr)
                vocab_lemma_orig = self._predict_from_vocab(form, 'lemma', xpos=xpos, upos=upos, feats=feats, debug=self.config.debug)
                if vocab_lemma_orig != '_' and vocab_lemma_orig is not None:
                    vocab_lemma = vocab_lemma_orig
                    if self.config.debug:
                        print(f"[DEBUG LEMMA] Found lemma '{vocab_lemma}' from original form", file=sys.stderr)
            
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
        
        # Ensure we always return a string (never None)
        if final_lemma is None:
            final_lemma = form
            if self.config.debug:
                print(f"[DEBUG LEMMA] final_lemma was None, falling back to form: '{final_lemma}'", file=sys.stderr)
        
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
        
        return final_lemma if final_lemma else form
    
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
                                if debug_entry or debug:
                                    print(f"[DEBUG LEMMA]   MATCH found! xpos='{analysis_xpos}' == '{xpos}'", file=sys.stderr)
                                matches.append((count, analysis))
                
                    # Fallback to UPOS+FEATS if no XPOS match
                    if not matches and upos and upos != '_':
                        if debug_entry or debug:
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
                                    if feats_match:
                                        count = analysis.get('count', 0)
                                        if debug_entry or debug:
                                            print(f"[DEBUG LEMMA]   MATCH found! upos='{analysis_upos}' == '{upos}'", file=sys.stderr)
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
        # BUT: For lemmatization, similarity matching is very unreliable and should be avoided
        # Similarity matching often finds wrong matches (e.g., "seruicios" -> "rossarios" -> "rosario")
        # Only use similarity matching for lemmatization if we have no other option AND the match is very strong
        if field == 'lemma':
            # For lemmatization, similarity matching is too unreliable - prefer returning form
            # Only use it if we're really desperate and the match is very strong (high threshold)
            if debug:
                print(f"[DEBUG LEMMA] Similarity matching for lemmatization is unreliable, skipping", file=sys.stderr)
            return form_lower
        
        # For non-lemma fields (UPOS, XPOS, FEATS), similarity matching is more acceptable
        if debug:
            print(f"[DEBUG] Trying similarity matching for form '{form}' (field: {field})", file=sys.stderr)
        similar = find_similar_words(form, vocab_to_check, self.config.similarity_threshold)
        if similar:
            best_match = similar[0][0]
            if debug:
                print(f"[DEBUG] Found similar words (showing top 5):", file=sys.stderr)
                for i, (match_word, score) in enumerate(similar[:5]):
                    print(f"[DEBUG]   [{i}] '{match_word}' (similarity: {score:.3f})", file=sys.stderr)
            similar_entry = vocab_to_check[best_match]
            
            # Handle array format
            if isinstance(similar_entry, list):
                result = similar_entry[0].get(field, '_') if similar_entry else '_'
                if debug:
                    print(f"[DEBUG] Similar entry '{best_match}' has {len(similar_entry)} analyses, using first: '{result}'", file=sys.stderr)
            else:
                result = similar_entry.get(field, '_')
                if debug:
                    print(f"[DEBUG] Similar entry '{best_match}' result: '{result}'", file=sys.stderr)
            
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
            if use_stdout and self.config.verbose:
                print("Warning: No sentences to write", file=sys.stderr)
            elif not use_stdout:
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
                                    
                                    # Removed verbose token context debug messages
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
                        
                        # If --guess-ud is enabled and we have a tagset, convert XPOS to UPOS/FEATS
                        tagset_def = getattr(self, 'tagset_def', None)
                        guess_ud = getattr(self.config, 'guess_ud', False)
                        if self.config.debug and token_idx == 0:
                            print(f"[DEBUG UD] Starting UD guessing: guess_ud={guess_ud}, tagset_def={tagset_def is not None}", file=sys.stderr)
                        if guess_ud and tagset_def and xpos and xpos != '_':
                            try:
                                from flexipipe.tagset import xpos_to_upos_feats
                                guessed_upos, guessed_feats = xpos_to_upos_feats(xpos, tagset_def)
                                if self.config.debug and token_idx < 3:
                                    print(f"[DEBUG UD] Token {token_idx} '{form}': XPOS '{xpos}' -> UPOS='{guessed_upos}', FEATS='{guessed_feats}'", file=sys.stderr)
                                # Always use guessed values when --guess-ud is enabled
                                if guessed_upos and guessed_upos != '_':
                                    upos = guessed_upos
                                if guessed_feats and guessed_feats != '_':
                                    feats = guessed_feats
                            except Exception as e:
                                if self.config.debug:
                                    print(f"[DEBUG UD] ERROR: Failed to guess UPOS/FEATS for XPOS '{xpos}': {e}", file=sys.stderr)
                                    import traceback
                                    traceback.print_exc(file=sys.stderr)
                        
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
                        
                        # If --guess-ud is enabled and we have a tagset, convert XPOS to UPOS/FEATS
                        tagset_def = getattr(self, 'tagset_def', None)
                        guess_ud = getattr(self.config, 'guess_ud', False)
                        if self.config.debug and token_idx == 0:
                            print(f"[DEBUG UD] Starting UD guessing: guess_ud={guess_ud}, tagset_def={tagset_def is not None}", file=sys.stderr)
                        if guess_ud and tagset_def and xpos and xpos != '_':
                            try:
                                from flexipipe.tagset import xpos_to_upos_feats
                                guessed_upos, guessed_feats = xpos_to_upos_feats(xpos, tagset_def)
                                if self.config.debug and token_idx < 3:
                                    print(f"[DEBUG UD] Token {token_idx} '{form}': XPOS '{xpos}' -> UPOS='{guessed_upos}', FEATS='{guessed_feats}'", file=sys.stderr)
                                # Always use guessed values when --guess-ud is enabled
                                if guessed_upos and guessed_upos != '_':
                                    upos = guessed_upos
                                if guessed_feats and guessed_feats != '_':
                                    feats = guessed_feats
                            except Exception as e:
                                if self.config.debug:
                                    print(f"[DEBUG UD] ERROR: Failed to guess UPOS/FEATS for XPOS '{xpos}': {e}", file=sys.stderr)
                                    import traceback
                                    traceback.print_exc(file=sys.stderr)
                        
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
                # When writing to stdout, don't print anything unless verbose
                if self.config.debug:
                    print(f"[DEBUG] Wrote to stdout, flushing...", file=sys.stderr)
                sys.stdout.flush()
            # When writing to file, always print confirmation
            if not use_stdout:
                print(f"Output written to {output_file}", file=sys.stderr)
    
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


