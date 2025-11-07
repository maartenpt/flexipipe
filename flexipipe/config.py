"""
Configuration classes for FlexiPipe.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FlexiPipeConfig:
    """Configuration for FlexiPipe tagger."""
    bert_model: str = "bert-base-multilingual-cased"  # Multilingual by default (supports 104 languages)
    train_tokenizer: bool = True  # Default: train tokenizer
    train_tagger: bool = True  # Default: train tagger
    train_parser: bool = True  # Default: train parser (full pipeline)
    train_lemmatizer: bool = True  # Default: train lemmatizer
    max_length: int = 512
    batch_size: int = 16  # Reduced for MPS memory constraints
    gradient_accumulation_steps: int = 2  # Effective batch size = 16 * 2 = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5  # Increased from 3 - BERT fine-tuning often needs more epochs
    output_dir: str = "models/flexipipe"
    similarity_threshold: float = 0.7
    use_vocabulary: bool = True
    vocab_priority: bool = False  # If True, vocabulary takes priority over model predictions for all tasks
    confidence_threshold: float = 0.7  # If model confidence < threshold, use vocabulary predictions (confidence-based blending)
    respect_existing: bool = True
    lemma_method: str = 'auto'  # 'bert', 'similarity', or 'auto' (default: try BERT first, fallback to similarity)
    # Historic document processing (neotag replacement)
    normalize: bool = False  # Normalize orthographic variants (e.g., "mediaeval" -> "medieval")
    conservative_normalization: bool = True  # Only normalize if high confidence (avoid over-normalization)
    train_normalizer: bool = True  # Train normalizer if normalization data is present (auto-detected)
    normalization_attr: str = 'reg'  # TEITOK attribute for normalization (default: 'reg', can be 'nform')
    expansion_attr: str = 'expan'  # TEITOK attribute for expansion (default: 'expan', can be 'fform')
    tag_on_normalized: bool = False  # Tag on normalized form instead of original orthography
    split_contractions: bool = False  # Split contractions (e.g., "destas" -> "de estas")
    aggressive_contraction_splitting: bool = False  # More aggressive splitting for historic texts
    language: Optional[str] = None  # Language code for language-specific contraction rules (e.g., 'es', 'pt', 'ltz')
    # Normalization inflection suffixes
    normalization_suffixes_file: Optional[Path] = None  # Optional JSON file with suffix list for normalization inflections
    lemma_anchor: str = 'both'  # How to derive suffixes: 'reg' | 'form' | 'both'
    xpos_attr: str = 'xpos'  # TEITOK attribute(s) for XPOS (comma-separated fallbacks allowed)
    use_xpos_for_tagging: bool = False  # If True, use XPOS for tagging/lemmatization; if False (default), use UPOS+FEATS
    # Parsing configuration
    parse: bool = False  # Whether to run parsing (predict head/deprel)
    tag_only: bool = False  # Only tag (UPOS/XPOS/FEATS), skip parsing
    parse_only: bool = False  # Only parse (assumes tags already exist), skip tagging
    debug: bool = False  # Enable debug output
    verbose: bool = False  # Enable verbose output (shows pipeline steps and timing)
    tokenization_method: str = 'auto'  # Tokenization method: 'ud' (rule-based), 'bert' (BERT-based), or 'auto' (default: auto - uses BERT if available, otherwise UD)
    guess_ud: bool = False  # If True, output guessed UPOS and FEATS derived from XPOS using tagset definition

