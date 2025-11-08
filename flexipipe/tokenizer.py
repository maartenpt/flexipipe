"""
Tokenizer module for FlexiPipe.

Supports both rule-based UD-style tokenization and BERT-based tokenization
for languages that require subword tokenization (e.g., Chinese, Japanese).
"""
import re
from typing import List, Optional, Union, Dict
from pathlib import Path

# Try to import transformers for BERT-based tokenization
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    PreTrainedTokenizer = None


def tokenize_words_ud_style(text: str) -> List[str]:
    """
    Tokenize text into words using UD-style tokenization rules.
    
    This is a rule-based tokenizer that works well for space-separated languages
    (e.g., English, Spanish, French). For languages without spaces (Chinese, Japanese)
    or languages that benefit from subword tokenization, use `tokenize_with_bert()`.
    
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


def tokenize_with_bert(
    text: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
    return_offsets: bool = False
) -> Union[List[str], tuple[List[str], List[tuple[int, int]]]]:
    """
    Tokenize text using a BERT tokenizer.
    
    This is useful for:
    - Languages without spaces (Chinese, Japanese, Thai)
    - Languages that benefit from subword tokenization
    - When you want to align with a BERT model's tokenization
    
    For Chinese/Japanese, BERT tokenizers typically produce subword tokens.
    This function returns the token strings as produced by the tokenizer.
    For word-level tokenization of Chinese, consider using specialized tools
    (e.g., jieba for Chinese, mecab for Japanese) or fine-tuned tokenizers.
    
    Args:
        text: Sentence string to tokenize
        tokenizer: Pre-trained tokenizer instance (if None, will load from model_name)
        model_name: Name of the model to load tokenizer from (e.g., 'bert-base-chinese')
        return_offsets: If True, also return character offsets for each token
        
    Returns:
        List of token strings, or tuple of (tokens, offsets) if return_offsets=True
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library required for BERT tokenization. "
            "Install with: pip install transformers"
        )
    
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the text
    # For Chinese/Japanese, BERT tokenizers will produce subword tokens
    # This is appropriate for these languages as they don't have word boundaries
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=return_offsets,
        return_attention_mask=False
    )
    
    if return_offsets:
        # Get token strings and their offsets
        token_ids = encoded['input_ids']
        offsets = encoded.get('offset_mapping', [])
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # Clean up tokens: remove special token markers (## for subwords, etc.)
        # But keep the actual token strings
        cleaned_tokens = []
        cleaned_offsets = []
        for token, offset in zip(tokens, offsets):
            # Skip special tokens
            if token in tokenizer.special_tokens_map.values():
                continue
            # Remove ## prefix for subword tokens (BERT WordPiece style)
            # but keep the token itself
            if token.startswith('##'):
                token = token[2:]
            cleaned_tokens.append(token)
            cleaned_offsets.append(offset)
        return cleaned_tokens, cleaned_offsets
    else:
        # Just return token strings
        token_ids = encoded['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # Clean up tokens: remove special token markers
        cleaned_tokens = []
        for token in tokens:
            # Skip special tokens
            if token in tokenizer.special_tokens_map.values():
                continue
            # Remove ## prefix for subword tokens (BERT WordPiece style)
            if token.startswith('##'):
                token = token[2:]
            cleaned_tokens.append(token)
        return cleaned_tokens


def tokenize(
    text: str,
    method: str = "auto",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
    bert_model: Optional[str] = None
) -> List[str]:
    """
    Tokenize text using the specified method.
    
    Args:
        text: Sentence string to tokenize
        method: Tokenization method - "ud" (rule-based), "bert" (BERT-based), or "auto" (default: auto - uses BERT if available, otherwise UD)
        tokenizer: Pre-trained tokenizer instance (for BERT method)
        model_name: Name of the model to load tokenizer from (for BERT method, deprecated - use bert_model)
        bert_model: Name of the BERT model to load tokenizer from (for BERT method, e.g., 'bert-base-chinese')
        
    Returns:
        List of token strings
    """
    # Handle deprecated model_name parameter
    if model_name and not bert_model:
        bert_model = model_name
    
    # Smart "auto" mode: try BERT if transformers is available, otherwise use UD
    if method == "auto":
        if TRANSFORMERS_AVAILABLE:
            # Try to use BERT tokenization if available
            try:
                # Use provided tokenizer, or try to load from bert_model, or use default multilingual
                if tokenizer:
                    return tokenize_with_bert(text, tokenizer=tokenizer)
                elif bert_model:
                    return tokenize_with_bert(text, model_name=bert_model)
                else:
                    # Default to multilingual BERT if available
                    try:
                        return tokenize_with_bert(text, model_name="bert-base-multilingual-cased")
                    except Exception:
                        # Fallback to UD if BERT loading fails
                        return tokenize_words_ud_style(text)
            except Exception:
                # If BERT tokenization fails, fallback to UD
                return tokenize_words_ud_style(text)
        else:
            # No transformers available, use UD
            return tokenize_words_ud_style(text)
    elif method == "bert":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for BERT tokenization. "
                "Install with: pip install transformers"
            )
        return tokenize_with_bert(text, tokenizer=tokenizer, model_name=bert_model)
    else:
        # Default to UD-style tokenization
        return tokenize_words_ud_style(text)


def train_tokenizer(
    sentences: List[List[Dict]],
    vocab_size: int = 30000,
    model_type: str = "bert",
    output_dir: Optional[Path] = None
) -> PreTrainedTokenizer:
    """
    Train a custom tokenizer from a corpus.
    
    This is useful for domain-specific tokenization or low-resource languages.
    
    Args:
        sentences: List of sentences, where each sentence is a list of token dicts
        vocab_size: Size of the vocabulary to build
        model_type: Type of tokenizer ("bert", "roberta", etc.)
        output_dir: Directory to save the trained tokenizer
        
    Returns:
        Trained tokenizer instance
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library required for training tokenizers. "
            "Install with: pip install transformers"
        )
    
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    
    # Extract text from sentences
    texts = []
    for sentence in sentences:
        # Join tokens with spaces (or use original text if available)
        if sentence and '_original_text' in sentence[0]:
            texts.append(sentence[0]['_original_text'])
        else:
            forms = [token.get('form', '') for token in sentence]
            texts.append(' '.join(forms))
    
    # Initialize tokenizer
    if model_type == "bert":
        tokenizer_model = models.WordPiece(unk_token="[UNK]")
    elif model_type == "roberta":
        tokenizer_model = models.BPE()
    else:
        tokenizer_model = models.WordPiece(unk_token="[UNK]")
    
    tokenizer_obj = Tokenizer(tokenizer_model)
    
    # Set pre-tokenizer
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Train the tokenizer
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    tokenizer_obj.train_from_iterator(texts, trainer=trainer)
    
    # Wrap in PreTrainedTokenizer
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        model_max_length=512,
        padding_side="right",
        truncation_side="right"
    )
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
    
    return tokenizer

