"""
Data Loading module for FlexiPipe.
"""
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

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
            # Also extract OrigForm= for transpositional parsing
            misc = parts[9] if len(parts) > 9 else '_'
            if misc and misc != '_':
                # Parse MISC column for Normalized= or Reg= (normalization)
                misc_parts = misc.split('|')
                norm_form = '_'
                expan_form = '_'
                orig_form = '_'
                for misc_part in misc_parts:
                    if misc_part.startswith('Normalized='):
                        norm_form = misc_part[11:]  # Extract value after "Normalized="
                    elif misc_part.startswith('Reg='):
                        norm_form = misc_part[4:]  # Extract value after "Reg=" (backward compatibility)
                    elif misc_part.startswith('OrigForm='):
                        orig_form = misc_part[9:]  # Extract value after "OrigForm="
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
                token['orig_form'] = orig_form if orig_form else '_'
            else:
                token['norm_form'] = '_'
                token['expan'] = '_'
                token['orig_form'] = '_'
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
from flexipipe.tokenizer import tokenize_words_ud_style



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



