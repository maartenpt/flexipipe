"""
Vocabulary module for FlexiPipe.
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

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


def load_unimorph_lexicon(file_path: Path, default_count: int = 1) -> Dict:
    """
    Load a UniMorph lexicon file and convert it to FlexiPipe vocabulary format.
    
    UniMorph format is tab-separated: lemma\tform\tfeatures
    Where features are in UniMorph format (e.g., "V;IND;SG;3;PRS" for verb, indicative, singular, 3rd person, present).
    Note: The lemma comes FIRST in UniMorph format, not the form.
    
    This function converts UniMorph features to UD FEATS format where possible.
    Note: UniMorph features are language-specific and may not always map cleanly to UD.
    
    Args:
        file_path: Path to UniMorph lexicon file (tab-separated: form\tlemma\tfeatures)
        default_count: Default count to use for entries (default: 1, since UniMorph doesn't have counts)
    
    Returns:
        Dictionary in FlexiPipe vocabulary format: {form: {upos, feats, lemma, count} or [{...}, ...]}
    """
    vocab = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                # Skip malformed lines
                continue
            
            # UniMorph format is: lemma\tform\tfeatures
            # NOT form\tlemma\tfeatures
            lemma = parts[0].strip()
            form = parts[1].strip()
            
            if not form or not lemma:
                continue
            
            # Parse UniMorph features (if present)
            features = parts[2].strip() if len(parts) > 2 else ''
            
            # Convert UniMorph features to UD FEATS format
            # UniMorph uses semicolon-separated features (e.g., "V;PST;3;SG")
            # UD uses pipe-separated key=value pairs (e.g., "Tense=Past|Person=3|Number=Sing")
            upos = '_'
            feats = '_'
            
            if features:
                # Basic mapping from UniMorph feature abbreviations to UD
                # This is a simplified mapping - full conversion would require language-specific rules
                unimorph_parts = [f.strip() for f in features.split(';') if f.strip()]
                ud_feats = []
                
                # Map common UniMorph POS tags to UPOS
                pos_mapping = {
                    'N': 'NOUN', 'V': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
                    'PRON': 'PRON', 'DET': 'DET', 'PREP': 'ADP', 'ADP': 'ADP',
                    'CONJ': 'CCONJ', 'SCONJ': 'SCONJ', 'NUM': 'NUM', 'PUNCT': 'PUNCT'
                }
                
                # Map common UniMorph features to UD FEATS
                feature_mapping = {
                    # Tense
                    'PST': 'Tense=Past', 'PRS': 'Tense=Pres', 'FUT': 'Tense=Fut',
                    # Person
                    '1': 'Person=1', '2': 'Person=2', '3': 'Person=3',
                    # Number
                    'SG': 'Number=Sing', 'PL': 'Number=Plur',
                    # Gender
                    'MASC': 'Gender=Masc', 'FEM': 'Gender=Fem', 'NEUT': 'Gender=Neut',
                    # Case
                    'NOM': 'Case=Nom', 'ACC': 'Case=Acc', 'GEN': 'Case=Gen', 'DAT': 'Case=Dat',
                    # Mood
                    'IND': 'Mood=Ind', 'SUB': 'Mood=Sub', 'IMP': 'Mood=Imp',
                    # Aspect
                    'IPFV': 'Aspect=Imp', 'PFV': 'Aspect=Perf',
                    # Voice
                    'ACT': 'Voice=Act', 'PASS': 'Voice=Pass',
                }
                
                for part in unimorph_parts:
                    # Check if it's a POS tag
                    if part in pos_mapping:
                        upos = pos_mapping[part]
                    # Check if it's a feature
                    elif part in feature_mapping:
                        ud_feats.append(feature_mapping[part])
                
                if ud_feats:
                    feats = '|'.join(ud_feats)
            
            # Create vocabulary entry
            entry = {
                'lemma': lemma.lower(),
                'count': default_count
            }
            
            if upos != '_':
                entry['upos'] = upos
            if feats != '_':
                entry['feats'] = feats
            
            # Add to vocabulary (handle multiple analyses for same form)
            vocab[form].append(entry)
    
    # Convert lists to single dict if only one analysis, otherwise keep as list
    result = {}
    for form, entries in vocab.items():
        if len(entries) == 1:
            result[form] = entries[0]
        else:
            result[form] = entries
    
    return result


def load_vocabulary_file(file_path: Path, default_count: int = 1) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
    """
    Load a vocabulary file in various formats (JSON, UniMorph).
    
    Supports:
    - FlexiPipe JSON format (with or without counts)
    - UniMorph lexicon format (tab-separated: form\tlemma\tfeatures)
    
    Args:
        file_path: Path to vocabulary file
        default_count: Default count to use for entries without counts (default: 1)
    
    Returns:
        Tuple of (vocab_dict, transitions_dict, metadata_dict)
        - vocab_dict: Vocabulary entries
        - transitions_dict: Transition probabilities (if available, None otherwise)
        - metadata_dict: Metadata (if available, None otherwise)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")
    
    # Detect format by extension or content
    if file_path.suffix.lower() == '.json':
        # JSON format (FlexiPipe vocabulary)
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Handle new format with metadata/vocab/transitions structure
        if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
            vocab = vocab_data.get('vocab', {})
            transitions = vocab_data.get('transitions', None)
            metadata = vocab_data.get('metadata', None)
        else:
            # Old format: just vocab dict
            vocab = vocab_data
            transitions = None
            metadata = None
        
        # Ensure all entries have counts (use default if missing)
        vocab = _ensure_counts(vocab, default_count)
        
        return vocab, transitions, metadata
    
    else:
        # Assume UniMorph format (tab-separated)
        vocab = load_unimorph_lexicon(file_path, default_count)
        return vocab, None, None


def _ensure_counts(vocab: Dict, default_count: int = 1) -> Dict:
    """
    Ensure all vocabulary entries have a count field.
    
    If an entry doesn't have a count, adds default_count.
    This allows vocabularies without counts to work seamlessly.
    
    Args:
        vocab: Vocabulary dictionary
        default_count: Default count to use if missing
    
    Returns:
        Vocabulary dictionary with all entries having counts
    """
    result = {}
    
    for form, entry in vocab.items():
        if isinstance(entry, list):
            # Multiple analyses
            entries_with_counts = []
            for analysis in entry:
                analysis_copy = analysis.copy()
                if 'count' not in analysis_copy:
                    analysis_copy['count'] = default_count
                entries_with_counts.append(analysis_copy)
            result[form] = entries_with_counts
        elif isinstance(entry, dict):
            # Single analysis
            entry_copy = entry.copy()
            if 'count' not in entry_copy:
                entry_copy['count'] = default_count
            result[form] = entry_copy
        else:
            # Unknown format - keep as is
            result[form] = entry
    
    return result


def build_vocab_from_sentences(sentences: List[List[Dict]]) -> Dict:
    """
    Build vocabulary from parsed sentences (CoNLL-U or TEITOK format).
    
    This is a shared function used by both training and create-vocab to ensure
    consistent vocabulary format. The vocabulary format is:
    - Single analysis: {form: {upos, xpos, feats, lemma, reg, expan, count}}
    - Multiple analyses: {form: [{upos, xpos, feats, lemma, reg, expan, count}, ...]}
    
    Args:
        sentences: List of sentences, where each sentence is a list of token dictionaries
                  Each token dict should have: form, upos, xpos, feats, lemma, norm_form, expan
    
    Returns:
        Dictionary mapping word forms to their analyses (single dict or list of dicts)
    """
    # Collect all annotations with frequency
    # Separate tracking for case-sensitive forms and lowercase forms
    all_annotations_case = defaultdict(lambda: defaultdict(int))  # form -> (upos, xpos, feats, lemma, norm, expan) -> count
    all_annotations_lower = defaultdict(lambda: defaultdict(int))  # form_lower -> (upos, xpos, feats, lemma, norm, expan) -> count
    
    for sentence in sentences:
        for token in sentence:
            form = token.get('form', '').strip()
            if not form or form == '_':
                continue
            
            form_lower = form.lower()
            upos = token.get('upos', '_')
            xpos = token.get('xpos', '_')
            feats = token.get('feats', '_')
            lemma = token.get('lemma', '_').lower() if token.get('lemma', '_') != '_' else '_'
            norm_form = token.get('norm_form', '_')
            expan_form = token.get('expan', '_')
            parts = token.get('parts', [])  # Contraction split forms (e.g., ["in", "dem"] for "im")
            
            # Store annotation combination and count frequency
            # parts is stored as tuple for hashing (convert list to tuple)
            parts_tuple = tuple(parts) if parts else ()
            annotation_key = (upos, xpos, feats, lemma, norm_form, expan_form, parts_tuple)
            all_annotations_case[form][annotation_key] += 1
            all_annotations_lower[form_lower][annotation_key] += 1
    
    vocab = {}
    
    # First, process case-sensitive forms (e.g., "Band", "Apple")
    for form, annotations in all_annotations_case.items():
        # Collect all annotation combinations (sorted by frequency, most frequent first)
        annotation_list = sorted(annotations.items(), key=lambda x: x[1], reverse=True)
        
        # Build entries for each annotation combination
        entries = []
        seen_combinations = set()
        
        for (upos, xpos, feats, lemma, norm_form, expan_form, parts), count in annotation_list:
            combination_key = (upos, xpos, feats, lemma, norm_form, expan_form, parts)
            if combination_key in seen_combinations:
                continue
            seen_combinations.add(combination_key)
            
            # Build entry (only include non-"_" fields, except lemma which is always included)
            entry = {}
            if upos != '_':
                entry['upos'] = upos
            if xpos != '_':
                entry['xpos'] = xpos
            if feats != '_':
                entry['feats'] = feats
            entry['lemma'] = lemma
            # Include normalization/expansion if present
            if norm_form and norm_form != '_':
                entry['reg'] = norm_form
            if expan_form and expan_form != '_':
                entry['expan'] = expan_form
            # Include parts (contraction split forms) if present
            if parts:
                entry['parts'] = list(parts)  # Convert tuple to list for JSON serialization
            # Include count/frequency
            entry['count'] = count
            
            if entry:
                entries.append(entry)
        
        # Store in vocabulary (case-sensitive entry)
        if entries:
            if len(entries) == 1:
                vocab[form] = entries[0]
            else:
                vocab[form] = entries
    
    # Then, process lowercase forms (for fallback when case-sensitive entry doesn't exist)
    for form_lower, annotations in all_annotations_lower.items():
        # Check if a case-sensitive form with same annotations already exists
        case_sensitive_exists = False
        case_sensitive_annotations = set()
        for case_form in all_annotations_case.keys():
            if case_form.lower() == form_lower:
                case_sensitive_exists = True
                for ann_key in all_annotations_case[case_form].keys():
                    case_sensitive_annotations.add(ann_key)
                break
        
        # If case-sensitive exists and has same annotations, skip lowercase (to avoid duplicates)
        if case_sensitive_exists:
            lowercase_annotations = set(annotations.keys())
            if lowercase_annotations == case_sensitive_annotations:
                continue  # Same annotations: skip lowercase entry
        
        # Collect all annotation combinations (sorted by frequency, most frequent first)
        annotation_list = sorted(annotations.items(), key=lambda x: x[1], reverse=True)
        
        # Build entries for each annotation combination
        entries = []
        seen_combinations = set()
        
        for (upos, xpos, feats, lemma, norm_form, expan_form, parts), count in annotation_list:
            combination_key = (upos, xpos, feats, lemma, norm_form, expan_form, parts)
            if combination_key in seen_combinations:
                continue
            seen_combinations.add(combination_key)
            
            # Build entry
            entry = {}
            if upos != '_':
                entry['upos'] = upos
            if xpos != '_':
                entry['xpos'] = xpos
            if feats != '_':
                entry['feats'] = feats
            entry['lemma'] = lemma
            if norm_form and norm_form != '_':
                entry['reg'] = norm_form
            if expan_form and expan_form != '_':
                entry['expan'] = expan_form
            # Include parts (contraction split forms) if present
            if parts:
                entry['parts'] = list(parts)  # Convert tuple to list for JSON serialization
            entry['count'] = count
            
            if entry:
                entries.append(entry)
        
        # Store in vocabulary (only if not already present as case-sensitive with same annotations)
        if entries and form_lower not in vocab:
            if len(entries) == 1:
                vocab[form_lower] = entries[0]
            else:
                vocab[form_lower] = entries
    
    return vocab
