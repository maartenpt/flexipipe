"""
Vocabulary-based dependency parser for FlexiPipe.

Provides fallback parsing when the neural parser is not available or fails.
Uses rule-based heuristics and vocabulary patterns to assign head and deprel.

Includes:
- Heuristic parser: Simple rule-based attachment
- Transition-based parser: MaltParser-style parser using transition probabilities
"""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import math

# Import tagset utilities for agreement checking
try:
    from flexipipe.tagset import xpos_to_upos_feats, check_agreement
    TAGSET_AVAILABLE = True
except ImportError:
    TAGSET_AVAILABLE = False
    def xpos_to_upos_feats(xpos: str, tagset_def: Dict) -> Tuple[str, str]:
        return '_', '_'
    def check_agreement(token1_feats: str, token2_feats: str, required_features: list = None) -> bool:
        return True


def parse_with_heuristics(sentence: List[Dict], use_xpos: bool = False) -> List[Dict]:
    """
    Parse a sentence using rule-based heuristics.
    
    This is a simple fallback parser that uses common dependency patterns
    based on POS tags. It's not as accurate as a trained parser, but provides
    reasonable results for basic use cases.
    
    Strategy:
    1. Find the root (typically the first/main verb, or first content word)
    2. Apply common attachment rules:
       - Determiners attach to following nouns
       - Adjectives attach to following nouns
       - Prepositions attach to following nouns
       - Auxiliaries attach to main verbs
       - Subjects attach to verbs
       - Objects attach to verbs
    
    Args:
        sentence: List of token dictionaries with 'upos' and/or 'xpos' fields
        use_xpos: If True, use XPOS for parsing; if False, use UPOS (default)
        
    Returns:
        List of token dictionaries with 'head' and 'deprel' fields added
    """
    if not sentence:
        return sentence
    
    tag_key = 'xpos' if use_xpos else 'upos'
    parsed = []
    
    # Find root (typically first verb, or first content word)
    root_idx = None
    for i, token in enumerate(sentence):
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        
        # Look for verbs (common root candidates)
        if pos.startswith('V') or (tag_key == 'upos' and pos == 'VERB'):
            root_idx = i
            break
    
    # If no verb found, use first content word (not punctuation, not function word)
    if root_idx is None:
        for i, token in enumerate(sentence):
            pos = token.get(tag_key, '_')
            if pos == '_':
                pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
            
            # Skip punctuation and common function words
            if pos in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') or pos.startswith('F'):
                continue
            
            # Prefer nouns, then other content words
            if (pos.startswith('N') or (tag_key == 'upos' and pos in ('NOUN', 'PROPN')) or
                pos.startswith('A') or (tag_key == 'upos' and pos == 'ADJ')):
                root_idx = i
                break
        
        # Last resort: use first non-punctuation token
        if root_idx is None:
            for i, token in enumerate(sentence):
                pos = token.get(tag_key, '_')
                if pos == '_':
                    pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
                if pos not in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') and not pos.startswith('F'):
                    root_idx = i
                    break
    
    # Default to first token if still no root found
    if root_idx is None:
        root_idx = 0
    
    # Assign heads and deprels
    for i, token in enumerate(sentence):
        parsed_token = token.copy()
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        
        if i == root_idx:
            # Root token
            parsed_token['head'] = 0
            parsed_token['deprel'] = 'root'
        else:
            # Apply attachment rules
            head, deprel = _find_head_and_deprel(i, sentence, root_idx, pos, tag_key, use_xpos)
            parsed_token['head'] = head
            parsed_token['deprel'] = deprel
        
        parsed.append(parsed_token)
    
    return parsed


def _find_head_and_deprel(
    token_idx: int,
    sentence: List[Dict],
    root_idx: int,
    pos: str,
    tag_key: str,
    use_xpos: bool
) -> Tuple[int, str]:
    """
    Find the head and dependency relation for a token using heuristics.
    
    Args:
        token_idx: Index of current token
        sentence: List of all tokens
        root_idx: Index of root token
        pos: POS tag of current token
        tag_key: 'upos' or 'xpos'
        use_xpos: Whether using XPOS tags
        
    Returns:
        Tuple of (head_index, deprel)
    """
    # Common patterns based on POS
    pos_lower = pos.lower()
    
    # Punctuation: attach to previous token
    if pos in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') or pos.startswith('F'):
        if token_idx > 0:
            # Attach to previous token (1-indexed: previous token is at token_idx-1, so head = token_idx)
            return token_idx, 'punct'  # Previous token is at index token_idx-1, in 1-indexed that's token_idx
        else:
            return root_idx + 1, 'punct'  # 1-indexed
    
    # Determiners: attach to following noun
    if (pos.startswith('D') or (tag_key == 'upos' and pos in ('DET', 'PRON')) or
        pos_lower.startswith('da') or pos_lower.startswith('di')):
        # Look ahead for noun
        for j in range(token_idx + 1, len(sentence)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (next_pos.startswith('N') or (tag_key == 'upos' and next_pos in ('NOUN', 'PROPN'))):
                return j + 1, 'det'  # 1-indexed
        # Fallback: attach to root
        return root_idx + 1, 'det'
    
    # Adjectives: attach to nearby noun (check both directions, prefer closer)
    if pos.startswith('A') or (tag_key == 'upos' and pos == 'ADJ'):
        # Check both directions and find closest noun
        best_noun_idx = None
        best_distance = float('inf')
        
        # Look ahead (for languages like Spanish where ADJ follows NOUN)
        for j in range(token_idx + 1, min(len(sentence), token_idx + 5)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (next_pos.startswith('N') or (tag_key == 'upos' and next_pos in ('NOUN', 'PROPN'))):
                distance = j - token_idx
                if distance < best_distance:
                    best_distance = distance
                    best_noun_idx = j
        
        # Look behind (for languages like English where ADJ precedes NOUN)
        for j in range(token_idx - 1, max(-1, token_idx - 5), -1):
            prev_pos = sentence[j].get(tag_key, '_')
            if prev_pos == '_':
                prev_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (prev_pos.startswith('N') or (tag_key == 'upos' and prev_pos in ('NOUN', 'PROPN'))):
                distance = token_idx - j
                if distance < best_distance:
                    best_distance = distance
                    best_noun_idx = j
        
        if best_noun_idx is not None:
            return best_noun_idx + 1, 'amod'  # 1-indexed
        
        # Fallback: attach to root
        return root_idx + 1, 'amod'
    
    # Prepositions: attach to following noun, head is root or previous verb
    if pos.startswith('S') or (tag_key == 'upos' and pos == 'ADP'):
        # Look ahead for noun
        for j in range(token_idx + 1, len(sentence)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (next_pos.startswith('N') or (tag_key == 'upos' and next_pos in ('NOUN', 'PROPN'))):
                # Preposition attaches to its object, object attaches to root
                return root_idx + 1, 'case'  # 1-indexed
        # Fallback
        return root_idx + 1, 'case'
    
    # Nouns: typically attach to verbs (as subjects/objects) or other nouns (as modifiers)
    if pos.startswith('N') or (tag_key == 'upos' and pos in ('NOUN', 'PROPN')):
        # Look for nearby verb
        for j in range(max(0, token_idx - 3), min(len(sentence), token_idx + 4)):
            if j == token_idx:
                continue
            other_pos = sentence[j].get(tag_key, '_')
            if other_pos == '_':
                other_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if other_pos.startswith('V') or (tag_key == 'upos' and other_pos == 'VERB'):
                # Subject if before verb, object if after
                if j > token_idx:
                    return j + 1, 'nsubj'  # 1-indexed
                else:
                    return j + 1, 'obj'  # 1-indexed
        # Fallback: attach to root
        return root_idx + 1, 'nmod'
    
    # Verbs: attach to root if not root itself, or to auxiliaries
    if pos.startswith('V') or (tag_key == 'upos' and pos == 'VERB'):
        if token_idx == root_idx:
            return 0, 'root'
        # Check for auxiliaries before
        for j in range(token_idx - 1, max(0, token_idx - 3), -1):
            prev_pos = sentence[j].get(tag_key, '_')
            if prev_pos == '_':
                prev_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            # Common auxiliary patterns
            if (prev_pos.startswith('V') and 
                sentence[j].get('form', '').lower() in ('have', 'has', 'had', 'be', 'is', 'are', 'was', 'were', 'will', 'would', 'can', 'could', 'should', 'may', 'might')):
                return j + 1, 'aux'  # 1-indexed
        # Otherwise attach to root
        return root_idx + 1, 'conj' if token_idx > root_idx else 'xcomp'
    
    # Pronouns: similar to nouns
    if (tag_key == 'upos' and pos == 'PRON') or pos.startswith('P'):
        # Look for verb
        for j in range(max(0, token_idx - 2), min(len(sentence), token_idx + 3)):
            if j == token_idx:
                continue
            other_pos = sentence[j].get(tag_key, '_')
            if other_pos == '_':
                other_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if other_pos.startswith('V') or (tag_key == 'upos' and other_pos == 'VERB'):
                if j > token_idx:
                    return j + 1, 'nsubj'  # 1-indexed
                else:
                    return j + 1, 'obj'  # 1-indexed
        return root_idx + 1, 'nmod'
    
    # Adverbs: attach to nearby verb or adjective
    if (tag_key == 'upos' and pos == 'ADV') or pos.startswith('R'):
        for j in range(max(0, token_idx - 2), min(len(sentence), token_idx + 3)):
            if j == token_idx:
                continue
            other_pos = sentence[j].get(tag_key, '_')
            if other_pos == '_':
                other_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if other_pos.startswith('V') or (tag_key == 'upos' and other_pos == 'VERB'):
                return j + 1, 'advmod'  # 1-indexed
            if other_pos.startswith('A') or (tag_key == 'upos' and other_pos == 'ADJ'):
                return j + 1, 'advmod'  # 1-indexed
        return root_idx + 1, 'advmod'
    
    # Conjunctions: attach to following content word
    if (tag_key == 'upos' and pos == 'CCONJ') or pos.startswith('C'):
        for j in range(token_idx + 1, min(len(sentence), token_idx + 3)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if next_pos not in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') and not next_pos.startswith('F'):
                return j + 1, 'cc'  # 1-indexed
        return root_idx + 1, 'cc'
    
    # Default: attach to root
    return root_idx + 1, 'dep'


def _extract_deprel_patterns_from_vocab(vocab: Dict) -> Dict:
    """
    Extract dependency relation patterns from vocabulary if available.
    
    If vocabulary was built from parsed CoNLL-U files, it may contain
    dependency information. This function extracts common patterns like:
    - (ADJ, NOUN) -> amod (and direction)
    - (DET, NOUN) -> det
    - (NOUN, VERB) -> nsubj/obj (and direction)
    
    Args:
        vocab: Vocabulary dictionary
        
    Returns:
        Dictionary of patterns: {(pos1, pos2, direction): (deprel, count)}
        where direction is 'before' or 'after'
    """
    patterns = defaultdict(lambda: defaultdict(int))
    
    # This would require vocabulary entries to have deprel information
    # For now, return empty dict - can be enhanced if vocab includes deprel data
    return {}


def _learn_attachment_preferences(sentence: List[Dict], vocab: Dict, tag_key: str) -> Dict:
    """
    Learn attachment preferences from the sentence structure and vocabulary.
    
    This analyzes the sentence to determine language-specific patterns,
    such as whether adjectives typically precede or follow nouns.
    
    Args:
        sentence: List of tokens
        vocab: Vocabulary dictionary
        tag_key: 'upos' or 'xpos'
        
    Returns:
        Dictionary with learned preferences, e.g.:
        {'adj_noun_order': 'after', 'det_noun_order': 'after', ...}
    """
    preferences = {}
    
    # Analyze ADJ-NOUN patterns in the sentence
    adj_noun_before = 0
    adj_noun_after = 0
    
    for i in range(len(sentence) - 1):
        pos1 = sentence[i].get(tag_key, '_')
        if pos1 == '_':
            pos1 = sentence[i].get('upos' if tag_key == 'xpos' else 'xpos', '_')
        pos2 = sentence[i + 1].get(tag_key, '_')
        if pos2 == '_':
            pos2 = sentence[i + 1].get('upos' if tag_key == 'xpos' else 'xpos', '_')
        
        # Check ADJ-NOUN patterns
        is_adj1 = pos1.startswith('A') or (tag_key == 'upos' and pos1 == 'ADJ')
        is_noun2 = pos2.startswith('N') or (tag_key == 'upos' and pos2 in ('NOUN', 'PROPN'))
        is_noun1 = pos1.startswith('N') or (tag_key == 'upos' and pos1 in ('NOUN', 'PROPN'))
        is_adj2 = pos2.startswith('A') or (tag_key == 'upos' and pos2 == 'ADJ')
        
        if is_adj1 and is_noun2:
            adj_noun_before += 1
        if is_noun1 and is_adj2:
            adj_noun_after += 1
    
    # Determine preference (default to 'after' for Romance languages)
    if adj_noun_after > adj_noun_before:
        preferences['adj_noun_order'] = 'after'
    elif adj_noun_before > adj_noun_after:
        preferences['adj_noun_order'] = 'before'
    else:
        preferences['adj_noun_order'] = 'either'  # No clear preference
    
    return preferences


def parse_with_vocab_patterns(
    sentence: List[Dict],
    vocab: Dict,
    transition_probs: Optional[Dict] = None,
    use_xpos: bool = False,
    tagset_def: Optional[Dict] = None
) -> List[Dict]:
    """
    Parse a sentence using vocabulary patterns and transition probabilities.
    
    This attempts to use dependency patterns from the vocabulary if available,
    falling back to heuristics if not.
    
    Args:
        sentence: List of token dictionaries
        vocab: Vocabulary dictionary (may contain deprel patterns)
        transition_probs: Transition probabilities (may contain deprel transitions)
        use_xpos: If True, use XPOS; if False, use UPOS
        
    Returns:
        List of token dictionaries with 'head' and 'deprel' fields
    """
    tag_key = 'xpos' if use_xpos else 'upos'
    
    # Learn attachment preferences from the sentence
    preferences = _learn_attachment_preferences(sentence, vocab, tag_key)
    
    # Extract dependency patterns from vocabulary if available
    deprel_patterns = _extract_deprel_patterns_from_vocab(vocab)
    
    # Use enhanced heuristics with learned preferences
    return parse_with_heuristics_enhanced(sentence, use_xpos, preferences, tagset_def)


def parse_with_heuristics_enhanced(
    sentence: List[Dict],
    use_xpos: bool = False,
    preferences: Optional[Dict] = None,
    tagset_def: Optional[Dict] = None
) -> List[Dict]:
    """
    Enhanced heuristic parser that uses learned preferences.
    
    Similar to parse_with_heuristics but uses language-specific preferences
    learned from the sentence structure.
    
    Args:
        sentence: List of token dictionaries with 'upos' and/or 'xpos' fields
        use_xpos: If True, use XPOS for parsing; if False, use UPOS (default)
        preferences: Dictionary of learned preferences (e.g., adj_noun_order)
        
    Returns:
        List of token dictionaries with 'head' and 'deprel' fields added
    """
    if not sentence:
        return sentence
    
    tag_key = 'xpos' if use_xpos else 'upos'
    parsed = []
    
    # Find root (typically first verb, or first content word)
    root_idx = None
    for i, token in enumerate(sentence):
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        
        # Look for verbs (common root candidates)
        if pos.startswith('V') or (tag_key == 'upos' and pos == 'VERB'):
            root_idx = i
            break
    
    # If no verb found, use first content word
    if root_idx is None:
        for i, token in enumerate(sentence):
            pos = token.get(tag_key, '_')
            if pos == '_':
                pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
            
            if pos in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') or pos.startswith('F'):
                continue
            
            if (pos.startswith('N') or (tag_key == 'upos' and pos in ('NOUN', 'PROPN')) or
                pos.startswith('A') or (tag_key == 'upos' and pos == 'ADJ')):
                root_idx = i
                break
        
        if root_idx is None:
            for i, token in enumerate(sentence):
                pos = token.get(tag_key, '_')
                if pos == '_':
                    pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
                if pos not in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') and not pos.startswith('F'):
                    root_idx = i
                    break
    
    if root_idx is None:
        root_idx = 0
    
    # Assign heads and deprels
    for i, token in enumerate(sentence):
        parsed_token = token.copy()
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        
        if i == root_idx:
            parsed_token['head'] = 0
            parsed_token['deprel'] = 'root'
        else:
            head, deprel = _find_head_and_deprel_enhanced(
                i, sentence, root_idx, pos, tag_key, use_xpos, preferences, tagset_def
            )
            parsed_token['head'] = head
            parsed_token['deprel'] = deprel
        
        parsed.append(parsed_token)
    
    return parsed


def _find_head_and_deprel_enhanced(
    token_idx: int,
    sentence: List[Dict],
    root_idx: int,
    pos: str,
    tag_key: str,
    use_xpos: bool,
    preferences: Optional[Dict] = None,
    tagset_def: Optional[Dict] = None
) -> Tuple[int, str]:
    """
    Enhanced version that uses learned preferences and agreement checking for better parsing.
    """
    if preferences is None:
        preferences = {}
    
    pos_lower = pos.lower()
    
    # Helper to get FEATS from token
    def get_token_feats(token: Dict) -> str:
        """Get FEATS from token, using tagset if needed."""
        feats = token.get('feats', '_')
        if feats and feats != '_':
            return feats
        # If no FEATS but have XPOS and tagset, try to extract
        if tagset_def and use_xpos:
            xpos = token.get('xpos', '_')
            if xpos and xpos != '_':
                _, extracted_feats = xpos_to_upos_feats(xpos, tagset_def)
                return extracted_feats
        return '_'
    
    # Punctuation: attach to previous token
    if pos in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') or pos.startswith('F'):
        if token_idx > 0:
            return token_idx, 'punct'
        else:
            return root_idx + 1, 'punct'
    
    # Determiners: attach to following noun (universal pattern)
    if (pos.startswith('D') or (tag_key == 'upos' and pos in ('DET', 'PRON')) or
        pos_lower.startswith('da') or pos_lower.startswith('di')):
        for j in range(token_idx + 1, len(sentence)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (next_pos.startswith('N') or (tag_key == 'upos' and next_pos in ('NOUN', 'PROPN'))):
                return j + 1, 'det'
        return root_idx + 1, 'det'
    
    # Adjectives: use learned preference for order + agreement checking
    if pos.startswith('A') or (tag_key == 'upos' and pos == 'ADJ'):
        adj_noun_order = preferences.get('adj_noun_order', 'either')
        best_noun_idx = None
        best_score = float('-inf')  # Use score instead of distance (higher is better)
        
        adj_feats = get_token_feats(sentence[token_idx])
        
        # Check both directions
        for j in range(token_idx + 1, min(len(sentence), token_idx + 5)):
            next_pos = sentence[j].get(tag_key, '_')
            if next_pos == '_':
                next_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (next_pos.startswith('N') or (tag_key == 'upos' and next_pos in ('NOUN', 'PROPN'))):
                distance = j - token_idx
                score = 1.0 / (distance + 1)  # Closer = higher score
                
                # Check agreement (Gender, Number)
                noun_feats = get_token_feats(sentence[j])
                if check_agreement(adj_feats, noun_feats, ['Gender', 'Number']):
                    score += 1.0  # Bonus for agreement
                
                # Prefer based on learned order
                if adj_noun_order == 'after':
                    score += 0.5  # Bonus for ADJ after NOUN
                
                if score > best_score:
                    best_score = score
                    best_noun_idx = j
        
        for j in range(token_idx - 1, max(-1, token_idx - 5), -1):
            prev_pos = sentence[j].get(tag_key, '_')
            if prev_pos == '_':
                prev_pos = sentence[j].get('upos' if tag_key == 'xpos' else 'xpos', '_')
            if (prev_pos.startswith('N') or (tag_key == 'upos' and prev_pos in ('NOUN', 'PROPN'))):
                distance = token_idx - j
                score = 1.0 / (distance + 1)  # Closer = higher score
                
                # Check agreement
                noun_feats = get_token_feats(sentence[j])
                if check_agreement(adj_feats, noun_feats, ['Gender', 'Number']):
                    score += 1.0  # Bonus for agreement
                
                # Prefer based on learned order
                if adj_noun_order == 'before':
                    score += 0.5  # Bonus for ADJ before NOUN
                elif adj_noun_order == 'after':
                    score -= 0.3  # Penalty for ADJ before NOUN when we prefer after
                
                if score > best_score:
                    best_score = score
                    best_noun_idx = j
        
        if best_noun_idx is not None:
            return best_noun_idx + 1, 'amod'
        return root_idx + 1, 'amod'
    
    # Rest of the rules remain the same (they're already bidirectional where needed)
    return _find_head_and_deprel(token_idx, sentence, root_idx, pos, tag_key, use_xpos)


def parse_with_transition_based(
    sentence: List[Dict],
    vocab: Optional[Dict] = None,
    transition_probs: Optional[Dict] = None,
    use_xpos: bool = False,
    tagset_def: Optional[Dict] = None,
    debug: bool = False
) -> List[Dict]:
    """
    Parse a sentence using a transition-based parser (MaltParser-style).
    
    Uses a greedy scoring approach to find the best attachments based on:
    - POS-based preferences (e.g., ADP should attach to following NOUN)
    - Dependency transition probabilities (if available from training corpus)
    - Distance (prefer closer attachments)
    - Agreement (if tagset available)
    
    Args:
        sentence: List of token dictionaries with POS tags
        vocab: Vocabulary dictionary (for POS-based scoring)
        transition_probs: Transition probabilities (may include 'deprel' transitions)
        use_xpos: If True, use XPOS tags; if False, use UPOS
        tagset_def: Tagset definition for agreement checking
        
    Returns:
        List of token dictionaries with 'head' and 'deprel' fields added
    """
    if not sentence:
        return sentence
    
    tag_key = 'xpos' if use_xpos else 'upos'
    n = len(sentence)
    
    # Extract POS tags for all tokens
    pos_tags = []
    for token in sentence:
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        pos_tags.append(pos)
    
    # Build dependency transition probabilities if available
    # Transitions are now stored as {'upos': {...}, 'xpos': {...}}
    deprel_transitions = {}
    if transition_probs and 'deprel' in transition_probs:
        all_deprel_transitions = transition_probs['deprel']
        # Select the appropriate format based on tag_key
        if isinstance(all_deprel_transitions, dict):
            # New format: {'upos': {...}, 'xpos': {...}}
            if tag_key in all_deprel_transitions:
                deprel_transitions = all_deprel_transitions[tag_key]
            # Debug: print if we have deprel transitions
            if debug and deprel_transitions:
                import sys
                print(f"[DEBUG PARSER] Using {len(deprel_transitions)} dependency transition patterns (tag_key={tag_key})", file=sys.stderr)
        else:
            # Old format: flat dict with string keys (backward compatibility)
            # Filter transitions based on tag format (UPOS vs XPOS)
            for key, prob in all_deprel_transitions.items():
                head_pos, dep_pos, deprel = key.split('|', 2)
                # Check if this transition matches our tag format
                is_upos_format = (head_pos in ['NOUN', 'VERB', 'ADJ', 'ADP', 'DET', 'PROPN', 'PRON', 'NUM', 'PUNCT', 'ADV', 'CCONJ', 'SCONJ', 'AUX', 'INTJ'] or
                                 dep_pos in ['NOUN', 'VERB', 'ADJ', 'ADP', 'DET', 'PROPN', 'PRON', 'NUM', 'PUNCT', 'ADV', 'CCONJ', 'SCONJ', 'AUX', 'INTJ'])
                
                if tag_key == 'xpos' and is_upos_format:
                    continue  # Skip UPOS transitions when using XPOS
                elif tag_key == 'upos' and not is_upos_format:
                    continue  # Skip XPOS transitions when using UPOS
                
                deprel_transitions[key] = prob
            
            # Debug: print if we have deprel transitions
            if debug and deprel_transitions:
                import sys
                print(f"[DEBUG PARSER] Using {len(deprel_transitions)} dependency transition patterns (tag_key={tag_key}, old format)", file=sys.stderr)
    
    # Initialize: all tokens point to root (will be updated)
    parsed = [token.copy() for token in sentence]
    for i in range(n):
        parsed[i]['head'] = 0
        parsed[i]['deprel'] = 'dep'
    
    # Find root (typically first/main verb)
    root_idx = _find_root(sentence, tag_key)
    if root_idx is not None:
        parsed[root_idx]['head'] = 0
        parsed[root_idx]['deprel'] = 'root'
    
    # Score all possible attachments
    attachment_scores = {}  # (dependent_idx, head_idx) -> score
    
    for dep_idx in range(n):
        if dep_idx == root_idx:
            continue
        
        dep_pos = pos_tags[dep_idx]
        dep_token = sentence[dep_idx]
        
        for head_idx in range(n):
            if head_idx == dep_idx:
                continue
            
            head_pos = pos_tags[head_idx]
            head_token = sentence[head_idx]
            
            # Calculate attachment score
            score = _score_attachment(
                dep_idx, dep_pos, dep_token,
                head_idx, head_pos, head_token,
                sentence, tag_key, deprel_transitions, tagset_def, debug
            )
            
            attachment_scores[(dep_idx, head_idx)] = score
    
    # Greedily assign heads based on scores (avoiding cycles)
    sorted_attachments = sorted(attachment_scores.items(), key=lambda x: x[1], reverse=True)
    
    assigned_heads = {}  # dep_idx -> head_idx
    
    for (dep_idx, head_idx), score in sorted_attachments:
        # Skip if already assigned or would create cycle
        if dep_idx in assigned_heads:
            continue
        
        # Check for cycles
        if _would_create_cycle(dep_idx, head_idx, assigned_heads):
            continue
        
        # Assign this attachment
        assigned_heads[dep_idx] = head_idx
        
        # Get POS tags for deprel determination (they're not in scope from the scoring loop)
        dep_pos_actual = pos_tags[dep_idx]
        head_pos_actual = pos_tags[head_idx]
        dep_token = sentence[dep_idx]
        head_token = sentence[head_idx]
        
        # Determine deprel
        deprel = _determine_deprel(
            dep_idx, dep_pos_actual, dep_token,
            head_idx, head_pos_actual, head_token,
            sentence, tag_key, tagset_def
        )
        
        parsed[dep_idx]['head'] = head_idx + 1  # 1-indexed
        parsed[dep_idx]['deprel'] = deprel
    
    # Handle any unassigned tokens (attach to root)
    for i in range(n):
        if i not in assigned_heads and i != root_idx:
            parsed[i]['head'] = (root_idx + 1) if root_idx is not None else 0
            parsed[i]['deprel'] = 'dep'
    
    return parsed


def _find_root(sentence: List[Dict], tag_key: str) -> Optional[int]:
    """Find the root token (typically first/main verb)."""
    # Look for verbs first
    for i, token in enumerate(sentence):
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        if pos.startswith('V') or (tag_key == 'upos' and pos == 'VERB'):
            return i
    
    # Fallback: first content word
    for i, token in enumerate(sentence):
        pos = token.get(tag_key, '_')
        if pos == '_':
            pos = token.get('upos' if tag_key == 'xpos' else 'xpos', '_')
        if pos not in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') and not pos.startswith('F'):
            return i
    
    return 0


def _score_attachment(
    dep_idx: int, dep_pos: str, dep_token: Dict,
    head_idx: int, head_pos: str, head_token: Dict,
    sentence: List[Dict], tag_key: str,
    deprel_transitions: Dict, tagset_def: Optional[Dict],
    debug: bool = False
) -> float:
    """
    Score a potential attachment of dep_token to head_token.
    
    Returns a score (higher is better) based on:
    - POS-based preferences
    - Dependency transition probabilities
    - Distance
    - Agreement
    """
    score = 0.0
    distance = abs(dep_idx - head_idx)
    
    # Distance penalty (prefer closer attachments)
    score -= distance * 0.1
    
    # POS-based preferences
    dep_lower = dep_pos.lower()
    head_lower = head_pos.lower()
    
    # Determiners attach to nouns
    if ((dep_pos.startswith('D') or (tag_key == 'upos' and dep_pos in ('DET', 'PRON'))) and
        (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
        if dep_idx < head_idx:  # DET before NOUN
            score += 5.0
    
    # Adjectives attach to nouns
    if ((dep_pos.startswith('A') or (tag_key == 'upos' and dep_pos == 'ADJ')) and
        (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
        score += 4.0
        # Check agreement if available
        if tagset_def and TAGSET_AVAILABLE:
            dep_feats = dep_token.get('feats', '_')
            head_feats = head_token.get('feats', '_')
            if check_agreement(dep_feats, head_feats, ['Gender', 'Number']):
                score += 2.0
    
    # Prepositions attach to nouns (case relation) - STRONG PREFERENCE
    if ((dep_pos.startswith('S') or (tag_key == 'upos' and dep_pos == 'ADP')) and
        (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
        if dep_idx < head_idx:  # ADP before NOUN
            score += 10.0  # Very strong preference for ADP -> NOUN
    
    # Nouns attach to verbs (subject/object)
    if ((dep_pos.startswith('N') or (tag_key == 'upos' and dep_pos in ('NOUN', 'PROPN'))) and
        (head_pos.startswith('V') or (tag_key == 'upos' and head_pos == 'VERB'))):
        score += 3.0
        if dep_idx < head_idx:  # NOUN before VERB (subject)
            score += 1.0
    
    # Dependency transition probabilities (if available)
    if deprel_transitions:
        # Keys are stored as strings: "head_pos|dep_pos|deprel"
        # Try common deprels based on POS patterns
        best_score = 0.0
        best_deprel = None
        
        # Determine likely deprel based on POS patterns
        likely_deprels = []
        if ((dep_pos.startswith('D') or (tag_key == 'upos' and dep_pos in ('DET', 'PRON'))) and
            (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
            likely_deprels = ['det', 'nmod']
        elif ((dep_pos.startswith('A') or (tag_key == 'upos' and dep_pos == 'ADJ')) and
              (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
            likely_deprels = ['amod', 'nmod']
        elif ((dep_pos.startswith('S') or (tag_key == 'upos' and dep_pos == 'ADP')) and
              (head_pos.startswith('N') or (tag_key == 'upos' and head_pos in ('NOUN', 'PROPN')))):
            likely_deprels = ['case', 'nmod']
        elif ((dep_pos.startswith('N') or (tag_key == 'upos' and dep_pos in ('NOUN', 'PROPN'))) and
              (head_pos.startswith('V') or (tag_key == 'upos' and head_pos == 'VERB'))):
            likely_deprels = ['nsubj', 'obj', 'obl', 'nmod']
        else:
            # Try all common deprels
            likely_deprels = ['nmod', 'amod', 'det', 'case', 'nsubj', 'obj', 'obl', 'advmod']
        
        # Try likely deprels first, then fall back to others
        for deprel_candidate in likely_deprels + ['dep']:
            key = f"{head_pos}|{dep_pos}|{deprel_candidate}"
            if key in deprel_transitions:
                prob = deprel_transitions[key]
                deprel_score = math.log(prob + 1e-10)
                if deprel_score > best_score:
                    best_score = deprel_score
                    best_deprel = deprel_candidate
        
        if best_score > 0:
            score += best_score
            # Debug: log when we use a dependency transition
            if deprel_transitions and len(deprel_transitions) > 0:  # Only if we have transitions
                import sys
                if hasattr(sys, '_parser_debug_count'):
                    sys._parser_debug_count += 1
                else:
                    sys._parser_debug_count = 1
                    if debug and sys._parser_debug_count <= 5:  # Only print first few
                        print(f"[DEBUG PARSER] Using transition: {key} (prob={prob:.6f}, score={best_score:.3f})", file=sys.stderr)
    
    return score


def _would_create_cycle(dep_idx: int, head_idx: int, assigned_heads: Dict[int, int]) -> bool:
    """Check if assigning dep_idx -> head_idx would create a cycle."""
    # Follow the chain from head_idx
    current = head_idx
    visited = set()
    while current in assigned_heads:
        if current == dep_idx:
            return True  # Cycle detected
        if current in visited:
            break  # Already checked this path
        visited.add(current)
        current = assigned_heads[current]
    return False


def _determine_deprel(
    dep_idx: int, dep_pos: str, dep_token: Dict,
    head_idx: int, head_pos: str, head_token: Dict,
    sentence: List[Dict], tag_key: str, tagset_def: Optional[Dict]
) -> str:
    """Determine the dependency relation for dep_token -> head_token."""
    dep_lower = dep_pos.lower()
    head_lower = head_pos.lower()
    
    # Punctuation
    if dep_pos in ('PUNCT', 'Fc', 'Fp', 'Fd', 'Fz') or dep_pos.startswith('F'):
        return 'punct'
    
    # Determiners
    if (dep_pos.startswith('D') or (tag_key == 'upos' and dep_pos in ('DET', 'PRON'))):
        return 'det'
    
    # Adjectives
    if dep_pos.startswith('A') or (tag_key == 'upos' and dep_pos == 'ADJ'):
        return 'amod'
    
    # Prepositions (case)
    if dep_pos.startswith('S') or (tag_key == 'upos' and dep_pos == 'ADP'):
        return 'case'
    
    # Nouns to verbs
    if ((dep_pos.startswith('N') or (tag_key == 'upos' and dep_pos in ('NOUN', 'PROPN'))) and
        (head_pos.startswith('V') or (tag_key == 'upos' and head_pos == 'VERB'))):
        if dep_idx < head_idx:
            return 'nsubj'
        else:
            return 'obj'
    
    # Default
    return 'dep'


def parse_sentence(
    sentence: List[Dict],
    vocab: Optional[Dict] = None,
    transition_probs: Optional[Dict] = None,
    use_xpos: bool = False,
    method: str = 'heuristics',
    tagset_def: Optional[Dict] = None,
    debug: bool = False
) -> List[Dict]:
    """
    Parse a sentence using the specified method.
    
    Args:
        sentence: List of token dictionaries with POS tags
        vocab: Vocabulary dictionary (for pattern-based parsing)
        transition_probs: Transition probabilities (may include 'deprel' for transition-based parsing)
        use_xpos: If True, use XPOS tags; if False, use UPOS
        method: Parsing method - 'heuristics' (default), 'vocab', or 'transition' (transition-based)
        tagset_def: Tagset definition for agreement checking
        
    Returns:
        List of token dictionaries with 'head' and 'deprel' fields added
    """
    import sys
    
    # Determine which method to use
    if method == 'transition' or (method == 'auto' and transition_probs and 'deprel' in transition_probs):
        # Check if we have transitions in the correct format
        tag_key = 'xpos' if use_xpos else 'upos'
        has_transitions = False
        if transition_probs and 'deprel' in transition_probs:
            deprel_transitions = transition_probs['deprel']
            if isinstance(deprel_transitions, dict) and tag_key in deprel_transitions:
                has_transitions = len(deprel_transitions[tag_key]) > 0
            elif isinstance(deprel_transitions, dict):
                # Old format - check if any transitions match
                has_transitions = len(deprel_transitions) > 0
        
        if has_transitions:
            if debug:
                print(f"[DEBUG PARSER] Using transition-based parser (MaltParser-style) with dependency transitions", file=sys.stderr)
            return parse_with_transition_based(sentence, vocab, transition_probs, use_xpos, tagset_def, debug)
        else:
            # Fall back to heuristics if no transitions available
            if debug:
                print(f"[DEBUG PARSER] Transition-based parser requested but no transitions available, falling back to heuristics", file=sys.stderr)
            if vocab:
                tag_key = 'xpos' if use_xpos else 'upos'
                preferences = _learn_attachment_preferences(sentence, vocab, tag_key)
                return parse_with_heuristics_enhanced(sentence, use_xpos, preferences, tagset_def)
            else:
                return parse_with_heuristics(sentence, use_xpos)
    elif method == 'vocab' and vocab:
        if debug:
            print(f"[DEBUG PARSER] Using vocabulary pattern-based parser", file=sys.stderr)
        return parse_with_vocab_patterns(sentence, vocab, transition_probs, use_xpos, tagset_def)
    else:
        # Always try to learn preferences from the sentence if vocab is available
        if vocab:
            tag_key = 'xpos' if use_xpos else 'upos'
            preferences = _learn_attachment_preferences(sentence, vocab, tag_key)
            if debug:
                print(f"[DEBUG PARSER] Using enhanced heuristic parser with learned preferences", file=sys.stderr)
            return parse_with_heuristics_enhanced(sentence, use_xpos, preferences, tagset_def)
        else:
            if debug:
                print(f"[DEBUG PARSER] Using basic heuristic parser", file=sys.stderr)
            return parse_with_heuristics(sentence, use_xpos)

