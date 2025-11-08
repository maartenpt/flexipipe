"""
Contractions module for FlexiPipe.
"""
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import unicodedata
import re

# Cache for learned contraction patterns (computed once per vocabulary)
_pattern_cache: Dict[str, Dict] = {}


def _normalize_accents(text: str) -> str:
    """Normalize accents for matching (e.g., 'jugár' -> 'jugar')."""
    # Remove combining diacritics
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def _build_contraction_patterns(vocab: Dict) -> Dict[str, List[Dict]]:
    """
    Build contraction patterns from vocabulary entries that have explicit parts.
    
    Returns a dictionary mapping suffix patterns to:
    - parts: list of suffix parts (e.g., ["lo"] or ["se", "lo"])
    - required_upos: set of UPOS tags that the base word must have (e.g., {"VERB"})
    - count: how many times this pattern was seen
    - examples: list of example contractions
    """
    patterns = defaultdict(lambda: {
        'parts': None,
        'required_upos': set(),
        'count': 0,
        'examples': []
    })
    
    for form, entry in vocab.items():
        analyses = entry if isinstance(entry, list) else [entry]
        for analysis in analyses:
            if 'parts' in analysis and analysis['parts']:
                parts = analysis['parts']
                if not parts or len(parts) < 2:
                    continue
                
                # Extract the suffix pattern
                # For "respirarlo" -> ["respirar", "lo"], the suffix is "lo"
                # For "jugárselo" -> ["jugar", "se", "lo"], the suffix is "selo"
                base_form = parts[0]
                suffix_parts = parts[1:]
                
                # Build the suffix string from parts (e.g., ["se", "lo"] -> "selo")
                suffix_string = ''.join(suffix_parts)
                
                # Try to find the suffix in the original form
                # Handle accent changes: "jugárselo" has "jugár" but base is "jugar"
                form_normalized = _normalize_accents(form.lower())
                base_normalized = _normalize_accents(base_form.lower())
                suffix_normalized = _normalize_accents(suffix_string.lower())
                
                # Check if form can be split as base + suffix (with possible accent/morphological changes)
                # The form should end with the suffix, and the base should be before it
                if form_normalized.endswith(suffix_normalized):
                    # Find where the base ends (allowing for accent changes and morphological variations)
                    # Try to find the base at the position where suffix would start
                    expected_base_end = len(form_normalized) - len(suffix_normalized)
                    if expected_base_end >= len(base_normalized):
                        # Check if the part before suffix matches the base (normalized)
                        base_part = form_normalized[:expected_base_end]
                        base_part_normalized = _normalize_accents(base_part)
                        
                        # Check if base_part matches base_normalized, or if it's a morphological variant
                        # (e.g., "pagá" matches "pagar" pattern, or "pagar" matches "pagá" in form)
                        base_matches = (
                            base_part_normalized.endswith(base_normalized) or
                            base_normalized.endswith(_normalize_base_for_clitic(base_part_normalized)) or
                            _normalize_base_for_clitic(base_part_normalized).endswith(_normalize_base_for_clitic(base_normalized))
                        )
                        
                        if base_matches:
                            # Extract the actual suffix from the original form
                            # This preserves the exact suffix as it appears (e.g., "selo" from "jugárselo")
                            actual_suffix = form_normalized[expected_base_end:]
                            
                            # Store pattern using the actual suffix from the form
                            pattern_key = actual_suffix
                            if patterns[pattern_key]['parts'] is None:
                                patterns[pattern_key]['parts'] = suffix_parts
                                patterns[pattern_key]['examples'] = []
                            
                            patterns[pattern_key]['count'] += analysis.get('count', 1)
                            if form not in patterns[pattern_key]['examples']:
                                patterns[pattern_key]['examples'].append(form)
                            
                            # Record required UPOS for this pattern
                            upos = analysis.get('upos', '_')
                            if upos and upos != '_':
                                patterns[pattern_key]['required_upos'].add(upos)
    
    # Filter patterns: only keep those with at least 2 examples (to avoid noise)
    filtered_patterns = {}
    for pattern_key, pattern_data in patterns.items():
        if pattern_data['count'] >= 2 and pattern_data['parts']:
            filtered_patterns[pattern_key] = {
                'parts': pattern_data['parts'],
                'required_upos': pattern_data['required_upos'],
                'count': pattern_data['count'],
                'examples': pattern_data['examples'][:5]  # Keep first 5 examples
            }
    
    return filtered_patterns


def _normalize_base_for_clitic(base: str) -> str:
    """
    Normalize base word for clitic matching.
    Handles morphological changes like dropping final 'r' in Portuguese (pagar -> pagá).
    """
    # Remove final 'r' (common in Portuguese before clitics)
    if base.endswith('r') and len(base) > 2:
        return base[:-1]
    return base


def split_contraction(form: str, vocab: Dict, aggressive: bool = False, language: Optional[str] = None) -> Optional[List[str]]:
    """
    Split contraction into component words (e.g., "destas" -> ["de", "estas"]).
    
    Uses vocabulary patterns from training data to identify potential contractions and split them.
    Language-agnostic: relies entirely on vocabulary patterns, no hardcoded language rules.
    
    Strategy:
    1. First check for explicit parts in vocabulary (most reliable)
       - If word has multiple entries (contraction and regular), prefer the contraction
    2. Then check learned patterns from vocabulary (for productive contractions)
    3. For pattern-based splitting, verify:
       - The base word exists in vocabulary (with morphological variations like dropped 'r')
       - The base word has appropriate UPOS (if pattern requires it)
    
    Args:
        form: Word form that might be a contraction
        vocab: Vocabulary dictionary (must contain patterns from training data)
        aggressive: If True, use more aggressive splitting (currently unused, kept for compatibility)
        language: Language code (currently unused, kept for compatibility)
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    form_lower = form.lower()
    
    # Step 1: Check for explicit parts in vocabulary (most reliable)
    # If word has multiple entries, prefer the one with parts (contraction) over regular word
    entry = vocab.get(form) or vocab.get(form_lower)
    if entry:
        analyses = entry if isinstance(entry, list) else [entry]
        
        # First, check if any analysis has parts - prefer those
        parts_analyses = [a for a in analyses if 'parts' in a and a['parts']]
        if parts_analyses:
            # Found explicit parts - use the most frequent one
            best_analysis = max(parts_analyses, key=lambda a: a.get('count', 0))
            return list(best_analysis['parts'])
        
        # If no parts found, check if this is ambiguous (could be contraction or regular word)
        # In this case, we don't split (prefer keeping as single word if no explicit parts)
    
    # Step 2: Try learned patterns (for productive contractions like "exportarlo")
    # Build patterns cache if not already built for this vocabulary
    vocab_id = id(vocab)  # Use object ID as cache key
    if vocab_id not in _pattern_cache:
        _pattern_cache[vocab_id] = _build_contraction_patterns(vocab)
    
    patterns = _pattern_cache[vocab_id]
    
    # Try each pattern (longest suffixes first for better matching)
    form_normalized = _normalize_accents(form_lower)
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
    
    for suffix, pattern_data in sorted_patterns:
        if not form_normalized.endswith(suffix):
            continue
        
        # Extract potential base word
        base_candidate = form_normalized[:-len(suffix)] if len(suffix) < len(form_normalized) else None
        if not base_candidate or len(base_candidate) < 2:
            continue
        
        # Check if base word exists in vocabulary
        # Handle morphological variations:
        # 1. Exact match (with accent normalization)
        # 2. With dropped final 'r' (e.g., "pagá" matches "pagar")
        # 3. With added final 'r' (e.g., "pagar" matches "pagá" pattern)
        base_found = False
        base_upos = set()
        base_candidates_to_try = [
            base_candidate,  # Exact normalized form
            base_candidate + 'r',  # Add 'r' (in case base is "pagá" but vocab has "pagar")
            _normalize_base_for_clitic(base_candidate)  # Remove 'r' (in case base is "pagar" but form has "pagá")
        ]
        # Remove duplicates
        base_candidates_to_try = list(dict.fromkeys(base_candidates_to_try))
        
        for candidate in base_candidates_to_try:
            # Try exact match first
            if candidate in vocab or candidate.title() in vocab:
                base_found = True
                base_entry = vocab.get(candidate) or vocab.get(candidate.title())
                if base_entry:
                    analyses = base_entry if isinstance(base_entry, list) else [base_entry]
                    for analysis in analyses:
                        upos = analysis.get('upos', '_')
                        if upos and upos != '_':
                            base_upos.add(upos)
                break
            
            # Try with accent variations (e.g., "jugár" -> "jugar")
            if not base_found:
                for vocab_word in vocab.keys():
                    vocab_normalized = _normalize_accents(vocab_word.lower())
                    if vocab_normalized == candidate:
                        base_found = True
                        base_entry = vocab.get(vocab_word) or vocab.get(vocab_word.lower())
                        if base_entry:
                            analyses = base_entry if isinstance(base_entry, list) else [base_entry]
                            for analysis in analyses:
                                upos = analysis.get('upos', '_')
                                if upos and upos != '_':
                                    base_upos.add(upos)
                        break
            
            if base_found:
                break
        
        if not base_found:
            continue
        
        # Check UPOS requirement if pattern specifies it
        required_upos = pattern_data['required_upos']
        if required_upos:
            # Pattern requires specific UPOS (e.g., VERB for "-arlo" patterns)
            if not base_upos.intersection(required_upos):
                # Base word doesn't have required UPOS - don't split
                # This prevents splitting "Carlos" (NOUN) even if it ends with "los"
                continue
        
        # Pattern matches! Split the contraction
        suffix_parts = pattern_data['parts']
        
        # Find the actual base form in vocab (with proper accents and case)
        # Prefer the form that appears most frequently in vocab and has required UPOS
        # Also handle morphological variations (e.g., "pagar" vs "pagá")
        best_base_form = base_candidate  # Fallback to normalized form
        best_count = 0
        
        # Try all candidate variations
        for candidate in base_candidates_to_try:
            for vocab_word in vocab.keys():
                vocab_normalized = _normalize_accents(vocab_word.lower())
                # Match exact normalized form, or handle 'r' dropping/adding
                if (vocab_normalized == candidate or
                    vocab_normalized == _normalize_base_for_clitic(candidate) or
                    _normalize_base_for_clitic(vocab_normalized) == candidate):
                    # Check if this base form has the required UPOS
                    base_entry = vocab.get(vocab_word) or vocab.get(vocab_word.lower())
                    if base_entry:
                        analyses = base_entry if isinstance(base_entry, list) else [base_entry]
                        for analysis in analyses:
                            # Check UPOS match if required
                            if required_upos:
                                upos = analysis.get('upos', '_')
                                if upos not in required_upos:
                                    continue
                            
                            # Use the form with highest count
                            count = analysis.get('count', 0)
                            if count > best_count:
                                best_count = count
                                best_base_form = vocab_word
        
        # If we found a good match, use it; otherwise use the normalized candidate
        if best_count > 0:
            return [best_base_form] + suffix_parts
        else:
            # Fallback: use normalized form (shouldn't happen if base_found was True)
            return [base_candidate] + suffix_parts
    
    # No pattern matched - do not split
    return None



