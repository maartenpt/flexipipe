"""
Lemmatization module for FlexiPipe.
"""
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys


def build_lemmatization_patterns(vocab: Dict) -> Tuple[Dict, Dict]:
    """
    Build lemmatization patterns from vocabulary (like TreeTagger/Neotag).
    
    Extracts suffix transformation patterns grouped by XPOS:
    - Example: "calidades" (NCFP000) -> "calidad" → pattern: -des -> -d for NCFP000
    
    IMPORTANT: If a vocabulary entry has a `reg` (normalized form) field, extract patterns
    from the `reg` form → lemma, NOT from the original form → lemma. This ensures that
    patterns are based on normalized forms, which is what we'll use for lemmatization.
    
    Patterns are stored as: {xpos: [(suffix_from, suffix_to, min_base_length, count), ...]}
    Sorted by suffix length (longest first) for longest-match application.
    
    Args:
        vocab: Vocabulary dictionary with word forms and their analyses
    
    Returns:
        Tuple of (patterns_dict, pattern_info_dict) where:
        - patterns_dict: {xpos: [(suffix_from, suffix_to, min_base, count), ...]}
        - pattern_info_dict: {(xpos, suffix_from, suffix_to): (min_base, suffix_len, count)}
    """
    patterns_by_xpos = defaultdict(list)  # xpos -> list of (suffix_from, suffix_to, min_length)
    
    for form, entry in vocab.items():
        # Skip XPOS-specific entries (they're redundant)
        if ':' in form:
            continue
        
        form_lower = form.lower()
        analyses = entry if isinstance(entry, list) else [entry]
        
        for analysis in analyses:
            lemma = analysis.get('lemma', '_')
            xpos = analysis.get('xpos', '_')
            reg = analysis.get('reg', '_')
            expan = analysis.get('expan', '_')
            
            if lemma == '_' or xpos == '_':
                continue
            
            # Skip entries with expan field - these are abbreviations, not morphological variants
            # The expansion is the actual form, so we shouldn't use the abbreviation for pattern building
            # Example: "sra" with expan "señora" should not create patterns from sra->señor/señora
            if expan and expan != '_' and expan.lower() != form_lower:
                continue
            
            # If entry has reg field, use reg form for pattern extraction (not original form)
            # This is crucial: lemmatization patterns should be based on normalized forms
            pattern_form = reg if reg and reg != '_' and reg != form else form_lower
            pattern_form_lower = pattern_form.lower()
            lemma_lower = lemma.lower()
            
            # Extract suffix transformation pattern (TreeTagger/Neotag style)
            # Strategy: find optimal prefix that gives best suffix pattern
            # Goal: prefer patterns like -des → -d over -es → '' (deletion patterns)
            # Example: "calidades" -> "calidad": should yield -des → -d, not -es → ''
            
            min_len = min(len(pattern_form_lower), len(lemma_lower))
            
            # Find longest common prefix (from the start)
            max_prefix_len = 0
            for i in range(min_len):
                if pattern_form_lower[i] == lemma_lower[i]:
                    max_prefix_len = i + 1
                else:
                    break
            
            if max_prefix_len > 0:
                # Try different prefix lengths to find the best pattern
                # Prefer patterns with non-empty suffix_to (transformation) over deletion (empty suffix_to)
                best_prefix_len = max_prefix_len
                best_suffix_from = pattern_form_lower[max_prefix_len:]
                best_suffix_to = lemma_lower[max_prefix_len:]
                
                # If we got a deletion pattern (empty suffix_to), try shorter prefixes
                if not best_suffix_to and len(best_suffix_from) > 1:
                    # Try progressively shorter prefixes to find a better pattern
                    for try_prefix_len in range(max_prefix_len - 1, 0, -1):
                        try_suffix_from = pattern_form_lower[try_prefix_len:]
                        try_suffix_to = lemma_lower[try_prefix_len:]
                        # Prefer this if it gives a non-empty suffix_to
                        if try_suffix_to:
                            best_prefix_len = try_prefix_len
                            best_suffix_from = try_suffix_from
                            best_suffix_to = try_suffix_to
                            break  # Stop at first non-empty suffix_to (longest prefix with transformation)
                
                suffix_from = best_suffix_from
                suffix_to = best_suffix_to
                min_base = best_prefix_len
                
                # Filter out unrealistic patterns:
                # 1. Very short suffix patterns that add characters (likely errors)
                #    Example: "o" -> "oto" is unrealistic (should be longer suffix or deletion)
                # 2. Patterns where suffix_to is much longer than suffix_from (unlikely morphological change)
                #    Example: "o" -> "oto" (1 char -> 3 chars) is suspicious
                if len(suffix_from) <= 1 and len(suffix_to) > len(suffix_from) + 1:
                    # Skip: very short suffix adding more than 1 character is unrealistic
                    continue
                if len(suffix_from) == 2 and len(suffix_to) > len(suffix_from) + 2:
                    # Skip: 2-char suffix adding more than 2 characters is suspicious
                    continue
                
                # IMPORTANT: Include "no change" patterns (form == lemma) as well
                # This prevents rare transformation patterns (like -a → -o for animate nouns)
                # from being over-applied to words that should have no change
                # Example: Most nouns ending in -a have lemma ending in -a (no change),
                # but a few animate nouns have lemma ending in -o. Without tracking
                # "no change" patterns, the rare -a → -o pattern gets applied incorrectly.
                patterns_by_xpos[xpos].append((suffix_from, suffix_to, min_base))
    
    # Count frequency of patterns (number of distinct lemma/form pairs per pattern)
    # This is the count of distinct lemma/form pairs, not token frequency
    pattern_counts = defaultdict(int)  # (xpos, suffix_from, suffix_to) -> count of distinct pairs
    
    for xpos, pattern_list in patterns_by_xpos.items():
        for suffix_from, suffix_to, min_base in pattern_list:
            pattern_counts[(xpos, suffix_from, suffix_to)] += 1
    
    # Build final patterns: keep only patterns that appear multiple times (more reliable)
    # Store count with each pattern for conflict resolution
    final_patterns = {}
    pattern_info = {}  # (xpos, suffix_from, suffix_to) -> (min_base, suffix_len, count)
    
    for xpos in patterns_by_xpos.keys():
        best_patterns = {}
        for suffix_from, suffix_to, min_base in patterns_by_xpos[xpos]:
            count = pattern_counts[(xpos, suffix_from, suffix_to)]
            # Require higher count for suspicious patterns:
            # - Patterns that change accented characters to unaccented (e.g., "ón" -> "o")
            #   These are often rare exceptions, not general rules
            # - Patterns with very short suffixes that change significantly
            min_required_count = 2
            if len(suffix_from) >= 2 and len(suffix_to) >= 1:
                # Check if pattern removes accent: contains accented char in suffix_from but not in suffix_to
                has_accent_in_from = any(c in suffix_from for c in 'áéíóúÁÉÍÓÚñÑçÇ')
                has_accent_in_to = any(c in suffix_to for c in 'áéíóúÁÉÍÓÚñÑçÇ')
                if has_accent_in_from and not has_accent_in_to:
                    # Accent removal pattern - require higher count (at least 5) to be reliable
                    min_required_count = 5
            
            if count < min_required_count:
                continue
            
            suffix_len = len(suffix_from)
            key = (suffix_from, suffix_to)
            candidate = (min_base, suffix_len, count)
            existing = best_patterns.get(key)
            if not existing or count > existing[2] or (count == existing[2] and min_base > existing[0]):
                best_patterns[key] = candidate
                # Store pattern info for conflict resolution (using best candidate)
                pattern_info[(xpos, suffix_from, suffix_to)] = (min_base, suffix_len, count)
        
        # Convert best patterns to sorted list
        xpos_patterns = []
        for (suffix_from, suffix_to), (min_base, suffix_len, count) in best_patterns.items():
            xpos_patterns.append((suffix_from, suffix_to, min_base, suffix_len, count))
        xpos_patterns.sort(key=lambda x: (x[3], x[4]), reverse=True)
        final_patterns[xpos] = [(p[0], p[1], p[2], p[4]) for p in xpos_patterns]
    
    return final_patterns, pattern_info


def apply_lemmatization_patterns(form: str, xpos: str, patterns: Dict, known_lemmas: Optional[set] = None, debug: bool = False) -> str:
    """
    Apply lemmatization patterns to OOV word (TreeTagger/Neotag style).
    
    Finds all matching patterns and applies the one with highest count of distinct lemma/form pairs.
    When multiple patterns match the same suffix length, picks the one with most examples.
    Example: 
    - "estudiantes" with patterns (-es, ""), (-des, "d"), (-edes, "ed") -> "estudiante" (uses longest: -edes)
    - For "palabrades" ending in -ades: if both (-ade, "") and (-ad, "") match, pick the one with highest count
    
    Args:
        form: Word form to lemmatize
        xpos: XPOS tag for pattern matching
        patterns: Dictionary of patterns {xpos: [(suffix_from, suffix_to, min_base, count), ...]}
        debug: If True, print debug information
    
    Returns:
        Lemma or '_' if no pattern matches
    """
    if not patterns or xpos not in patterns:
        if debug:
            print(f"[DEBUG LEMMA PATTERNS] No patterns for XPOS '{xpos}'", file=sys.stderr)
        return '_'
    
    form_lower = form.lower()
    pattern_list = patterns[xpos]
    
    if debug:
        print(f"[DEBUG LEMMA PATTERNS] Found {len(pattern_list)} patterns for XPOS '{xpos}'", file=sys.stderr)
    
    # Find all matching patterns (patterns where suffix_from matches the end of the form)
    matching_patterns = []
    
    for pattern_tuple in pattern_list:
        if len(pattern_tuple) == 4:
            suffix_from, suffix_to, min_base, count = pattern_tuple
        else:
            # Backward compatibility: old format without count
            suffix_from, suffix_to, min_base = pattern_tuple[:3]
            count = 1  # Default count if not available
        
        # Check if form matches this pattern
        if suffix_from:
            # Include deletion patterns (empty suffix_to) if they have high enough count
            # Deletion patterns like -es → '' are valid for cases like "mercedes" → "merced"
            # But we need to be careful - only allow deletion if count is high enough (reliable)
            if not suffix_to and suffix_from:
                # Empty suffix_to: deletion pattern (e.g., -es → '' for "mercedes" -> "merced")
                # Only allow if count is high enough (at least 3) to be reliable
                if count < 3:
                    # Skip unreliable deletion patterns
                    continue
            if form_lower.endswith(suffix_from):
                base = form_lower[:-len(suffix_from)]
                if len(base) >= min_base:
                    lemma = base + suffix_to  # Will be just "base" if suffix_to is empty
                    # Verify lemma is reasonable (not empty, not too short)
                    if len(lemma) >= 2:
                        # Additional validation: filter out unrealistic patterns at application time
                        # This catches patterns that might have slipped through during building
                        # 1. Very short suffix patterns that add characters (likely errors)
                        if len(suffix_from) <= 1 and len(suffix_to) > len(suffix_from) + 1:
                            # Skip: very short suffix adding more than 1 character is unrealistic
                            # Example: "o" -> "oto" (1 char -> 3 chars) is suspicious
                            if debug:
                                print(f"[DEBUG LEMMA PATTERNS] Skipping unrealistic pattern: suffix_from='{suffix_from}' -> suffix_to='{suffix_to}' (too short, adds too many chars)", file=sys.stderr)
                            continue
                        if len(suffix_from) == 2 and len(suffix_to) > len(suffix_from) + 2:
                            # Skip: 2-char suffix adding more than 2 characters is suspicious
                            if debug:
                                print(f"[DEBUG LEMMA PATTERNS] Skipping unrealistic pattern: suffix_from='{suffix_from}' -> suffix_to='{suffix_to}' (adds too many chars)", file=sys.stderr)
                            continue
                        
                        # Store: (suffix_length, count, lemma)
                        # "No change" patterns (suffix_from == suffix_to) are valid and important
                        # Deletion patterns (suffix_to == '') are also valid if count is high
                        lemma_known = 1 if known_lemmas and lemma in known_lemmas else 0
                        matching_patterns.append((lemma_known, len(suffix_from), count, lemma))
                        if debug:
                            print(f"[DEBUG LEMMA PATTERNS] Pattern matches: suffix_from='{suffix_from}' -> suffix_to='{suffix_to}', lemma='{lemma}', count={count}", file=sys.stderr)
        elif suffix_to:
            # Pattern: add suffix_to (less common, but possible)
            if len(form_lower) >= min_base:
                lemma = form_lower + suffix_to
                if len(lemma) >= 2:
                    lemma_known = 1 if known_lemmas and lemma in known_lemmas else 0
                    matching_patterns.append((lemma_known, 0, count, lemma))  # Suffix length 0 for add patterns
    
    if not matching_patterns:
        if debug:
            print(f"[DEBUG LEMMA PATTERNS] No matching patterns for form '{form}'", file=sys.stderr)
        return '_'
    
    # Resolve conflicts: if multiple patterns match, prefer:
    # 1. Longest suffix (most specific match)
    # 2. Highest count (most distinct lemma/form pairs) when suffix lengths are equal
    matching_patterns.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    
    if debug:
        best = matching_patterns[0]
        print(f"[DEBUG LEMMA PATTERNS] Selected best pattern: known={bool(best[0])}, suffix_length={best[1]}, count={best[2]}, lemma='{best[3]}'", file=sys.stderr)
    
    # Return lemma from the best matching pattern
    return matching_patterns[0][3]

