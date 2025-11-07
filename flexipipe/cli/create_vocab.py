#!/usr/bin/env python3
"""
Extract vocabulary from TEITOK XML files for FlexiPipe.

Recursively processes all XML files in a folder and creates a vocabulary JSON file
with word-level annotations (form, lemma, upos, xpos, feats).
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import xml.etree.ElementTree as ET


def extract_form_from_tok(tok):
    """Extract form from <tok> element: use @form attribute if present, otherwise innerText."""
    form = tok.get('form', '').strip()
    if not form:
        # Use innerText if @form not present
        form = (tok.text or '').strip()
    return form


def get_attribute_with_fallback(elem, attr_names: str) -> str:
    """
    Get attribute value with fallback to multiple attributes (TEITOK inheritance).
    
    Args:
        elem: XML element (<tok> or <dtok>)
        attr_names: Comma-separated attribute names (e.g., 'nform,fform' or 'xpos')
    
    Returns:
        First non-empty attribute value found, or empty string if none found
    """
    attr_list = [a.strip() for a in attr_names.split(',')]
    for attr_name in attr_list:
        value = elem.get(attr_name, '').strip()
        if value:
            return value
    return ''


def extract_vocab_from_teitok_xml(file_path: Path, xpos_attr: str = 'xpos', reg_attr: str = 'reg', expan_attr: str = 'expan', track_transitions: bool = False):
    """
    Extract vocabulary entries from a TEITOK XML file.
    
    Args:
        file_path: Path to TEITOK XML file
        xpos_attr: Attribute name(s) for XPOS (default: 'xpos', can be 'pos' or 'msd', or comma-separated like 'pos,msd')
        reg_attr: Attribute name(s) for normalization/regularization (default: 'reg', can be 'nform' or 'nform,fform' for inheritance)
        track_transitions: If True, also track tag transition probabilities for Viterbi
    
    Returns:
        If track_transitions=False: Dictionary mapping (form, form_lower, annotation_key) -> count
        If track_transitions=True: Tuple of (word_annotations, transitions) where:
            - word_annotations: Dictionary mapping (form, form_lower, annotation_key) -> count
            - transitions: Dictionary with 'upos', 'xpos', 'start', 'sentences' keys
    """
    word_annotations = defaultdict(int)  # (form, form_lower, upos, xpos, feats, lemma, norm_form, expan_form) -> count
    
    # Track transitions if requested
    transitions = None
    capitalization_info = None
    if track_transitions:
        transitions = {
            'upos': defaultdict(int),  # (prev_upos, curr_upos) -> count
            'xpos': defaultdict(int),  # (prev_xpos, curr_xpos) -> count
            'start': defaultdict(int),  # upos -> count (sentence-start)
            'sentences': 0
        }
        # Track capitalization statistics: for each tag, count capitalized vs lowercase in non-initial positions
        # Structure: {tag: {'capitalized': count, 'lowercase': count}}
        capitalization_info = {
            'upos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0}),
            'xpos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0})
        }
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Helper function to check if a token is sentence-final punctuation
        def is_sentence_final_punct(form: str) -> bool:
            """Check if form is sentence-final punctuation."""
            if not form:
                return False
            # Common sentence-final punctuation
            return form in {'.', '!', '?', '…', '…', '。', '！', '？'}
        
        # Helper function to check if word is all-caps (should be ignored)
        def is_all_caps(form: str) -> bool:
            """Check if word is all uppercase (likely abbreviation or emphasis, not capitalization rule)."""
            if not form:
                return False
            # Check if all alphabetic characters are uppercase and there's at least one letter
            has_letter = any(c.isalpha() for c in form)
            return has_letter and form.isupper()
        
        for s in root.findall('.//s'):
            prev_upos = None
            prev_xpos = None
            is_first_token = True
            prev_form = None  # Track previous token form for punctuation-based sentence detection
            
            token_idx = 0
            for tok in s.findall('.//tok'):
                # Check if this tok has dtok children (contraction)
                dtoks = tok.findall('.//dtok')
                
                if dtoks:
                    # Contraction: process each dtok separately
                    for dt in dtoks:
                        # dtok always has @form attribute
                        form = dt.get('form', '').strip()
                        if not form:
                            continue
                        
                        form_lower = form.lower()
                        upos = dt.get('upos', '_')
                        # Use configurable xpos attribute(s) with fallback
                        xpos = get_attribute_with_fallback(dt, xpos_attr)
                        xpos = xpos if xpos else '_'
                        
                        # Get normalization and expansion with specified attributes (with fallback)
                        norm_form = get_attribute_with_fallback(dt, reg_attr) or '_'
                        expan_form = get_attribute_with_fallback(dt, expan_attr) or '_'
                        
                        # Skip entries without XPOS - they cause problems during tagging
                        # EXCEPTION: Include entries without XPOS if they have a normalization (reg) field
                        # This is important for normalization - we need form->reg mappings even without POS tags
                        if (not xpos or xpos == '_') and (not norm_form or norm_form == '_'):
                            continue
                        
                        feats = dt.get('feats', '_')
                        lemma = dt.get('lemma', '_').lower() if dt.get('lemma', '_') != '_' else '_'
                        
                        # Track capitalization statistics: only for non-initial positions
                        # Skip all-caps words for capitalization tracking (abbreviations, emphasis, etc.)
                        is_all_caps_word = is_all_caps(form)
                        
                        is_capitalized = form and form[0].isupper() and form_lower != form
                        # Check if sentence-initial: either first token in sentence OR after sentence-final punctuation
                        is_sentence_initial = (token_idx == 0) or (prev_form and is_sentence_final_punct(prev_form))
                        
                        # Only track capitalization if not all-caps and not sentence-initial
                        if capitalization_info and not is_sentence_initial and not is_all_caps_word:
                            # Track statistics for this tag in non-initial position
                            if upos and upos != '_':
                                if is_capitalized:
                                    capitalization_info['upos'][upos]['capitalized'] += 1
                                else:
                                    capitalization_info['upos'][upos]['lowercase'] += 1
                            if xpos and xpos != '_':
                                if is_capitalized:
                                    capitalization_info['xpos'][xpos]['capitalized'] += 1
                                else:
                                    capitalization_info['xpos'][xpos]['lowercase'] += 1
                        
                        # Store with both original case and lowercase for proper handling
                        annotation_key = (form, form_lower, upos, xpos, feats, lemma, norm_form, expan_form)
                        word_annotations[annotation_key] += 1
                        
                        prev_form = form
                        token_idx += 1
                        
                        # Track transitions for Viterbi
                        if track_transitions and transitions:
                            if is_first_token:
                                transitions['start'][upos] += 1
                                prev_upos = upos
                                prev_xpos = xpos
                                is_first_token = False
                            else:
                                if prev_upos and upos != '_':
                                    transitions['upos'][(prev_upos, upos)] += 1
                                if prev_xpos and xpos != '_':
                                    transitions['xpos'][(prev_xpos, xpos)] += 1
                                prev_upos = upos
                                prev_xpos = xpos
                else:
                    # Regular token: use @form or innerText
                    form = extract_form_from_tok(tok)
                    if not form:
                        continue
                    
                    form_lower = form.lower()
                    upos = tok.get('upos', '_')
                    # Use configurable xpos attribute(s) with fallback
                    xpos = get_attribute_with_fallback(tok, xpos_attr)
                    xpos = xpos if xpos else '_'
                    
                    # Get normalization and expansion with specified attributes (with fallback)
                    norm_form = get_attribute_with_fallback(tok, reg_attr) or '_'
                    expan_form = get_attribute_with_fallback(tok, expan_attr) or '_'
                    
                    # Skip entries without XPOS - they cause problems during tagging
                    # EXCEPTION: Include entries without XPOS if they have a normalization (reg) field
                    # This is important for normalization - we need form->reg mappings even without POS tags
                    if (not xpos or xpos == '_') and (not norm_form or norm_form == '_'):
                        continue
                    
                    feats = tok.get('feats', '_')
                    lemma = tok.get('lemma', '_').lower() if tok.get('lemma', '_') != '_' else '_'
                    
                    # Track capitalization statistics: only for non-initial positions
                    # Skip all-caps words for capitalization tracking (abbreviations, emphasis, etc.)
                    is_all_caps_word = is_all_caps(form)
                    
                    is_capitalized = form and form[0].isupper() and form_lower != form
                    # Check if sentence-initial: either first token in sentence OR after sentence-final punctuation
                    is_sentence_initial = (token_idx == 0) or (prev_form and is_sentence_final_punct(prev_form))
                    
                    # Only track capitalization if not all-caps and not sentence-initial
                    if capitalization_info and not is_sentence_initial and not is_all_caps_word:
                        # Track statistics for this tag in non-initial position
                        if upos and upos != '_':
                            if is_capitalized:
                                capitalization_info['upos'][upos]['capitalized'] += 1
                            else:
                                capitalization_info['upos'][upos]['lowercase'] += 1
                        if xpos and xpos != '_':
                            if is_capitalized:
                                capitalization_info['xpos'][xpos]['capitalized'] += 1
                            else:
                                capitalization_info['xpos'][xpos]['lowercase'] += 1
                    
                    # Store with both original case and lowercase for proper handling
                    annotation_key = (form, form_lower, upos, xpos, feats, lemma, norm_form, expan_form)
                    word_annotations[annotation_key] += 1
                    
                    prev_form = form
                    token_idx += 1
                    
                    # Track transitions for Viterbi
                    if track_transitions and transitions:
                        if is_first_token:
                            transitions['start'][upos] += 1
                            prev_upos = upos
                            prev_xpos = xpos
                            is_first_token = False
                        else:
                            if prev_upos and upos != '_':
                                transitions['upos'][(prev_upos, upos)] += 1
                            if prev_xpos and xpos != '_':
                                transitions['xpos'][(prev_xpos, xpos)] += 1
                            prev_upos = upos
                            prev_xpos = xpos
            
            # Count sentence
            if track_transitions and transitions and not is_first_token:
                transitions['sentences'] += 1
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    if track_transitions:
        return word_annotations, transitions, capitalization_info
    else:
        # Return empty capitalization_info when not tracking transitions
        return word_annotations, None, {'upos': {}, 'xpos': {}}


def build_vocabulary_from_folder(folder_path: Path, xpos_attr: str = 'xpos', reg_attr: str = 'reg', expan_attr: str = 'expan', debug: bool = False):
    """
    Build vocabulary from all TEITOK XML files in a folder (recursively).
    
    Args:
        folder_path: Root folder to search for XML files
        xpos_attr: Attribute name for XPOS (default: 'xpos', can be 'pos' or 'msd')
        reg_attr: Attribute name for normalization/regularization (default: 'reg', can be 'nform')
    
    Returns:
        Tuple of (vocab_dict, transition_probs) where:
        - vocab_dict: Dictionary with vocabulary entries in FlexiPipe format
        - transition_probs: Dictionary of transition probabilities for Viterbi tagging
    """
    # Collect all annotations with frequency
    # Separate tracking for case-sensitive forms and lowercase forms
    all_annotations_case = defaultdict(lambda: defaultdict(int))  # form (original case) -> (upos, xpos, feats, lemma, norm_form, expan_form) -> count
    all_annotations_lower = defaultdict(lambda: defaultdict(int))  # form_lower -> (upos, xpos, feats, lemma, norm_form, expan_form) -> count
    
    # Track transition probabilities for Viterbi tagging
    # Track tag sequences within sentences
    upos_transitions = defaultdict(int)  # (prev_upos, curr_upos) -> count
    xpos_transitions = defaultdict(int)  # (prev_xpos, curr_xpos) -> count
    upos_start_counts = defaultdict(int)  # upos -> count (for sentence-start probabilities)
    total_sentences = 0
    
    # Track capitalization statistics: for each tag, count capitalized vs lowercase in non-initial positions
    capitalizable_tags = {
        'upos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0}),
        'xpos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0})
    }
    
    # Find all XML files recursively
    xml_files = list(folder_path.rglob('*.xml'))
    
    if not xml_files:
        print(f"Warning: No XML files found in {folder_path}", file=sys.stderr)
        return {}, {}
    
    print(f"Found {len(xml_files)} XML files to process...", file=sys.stderr)
    
    # Progress reporting: show every 10% or every 100 files, whichever is more frequent
    total_files = len(xml_files)
    progress_interval = max(1, min(100, total_files // 10)) if total_files > 0 else 1
    
    for file_idx, xml_file in enumerate(xml_files, 1):
        if debug:
            print(f"Processing: {xml_file}", file=sys.stderr)
        elif file_idx % progress_interval == 0 or file_idx == total_files:
            # Show progress every N files or at the end
            percent = (file_idx * 100) // total_files if total_files > 0 else 0
            print(f"Processing: {file_idx}/{total_files} files ({percent}%)...", file=sys.stderr, end='\r')
        
        word_annotations, transitions, capitalization_info = extract_vocab_from_teitok_xml(xml_file, xpos_attr, reg_attr, expan_attr, track_transitions=True)
        
        # Merge into main annotations dictionary
        # Track both case-sensitive and lowercase separately
        for (form, form_lower, upos, xpos, feats, lemma, norm_form, expan_form), count in word_annotations.items():
            annotation_key = (upos, xpos, feats, lemma, norm_form, expan_form)
            
            # Always track lowercase (for fallback)
            all_annotations_lower[form_lower][annotation_key] += count
            
            # Track original case only if it differs from lowercase
            # This preserves case-sensitive distinctions (e.g., "Band" vs "band")
            if form != form_lower:
                all_annotations_case[form][annotation_key] += count
        
        # Merge transition probabilities
        if transitions:
            for (prev_upos, curr_upos), count in transitions.get('upos', {}).items():
                upos_transitions[(prev_upos, curr_upos)] += count
            for (prev_xpos, curr_xpos), count in transitions.get('xpos', {}).items():
                xpos_transitions[(prev_xpos, curr_xpos)] += count
            for upos, count in transitions.get('start', {}).items():
                upos_start_counts[upos] += count
            total_sentences += transitions.get('sentences', 0)
        
        # Merge capitalization statistics
        if capitalization_info:
            for tag, stats in capitalization_info['upos'].items():
                capitalizable_tags['upos'][tag]['capitalized'] += stats['capitalized']
                capitalizable_tags['upos'][tag]['lowercase'] += stats['lowercase']
            for tag, stats in capitalization_info['xpos'].items():
                capitalizable_tags['xpos'][tag]['capitalized'] += stats['capitalized']
                capitalizable_tags['xpos'][tag]['lowercase'] += stats['lowercase']
    
    # Print newline at the end to clear the progress line (if progress was shown)
    if not debug and total_files > 0:
        print()  # New line after all files processed
    
    # Build vocabulary using arrays for ambiguous words
    vocab = {}
    
    # Convert transition counts to probabilities (with smoothing)
    transition_probs = {}
    if total_sentences > 0:
        # UPOS transition probabilities
        upos_trans_probs = {}
        # Count total transitions from each state
        upos_from_counts = defaultdict(int)
        for (prev, curr), count in upos_transitions.items():
            upos_from_counts[prev] += count
        
        # Calculate probabilities (add smoothing for unseen transitions)
        smoothing = 0.01  # Small smoothing factor
        all_upos = set()
        for (prev, curr), count in upos_transitions.items():
            all_upos.add(prev)
            all_upos.add(curr)
        
        for prev in all_upos:
            total = upos_from_counts[prev] + smoothing * len(all_upos)
            upos_trans_probs[prev] = {}
            for curr in all_upos:
                count = upos_transitions.get((prev, curr), 0)
                prob = (count + smoothing) / total
                upos_trans_probs[prev][curr] = prob
        
        # XPOS transition probabilities (similar approach)
        xpos_trans_probs = {}
        xpos_from_counts = defaultdict(int)
        for (prev, curr), count in xpos_transitions.items():
            xpos_from_counts[prev] += count
        
        all_xpos = set()
        for (prev, curr), count in xpos_transitions.items():
            all_xpos.add(prev)
            all_xpos.add(curr)
        
        for prev in all_xpos:
            total = xpos_from_counts[prev] + smoothing * len(all_xpos)
            xpos_trans_probs[prev] = {}
            for curr in all_xpos:
                count = xpos_transitions.get((prev, curr), 0)
                prob = (count + smoothing) / total
                xpos_trans_probs[prev][curr] = prob
        
        # Start probabilities (sentence-initial)
        start_probs = {}
        total_starts = sum(upos_start_counts.values())
        if total_starts > 0:
            for upos in all_upos:
                count = upos_start_counts.get(upos, 0)
                start_probs[upos] = (count + smoothing) / (total_starts + smoothing * len(all_upos))
        else:
            # Uniform distribution if no data
            for upos in all_upos:
                start_probs[upos] = 1.0 / len(all_upos) if all_upos else 0.0
        
        transition_probs = {
            'upos': upos_trans_probs,
            'xpos': xpos_trans_probs,
            'start': start_probs,
            'sentences': total_sentences
        }
    
    # Convert annotation dictionaries to sentence format for shared vocabulary builder
    # This ensures consistency between create-vocab and training vocab building
    from flexipipe.vocabulary import build_vocab_from_sentences
    
    # Build "fake" sentences from annotations (for shared function)
    # Each token appears as many times as its count
    fake_sentences = []
    for form, annotations in all_annotations_case.items():
        for (upos, xpos, feats, lemma, norm_form, expan_form), count in annotations.items():
            for _ in range(count):
                fake_sentences.append([{
                    'form': form,
                    'upos': upos,
                    'xpos': xpos,
                    'feats': feats,
                    'lemma': lemma,
                    'norm_form': norm_form,
                    'expan': expan_form
                }])
    
    for form_lower, annotations in all_annotations_lower.items():
        # Only add if no case-sensitive form exists with same annotations
        case_sensitive_exists = False
        case_sensitive_annotations = set()
        for case_form in all_annotations_case.keys():
            if case_form.lower() == form_lower:
                case_sensitive_exists = True
                for ann_key in all_annotations_case[case_form].keys():
                    case_sensitive_annotations.add(ann_key)
                break
        
        if case_sensitive_exists:
            lowercase_annotations = set(annotations.keys())
            if lowercase_annotations == case_sensitive_annotations:
                continue  # Skip if same as case-sensitive
        
        for (upos, xpos, feats, lemma, norm_form, expan_form), count in annotations.items():
            for _ in range(count):
                fake_sentences.append([{
                    'form': form_lower,
                    'upos': upos,
                    'xpos': xpos,
                    'feats': feats,
                    'lemma': lemma,
                    'norm_form': norm_form,
                    'expan': expan_form
                }])
    
    # Use shared vocabulary building function
    vocab = build_vocab_from_sentences(fake_sentences)
    
    # Return both vocabulary and transition probabilities
    return vocab, transition_probs, capitalizable_tags


def main():
    parser = argparse.ArgumentParser(
        prog='flexipipe create-vocab',
        description='Extract vocabulary from TEITOK XML files or CoNLL-U files for FlexiPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract vocabulary from TEITOK XML folder with default attributes
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json
  
  # Extract vocabulary from CoNLL-U file(s)
  python -m flexipipe create-vocab train.conllu dev.conllu --output model_vocab.json
  
  # Extract vocabulary from multiple sources (folders and files)
  python -m flexipipe create-vocab /path/to/xml/folder train.conllu --output combined_vocab.json
  
  # Extract vocabulary using 'pos' attribute instead of 'xpos' (XML only)
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json --xpos-attr pos
  
  # Extract vocabulary using 'msd' attribute for XPOS (XML only)
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json --xpos-attr msd
  
  # Extract vocabulary using 'nform' attribute for normalization instead of 'reg' (XML only)
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json --reg nform
  
  # Extract vocabulary with attribute inheritance (TEITOK style, XML only)
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json --xpos-attr pos,msd --reg nform,fform --expan expan,fform
  
  # Extract vocabulary with custom corpus name
  python -m flexipipe create-vocab /path/to/xml/folder --output custom_vocab.json --corpus-name "Old Portuguese Letters"
        """
    )
    
    parser.add_argument('folders', type=Path, nargs='+',
                       help='One or more root folders containing TEITOK XML files (will search recursively), '
                            'or individual CoNLL-U files (.conllu). '
                            'Vocabularies from multiple sources will be merged together.')
    parser.add_argument('--output', '-o', type=Path, default=Path('custom_vocab.json'),
                       help='Output vocabulary file (default: custom_vocab.json)')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='Attribute name(s) for XPOS in TEITOK XML (default: xpos). '
                            'Use "pos" or "msd" for corpora not in UD format. '
                            'For inheritance, use comma-separated values like "pos,msd" (tries @pos first, then @msd).')
    parser.add_argument('--language', type=str,
                       help='Language code (e.g., "es" for Spanish, "de" for German) for capitalization rules. '
                            'Used to determine which tags can be capitalized in non-initial positions.')
    parser.add_argument('--reg', default='reg',
                       help='Attribute name(s) for normalization/regularization in TEITOK XML (default: reg, can be nform). '
                            'For inheritance, use comma-separated values like "nform,fform" (tries @nform first, then @fform). '
                            'This corresponds to the Reg= field in CoNLL-U MISC column.')
    parser.add_argument('--expan', default='expan',
                       help='Attribute name(s) for expansion in TEITOK XML (default: expan; older projects may use fform). '
                            'For inheritance, use comma-separated values like "expan,fform".')
    parser.add_argument('--corpus-name', type=str, default=None,
                       help='Name of the corpus (default: folder name)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (prints each filename being processed)')
    
    args = parser.parse_args()
    
    # Import CoNLL-U loading function
    from flexipipe.data_loading import load_conllu_file
    
    # Separate folders (for XML) and files (for CoNLL-U)
    xml_folders = []
    conllu_files = []
    
    for path in args.folders:
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        
        if path.is_file() and path.suffix.lower() in ['.conllu', '.conll']:
            conllu_files.append(path)
        elif path.is_dir():
            xml_folders.append(path)
        else:
            print(f"Error: {path} is neither a directory nor a CoNLL-U file (.conllu/.conll)", file=sys.stderr)
            sys.exit(1)
    
    # Build vocabulary from all sources
    total_sources = len(xml_folders) + len(conllu_files)
    print(f"Extracting vocabulary from {total_sources} source(s):", file=sys.stderr)
    for folder in xml_folders:
        print(f"  - {folder} (XML folder)", file=sys.stderr)
    for conllu_file in conllu_files:
        print(f"  - {conllu_file} (CoNLL-U file)", file=sys.stderr)
    
    if xml_folders:
        print(f"Using XPOS attribute: {args.xpos_attr}", file=sys.stderr)
        print(f"Using normalization attribute: {args.reg}", file=sys.stderr)
        print(f"Using expansion attribute: {args.expan}", file=sys.stderr)
    
    # Process each source and merge vocabularies
    merged_vocab = {}
    # Store transitions in nested format: {prev: {curr: prob}} for upos/xpos, {tag: prob} for start
    merged_transitions = {
        'upos': defaultdict(lambda: defaultdict(float)),  # {prev: {curr: prob}}
        'xpos': defaultdict(lambda: defaultdict(float)),  # {prev: {curr: prob}}
        'start': defaultdict(float),  # {tag: prob}
        'sentences': 0
    }
    merged_capitalizable_tags = {
        'upos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0}),
        'xpos': defaultdict(lambda: {'capitalized': 0, 'lowercase': 0})
    }
    
    source_idx = 0
    
    # Process XML folders
    for folder in xml_folders:
        source_idx += 1
        print(f"\n[{source_idx}/{total_sources}] Processing XML folder: {folder}", file=sys.stderr)
        vocab, transition_probs, capitalizable_tags = build_vocabulary_from_folder(folder, args.xpos_attr, args.reg, args.expan, debug=args.debug)
        
        if not vocab:
            print(f"Warning: No vocabulary entries found in {folder}. Skipping.", file=sys.stderr)
            continue
        
        # Merge vocabularies: combine entries and accumulate counts
        for form, entry in vocab.items():
            if form in merged_vocab:
                # Entry already exists - need to merge
                existing_entry = merged_vocab[form]
                new_entry = entry
                
                # Handle both single dict and list formats
                existing_list = existing_entry if isinstance(existing_entry, list) else [existing_entry]
                new_list = new_entry if isinstance(new_entry, list) else [new_entry]
                
                # Merge analyses: combine by annotation key and sum counts
                merged_analyses = {}
                for analysis in existing_list + new_list:
                    # Create key from annotation (excluding count)
                    key = (
                        analysis.get('upos', '_'),
                        analysis.get('xpos', '_'),
                        analysis.get('feats', '_'),
                        analysis.get('lemma', '_'),
                        analysis.get('reg', '_'),
                        analysis.get('expan', '_')
                    )
                    if key in merged_analyses:
                        # Same annotation: add counts
                        merged_analyses[key]['count'] += analysis.get('count', 1)
                    else:
                        # New annotation: add it
                        merged_analyses[key] = analysis.copy()
                
                # Convert back to list or single dict
                merged_list = list(merged_analyses.values())
                if len(merged_list) == 1:
                    merged_vocab[form] = merged_list[0]
                else:
                    merged_vocab[form] = merged_list
            else:
                # New entry: add it directly
                merged_vocab[form] = entry
        
        # Merge transition probabilities
        # Note: transition_probs has structure: {'upos': {prev: {curr: prob}}, 'xpos': {prev: {curr: prob}}, 'start': {tag: prob}, 'sentences': N}
        # We merge by averaging probabilities (weighted by sentence count if available)
        if transition_probs:
            folder_sentences = transition_probs.get('sentences', 1)
            total_sentences_before = merged_transitions['sentences']
            
            for trans_type in ['upos', 'xpos']:
                if trans_type in transition_probs:
                    # transition_probs[trans_type] is {prev: {curr: prob}}
                    for prev, curr_dict in transition_probs[trans_type].items():
                        for curr, prob in curr_dict.items():
                            # Weight by sentence count for proper averaging
                            if total_sentences_before > 0:
                                # Weighted average: (old_total * old_prob + new_sentences * new_prob) / (old_total + new_sentences)
                                old_prob = merged_transitions[trans_type][prev].get(curr, 0.0)
                                if old_prob > 0:
                                    merged_transitions[trans_type][prev][curr] = (
                                        (total_sentences_before * old_prob + folder_sentences * prob) /
                                        (total_sentences_before + folder_sentences)
                                    )
                                else:
                                    merged_transitions[trans_type][prev][curr] = prob
                            else:
                                merged_transitions[trans_type][prev][curr] = prob
            
            if 'start' in transition_probs:
                # transition_probs['start'] is {tag: prob}
                for tag, prob in transition_probs['start'].items():
                    # Weighted average for start probabilities
                    if total_sentences_before > 0:
                        old_prob = merged_transitions['start'].get(tag, 0.0)
                        if old_prob > 0:
                            merged_transitions['start'][tag] = (
                                (total_sentences_before * old_prob + folder_sentences * prob) /
                                (total_sentences_before + folder_sentences)
                            )
                        else:
                            merged_transitions['start'][tag] = prob
                    else:
                        merged_transitions['start'][tag] = prob
            
            merged_transitions['sentences'] += transition_probs.get('sentences', 0)
        
        # Merge capitalization statistics
        for tag, stats in capitalizable_tags['upos'].items():
            merged_capitalizable_tags['upos'][tag]['capitalized'] += stats['capitalized']
            merged_capitalizable_tags['upos'][tag]['lowercase'] += stats['lowercase']
        for tag, stats in capitalizable_tags['xpos'].items():
            merged_capitalizable_tags['xpos'][tag]['capitalized'] += stats['capitalized']
            merged_capitalizable_tags['xpos'][tag]['lowercase'] += stats['lowercase']
    
    # Process CoNLL-U files
    from flexipipe.vocabulary import build_vocab_from_sentences
    for conllu_file in conllu_files:
        source_idx += 1
        print(f"\n[{source_idx}/{total_sources}] Processing CoNLL-U file: {conllu_file}", file=sys.stderr)
        sentences = load_conllu_file(conllu_file)
        print(f"  Loaded {len(sentences)} sentences", file=sys.stderr)
        
        # Build vocabulary from sentences (no transition probabilities for CoNLL-U files)
        vocab = build_vocab_from_sentences(sentences)
        
        if not vocab:
            print(f"Warning: No vocabulary entries found in {conllu_file}. Skipping.", file=sys.stderr)
            continue
        
        # Merge vocabularies (same logic as XML folders)
        for form, entry in vocab.items():
            if form in merged_vocab:
                existing_entry = merged_vocab[form]
                new_entry = entry
                existing_list = existing_entry if isinstance(existing_entry, list) else [existing_entry]
                new_list = new_entry if isinstance(new_entry, list) else [new_entry]
                merged_analyses = {}
                for analysis in existing_list + new_list:
                    key = (
                        analysis.get('upos', '_'),
                        analysis.get('xpos', '_'),
                        analysis.get('feats', '_'),
                        analysis.get('lemma', '_'),
                        analysis.get('reg', '_'),
                        analysis.get('expan', '_')
                    )
                    if key in merged_analyses:
                        merged_analyses[key]['count'] += analysis.get('count', 1)
                    else:
                        merged_analyses[key] = analysis.copy()
                merged_list = list(merged_analyses.values())
                if len(merged_list) == 1:
                    merged_vocab[form] = merged_list[0]
                else:
                    merged_vocab[form] = merged_list
            else:
                merged_vocab[form] = entry
    
    # Convert defaultdicts to regular dicts for JSON serialization
    transition_probs_final = {
        'upos': {prev: dict(curr_dict) for prev, curr_dict in merged_transitions['upos'].items()},
        'xpos': {prev: dict(curr_dict) for prev, curr_dict in merged_transitions['xpos'].items()},
        'start': dict(merged_transitions['start']),
        'sentences': merged_transitions['sentences']
    }
    
    vocab = merged_vocab
    transition_probs = transition_probs_final
    
    if not vocab:
        print("Warning: No vocabulary entries found. Check that XML files contain <tok> or <dtok> elements.", file=sys.stderr)
        sys.exit(1)
    
    # Calculate statistics
    word_entries = sum(1 for k in vocab.keys() if ':' not in k)
    xpos_entries = len(vocab) - word_entries
    
    # Count ambiguous words (those with multiple analyses)
    ambiguous_words = sum(1 for v in vocab.values() if isinstance(v, list))
    
    # Count total word types (unique forms)
    total_word_types = word_entries
    
    # Count total analyses (sum of all analyses for all words)
    total_analyses = 0
    for entry in vocab.values():
        if isinstance(entry, list):
            total_analyses += len(entry)
        else:
            total_analyses += 1
    
    # Transition statistics
    upos_trans_count = sum(len(v) for v in transition_probs.get('upos', {}).values()) if transition_probs else 0
    xpos_trans_count = sum(len(v) for v in transition_probs.get('xpos', {}).values()) if transition_probs else 0
    start_count = len(transition_probs.get('start', {})) if transition_probs else 0
    
    # Determine corpus name
    # If explicit corpus name provided, use it
    if args.corpus_name:
        corpus_name = args.corpus_name
    else:
        # For single source, use its name
        if total_sources == 1:
            source = xml_folders[0] if xml_folders else conllu_files[0]
            if source.is_dir():
                # Check if folder name is generic (like "xmlfiles", "xml", "data", etc.)
                generic_folder_names = {'xmlfiles', 'xml', 'data', 'files', 'xml_data', 'xml_files', 'source', 'src'}
                folder_name_lower = source.name.lower()
                if folder_name_lower in generic_folder_names and source.parent != source:
                    corpus_name = source.parent.name
                else:
                    corpus_name = source.name
            else:
                # CoNLL-U file: use filename without extension
                corpus_name = source.stem
        else:
            # Multiple sources: use combined name
            corpus_name = f"combined_{total_sources}_sources"
    
    # Count XML files processed from all folders
    xml_file_count = 0
    for folder in xml_folders:
        xml_files = list(folder.rglob('*.xml'))
        xml_file_count += len(xml_files)
    
    # Create metadata
    metadata = {
        'corpus_name': corpus_name,
        'creation_date': datetime.now().isoformat(),
        'source_folders': [str(f) for f in xml_folders],
        'source_files': [str(f) for f in conllu_files],
        'xpos_attr': args.xpos_attr,
        'reg_attr': args.reg,
        'language': args.language if args.language else None,
        'capitalizable_tags': {
            'upos': {tag: dict(stats) for tag, stats in merged_capitalizable_tags['upos'].items()},
            'xpos': {tag: dict(stats) for tag, stats in merged_capitalizable_tags['xpos'].items()}
        },
        'vocab_stats': {
            'total_entries': len(vocab),
            'word_entries': word_entries,
            'xpos_specific_entries': xpos_entries,
            'ambiguous_words': ambiguous_words,
            'total_word_types': total_word_types,
            'total_analyses': total_analyses
        },
        'transition_stats': {
            'upos_transitions': upos_trans_count,
            'xpos_transitions': xpos_trans_count,
            'start_states': start_count,
            'has_transitions': bool(transition_probs)
        },
        'source_stats': {
            'xml_files_processed': xml_file_count
        }
    }
    
    # Save vocabulary with transition probabilities and metadata
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': metadata,
        'vocab': vocab,
        'transitions': transition_probs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nVocabulary saved to: {output_path}", file=sys.stderr)
    print(f"\nVocabulary Statistics:", file=sys.stderr)
    print(f"  Total entries: {len(vocab)}", file=sys.stderr)
    print(f"  Word entries: {word_entries}", file=sys.stderr)
    print(f"  XPOS-specific lemma entries: {xpos_entries}", file=sys.stderr)
    print(f"  Ambiguous words: {ambiguous_words}", file=sys.stderr)
    print(f"  Total analyses: {total_analyses}", file=sys.stderr)
    
    if transition_probs:
        print(f"\nTransition Probabilities:", file=sys.stderr)
        print(f"  UPOS transitions: {upos_trans_count}", file=sys.stderr)
        print(f"  XPOS transitions: {xpos_trans_count}", file=sys.stderr)
        print(f"  Start states: {start_count}", file=sys.stderr)
    
    print(f"\nSource Information:", file=sys.stderr)
    print(f"  Corpus: {corpus_name}", file=sys.stderr)
    print(f"  XML files processed: {xml_file_count}", file=sys.stderr)
    print(f"  Created: {metadata['creation_date']}", file=sys.stderr)


if __name__ == '__main__':
    main()

