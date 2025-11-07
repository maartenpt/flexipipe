#!/usr/bin/env python3
"""
Tag command for FlexiPipe - supports stdin/stdout for pipeline usage.
"""

import sys
import argparse
import json
from pathlib import Path

# Import from appropriate modules
from flexipipe.config import FlexiPipeConfig
from flexipipe.tagger import FlexiPipeTagger
from flexipipe.core import (
    set_conllu_expansion_key,
    load_teitok_xml, build_vocabulary
)

def main():
    parser = argparse.ArgumentParser(
        description='FlexiPipe tag: Tag text (supports stdin/stdout for pipelines)',
        prog='flexipipe tag'
    )
    parser.add_argument('input', nargs='?', type=str, default='-',
                       help='Input file (use "-" or omit for stdin)')
    parser.add_argument('--output', '-o', type=str, default='-',
                       help='Output file (use "-" or omit for stdout, default: stdout)')
    parser.add_argument('--input-format', choices=['conllu', 'teitok', 'plain', 'auto'], default='auto',
                       help='Input format (default: auto - detected from file extension)')
    parser.add_argument('--output-format', choices=['conllu', 'plain', 'text', 'plain-tagged', 'teitok'],
                       help='Output format (defaults to input format or conllu)')
    parser.add_argument('--segment', action='store_true',
                       help='Segment raw text into sentences')
    parser.add_argument('--tokenize', action='store_true',
                       help='Tokenize sentences into words')
    parser.add_argument('--tokenization-method', choices=['ud', 'bert', 'auto'], default='auto',
                       help='Tokenization method: "ud" (rule-based), "bert" (BERT-based, for Chinese/Japanese/etc.), or "auto" (default: auto - uses BERT if available, otherwise UD)')
    parser.add_argument('--model', type=Path, help='Path to trained model')
    parser.add_argument('--bert-model', default='bert-base-multilingual-cased',
                       help='BERT base model if no trained model')
    parser.add_argument('--vocab', type=Path, nargs='+',
                       help='Vocabulary file(s) (JSON). Multiple files can be specified; later files override earlier ones for the same words.')
    parser.add_argument('--lexicon', type=Path, nargs='+',
                       help='Lexicon file(s) (UniMorph or JSON without counts). Used as fallback for OOV words only, not merged with vocabulary.')
    parser.add_argument('--vocab-priority', action='store_true',
                       help='Give vocabulary priority over model predictions')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='If model confidence < threshold, use vocabulary predictions (confidence-based blending, default: 0.7)')
    parser.add_argument('--respect-existing', action='store_true', default=True,
                       help='Respect existing annotations (default: True)')
    parser.add_argument('--no-respect-existing', dest='respect_existing', action='store_false',
                       help='Ignore existing annotations')
    parser.add_argument('--parse', action='store_true',
                       help='Run parsing (predict head and deprel)')
    parser.add_argument('--tag-only', action='store_true',
                       help='Only tag (UPOS/XPOS/FEATS), skip parsing')
    parser.add_argument('--parse-only', action='store_true',
                       help='Only parse (assumes tags already exist)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize orthographic variants')
    parser.add_argument('--normalization-strategy', choices=['conservative', 'similarity', 'aggressive'], 
                       default='conservative',
                       help='Normalization strategy: conservative (only explicit mappings, default), similarity (use similarity matching), aggressive (not yet implemented)')
    parser.add_argument('--normalization-attr', default='reg',
                       help='TEITOK attribute name for normalization')
    parser.add_argument('--expan', default='expan',
                       help='TEITOK attribute name or CoNLL-U MISC key for expansion')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='TEITOK attribute name(s) for XPOS')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (shows pipeline steps and timing)')
    parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                       help='Anchor for learning inflection suffixes')
    parser.add_argument('--use-xpos-for-tagging', action='store_true',
                       help='Use XPOS for tagging/lemmatization instead of UPOS+FEATS (default: use UPOS+FEATS)')
    parser.add_argument('--tagset', type=Path,
                       help='Path to TEITOK tagset XML file (for XPOS to UPOS/FEATS mapping and agreement checking)')
    parser.add_argument('--guess-ud', action='store_true',
                       help='Output guessed UPOS and FEATS derived from XPOS using tagset definition')
    
    args = parser.parse_args()
    
    # Handle stdin/stdout
    input_path = None
    output_path = None
    use_stdin = (args.input == '-' or args.input is None or not args.input)
    use_stdout = (args.output == '-' or args.output is None or not args.output)
    is_stdin = False  # Track if we're reading from stdin for verbose output
    
    # Check if input is from terminal or pipe
    if use_stdin:
        if sys.stdin.isatty():
            print("Error: No input provided and stdin is a terminal", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        is_stdin = True
        # Create temporary file from stdin
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(sys.stdin.read())
            input_path = Path(tmp.name)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
    
    if not use_stdout:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure expansion key
    set_conllu_expansion_key(args.expan)
    load_teitok_xml._xpos_attr = args.xpos_attr
    load_teitok_xml._expan_attr = args.expan
    
    # Determine parse/tag settings
    parse_enabled = args.parse
    if args.tag_only:
        parse_enabled = False
    if args.parse_only:
        parse_enabled = True
    
    # Map normalization strategy to config
    normalization_strategy = args.normalization_strategy
    conservative_normalization = (normalization_strategy == 'conservative')
    if normalization_strategy == 'aggressive':
        print("Warning: 'aggressive' normalization strategy is not yet implemented, using 'similarity' instead", file=sys.stderr)
        normalization_strategy = 'similarity'
        conservative_normalization = False
    
    # Set similarity threshold based on normalization strategy
    similarity_threshold = 0.8 if conservative_normalization else 0.7
    
    # Build config
    config = FlexiPipeConfig(
        bert_model=args.bert_model,
        respect_existing=args.respect_existing,
        parse=parse_enabled,
        tag_only=args.tag_only,
        parse_only=args.parse_only,
        vocab_priority=args.vocab_priority,
        confidence_threshold=args.confidence_threshold,
        normalize=args.normalize,
        conservative_normalization=conservative_normalization,
        similarity_threshold=similarity_threshold,
        normalization_attr=args.normalization_attr,
        expansion_attr=args.expan,
        xpos_attr=args.xpos_attr,
        debug=args.debug,
        verbose=args.verbose,
        lemma_anchor=args.lemma_anchor,
        use_xpos_for_tagging=args.use_xpos_for_tagging,
        tokenization_method=args.tokenization_method,
        guess_ud=args.guess_ud,
    )
    
    # Load tagset if provided
    tagset_def = None
    if args.tagset:
        try:
            from flexipipe.tagset import parse_teitok_tagset
            tagset_def = parse_teitok_tagset(args.tagset)
            if args.debug:
                print(f"[DEBUG] Loaded tagset from {args.tagset}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load tagset from {args.tagset}: {e}", file=sys.stderr)
    
    # Load vocabulary (support multiple files - later files override earlier ones)
    vocab = {}
    transition_probs = None
    vocab_metadata = None
    vocab_file_paths = []
    
    # Load lexicons separately (used as fallback for OOV words only)
    lexicons = {}
    lexicon_file_paths = []
    
    if args.lexicon:
        from flexipipe.vocabulary import load_vocabulary_file
        
        for lexicon_file in args.lexicon:
            lexicon_file_path = Path(lexicon_file)
            lexicon_from_file, _, _ = load_vocabulary_file(lexicon_file_path, default_count=1)
            
            # Merge lexicons (simple merge, no weighting since they're all fallback)
            for form, entry in lexicon_from_file.items():
                if form not in lexicons:
                    lexicons[form] = entry
                else:
                    # Merge entries: combine into list if needed
                    existing_entry = lexicons[form]
                    new_entry = entry
                    existing_list = existing_entry if isinstance(existing_entry, list) else [existing_entry]
                    new_list = new_entry if isinstance(new_entry, list) else [new_entry]
                    # Combine all analyses
                    combined = existing_list + new_list
                    lexicons[form] = combined[0] if len(combined) == 1 else combined
            
            lexicon_file_paths.append(lexicon_file_path)
        
        if lexicons:
            if args.debug:
                print(f"Loaded {len(lexicons)} lexicon entries from {len(lexicon_file_paths)} file(s) (OOV fallback only)", file=sys.stderr)
            # Will be reported in verbose mode by tagger
    
    if args.vocab:
        # Process vocab files in order - later files override earlier ones
        from flexipipe.vocabulary import load_vocabulary_file
        
        for vocab_file in args.vocab:
            vocab_file_path = Path(vocab_file)
            vocab_from_file, transitions_from_file, metadata_from_file = load_vocabulary_file(vocab_file_path, default_count=1)
            
            # Merge vocabularies: combine analyses, but prioritize later files
            # This preserves all analyses from all vocab files, but gives higher weight to later ones
            for form, entry in vocab_from_file.items():
                if form not in vocab:
                    # New word: just add it
                    vocab[form] = entry
                else:
                    # Word exists: merge analyses, prioritizing later vocab
                    existing_entry = vocab[form]
                    new_entry = entry
                    
                    # Handle both single dict and list formats
                    existing_list = existing_entry if isinstance(existing_entry, list) else [existing_entry]
                    new_list = new_entry if isinstance(new_entry, list) else [new_entry]
                    
                    # Merge analyses: combine by annotation key, but weight later vocab more heavily
                    # Weight factor: later vocab counts are multiplied by this to give them priority
                    weight_factor = 10  # Later vocab counts are worth 10x for sorting purposes
                    merged_analyses = {}
                    
                    # First, add existing analyses (from earlier vocab files)
                    # These get base priority (no weight boost)
                    for analysis in existing_list:
                        key = (
                            analysis.get('upos', '_'),
                            analysis.get('xpos', '_'),
                            analysis.get('feats', '_'),
                            analysis.get('lemma', '_'),
                            analysis.get('reg', '_'),
                            analysis.get('expan', '_')
                        )
                        if key not in merged_analyses:
                            # New annotation: add it with base priority
                            merged_analyses[key] = analysis.copy()
                            merged_analyses[key]['_original_count'] = analysis.get('count', 1)
                            # Don't apply weight boost to earlier vocab entries
                            merged_analyses[key]['_weighted_count'] = analysis.get('count', 1)
                    
                    # Then, add new analyses (from later vocab file) with weight boost
                    for analysis in new_list:
                        key = (
                            analysis.get('upos', '_'),
                            analysis.get('xpos', '_'),
                            analysis.get('feats', '_'),
                            analysis.get('lemma', '_'),
                            analysis.get('reg', '_'),
                            analysis.get('expan', '_')
                        )
                        if key in merged_analyses:
                            # Same annotation: combine counts, but boost later vocab for sorting
                            existing_original = merged_analyses[key].get('_original_count', 0)
                            existing_weighted = merged_analyses[key].get('_weighted_count', existing_original)
                            new_count = analysis.get('count', 1)
                            # Update original count (actual sum)
                            merged_analyses[key]['_original_count'] = existing_original + new_count
                            # Update weighted count for sorting (later vocab gets boost)
                            merged_analyses[key]['_weighted_count'] = existing_weighted + (new_count * weight_factor)
                            # Update display count to weighted count for sorting
                            merged_analyses[key]['count'] = merged_analyses[key]['_weighted_count']
                        else:
                            # New annotation: add it with weight boost
                            merged_analyses[key] = analysis.copy()
                            new_count = analysis.get('count', 1)
                            merged_analyses[key]['_original_count'] = new_count
                            merged_analyses[key]['_weighted_count'] = new_count * weight_factor
                            merged_analyses[key]['count'] = merged_analyses[key]['_weighted_count']
                    
                    # Convert back to list or single dict, sorted by weighted count (descending)
                    merged_list = list(merged_analyses.values())
                    merged_list.sort(key=lambda a: a.get('_weighted_count', a.get('count', 0)), reverse=True)
                    
                    # Restore original count for display (weighted count was only for sorting)
                    for a in merged_list:
                        if '_original_count' in a:
                            a['count'] = a['_original_count']
                            # Optionally remove internal fields (keep for debugging for now)
                            # a.pop('_original_count', None)
                            # a.pop('_weighted_count', None)
                    
                    # If only one analysis, return as dict; otherwise as list
                    if len(merged_list) == 1:
                        vocab[form] = merged_list[0]
                    else:
                        vocab[form] = merged_list
            
            # Merge transition probabilities (weighted average by sentence count)
            if transitions_from_file:
                if transition_probs is None:
                    transition_probs = {}
                
                # Get sentence counts from metadata for weighting
                file_sentences = metadata_from_file.get('sentence_count', 1) if metadata_from_file else 1
                total_sentences = vocab_metadata.get('sentence_count', 0) if vocab_metadata else 0
                
                for trans_type, trans_dict in transitions_from_file.items():
                    if not isinstance(trans_dict, dict):
                        # Skip if not a dictionary (shouldn't happen, but be safe)
                        continue
                    
                    if trans_type not in transition_probs:
                        transition_probs[trans_type] = {}
                    
                    # Special handling for deprel transitions: they're stored as {'upos': {...}, 'xpos': {...}}
                    # or as a flat dict (old format) with string keys like "head_pos|dep_pos|deprel"
                    if trans_type == 'deprel':
                        # Check if new format (nested dict with 'upos' and 'xpos' keys)
                        if isinstance(trans_dict, dict) and 'upos' in trans_dict or 'xpos' in trans_dict:
                            # New format: {'upos': {...}, 'xpos': {...}}
                            if trans_type not in transition_probs:
                                transition_probs[trans_type] = {}
                            for tag_format in ['upos', 'xpos']:
                                if tag_format in trans_dict:
                                    if tag_format not in transition_probs[trans_type]:
                                        transition_probs[trans_type][tag_format] = {}
                                    format_dict = trans_dict[tag_format]
                                    for key, prob in format_dict.items():
                                        if isinstance(prob, (int, float)):
                                            if key in transition_probs[trans_type][tag_format]:
                                                # Weighted average: combine probabilities
                                                old_prob = transition_probs[trans_type][tag_format][key]
                                                if total_sentences > 0:
                                                    new_prob = (old_prob * total_sentences + prob * file_sentences) / (total_sentences + file_sentences)
                                                else:
                                                    new_prob = prob
                                                transition_probs[trans_type][tag_format][key] = new_prob
                                            else:
                                                transition_probs[trans_type][tag_format][key] = prob
                        else:
                            # Old format: flat dict with string keys (backward compatibility)
                            for key, prob in trans_dict.items():
                                if isinstance(prob, (int, float)):
                                    if key in transition_probs[trans_type]:
                                        # Weighted average: combine probabilities
                                        old_prob = transition_probs[trans_type][key]
                                        if total_sentences > 0:
                                            new_prob = (old_prob * total_sentences + prob * file_sentences) / (total_sentences + file_sentences)
                                        else:
                                            new_prob = prob
                                        transition_probs[trans_type][key] = new_prob
                                    else:
                                        transition_probs[trans_type][key] = prob
                        continue
                    
                    # Check if this is a nested structure (upos/xpos) or flat structure (start)
                    # Nested: {prev_tag: {next_tag: prob}}
                    # Flat: {tag: prob}
                    is_nested = False
                    if trans_dict:
                        # Check first value to determine structure
                        first_value = next(iter(trans_dict.values()))
                        is_nested = isinstance(first_value, dict)
                    
                    if is_nested:
                        # Nested structure (upos, xpos transitions)
                        for prev_tag, next_dict in trans_dict.items():
                            if not isinstance(next_dict, dict):
                                continue
                            if prev_tag not in transition_probs[trans_type]:
                                transition_probs[trans_type][prev_tag] = {}
                            
                            for next_tag, prob in next_dict.items():
                                if next_tag in transition_probs[trans_type][prev_tag]:
                                    # Weighted average: combine probabilities
                                    old_prob = transition_probs[trans_type][prev_tag][next_tag]
                                    if total_sentences > 0:
                                        # Weighted average
                                        new_prob = (old_prob * total_sentences + prob * file_sentences) / (total_sentences + file_sentences)
                                    else:
                                        new_prob = prob
                                    transition_probs[trans_type][prev_tag][next_tag] = new_prob
                                else:
                                    transition_probs[trans_type][prev_tag][next_tag] = prob
                    else:
                        # Flat structure (start probabilities)
                        for tag, prob in trans_dict.items():
                            if isinstance(prob, (int, float)):
                                if tag in transition_probs[trans_type]:
                                    # Weighted average: combine probabilities
                                    old_prob = transition_probs[trans_type][tag]
                                    if total_sentences > 0:
                                        new_prob = (old_prob * total_sentences + prob * file_sentences) / (total_sentences + file_sentences)
                                    else:
                                        new_prob = prob
                                    transition_probs[trans_type][tag] = new_prob
                                else:
                                    transition_probs[trans_type][tag] = prob
                
                # Update total sentence count
                if vocab_metadata:
                    vocab_metadata['sentence_count'] = total_sentences + file_sentences
                elif metadata_from_file:
                    vocab_metadata = metadata_from_file.copy()
            
            # Merge metadata (later files override)
            if metadata_from_file:
                if vocab_metadata is None:
                    vocab_metadata = {}
                # Merge capitalization info
                if 'capitalizable_tags' in metadata_from_file:
                    if 'capitalizable_tags' not in vocab_metadata:
                        vocab_metadata['capitalizable_tags'] = {'upos': {}, 'xpos': {}}
                    # Merge capitalization stats (weighted by counts)
                    for tag_type in ['upos', 'xpos']:
                        if tag_type in metadata_from_file.get('capitalizable_tags', {}):
                            for tag, stats in metadata_from_file['capitalizable_tags'][tag_type].items():
                                if tag in vocab_metadata['capitalizable_tags'][tag_type]:
                                    # Combine counts
                                    old_stats = vocab_metadata['capitalizable_tags'][tag_type][tag]
                                    vocab_metadata['capitalizable_tags'][tag_type][tag] = {
                                        'capitalized': old_stats.get('capitalized', 0) + stats.get('capitalized', 0),
                                        'lowercase': old_stats.get('lowercase', 0) + stats.get('lowercase', 0)
                                    }
                                else:
                                    vocab_metadata['capitalizable_tags'][tag_type][tag] = stats.copy()
                
            # Other metadata: later files override
            for key, value in metadata_from_file.items():
                if key != 'capitalizable_tags':
                    vocab_metadata[key] = value
            
            # Check if tagset is in metadata
            if metadata_from_file and 'tagset' in metadata_from_file:
                tagset_def = metadata_from_file['tagset']
                if args.debug:
                    print(f"[DEBUG] Loaded tagset from vocabulary metadata", file=sys.stderr)
            
            # If language is in metadata but not in config, use it
            if metadata_from_file and metadata_from_file.get('language') and not config.language:
                config.language = metadata_from_file.get('language')
            
            vocab_file_paths.append(vocab_file)
    
    # Create tagger
    tagger = FlexiPipeTagger(config, vocab, model_path=args.model if args.model else None, transition_probs=transition_probs, vocab_metadata=vocab_metadata, lexicon=lexicons if lexicons else None)
    # Store stdin flag for verbose output
    tagger._is_stdin = is_stdin
    # Store tagset for parser and UD guessing
    if tagset_def:
        tagger.tagset_def = tagset_def
        if args.tagset:
            tagger._tagset_file = str(args.tagset)
        if args.debug:
            print(f"[DEBUG] Stored tagset in tagger (tagset_def is not None: {tagset_def is not None})", file=sys.stderr)
    elif args.debug:
        print(f"[DEBUG] No tagset to store (tagset_def is None)", file=sys.stderr)
    # Store vocabulary file path(s) for revision statement
    if vocab_file_paths:
        # Store as comma-separated list or just the most specific (last) one
        tagger.vocab_file_path = vocab_file_paths[-1]  # Most specific vocab file
        tagger.vocab_file_paths = vocab_file_paths  # All vocab files
    if args.model:
        tagger.load_model(args.model)
    
    # Auto-detect format
    input_format = args.input_format
    if input_format == 'auto':
        if use_stdin:
            input_format = 'plain'
        else:
            input_ext = input_path.suffix.lower()
            if input_ext == '.xml':
                input_format = 'teitok'
            elif input_ext in ('.conllu', '.conll'):
                input_format = 'conllu'
            else:
                input_format = 'plain'
    
    # Determine output format
    output_format = args.output_format or input_format
    if not args.output_format and input_format == 'plain':
        output_format = 'conllu'
    
    # Handle segment/tokenize flags:
    # - For 'teitok' and 'conllu': already tokenized/segmented, disable these flags
    # - For 'plain': auto-enable if not explicitly set
    # - For 'raw': always enabled
    if input_format in ('teitok', 'conllu'):
        # Input is already tokenized and segmented - disable segment/tokenize
        segment = False
        tokenize = False
        if args.segment or args.tokenize:
            print(f"Note: --segment and --tokenize are ignored for {input_format} input (already tokenized/segmented)", file=sys.stderr)
    elif input_format == 'plain':
        # Auto-enable for plain text if not explicitly set
        segment = args.segment
        tokenize = args.tokenize
    else:
        # For 'raw' or other formats, use flags as provided
        segment = args.segment
        tokenize = args.tokenize
    
    # Tag
    tagged = tagger.tag(input_path, None, input_format, segment=segment, tokenize=tokenize)
    
    # Write output
    if use_stdout:
        tagger.write_output(tagged, None, output_format)
    else:
        tagger.write_output(tagged, output_path, output_format)
    
    # Clean up temporary file if created
    if use_stdin and input_path.exists():
        input_path.unlink()

if __name__ == '__main__':
    main()

