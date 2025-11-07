#!/usr/bin/env python3
"""
Analyze command for FlexiPipe.
"""

import sys
import argparse
import json
from pathlib import Path

from flexipipe.core import (
    FlexiPipeConfig, FlexiPipeTagger, set_conllu_expansion_key,
    load_teitok_xml
)

def main():
    parser = argparse.ArgumentParser(
        description='FlexiPipe analyze: Analyze vocabulary and suffixes',
        prog='flexipipe analyze'
    )
    parser.add_argument('--model', type=Path, help='Path to trained model')
    parser.add_argument('--vocab', type=Path, help='Vocabulary JSON to analyze')
    parser.add_argument('--lexicon', type=Path, help='Lexicon file (UniMorph or JSON) to analyze')
    parser.add_argument('--normalization-suffixes', type=Path,
                       help='External suffix list (JSON)')
    parser.add_argument('--expan', default='expan',
                       help='TEITOK attribute name for expansion')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='TEITOK attribute name(s) for XPOS')
    parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                       help='Anchor for deriving inflection suffixes')
    parser.add_argument('--output', type=Path, help='Write analysis JSON to this file (default: stdout)')
    parser.add_argument('--word', type=str, help='Analyze a single word (shows vocabulary, normalization, and lemmatization analysis)')
    parser.add_argument('--xpos', type=str, help='XPOS tag for word analysis (use with --word)')
    parser.add_argument('--upos', type=str, help='UPOS tag for word analysis (use with --word)')
    
    args = parser.parse_args()
    
    # Configure
    set_conllu_expansion_key(args.expan)
    load_teitok_xml._xpos_attr = args.xpos_attr
    load_teitok_xml._expan_attr = args.expan
    
    config = FlexiPipeConfig(
        normalize=True,
        conservative_normalization=True,
        normalization_suffixes_file=args.normalization_suffixes,
        lemma_anchor=args.lemma_anchor,
        train_tokenizer=False,
        train_tagger=False,
        train_parser=False,
        train_lemmatizer=False,
        train_normalizer=False,
    )
    
    # Load vocabulary
    from flexipipe.vocabulary import load_vocabulary_file
    
    vocab = {}
    if args.vocab:
        vocab, _, _ = load_vocabulary_file(args.vocab, default_count=1)
    elif args.model and Path(args.model).exists():
        model_vocab_file = Path(args.model) / 'model_vocab.json'
        if model_vocab_file.exists():
            vocab, _, _ = load_vocabulary_file(model_vocab_file, default_count=1)
    
    # Load lexicon if provided
    lexicon = {}
    if args.lexicon:
        from flexipipe.vocabulary import load_vocabulary_file
        lexicon, _, _ = load_vocabulary_file(args.lexicon, default_count=1)
        if lexicon:
            print(f"Loaded {len(lexicon)} lexicon entries (OOV fallback only)", file=sys.stderr)
    
    # Analyze
    tagger = FlexiPipeTagger(config, vocab=vocab, model_path=args.model if args.model else None, lexicon=lexicon if lexicon else None)
    if not tagger.vocab:
        tagger.vocab = vocab
    tagger._build_normalization_inflection_suffixes()
    suffixes = tagger.inflection_suffixes or []
    
    # Word lookup mode
    if args.word:
        # Build lemmatization patterns if needed
        if tagger.vocab and not tagger.lemmatization_patterns:
            tagger._build_lemmatization_patterns()
        from flexipipe.normalization import normalize_word
        from flexipipe.vocabulary import find_similar_words
        
        word = args.word
        xpos = args.xpos
        upos = args.upos
        
        analysis = {
            'word': word,
            'xpos': xpos,
            'upos': upos,
        }
        
        # Vocabulary lookup
        vocab_analysis = {}
        word_lower = word.lower()
        
        # Check exact case
        if word in tagger.vocab:
            entry = tagger.vocab[word]
            vocab_analysis['exact_case'] = entry
        # Check lowercase
        if word_lower in tagger.vocab:
            entry = tagger.vocab[word_lower]
            vocab_analysis['lowercase'] = entry
        # Check form:xpos
        if xpos:
            key = f"{word}:{xpos}"
            if key in tagger.vocab:
                vocab_analysis['form_xpos'] = {key: tagger.vocab[key]}
            key_lower = f"{word_lower}:{xpos}"
            if key_lower in tagger.vocab:
                vocab_analysis['form_xpos_lower'] = {key_lower: tagger.vocab[key_lower]}
        
        analysis['vocabulary'] = vocab_analysis
        
        # Lexicon lookup (if available)
        lexicon_analysis = {}
        if tagger.lexicon:
            word_lower = word.lower()
            # Check exact case
            if word in tagger.lexicon:
                entry = tagger.lexicon[word]
                lexicon_analysis['exact_case'] = entry
            # Check lowercase
            if word_lower in tagger.lexicon:
                entry = tagger.lexicon[word_lower]
                lexicon_analysis['lowercase'] = entry
            # Check form:xpos
            if xpos:
                key = f"{word}:{xpos}"
                if key in tagger.lexicon:
                    lexicon_analysis['form_xpos'] = {key: tagger.lexicon[key]}
                key_lower = f"{word_lower}:{xpos}"
                if key_lower in tagger.lexicon:
                    lexicon_analysis['form_xpos_lower'] = {key_lower: tagger.lexicon[key_lower]}
        analysis['lexicon'] = lexicon_analysis
        
        # Normalization analysis
        norm_analysis = {}
        if tagger.vocab:
            normalized = normalize_word(
                word,
                tagger.vocab,
                conservative=config.conservative_normalization,
                similarity_threshold=0.8 if config.conservative_normalization else 0.7,
                inflection_suffixes=suffixes
            )
            norm_analysis['normalized_form'] = normalized
            norm_analysis['normalization_applied'] = normalized is not None
        else:
            norm_analysis['normalized_form'] = None
            norm_analysis['normalization_applied'] = False
        
        analysis['normalization'] = norm_analysis
        
        # Lemmatization analysis
        lemma_analysis = {}
        # Enable debug temporarily to capture analysis
        original_debug = config.debug
        config.debug = True
        import io
        import contextlib
        
        debug_output = io.StringIO()
        with contextlib.redirect_stderr(debug_output):
            # Try to get lemma
            lemma_form = norm_analysis['normalized_form'] if norm_analysis['normalized_form'] else word
            vocab_lemma = tagger._predict_from_vocab(lemma_form, 'lemma', xpos=xpos, upos=upos, debug=True)
            lemma_analysis['vocab_lemma'] = vocab_lemma if vocab_lemma != '_' else None
        
        debug_text = debug_output.getvalue()
        lemma_analysis['debug_trace'] = debug_text.split('\n') if debug_text else []
        
        # Pattern-based analysis
        if xpos and tagger.lemmatization_patterns and xpos in tagger.lemmatization_patterns:
            patterns = tagger.lemmatization_patterns[xpos]
            matching_patterns = []
            form_lower = word.lower()
            for pattern_tuple in patterns:
                if len(pattern_tuple) == 4:
                    suffix_from, suffix_to, min_base, count = pattern_tuple
                else:
                    suffix_from, suffix_to, min_base = pattern_tuple[:3]
                    count = 1
                if suffix_from and form_lower.endswith(suffix_from):
                    base = form_lower[:-len(suffix_from)]
                    if len(base) >= min_base:
                        lemma = base + suffix_to
                        if len(lemma) >= 2:
                            matching_patterns.append({
                                'suffix_from': suffix_from,
                                'suffix_to': suffix_to,
                                'lemma': lemma,
                                'count': count,
                                'suffix_length': len(suffix_from)
                            })
            matching_patterns.sort(key=lambda x: (x['suffix_length'], x['count']), reverse=True)
            lemma_analysis['pattern_based'] = {
                'patterns_available': len(patterns),
                'matching_patterns': matching_patterns,
                'selected_pattern': matching_patterns[0] if matching_patterns else None
            }
        else:
            lemma_analysis['pattern_based'] = {
                'patterns_available': 0,
                'matching_patterns': [],
                'selected_pattern': None
            }
        
        # Similarity matching
        if tagger.vocab:
            similar = find_similar_words(word, tagger.vocab, threshold=0.7)
            lemma_analysis['similarity_matching'] = {
                'similar_words': [(w, score) for w, score in similar[:10]],
                'best_match': similar[0][0] if similar else None,
                'best_match_lemma': None
            }
            if similar:
                best_entry = tagger.vocab[similar[0][0]]
                if isinstance(best_entry, list):
                    best_lemma = best_entry[0].get('lemma', '_') if best_entry else '_'
                else:
                    best_lemma = best_entry.get('lemma', '_')
                lemma_analysis['similarity_matching']['best_match_lemma'] = best_lemma if best_lemma != '_' else None
        else:
            lemma_analysis['similarity_matching'] = {
                'similar_words': [],
                'best_match': None,
                'best_match_lemma': None
            }
        
        config.debug = original_debug
        analysis['lemmatization'] = lemma_analysis
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as out:
                json.dump(analysis, out, ensure_ascii=False, indent=2)
            print(f"Wrote word analysis to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
    else:
        # Suffix analysis mode (original behavior)
        analysis = {
            'lemma_anchor': config.lemma_anchor,
            'source': 'external' if args.normalization_suffixes else 'derived',
            'num_suffixes': len(suffixes),
            'suffixes': suffixes,
        }
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as out:
                json.dump(analysis, out, ensure_ascii=False, indent=2)
            print(f"Wrote suffix analysis to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

