#!/usr/bin/env python3
"""
Train command for FlexiPipe.
"""

import sys
import argparse
from pathlib import Path

from flexipipe.config import FlexiPipeConfig
from flexipipe.tagger import FlexiPipeTagger
from flexipipe.core import (
    set_conllu_expansion_key,
    load_teitok_xml, build_vocabulary
)

def main():
    parser = argparse.ArgumentParser(
        description='FlexiPipe train: Train a model',
        prog='flexipipe train'
    )
    parser.add_argument('--data-dir', type=Path,
                       help='UD treebank directory')
    parser.add_argument('--train-dir', type=Path,
                       help='Directory containing CoNLL-U training files')
    parser.add_argument('--dev-dir', type=Path,
                       help='Directory containing CoNLL-U development files')
    parser.add_argument('--bert-model', default='bert-base-multilingual-cased',
                       help='BERT base model')
    parser.add_argument('--no-tokenizer', dest='train_tokenizer', action='store_false', default=True)
    parser.add_argument('--no-tagger', dest='train_tagger', action='store_false', default=True)
    parser.add_argument('--no-parser', dest='train_parser', action='store_false', default=True)
    parser.add_argument('--no-lemmatizer', dest='train_lemmatizer', action='store_false', default=True)
    parser.add_argument('--no-normalizer', dest='train_normalizer', action='store_false', default=True)
    parser.add_argument('--normalization-attr', default='reg',
                       help='TEITOK attribute name for normalization')
    parser.add_argument('--expan', default='expan',
                       help='TEITOK attribute name for expansion')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='TEITOK attribute name(s) for XPOS')
    parser.add_argument('--output-dir', type=Path, default=Path('models/flexipipe'),
                       help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--num-epochs', type=int, default=3)
    
    args = parser.parse_args()
    
    # Find training files
    train_files = []
    dev_files = None
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        train_files = list(data_dir.glob('*-ud-train.conllu'))
        dev_files_list = list(data_dir.glob('*-ud-dev.conllu'))
        if not train_files:
            print(f"Error: No *-ud-train.conllu file found in {data_dir}", file=sys.stderr)
            sys.exit(1)
        if dev_files_list:
            dev_files = dev_files_list
    elif args.train_dir:
        train_files = list(args.train_dir.glob('*.conllu'))
        if not train_files:
            print(f"Error: No .conllu files found in {args.train_dir}", file=sys.stderr)
            sys.exit(1)
        if args.dev_dir:
            dev_files = list(args.dev_dir.glob('*.conllu'))
    else:
        print("Error: Either --data-dir or --train-dir must be specified", file=sys.stderr)
        sys.exit(1)
    
    # Configure
    set_conllu_expansion_key(args.expan)
    load_teitok_xml._xpos_attr = args.xpos_attr
    load_teitok_xml._expan_attr = args.expan
    
    config = FlexiPipeConfig(
        bert_model=args.bert_model,
        train_tokenizer=args.train_tokenizer,
        train_tagger=args.train_tagger,
        train_parser=args.train_parser,
        train_lemmatizer=args.train_lemmatizer,
        output_dir=str(args.output_dir),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )
    
    # Build vocabulary
    vocab = build_vocabulary(train_files + (dev_files or []))
    print(f"Built vocabulary with {len(vocab)} entries", file=sys.stderr)
    
    # Train
    tagger = FlexiPipeTagger(config, vocab)
    tagger.train(train_files, dev_files)
    
    print(f"Training complete. Model saved to: {args.output_dir}", file=sys.stderr)

if __name__ == '__main__':
    main()

