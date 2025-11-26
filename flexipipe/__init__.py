"""flexipipe – modular NLP pipeline combining BERT and flexitag."""

__version__ = "0.1.0"

from .doc import Document, Sentence, Token, SubToken, apply_nlpform
from .engine import FlexitagFallback
from .teitok import load_teitok, save_teitok, dump_teitok, update_teitok
# Import insert_tokens_into_teitok implementation
from .insert_tokens import insert_tokens_into_teitok
from .pipeline import FlexiPipeline, PipelineConfig
from .conllu import document_to_conllu, conllu_to_document
from .train import train_ud_treebank
from .check import evaluate_model
from .tag_mapping import TagMapping, build_tag_mapping_from_paths

# Ensure language detectors register on import
from . import language_detectors as _language_detectors  # noqa: F401

__all__ = [
    "Document",
    "Sentence",
    "Token",
    "SubToken",
    "apply_nlpform",
    "FlexitagFallback",
    "FlexiPipeline",
    "PipelineConfig",
    "load_teitok",
    "save_teitok",
    "dump_teitok",
    "update_teitok",
    "document_to_conllu",
    "conllu_to_document",
    "train_ud_treebank",
    "evaluate_model",
    "TagMapping",
    "build_tag_mapping_from_paths",
]
