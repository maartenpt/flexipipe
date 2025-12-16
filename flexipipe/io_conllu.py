from __future__ import annotations

from pathlib import Path
from typing import Optional

from .conllu import conllu_to_document, document_to_conllu
from .doc import Document
from .engine import assign_doc_id_from_path
from .io_registry import InputEntry, OutputEntry, registry


def _load_conllu(*, args, stdin_content: Optional[str] = None) -> Document:
    if stdin_content is not None:
        text = stdin_content
        doc = conllu_to_document(text)
        doc.meta.setdefault("source", "stdin")
        return doc
    input_path = getattr(args, "input", None)
    if not input_path:
        raise SystemExit("ConLL-U input requires --input or data on STDIN.")
    path = Path(input_path)
    from .file_utils import read_text_file
    text = read_text_file(path)
    doc = conllu_to_document(text, doc_id=path.stem)
    doc.meta.setdefault("source_path", str(path))
    assign_doc_id_from_path(doc, str(path))
    return doc


def _save_conllu(
    document: Document,
    *,
    args,
    output_path: Optional[str],
    entity_format: str,
    model_info: Optional[str] = None,
) -> None:
    conllu_text = document_to_conllu(
        document,
        model=model_info,
        create_implicit_mwt=getattr(args, "create_implicit_mwt", False),
        entity_format=entity_format,
    )
    if output_path:
        Path(output_path).write_text(conllu_text, encoding="utf-8")
    else:
        print(conllu_text, end="")


registry.register_input(
    InputEntry(
        name="conllu",
        aliases=("conll-u", "conllu-ne"),
        loader=_load_conllu,
        supports_stdin=True,
    )
)

registry.register_output(
    OutputEntry(
        name="conllu",
        aliases=("conll-u",),
        saver=lambda document, *, args, output_path, model_info=None: _save_conllu(
            document,
            args=args,
            output_path=output_path,
            entity_format="iob",
            model_info=model_info,
        ),
        description="Classic CoNLL-U output (IOB entities).",
    )
)

registry.register_output(
    OutputEntry(
        name="conllu-ne",
        aliases=(),
        saver=lambda document, *, args, output_path, model_info=None: _save_conllu(
            document,
            args=args,
            output_path=output_path,
            entity_format="ne",
            model_info=model_info,
        ),
        description="CoNLL-U with named entities in # sent_id metadata.",
    )
)

