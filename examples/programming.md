## Programmatic Use of Flexipipe (spaCy, Stanza, or stand-alone)

Flexipipe is primarily intended as an easy-to-use uniform front-end for existing tools, 
making it easy to process data with a variety of NLP pipeline without having to adept to
the different structure of each one.

However, you can of course also use flexipipe programmatically as a Python modules. 
The core functioning of flexipipe is that it uses a uniform document
model across various tool (a `Document`), that you can use by itself, bridge it to 
existing NLP pipeline architecture like SpaCy or Stanza, or feed the results into 
pipelines that rely on annotated data. Below are basic examples of each type 
of use.

### A. Minimal stand-alone usage
```python
from flexipipe import pipeline

text = "This is a test sentence."

# Auto-select a model for a language (default backend order)
doc = pipeline.process_text(text, language="en")

for sent in doc.sentences:
    for tok in sent.tokens:
        print(tok.text, tok.lemma, tok.upos, tok.xpos, tok.head, tok.deprel)

#  Choose a backend explicitly:

doc = pipeline.process_text(
    text,
    language="en",
    backend="udpipe",   # or "stanza", "spacy", "flexitag", etc.
)
```

### B. Using Stanza via Flexipipe

```python
from flexipipe import pipeline

# example text in Vietnamese (from the Universal Declaration of Human Rights)
text = Tất cả mọi người sinh ra đều được tự do và bình đẳng về nhân phẩm và quyền lợi." 
doc = pipeline.process_text(
    text,
    language="vie",          # normalized ISO code
    backend="stanza",
    download_model=True,    # auto-download if missing
)
```
Flexipipe uses the curated Stanza registry to request only processors that exist for the package (skips missing NER, etc.).

### C. Using UDPipe (REST) via Flexipipe

```python
from flexipipe.backends import udpipe

doc = udpipe.tag(
    "Bonjour tout le monde.",
    model="french-gsd-ud-2.17-251125",   # curated name from the registry
    refresh_cache=False,                 # set True to force registry refresh
)
```

### D. Convert Flexipipe `Document` -> spaCy `Doc`

```python
import spacy
from flexipipe.spacy_bridge import document_to_spacy

doc = pipeline.process_text("Another test.", language="en", backend="udpipe")
nlp = spacy.blank("en")  # add spaCy components if desired
spacy_doc = document_to_spacy(doc, nlp)

for sent in spacy_doc.sents:
    for token in sent:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.head.text)
```
Mapped fields:
- tokens: `text`, `lemma_`, `pos_` (UPOS), `tag_` (XPOS), `morph`
- syntax: `head`, `dep_`
- sentences: `Doc.sents`

### E. Persisting / feeding other tools
- **CoNLL-U**: `from flexipipe.conllu import document_to_conllu`
- **JSON payload**: `from flexipipe.doc_utils import document_to_json_payload`
- **TEI/TEITOK**: `from flexipipe.teitok import save_teitok`
- **Custom frameworks**: the `Document` is a plain Python object with `sentences` and `tokens`; serialize or adapt as needed.

### F. Practical tips
- Use `refresh_cache=True` (CLI `--refresh-cache`) to force registry/language mapping refreshes when new models or language codes appear.
- Backends that auto-download (Stanza, UDPipe REST) may log download progress the first time; subsequent runs use cached files.
- Keep the original Flexipipe `Document` if you plan to write TEI/CoNLL-U or pass data to another backend; the spaCy `Doc` is mainly for spaCy-only processing.

### G. Free click-to-run options (Colab)
We don’t want a paid Codespaces flow. For a free option, use Google Colab:

1) Upload a small example notebook from this repo (or create one with the snippets above).  
2) In Colab, install Flexipipe with the needed extras (note: large downloads on first run):
```bash
!pip install "git+https://github.com/ufal/flexipipe.git#egg=flexipipe[stanza,spacy]"
```
3) Run the code cells; models/cache live only for the session.  

If you prefer a view-only render, use nbviewer on any notebook in the repo (no execution). Binder/JupyterLite are not recommended for Flexipipe because of native deps and large model downloads.
