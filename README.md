# Flexipipe

Flexipipe is a modular NLP pipeline for Universal Dependencies data that glues
rule-based components, the legacy flexitag tagger, and multiple neural
backends (SpaCy, Stanza, Flair, UDPipe REST, and
UDMorph REST).  It can ingest raw text, CoNLL-U, and TEITOK XML, preserves
existing annotations, and exports both CoNLL-U (with optional implicit MWTs
and IOB NER) and TEITOK with nested `<tok>`/`<dtok>`/`<name>` structures.

The CLI centers around several workflows:

| Command | Purpose |
| --- | --- |
| `python -m flexipipe process` | Tag/parse/normalize input with any backend |
| `python -m flexipipe benchmark` | Evaluate backends against gold treebanks |
| `python -m flexipipe train` | Train flexitag or (where implemented) neural backends |
| `python -m flexipipe convert` | Convert between formats (tagged files, treebanks, lexicons) |
| `python -m flexipipe info` | List backends, models, examples, or tasks |
| `python -m flexipipe config` | Inspect or change defaults (models dir, backend, output format, implicit MWT) |

Use `python -m flexipipe info backends`, `python -m flexipipe info models --backend <name>`, 
`python -m flexipipe info examples`, or `python -m flexipipe info tasks` to explore
available integrations, models, example texts, and supported tasks.

---

## Key Features

* **Multi-backend orchestration**
  * `flexitag` (built-in Viterbi model with rule-based sentence segmentation & UD-style tokenization)
  * `spacy` (full [SpaCy](http://spacy.io) pipeline, TEI/CoNLL conversions, automatic model discovery, optional downloads)
  * `stanza` ([Stanza](https://stanfordnlp.github.io/stanza/) with raw or tokenized input, package selection, download-on-demand, suppressed logging)
  * `flair` ([Flair](https://flairnlp.github.io/) multi-tagger with confidence scores, automatic contraction handling)
  * `transformers` ([HuggingFace Transformers](https://huggingface.co/) token-classification models for POS & NER with detailed metadata)
  * `udpipe` ([UDPIPE](https://lindat.mff.cuni.cz/services/udpipe) REST backends with batching, debug logging, and URL overrides)
  * `udmorph` ([UDMorph](https://lindat.mff.cuni.cz/services/teitok-live/udmorph/) REST backends with batching, and debug logging)
* **Input flexibility**: auto-detects format or accepts `--input-format` (`auto`, `conllu`, `tei`, `raw`).
  Raw mode can read from STDIN (use `--input -` or pipe text).
* **Backend chaining**: Pipe output from one backend to another to combine different tools (e.g., UDPipe for tagging/parsing, then NameTag for NER).
* **Document representation**:
  * Supports multi-word tokens with subtokens, tokids, `space_after`, and alignment metadata.
  * Named entities stored per-sentence and exported as CoNLL-U `Entity=B-ORG` and TEITOK `<name>` spans.
  * Confidence slots (`upos_confidence`, `lemma_confidence`, etc.) allow neural fallbacks.
* **Output control**:
  * CoNLL-U writer can append implicit MWT ranges and selective `TokId` output.
  * TEITOK writer emits `<dtok/>` without extra whitespace, nests `<name>` blocks, and respects `_` skipping rules.
  * Configure default output format and implicit MWT behaviour via `flexipipe config`.
* **Evaluation tooling (`check`)**
  * Aligns tokens via tokids and SequenceMatcher fallbacks.
  * Computes UPOS/XPOS/FEATS/lemma/UAS/LAS, splitting stats, and partial-feats accuracy.
  * Optional debug dumps: duplication warnings, tokenization diff samples, backend metadata.
* **Centralised model storage**
  * Models live under `~/.flexipipe/models/<backend>/` (overridable via `FLEXIPIPE_MODELS_DIR` or `config --set-models-dir`).
  * Environment helpers ensure SpaCy/Stanza/Flair look inside the shared directory first.
* **Rich configuration**
  * `config.json` (created automatically) tracks:
    * `models_dir`
    * default backend (used when `--backend` is omitted)
    * default output format (`tei` or `conllu`)
    * default `create_implicit_mwt` flag
  * `flexipipe config --show` prints the active settings and where they originate.
  * `--list-models` caches per-backend model catalogs under `~/.flexipipe/cache/`; refresh anytime with `--refresh-cache`.

---

## Installation & Requirements

1. **Python 3.8+** (Python 3.11 recommended)
2. Install Python requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
   Note: `torch>=2.6.0` is required for security (CVE-2025-32434). The transformers backend will check this at runtime.
3. For neural backends install their optional extras on demand:
   ```bash
   # SpaCy
   python -m pip install "spacy[transformers]"
   # Stanza + Flair
   python -m pip install stanza flair
   # REST backends use requests / urllib (already required)
   ```
4. Build the native flexitag modules (`flexitag`, `viterbi_cpp`) if needed via CMake
   (see `README_CPP.md` for details).
   
   **Note**: C++ dependencies (pugixml, rapidjson) are automatically fetched via CMake FetchContent during build. No manual installation needed.

---

## Quick Start

### Tagging

```bash
# Raw text via STDIN, using default backend and output config
echo "Don't even think he wouldn't do it." | python -m flexipipe process --input -

# Process a CoNLL-U file with SpaCy and export TEITOK XML
python -m flexipipe process \
  --input data/en.conllu \
  --backend spacy \
  --model en_core_web_md \
  --output-format tei \
  --output out.xml

# Use UDPipe REST in raw mode with debug logging
python -m flexipipe process \
  --backend udpipe \
  --udpipe-model english-ewt-ud-2.15-241121 \
  --input-format raw \
  --debug \
  --input story.txt

# Chain backends: use UDPipe for tagging/parsing, then NameTag for NER
echo "Mary bought a new bicycle in Germany." | \
  python -m flexipipe process --backend udpipe | \
  python -m flexipipe process --backend nametag --output-format conllu-ne
```

Important switches:

| Flag | Description |
| --- | --- |
| `--backend` | Selects backend (`flexitag`, `spacy`, `stanza`, `flair`, `udpipe`, `udmorph`, `nametag`) |
| `--model` / `--language` | Backend-specific model hint. SpaCy resolves installed/downloadable names. |
| `--language English` (SpaCy) | Without `--model`, Flexipipe auto-uses SpaCy’s default core model (e.g., `en_core_web_sm`, if installed). |
| `--download-model` | Auto-fetch SpaCy/Stanza/Flair models when missing. |
| `--output-format` | `tei`, `conllu`, or `json`. Falls back to configuration default. |
| `--create-implicit-mwt` | Rebuilds implicit MWT ranges in output (default configurable). |
### Benchmarking

```bash
# Run a single backend/model evaluation (quick validation)
python -m flexipipe benchmark --test \
  --test-file UD_English-EWT/en_ewt-ud-test.conllu \
  --backend spacy \
  --model en_core_web_trf \
  --mode tokenized \
  --verbose --debug

# Run benchmark sweep across languages/backends
python -m flexipipe benchmark --run \
  --languages en nl de \
  --backends spacy stanza

# Show stored benchmark results
python -m flexipipe benchmark --show \
  --sort-by upos
```

* Accepts the same backend selection flags as `process`.
* `--mode` chooses how the gold data is fed to the backend (`raw`, `tokenized`,
  `split`, or `auto`).
* `--test` runs a single evaluation (replaces the old `check` command).
* `--run` performs a benchmark sweep across multiple languages/backends.
* `--show` displays stored benchmark results with various sorting options.

### Training

```bash
# Train flexitag on a UD treebank
python -m flexipipe train \
  --backend flexitag \
  --ud-data /path/to/UD_English-EWT \
  --output-dir models/flexitag-en

# Kick off backend-owned training (where implemented)
python -m flexipipe train \
  --backend spacy \
  --model en_core_web_md \
  --train-data data/train.conllu \
  --dev-data data/dev.conllu \
  --output-dir models/spacy-en
```

Flexitag training supports UD treebank directories (`--ud-data`) with automatic
tag-attribute selection. Neural training delegates to the backend’s own API;
some backends will raise `NotImplementedError` until training hooks are fully
implemented.

### Configuration

```bash
# Pick default backend/output, move models to an external drive, and enable implicit MWTs
python -m flexipipe config \
  --set-models-dir /Volumes/Data2/Flexipipe \
  --set-default-backend spacy \
  --set-default-output-format conllu \
  --set-default-create-implicit-mwt true

# Inspect the resulting config.json
python -m flexipipe config --show
```

---

## Backends Overview

| Backend | Mode(s) | Highlights | Notes |
| --- | --- | --- | --- |
| `flexitag` | Raw / tokenized | Built-in Viterbi tagger, rule-based segmentation, lexicon-aware | Requires flexitag model (`model_vocab.json`). |
| `spacy` | Raw + tokenized | NER, dependency parsing, automatic model discovery & download, centralized model dir support | Pre-tokenized mode preserves tokids/MWTs; raw mode hands segmentation to SpaCy. |
| `stanza` | Raw + tokenized | Full UD pipeline, package selection (e.g., `cs_cac`), SpaceAfter inference, suppressed INFO logging | Set `--download-model` or provide `--model`/`--language`. |
| `flair` | Raw-focused | Multi-task taggers (POS + NER), confidence scores, contraction alignment | Works best in raw mode; auto-converts results back to original tokens. |
| `transformers` | Raw + tokenized | HuggingFace Transformers POS/NER with detailed model metadata (tasks, base model, training data, techniques) | Requires `--model <huggingface_id>` plus optional `--transformers-task`, `--transformers-device`, etc. |
| `udpipe` | Raw + tokenized | REST integration with batching, curl debug output, token/parse tasks, default Lindat endpoint | Provide `--udpipe-model`, optional `--udpipe-param KEY=VALUE`. |
| `udmorph` | Tokenized | REST morph-only tagging, curl debug output, language-sorted model listing | Requires `--udmorph-model`. |
| `nametag` | Raw + tokenized | REST NER service, supports 21 languages, NameTag 3 (default), curl debug output | Provide `--nametag-model` or `--language`, optional `--nametag-version` (1/2/3), `--nametag-param KEY=VALUE`. |
### HuggingFace Transformers backend

The new `transformers` backend plugs Flexipipe directly into HuggingFace token-classification
models (POS tagging or NER). Models are described in the transformers registry with extra
metadata—tasks, base model, training corpora, and training techniques—so `python -m flexipipe info models --backend transformers`
shows not just names but what each model actually does.

Usage example:

```bash
echo "Why do we need an Old French model?" | \
  python -m flexipipe process \
    --backend transformers \
    --model Davlan/bert-base-multilingual-cased-ner-hrl \
    --transformers-device cpu \
    --output-format conllu
```

Key CLI switches:

| Flag | Purpose |
| --- | --- |
| `--model` | Required HuggingFace repo/model ID (e.g., `vblagoje/bert-english-uncased-finetuned-pos`). |
| `--transformers-task` | Override automatic task detection (`tag` or `ner`). |
| `--transformers-device` | Choose runtime device (`cpu`, `cuda`, `cuda:0`, `mps`, ...). |
| `--transformers-adapter` | Load a specific adapters hub adapter (if the model exposes adapters). |
| `--transformers-revision` | Pin a specific revision/tag/commit. |
| `--transformers-trust-remote-code` | Allow custom model code (required for some community repos). |

The backend aligns sub-word predictions back to document tokens, fills `upos` or
sentence-level NER spans, and records per-token confidence scores. Training hooks
will follow later (multi-task fine-tuning over arbitrary corpora).

Each backend exposes `list_*_models_display()` used by `--list-models` to show
installed vs available models, languages, and statuses (deduplicated where
possible).

---

## Input & Output Details

### Input Formats

* **CoNLL-U**: `--input-format conllu` or auto-detected via `.conllu`.
* **TEITOK XML**: `--input-format tei`. The reader converts `<tok>` and
  `<dtok>` to the internal document representation (preserving TokId).
* **Raw text / STDIN**: `--input-format raw` or `--input -`. When using
  flexitag, raw text is segmented/tokenized before tagging. For neural
  backends, raw text is passed through untouched so the backend can handle
  segmentation itself.
* **Tokenized CoNLL-U predictions**: `check` can operate in `tokenized` mode,
  merging UD MWTs as needed before evaluation.

### Output Formats

* **CoNLL-U**: `document_to_conllu` adds `Entity=` entries, writes
  TokId only when it originates from input, and can rebuild implicit MWT ranges.
* **TEITOK**: `dump_teitok` produces clean `<dtok/>` blocks, adds `<name>`
  wrappers for entity spans, and avoids redundant `_` attributes (except real
  underscores in lemma/form). TokIds are emitted as `xml:id`.
* **Intermediates**: `check` stores predicted and detagged corpora for auditing.

---

## Named Entity Recognition

* SpaCy and other NER-capable backends populate `Sentence.entities`.
* CoNLL-U output encodes entities as `Entity=B-ORG`, `Entity=I-LOC`, etc.
* TEITOK output wraps the affected `<tok>` elements in `<name type="ORG">`.
* Entities can carry arbitrary attributes (copied to TEITOK `<name>` attributes).

---

## Multi-Word Tokens (MWTs)

* Existing MWTs are preserved via `tokid` alignment.
* `_create_implicit_mwt` can synthesize MWT ranges for contractions (based on
  `SpaceAfter=No`) even if the backend did not output them.
* Enable per run with `--create-implicit-mwt` or set the default with
  `flexipipe config --set-default-create-implicit-mwt true`.
* TEITOK exporter keeps `<tok>` text content even when `<dtok>` children exist.

---

## Model & Data Management

* **Shared model directory**: `get_backend_models_dir(backend)` ensures each
  backend uses a subdirectory under `~/.flexipipe/models`.
* **Environment overrides**:
  * `FLEXIPIPE_MODELS_DIR` – forces a different root.
  * Backend-specific envs (e.g., Stanza’s `STANZA_RESOURCES_DIR`) are set to
    refer to the shared directory before imports happen.
* **`config.json`** lives in `~/.flexipipe/`. Editing via the CLI is preferred.

---

## Information Commands

```bash
# List all available backends
python -m flexipipe info backends

# List available models for a backend
python -m flexipipe info models --backend transformers

# List available example texts (UDHR)
python -m flexipipe info examples

# List all supported tasks
python -m flexipipe info tasks
```

## Format Conversion

The `convert` command supports multiple conversion types:

### Convert TEITOK to UD-style CoNLL-U splits

```bash
# Convert TEITOK corpus to train/dev/test splits
python -m flexipipe convert \
  --type treebank \
  --input /path/to/teitok/corpus \
  --output /path/to/output \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1

# Convert CoNLL-U file(s) to splits (handles missing sections)
python -m flexipipe convert \
  --type treebank \
  --input test.conllu \
  --output splits/

# Rerunning preserves existing splits based on sent_id
# (prevents test set contamination when data is updated)
```

**Features:**
- **Longer sent_ids**: Includes source filename (e.g., `corpus-s1`, `article-s2`) for tracking
- **Split preservation**: When rerunning, existing splits are preserved based on `sent_id` to avoid test contamination
- **Handles missing sections**: Can split treebanks that only have test, or only train+test sections
- **CoNLL-U input support**: Can convert from CoNLL-U files directly (not just TEITOK)

### Using Example Texts

```bash
# Process UDHR example for a language
python -m flexipipe process \
  --backend classla \
  --language bg \
  --example udhr
```

---

## Evaluation & Debugging Tips

* Use `--debug` for both `process` and `benchmark` to enable:
  * Curl representations of REST payloads (UDPipe/UDMorph).
  * Tokenization difference samples.
  * Backend-specific log statements (e.g., SPD request durations).
* `--verbose` prints high-level progress plus evaluation summaries.
* `tmp/` directories (e.g., `tmp_spacy_raw/`) capture per-run artifacts for
  manual inspection.

---

## Project Layout

```
flexipipe/              # Main Python package (CLI, backends, converters)
flexitag/               # C++ flexitag sources and bindings
src/                    # Additional C++ helpers (tokenizer, TEITOK writer, etc.)
README_CPP.md           # Native build instructions
```

**Note**: Third-party C++ dependencies (pugixml, rapidjson) are automatically fetched via CMake FetchContent during build and are not stored in the repository.

---

## Contributing & Further Work

* Ensure new features respect existing document structures (`tokid`,
  `Sentence.entities`, `space_after`).
* Keep README sections in sync when adding backends, CLI switches, or config
  keys.
* The transformers backend is fully implemented with model discovery and metadata support.
* Training hooks are available for flexitag and some neural backends.

Please open issues or start discussions if you bump into missing features,
incomplete documentation, or ideas for new backend integrations.

