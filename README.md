# FlexiPipe: Transformer-based Universal Dependencies Tagger and Parser

A fast, accurate FlexiPipe tagger and parser that respects existing annotations and handles contractions, OOV items, and multiple input formats. Full pipeline support for raw text processing.

## Features

- **BERT-based tagging and parsing**: Uses fine-tuned transformer models for UPOS/XPOS/FEATS/LEMMA prediction and dependency parsing
- **Auto-detection of components**: Automatically detects and trains only available components (lemmatizer, parser, normalizer) based on training data
- **Normalization training**: Integrated normalization head that learns orthographic variants from training data (TEITOK @nform/@reg or CoNLL-U Reg=)
- **Tokenizer training**: Train custom WordPiece tokenizers from corpus data (default: enabled)
- **Sentence segmentation**: Rule-based sentence splitting for raw text input
- **Word tokenization**: UD-style tokenization that handles contractions, compounds, and punctuation
- **Context-aware lemmatization**: Uses UPOS/XPOS/FEATS embeddings for better lemma prediction
- **Respects existing annotations**: If UPOS is already provided, it's preserved (LLM-provided annotations take precedence)
- **Contraction handling**: Handles multi-word tokens (MWT) and contractions
- **OOV similarity matching**: Finds similar words based on endings/beginnings for unknown words
- **Vocabulary support**: Word-level vocabulary with linguistic annotations (separate from tokenizer's subword vocab)
- **External vocabulary**: Extend/override model vocabulary without retraining
- **Multiple formats**: Supports CoNLL-U (including VRT format with 1-3 columns), TEITOK XML, plain text, and raw text
- **Full pipeline**: Raw text → sentences → tokens → tags → parse
- **Historic document support**: Normalization and contraction splitting for neotag replacement
- **Fast inference**: Optimized for speed during tagging
- **Accuracy calculation**: Built-in metrics for evaluation (UPOS, XPOS, UFeats, AllTags, Lemma, UAS, LAS, MLAS, BLEX)

## Technologies Used

### Core Architecture

- **BERT (Bidirectional Encoder Representations from Transformers)**: Base transformer model for contextual word representations
  - Default: `bert-base-multilingual-cased` (supports 104 languages, can be changed with `--bert-model`)
  - Language-specific models (e.g., `bert-base-german-cased`) often perform better for specific languages
  - Provides contextual embeddings that capture word sense and syntactic relationships
  - Fine-tuned on Universal Dependencies treebanks for tagging and parsing tasks
  - **Language-agnostic**: The pipeline itself is language-agnostic; you can use any pre-trained BERT model

- **Multi-task Learning**: Single model predicts multiple outputs simultaneously
  - UPOS, XPOS, FEATS: Classification heads (2-layer MLP with GELU activation)
  - LEMMA: Context-aware lemmatization using UPOS/XPOS/FEATS embeddings + BERT embeddings
  - NORM: Normalization head for orthographic variant prediction (trained if normalization data is present)
  - Dependency Parsing: Biaffine attention mechanism for head prediction

### Tokenization

- **WordPiece Tokenization**: Subword tokenization used by BERT
  - Trained on corpus during model training (default: enabled)
  - Handles OOV words by splitting into subword units
  - Vocabulary size: 30,000 tokens (standard BERT size)
  - Saved as `vocab.txt` in model directory

- **UD-style Word Tokenization**: Rule-based word-level tokenization
  - Handles contractions (e.g., `d'`, `l'`, `n'`)
  - Separates punctuation from words
  - Preserves hyphenated compounds
  - Used for sentence-level tokenization before BERT subword tokenization

### Sentence Segmentation

- **Rule-based Segmentation**: Regex-based sentence boundary detection
  - Detects sentence endings: `.`, `!`, `?`
  - Handles German characters (Ä, Ö, Ü)
  - Splits on sentence endings followed by capital letters or end of text
  - Used for raw text input processing

### Lemmatization

- **Context-aware Architecture**: Uses linguistic context for better accuracy
  - BERT embeddings (768-dim) + UPOS embeddings (32-dim) + XPOS embeddings (64-dim) + FEATS embeddings (32-dim)
  - Combined embeddings fed into MLP classification head
  - Predicts lemma labels from training vocabulary
  - Falls back to vocabulary lookup with XPOS context (like neotag strategy)

- **Vocabulary-based Fallback**: 
  - XPOS-aware lookup: `(form, XPOS)` keys for context-aware lemmatization
  - Similarity matching: OOV words matched to similar words with transformation rules
  - Example: "glayed" → "glay" (similar to "played" → "play")

- **Method Selection** (`--lemma-method`):
  - **`bert`**: Prioritize BERT predictions (best for well-resourced languages with standard orthography)
  - **`similarity`**: Prioritize vocabulary/similarity matching (best for Low Resource Languages or historic texts with orthographic variation)
  - **`auto`**: Try BERT first, then vocabulary/similarity (default, general-purpose)

### Dependency Parsing

- **Biaffine Attention**: Neural architecture for dependency head prediction
  - Arc dimension: 500 (configurable)
  - Computes scores for all possible head-dependent pairs: `[batch, seq_len, seq_len]`
  - Memory-efficient: Processes in chunks to avoid memory issues
  - Predicts dependency relations (DEPREL) using separate MLP head

### Vocabulary Management

- **Model Vocabulary**: Word-level vocabulary built from training data
  - Contains: form, lemma, upos, xpos, feats
  - Frequency-based: Uses most frequent annotation combination for ambiguous words
  - Saved as `model_vocab.json` (separate from tokenizer's `vocab.txt`)
  - Automatically loaded when model is loaded

- **External Vocabulary**: Extends/overrides model vocabulary
  - JSON format with same structure as model vocabulary
  - External entries override model entries (priority system)
  - No retraining needed: just add words to external vocab file
  - Supports XPOS-specific lemma entries: `"word:xpos": {"lemma": "..."}`

### Training Techniques

- **Learning Rate Scheduling**:
  - Warmup: 500 steps (or 10% of training)
  - Decay: Cosine annealing
  - Default learning rate: 2e-5

- **Loss Weighting**: Different tasks weighted differently
  - UPOS: 2.0x (most important)
  - XPOS: 1.5x
  - FEATS: 1.0x
  - LEMMA: 1.5x
  - NORM: 1.0x (normalization)
  - Parsing: 1.0x

- **Gradient Accumulation**: Simulates larger batch sizes
  - Default: 2 steps (effective batch = batch_size × 2)
  - Automatically increased for parser training on MPS devices

- **Early Stopping**: Prevents overfitting
  - Patience: 3 epochs without improvement
  - Saves best model based on validation performance

- **Memory Optimization**:
  - Automatic batch size reduction for parser training on MPS (Apple Silicon)
  - Max length reduction (512 → 256) for parser training on MPS
  - Chunked processing for biaffine attention

## Installation

```bash
pip install transformers torch datasets scikit-learn accelerate tokenizers
```

**Required packages**:
- `transformers`: HuggingFace transformers library (for BERT models and Trainer API)
- `torch`: PyTorch (for neural network operations)
- `datasets`: HuggingFace datasets library (for data handling)
- `scikit-learn`: For metrics calculation
- `accelerate>=0.26.0`: Required for training (distributed training and device management)
- `tokenizers`: Required for tokenizer training (optional, but recommended)

## Model Files

When a model is trained, the following files are saved in the output directory:

- **`pytorch_model.bin`**: Model weights (PyTorch state dict)
- **`config.json`**: HuggingFace BERT base model configuration (vocab_size, hidden_size, etc.)
  - This is the BERT model's architecture config, not training settings
  - Used by HuggingFace to reconstruct the model architecture
- **`tokenizer files`**: Tokenizer configuration (vocab.txt, tokenizer_config.json, etc.)
- **`label_mappings.json`**: Label mappings for all tasks (UPOS, XPOS, FEATS, LEMMA, DEPREL, NORM)
  - Used during tagging/parsing to map predictions back to labels
- **`model_vocab.json`**: Word-level vocabulary with linguistic annotations (form, lemma, upos, xpos, feats)
  - Separate from tokenizer's vocab.txt (which contains subword tokens)
  - Used for OOV handling and vocabulary-based fallbacks
- **`training_config.json`**: **Complete training configuration** (all settings used during training)
  - BERT model used (`bert_model`)
  - Which components were trained (`train_tokenizer`, `train_tagger`, `train_parser`, `train_lemmatizer`, `train_normalizer`)
  - Hyperparameters (`batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_epochs`, `max_length`)
  - Normalization attribute (`normalization_attr`)
  - Label counts for each task
  - **Used during tagging**: Automatically loaded to determine which components are available

**Note**: `config.json` is purely informational (HuggingFace BERT architecture). `training_config.json` contains the actual training settings and is used during tagging to determine which components the model supports.

## Usage

### Training

Train a model on CoNLL-U files:

**Using UD treebank directory (recommended)**:
```bash
# For German (using language-specific model for better performance)
python flexipipe.py train \
    --data-dir /path/to/UD_German-GSD \
    --bert-model bert-base-german-cased \
    --output-dir models/flexipipe-german \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 \
    --num-epochs 5

# For any language (using multilingual model)
python flexipipe.py train \
    --data-dir /path/to/UD_Treebank \
    --bert-model bert-base-multilingual-cased \
    --output-dir models/flexipipe \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --num-epochs 5
```

**Using separate train/dev directories**:
```bash
python flexipipe.py train \
    --train-dir path/to/training/conllu/files \
    --dev-dir path/to/dev/conllu/files \
    --bert-model bert-base-german-cased \
    --output-dir models/flexipipe \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --num-epochs 5
```

#### Training Options

**Data Options**:
- `--data-dir`: UD treebank directory (automatically finds `*-ud-train.conllu`, `*-ud-dev.conllu`, `*-ud-test.conllu`)
- `--train-dir`: Directory containing CoNLL-U training files (alternative to `--data-dir`)
- `--dev-dir`: Directory containing CoNLL-U development files (alternative to `--data-dir`)

**Model Options**:
- `--bert-model`: BERT base model to use (default: `bert-base-multilingual-cased`)
  - **Multilingual**: `bert-base-multilingual-cased` (default, supports 104 languages)
  - **Language-specific**: Often better performance for specific languages
    - German: `bert-base-german-cased`, `dbmdz/bert-base-german-cased`
    - Portuguese: `neuralmind/bert-base-portuguese-cased`
    - Spanish: `dccuchile/bert-base-spanish-wwm-uncased`
    - French: `camembert-base`
    - And many more from HuggingFace Model Hub
  - **Other models**: `bert-base-uncased`, `xlm-roberta-base`, `distilbert-base-multilingual-cased`
  - **Language-agnostic pipeline**: Works with any BERT-based model; just specify the appropriate one for your language
- `--output-dir`: Output directory for trained model (default: `models/flexipipe`)

**Component Selection** (default: all enabled, auto-detected from data):
- `--no-tokenizer`: Disable tokenizer training (uses base BERT tokenizer)
- `--no-tagger`: Disable tagger training (UPOS/XPOS/FEATS)
- `--no-parser`: Disable parser training (HEAD/DEPREL)
- `--no-lemmatizer`: Disable lemmatizer training (LEMMA)
- `--no-normalizer`: Disable normalizer training (NORM)
  - **Auto-detection**: Normalizer is automatically enabled if normalization data is found in training files
  - **TEITOK**: Reads `@nform` or `@reg` attributes (configurable with `--normalization-attr`)
  - **CoNLL-U**: Reads `Reg=` from MISC column
  - If no normalization data is found, normalizer training is automatically disabled

**Normalization Options**:
- `--normalization-attr`: TEITOK attribute name for normalization (default: `nform`, can be `reg`)
  - Also used when reading `Reg=` from CoNLL-U MISC column
  - Allows different attribute names for different corpora

**Training Hyperparameters**:
- `--batch-size`: Training batch size (default: 16)
  - Effective batch size = `batch_size × gradient_accumulation_steps`
  - Automatically reduced for parser training on MPS devices
- `--gradient-accumulation-steps`: Number of gradient accumulation steps (default: 2)
  - Simulates larger batch sizes without using more memory
  - Automatically increased for parser training to maintain effective batch size
- `--learning-rate`: Learning rate (default: 2e-5)
  - Typical range: 1e-5 to 5e-5 for BERT fine-tuning
- `--num-epochs`: Number of training epochs (default: 3)

**Note**: By default, all components (tokenizer, tagger, parser, lemmatizer, normalizer) are trained if data is available. The system automatically detects which components to train:
- **Lemmatizer**: Enabled if `lemma` data is present (not "_")
- **Parser**: Enabled if `head` or `deprel` data is present (not "0" or "_")
- **Normalizer**: Enabled if normalization data is present (`@nform`/`@reg` in TEITOK or `Reg=` in CoNLL-U MISC)
- Use `--no-*` flags to explicitly disable components even if data is present

### Tagging

Tag sentences from a file:

**CoNLL-U input**:
```bash
python flexipipe.py tag input.conllu \
    --output output.conllu \
    --format conllu \
    --model models/flexipipe \
    --vocab vocabulary.json \
    --respect-existing
```

**Plain text input (one sentence per line)**:
```bash
python flexipipe.py tag input.txt \
    --output output.conllu \
    --format plain \
    --model models/flexipipe
```

**Raw text input (automatic sentence segmentation and tokenization)**:
```bash
python flexipipe.py tag raw_text.txt \
    --output output.conllu \
    --format raw \
    --model models/flexipipe
```

**Plain text with manual segmentation and tokenization**:
```bash
python flexipipe.py tag input.txt \
    --output output.conllu \
    --format plain \
    --segment \
    --tokenize \
    --model models/flexipipe
```

**Auto-detect format from file extension**:
```bash
python flexipipe.py tag input.txt \
    --output output.conllu \
    --format auto \
    --model models/flexipipe
```

#### Tagging Options

**Input/Output Options**:
- `input`: Input file (required) - CoNLL-U, TEITOK XML, or plain text
- `--output`: Output file (default: stdout)
- `--format`: Input format (`conllu`, `teitok`, `plain`, `text`, `raw`, or `auto`)
  - `conllu`: Standard CoNLL-U format
  - `teitok`: TEITOK XML format
  - `plain`/`text`: Plain text (one sentence per line, or use `--segment` for unsegmented text)
  - `raw`: Raw text (automatically segments sentences and tokenizes words)
  - `auto`: Auto-detect format from file extension
- `--output-format`: Output format (`conllu`, `plain`, `text`, or `plain-tagged`)
  - Defaults to input format or `conllu` for tagged output
  - `plain-tagged`: Plain text with UPOS tags (format: `word/UPOS word/UPOS ...`)

**Text Processing Options**:
- `--segment`: Segment raw text into sentences (for `plain`/`text` format)
- `--tokenize`: Tokenize sentences into words using UD-style rules (for `plain`/`text` format)

**Model Options**:
- `--model`: Path to trained model (if not provided, uses base BERT)
- `--bert-model`: BERT base model if no trained model (default: `bert-base-multilingual-cased`)
  - Language-specific models often perform better (e.g., `bert-base-german-cased` for German)

**Vocabulary Options**:
- `--vocab`: JSON vocabulary file(s) for tuning to local corpus
  - Format: `{"word": {"upos": "...", "xpos": "...", "feats": "...", "lemma": "..."}, "word:xpos": {"lemma": "..."}}`
  - Automatically merged with model vocabulary (external vocab overrides model vocab)
  - **Multiple vocab files supported**: Specify multiple files (e.g., `--vocab general.json local.json`)
    - Later files override earlier ones for the same words
    - All analyses are preserved, but later vocab files get higher priority for sorting
- `--vocab-priority`: Give vocabulary priority over model predictions for all tasks (UPOS/XPOS/FEATS/LEMMA)
  - Useful for tuning to local corpus without retraining
  - When enabled: vocabulary checked first, model predictions as fallback
  - When disabled: model predictions first, vocabulary as fallback
- `--confidence-threshold`: Confidence threshold for confidence-based blending (default: 0.7)
  - If model confidence < threshold, use vocabulary predictions instead
  - Helps with OOV words and domain-specific terms where model is uncertain
  - Works automatically when `--vocab` is provided (no need for `--vocab-priority`)
  - Example: If model predicts UPOS with 0.5 confidence, vocabulary prediction is used if available

**Annotation Options**:
- `--respect-existing`: Preserve existing annotations in input (default: True)
- `--no-respect-existing`: Ignore existing annotations and retag everything

**Parsing Options**:
- `--parse`: Run dependency parsing (predict head and deprel)
  - Requires model trained with `--train-parser` (default: enabled)
- `--tag-only`: Only tag (UPOS/XPOS/FEATS), skip parsing
- `--parse-only`: Only parse (assumes tags already exist), skip tagging
- `--lemma-method`: Lemmatization method (choices: `bert`, `similarity`, `auto`)
  - `bert`: Use BERT model predictions first, fallback to vocabulary/similarity
  - `similarity`: Use vocabulary/similarity matching first, fallback to BERT
  - `auto`: Try BERT first, then vocabulary/similarity (default)
  - **Note**: For Low Resource Languages (LRL) or historic documents with large orthographic variation, `similarity` often outperforms BERT. Use `--lemma-method similarity` for better accuracy on such texts.

**Historic Document Processing** (neotag replacement):
- `--normalize`: Normalize orthographic variants (e.g., "mediaeval" -> "medieval")
  - **Can work without a trained model**: Vocabulary-based normalization doesn't require a model - just provide `--vocab` with normalization mappings (see example below)
  - **Priority order**:
    1. **Model's normalizer** (if model was trained with normalization): Uses the trained normalizer head from the main model (highest priority during tagging)
    2. **Explicit normalization in vocabulary** (`reg` field): If vocabulary entry has `reg`, uses that (domain-specific mappings)
    3. **Vocabulary-based similarity**: Falls back to similarity matching
  - **Note**: The normalizer is integrated into the main model (`--model`). When a model trained with normalization is loaded, the normalizer is automatically available - no separate normalizer model needed.
  - **Local vocabulary integration**: The external vocabulary (`--vocab`) is especially important for historic documents, as it can provide domain-specific normalization mappings based on transcription standards, region, period, and register
  - Conservative by default (only normalizes if high confidence)
  
  **Normalization-only mode (no model required)**:
  ```bash
  # Step 1: Create vocabulary from TEITOK XML files (extracts forms, lemmas, UPOS, XPOS, FEATS, and reg normalization)
  python flexipipe_create_vocab.py /path/to/teitok/xml/folder \
      --output historic_vocab.json \
      --reg reg  # Use @reg attribute for normalization (default)
  
  # Step 2: Normalize text using only vocabulary (no model needed)
  # Input can be TEITOK XML, CoNLL-U, or plain text
  python flexipipe.py tag historic_text.xml \
      --vocab historic_vocab.json \
      --normalize \
      --output normalized.conllu \
      --format teitok  # or 'conllu', 'plain', 'raw', etc.
  
  # For plain text input (requires segmentation and tokenization):
  python flexipipe.py tag historic_text.txt \
      --vocab historic_vocab.json \
      --normalize \
      --format raw \
      --output normalized.conllu
  ```
  
  **Note**: The vocabulary-based normalization uses similarity matching to find the closest normalized form in the vocabulary. Words with explicit `reg` fields in the vocabulary are normalized directly; otherwise, similarity-based matching is used.
- `--normalization-attr`: TEITOK attribute name for normalization (default: `reg`, can be `nform`)
  - Used when reading TEITOK XML files
  - Also used when reading `Reg=` from CoNLL-U MISC column
  - Allows different attribute names for different corpora
- `--no-conservative-normalization`: Disable conservative normalization (more aggressive)
- `--tag-on-normalized`: Tag on normalized form instead of original orthography
  - Requires `--normalize`
  - Original form preserved in MISC column
- `--split-contractions`: Split contractions (e.g., "destas" -> "de estas")
  - Useful for historic texts where more things are written together
  - Uses vocabulary to identify potential contractions
- `--aggressive-contraction-splitting`: More aggressive splitting patterns for historic texts
  - Requires `--split-contractions`
  - Handles patterns like Spanish "destas", "dellos", etc.
- `--language`: Language code for language-specific contraction rules
  - `es` (Spanish): Handles verb+clitics without hyphens (e.g., "dámelo" = "dé" + "me" + "lo")
  - `pt` (Portuguese): Handles hyphenated clitics (e.g., "faze-lo" = "faze" + "lo")
  - `ltz` or `lb` (Luxembourgish): Handles apostrophe contractions (e.g., "d'XXX" = "de" + "XXX")
  - Enables rule-based splitting for modern languages

**Ambiguity Handling**:
- If a word exists as a single word in vocabulary, it's treated as ambiguous
- Ambiguous words are only split if there's strong evidence (e.g., verb part exists in vocab)
- This prevents over-splitting (e.g., "kárate" stays as single word, not "kára" + "te")
- Portuguese "pelo" can be "hair" (single word) or "por" + "lo" (contraction) - only splits if verb part exists

**Note**: The parser automatically uses existing UPOS/FEATS/XPOS when available (via `--respect-existing`), so the workflow of normalize → tag → correct → parse is fully supported.

### Calculate Accuracy

Compare predictions against gold standard:

```bash
python flexipipe.py calculate-accuracy \
    gold.conllu pred.conllu \
    --format conllu
```

**Accuracy Metrics**:
- **Words**: Total number of tokens
- **Sentences**: Total number of sentences
- **UPOS**: Universal part-of-speech accuracy
- **XPOS**: Language-specific part-of-speech accuracy
- **UFeats**: Universal morphological features accuracy
- **AllTags**: UPOS + XPOS + FEATS all correct
- **Lemma**: Lemmatization accuracy
- **UAS**: Unlabeled Attachment Score (head prediction accuracy)
- **LAS**: Labeled Attachment Score (head + deprel prediction accuracy)
- **MLAS**: Morphological LAS (includes morphological features)
- **BLEX**: Bilexical dependency accuracy (head + deprel + head's form/lemma)

## Input Formats

### CoNLL-U Format

Full CoNLL-U (10 columns):
```
1	Die	die	DET	ART	Definite=Def|PronType=Art	2	det	_	_
2	Motivation	Motivation	NOUN	NN	Case=Nom|Number=Sing	0	root	_	_
```

**Normalization in MISC column**:
- Normalization is stored as `Reg=` in the MISC column (10th column)
- Example: `1	Die	die	DET	ART	...	_	Reg=die`
- Multiple MISC fields separated by `|`: `Reg=die|OrigForm=Die`
- Automatically read during training and tagging

VRT format (1-3 columns):
```
Die
die	DET
```

### Plain Text Format

One sentence per line (blank lines separate sentences):
```
Die Motivation hat geändert .
D'Wieler ginn e Sonndeg wielen .
```

The tagger will:
- Tokenize by whitespace (simple word splitting)
- Or use `--segment` and `--tokenize` for automatic segmentation and UD-style tokenization
- Assign UPOS/XPOS/FEATS tags using the model
- Output in CoNLL-U format by default (or specify `--output-format plain-tagged` for `word/UPOS` format)

### TEITOK XML Format

```xml
<s id="s-1">
  <tok id="w-1" lemma="die" upos="DET" nform="die">Die</tok>
  <tok id="w-2" reg="zur">
    <dtok id="w-2.1" form="zu" upos="ADP"/>
    <dtok id="w-2.2" form="der" upos="DET"/>
  </tok>
</s>
```

**Normalization attributes**:
- Default: `@nform` (can be changed with `--normalization-attr`)
- Fallback: `@reg` (if `@nform` not found)
- Both attributes are checked automatically
- Normalization is stored in `norm_form` field and written as `Reg=` in CoNLL-U MISC column

## Vocabulary File Format

### Model Vocabulary

Built automatically during training, saved as `model_vocab.json`. Contains word-level entries with linguistic annotations extracted from training data.

### External Vocabulary

FlexiPipe supports multiple vocabulary formats:

1. **FlexiPipe JSON format** (recommended): JSON file for tuning to local corpus (extends/overrides model vocabulary). Can be created using `flexipipe create-vocab`.
2. **UniMorph lexicon format**: Tab-separated format (`form\tlemma\tfeatures`) for morphological lexicons.

### Format Specification

#### FlexiPipe JSON Format

**Single Analysis (Unambiguous Word)**:
```json
{
  "word": {
    "upos": "NOUN",
    "xpos": "NN",
    "feats": "Case=Nom|Number=Sing",
    "lemma": "word",
    "count": 150
  },
  "mediaeval": {
    "reg": "medieval",
    "upos": "ADJ",
    "lemma": "medieval",
    "count": 5
  }
}
```

**Note**: 
- The `reg` field is optional and used for explicit normalization mappings (corresponds to `Reg=` in CoNLL-U and `@reg` in TEITOK). This is especially useful for historic documents where orthographic variants depend on transcription standards, region, period, and register. If `reg` is present, it takes priority over similarity-based normalization.
- The `count` field is **optional**. If missing, a default count of 1 is used. Counts are useful for disambiguation when a word has multiple possible analyses (see below). Vocabularies without counts work perfectly fine for lemmatization and tagging.

**Multiple Analyses (Ambiguous Word)**:
For words with multiple possible analyses (different UPOS/XPOS/FEATS/LEMMA combinations), use an array. The `count` field (if present) helps determine the most likely analysis:
```json
{
  "band": [
    {
      "upos": "NOUN",
      "xpos": "NN",
      "feats": "Case=Nom|Number=Sing",
      "lemma": "band",
      "count": 120
    },
    {
      "upos": "VERB",
      "xpos": "VBN",
      "feats": "Tense=Past|VerbForm=Part",
      "lemma": "bind",
      "count": 15
    }
  ]
}
```
When disambiguating, the tagger uses the `count` field to prefer the most frequent analysis (e.g., "band" as NOUN with count=120 is more likely than VERB with count=15).

**Case-Sensitive Distinctions**:
Case-sensitive forms are stored separately to handle language-specific distinctions:
```json
{
  "Band": {
    "upos": "NOUN",
    "xpos": "NN",
    "lemma": "Band"
  },
  "band": {
    "upos": "VERB",
    "xpos": "VVFIN",
    "lemma": "binden"
  }
}
```

**Field Inclusion Rules**:
- **Fields set to "_" are omitted** (except `lemma` which is always included)
- **`lemma` is always included** (even if "_")
- **`upos`, `xpos`, `feats`**: Only included if not "_"

**Example - Minimal Vocabulary**:
```json
{
  "word": {
    "lemma": "word"
  },
  "another": {
    "upos": "VERB",
    "lemma": "another"
  },
  "ambiguous": [
    {
      "upos": "NOUN",
      "lemma": "ambiguous"
    },
    {
      "upos": "ADJ",
      "lemma": "ambiguous"
    }
  ]
}
```

#### UniMorph Lexicon Format

FlexiPipe can load UniMorph lexicons directly. UniMorph format is tab-separated:

```
lemma	form	features
```

**Example**:
```
lächeln	lächelt	V;IND;SG;3;PRS
gehen	geht	V;IND;SG;3;PRS
Haus	Häuser	N;PL;NOM
```

**Format details**:
- **lemma**: The lemma (dictionary form) - first column
- **form**: The inflected word form - second column
- **features**: UniMorph features (semicolon-separated, e.g., `V;IND;SG;3;PRS`) - third column

**Note**: UniMorph format is `lemma\tform\tfeatures`, not `form\tlemma\tfeatures`. The lemma comes first.

FlexiPipe automatically converts UniMorph features to UD FEATS format where possible:
- UniMorph POS tags (e.g., `V`, `N`, `ADJ`) are mapped to UPOS
- Common UniMorph features are converted to UD format:
  - `PST` → `Tense=Past`, `PRS` → `Tense=Pres`, `FUT` → `Tense=Fut`
  - `SG` → `Number=Sing`, `PL` → `Number=Plur`
  - `1`, `2`, `3` → `Person=1`, `Person=2`, `Person=3`
  - `NOM`, `ACC`, `GEN`, `DAT` → `Case=Nom`, `Case=Acc`, `Case=Gen`, `Case=Dat`
  - And more...

**Usage**:
```bash
# Use as lexicon (OOV fallback only)
python -m flexipipe tag input.txt --lexicon lexicon.tsv --vocab corpus_vocab.json --output output.conllu

# Or use lexicon alone (if no corpus vocabulary available)
python -m flexipipe tag input.txt --lexicon lexicon.tsv --output output.conllu
```

**Important**: Lexicons are used as **fallback for OOV words only**. They are not merged with vocabulary files. This means:
- If a word is in the vocabulary (from `--vocab`), the vocabulary entry is used
- If a word is OOV (not in vocabulary), the lexicon is checked as a fallback
- This ensures corpus-specific vocabulary (with counts) takes priority over general lexicons

UniMorph lexicons are particularly useful for:
- Low-resource languages with available UniMorph data
- Adding morphological information from external lexicons
- Supplementing corpus-derived vocabularies with comprehensive morphological paradigms

**Note**: UniMorph entries are assigned a default count of 1 (since UniMorph doesn't include frequency information). This works fine for lemmatization and tagging.

**Vocabulary Features**:
- **Arrays for ambiguity**: Multiple analyses stored as arrays, sorted by frequency (most frequent first)
- **Case-sensitive**: Preserves case distinctions (e.g., "Band" vs "band" in German)
- **Context-aware lookup**: Uses XPOS/UPOS context to disambiguate when available
- **Priority**: External vocabulary overrides model vocabulary for same words
- **Compact format**: Only includes non-"_" fields (except lemma)

## How It Works

### Vocabulary System

1. **Model Vocabulary**: Built from training data during training
   - Contains all words seen in training with their annotations
   - Uses arrays for ambiguous words (multiple analyses)
   - Preserves case-sensitive distinctions
   - Saved as `model_vocab.json` in model directory
   - Automatically loaded when model is loaded

2. **External Vocabulary**: Extends/overrides model vocabulary
   - Loaded via `--vocab` argument
   - Can be created using `flexipipe_create_vocab.py` from TEITOK XML files
   - Merged with model vocabulary (external entries override model entries)
   - No retraining needed: just add words to external vocab file
   - Supports arrays for ambiguous words
   - Supports case-sensitive distinctions

3. **Lookup Strategy**:
   - **Case-sensitive**: Tries exact case match first (e.g., "Band"), then falls back to lowercase (e.g., "band")
   - **Context-aware**: Uses XPOS/UPOS context to disambiguate array entries
   - **Fallback**: If no exact match, uses most frequent (first) analysis in array

4. **Priority System**:
   - **`--vocab-priority`**: Vocabulary checked first, model predictions as fallback (always use vocab if available)
   - **Confidence-based blending** (default when `--vocab` is provided): Model predictions first, but if model confidence < threshold, use vocabulary instead
     - Helps with OOV words and domain-specific terms where model is uncertain
     - Configurable via `--confidence-threshold` (default: 0.7)
     - Example: Model predicts with 0.5 confidence → vocabulary prediction used if available
   - **Without `--vocab-priority` and low confidence**: Model predictions first, vocabulary as fallback

### Annotation Pipeline

1. **Respects Existing Annotations**: If UPOS/XPOS/FEATS are already provided in the input, they are preserved (unless `--no-respect-existing` is used)

2. **OOV Handling**: For unknown words:
   - First checks vocabulary file (if `--vocab-priority` enabled)
   - Then uses model predictions
   - Then uses vocabulary lookup (if `--vocab-priority` disabled)
   - Finally uses similarity matching (endings/beginnings)
   - Falls back to default (usually NOUN)

3. **Context-aware Lemmatization**:
   - Uses BERT embeddings + UPOS/XPOS/FEATS embeddings
   - Predicts lemma labels from training vocabulary
   - Falls back to XPOS-aware vocabulary lookup: `(form, XPOS)` keys
   - Similarity matching with transformation rules for OOV words

4. **Contraction Handling**: 
   - Detects multi-word tokens in input
   - Expands contractions to component parts
   - Maintains proper alignment

5. **Similarity Matching**: 
   - Checks word endings (last 3-6 characters)
   - Checks word beginnings (first 2-4 characters)
   - Scores by length similarity
   - Uses threshold (default: 0.7) to filter candidates
   - For lemmatization: applies transformation rules (e.g., remove `-ed`, `-ing` suffixes)

### Memory Management

- **MPS (Apple Silicon) Optimization**:
  - Automatic batch size reduction for parser training (16 → 4 or 1)
  - Max length reduction (512 → 256) for parser training
  - Gradient accumulation increased to maintain effective batch size
  - Chunked processing for biaffine attention (64 tokens at a time)

- **CUDA/CPU**: Uses standard batch sizes with automatic reduction only for parser training

## Performance

### Expected Accuracy (German-GSD)

With proper training (5 epochs, MLP heads, loss weighting, learning rate scheduling):
- **UPOS**: ~96-97% (SOTA: UDPIPE2 ~97%)
- **XPOS**: ~97-98%
- **UFeats**: ~91-92%
- **AllTags**: ~89-90%
- **Lemma**: ~97-98% (with context-aware lemmatization)
- **UAS**: ~89%
- **LAS**: ~85%
- **MLAS**: ~68%
- **BLEX**: ~77%

### Training Time

- **Mac Studio (M1 Max)**: ~2-4 hours per epoch (with parser training)
- **NVIDIA GPU**: ~30-60 minutes per epoch (with parser training)
- **CPU**: Significantly slower (not recommended for training)

## Integration Examples

### With annotate_transpose.py

The tagger can be used to improve tagging accuracy when LLM-provided UPOS annotations are available:

```python
# In annotate_transpose.py workflow:
# 1. LLM provides UPOS annotations in JSON
# 2. FlexiPipe respects those annotations (--respect-existing)
# 3. Only fills in missing annotations (XPOS, FEATS, LEMMA)
# 4. Uses BERT for better accuracy on unknown words
```

### Creating a Custom Vocabulary

Create a vocabulary file from TEITOK XML files:

```bash
# Create vocabulary from TEITOK XML files (recursive)
python flexipipe_create_vocab.py \
    --folder /path/to/teitok/xml/files \
    --output custom_vocab.json \
    --xpos-attr xpos

# Use different XPOS attribute (e.g., 'pos' or 'msd')
python flexipipe_create_vocab.py \
    --folder /path/to/corpus \
    --output vocab.json \
    --xpos-attr pos
```

**Features**:
- Recursively processes all XML files in the folder
- Handles TEITOK format with `<tok>` and `<dtok>` elements
- Supports configurable XPOS attribute (`xpos`, `pos`, `msd`)
- Creates arrays for ambiguous words (multiple analyses)
- Preserves case-sensitive distinctions
- Omits fields set to "_" (except `lemma`)

### Tuning to Local Corpus

```bash
# 1. Train base model on UD treebank
python flexipipe.py train --data-dir /path/to/UD_German-GSD --output-dir models/base

# 2. Create local vocabulary file from your corpus
python flexipipe_create_vocab.py \
    --folder /path/to/local/corpus \
    --output local_vocab.json

# 3a. Use with vocab priority for domain-specific tagging (always use vocab if available)
python flexipipe.py tag input.txt \
    --model models/base \
    --vocab local_vocab.json \
    --vocab-priority \
    --output output.conllu

# 3b. Use with confidence-based blending (use vocab when model is uncertain)
python flexipipe.py tag input.txt \
    --model models/base \
    --vocab local_vocab.json \
    --confidence-threshold 0.7 \
    --output output.conllu

# 3c. Combine multiple vocab files (general + specific)
python flexipipe.py tag input.txt \
    --model models/base \
    --vocab general_vocab.json local_vocab.json \
    --confidence-threshold 0.7 \
    --output output.conllu
```

**Confidence-based blending** (option 3b) is recommended for most use cases:
- Uses model predictions when confident (high accuracy)
- Falls back to vocabulary when model is uncertain (OOV words, domain-specific terms)
- Best of both worlds: model accuracy + vocabulary coverage

### Historic Document Processing (neotag replacement)

Workflow: normalize → tag → correct → parse

```bash
# Step 1: Normalize and tag
python flexipipe.py tag historic_text.conllu \
    --model models/base \
    --vocab historic_vocab.json \
    --normalize \
    --tag-on-normalized \
    --split-contractions \
    --aggressive-contraction-splitting \
    --lemma-method similarity \
    --output normalized_tagged.conllu

# Step 2: (Manual correction of tags/lemmas in normalized_tagged.conllu)

# Step 3: Parse with corrected tags
python flexipipe.py tag corrected_tagged.conllu \
    --model models/base \
    --parse-only \
    --respect-existing \
    --output final_parsed.conllu
```

**Features**:
- **Normalization**: Integrated into main model training (if normalization data is present)
  - **During training**: Automatically detects normalization data from TEITOK (`@nform`/`@reg`) or CoNLL-U (`Reg=`)
  - **During tagging**: Uses trained normalizer if available, falls back to vocabulary-based if not
  - Conservative by default (only normalizes if high confidence)
- **Normalization formats**:
  - **TEITOK**: `@nform` or `@reg` attributes (configurable with `--normalization-attr`)
  - **CoNLL-U**: `Reg=` in MISC column (standard UD format)
  - **Output**: Always writes `Reg=` in CoNLL-U MISC column
- **Contraction splitting**: Handles historic texts where more things are written together
- **Original forms preserved**: Stored in MISC column (OrigForm, Reg=, SplitForms)
- **Parser respects existing tags**: Uses corrected UPOS/FEATS/XPOS for better parsing

**Training with normalization data**:

```bash
# Training data with normalization (TEITOK XML with @nform/@reg)
python flexipipe.py train \
    --data-dir /path/to/training/data \
    --bert-model bert-base-portuguese-cased \
    --output-dir models/flexipipe-pt-historic \
    --normalization-attr nform  # or 'reg' if your corpus uses @reg
    # Normalizer automatically enabled if normalization data is found

# Training data with normalization (CoNLL-U with Reg= in MISC)
python flexipipe.py train \
    --data-dir /path/to/training/data \
    --bert-model bert-base-portuguese-cased \
    --output-dir models/flexipipe-pt-historic
    # Automatically reads Reg= from MISC column
```

**Normalization during tagging**:

```bash
# If model was trained with normalization, it's used automatically
# Local vocabulary provides domain-specific normalization mappings
python flexipipe.py tag old_portuguese.conllu \
    --model models/flexipipe-pt-historic \
    --vocab portuguese_historic_vocab.json \
    --normalize \
    --output normalized.conllu

# Vocabulary with explicit normalization mappings:
# {
#   "vossa": {"reg": "vossa", "upos": "PRON", ...},
#   "quẽ": {"reg": "que", "upos": "SCONJ", ...},
#   "mediaeval": {"reg": "medieval", "upos": "ADJ", ...}
# }
```

**Vocabulary normalization priority**:
1. **Explicit `reg` field**: If vocabulary entry has `reg`, uses that directly (highest priority)
2. **Trained normalizer**: If model was trained with normalization data, uses model predictions
3. **Similarity matching**: Falls back to finding similar words in vocabulary

This allows domain-specific normalization (transcription standards, region, period, register) through the local vocabulary without retraining the model.

## Troubleshooting

### Memory Issues on MPS

If you get "MPS backend out of memory" errors:
- The code automatically reduces batch size for parser training
- If still failing, try reducing `--batch-size` further (e.g., 8 or 4)
- Or disable parser training: `--no-parser`

### Low Accuracy

- Ensure model was trained with all components: `--train-tagger --train-lemmatizer --train-parser`
- Check that test set wasn't used in training
- Use `--no-respect-existing` when calculating accuracy to avoid data leakage
- Increase training epochs: `--num-epochs 5` or more
- Try different learning rates: `--learning-rate 1e-5` to `5e-5`

### Low Lemma Accuracy (especially for LRL/historic texts)

- Try `--lemma-method similarity` instead of default `auto` or `bert`
- Similarity-based lemmatization often outperforms BERT for:
  - Low Resource Languages (LRL)
  - Historic documents with orthographic variation
  - Texts with non-standard spelling
- Use `--lemma-method similarity` when calculating accuracy to test performance:
  ```bash
  python flexipipe.py tag input.conllu --model models/flexipipe --output pred.conllu --lemma-method similarity
  python flexipipe.py calculate-accuracy gold.conllu pred.conllu
  ```

### Tokenizer Warnings

If you see "tokenizers parallelism" warnings:
- These are harmless and suppressed automatically
- Or set `TOKENIZERS_PARALLELISM=false` environment variable

## Future Enhancements

See `dev/TODO.md` for a complete list of planned enhancements and improvements.

**Note**: The pipeline itself is already language-agnostic and supports all languages through any BERT model. Language-specific models (e.g., `bert-base-german-cased`) typically perform better than multilingual models for individual languages, but multilingual models enable handling multiple languages with a single model.
