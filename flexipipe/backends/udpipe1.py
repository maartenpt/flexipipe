"""Backend spec and implementation for the UDPipe CLI backend (udpipe1)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document, document_to_conllu, parse_conllu_from_backend
from ..doc import Document
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult


def _document_to_plain_text(document: Document) -> str:
    """Convert a Document to plain text for UDPipe processing."""
    sentences = []
    for sentence in document.sentences:
        if sentence.text:
            sentences.append(sentence.text)
        else:
            # Reconstruct from tokens
            tokens = []
            for token in sentence.tokens:
                if token.is_mwt and token.subtokens:
                    tokens.extend([st.form for st in token.subtokens])
                else:
                    tokens.append(token.form)
            sentences.append(" ".join(tokens))
    return "\n".join(sentences)


class UDPipeCLIBackend(BackendManager):
    DEFAULT_TOKENIZER_OPTIONS = (
        "epochs=80,batch_size=50,segment_size=50,learning_rate=0.005,"
        "learning_rate_final=0.0005,dropout=0.1,early_stopping=1,tokenize_url=1,"
        "allow_spaces=0,dimension=24"
    )

    @staticmethod
    def _detect_annotation_coverage(conllu_path: Path) -> Dict[str, bool]:
        """
        Inspect a CoNLL-U file and detect which annotation columns contain data.

        Returns a dictionary with flags for lemma, upos, xpos, feats, head, deprel.
        """
        coverage = {
            "lemma": False,
            "upos": False,
            "xpos": False,
            "feats": False,
            "head": False,
            "deprel": False,
        }
        required = set(coverage.keys())

        try:
            with conllu_path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 8:
                        continue
                    token_id = parts[0]
                    if "-" in token_id or "." in token_id:
                        # Skip multi-word tokens and empty nodes
                        continue
                    lemma, upos, xpos, feats = parts[2:6]
                    head = parts[6]
                    deprel = parts[7] if len(parts) > 7 else ""

                    if lemma and lemma != "_":
                        coverage["lemma"] = True
                    if upos and upos != "_":
                        coverage["upos"] = True
                    if xpos and xpos != "_":
                        coverage["xpos"] = True
                    if feats and feats != "_":
                        coverage["feats"] = True
                    if head and head != "_":
                        coverage["head"] = True
                    if deprel and deprel != "_":
                        coverage["deprel"] = True

                    if all(coverage[key] for key in required):
                        break
        except OSError:
            # If the file can't be read, leave coverage as False for all fields
            pass

        return coverage

    @staticmethod
    def _split_udpipe_options(options: Optional[str], default: Optional[str]) -> List[str]:
        text = options if options is not None else default
        if not text:
            return []
        cleaned = text.strip()
        if not cleaned:
            return []
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace(":", ",")
        parts = [part for part in cleaned.split(",") if part]
        return parts

    DEFAULT_TAGGER_OPTIONS = ""
    DEFAULT_PARSER_OPTIONS = ""
    """
    UDPipe CLI backend for tagging and training.
    
    This backend uses the UDPipe CLI tool for fast training and inference.
    but available for use in debug_accuracy and as a backend for UDMorph.
    """
    
    def __init__(
        self,
        model: str,
        udpipe_binary: str = "udpipe",
        timeout: Optional[int] = None,
        verbose: bool = False,
        require_model_exists: bool = True,
    ):
        """
        Initialize UDPipe CLI backend.
        
        Args:
            model: Path to UDPipe model file (.udpipe) or model name (will search in models_dir/udpipe1/)
            udpipe_binary: Path to UDPipe binary (default: "udpipe" in PATH)
            timeout: Timeout for subprocess calls (seconds)
            verbose: Whether to print verbose output
        """
        # Try to resolve model path
        model_path = Path(model).expanduser()
        
        resolved_path = None
        if model_path.exists():
            resolved_path = model_path
        elif not model_path.is_absolute():
            models_dir = get_backend_models_dir("udpipe1", create=False)
            candidates = [
                models_dir / f"{model}.udpipe",
                models_dir / model,
            ]
            for cand in candidates:
                if cand.exists():
                    resolved_path = cand
                    break
            if resolved_path is None and require_model_exists:
                raise FileNotFoundError(
                    f"UDPipe model not found: {model}. "
                    f"Tried: {candidates[0]}, {candidates[1]}"
                )
            if resolved_path is None:
                resolved_path = (models_dir / f"{model}.udpipe").resolve()
        else:
            if require_model_exists:
                raise FileNotFoundError(f"UDPipe model not found: {model_path}")
            resolved_path = model_path
        
        self._model = resolved_path.resolve()
        
        self._udpipe_binary = udpipe_binary
        self._timeout = timeout
        self._verbose = verbose
        self._backend_name = "UDPipe CLI"
        self._model_name = self._model.stem
    
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
        **kwargs,
    ) -> NeuralResult:
        """
        Tag a document using UDPipe CLI.
        
        Args:
            document: Input document
            use_raw_text: If True, send raw text; if False, send pre-tokenized CoNLL-U
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Tagged document
        """
        if use_raw_text:
            # Convert to plain text
            input_text = _document_to_plain_text(document)
            input_format = "plain"
        else:
            # Convert to CoNLL-U
            input_text = document_to_conllu(document, create_implicit_mwt=False)
            input_format = "conllu"
        
        # Write input to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=f".{input_format}", delete=False, encoding="utf-8") as f:
            f.write(input_text)
            input_file = Path(f.name)
        
        try:
            # Build UDPipe command
            # UDPipe CLI: udpipe [--tokenize] [--tag] [--parse] [--input=FORMAT] [--output=FORMAT] MODEL INPUT
            # Note: --no-tokenize doesn't exist; omit --tokenize for pre-tokenized input
            cmd = [self._udpipe_binary]
            
            if use_raw_text:
                cmd.append("--tokenize")
            # For pre-tokenized input, just omit --tokenize (don't use --no-tokenize)
            
            # Add components based on what's requested
            if components is None:
                # Default: tag and parse
                cmd.extend(["--tag", "--parse"])
            else:
                if "tagger" in components or "tag" in components:
                    cmd.append("--tag")
                if "parser" in components or "parse" in components:
                    cmd.append("--parse")
            
            cmd.extend([
                f"--input={input_format}",
                "--output=conllu",
                str(self._model),
                str(input_file),
            ])
            
            # Run UDPipe
            if self._verbose:
                print(f"[flexipipe] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,  # Don't raise on non-zero exit, check manually
            )
            
            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                
                # If parser is not available, try again without --parse
                if "No parser defined" in error_msg and "--parse" in cmd:
                    if self._verbose:
                        print(f"[flexipipe] Parser not available, retrying without --parse")
                    # Remove --parse from command and try again
                    cmd_no_parse = [c for c in cmd if c != "--parse"]
                    result = subprocess.run(
                        cmd_no_parse,
                        capture_output=True,
                        text=True,
                        timeout=self._timeout,
                        check=False,
                    )
                    if result.returncode == 0:
                        # Success without parser
                        output_text = result.stdout
                        if output_text.strip():
                            tagged_doc = parse_conllu_from_backend(output_text, document)
                            return NeuralResult(document=tagged_doc, stats={})
                
                # If we get here, it's a real error
                if self._verbose:
                    print(f"[flexipipe] UDPipe stderr: {error_msg}")
                raise RuntimeError(
                    f"UDPipe CLI failed with exit code {result.returncode}: {error_msg}"
                )
            
            # UDPipe writes to stdout
            output_text = result.stdout
            
            if not output_text.strip():
                error_msg = result.stderr.strip() if result.stderr else "No output produced"
                raise RuntimeError(f"UDPipe produced no output. stderr: {error_msg}")
            
            # Parse output
            tagged_doc = parse_conllu_from_backend(output_text, document)
            
            return NeuralResult(document=tagged_doc, stats={})
            
        finally:
            # Clean up temporary files
            input_file.unlink(missing_ok=True)
    
    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        language: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: bool = False,
        tokenizer_options: Optional[str] = None,
        tagger_options: Optional[str] = None,
        parser_options: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Train a UDPipe model using UDPipe CLI.
        
        Args:
            train_data: Path to training CoNLL-U file (or Document/List[Document] - will be converted)
            output_dir: Directory to save the trained model
            dev_data: Optional path to dev CoNLL-U file (or Document/List[Document])
            language: Language code (optional)
            model_name: Model name (optional, defaults to output_dir name)
            verbose: Whether to print verbose output
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Path to the trained model file
        """
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            model_name = output_dir.name
        
        model_path = output_dir / f"{model_name}.udpipe"
        
        # Convert train_data to Path if it's a Document or List[Document]
        train_path: Path
        if isinstance(train_data, (Document, list)):
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".conllu", delete=False, encoding="utf-8") as f:
                if isinstance(train_data, Document):
                    conllu_text = document_to_conllu(train_data, create_implicit_mwt=False)
                else:
                    # List of documents
                    conllu_parts = [document_to_conllu(doc, create_implicit_mwt=False) for doc in train_data]
                    conllu_text = "\n\n".join(conllu_parts)
                f.write(conllu_text)
                train_path = Path(f.name)
        else:
            train_path = Path(train_data).expanduser().resolve()
            if train_path.is_dir():
                from ..train import _find_ud_splits  # Local import to avoid circular at top-level
                splits = _find_ud_splits(train_path)
                if "train" not in splits:
                    raise ValueError(f"No train split found in directory: {train_path}")
                train_path = splits["train"]
                if dev_data is None and "dev" in splits:
                    dev_path = splits["dev"]
        
        # Convert dev_data to Path if needed
        dev_path: Optional[Path] = None
        if dev_data:
            if isinstance(dev_data, (Document, list)):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".conllu", delete=False, encoding="utf-8") as f:
                    if isinstance(dev_data, Document):
                        conllu_text = document_to_conllu(dev_data, create_implicit_mwt=False)
                    else:
                        conllu_parts = [document_to_conllu(doc, create_implicit_mwt=False) for doc in dev_data]
                        conllu_text = "\n\n".join(conllu_parts)
                    f.write(conllu_text)
                    dev_path = Path(f.name)
            else:
                dev_path = Path(dev_data).expanduser().resolve()
        
        try:
            # Build UDPipe train command
            coverage = self._detect_annotation_coverage(train_path)
            has_tagger_annotations = any(
                coverage[key] for key in ("lemma", "upos", "xpos", "feats")
            )
            has_parser_annotations = coverage["head"] and coverage["deprel"]

            tokenizer_opts = self._split_udpipe_options(tokenizer_options, self.DEFAULT_TOKENIZER_OPTIONS)
            tagger_opts = self._split_udpipe_options(tagger_options, self.DEFAULT_TAGGER_OPTIONS)
            parser_opts = self._split_udpipe_options(parser_options, self.DEFAULT_PARSER_OPTIONS)

            tagger_disabled_by_user = any(opt.lower() == "none" for opt in tagger_opts)
            parser_disabled_by_user = any(opt.lower() == "none" for opt in parser_opts)

            user_set_tagger = tagger_options is not None
            user_set_parser = parser_options is not None

            cmd = [
                self._udpipe_binary,
                "--train",
            ]

            for opt in tokenizer_opts:
                cmd.append(f"--tokenizer={opt}")

            if tagger_disabled_by_user:
                cmd.append("--tagger=none")
            elif not has_tagger_annotations:
                if user_set_tagger and tagger_opts:
                    print(
                        "[flexipipe] WARNING: Training data lacks lemma/upos/xpos/feats annotations; "
                        "ignoring provided UDPipe tagger options and disabling tagger training."
                    )
                else:
                    print(
                        "[flexipipe] Training data lacks lemma/upos/xpos/feats annotations; "
                        "skipping UDPipe tagger training."
                    )
                cmd.append("--tagger=none")
            else:
                for opt in tagger_opts:
                    cmd.append(f"--tagger={opt}")

            if parser_disabled_by_user:
                cmd.append("--parser=none")
            elif not has_parser_annotations:
                if not user_set_parser:
                    print(
                        "[flexipipe] Training data has no dependency annotations (HEAD/DEPREL); "
                        "skipping UDPipe parser training."
                    )
                else:
                    print(
                        "[flexipipe] WARNING: Training data has no dependency annotations (HEAD/DEPREL); "
                        "ignoring provided UDPipe parser options and disabling parser training."
                    )
                cmd.append("--parser=none")
            else:
                for opt in parser_opts:
                    cmd.append(f"--parser={opt}")

            if dev_path:
                cmd.append(f"--heldout={dev_path}")

            cmd.extend(
                [
                    str(model_path),
                    str(train_path),
                ]
            )
            
            if verbose:
                print(f"[flexipipe] Training UDPipe model: {' '.join(cmd)}")
            
            if verbose:
                try:
                    subprocess.run(
                        cmd,
                        text=True,
                        timeout=self._timeout,
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    err_text = exc.stderr or exc.stdout or str(exc)
                    raise RuntimeError(
                        f"UDPipe training failed with exit code {exc.returncode}: {err_text}"
                    ) from exc
            else:
                # Run training with captured output so we can surface errors
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                )
                
                # Check for errors
                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    raise RuntimeError(
                        f"UDPipe training failed with exit code {result.returncode}: {error_msg}"
                    )
            
            if not model_path.exists():
                raise RuntimeError(f"UDPipe training failed: model file not created at {model_path}")
            
            return model_path
        finally:
            # Clean up temporary files if we created them
            if isinstance(train_data, (Document, list)) and train_path.exists():
                train_path.unlink(missing_ok=True)
            if dev_path and isinstance(dev_data, (Document, list)) and dev_path.exists():
                dev_path.unlink(missing_ok=True)
    
    @property
    def supports_training(self) -> bool:
        """UDPipe CLI supports training."""
        return True


MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _parse_udpipe1_model_filename(filename: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a UDPipe CLI model filename in the format 'iso-project.udpipe'.
    
    Args:
        filename: Model filename (e.g., 'luo-dho-project.udpipe' or 'en-ewt.udpipe')
    
    Returns:
        Tuple of (iso_code, project_name) or (None, None) if parsing fails
    """
    # Remove .udpipe extension
    base = filename.replace(".udpipe", "")
    
    # Pattern: iso-project where iso can contain hyphens
    # We need to find the last hyphen that separates iso from project
    # Examples:
    #   "luo-dho-project" -> iso="luo-dho", project="project"
    #   "en-ewt" -> iso="en", project="ewt"
    #   "cs-cac" -> iso="cs", project="cac"
    
    # Try to split on the last hyphen
    parts = base.rsplit("-", 1)
    if len(parts) == 2:
        iso_code, project = parts
        return iso_code, project
    
    # If no hyphen, assume the whole thing is the model name
    return None, base


def get_udpipe1_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Get UDPipe CLI model entries from local directory.
    
    Models are stored in models_dir/udpipe1/ with format 'iso-project.udpipe'.
    Metadata is enriched from UDMorph cache since models come from the same source.
    """
    cache_key = "udpipe1:local"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            # Re-check filesystem to update installed/local_path flags even when using cache
            models_dir = get_backend_models_dir("udpipe1", create=False)
            for model_name, entry in cached.items():
                expected_path = models_dir / f"{model_name}.udpipe"
                if expected_path.exists():
                    entry["installed"] = True
                    entry["local_path"] = str(expected_path)
                else:
                    entry["installed"] = False
            if verbose:
                print("[flexipipe] Using cached UDPipe CLI model list (use --refresh-cache to update).")
            return cached
    
    if verbose:
        print("[flexipipe] Scanning for UDPipe CLI models...")
    
    # Get models directory
    models_dir = get_backend_models_dir("udpipe1", create=False)
    
    # Load UDMorph metadata to enrich model entries
    udmorph_metadata: Dict[str, Dict[str, Any]] = {}
    try:
        from ..backends.udmorph import get_udmorph_model_entries
        udmorph_metadata = get_udmorph_model_entries(use_cache=True, refresh_cache=False, verbose=False)
    except Exception:
        # If UDMorph metadata is not available, continue without it
        pass
    
    prepared_models: Dict[str, Dict[str, Any]] = {}
    
    # Scan for .udpipe files
    if models_dir.exists():
        for model_file in models_dir.glob("*.udpipe"):
            model_name = model_file.stem  # filename without .udpipe
            iso_code, project = _parse_udpipe1_model_filename(model_file.name)
            
            # Try to find matching UDMorph model for metadata
            # UDMorph models use the format 'iso-project' as the key
            model_key = f"{iso_code}-{project}" if iso_code and project else model_name
            udmorph_entry = udmorph_metadata.get(model_key)
            
            # Extract language info from UDMorph metadata if available
            language_code = iso_code
            language_name = None
            features = "tokenizer, tagger, parser"
            
            if udmorph_entry:
                language_code = udmorph_entry.get(LANGUAGE_FIELD_ISO) or iso_code
                language_name = udmorph_entry.get(LANGUAGE_FIELD_NAME)
                features = udmorph_entry.get("features", features)
            
            entry = build_model_entry(
                "udpipe1",
                model_name,
                language_code=language_code,
                language_name=language_name,
                features=features,
                name=model_name,
                local_path=str(model_file),
                installed=True,
            )
            prepared_models[model_name] = entry
    
    # Only write to cache if refresh_cache is True (explicit refresh)
    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, prepared_models)
        except (OSError, PermissionError):
            # If we can't write cache, that's okay - we'll just return the entries without caching
            pass
    return prepared_models


def list_udpipe1_models_display(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> int:
    """
    List available UDPipe CLI models with formatted output.
    Prints formatted output and returns exit code (0 for success, 1 for error).
    """
    try:
        prepared_models = get_udpipe1_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=True,
        )
        
        if not prepared_models:
            print("[flexipipe] No UDPipe CLI models found in local directory.")
            print(f"[flexipipe] Models should be placed in: {get_backend_models_dir('udpipe1', create=False)}")
            return 0
        
        print(f"\nAvailable UDPipe CLI models:")
        print(f"{'Model Name':<40} {'ISO':<8} {'Language':<20} {'Features':<30}")
        print("=" * 110)
        
        sorted_items = sorted(
            prepared_models.items(),
            key=lambda x: (
                x[1].get(LANGUAGE_FIELD_ISO) or x[1].get(LANGUAGE_FIELD_NAME) or "",
                x[0]
            )
        )
        
        for model_name, entry in sorted_items:
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            features = entry.get("features", "unknown")
            print(f"{model_name:<40} {lang_iso:<8} {lang_display:<20} {features:<30}")
        
        unique_languages = set()
        for entry in prepared_models.values():
            lang_iso = entry.get(LANGUAGE_FIELD_ISO)
            if lang_iso:
                unique_languages.add(lang_iso)
        
        print(f"\nTotal: {len(prepared_models)} model(s) for {len(unique_languages)} language(s)")
        return 0
    except Exception as e:
        print(f"[flexipipe] Error listing UDPipe CLI models: {e}", file=__import__("sys").stderr)
        return 1


def _create_udpipe1_backend(
    *,
    model: str | None = None,
    model_path: str | None = None,
    model_name: str | None = None,
    udpipe_binary: str = "udpipe",
    timeout: float | None = None,
    verbose: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> UDPipeCLIBackend:
    """Instantiate the UDPipe CLI backend."""

    # Accept and drop download-specific flags that higher layers might set
    kwargs.pop("download_model", None)

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected UDPipe CLI backend arguments: {unexpected}")

    resolved_model = model or model_path or model_name
    if not resolved_model:
        raise ValueError("UDPipe CLI backend requires model path. Provide --model.")

    return UDPipeCLIBackend(
        model=resolved_model,
        udpipe_binary=udpipe_binary,
        timeout=timeout,
        verbose=verbose,
        require_model_exists=not training,
    )


BACKEND_SPEC = BackendSpec(
    name="udpipe1",
    description="UDPipe CLI - Local UDPipe command-line tool (fast training)",
    factory=_create_udpipe1_backend,
    get_model_entries=get_udpipe1_model_entries,
    list_models=list_udpipe1_models_display,
    supports_training=True,
    is_rest=False,
    url="https://github.com/ufal/udpipe",
)
