"""Backend and registry spec for ClassLA."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..backend_spec import BackendSpec
from ..doc import Document, Entity, Sentence, Token
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_registry import fetch_remote_registry, get_registry_url
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _get_fallback_classla_models() -> Dict[tuple[str, str], Dict[str, Any]]:
    """Get fallback hardcoded ClassLA models (used when registry is unavailable)."""
    return {
        ("hr", "standard"): {"package": "set", "name": "Croatian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("hr", "nonstandard"): {"package": "set", "name": "Croatian (nonstandard)", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("sr", "standard"): {"package": "set", "name": "Serbian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("sr", "nonstandard"): {"package": "set", "name": "Serbian (nonstandard)", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("bg", "standard"): {"package": "btb", "name": "Bulgarian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("mk", "standard"): {"package": "mk", "name": "Macedonian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats"]},  # No depparse
        ("sl", "standard"): {"package": "ssj", "name": "Slovenian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse"]},  # Has depparse
    }


def get_classla_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    del kwargs
    cache_key = "classla"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached:
            fixed = False
            for model_key, entry in cached.items():
                if isinstance(entry, dict) and not entry.get("language_iso") and "-" in model_key:
                    entry["language_iso"] = model_key.split("-")[0].lower()
                    fixed = True
            if fixed and refresh_cache:
                try:
                    write_model_cache_entry(cache_key, cached)
                except (OSError, PermissionError):
                    pass
            if cache_entries_standardized(cached):
                return cached

    # Try to fetch from remote registry first
    known_models: Dict[tuple[str, str], Dict[str, Any]] = {}
    try:
        registry = fetch_remote_registry(
            backend="classla",
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=verbose,
        )
        if registry:
            # Extract models from registry structure
            sources = registry.get("sources", {})
            for source_type in ["official", "flexipipe", "community"]:
                if source_type in sources:
                    for model_entry in sources[source_type]:
                        model_name = model_entry.get("model")
                        if model_name and "-" in model_name:
                            # Parse model name like "hr-standard" or "mk-standard"
                            parts = model_name.split("-", 1)
                            if len(parts) == 2:
                                lang_code, variant = parts
                                if variant in ("standard", "nonstandard"):
                                    known_models[(lang_code, variant)] = {
                                        "package": model_entry.get("package", ""),
                                        "name": model_entry.get("language_name") or model_entry.get("name", lang_code.upper()),
                                        "default_features": model_entry.get("features", "").split(", ") if model_entry.get("features") else [],
                                    }
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] ClassLA registry unavailable ({exc}), using fallback models.")
    
    # Fall back to hardcoded models if registry is empty or unavailable
    if not known_models:
        known_models = _get_fallback_classla_models()

    result: Dict[str, Dict[str, str]] = {}
    installed_models: Dict[str, Dict[str, str]] = {}
    # Track available processors for each model
    model_processors: Dict[str, Set[str]] = {}
    classla_resources = get_backend_models_dir("classla", create=False)
    if classla_resources.exists():
        for lang_dir in classla_resources.iterdir():
            if not lang_dir.is_dir():
                continue
            lang_code = lang_dir.name
            for processor_dir in lang_dir.iterdir():
                if not processor_dir.is_dir():
                    continue
                processor_name = processor_dir.name
                # Check for .pt files (including .pt.zip files)
                has_model_file = False
                model_file = None
                # First check for regular .pt files
                for f in processor_dir.glob("*.pt"):
                    if not f.name.endswith(".zip"):
                        model_file = f
                        has_model_file = True
                        break
                # If no regular .pt file, check for .pt.zip
                if not model_file:
                    for f in processor_dir.glob("*.pt.zip"):
                        model_file = f
                        has_model_file = True
                        break
                
                if has_model_file and model_file:
                    package = model_file.stem.replace(".zip", "")
                    variant = "nonstandard" if "nonstandard" in processor_dir.parts else "standard"
                    model_key = f"{lang_code}-{variant}"
                    installed_models.setdefault(
                        model_key,
                        {"lang": lang_code, "package": package, "variant": variant},
                    )
                    # Track which processors are available for this model
                    if model_key not in model_processors:
                        model_processors[model_key] = set()
                    model_processors[model_key].add(processor_name)

    # Check which models are actually installed
    from ..model_storage import is_model_installed
    
    for (lang_code, variant), model_info in known_models.items():
        package = model_info["package"]
        model_key = f"{lang_code}-{variant}"
        model_entry = installed_models.get(
            model_key,
            {"lang": lang_code, "package": package, "variant": variant},
        )
        lang_name = model_info.get("name", lang_code.upper())
        entry = build_model_entry(
            backend="classla",
            model_id=model_key,
            model_name=model_key,
            language_code=model_entry.get("lang", lang_code),
            language_name=lang_name,
            package=model_entry.get("package", package),
            description=f"ClassLA model for {lang_name} ({variant})",
        )
        entry["language_iso"] = entry.get("language_iso") or lang_code.lower()
        entry["package"] = model_entry.get("package", package)
        entry["variant"] = model_entry.get("variant", variant)
        
        # Set status based on whether model is actually installed
        try:
            if is_model_installed("classla", model_key):
                entry["status"] = "installed"
            else:
                entry["status"] = "available"
        except Exception:
            # If check fails, default to available
            entry["status"] = "available"
        
        # Set features based on available processors (if installed) or default features (if not installed)
        available_processors = model_processors.get(model_key, set())
        features_list = []
        
        if available_processors:
            # Model is installed - detect features from actual processors
            # Map processor names to feature names
            processor_to_feature = {
                "tokenize": "tokenization",
                "pos": "upos",
                "lemma": "lemma",
                "depparse": "depparse",
                "ner": "ner",
            }
            for proc, feat in processor_to_feature.items():
                if proc in available_processors:
                    features_list.append(feat)
            # Also include xpos and feats if pos is available (POS tagger provides these)
            if "pos" in available_processors:
                if "xpos" not in features_list:
                    features_list.append("xpos")
                if "feats" not in features_list:
                    features_list.append("feats")
        else:
            # Model is not installed - use default features from known_models
            default_features = model_info.get("default_features", [])
            features_list = default_features.copy()
        
        if features_list:
            entry["features"] = ", ".join(features_list)
        
        result[model_key] = entry

    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, result)
        except (OSError, PermissionError):
            pass
    return result


def _classla_doc_to_document(
    classla_doc,
    original_doc: Optional[Document] = None,
) -> Document:
    doc = Document(id="")
    if original_doc:
        doc.id = original_doc.id
        doc.meta = original_doc.meta.copy()
        doc.attrs = original_doc.attrs.copy()

    for stanza_sent in classla_doc.sentences:
        sent_id = getattr(stanza_sent, "sent_id", None) or ""
        sentence = Sentence(id=sent_id, sent_id=sent_id, text=stanza_sent.text, tokens=[])

        if hasattr(stanza_sent, "ents") and stanza_sent.ents:
            for ent in stanza_sent.ents:
                try:
                    token_start = getattr(ent, "start", None)
                    token_end = getattr(ent, "end", None)
                    if hasattr(ent, "tokens") and ent.tokens:
                        token_start = ent.tokens[0].id
                        token_end = ent.tokens[-1].id
                    if token_start is None or token_end is None:
                        continue
                    start_idx = int(str(token_start).split("-")[0])
                    end_idx = int(str(token_end).split("-")[-1])
                    entity = Entity(
                        start=start_idx + 1,
                        end=end_idx,
                        label=getattr(ent, "type", ""),
                        text=getattr(ent, "text", ""),
                    )
                    sentence.entities.append(entity)
                except Exception:
                    continue

        for token_idx, stanza_token in enumerate(stanza_sent.tokens):
            def _to_int_id(token_id):
                if token_id is None:
                    return token_idx + 1
                token_str = str(token_id)
                if "-" in token_str:
                    token_str = token_str.split("-")[0]
                try:
                    return int(token_str)
                except (ValueError, TypeError):
                    return token_idx + 1

            if hasattr(stanza_token, "words") and len(stanza_token.words) > 1:
                subtokens = []
                for word in stanza_token.words:
                    subtokens.append(
                        Token(
                            id=_to_int_id(word.id),
                            form=word.text,
                            lemma=word.lemma or "",
                            upos=word.upos or "",
                            xpos=word.xpos or "",
                            feats=word.feats or "",
                            head=word.head if word.head else 0,
                            deprel=word.deprel or "",
                            space_after=("SpaceAfter=No" not in (word.misc or "")) if word.misc else True,
                        )
                    )
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=stanza_token.words[0].lemma if stanza_token.words else "",
                    upos=stanza_token.words[0].upos if stanza_token.words else "",
                    xpos=stanza_token.words[0].xpos if stanza_token.words else "",
                    feats=stanza_token.words[0].feats if stanza_token.words else "",
                    head=stanza_token.words[0].head if stanza_token.words and stanza_token.words[0].head else 0,
                    deprel=stanza_token.words[0].deprel if stanza_token.words else "",
                    is_mwt=True,
                    subtokens=subtokens,
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if stanza_token.misc else True,
                )
                token.parts = [st.form for st in subtokens]
            else:
                word = stanza_token.words[0] if hasattr(stanza_token, "words") and stanza_token.words else stanza_token
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=getattr(word, "lemma", "") or "",
                    upos=getattr(word, "upos", "") or "",
                    xpos=getattr(word, "xpos", "") or "",
                    feats=getattr(word, "feats", "") or "",
                    head=getattr(word, "head", 0) or 0,
                    deprel=getattr(word, "deprel", "") or "",
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if getattr(stanza_token, "misc", None) else True,
                )
            sentence.tokens.append(token)

        if sentence.tokens:
            sentence.tokens[-1].space_after = None
        doc.sentences.append(sentence)
    return doc


class ClassLABackend(BackendManager):
    """ClassLA-based neural backend."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        package: Optional[str] = None,
        processors: Optional[str] = None,
        use_gpu: bool = False,
        download_model: bool = False,
        verbose: bool = False,
        type: Optional[str] = None,
    ):
        from ..model_storage import setup_backend_environment

        setup_backend_environment("classla")

        if not verbose:
            logging.getLogger("classla").setLevel(logging.WARNING)

        try:
            import classla
        except ImportError as exc:
            raise ImportError("ClassLA backend requires the 'classla' package. Install it with: pip install classla") from exc
        try:
            from classla.pipeline.core import ResourcesFileNotFound  # type: ignore
        except ImportError:
            ResourcesFileNotFound = Exception

        self.classla = classla
        self._resources_error = ResourcesFileNotFound
        self._language = language or (model_name.split("-")[0] if model_name and "-" in model_name else model_name) or "hr"
        default_package = {"hr": "set", "sr": "set", "bg": "btb", "mk": "mk", "sl": "ssj"}
        self._package = package or default_package.get(self._language)
        
        # Determine default processors based on model capabilities
        default_processors = ["tokenize", "pos", "lemma"]
        # Check model registry to see what features are available
        if processors is None and model_name:
            try:
                from .classla import get_classla_model_entries  # circular-safe import (self-file)
                entries = get_classla_model_entries(use_cache=True, refresh_cache=False)
                entry = entries.get(model_name)
                if entry:
                    features_str = entry.get("features", "")
                    if features_str:
                        features = [f.strip() for f in features_str.split(",")]
                        # Map features to processors
                        feature_to_processor = {
                            "tokenization": "tokenize",
                            "upos": "pos",
                            "xpos": "pos",  # xpos comes from pos processor
                            "feats": "pos",  # feats comes from pos processor
                            "lemma": "lemma",
                            "depparse": "depparse",
                            "ner": "ner",
                        }
                        available_processors = []
                        # Always include tokenize (it's always available for ClassLA models)
                        available_processors.append("tokenize")
                        for feat in features:
                            proc = feature_to_processor.get(feat)
                            if proc and proc not in available_processors:
                                available_processors.append(proc)
                        if available_processors:
                            default_processors = available_processors
            except Exception:
                # If registry lookup fails, use default
                pass
        
        if processors:
            self._processors = processors
        else:
            self._processors = ",".join(default_processors)
        self._use_gpu = use_gpu
        self._download = download_model
        self._verbose = verbose
        if model_name and "-" in model_name and not type:
            parts = model_name.split("-", 1)
            if len(parts) == 2 and parts[1] in ("standard", "nonstandard"):
                self._type = parts[1]
            else:
                self._type = type or "standard"
        else:
            self._type = type or "standard"
        self._pipelines: Dict[bool, classla.Pipeline] = {}

    def _build_pipeline(self, pretokenized: bool):
        if not self._verbose:
            classla_logger = logging.getLogger("classla")
            classla_logger.setLevel(logging.WARNING)
            for handler in classla_logger.handlers:
                handler.setLevel(logging.WARNING)
            classla_logger.propagate = False

        from ..model_storage import get_backend_models_dir

        classla_dir = get_backend_models_dir("classla", create=False)

        config: Dict[str, Union[str, bool]] = {
            "lang": self._language,
            "processors": self._processors,
            "use_gpu": self._use_gpu,
            "type": self._type,
            "dir": str(classla_dir),
        }
        if pretokenized:
            config["tokenize_pretokenized"] = True

        try:
            return self.classla.Pipeline(**config)
        except (ValueError, TypeError) as e:
            # Check if this is a JSON parsing error from ClassLA's resource loading
            error_str = str(e)
            if "Expecting value" in error_str or "JSONDecodeError" in error_str:
                # Check if resources.json is empty or corrupted and try to fix it
                resources_file = classla_dir / "resources.json"
                if resources_file.exists():
                    try:
                        import json
                        with open(resources_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if not content:
                                # Empty file - delete it and try to rebuild
                                if self._verbose:
                                    print(f"[flexipipe] Detected empty resources.json, attempting to rebuild...", file=sys.stderr)
                                resources_file.unlink()
                            else:
                                # Try to parse - if it fails, it's corrupted
                                json.loads(content)
                                # If we get here, JSON is valid but ClassLA still failed
                                # This might be a different issue
                                raise RuntimeError(
                                    f"ClassLA failed to load resources despite valid JSON file. "
                                    f"This may indicate a ClassLA version mismatch or corrupted model files. "
                                    f"Try re-downloading: classla.download('{self._language}', type='{self._type}')"
                                ) from e
                    except (json.JSONDecodeError, ValueError):
                        # Corrupted JSON - delete it and try to rebuild
                        if self._verbose:
                            print(f"[flexipipe] Detected corrupted resources.json, attempting to rebuild...", file=sys.stderr)
                        resources_file.unlink()
                
                # Always try to rebuild when we detect corrupted/empty resources.json
                # This is a recovery mechanism, not a download request
                if self._verbose:
                    print(f"[flexipipe] Rebuilding ClassLA resources for {self._language} (type: {self._type})...", file=sys.stderr)
                try:
                    self.classla.download(self._language, type=self._type, verbose=self._verbose)
                    # Retry pipeline creation after download
                    return self.classla.Pipeline(**config)
                except Exception as download_error:
                    raise RuntimeError(
                        f"Failed to rebuild ClassLA resources: {download_error}. "
                        f"Try manually: classla.download('{self._language}', type='{self._type}')"
                    ) from download_error
            error_str = str(e)
            if "not enough values to unpack" in error_str or "expected 2" in error_str:
                config_minimal = {
                    "lang": self._language,
                    "processors": self._processors,
                    "use_gpu": self._use_gpu,
                    "type": self._type,
                }
                pipeline = self.classla.Pipeline(**config_minimal)
                if pretokenized and hasattr(pipeline, "tokenize_pretokenized"):
                    pipeline.tokenize_pretokenized = True
                return pipeline
            raise
        except Exception as e:
            # Catch any Exception (including ResourcesFileNotFound) to check if it's a resources error
            error_str = str(e)
            is_resources_error = (
                "Resources file not found" in error_str or
                "resources" in error_str.lower() or
                isinstance(e, self._resources_error)
            )
            
            # Check if this is a missing pretrain/vector file error
            is_missing_pretrain = (
                "vector file is not provided" in error_str.lower() or
                ("pretrain" in error_str.lower() and ("not found" in error_str.lower() or "missing" in error_str.lower()))
            )
            
            if is_missing_pretrain:
                # If download is explicitly requested, try to download the complete model
                if self._download:
                    if self._verbose:
                        print(f"[flexipipe] Missing pretrain/vector file detected. Downloading complete ClassLA model for {self._language} (type: {self._type})...", flush=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    try:
                        # Download with all required processors, including pretrain
                        processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                        # Always include pretrain if pos is requested (POS tagger requires pretrain vectors)
                        if "pos" in processors_list and "pretrain" not in processors_list:
                            processors_list.append("pretrain")
                        processors_dict = {proc: True for proc in processors_list}
                        self.classla.download(
                            self._language, 
                            type=self._type, 
                            processors=processors_dict,
                            verbose=self._verbose
                        )
                        if self._verbose:
                            sys.stdout.flush()
                            sys.stderr.flush()
                            print(f"[flexipipe] ClassLA model download completed", flush=True)
                        # Retry pipeline creation after download
                        return self.classla.Pipeline(**config)
                    except Exception as download_error:
                        raise RuntimeError(
                            f"Failed to download ClassLA model: {download_error}. "
                            f"Try manually: classla.download('{self._language}', type='{self._type}')"
                        ) from download_error
                # If download is not requested, raise error with instructions
                raise RuntimeError(
                    f"ClassLA model for language '{self._language}' (type: {self._type}) is incomplete. "
                    f"The pretrain/vector file is missing, which is required for the POS tagger. "
                    f"Use --download-model to download the complete model, or run: "
                    f"classla.download('{self._language}', type='{self._type}')"
                ) from e
            
            if is_resources_error:
                # Check if resources.json is missing or empty - if so, try to rebuild automatically
                resources_file = classla_dir / "resources.json"
                should_auto_rebuild = False
                
                if not resources_file.exists():
                    should_auto_rebuild = True
                    if self._verbose:
                        print(f"[flexipipe] Resources file not found, attempting to rebuild...", file=sys.stderr)
                elif resources_file.exists():
                    # Check if file is empty
                    try:
                        if resources_file.stat().st_size == 0:
                            should_auto_rebuild = True
                            if self._verbose:
                                print(f"[flexipipe] Resources file is empty, attempting to rebuild...", file=sys.stderr)
                            resources_file.unlink()
                    except OSError:
                        pass
                
                # Auto-rebuild only for corrupted/empty resources.json (recovery mechanism)
                # For missing models, require explicit --download-model flag
                if should_auto_rebuild:
                    if self._verbose:
                        print(f"[flexipipe] Rebuilding ClassLA resources for {self._language} (type: {self._type})...", file=sys.stderr)
                    try:
                        # Only download the processors we actually need, not all available processors
                        # This prevents downloading unnecessary models (like Ukrainian lemmatizer)
                        # Processors should be a dict mapping processor names to True/False or package names
                        processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                        # Always include pretrain if pos is requested (POS tagger requires pretrain vectors)
                        if "pos" in processors_list and "pretrain" not in processors_list:
                            processors_list.append("pretrain")
                        processors_dict = {proc: True for proc in processors_list}
                        self.classla.download(
                            self._language, 
                            type=self._type, 
                            processors=processors_dict,
                            verbose=self._verbose
                        )
                        # Retry pipeline creation after download
                        return self.classla.Pipeline(**config)
                    except Exception as download_error:
                        raise RuntimeError(
                            f"Failed to rebuild ClassLA resources: {download_error}. "
                            f"Try manually: classla.download('{self._language}', type='{self._type}')"
                        ) from download_error
                # If resources.json is fine but model files are missing, require explicit download
                raise RuntimeError(
                    f"ClassLA model not found for language '{self._language}' "
                    f"(package: {self._package}, type: {self._type}). "
                    f"Use --download-model to install, or run: classla.download('{self._language}', type='{self._type}')"
                ) from e
            
            # Check if it's a file/OS error for depparse fallback
            error_str_lower = error_str.lower()
            if "depparse" in error_str_lower or "parser" in error_str_lower or "no such file" in error_str_lower:
                processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                if "depparse" in processors_list:
                    processors_list.remove("depparse")
                    config_fallback = dict(config)
                    config_fallback["processors"] = ",".join(processors_list)
                    return self.classla.Pipeline(**config_fallback)
            
            # Not handled - re-raise
            raise

    def _get_pipeline(self, pretokenized: bool):
        if pretokenized not in self._pipelines:
            self._pipelines[pretokenized] = self._build_pipeline(pretokenized)
        return self._pipelines[pretokenized]

    def _run_raw(self, document: Document):
        pipeline = self._get_pipeline(pretokenized=False)
        text = "\n".join(sent.text for sent in document.sentences if sent.text)
        return pipeline(text)

    def _run_pretokenized(self, document: Document):
        pipeline = self._get_pipeline(pretokenized=True)
        pretokenized = [[token.form for token in sentence.tokens] for sentence in document.sentences]
        return pipeline(pretokenized)

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components

        start_time = time.time()
        if use_raw_text or not document.sentences:
            classla_doc = self._run_raw(document)
            result_doc = _classla_doc_to_document(classla_doc)
        else:
            classla_doc = self._run_pretokenized(document)
            result_doc = _classla_doc_to_document(classla_doc, original_doc=document)

        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0.0,
        }
        return NeuralResult(document=result_doc, stats=stats)

    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs,
    ) -> Path:
        raise NotImplementedError(
            "ClassLA training is not yet integrated into flexipipe. "
            "Use the official classla-train workflow to build models."
        )

    def supports_training(self) -> bool:
        return False


def _list_classla_models(*args, **kwargs) -> int:
    entries = get_classla_model_entries(*args, **kwargs)
    print(f"\nAvailable ClassLA models:")
    print(f"{'Model ID':<20} {'ISO':<6} {'Language':<20} {'Package':<15} {'Variant':<15} {'Status':<25}")
    print("=" * 101)
    for key in sorted(entries.keys()):
        model_info = entries[key]
        model_id = key
        iso = (model_info.get("language_iso") or "")[:6]
        lang = model_info.get("language_name", "")
        pkg = model_info.get("package", "")
        variant = model_info.get("variant", "standard")
        status = model_info.get("status", "Available")
        print(f"{model_id:<20} {iso:<6} {lang:<20} {pkg:<15} {variant:<15} {status:<25}")
    print(f"\nTotal: {len(entries)} model(s)")
    print("\nClassLA models are downloaded automatically on first use")
    print("Features: tokenization, lemma, upos, xpos, feats, depparse, NER")
    print("\nUsage: --backend classla --model <Model ID>")
    return 0


def _create_classla_backend(
    *,
    model_name: str | None = None,
    language: str | None = None,
    package: str | None = None,
    processors: str | None = None,
    use_gpu: bool = False,
    download_model: bool = False,
    verbose: bool = False,
    type: str | None = None,
    training: bool = False,
    **kwargs: Any,
) -> ClassLABackend:
    _ = training
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected ClassLA backend arguments: {unexpected}")
    return ClassLABackend(
        model_name=model_name,
        language=language,
        package=package,
        processors=processors,
        use_gpu=use_gpu,
        download_model=download_model,
        verbose=verbose,
        type=type,
    )


BACKEND_SPEC = BackendSpec(
    name="classla",
    description="ClassLA - Fork of Stanza for South Slavic languages",
    factory=_create_classla_backend,
    get_model_entries=get_classla_model_entries,
    list_models=_list_classla_models,
    supports_training=False,
    is_rest=False,
    url="https://github.com/clarinsi/classla",
)

