"""TreeTagger backend implementation and registry spec."""

from __future__ import annotations

import gzip
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..backend_spec import BackendSpec
from ..doc import Document, Sentence, Token
from ..language_utils import LANGUAGE_FIELD_ISO, LANGUAGE_FIELD_NAME, build_model_entry
from ..model_registry import fetch_remote_registry, get_registry_url
from ..model_storage import get_backend_models_dir, write_model_cache_entry, read_model_cache_entry
from ..neural_backend import BackendManager, NeuralResult

try:
    import requests
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "TreeTagger backend requires the 'requests' package for downloading models. "
        "Install it with: pip install requests"
    ) from exc

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
DEFAULT_TREETAGGER_REGISTRY_URL = get_registry_url("treetagger")


def _entries_from_registry_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = payload.get("sources", {}) if isinstance(payload, dict) else {}
    entries: Dict[str, Dict[str, Any]] = {}
    for source_name, models in sources.items():
        if not isinstance(models, list):
            continue
        for model_info in models:
            if not isinstance(model_info, dict):
                continue
            model_name = model_info.get("model")
            if not model_name:
                continue
            entry = build_model_entry(
                "treetagger",
                model_name,
                language_code=model_info.get("language_iso"),
                language_name=model_info.get("language_name"),
                features=model_info.get("features", "lemma,xpos"),
                components=model_info.get("components", ["tagger"]),
                preferred=model_info.get("preferred", False),
            )
            for extra_key in (
                "download_url",
                "licence",
                "parameter_file",
                "description",
                "tasks",
                "checksum",
                "checksum_type",
            ):
                if model_info.get(extra_key) is not None:
                    entry[extra_key] = model_info[extra_key]
            entry["source"] = source_name
            entries[model_name] = entry
    return entries


def get_treetagger_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Return curated TreeTagger model entries from the flexipipe-models registry."""
    registry_url = DEFAULT_TREETAGGER_REGISTRY_URL
    cache_key = f"treetagger:{registry_url}"

    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached:
            return cached

    registry_payload = fetch_remote_registry(
        backend="treetagger",
        url=registry_url,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        verbose=verbose,
    )
    entries = _entries_from_registry_payload(registry_payload)

    if entries:
        try:
            write_model_cache_entry(cache_key, entries)
        except (OSError, PermissionError):
            pass  # best effort
    return entries


def list_treetagger_models(*, use_cache: bool = True, refresh_cache: bool = False) -> int:
    """Print curated TreeTagger models."""
    try:
        entries = get_treetagger_model_entries(use_cache=use_cache, refresh_cache=refresh_cache, verbose=True)
    except Exception as exc:
        print(f"[flexipipe] Error loading TreeTagger registry: {exc}")
        return 1

    if not entries:
        print("[flexipipe] No TreeTagger models available in registry.")
        return 0

    print(f"\nAvailable TreeTagger models:")
    print(f"{'Model Name':<35} {'ISO':<6} {'Language':<20} {'Preferred':<10}")
    print("=" * 75)
    for model_name in sorted(entries.keys()):
        entry = entries[model_name]
        iso = entry.get(LANGUAGE_FIELD_ISO, "")
        lang = entry.get(LANGUAGE_FIELD_NAME, "")
        preferred = "yes" if entry.get("preferred") else ""
        print(f"{model_name:<35} {iso:<6} {lang:<20} {preferred:<10}")
    print(f"\nTotal: {len(entries)} model(s)")
    return 0


def _download_file(url: str, destination: Path, *, verbose: bool = False) -> None:
    if verbose:
        print(f"[treetagger] Downloading {url} -> {destination}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                handle.write(chunk)


def _decompress_archive(source: Path, target_dir: Path, *, expected_file: Optional[str] = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".gz" and not source.name.endswith(".tar.gz"):
        target_file = target_dir / (expected_file or source.stem)
        with gzip.open(source, "rb") as src, target_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return target_file
    if source.suffixes[-2:] == [".tar", ".gz"] or source.suffix.endswith("tgz"):
        with tarfile.open(source, "r:*") as tar:
            tar.extractall(target_dir)
    elif source.suffix == ".zip":
        with zipfile.ZipFile(source, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    else:
        target_file = target_dir / source.name
        shutil.copy(source, target_file)
        return target_file

    # Try to locate expected parameter file
    if expected_file:
        candidate = target_dir / expected_file
        if candidate.exists():
            return candidate
    # Fallback: pick first .par file
    for par in target_dir.rglob("*.par"):
        return par
    raise FileNotFoundError(
        f"Could not locate TreeTagger parameter file inside archive {source.name}. "
        "Specify 'parameter_file' in the registry entry."
    )


def _select_treetagger_entry(
    model_name: Optional[str],
    language: Optional[str],
    entries: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if model_name and model_name in entries:
        return model_name, entries[model_name]
    lang_norm = (language or "").lower()
    if lang_norm:
        for candidate_name, entry in entries.items():
            iso = (entry.get(LANGUAGE_FIELD_ISO) or "").lower()
            lang_display = (entry.get(LANGUAGE_FIELD_NAME) or "").lower()
            if lang_norm in {iso, lang_display}:
                return candidate_name, entry
    from ..model_storage import is_running_from_teitok
    teitok_msg = "" if is_running_from_teitok() else " Run 'python -m flexipipe info models --backend treetagger' for the list of models."
    raise ValueError(
        "TreeTagger backend requires --treetagger-model (matching the curated registry) "
        "or a language code (--language) that can be mapped to a curated entry." + teitok_msg
    )


def _ensure_model_available_from_entry(
    entry: Dict[str, Any],
    *,
    download_model: bool,
    verbose: bool,
) -> Path:
    model_name = entry.get("model")
    models_dir = get_backend_models_dir("treetagger")
    model_dir = models_dir / model_name
    parameter_file_name = entry.get("parameter_file") or f"{model_name}.par"
    candidate = model_dir / parameter_file_name
    if candidate.exists():
        return candidate
    if not download_model:
        raise SystemExit(
            f"TreeTagger model '{model_name}' is not available locally. "
            "Re-run with --download-model to fetch it automatically."
        )

    download_url = entry.get("download_url")
    if not download_url:
        raise SystemExit(
            f"TreeTagger model '{model_name}' does not define a download_url in the registry. "
            "Install it manually and pass --treetagger-model-path."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(download_url).name
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=f"-{suffix}")
    tmp_path = Path(tmp_path_str)
    os.close(tmp_fd)
    try:
        _download_file(download_url, tmp_path, verbose=verbose)
        parameter_path = _decompress_archive(tmp_path, model_dir, expected_file=parameter_file_name)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return parameter_path


class TreeTaggerBackend(BackendManager):
    """Backend that runs the local TreeTagger binary."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        model_path: Optional[str | Path] = None,
        language: Optional[str] = None,
        binary: Optional[str | Path] = None,
        download_model: bool = False,
        extra_args: Optional[Iterable[str]] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.binary = self._resolve_binary(binary)
        self.extra_args = list(extra_args or [])
        default_args = ["-quiet", "-token", "-lemma"]
        if not self.extra_args:
            self.extra_args = default_args
        else:
            for arg in default_args:
                if arg not in self.extra_args:
                    self.extra_args.insert(0, arg)

        self.model_metadata: Optional[Dict[str, Any]] = None
        if model_path:
            model_file = Path(model_path).expanduser().resolve()
            if not model_file.exists():
                raise ValueError(f"TreeTagger parameter file not found: {model_file}")
            self.model_path = model_file
            self.model_name = model_name or model_file.stem
            self.model_metadata = {"model": self.model_name}
        else:
            entries = get_treetagger_model_entries(
                use_cache=not download_model,
                refresh_cache=False,
                verbose=verbose,
            )
            resolved_name, entry = _select_treetagger_entry(model_name, language, entries)
            self.model_name = resolved_name
            self.model_metadata = entry
            self.model_path = _ensure_model_available_from_entry(
                entry,
                download_model=download_model,
                verbose=verbose,
            )
        if not self.model_name:
            raise ValueError("TreeTagger backend requires a valid model name.")
        else:
            self.model_name = str(self.model_name)

    @staticmethod
    def _resolve_binary(binary: Optional[str | Path]) -> Path:
        candidates = []
        if binary:
            candidates.append(str(binary))
        candidates.append("tree-tagger")
        candidates.append("tagger")
        for candidate in candidates:
            resolved = shutil.which(str(candidate))
            if resolved:
                return Path(resolved).resolve()
        raise SystemExit(
            "TreeTagger executable not found. Install TreeTagger and ensure 'tree-tagger' is on PATH, "
            "or provide --treetagger-binary."
        )

    def supports_training(self) -> bool:  # pragma: no cover - trivial
        return False

    def train(
        self,
        train_data: Any,
        output_dir: Path,
        *,
        dev_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Path:  # pragma: no cover - not supported
        raise NotImplementedError("TreeTagger backend does not support training.")

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components, use_raw_text

        flat_tokens: List[str] = []
        token_map: List[tuple[int, int]] = []
        for sent_idx, sentence in enumerate(document.sentences):
            for tok_idx, token in enumerate(sentence.tokens):
                form = token.form or token.text or ""
                cleaned = (form if form else "_").replace("\t", " ")
                flat_tokens.append(cleaned)
                token_map.append((sent_idx, tok_idx))

        input_data = "\n".join(flat_tokens) + ("\n" if flat_tokens else "")
        if not input_data.strip():
            return NeuralResult(
                document=document,
                stats={
                    "backend": "treetagger",
                    "token_count": 0,
                    "model": self.model_name or self.model_path.stem,
                },
            )

        cmd = [str(self.binary), *self.extra_args, str(self.model_path)]
        if self.verbose:
            print(f"[treetagger] Running: {' '.join(shlex.quote(part) for part in cmd)}")
        start = time.time()
        process = subprocess.run(
            cmd,
            input=input_data,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.time() - start
        if process.returncode != 0:
            raise RuntimeError(
                f"TreeTagger failed with exit code {process.returncode}: {process.stderr.strip()}"
            )

        lines = []
        for raw_line in process.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith(("reading parameters", "tagging", "finished")):
                continue
            lines.append(line)
        output_lines = lines
        if len(output_lines) != len(token_map):
            raise RuntimeError(
                f"TreeTagger returned {len(output_lines)} tagged tokens, expected {len(token_map)}. "
                "Ensure --treetagger-extra-args includes '-token'."
            )

        tagged_doc = Document(
            id=document.id,
            meta=dict(document.meta),
            attrs=dict(getattr(document, "attrs", {})) if hasattr(document, "attrs") else {},
        )
        tagged_doc.spans = {}
        if hasattr(document, "spans") and isinstance(document.spans, dict):
            tagged_doc.spans = {k: list(v) for k, v in document.spans.items()}

        # Preserve file-level attributes with model info
        file_level_attrs = tagged_doc.meta.setdefault("_file_level_attrs", {})
        model_label = self.model_name or self.model_path.stem
        file_level_attrs["treetagger_model"] = model_label
        if self.model_metadata:
            licence = self.model_metadata.get("licence")
            if licence:
                file_level_attrs["treetagger_model_licence"] = licence
            description = self.model_metadata.get("description")
            if description and "treetagger_model_description" not in file_level_attrs:
                file_level_attrs["treetagger_model_description"] = description

        line_index = 0
        for sent_idx, sentence in enumerate(document.sentences):
            new_sentence = Sentence(
                id=sentence.id,
                sent_id=sentence.sent_id,
                text=sentence.text,
                tokens=[],
                entities=list(sentence.entities),
                spans={layer: list(spans) for layer, spans in getattr(sentence, "spans", {}).items()},
                attrs=dict(getattr(sentence, "attrs", {})),
                source_id=sentence.source_id,
                char_start=sentence.char_start,
                char_end=sentence.char_end,
                byte_start=sentence.byte_start,
                byte_end=sentence.byte_end,
            )
            for token_idx, original_token in enumerate(sentence.tokens):
                line = output_lines[line_index]
                line_index += 1
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split()
                form = parts[0] if parts else original_token.form
                xpos = parts[1] if len(parts) >= 2 else original_token.xpos or "_"
                lemma = parts[2] if len(parts) >= 3 else original_token.lemma or "_"
                if lemma in {"<unknown>", "<unknown>.", "UNKNOWN"}:
                    lemma = original_token.lemma or original_token.form

                new_token = Token(
                    id=original_token.id,
                    form=original_token.form or form,
                    lemma=lemma or original_token.lemma,
                    upos=original_token.upos or "_",
                    xpos=xpos,
                    feats=original_token.feats,
                    is_mwt=original_token.is_mwt,
                    mwt_start=original_token.mwt_start,
                    mwt_end=original_token.mwt_end,
                    parts=list(original_token.parts),
                    subtokens=list(original_token.subtokens),
                    source=original_token.source,
                    source_id=original_token.source_id,
                    head=original_token.head,
                    deprel=original_token.deprel,
                    deps=original_token.deps,
                    misc=original_token.misc,
                    char_start=original_token.char_start,
                    char_end=original_token.char_end,
                    byte_start=original_token.byte_start,
                    byte_end=original_token.byte_end,
                    attrs=dict(getattr(original_token, "attrs", {})),
                )
                new_token.space_after = original_token.space_after
                new_sentence.tokens.append(new_token)
            tagged_doc.sentences.append(new_sentence)

        stats = {
            "backend": "treetagger",
            "token_count": len(flat_tokens),
            "elapsed_seconds": elapsed,
            "model": model_label,
        }
        return NeuralResult(document=tagged_doc, stats=stats)


def _create_treetagger_backend(
    *,
    model_name: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    language: Optional[str] = None,
    binary: Optional[str | Path] = None,
    download_model: bool = False,
    treetagger_extra_args: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> TreeTaggerBackend:
    del kwargs
    resolved_model_name = model_name
    return TreeTaggerBackend(
        model_name=resolved_model_name,
        model_path=model_path,
        language=language,
        binary=binary,
        download_model=download_model,
        extra_args=treetagger_extra_args,
        verbose=verbose,
    )


BACKEND_SPEC = BackendSpec(
    name="treetagger",
    description="TreeTagger POS and lemmatization backend (local binary).",
    factory=_create_treetagger_backend,
    get_model_entries=get_treetagger_model_entries,
    list_models=list_treetagger_models,
    supports_training=False,
    is_rest=False,
    url="https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/",
    model_registry_url=DEFAULT_TREETAGGER_REGISTRY_URL,
    install_instructions="treetagger requires the TreeTagger CLI binary to be installed separately (see https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)",
)


