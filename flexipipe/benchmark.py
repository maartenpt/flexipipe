from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .backend_registry import (
    create_backend,
    get_backend_info,
    get_model_entries,
    get_model_entries_dict,
    list_backends,
)
from .language_utils import language_matches_entry, resolve_language_query, build_model_entry
from .check import evaluate_model
from .backends.flexitag import resolve_flexitag_model_path
from .model_storage import get_backend_models_dir, get_flexipipe_models_dir

# Default paths under the flexipipe models directory
# Don't create directory at module level - only create when actually needed
BENCHMARK_ROOT = get_flexipipe_models_dir(create=False) / "benchmark"
DEFAULT_RESULTS_PATH = BENCHMARK_ROOT / "results.json"
DEFAULT_TREEBANK_CATALOG = BENCHMARK_ROOT / "treebanks.json"
DEFAULT_MODEL_CATALOG = BENCHMARK_ROOT / "models.json"
DISABLED_MODELS_PATH = BENCHMARK_ROOT / "disabled_models.json"


def load_disabled_models() -> set[tuple[str, str]]:
    """Load disabled models from file. Returns set of (backend, model) tuples."""
    if not DISABLED_MODELS_PATH.exists():
        return set()
    try:
        with DISABLED_MODELS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return {tuple(item) for item in data if isinstance(item, list) and len(item) >= 2}
        return set()
    except (json.JSONDecodeError, OSError):
        return set()


def save_disabled_models(disabled: set[tuple[str, str]]) -> None:
    """Save disabled models to file."""
    DISABLED_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISABLED_MODELS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(sorted(list(disabled)), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def mark_model_disabled(backend: str, model: str) -> None:
    """Mark a model as disabled."""
    disabled = load_disabled_models()
    disabled.add((backend.lower(), model))
    save_disabled_models(disabled)


def is_model_disabled(backend: str, model: str) -> bool:
    """Check if a model is disabled."""
    disabled = load_disabled_models()
    return (backend.lower(), model) in disabled


@dataclass
class BenchmarkJob:
    language: str
    backend: str
    model: Optional[str]
    treebank: Path
    tasks: tuple[str, ...]
    mode: str = "auto"  # Evaluation mode: auto, raw, tokenized, split


class BenchmarkStorage:
    """Simple JSON-backed storage for benchmark results."""

    def __init__(self, path: Path):
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []

    def save(self, rows: list[dict]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def append(self, row: dict) -> None:
        """Append a result row, overwriting any existing row with the same key (language/backend/model/treebank/mode)."""
        rows = self.load()
        # Build key for this row
        row_key = (
            row.get("language", ""),
            row.get("backend", ""),
            row.get("model", ""),
            str(row.get("treebank", "")),
            row.get("mode", "auto"),  # Include mode in key
        )
        # Remove any existing row with the same key
        rows = [
            r for r in rows
            if (
                r.get("language", ""),
                r.get("backend", ""),
                r.get("model", ""),
                str(r.get("treebank", "")),
                r.get("mode", "auto"),
            ) != row_key
        ]
        # Append the new row
        rows.append(row)
        self.save(rows)


class BenchmarkRunner:
    """Coordinates benchmark execution for backend/model pairs."""

    def __init__(
        self,
        storage: BenchmarkStorage,
        *,
        treebank_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        verbose: bool = False,
        dry_run: bool = False,
        treebank_catalog: Optional[list[dict]] = None,
        download_models: bool = False,
    ):
        self.storage = storage
        self.treebank_root = treebank_root.expanduser() if treebank_root else None
        BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root or (BENCHMARK_ROOT / "runs")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.dry_run = dry_run
        self.download_models = download_models
        self.treebank_catalog = treebank_catalog or self._build_treebank_catalog()

    # ------------------------------------------------------------------ discovery
    def discover_languages(self) -> list[str]:
        """Infer available languages from treebank root."""
        if self.treebank_catalog:
            return sorted({entry["language"] for entry in self.treebank_catalog})
        if not self.treebank_root or not self.treebank_root.exists():
            return []
        langs: set[str] = set()
        for path in self.treebank_root.rglob("*-ud-*.conllu"):
            name = path.name
            if "-" in name:
                prefix = name.split("-")[0]
                langs.add(prefix.lower())
        return sorted(langs)

    def discover_treebanks(self, language: str, limit: Optional[int] = None) -> list[Path]:
        """Return CoNLL-U paths matching the given language code."""
        if self.treebank_catalog:
            filtered = [
                Path(entry["path"])
                for entry in self.treebank_catalog
                if entry["language"] == language.lower()
            ]
            if limit:
                filtered = filtered[:limit]
            return filtered
        if not self.treebank_root or not self.treebank_root.exists():
            return []
        matches: list[Path] = []
        lowered = language.lower()
        for path in sorted(self.treebank_root.rglob("*-ud-*.conllu")):
            if lowered in path.name.lower():
                matches.append(path)
                if limit and len(matches) >= limit:
                    break
        return matches

    def discover_models_for_language(self, backend: str, language: str) -> list[str]:
        """Return model names for a backend that match the requested language."""
        # Check if this is a REST backend - REST backends should always try to fetch from service
        backend_info = get_backend_info(backend)
        is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
        
        # For REST backends, try fetching from service if cache is empty or no models found
        use_cache = True
        refresh_cache = False
        
        try:
            entries = get_model_entries(
                backend,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                verbose=False,
            )
            
            # If no entries found and it's a REST backend, try refreshing
            if is_rest_backend and (not entries or len(entries) == 0):
                if self.verbose:
                    print(f"[benchmark] No cached models for REST backend {backend}, fetching from service...")
                entries = get_model_entries(
                    backend,
                    use_cache=False,  # Force fetch from service
                    refresh_cache=True,
                    verbose=self.verbose,
                )
        except Exception as exc:
            if self.verbose:
                print(f"[benchmark] Error fetching models for {backend}: {exc}")
            return []
        
        query = resolve_language_query(language)
        models: list[str] = []
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if language_matches_entry(entry, query, allow_fuzzy=True):
                model_name = entry.get("model")
                if model_name:
                    models.append(model_name)
        # Preserve order but remove duplicates
        seen: set[str] = set()
        unique: list[str] = []
        for model in models:
            if model not in seen:
                unique.append(model)
                seen.add(model)
        return unique

    def _build_treebank_catalog(self) -> list[dict]:
        if not self.treebank_root or not self.treebank_root.exists():
            return []
        catalog: list[dict] = []
        root = self.treebank_root
        for path in sorted(root.rglob("*-ud-test.conllu")):
            file_name = path.name
            treebank_id = file_name.split("-ud-")[0]
            lang = treebank_id.split("_")[0].lower()
            catalog.append(
                {
                    "id": treebank_id,
                    "language": lang,
                    "path": str(path),
                }
            )
        return catalog
    
    def build_model_catalog(self) -> list[dict]:
        catalog: list[dict] = []
        from types import SimpleNamespace
        default_args = SimpleNamespace(refresh_cache=False, verbose=False)
        for backend_name in get_model_entries_dict.keys():
            entry_getter = get_model_entries_dict[backend_name]
            try:
                entries = entry_getter(
                    default_args,
                    use_cache=True,
                    refresh_cache=False,
                    verbose=False,
                )
            except TypeError:
                try:
                    entries = entry_getter(
                        use_cache=True,
                        refresh_cache=False,
                        verbose=False,
                    )
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(entries, tuple):
                entries = entries[0]
            if not isinstance(entries, dict):
                continue
            for entry_key, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                model_name = entry.get("model")
                if not model_name:
                    continue
                # Skip disabled models
                if is_model_disabled(backend_name, model_name):
                    continue
                language_iso = (entry.get("language_iso") or "").lower()
                language_name = entry.get("language_name")
                model_entry = {
                    "backend": backend_name,
                    "model": model_name,
                    "language": language_iso,
                    "language_name": language_name,
                    "description": entry.get("description"),
                    "entry_id": entry_key,
                }
                catalog.append(model_entry)
        return catalog

    # ------------------------------------------------------------------- execution
    def run_jobs(self, jobs: Iterable[BenchmarkJob]) -> None:
        for job in jobs:
            self._execute_job(job)

    def _ensure_conllu(self, treebank_path: Path) -> Path:
        """Ensure treebank is in CoNLL-U format, converting from TEI if needed."""
        if treebank_path.suffix.lower() in (".conllu", ".conll"):
            return treebank_path
        
        # Check if it's TEI format
        try:
            with treebank_path.open("r", encoding="utf-8", errors="ignore") as handle:
                sample = handle.read(4096)
            if "<tok" in sample or "<TEI" in sample or sample.lstrip().startswith("<TEI"):
                # Convert TEI to CoNLL-U
                from .teitok import load_teitok
                from .conllu import document_to_conllu
                
                if self.verbose:
                    print(f"[benchmark] Converting TEI to CoNLL-U: {treebank_path.name}")
                
                # Load TEI document
                doc = load_teitok(str(treebank_path))
                
                # Create temporary CoNLL-U file
                conllu_path = treebank_path.with_suffix(".conllu")
                if not conllu_path.exists():
                    with conllu_path.open("w", encoding="utf-8") as handle:
                        handle.write(document_to_conllu(doc))
                    if self.verbose:
                        print(f"[benchmark] Created CoNLL-U file: {conllu_path.name}")
                
                return conllu_path
        except Exception as exc:
            if self.verbose:
                print(f"[benchmark] Warning: Could not convert {treebank_path.name} to CoNLL-U: {exc}")
        
        # If not TEI or conversion failed, return original path
        return treebank_path

    def _execute_job(self, job: BenchmarkJob) -> None:
        print(
            f"[benchmark] Running {job.language} | {job.backend} | "
            f"{job.model or '<auto>'} | {Path(job.treebank).name}"
        )
        if self.dry_run:
            return

        # Ensure treebank is in CoNLL-U format
        treebank_path = self._ensure_conllu(job.treebank)

        output_dir = (
            self.output_root
            / job.language
            / job.backend
            / (job.model or "default")
            / treebank_path.stem
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        neural_backend = None
        model_path: Optional[Path] = None

        try:
            if job.backend == "flexitag":
                try:
                    model_path = resolve_flexitag_model_path(
                        model_name=job.model,
                        params_path=None,
                    )
                except (ValueError, FileNotFoundError) as exc:
                    error_msg = str(exc)
                    print(
                        f"[benchmark] ⚠ Skipping {job.backend}/{job.model or 'default'}: {error_msg}",
                        file=sys.stderr
                    )
                    return  # Skip this job
            else:
                backend_info = get_backend_info(job.backend)
                if backend_info is None:
                    raise ValueError(f"Unknown backend '{job.backend}'")
                create_kwargs: dict = {}
                if job.model:
                    # REST backends (UDPipe, UDMorph, NameTag) use "model" instead of "model_name"
                    if job.backend in ("udpipe", "udmorph", "nametag"):
                        create_kwargs["model"] = job.model
                    else:
                        create_kwargs["model_name"] = job.model
                if job.language:
                    create_kwargs.setdefault("language", job.language)
                # Add download_model flag if requested
                # Note: Downloads happen during backend creation, BEFORE timing starts
                # This ensures download time is not included in benchmark speed measurements
                if self.download_models:
                    create_kwargs["download_model"] = True
                # Provide default endpoint URLs for REST backends
                if job.backend == "udpipe":
                    create_kwargs.setdefault(
                        "endpoint_url",
                        "https://lindat.mff.cuni.cz/services/udpipe/api/process"
                    )
                elif job.backend == "udmorph":
                    create_kwargs.setdefault(
                        "endpoint_url",
                        "https://lindat.mff.cuni.cz/services/teitok-live/udmorph/index.php?action=tag&act=tag"
                    )
                elif job.backend == "nametag":
                    create_kwargs.setdefault(
                        "endpoint_url",
                        "https://lindat.mff.cuni.cz/services/nametag/api/recognize"
                    )
                try:
                    # Backend creation (including any downloads) happens BEFORE timing starts
                    neural_backend = create_backend(job.backend, training=False, **create_kwargs)
                except SystemExit as exc:
                    # SystemExit might be raised by spacy.cli.download when model doesn't exist
                    # Check if we were trying to download a model
                    if self.download_models and job.model:
                        error_msg = str(exc) if exc.args else f"Model download failed (exit code {exc.code})"
                        print(
                            f"[benchmark] ⚠ Skipping {job.backend}/{job.model}: {error_msg}",
                            file=sys.stderr
                        )
                        return  # Skip this job
                    # For other SystemExits, re-raise
                    raise
                except (ValueError, OSError) as exc:
                    error_msg = str(exc)
                    # Check for various error patterns that indicate model not found/download failed
                    error_lower = error_msg.lower()
                    is_model_error = (
                        "not found" in error_lower
                        or "install" in error_lower
                        or "can't find" in error_lower
                        or "no compatible package" in error_lower
                        or "failed to download" in error_lower
                        or "doesn't exist" in error_lower
                        or "not available" in error_lower
                        or "could not find" in error_lower
                    )
                    if is_model_error:
                        # Mark model as disabled if download was attempted and failed
                        if self.download_models and job.model:
                            mark_model_disabled(job.backend, job.model)
                            print(
                                f"[benchmark] ⚠ Disabled {job.backend}/{job.model}: {error_msg}",
                                file=sys.stderr
                            )
                        else:
                            print(
                                f"[benchmark] ⚠ Skipping {job.backend}/{job.model or 'default'}: {error_msg}",
                                file=sys.stderr
                            )
                            if not self.download_models:
                                print(
                                    f"[benchmark] Tip: Use --download-models to automatically download missing models.",
                                    file=sys.stderr
                                )
                        return  # Skip this job
                    # For other ValueErrors/OSErrors, re-raise
                    raise
                except Exception as exc:
                    # Catch any other exceptions during backend creation (e.g., network errors, import errors)
                    error_msg = str(exc)
                    error_lower = error_msg.lower()
                    # Check if it's a model-related error
                    is_model_error = (
                        "not found" in error_lower
                        or "install" in error_lower
                        or "can't find" in error_lower
                        or "no compatible package" in error_lower
                        or "failed to download" in error_lower
                        or "doesn't exist" in error_lower
                        or "not available" in error_lower
                        or "could not find" in error_lower
                        or ("download" in error_lower and ("failed" in error_lower or "error" in error_lower))
                    )
                    # Also treat as model error if we were trying to download and got any exception
                    if self.download_models and job.model and not is_model_error:
                        # If we were downloading and got an unexpected error, it's likely a download failure
                        is_model_error = True
                    if is_model_error:
                        print(
                            f"[benchmark] ⚠ Skipping {job.backend}/{job.model or 'default'}: {error_msg}",
                            file=sys.stderr
                        )
                        if not self.download_models:
                            print(
                                f"[benchmark] Tip: Use --download-models to automatically download missing models.",
                                file=sys.stderr
                            )
                        return  # Skip this job
                    # For unexpected errors, re-raise
                    raise

            # Timing starts AFTER backend creation (downloads are not included in speed measurements)
            start = time.time()
            try:
                metrics_path = evaluate_model(
                    gold_path=treebank_path,  # Use converted CoNLL-U path
                    output_dir=output_dir,
                    gold_format="conllu",  # Always use CoNLL-U format for benchmark
                    model_path=model_path,
                    neural_backend=neural_backend,
                    verbose=self.verbose,
                    mode=job.mode,
                    create_implicit_mwt=False,
                )
            except Exception as exc:
                print(
                    f"[benchmark] ✗ Failed {job.backend}/{job.model or 'default'} "
                    f"on {Path(job.treebank).name}: {exc}",
                    file=sys.stderr
                )
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return  # Skip this job
            elapsed = time.time() - start
            summary = self._load_metrics(metrics_path)
            model_size = self._compute_model_size_bytes(job, model_path)
            entry = self._build_result_entry(job, summary, elapsed, model_size)
            self.storage.append(entry)
            metrics_dict = summary.get("metrics", {})
            upos_acc = metrics_dict.get("upos", {}).get("accuracy")
            xpos_acc = metrics_dict.get("xpos", {}).get("accuracy")
            feats_partial_acc = metrics_dict.get("feats_partial", {}).get("accuracy")
            lemma_acc = metrics_dict.get("lemma", {}).get("accuracy")
            uas = summary.get("uas")
            las = summary.get("las")
            parts = []
            if upos_acc is not None:
                parts.append(f"UPOS: {upos_acc*100:.1f}%")
            if xpos_acc is not None:
                parts.append(f"XPOS: {xpos_acc*100:.1f}%")
            if feats_partial_acc is not None:
                parts.append(f"FEATS_PARTIAL: {feats_partial_acc*100:.1f}%")
            if lemma_acc is not None:
                parts.append(f"LEMMA: {lemma_acc*100:.1f}%")
            if uas is not None:
                parts.append(f"UAS: {uas*100:.1f}%")
            if las is not None:
                parts.append(f"LAS: {las*100:.1f}%")
            metrics_str = ", ".join(parts) if parts else "no metrics"
            print(
                f"[benchmark] ✓ Completed {job.backend}/{job.model or 'default'} "
                f"on {Path(job.treebank).name} - {metrics_str}"
            )
        finally:
            if neural_backend and hasattr(neural_backend, "close"):
                try:
                    neural_backend.close()
                except Exception:
                    pass

    def _load_metrics(self, metrics_path: Path) -> dict:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_result_entry(
        self,
        job: BenchmarkJob,
        summary: dict,
        elapsed: float,
        model_size_bytes: Optional[int],
    ) -> dict:
        metrics = summary.get("metrics", {})
        token_metric = metrics.get("tokenization") or {}
        token_total = token_metric.get("total") or 0
        tokens_per_sec = (token_total / elapsed) if elapsed > 0 and token_total else None
        return {
            "language": job.language,
            "backend": job.backend,
            "model": job.model,
            "treebank": str(job.treebank),
            "tasks": list(job.tasks),
            "mode": job.mode,
            "metrics": metrics,
            "uas": summary.get("uas"),
            "las": summary.get("las"),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
            "model_size_bytes": model_size_bytes,
            "treebank_root": str(self.treebank_root) if self.treebank_root else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _compute_model_size_bytes(
        self,
        job: BenchmarkJob,
        model_path: Optional[Path],
    ) -> Optional[int]:
        info = get_backend_info(job.backend)
        if info and info.is_rest:
            return 0
        candidate: Optional[Path] = None
        if model_path and model_path.exists():
            candidate = model_path
        else:
            candidates: list[Path] = []
            if job.model:
                # Try to get backend directory first for backend-specific paths
                try:
                    backend_dir = get_backend_models_dir(job.backend, create=False)
                    # For ClassLA, models are stored as lang/processor/package.pt
                    # Model name format is "lang-type" (e.g., "bg-standard")
                    # Priority: check lang directory first (most reliable)
                    if job.backend == "classla" and "-" in job.model:
                        lang_code = job.model.split("-")[0]
                        candidates.append(backend_dir / lang_code)
                    # For Stanza, models are stored as lang/processor/package.pt
                    # Model name format is usually "lang_package" (e.g., "bg_btb")
                    elif job.backend == "stanza" and "_" in job.model:
                        lang_code = job.model.split("_")[0]
                        candidates.append(backend_dir / lang_code)
                    # Also try the full model name path
                    candidates.append(backend_dir / job.model)
                except Exception as exc:
                    if self.verbose:
                        print(f"[benchmark] Warning: Could not determine backend models directory: {exc}")
                # Fallback: try as direct path
                candidates.append(Path(job.model).expanduser())
            # Check candidates in order
            for path in candidates:
                if path.exists():
                    candidate = path
                    if self.verbose:
                        print(f"[benchmark] Found model at: {candidate}")
                    break
        if not candidate:
            if self.verbose:
                print(f"[benchmark] Warning: Could not find model path for {job.backend}/{job.model}")
            return None
        if candidate.is_file():
            return candidate.stat().st_size
        # For directories, sum all file sizes recursively
        try:
            total = 0
            for path in candidate.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
            return total if total > 0 else None
        except (OSError, PermissionError) as exc:
            if self.verbose:
                print(f"[benchmark] Warning: Could not calculate model size for {candidate}: {exc}")
            return None


# --------------------------------------------------------------------------- CLI
def parse_model_map(entries: Optional[Sequence[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not entries:
        return mapping
    for item in entries:
        if "=" not in item:
            raise SystemExit(f"Invalid --models entry '{item}'. Expected BACKEND=MODEL format.")
        backend, model = item.split("=", 1)
        backend = backend.strip().lower()
        model = model.strip()
        if not backend or not model:
            raise SystemExit(f"Invalid --models entry '{item}'.")
        mapping[backend] = model
    return mapping


def parse_tasks_arg(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ("tagging", "parsing", "ner")
    tasks = [part.strip().lower() for part in value.split(",") if part.strip()]
    return tuple(tasks or ["tagging"])


def _print_treebank_catalog(catalog: list[dict], output_format: str = "table") -> None:
    if not catalog:
        if output_format == "json":
            import json
            print(json.dumps({"treebanks": []}, indent=2, ensure_ascii=False))
        else:
            print("[benchmark] No treebanks available.")
        return
    
    if output_format == "json":
        import json
        print(json.dumps({"treebanks": catalog, "total": len(catalog)}, indent=2, ensure_ascii=False))
        return
    
    # Table format
    header = f"{'Language':<10} {'Treebank ID':<20} {'Path'}"
    print(header)
    print("-" * len(header))
    for entry in catalog:
        lang = entry.get("language") or "-"
        tbid = entry.get("id") or "-"
        path = entry.get("path") or "-"
        print(f"{lang:<10} {tbid:<20} {path}")


def _print_model_catalog(catalog: list[dict], output_format: str = "table") -> None:
    if not catalog:
        if output_format == "json":
            import json
            print(json.dumps({"models": []}, indent=2, ensure_ascii=False))
        else:
            print("[benchmark] No models available.")
        return
    
    if output_format == "json":
        import json
        print(json.dumps({"models": catalog, "total": len(catalog)}, indent=2, ensure_ascii=False))
        return
    
    # Table format
    header = f"{'Language':<10} {'Backend':<10} {'Model'}"
    print(header)
    print("-" * len(header))
    for entry in catalog:
        lang = entry.get("language") or "-"
        backend = entry.get("backend") or "-"
        model = entry.get("model") or "-"
        print(f"{lang:<10} {backend:<10} {model}")


def _normalize_language_code(value: Optional[str], fallback: Optional[str] = None) -> str:
    if value:
        normalized = value.strip().lower()
        if normalized:
            return normalized
    if fallback:
        normalized = fallback.strip().lower()
        if normalized:
            return normalized
    return ""


def _print_language_coverage(treebanks: list[dict], models: list[dict], output_format: str = "table") -> None:
    treebank_langs = {
        _normalize_language_code(entry.get("language")) for entry in treebanks
    }
    model_langs = {
        _normalize_language_code(entry.get("language"), entry.get("language_name"))
        for entry in models
    }
    languages = sorted(
        {lang for lang in treebank_langs | model_langs if lang}
    )
    if not languages:
        if output_format == "json":
            import json
            print(json.dumps({"languages": [], "summary": {}}, indent=2, ensure_ascii=False))
        else:
            print("[benchmark] No languages detected in catalogs.")
        return
    
    # Collect data for both JSON and table output
    languages_data = []
    for lang in languages:
        models_for_lang = [
            f"{entry.get('backend')}:{entry.get('model')}"
            for entry in models
            if _normalize_language_code(entry.get("language"), entry.get("language_name")) == lang
        ]
        treebanks_for_lang = [
            entry.get("id") or Path(entry.get("path", "")).stem
            for entry in treebanks
            if _normalize_language_code(entry.get("language")) == lang
        ]
        model_count = len(models_for_lang)
        treebank_count = len(treebanks_for_lang)
        if model_count and treebank_count:
            status = f"{model_count} models x {treebank_count} tests"
        elif model_count:
            status = f"{model_count} models only"
        elif treebank_count:
            status = f"{treebank_count} tests only"
        else:
            status = "none"
        
        languages_data.append({
            "language": lang,
            "models": models_for_lang,
            "treebanks": treebanks_for_lang,
            "model_count": model_count,
            "treebank_count": treebank_count,
            "status": status,
        })
    
    if output_format == "json":
        import json
        # Calculate summary
        lang_count = len(languages_data)
        with_both = sum(1 for lang_data in languages_data if lang_data["model_count"] > 0 and lang_data["treebank_count"] > 0)
        without_models = sum(1 for lang_data in languages_data if lang_data["model_count"] == 0)
        without_tests = sum(1 for lang_data in languages_data if lang_data["treebank_count"] == 0)
        benchmark_runs = sum(lang_data["model_count"] * lang_data["treebank_count"] for lang_data in languages_data)
        
        result = {
            "languages": languages_data,
            "summary": {
                "languages_covered": lang_count,
                "languages_with_both_models_and_tests": with_both,
                "languages_without_models": without_models,
                "languages_without_tests": without_tests,
                "potential_benchmark_runs": benchmark_runs,
            }
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Table format
    header = f"{'Language':<10} {'Models':<40} {'Treebanks':<30} {'Status':<20}"
    print(header)
    print("-" * len(header))
    for lang_data in languages_data:
        models_str = ", ".join(lang_data["models"]) if lang_data["models"] else "-"
        treebanks_str = ", ".join(lang_data["treebanks"]) if lang_data["treebanks"] else "-"
        print(
            f"{lang_data['language']:<10} {models_str:<40.40} {treebanks_str:<30.30} {lang_data['status']:<20}"
        )
    
    # Calculate summary for table format
    lang_count = len(languages_data)
    with_both = sum(1 for lang_data in languages_data if lang_data["model_count"] > 0 and lang_data["treebank_count"] > 0)
    without_models = sum(1 for lang_data in languages_data if lang_data["model_count"] == 0)
    without_tests = sum(1 for lang_data in languages_data if lang_data["treebank_count"] == 0)
    benchmark_runs = sum(lang_data["model_count"] * lang_data["treebank_count"] for lang_data in languages_data)
    
    print("\nSummary:")
    print(f"  Languages covered: {lang_count}")
    print(f"  Languages with both models and tests: {with_both}")
    print(f"  Languages without models: {without_models}")
    print(f"  Languages without tests: {without_tests}")
    print(f"  Potential benchmark runs (models × tests): {benchmark_runs}")
def _compute_averages(rows: list[dict]) -> list[dict]:
    """Group results by language/backend/model/mode and compute average metrics."""
    from collections import defaultdict
    
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("language", ""),
            row.get("backend", ""),
            row.get("model", ""),
            row.get("mode", "auto"),  # Include mode in grouping
        )
        grouped[key].append(row)
    
    averaged = []
    for (language, backend, model, mode), group_rows in grouped.items():
        if not group_rows:
            continue
        
        # Collect all metric values
        upos_values = []
        xpos_values = []
        feats_partial_values = []
        lemma_values = []
        uas_values = []
        las_values = []
        tps_values = []
        size_bytes = None
        
        for row in group_rows:
            metrics = row.get("metrics", {})
            upos = metrics.get("upos", {}).get("accuracy")
            xpos = metrics.get("xpos", {}).get("accuracy")
            feats_partial = metrics.get("feats_partial", {}).get("accuracy")
            lemma = metrics.get("lemma", {}).get("accuracy")
            uas = row.get("uas")
            las = row.get("las")
            tps = row.get("tokens_per_second")
            
            if upos is not None:
                upos_values.append(upos)
            if xpos is not None:
                xpos_values.append(xpos)
            if feats_partial is not None:
                feats_partial_values.append(feats_partial)
            if lemma is not None:
                lemma_values.append(lemma)
            if uas is not None:
                uas_values.append(uas)
            if las is not None:
                las_values.append(las)
            if tps is not None:
                tps_values.append(tps)
            if size_bytes is None:
                size_bytes = row.get("model_size_bytes")
        
        # Compute averages
        avg_upos = sum(upos_values) / len(upos_values) if upos_values else None
        avg_xpos = sum(xpos_values) / len(xpos_values) if xpos_values else None
        avg_feats_partial = sum(feats_partial_values) / len(feats_partial_values) if feats_partial_values else None
        avg_lemma = sum(lemma_values) / len(lemma_values) if lemma_values else None
        avg_uas = sum(uas_values) / len(uas_values) if uas_values else None
        avg_las = sum(las_values) / len(las_values) if las_values else None
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else None
        
        averaged.append({
            "language": language,
            "backend": backend,
            "model": model,
            "mode": mode,  # Include mode in output
            "treebank_count": len(group_rows),
            "metrics": {
                "upos": {"accuracy": avg_upos},
                "xpos": {"accuracy": avg_xpos},
                "feats_partial": {"accuracy": avg_feats_partial},
                "lemma": {"accuracy": avg_lemma},
            },
            "uas": avg_uas,
            "las": avg_las,
            "tokens_per_second": avg_tps,
            "model_size_bytes": size_bytes,
        })
    
    return averaged


def handle_show(storage: BenchmarkStorage, args: argparse.Namespace) -> None:
    rows = storage.load()
    output_format = getattr(args, "output_format", "table")
    
    # Filter by language
    if getattr(args, "language", None):
        language_filter = getattr(args, "language", "").lower()
        rows = [row for row in rows if row.get("language", "").lower() == language_filter]
    # Filter by languages list if provided (and no single --language was given)
    elif getattr(args, "languages", None):
        languages_set = {lang.lower() for lang in getattr(args, "languages", [])}
        rows = [row for row in rows if row.get("language", "").lower() in languages_set]
    # Filter by backend (check both --backend and --backends for compatibility)
    backend_filter = getattr(args, "backend", None)
    if not backend_filter and getattr(args, "backends", None):
        # If --backends was provided, use the first one for filtering
        backends_list = getattr(args, "backends", [])
        if backends_list and len(backends_list) == 1:
            backend_filter = backends_list[0]
    if backend_filter:
        backend_filter = backend_filter.lower()
        original_count = len(rows)
        rows = [row for row in rows if row.get("backend", "").lower() == backend_filter]
        if getattr(args, "debug", False) or getattr(args, "verbose", False):
            import sys
            print(f"[benchmark] Filtered by backend '{backend_filter}': {original_count} -> {len(rows)} results", file=sys.stderr)
    # Filter by mode
    if getattr(args, "mode", None):
        mode_filter = getattr(args, "mode", "").lower()
        rows = [row for row in rows if row.get("mode", "auto").lower() == mode_filter]
    if not rows:
        if output_format == "json":
            import json
            print(json.dumps({"results": []}, indent=2, ensure_ascii=False))
        else:
            print("[benchmark] No results recorded yet.")
        return
    
    # Compute averages if requested
    if getattr(args, "average", False):
        rows = _compute_averages(rows)
        treebank_col = "Treebanks"
    else:
        treebank_col = "Treebank"
    
    # Sort by metric (default: UPOS%)
    sort_key = getattr(args, "sort_by", "upos") or "upos"
    def sort_key_func(row: dict) -> tuple:
        """Return a sort key tuple: (is_none, -value) so None values go last and higher values come first."""
        metrics = row.get("metrics", {})
        if sort_key == "upos":
            value = metrics.get("upos", {}).get("accuracy")
        elif sort_key == "xpos":
            value = metrics.get("xpos", {}).get("accuracy")
        elif sort_key == "feats_partial" or sort_key == "featsp":
            value = metrics.get("feats_partial", {}).get("accuracy")
        elif sort_key == "lemma":
            value = metrics.get("lemma", {}).get("accuracy")
        elif sort_key == "uas":
            value = row.get("uas")
        elif sort_key == "las":
            value = row.get("las")
        elif sort_key == "tokens_per_second" or sort_key == "tps":
            value = row.get("tokens_per_second")
        else:
            value = metrics.get("upos", {}).get("accuracy")
        # Return (is_none, -value) so None values sort last and higher values come first
        return (value is None, -(value or 0))
    
    rows = sorted(rows, key=sort_key_func)
    
    # Output JSON if requested
    if output_format == "json":
        import json
        # Convert rows to JSON-friendly format
        json_rows = []
        for row in rows:
            json_row = {
                "language": row.get("language"),
                "backend": row.get("backend"),
                "model": row.get("model"),
                "mode": row.get("mode", "auto"),
            }
            if getattr(args, "average", False):
                json_row["treebank_count"] = row.get("treebank_count", 0)
            else:
                json_row["treebank"] = row.get("treebank")
            
            metrics = row.get("metrics", {})
            json_row["metrics"] = {
                "upos": metrics.get("upos", {}).get("accuracy"),
                "xpos": metrics.get("xpos", {}).get("accuracy"),
                "feats_partial": metrics.get("feats_partial", {}).get("accuracy"),
                "lemma": metrics.get("lemma", {}).get("accuracy"),
            }
            json_row["uas"] = row.get("uas")
            json_row["las"] = row.get("las")
            json_row["tokens_per_second"] = row.get("tokens_per_second")
            json_row["model_size_bytes"] = row.get("model_size_bytes")
            json_rows.append(json_row)
        
        result = {
            "results": json_rows,
            "total": len(json_rows),
            "averaged": getattr(args, "average", False),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # Check if we have multiple modes in the results
    modes_in_results = {row.get("mode", "auto") for row in rows}
    show_mode_column = len(modes_in_results) > 1
    
    if show_mode_column:
        header = (
            f"{'Language':<10} {'Backend':<10} {'Model':<25} {'Mode':<10} "
            f"{treebank_col:<30} {'UPOS%':>7} {'XPOS%':>7} {'FEATSP%':>7} {'LEMMA%':>7} "
            f"{'UAS%':>7} {'LAS%':>7} {'Tok/s':>9} {'SizeMB':>8}"
        )
    else:
        header = (
            f"{'Language':<10} {'Backend':<10} {'Model':<25} "
            f"{treebank_col:<30} {'UPOS%':>7} {'XPOS%':>7} {'FEATSP%':>7} {'LEMMA%':>7} "
            f"{'UAS%':>7} {'LAS%':>7} {'Tok/s':>9} {'SizeMB':>8}"
        )
    print(header)
    print("-" * len(header))
    for row in rows:
        metrics = row.get("metrics", {})
        upos = metrics.get("upos", {}).get("accuracy")
        xpos = metrics.get("xpos", {}).get("accuracy")
        feats_partial = metrics.get("feats_partial", {}).get("accuracy")
        lemma = metrics.get("lemma", {}).get("accuracy")
        uas = row.get("uas")
        las = row.get("las")
        tps = row.get("tokens_per_second")
        model = row.get("model") or "-"
        if getattr(args, "average", False):
            treebank = f"{row.get('treebank_count', 0)} treebanks"
        else:
            treebank = Path(row.get("treebank", "")).name
        upos_str = f"{upos*100:5.1f}%" if upos is not None else "  n/a"
        xpos_str = f"{xpos*100:5.1f}%" if xpos is not None else "  n/a"
        feats_partial_str = f"{feats_partial*100:5.1f}%" if feats_partial is not None else "  n/a"
        lemma_str = f"{lemma*100:5.1f}%" if lemma is not None else "  n/a"
        uas_str = f"{uas*100:5.1f}%" if uas is not None else "  n/a"
        las_str = f"{las*100:5.1f}%" if las is not None else "  n/a"
        tps_str = f"{tps:8.0f}" if tps is not None else "    n/a"
        size_bytes = row.get("model_size_bytes")
        size_str = (
            f"{(size_bytes / (1024**2)):6.1f}" if size_bytes is not None else "   n/a"
        )
        mode_str = row.get("mode", "auto")
        if show_mode_column:
            print(
                f"{row.get('language','-'):<10} "
                f"{row.get('backend','-'):<10} "
                f"{model:<25.25} "
                f"{mode_str:<10} "
                f"{treebank:<30.30} "
                f"{upos_str:>7} "
                f"{xpos_str:>7} "
                f"{feats_partial_str:>7} "
                f"{lemma_str:>7} "
                f"{uas_str:>7} "
                f"{las_str:>7} "
                f"{tps_str:>9} "
                f"{size_str:>8}"
            )
        else:
            print(
                f"{row.get('language','-'):<10} "
                f"{row.get('backend','-'):<10} "
                f"{model:<25.25} "
                f"{treebank:<30.30} "
                f"{upos_str:>7} "
                f"{xpos_str:>7} "
                f"{feats_partial_str:>7} "
                f"{lemma_str:>7} "
                f"{uas_str:>7} "
                f"{las_str:>7} "
                f"{tps_str:>9} "
                f"{size_str:>8}"
            )
    
    # Print summary: X benchmarks for Y models in Z languages on W treebanks
    num_benchmarks = len(rows)
    unique_models = set()
    unique_languages = set()
    unique_treebanks = set()
    for row in rows:
        backend = row.get("backend", "")
        model = row.get("model", "")
        if backend and model:
            unique_models.add((backend, model))
        language = row.get("language", "")
        if language:
            unique_languages.add(language)
        treebank = row.get("treebank", "")
        if treebank:
            unique_treebanks.add(treebank)
    
    num_models = len(unique_models)
    num_languages = len(unique_languages)
    num_treebanks = len(unique_treebanks)
    print(f"\n{num_benchmarks} benchmark(s) for {num_models} model(s) in {num_languages} language(s) on {num_treebanks} treebank(s)")


def handle_test(args: argparse.Namespace, runner: BenchmarkRunner) -> None:
    # Allow --test to work standalone (like the old 'check' command)
    # If treebank is provided as a list, take first element
    treebank_path = None
    if hasattr(args, 'treebank') and args.treebank:
        treebank_path = Path(args.treebank[0] if isinstance(args.treebank, list) else args.treebank).expanduser()
    elif hasattr(args, 'input') and args.input:
        # Allow using --input as treebank path for standalone use
        treebank_path = Path(args.input).expanduser()
    
    if not treebank_path or not treebank_path.exists():
        if not hasattr(args, 'language') or not args.language or not hasattr(args, 'backend') or not args.backend:
            raise SystemExit(
                "--test requires --language, --backend, --model, and --treebank (or --input).\n"
                "Example: benchmark --test --language en --backend spacy --model en_core_news_sm --treebank test.conllu"
            )
        raise SystemExit(f"Treebank file '{treebank_path}' does not exist.")
    
    # Auto-detect language from treebank if not provided
    language = getattr(args, 'language', None)
    if not language and treebank_path:
        # Try to infer from filename (e.g., en_ewt-ud-test.conllu -> en)
        stem = treebank_path.stem
        if '_' in stem:
            language = stem.split('_')[0]
        elif '-' in stem:
            language = stem.split('-')[0]
    
    if not language:
        raise SystemExit("--test requires --language (or provide a treebank filename with language prefix).")
    
    backend = getattr(args, 'backend', None)
    if not backend:
        raise SystemExit("--test requires --backend.")
    
    job = BenchmarkJob(
        language=language,
        backend=backend.lower(),
        model=getattr(args, "model", None),
        treebank=treebank_path,
        tasks=parse_tasks_arg(getattr(args, "tasks", None)),
        mode=getattr(args, "mode", "auto"),
    )
    runner._execute_job(job)


def handle_debug_flexitag(args: argparse.Namespace) -> None:
    """Handle --debug-flexitag flag (replaces debug-accuracy command)."""
    from pathlib import Path
    from .scripts import debug_accuracy
    from .conllu import conllu_to_document
    from .check import _detag_document
    from .engine import FlexitagFallback
    
    model_path = Path(getattr(args, "debug_flexitag_model", None) or "")
    test_path = Path(getattr(args, "debug_flexitag_test", None) or "")
    
    if not model_path or not model_path.exists():
        raise SystemExit(f"Error: Model file not found. Use --debug-flexitag-model <path>")
    
    if not test_path or not test_path.exists():
        raise SystemExit(f"Error: Test file not found. Use --debug-flexitag-test <path>")
    
    # Load vocabulary and determine tagpos
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print(f"[flexipipe] Loading vocabulary from {model_path}...")
    vocab = debug_accuracy.load_vocab(model_path)
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print(f"[flexipipe] Loaded {len(vocab)} vocabulary entries")
    
    # Get tagpos from model
    tagpos = debug_accuracy.get_tagpos_from_model(model_path)
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print(f"[flexipipe] Using tagpos: {tagpos}")
    
    # Load gold standard
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print(f"[flexipipe] Loading gold standard from {test_path}...")
    gold_text = test_path.read_text(encoding="utf-8", errors="replace")
    gold_doc = conllu_to_document(gold_text, doc_id=test_path.stem)
    
    # CRITICAL: Detag the input to avoid data leakage
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print("[flexipipe] Detagging input to prevent data leakage...")
    input_doc = _detag_document(gold_doc)
    
    # Tag with flexitag
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print("[flexipipe] Tagging with flexitag...")
    flexitag_options = {"tagpos": tagpos, "overwrite": True}
    if getattr(args, "debug", False):
        flexitag_options["debug"] = True
    endlen = getattr(args, "debug_flexitag_endlen", None)
    if endlen is not None:
        flexitag_options["endlen"] = endlen
    fallback = FlexitagFallback(str(model_path), options=flexitag_options, debug=getattr(args, "debug", False))
    
    # Measure total time (including Python overhead)
    import time
    total_start_time = time.time()
    pred_result = fallback.tag(input_doc)
    total_elapsed_time = time.time() - total_start_time
    pred_doc = pred_result.document
    
    # Analyze errors
    if getattr(args, "verbose", False) or getattr(args, "debug", False):
        print("[flexipipe] Analyzing errors...")
    stats = debug_accuracy.analyze_errors(gold_doc, pred_doc, vocab, tagpos=tagpos, verbose=getattr(args, "debug", False))
    
    # Extract timing stats from pred_result
    elapsed_seconds = 0.0
    word_count = sum(len(sent.tokens) for sent in gold_doc.sentences)
    if hasattr(pred_result, 'stats') and pred_result.stats:
        stats_dict = pred_result.stats
        if isinstance(stats_dict, dict) and stats_dict:
            elapsed_val = stats_dict.get("elapsed_seconds")
            if elapsed_val is not None:
                elapsed_seconds = float(elapsed_val)
            word_count_val = stats_dict.get("word_count")
            if word_count_val is not None:
                word_count = int(word_count_val)
    
    # Use total_elapsed_time if C++ timing is not available
    if elapsed_seconds == 0.0 or elapsed_seconds < 0.001:
        elapsed_seconds = total_elapsed_time
    
    # Store timing and sentence info in stats dict
    sentence_count = len(gold_doc.sentences) if gold_doc.sentences else 0
    stats["elapsed_seconds"] = float(elapsed_seconds)
    stats["total_elapsed_seconds"] = float(total_elapsed_time)
    stats["word_count"] = int(word_count)
    stats["sentence_count"] = int(sentence_count)
    
    # Calculate speeds
    timing_for_speed = elapsed_seconds if elapsed_seconds > 0 else total_elapsed_time
    if timing_for_speed > 0:
        stats["speed"] = float(stats["word_count"]) / timing_for_speed
        stats["sent_speed"] = float(stats["sentence_count"]) / timing_for_speed
    else:
        stats["speed"] = 0.0
        stats["sent_speed"] = 0.0
    
    # Print report
    debug_accuracy.print_report(stats, tagpos=tagpos, verbose=getattr(args, "debug", False))
    
    # Save output if requested
    output_path = getattr(args, "debug_flexitag_output", None)
    if output_path:
        output_path = Path(output_path)
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        if getattr(args, "verbose", False) or getattr(args, "debug", False):
            print(f"\n[flexipipe] Statistics saved to {output_path}")


def handle_run(args: argparse.Namespace, runner: BenchmarkRunner) -> None:
    # Get model and treebank catalogs to filter intelligently
    models_catalog = None
    if args.models_file:
        models_path = Path(args.models_file).expanduser()
        if models_path.exists():
            with models_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    models_catalog = data
    if models_catalog is None and DEFAULT_MODEL_CATALOG.exists():
        with DEFAULT_MODEL_CATALOG.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                models_catalog = data
    if models_catalog is None:
        models_catalog = runner.build_model_catalog()
    
    treebanks_catalog = runner.treebank_catalog or runner._build_treebank_catalog()
    
    # Determine which backends we'll be using
    # Handle "all" for backends first, so we can filter languages based on selected backends
    if args.backends and "all" in [b.lower() for b in args.backends]:
        all_backends = list_backends(include_hidden=False)
        # For "all" backends, we'll filter later based on models
        backends = sorted(all_backends.keys())
        if runner.verbose:
            print(f"[benchmark] Using all {len(backends)} available backend(s)")
    else:
        backends = [b.lower() for b in (args.backends or [])]
        if not backends:
            raise SystemExit("No backends specified. Provide --backends backend1 backend2 ... or --backends all")
        # Validate that explicitly specified backends exist (including hidden ones)
        all_backends = list_backends(include_hidden=True)
        invalid_backends = [b for b in backends if b not in all_backends]
        if invalid_backends:
            raise SystemExit(f"Unknown backend(s): {', '.join(invalid_backends)}")
    
    # Handle "all" for languages - filter to languages that have models for the selected backends
    if args.languages and "all" in [lang.lower() for lang in args.languages]:
        # Find languages that have models for at least one of the selected backends
        # For REST backends, we need to fetch models from the service
        treebank_langs = {
            _normalize_language_code(entry.get("language")) for entry in treebanks_catalog
        }
        
        # Collect languages that have models for the selected backends
        model_langs = set()
        for backend_name in backends:
            # Check if this is a REST backend - if so, fetch models from service
            backend_info = get_backend_info(backend_name)
            is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
            
            try:
                # For REST backends, force refresh to get current models
                # For non-REST backends, use cache
                entries = get_model_entries(
                    backend_name,
                    use_cache=not is_rest_backend,  # Don't use cache for REST backends
                    refresh_cache=is_rest_backend,  # Force refresh for REST backends
                    verbose=runner.verbose,
                )
                
                # Extract languages from model entries
                for entry in entries.values():
                    if not isinstance(entry, dict):
                        continue
                    lang_code = _normalize_language_code(
                        entry.get("language"), 
                        entry.get("language_name")
                    )
                    if lang_code:
                        model_langs.add(lang_code)
            except Exception as exc:
                if runner.verbose:
                    print(f"[benchmark] Warning: Could not fetch models for {backend_name}: {exc}")
                # Also check the catalog as fallback
                for entry in models_catalog:
                    if entry.get("backend") == backend_name:
                        lang_code = _normalize_language_code(
                            entry.get("language"), 
                            entry.get("language_name")
                        )
                        if lang_code:
                            model_langs.add(lang_code)
        
        # Only include languages that have both models (for selected backends) and tests
        languages = sorted([lang for lang in treebank_langs & model_langs if lang])
        if not languages:
            raise SystemExit(
                f"No languages with both models (for selected backend(s): {', '.join(backends)}) and tests found. "
                "Use --list-tests to see available combinations."
            )
        if runner.verbose:
            print(f"[benchmark] Filtered to {len(languages)} language(s) with both models and tests")
    else:
        languages = args.languages or runner.discover_languages()
        if not languages:
            raise SystemExit("No languages specified and none discovered. Provide --languages or --treebank-root.")
    model_map = parse_model_map(args.models)
    tasks = parse_tasks_arg(args.tasks)
    explicit_treebanks: Optional[list[Path]] = None
    if args.treebank:
        explicit_treebanks = []
        for tb in args.treebank:
            tb_path = Path(tb).expanduser()
            if not tb_path.exists():
                print(f"[benchmark] Warning: treebank file '{tb_path}' does not exist; skipping.")
                continue
            explicit_treebanks.append(tb_path)
        if not explicit_treebanks:
            print("[benchmark] No valid treebank paths supplied via --treebank.")
    jobs: list[BenchmarkJob] = []
    for language in languages:
        if explicit_treebanks is not None:
            treebanks = [
                tb for tb in explicit_treebanks if language.lower() in tb.name.lower()
            ] or explicit_treebanks
        else:
            treebanks = runner.discover_treebanks(language, limit=args.limit_treebanks)
        if not treebanks:
            print(f"[benchmark] Warning: no treebanks found for language '{language}'")
            continue
        for backend in backends:
            if backend in model_map:
                model_names = [model_map[backend]]
            else:
                model_names = runner.discover_models_for_language(backend, language)
            
            # Check if this is a REST backend
            backend_info = get_backend_info(backend)
            is_rest_backend = backend_info and backend_info.is_rest if backend_info else False
            
            # For REST backends, if no models found, try to discover them from the service
            # Some REST backends (like UDPipe) require specific model names, so we can't just use language codes
            if not model_names and is_rest_backend:
                # Try to refresh the cache and discover models again
                try:
                    if runner.verbose:
                        print(f"[benchmark] No models found for REST backend {backend}, trying to refresh model list...")
                    # Try refreshing the cache to get models from the REST service
                    entries = get_model_entries(
                        backend,
                        use_cache=False,  # Force refresh
                        refresh_cache=True,
                        verbose=runner.verbose,
                    )
                    query = resolve_language_query(language)
                    for entry in entries.values():
                        if not isinstance(entry, dict):
                            continue
                        if language_matches_entry(entry, query, allow_fuzzy=True):
                            model_name = entry.get("model")
                            if model_name:
                                model_names.append(model_name)
                    # Remove duplicates
                    seen: set[str] = set()
                    unique: list[str] = []
                    for model in model_names:
                        if model not in seen:
                            unique.append(model)
                            seen.add(model)
                    model_names = unique
                except Exception as exc:
                    if runner.verbose:
                        print(f"[benchmark] Could not refresh models for REST backend {backend}: {exc}")
                    # If refresh fails, we'll still try to proceed - the backend creation will handle errors
            
            # For REST backends, if still no models found, we should still try to create jobs
            # The backend will handle validation and provide appropriate error messages
            # For non-REST backends, skip if no models found
            if backend != "flexitag" and not model_names and not is_rest_backend:
                print(
                    f"[benchmark] Skipping {backend} for {language}: no models available "
                    "locally or in registry."
                )
                continue
            if backend == "flexitag" and not model_names:
                print(
                    f"[benchmark] Warning: no flexitag models found for {language}. "
                    "Install or copy them into FLEXIPIPE_MODELS_DIR/flexitag/. Skipping flexitag."
                )
                continue
            
            # For REST backends with no models, we still want to try (backend will validate)
            # But we need at least one model name to create jobs
            # If still no models, skip with a warning
            if is_rest_backend and not model_names:
                print(
                    f"[benchmark] Warning: no models found for REST backend {backend} and language {language}. "
                    "The backend may require explicit model names. Skipping."
                )
                continue
            
            limited_treebanks = treebanks[: args.limit_treebanks] if args.limit_treebanks else treebanks
            for tb in limited_treebanks:
                for model_name in model_names:
                    jobs.append(
                        BenchmarkJob(
                            language=language,
                            backend=backend,
                            model=model_name,
                            treebank=tb,
                            tasks=tasks,
                            mode=getattr(args, "mode", "auto"),
                        )
                    )
    if not jobs:
        print("[benchmark] No jobs queued.")
        return
    
    # Filter out jobs that already have results unless --force is set
    force = getattr(args, "force", False)
    if not force:
        existing_results = runner.storage.load()
        mode = getattr(args, "mode", "auto")
        existing_keys = {
            (
                row.get("language", ""),
                row.get("backend", ""),
                row.get("model", ""),
                str(row.get("treebank", "")),
                row.get("mode", "auto"),  # Include mode in key
            )
            for row in existing_results
        }
        original_count = len(jobs)
        jobs = [
            job
            for job in jobs
            if (
                job.language,
                job.backend,
                job.model or "",
                str(job.treebank),
                job.mode,  # Include mode in key
            ) not in existing_keys
        ]
        skipped = original_count - len(jobs)
        if skipped > 0:
            print(f"[benchmark] Skipping {skipped} job(s) that already have results (use --force to re-run).")
    
    if not jobs:
        print("[benchmark] No new jobs to run (all already completed).")
        return
    
    jobs_to_run = jobs[: args.limit_jobs] if args.limit_jobs else jobs
    print(f"[benchmark] Queued {len(jobs_to_run)} benchmark job(s).")
    runner.run_jobs(jobs_to_run)


def run_cli(args: argparse.Namespace) -> None:
    storage_path = Path(args.results_file).expanduser() if args.results_file else DEFAULT_RESULTS_PATH
    storage = BenchmarkStorage(storage_path)
    treebank_root = Path(args.treebank_root).expanduser() if args.treebank_root else None
    output_root = Path(args.output_dir).expanduser() if args.output_dir else None
    treebank_catalog = None
    if args.treebanks_file:
        catalog_path = Path(args.treebanks_file).expanduser()
        if catalog_path.exists():
            with catalog_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    treebank_catalog = data
    elif DEFAULT_TREEBANK_CATALOG.exists():
        with DEFAULT_TREEBANK_CATALOG.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                treebank_catalog = data
    models_catalog = None
    if args.models_file:
        models_path = Path(args.models_file).expanduser()
        if models_path.exists():
            with models_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    models_catalog = data
    elif DEFAULT_MODEL_CATALOG.exists():
        with DEFAULT_MODEL_CATALOG.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                models_catalog = data
    runner = BenchmarkRunner(
        storage,
        treebank_root=treebank_root,
        output_root=output_root,
        verbose=args.verbose,
        dry_run=args.dry_run,
        treebank_catalog=treebank_catalog,
        download_models=getattr(args, "download_models", False),
    )
    if getattr(args, "export_treebanks", None) is not None:
        export_treebanks = getattr(args, "export_treebanks", None)
        export_path = (
            DEFAULT_TREEBANK_CATALOG
            if export_treebanks == "__DEFAULT_TREEBANK__"
            else Path(export_treebanks).expanduser()
        )
        catalog = runner.treebank_catalog or runner._build_treebank_catalog()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(catalog, handle, indent=2)
            handle.write("\n")
        print(f"[benchmark] Treebank catalog exported to {export_path}")
        if not (getattr(args, "run", False) or getattr(args, "test", False) or getattr(args, "show", False)):
            return
    if getattr(args, "export_models", None) is not None:
        export_models = getattr(args, "export_models", None)
        if export_models.lower() == "web":
            # Special case: write to web/models.json relative to flexipipe package
            import os
            flexipipe_package = Path(__file__).parent.parent
            web_dir = flexipipe_package / "web"
            if not web_dir.exists():
                web_dir = Path.cwd() / "web"
            export_path = web_dir / "models.json"
        else:
            export_path = (
                DEFAULT_MODEL_CATALOG
                if export_models == "__DEFAULT_MODEL__"
                else Path(export_models).expanduser()
            )
        model_catalog = runner.build_model_catalog()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(model_catalog, handle, indent=2)
            handle.write("\n")
        print(f"[benchmark] Model catalog exported to {export_path}")
        if not (getattr(args, "run", False) or getattr(args, "test", False) or getattr(args, "show", False)):
            return

    handled_listing = False
    output_format = getattr(args, "output_format", "table")
    if getattr(args, "list_treebanks", False):
        catalog = runner.treebank_catalog or runner._build_treebank_catalog()
        runner.treebank_catalog = catalog
        _print_treebank_catalog(catalog, output_format=output_format)
        handled_listing = True
    if getattr(args, "list_models", False):
        catalog = models_catalog or runner.build_model_catalog()
        _print_model_catalog(catalog, output_format=output_format)
        handled_listing = True
    if getattr(args, "list_tests", False):
        treebanks_data = runner.treebank_catalog or runner._build_treebank_catalog()
        models_data = models_catalog or runner.build_model_catalog()
        _print_language_coverage(treebanks_data, models_data, output_format=output_format)
        handled_listing = True
    if handled_listing and not (getattr(args, "run", False) or getattr(args, "test", False) or getattr(args, "show", False)):
        return

    if not (getattr(args, "run", False) or getattr(args, "test", False) or getattr(args, "show", False)):
        if runner.treebank_root:
            catalog = runner._build_treebank_catalog()
            runner.treebank_catalog = catalog
            target = DEFAULT_TREEBANK_CATALOG
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(catalog, handle, indent=2)
                handle.write("\n")
            print(f"[benchmark] Treebank catalog written to {target}")
            if getattr(args, "export_models", None) is None:
                model_catalog = runner.build_model_catalog()
                DEFAULT_MODEL_CATALOG.parent.mkdir(parents=True, exist_ok=True)
                with DEFAULT_MODEL_CATALOG.open("w", encoding="utf-8") as handle:
                    json.dump(model_catalog, handle, indent=2)
                    handle.write("\n")
                print(f"[benchmark] Model catalog written to {DEFAULT_MODEL_CATALOG}")
            print("[benchmark] Re-run with --backends ... to start benchmarking.")
            return
        raise SystemExit("No action specified. Use --run, --test, or --show.")

    # Handle --export-benchmark
    if getattr(args, "export_benchmark", None):
        export_path = getattr(args, "export_benchmark")
        if export_path.lower() == "web":
            # Special case: write to web/benchmark.json relative to flexipipe package
            import os
            # Try to find web directory relative to flexipipe package
            flexipipe_package = Path(__file__).parent.parent
            web_dir = flexipipe_package / "web"
            if not web_dir.exists():
                # Fallback: try to find it relative to current working directory
                web_dir = Path.cwd() / "web"
            export_path = str(web_dir / "benchmark.json")
        else:
            export_path = Path(export_path).expanduser()
        
        # Get benchmark results
        rows = storage.load()
        output_format = getattr(args, "output_format", "json")
        
        # Convert to JSON format
        json_rows = []
        for row in rows:
            json_row = {
                "language": row.get("language"),
                "backend": row.get("backend"),
                "model": row.get("model"),
                "mode": row.get("mode", "auto"),
                "treebank": str(row.get("treebank", "")),
            }
            metrics = row.get("metrics", {})
            json_row["metrics"] = {
                "upos": metrics.get("upos", {}).get("accuracy") if isinstance(metrics.get("upos"), dict) else metrics.get("upos"),
                "xpos": metrics.get("xpos", {}).get("accuracy") if isinstance(metrics.get("xpos"), dict) else metrics.get("xpos"),
                "feats_partial": metrics.get("feats_partial", {}).get("accuracy") if isinstance(metrics.get("feats_partial"), dict) else metrics.get("feats_partial"),
                "lemma": metrics.get("lemma", {}).get("accuracy") if isinstance(metrics.get("lemma"), dict) else metrics.get("lemma"),
            }
            json_row["uas"] = row.get("uas")
            json_row["las"] = row.get("las")
            json_row["tokens_per_second"] = row.get("tokens_per_second")
            json_row["model_size_bytes"] = row.get("model_size_bytes")
            json_rows.append(json_row)
        
        result = {
            "results": json_rows,
            "total": len(json_rows),
        }
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"[benchmark] Benchmark results exported to {export_path}")
        return
    
    if getattr(args, "debug_flexitag", False):
        handle_debug_flexitag(args)
        return
    if getattr(args, "show", False):
        handle_show(storage, args)
        return
    if getattr(args, "test", False):
        handle_test(args, runner)
        return
    if getattr(args, "run", False):
        handle_run(args, runner)
        return
    raise SystemExit("No action specified. Use --run, --test, or --show.")

