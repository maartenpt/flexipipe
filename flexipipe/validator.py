from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from .ud_validator import run_ud_validator

if TYPE_CHECKING:
    from .doc import Document

UDAPI_HINT = (
    "[flexipipe] UD validator depends on the 'udapi' package. "
    "Install it with: pip install udapi (auto-install only covers flexipipe extras)."
)


def run_validator_cli(
    treebank_path: Path,
    *,
    version: str,
    url: Optional[str],
    rules: Optional[str],
    extra_args: Optional[List[str]],
    lang: Optional[str],
    verbose: bool,
    smart_validation: bool = True,
) -> int:
    """
    Run the UD validator with smart validation that filters out errors about
    missing annotations if those annotations aren't present in the file.
    
    Args:
        smart_validation: If True (default), filter errors about missing annotations
            that aren't present in the file. If False, show all errors.
    """
    if importlib.util.find_spec("udapi") is None:  # type: ignore[attr-defined]
        print(UDAPI_HINT, file=sys.stderr)
        return 1

    language = lang or infer_language_from_path(treebank_path)
    if not language:
        print(
            "[flexipipe] UD validator requires a language code (--validator-lang). "
            "Provide one explicitly or include '# language = xx' comments in the CoNLL-U headers.",
            file=sys.stderr,
        )
        return 1

    extras = list(extra_args or [])
    if not _extra_args_include_lang(extras):
        extras.extend(["--lang", language])

    # Detect annotation coverage if smart validation is enabled
    annotation_coverage = None
    if smart_validation and not rules:
        # Only do smart validation if user hasn't explicitly set validation rules
        annotation_coverage = detect_annotation_coverage(treebank_path)
        if verbose:
            print(
                "[flexipipe] Using smart validation (filtering errors about missing annotations)",
                file=sys.stderr,
            )
    elif verbose:
        if rules:
            print(
                f"[flexipipe] Using raw validation with rules: {rules}",
                file=sys.stderr,
            )
        else:
            print(
                "[flexipipe] Using raw validation (showing all errors)",
                file=sys.stderr,
            )

    try:
        result = run_ud_validator(
            treebank_path,
            version=version,
            url=url,
            rules=rules,
            extra_args=extras,
            verbose=verbose,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        if exc.name == "udapi":
            print(UDAPI_HINT, file=sys.stderr)
            return 1
        raise

    # Filter output if smart validation is enabled
    filtered_stdout = result.stdout
    filtered_stderr = result.stderr
    filtered_returncode = result.returncode
    
    if smart_validation and annotation_coverage:
        filtered_stdout, filtered_stderr, filtered_returncode = _filter_validator_output(
            result.stdout or "",
            result.stderr or "",
            result.returncode,
            annotation_coverage,
        )

    if filtered_stdout:
        sys.stdout.write(filtered_stdout)
        if not filtered_stdout.endswith("\n"):
            sys.stdout.write("\n")
    if filtered_stderr:
        sys.stderr.write(filtered_stderr)
        if not filtered_stderr.endswith("\n"):
            sys.stderr.write("\n")
    # Only print failure message if there are actual errors (not just filtered ones)
    if filtered_returncode != 0 and (filtered_stdout or filtered_stderr):
        # Check if output indicates failure
        output_text = (filtered_stdout or "") + (filtered_stderr or "")
        if "*** FAILED ***" in output_text or "FAILED" in output_text.upper():
            # Don't show temp file paths in error messages
            display_path = treebank_path
            if "/tmp/" in str(treebank_path) or treebank_path.name.startswith("tmp"):
                display_path = "output"
            print(
                f"[flexipipe] UD validator failed for {display_path} (exit code {filtered_returncode})",
                file=sys.stderr,
            )
    return filtered_returncode


def infer_language_from_document(
    doc: Optional["Document"],
    detection_result: Optional[dict],
    arg_language: Optional[str],
) -> Optional[str]:
    candidates: List[Optional[str]] = []
    if arg_language:
        candidates.append(arg_language)
    if doc:
        meta = getattr(doc, "meta", {}) or {}
        attrs = getattr(doc, "attrs", {}) or {}
        candidates.extend(
            [
                meta.get("language"),
                meta.get("lang"),
                attrs.get("language"),
                attrs.get("lang"),
            ]
        )
    if detection_result:
        candidates.extend(
            [
                detection_result.get("language"),
                detection_result.get("language_iso"),
                detection_result.get("language_code"),
            ]
        )
    for candidate in candidates:
        if candidate:
            return candidate.strip()
    return None


def infer_language_from_path(path: Path) -> Optional[str]:
    if path.is_dir():
        for candidate in sorted(path.rglob("*.conllu")):
            lang = _extract_language_from_file(candidate)
            if lang:
                return lang
        return _guess_language_from_name(path.name)
    lang = _extract_language_from_file(path)
    if lang:
        return lang
    return _guess_language_from_name(path.stem)


def _extract_language_from_file(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(500):
                line = handle.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = [part.strip() for part in line[1:].split("=", 1)]
                if key.lower().startswith("language"):
                    return value
    except OSError:
        return None
    return None


def _guess_language_from_name(name: str) -> Optional[str]:
    match = re.match(r"([a-z]{2,3})(?:[_-]|$)", name.lower())
    if match:
        return match.group(1)
    return None


def _extra_args_include_lang(args: List[str]) -> bool:
    for arg in args:
        if arg == "--lang" or arg.startswith("--lang="):
            return True
    return False


def detect_annotation_coverage(conllu_path: Path) -> Dict[str, bool]:
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
                    try:
                        int(head)
                        coverage["head"] = True
                    except ValueError:
                        pass
                if deprel and deprel != "_":
                    coverage["deprel"] = True

                if all(coverage[key] for key in required):
                    break
    except OSError:
        # If the file can't be read, leave coverage as False for all fields
        pass

    return coverage


def _filter_validator_output(
    stdout: str,
    stderr: str,
    returncode: int,
    coverage: Dict[str, bool],
) -> tuple[str, str, int]:
    """
    Filter validator output to remove errors about missing annotations
    that aren't present in the file.
    
    Returns (filtered_stdout, filtered_stderr, filtered_returncode).
    """
    # Patterns to match errors about missing annotations
    # These patterns match common UD validator error messages
    patterns_to_filter = []
    
    # Filter head/deprel errors if parsing info is missing
    if not coverage.get("head") or not coverage.get("deprel"):
        patterns_to_filter.extend([
            r"\[L\d+.*invalid-head\]",
            r"\[L\d+.*unknown-head\]",
            r"\[L\d+.*Invalid HEAD",
            r"\[L\d+.*Undefined HEAD",
            r"\[L\d+.*missing.*head",
            r"\[L\d+.*deprel.*error",
            r"\[L\d+.*dependency",
        ])
    
    # Filter upos/feats errors if tagging info is missing
    if not coverage.get("upos"):
        patterns_to_filter.extend([
            r"\[L\d+.*upos.*error",
            r"\[L\d+.*UPOS.*invalid",
            r"\[L\d+.*Part-of-speech",
        ])
    
    if not coverage.get("feats"):
        patterns_to_filter.extend([
            r"\[L\d+.*feats.*error",
            r"\[L\d+.*FEATS.*invalid",
            r"\[L\d+.*Morphological.*features",
        ])
    
    if not patterns_to_filter:
        # Nothing to filter
        return stdout, stderr, returncode
    
    # Compile patterns
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns_to_filter]
    
    def should_keep_line(line: str) -> bool:
        """Check if a line should be kept (not filtered)."""
        for pattern in compiled_patterns:
            if pattern.search(line):
                return False
        return True
    
    # Filter stdout and stderr
    filtered_stdout_lines = [line for line in stdout.splitlines(keepends=True) if should_keep_line(line)]
    filtered_stderr_lines = [line for line in stderr.splitlines(keepends=True) if should_keep_line(line)]
    
    # Count remaining errors by type
    format_error_count = sum(1 for line in filtered_stdout_lines + filtered_stderr_lines if "[L" in line and "Format" in line)
    syntax_error_count = sum(1 for line in filtered_stdout_lines + filtered_stderr_lines if "[L" in line and "Syntax" in line)
    
    # Rebuild output with updated error counts
    filtered_lines = []
    for line in filtered_stdout_lines + filtered_stderr_lines:
        # Replace error count lines with updated counts
        if re.match(r"Format errors?: \d+", line, re.IGNORECASE):
            if format_error_count > 0:
                filtered_lines.append(f"Format errors: {format_error_count}\n")
            # Skip if count is 0
        elif re.match(r"Syntax errors?: \d+", line, re.IGNORECASE):
            if syntax_error_count > 0:
                filtered_lines.append(f"Syntax errors: {syntax_error_count}\n")
            # Skip if count is 0
        elif "*** FAILED ***" in line:
            # Update FAILED line if there are no errors left
            if format_error_count == 0 and syntax_error_count == 0:
                filtered_lines.append("*** PASSED *** (after filtering missing annotation errors)\n")
            else:
                total_errors = format_error_count + syntax_error_count
                filtered_lines.append(f"*** FAILED *** with {total_errors} errors\n")
        else:
            filtered_lines.append(line)
    
    filtered_stdout = "".join(filtered_lines)
    filtered_stderr = ""
    
    # Determine return code
    if format_error_count == 0 and syntax_error_count == 0:
        filtered_returncode = 0
    else:
        filtered_returncode = returncode
    
    return filtered_stdout, filtered_stderr, filtered_returncode

