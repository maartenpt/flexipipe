"""Example data loader for flexipipe."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional

EXAMPLES_FILENAME = "examples-udhr.json"
REMOTE_URL = "https://raw.githubusercontent.com/ufal/flexipipe-models/main/resources/examples-udhr.json"


def _get_examples_dir(create: bool = True) -> Path:
    """Return the examples directory under the configured models directory."""
    from .model_storage import get_flexipipe_models_dir
    models_dir = get_flexipipe_models_dir(create=create)
    examples_dir = models_dir / "examples"
    if create:
        examples_dir.mkdir(parents=True, exist_ok=True)
    return examples_dir


def _download_examples_file(destination: Path) -> bool:
    """Download the examples file from the remote repository."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REMOTE_URL) as response:
            data = response.read().decode("utf-8")
        if not data.strip():
            # Empty response - don't write empty file
            return False
        destination.write_text(data, encoding="utf-8")
        return True
    except Exception as e:
        # Log the error for debugging
        import sys
        if sys.stderr:
            print(f"[flexipipe] Warning: Failed to download examples file: {e}", file=sys.stderr)
        return False


def load_examples(refresh: bool = False) -> Dict[str, Dict[str, str]]:
    """Load example data as a dictionary keyed by language."""
    examples_dir = _get_examples_dir(create=True)
    examples_file = examples_dir / EXAMPLES_FILENAME

    if refresh or not examples_file.exists():
        if not _download_examples_file(examples_file):
            if not examples_file.exists():
                raise FileNotFoundError(
                    f"Could not download {EXAMPLES_FILENAME} and no local file available."
                )

    with examples_file.open("r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise FileNotFoundError(
                f"Examples file {examples_file} is empty. Try deleting it and running again."
            )
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupted examples file {examples_file}: {e}. "
                f"Try deleting it and running again to re-download."
            ) from e

    normalized = {}
    for lang_code, entry in data.items():
        if "text" not in entry:
            continue
        normalized[lang_code.lower()] = entry

    return normalized


def get_example_text(language: str, refresh: bool = False) -> Optional[str]:
    """Get example text for a specific language code."""
    examples = load_examples(refresh=refresh)
    lang_lower = language.lower()
    if lang_lower in examples:
        return examples[lang_lower].get("text")
    from .language_mapping import normalize_language_code

    _, iso2, iso3 = normalize_language_code(language)
    for code in (iso2, iso3):
        if code and code.lower() in examples:
            return examples[code.lower()].get("text")
    return None

