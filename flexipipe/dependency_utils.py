from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Optional

from .model_storage import get_auto_install_extras, get_prompt_install_extras


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None  # type: ignore[attr-defined]


def _run_pip_install(extra_name: str) -> bool:
    """Install optional dependencies for a backend extra.
    
    If flexipipe is installed from PyPI, installs flexipipe[extra_name].
    If flexipipe is installed in editable mode or run from source, installs dependencies directly.
    """
    # Hardcoded mapping of extra names to their dependencies (matches setup.py)
    EXTRA_DEPENDENCIES = {
        "spacy": ["spacy>=3.7.0"],
        "stanza": ["stanza>=1.8.0"],
        "classla": ["classla>=2.1.0"],
        "flair": ["flair>=0.13.0", "torch>=2.6.0"],
        "transformers": [
            "torch>=2.6.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "scikit-learn>=1.0.0",
            "accelerate>=0.20.0",
        ],
        "nametag": ["requests>=2.31.0"],
        "udpipe": ["requests>=2.31.0"],
        "udmorph": ["requests>=2.31.0"],
    }
    
    deps = EXTRA_DEPENDENCIES.get(extra_name)
    if not deps:
        print(f"[flexipipe] Error: Unknown extra '{extra_name}'", file=sys.stderr)
        return False
    
    # Check if flexipipe is installed from PyPI (not editable/dev mode)
    # If it's an editable install or run from source, install dependencies directly
    use_extra_syntax = False
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution("flexipipe")
        # Check if it's installed in editable mode
        # Editable installs typically have the location pointing to the source directory
        # or have .egg-link files. For simplicity, check if location contains common dev paths
        location = dist.location
        if location and (
            "site-packages" in location or 
            "dist-packages" in location or
            location.endswith(".egg")
        ):
            # Likely installed from PyPI or as a regular package - try extra syntax
            use_extra_syntax = True
    except (pkg_resources.DistributionNotFound, pkg_resources.RequirementParseError, ImportError, AttributeError):
        # flexipipe is not installed as a package - install dependencies directly
        pass
    
    if use_extra_syntax:
        # Try installing via flexipipe[extra] syntax (for PyPI installations)
        cmd = [sys.executable, "-m", "pip", "install", f"flexipipe[{extra_name}]"]
        print(f"[flexipipe] Installing optional dependency via: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            return True
        except subprocess.CalledProcessError:
            # Extra syntax failed - fall back to direct dependencies
            # (might be editable install or flexipipe not on PyPI)
            pass
    
    # Install dependencies directly (works for editable installs, source runs, and as fallback)
    cmd = [sys.executable, "-m", "pip", "install"] + deps
    print(f"[flexipipe] Installing optional dependency via: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        importlib.invalidate_caches()
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[flexipipe] Failed to install dependencies for '{extra_name}': {exc}", file=sys.stderr)
        return False


def ensure_extra_installed(
    extra_name: str,
    *,
    module_name: str,
    friendly_name: str,
    allow_prompt: Optional[bool] = None,
) -> None:
    """
    Ensure that an optional extra dependency is installed.

    Args:
        extra_name: Name of the install extra (e.g. "spacy" for pip install flexipipe[spacy])
        module_name: Module that must be importable after installation
        friendly_name: Human readable backend name for messaging
        allow_prompt: Override whether prompting is allowed (defaults to stdin isatty)
    """
    if _module_available(module_name):
        return

    auto_install = get_auto_install_extras()
    prompt_install = get_prompt_install_extras()

    if allow_prompt is None:
        allow_prompt = sys.stdin.isatty()

    hint = (
        "Install it manually with: pip install \"flexipipe[{extra}]\". "
        "You can enable automatic installs via "
        "`python -m flexipipe config --set-auto-install-extras true`."
    )

    if auto_install:
        if _run_pip_install(extra_name) and _module_available(module_name):
            return
        raise ImportError(
            f"{friendly_name} backend requires optional dependency '{extra_name}', "
            f"but automatic installation failed. {hint.format(extra=extra_name)}"
        )

    if prompt_install and allow_prompt:
        answer = input(
            f"{friendly_name} backend requires optional dependency '{extra_name}'. "
            f"Install it now via pip? [Y/n]: "
        ).strip().lower()
        if answer in ("", "y", "yes"):
            if _run_pip_install(extra_name) and _module_available(module_name):
                return
            raise ImportError(
                f"{friendly_name} backend requires '{extra_name}', "
                f"but the installation attempt failed. {hint.format(extra=extra_name)}"
            )

    raise ImportError(
        f"{friendly_name} backend requires optional dependency '{extra_name}'. "
        f"{hint.format(extra=extra_name)} "
        "To continue receiving prompts, ensure "
        "`python -m flexipipe config --set-prompt-install-extras true` is enabled."
    )

