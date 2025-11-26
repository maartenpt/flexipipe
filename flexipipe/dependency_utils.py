from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Optional

from .model_storage import get_auto_install_extras, get_prompt_install_extras


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None  # type: ignore[attr-defined]


def _run_pip_install(extra_name: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", f"flexipipe[{extra_name}]"]
    print(f"[flexipipe] Installing optional dependency via: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        importlib.invalidate_caches()
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[flexipipe] Failed to install flexipipe[{extra_name}]: {exc}", file=sys.stderr)
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

