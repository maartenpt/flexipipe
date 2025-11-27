from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests

from .model_registry import fetch_remote_registry, get_registry_url
from .model_storage import get_flexipipe_models_dir

DEFAULT_UD_VALIDATOR_URLS = {
    "2.17": "https://lindat.mff.cuni.cz/repository/download/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/ud-tools-v2.17.tgz",
}


def ensure_ud_tools(
    version: Optional[str],
    *,
    url: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """
    Ensure that the UD validator tools for the requested version are available locally.
    Downloads and extracts them into <models-dir>/ud-tools/<version>/ if necessary.
    """
    resolved_version, source_entry = _resolve_ud_tools_source(version, url=url, verbose=verbose)
    version = resolved_version
    download_url = source_entry["download_url"]
    handle_id_hint = source_entry.get("handle_id")
    item_uuid_hint = source_entry.get("item_uuid")
    filename_hint = source_entry.get("filename")
    models_dir = get_flexipipe_models_dir(create=True)
    ud_tools_root = models_dir / "ud-tools" / version
    validate_script = _locate_validate_script(ud_tools_root)
    if validate_script is not None:
        return ud_tools_root


    ud_tools_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / f"ud-tools-v{version}.tgz"
        if verbose:
            print(f"[flexipipe] Downloading UD validator ({version}) from {download_url}")
        _download_file(
            download_url,
            archive_path,
            handle_id=handle_id_hint,
            item_uuid=item_uuid_hint,
            filename=filename_hint,
        )
        if verbose:
            print(f"[flexipipe] Extracting UD validator to {ud_tools_root}")
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(tmpdir)

        # Find the extracted directory (some archives include a top-level folder)
        extracted_entries = [p for p in Path(tmpdir).iterdir() if p.is_dir()]
        if len(extracted_entries) == 1:
            extracted_dir = extracted_entries[0]
        else:
            extracted_dir = Path(tmpdir)

        # Clean target dir before copying
        for child in ud_tools_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        for item in extracted_dir.iterdir():
            target = ud_tools_root / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    validate_script = _locate_validate_script(ud_tools_root)
    if validate_script is None:
        raise FileNotFoundError(
            f"Unable to locate validate.py inside UD tools version {version}. "
            "Please ensure the archive contains the validator."
        )
    return ud_tools_root


def run_ud_validator(
    treebank_path: Path,
    *,
    version: Optional[str] = None,
    url: Optional[str] = None,
    rules: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run the UD validator on the provided treebank path. Returns the completed process.
    """
    ud_tools_root = ensure_ud_tools(version, url=url, verbose=verbose)
    validate_script = _locate_validate_script(ud_tools_root)
    if validate_script is None:
        raise FileNotFoundError(
            f"validate.py not found under {ud_tools_root}. The ud-tools archive may be incomplete."
        )

    cmd = [sys.executable, str(validate_script)]
    level_arg = _normalize_validation_level(rules)
    if level_arg:
        cmd.extend(["--level", level_arg])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(treebank_path))

    if verbose:
        print(f"[flexipipe] Running UD validator: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _download_file(
    url: str,
    destination: Path,
    *,
    handle_id: Optional[str] = None,
    item_uuid: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    last_exc: Optional[Exception] = None
    try:
        _download_via_requests(url, destination)
        return
    except Exception as exc:
        last_exc = exc

    if _download_with_curl(url, destination):
        return

    if _download_from_allzip(
        url,
        destination,
        handle_id_hint=handle_id,
        item_uuid_hint=item_uuid,
        filename_hint=filename,
    ):
        return

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to download {url}")


def _download_via_requests(url: str, destination: Path) -> None:
    headers = {
        "User-Agent": "flexipipe/1.0 (+https://github.com/ufal/flexipipe)",
        "Accept": "*/*",
    }
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def _locate_validate_script(root: Path) -> Optional[Path]:
    candidates = [
        root / "validate.py",
        root / "tools" / "validate.py",
        root / "ud-tools" / "validate.py",
        root / "ud_tools" / "validate.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob("validate.py"))
    if matches:
        return matches[0]
    return None


def _download_with_curl(url: str, destination: Path) -> bool:
    """Fallback downloader that shells out to curl when requests hits a 404/403."""
    curl_path = shutil.which("curl")
    if not curl_path:
        return False
    cmd = [
        curl_path,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--output",
        str(destination),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    # Clean up partial file if curl failed
    try:
        destination.unlink()
    except FileNotFoundError:
        pass
    return False


def _resolve_ud_tools_source(
    requested_version: Optional[str],
    *,
    url: Optional[str],
    verbose: bool,
) -> tuple[str, dict]:
    """
    Determine which version/URL to use for ud-tools downloads.
    Resolution order:
      1. Explicit --ud-tools-url wins (paired with requested_version or 'custom').
      2. Remote manifest (registries/ud-tools.json) provides latest + per-version URLs.
      3. Local fallback map (DEFAULT_UD_VALIDATOR_URLS) if manifest missing or version absent.
    """
    if url:
        version = requested_version or "custom"
        entry = _populate_source_entry({"download_url": url})
        return version, entry

    manifest = fetch_remote_registry(backend="ud-tools", verbose=verbose)
    versions = manifest.get("versions") or {}
    latest = manifest.get("latest")

    version = requested_version or latest
    if version == "latest":
        version = latest

    if version and version in versions:
        entry = dict(versions[version])
        download_url = entry.get("download_url")
        if download_url:
            populated = _populate_source_entry(entry)
            return version, populated

    # Fall back to default table
    if not version and DEFAULT_UD_VALIDATOR_URLS:
        # Pick the lexicographically highest version (assuming semantic-ish versions like 2.17, 2.18, etc.)
        version = sorted(DEFAULT_UD_VALIDATOR_URLS.keys())[-1]
    if version and version in DEFAULT_UD_VALIDATOR_URLS:
        entry = _populate_source_entry({"download_url": DEFAULT_UD_VALIDATOR_URLS[version]})
        return version, entry

    # As a last resort, if we know the manifest URL but not the version, raise a helpful error
    manifest_url = get_registry_url("ud-tools")
    raise ValueError(
        "Unable to resolve UD validator download URL. "
        f"Requested version: '{requested_version or 'latest'}'. "
        f"Checked manifest at {manifest_url}. "
        "Provide --ud-tools-url or update flexipipe-models/registries/ud-tools.json."
    )


def _populate_source_entry(entry: dict) -> dict:
    download_url = entry.get("download_url")
    if not download_url:
        return entry
    parsed = urlparse(download_url)
    path = parsed.path
    filename = Path(path).name
    if filename:
        entry.setdefault("filename", filename)
    item_uuid = _extract_uuid_from_download_path(path)
    if item_uuid:
        entry.setdefault("item_uuid", item_uuid)
    return entry


def _normalize_validation_level(rules: Optional[str]) -> Optional[str]:
    if not rules:
        return None
    normalized = str(rules).strip()
    if not normalized:
        return None
    LEVEL_MAP = {
        "basic": "1",
        "format": "2",
        "core": "3",
        "content": "3",
        "lang": "4",
        "language": "4",
        "labels": "4",
        "all": "5",
        "full": "5",
    }
    lower = normalized.lower()
    if lower in LEVEL_MAP:
        return LEVEL_MAP[lower]
    if normalized.isdigit():
        return normalized
    return None


def _download_from_allzip(
    original_url: str,
    destination: Path,
    *,
    handle_id_hint: Optional[str] = None,
    item_uuid_hint: Optional[str] = None,
    filename_hint: Optional[str] = None,
) -> bool:
    parsed = urlparse(original_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    item_uuid = item_uuid_hint or _extract_uuid_from_download_path(parsed.path)
    if not item_uuid:
        return False

    handle_id = handle_id_hint or _fetch_handle_id(base_url, item_uuid)
    if not handle_id:
        return False

    filename = filename_hint or Path(parsed.path).name
    if not filename:
        return False

    allzip_url = f"{base_url}/repository/server/api/core/items/{item_uuid}/allzip?handleId={handle_id}"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            _download_via_requests(allzip_url, tmp_path)
        except Exception:
            if not _download_with_curl(allzip_url, tmp_path):
                return False

        with zipfile.ZipFile(tmp_path, "r") as zip_handle:
            member_name = _find_zip_member(zip_handle, filename)
            if not member_name:
                return False
            with zip_handle.open(member_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def _fetch_handle_id(base_url: str, item_uuid: str) -> Optional[str]:
    api_url = f"{base_url}/repository/server/api/core/items/{item_uuid}"
    headers = {
        "User-Agent": "flexipipe/1.0 (+https://github.com/ufal/flexipipe)",
        "Accept": "application/json",
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("handle")
    except Exception:
        return None


def _extract_uuid_from_download_path(path: str) -> Optional[str]:
    parts = [part for part in path.split("/") if part]
    for idx, part in enumerate(parts):
        if part == "download" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _find_zip_member(zip_handle: zipfile.ZipFile, target_name: str) -> Optional[str]:
    normalized_target = target_name.strip("/")
    for name in zip_handle.namelist():
        if name.rstrip("/").endswith(normalized_target):
            return name
    return None


