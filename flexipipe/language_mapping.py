"""
Comprehensive language mapping table.

This module provides a mapping of language identifiers to standardized language metadata.
It loads from a JSON file in flexipipe-models/resources/languages.json (if available),
with a fallback to a hard-coded list. This allows manual corrections for edge cases.

The mapping includes:
- ISO 639-1 (2-letter codes)
- ISO 639-2 (3-letter codes)
- ISO 639-3 (3-letter codes, more comprehensive)
- Language names (various capitalizations and variants)
- Common variants and aliases
- Non-standard codes (e.g., "old_ch" for Old Church Slavonic)

This mapping is compiled once and used for fast language matching and normalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    requests = None

# Comprehensive language mapping
# Format: (iso_639_1, iso_639_2, iso_639_3, primary_name, [variants])
_LANGUAGE_MAPPINGS: List[Tuple[str, str, str, str, List[str]]] = [
    # Format: (iso_639_1, iso_639_2, iso_639_3, primary_name, [variants])
    # Common languages with multiple identifiers
    ("af", "afr", "afr", "Afrikaans", ["afrikaans"]),
    ("ar", "ara", "ara", "Arabic", ["arabic", "العربية"]),
    ("bg", "bul", "bul", "Bulgarian", ["bulgarian", "български"]),
    ("ca", "cat", "cat", "Catalan", ["catalan", "català"]),
    ("cs", "ces", "ces", "Czech", ["czech", "čeština"]),
    ("da", "dan", "dan", "Danish", ["danish", "dansk"]),
    ("de", "deu", "deu", "German", ["german", "deutsch"]),
    ("el", "ell", "ell", "Greek", ["greek", "ελληνικά"]),
    ("en", "eng", "eng", "English", ["english"]),
    ("es", "spa", "spa", "Spanish", ["spanish", "español", "espanol"]),
    ("et", "est", "est", "Estonian", ["estonian", "eesti"]),
    ("eu", "eus", "eus", "Basque", ["basque", "euskara"]),
    ("fa", "fas", "fas", "Persian", ["persian", "farsi", "فارسی"]),
    ("fi", "fin", "fin", "Finnish", ["finnish", "suomi"]),
    ("fr", "fra", "fra", "French", ["french", "français", "francais"]),
    ("ga", "gle", "gle", "Irish", ["irish", "gaeilge"]),
    ("gl", "glg", "glg", "Galician", ["galician", "galego"]),
    ("he", "heb", "heb", "Hebrew", ["hebrew", "עברית"]),
    ("hi", "hin", "hin", "Hindi", ["hindi", "हिन्दी"]),
    ("hr", "hrv", "hrv", "Croatian", ["croatian", "hrvatski"]),
    ("hu", "hun", "hun", "Hungarian", ["hungarian", "magyar"]),
    ("id", "ind", "ind", "Indonesian", ["indonesian", "bahasa indonesia"]),
    ("it", "ita", "ita", "Italian", ["italian", "italiano"]),
    ("ja", "jpn", "jpn", "Japanese", ["japanese", "日本語"]),
    ("ko", "kor", "kor", "Korean", ["korean", "한국어"]),
    ("la", "lat", "lat", "Latin", ["latin", "latina"]),
    ("lv", "lav", "lav", "Latvian", ["latvian", "latviešu"]),
    ("nl", "nld", "nld", "Dutch", ["dutch", "nederlands"]),
    ("no", "nor", "nor", "Norwegian", ["norwegian", "norsk"]),
    ("pl", "pol", "pol", "Polish", ["polish", "polski"]),
    ("pt", "por", "por", "Portuguese", ["portuguese", "português", "portugues"]),
    ("ro", "ron", "ron", "Romanian", ["romanian", "română", "romana"]),
    ("ru", "rus", "rus", "Russian", ["russian", "русский"]),
    ("sk", "slk", "slk", "Slovak", ["slovak", "slovenčina"]),
    ("sl", "slv", "slv", "Slovenian", ["slovenian", "slovenski"]),
    ("sv", "swe", "swe", "Swedish", ["swedish", "svenska"]),
    ("tr", "tur", "tur", "Turkish", ["turkish", "türkçe", "turkce"]),
    ("uk", "ukr", "ukr", "Ukrainian", ["ukrainian", "українська"]),
    ("vi", "vie", "vie", "Vietnamese", ["vietnamese", "tiếng việt", "tieng viet"]),
    ("zh", "zho", "zho", "Chinese", ["chinese", "中文", "mandarin"]),
    # Additional languages
    ("is", "isl", "isl", "Icelandic", ["icelandic", "íslenska"]),
    ("got", "got", "got", "Gothic", ["gothic"]),
    ("sq", "sqi", "sqi", "Albanian", ["albanian", "shqip"]),
    ("hy", "hye", "hye", "Armenian", ["armenian", "հայերեն"]),
    ("be", "bel", "bel", "Belarusian", ["belarusian", "беларуская"]),
    ("bor", None, "bor", "Bororo", ["bororo"]),  # ISO 639-3 only
    ("xcl", None, "xcl", "Classical Armenian", ["classical_armenian", "classical armenian"]),  # ISO 639-3 only
    ("lzh", None, "lzh", "Classical Chinese", ["classical_chinese", "classical chinese", "文言"]),  # ISO 639-3 only
    ("cop", "cop", "cop", "Coptic", ["coptic"]),
    ("fo", "fao", "fao", "Faroese", ["faroese", "føroyskt"]),
    ("ka", "kat", "kat", "Georgian", ["georgian", "ქართული"]),
    ("hsb", None, "hsb", "Upper Sorbian", ["upper sorbian", "sorbian (upper)"]),  # ISO 639-3 only
    # Historical and non-standard languages
    ("cu", "chu", "chu", "Church Slavonic", ["church slavonic", "old church slavonic", "old_church_slavonic", "old_ch"]),
    ("orv", None, "orv", "Old East Slavic", ["old east slavic", "old_east_slavic", "old russian", "ruthenian"]),
    # Add more languages as needed
]

# Default remote language mapping URL
DEFAULT_LANGUAGE_MAPPING_URL = "https://raw.githubusercontent.com/ufal/flexipipe-models/main/resources/languages.json"

# Build lookup dictionaries for fast access
_LANGUAGE_BY_CODE: Dict[str, Dict[str, str]] = {}
_LANGUAGE_BY_NAME: Dict[str, Dict[str, str]] = {}
_LANGUAGE_MAPPINGS_LOADED = False


def _load_language_mappings_from_json(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Load language mappings from a JSON file.
    
    The JSON file should have the following structure:
    {
        "languages": [
            {
                "iso_639_1": "en",
                "iso_639_2": "eng",
                "iso_639_3": "eng",
                "primary_name": "English",
                "variants": ["english"],
                "aliases": ["en", "eng"]  // optional, additional aliases
            },
            ...
        ]
    }
    
    Args:
        url: URL to the JSON file. If None, uses the default from flexipipe-models.
        use_cache: Whether to use cached version if available.
        verbose: Whether to print debug messages.
        
    Returns:
        List of language mapping dictionaries, or None if loading fails.
    """
    from .model_storage import get_cache_dir, read_model_cache_entry, write_model_cache_entry
    
    mapping_url = url or DEFAULT_LANGUAGE_MAPPING_URL
    
    # Try to load from cache first
    if use_cache:
        cache_key = "language_mappings"
        cache_dir = get_cache_dir()
        if cache_dir:
            cached = read_model_cache_entry(cache_key, max_age_seconds=86400)  # 24 hours
            if cached and isinstance(cached, list):
                if verbose:
                    print(f"[flexipipe] Loaded {len(cached)} language mappings from cache", file=__import__("sys").stderr)
                return cached
    
    # Try to fetch from remote URL
    if mapping_url.startswith("http://") or mapping_url.startswith("https://"):
        if not requests:
            if verbose:
                print("[flexipipe] Warning: 'requests' not available, cannot fetch language mappings from remote", file=sys.stderr)
            return None
        
        try:
            response = requests.get(mapping_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and "languages" in data:
                languages = data["languages"]
            elif isinstance(data, list):
                languages = data
            else:
                if verbose:
                    print(f"[flexipipe] Warning: Invalid language mapping JSON format", file=sys.stderr)
                return None
            
            # Cache the result
            if use_cache and cache_dir:
                try:
                    write_model_cache_entry(cache_key, languages)
                except (OSError, PermissionError):
                    pass
            
            if verbose:
                print(f"[flexipipe] Loaded {len(languages)} language mappings from {mapping_url}", file=sys.stderr)
            return languages
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: Failed to load language mappings from {mapping_url}: {exc}", file=sys.stderr)
            return None
    
    # Try to load from local file
    elif mapping_url.startswith("file://"):
        file_path = Path(mapping_url[7:])
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "languages" in data:
                languages = data["languages"]
            elif isinstance(data, list):
                languages = data
            else:
                return None
            
            if verbose:
                print(f"[flexipipe] Loaded {len(languages)} language mappings from {file_path}", file=sys.stderr)
            return languages
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: Failed to load language mappings from {file_path}: {exc}", file=sys.stderr)
            return None
    
    return None


def _build_language_mappings() -> None:
    """Build lookup dictionaries from language mappings."""
    global _LANGUAGE_BY_CODE, _LANGUAGE_BY_NAME, _LANGUAGE_MAPPINGS_LOADED
    
    if _LANGUAGE_MAPPINGS_LOADED:  # Already built
        return
    
    # Try to load from JSON file first (allows manual corrections)
    json_mappings = _load_language_mappings_from_json(use_cache=True, verbose=False)
    
    # Start with hard-coded mappings as base
    mappings_to_process: List[Tuple[Optional[str], Optional[str], Optional[str], str, List[str]]] = list(_LANGUAGE_MAPPINGS)
    
    # Process JSON mappings and add/override hard-coded ones
    # JSON takes precedence for corrections
    if json_mappings:
        # Create a set of existing codes to check for duplicates
        existing_codes = set()
        for iso_1, iso_2, iso_3, _, _ in mappings_to_process:
            if iso_1:
                existing_codes.add(iso_1.lower())
            if iso_2:
                existing_codes.add(iso_2.lower())
            if iso_3:
                existing_codes.add(iso_3.lower())
        
        # Process JSON mappings
        for lang_entry in json_mappings:
            if not isinstance(lang_entry, dict):
                continue
            
            iso_1 = lang_entry.get("iso_639_1")
            iso_2 = lang_entry.get("iso_639_2")
            iso_3 = lang_entry.get("iso_639_3")
            primary_name = lang_entry.get("primary_name", "")
            variants = lang_entry.get("variants", [])
            aliases = lang_entry.get("aliases", [])
            
            # Combine variants and aliases
            all_variants = list(variants) + list(aliases)
            
            # Convert None to empty string for consistency with hard-coded format
            iso_1 = iso_1 if iso_1 else None
            iso_2 = iso_2 if iso_2 else None
            iso_3 = iso_3 if iso_3 else None
            
            # Check if this language already exists (by any of its codes)
            # If so, replace the existing entry; otherwise, add it
            lang_codes = [c.lower() for c in [iso_1, iso_2, iso_3] if c]
            found_existing = False
            for i, (existing_iso_1, existing_iso_2, existing_iso_3, _, _) in enumerate(mappings_to_process):
                existing_codes_for_entry = [c.lower() for c in [existing_iso_1, existing_iso_2, existing_iso_3] if c]
                if any(code in existing_codes_for_entry for code in lang_codes if code):
                    # Replace existing entry with JSON version
                    mappings_to_process[i] = (iso_1, iso_2, iso_3, primary_name, all_variants)
                    found_existing = True
                    break
            
            if not found_existing:
                # Add new entry
                mappings_to_process.append((iso_1, iso_2, iso_3, primary_name, all_variants))
    
    # Build lookup dictionaries
    for iso_1, iso_2, iso_3, primary_name, variants in mappings_to_process:
        lang_data = {
            "iso_639_1": iso_1,
            "iso_639_2": iso_2,
            "iso_639_3": iso_3,
            "primary_name": primary_name,
        }
        
        # Index by all codes (skip None values)
        if iso_1:
            _LANGUAGE_BY_CODE[iso_1.lower()] = lang_data
        if iso_2:
            _LANGUAGE_BY_CODE[iso_2.lower()] = lang_data
        if iso_3:
            _LANGUAGE_BY_CODE[iso_3.lower()] = lang_data
        
        # Index by primary name (various capitalizations)
        if primary_name:
            _LANGUAGE_BY_NAME[primary_name.lower()] = lang_data
            _LANGUAGE_BY_NAME[primary_name.title()] = lang_data
            _LANGUAGE_BY_NAME[primary_name.upper()] = lang_data
        
        # Index by variants
        for variant in variants:
            if variant:
                _LANGUAGE_BY_NAME[variant.lower()] = lang_data
                _LANGUAGE_BY_NAME[variant.title()] = lang_data
                _LANGUAGE_BY_NAME[variant.upper()] = lang_data
    
    _LANGUAGE_MAPPINGS_LOADED = True


def normalize_language_code(language: str) -> Tuple[str, str, str]:
    """
    Normalize a language identifier to standardized ISO codes.
    
    Args:
        language: Language identifier (code or name)
        
    Returns:
        Tuple of (iso_639_1, iso_639_2, iso_639_3) or (None, None, None) if not found
    """
    _build_language_mappings()
    
    lang_lower = language.lower().strip()
    
    # Try code lookup first
    if lang_lower in _LANGUAGE_BY_CODE:
        lang_data = _LANGUAGE_BY_CODE[lang_lower]
        return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    # Try name lookup
    if lang_lower in _LANGUAGE_BY_NAME:
        lang_data = _LANGUAGE_BY_NAME[lang_lower]
        return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    # Try partial matches (e.g., "spanish" matches "Spanish")
    for key, lang_data in _LANGUAGE_BY_NAME.items():
        if lang_lower in key or key in lang_lower:
            return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    return (None, None, None)


def get_language_metadata(language: str) -> Dict[str, str]:
    """
    Get comprehensive language metadata for a language identifier.
    
    Args:
        language: Language identifier (code or name)
        
    Returns:
        Dictionary with iso_639_1, iso_639_2, iso_639_3, primary_name, and variants
    """
    _build_language_mappings()
    
    lang_lower = language.lower().strip()
    
    # Try code lookup first
    if lang_lower in _LANGUAGE_BY_CODE:
        return _LANGUAGE_BY_CODE[lang_lower].copy()
    
    # Try name lookup
    if lang_lower in _LANGUAGE_BY_NAME:
        return _LANGUAGE_BY_NAME[lang_lower].copy()
    
    # Try partial matches
    for key, lang_data in _LANGUAGE_BY_NAME.items():
        if lang_lower in key or key in lang_lower:
            return lang_data.copy()
    
    # Return None values if not found
    return {
        "iso_639_1": None,
        "iso_639_2": None,
        "iso_639_3": None,
        "primary_name": None,
    }

