"""
TEITOK settings.xml parser and integration.

This module handles reading and parsing TEITOK settings.xml files to extract:
- Attribute mappings (XML tags -> internal attribute names)
- Default language settings
- CQP corpus attributes (pattributes/sattributes)
- XML file defaults
- TEI header metadata defaults

TEITOK settings.xml structure:
- /cqp: CQP corpus configuration (pattributes, sattributes)
- /xmlfiles: XML file defaults (language, etc.)
- /teiheader: Text-level metadata defaults
- /neotag/parameters/item: Tagger-specific parameters (handled by C++ code)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Standard TEITOK attribute mappings
# These are the default mappings if not specified in settings.xml
DEFAULT_ATTRIBUTE_MAPPINGS = {
    "xpos": ["xpos", "msd", "pos"],
    "reg": ["reg", "nform"],
    "expan": ["expan", "fform"],
    "lemma": ["lemma"],
    "tokid": ["id", "xml:id"],
}

# Standard CQP attributes that map to internal attributes
CQP_ATTRIBUTE_MAPPINGS = {
    "pos": "xpos",
    "msd": "xpos",
    "lemma": "lemma",
    "reg": "reg",
    "nform": "reg",
    "expan": "expan",
    "fform": "expan",
}


class TeitokSettings:
    """
    Parsed TEITOK settings.xml configuration.
    
    Attributes:
        attribute_mappings: Dict mapping internal attribute names to lists of XML attribute names
        default_language: Default language code from settings
        cqp_pattributes: CQP positional attributes (token-level)
        cqp_sattributes: CQP structural attributes (sentence/document-level)
        xmlfile_defaults: Default values for XML file processing
        teiheader_defaults: Default TEI header metadata
    """
    
    def __init__(self):
        self.attribute_mappings: Dict[str, List[str]] = dict(DEFAULT_ATTRIBUTE_MAPPINGS)
        self.default_language: Optional[str] = None
        self.cqp_pattributes: List[str] = []
        self.cqp_sattributes: List[str] = []
        self.xmlfile_defaults: Dict[str, str] = {}
        self.teiheader_defaults: Dict[str, str] = {}
        self.settings_path: Optional[Path] = None
    
    @classmethod
    def load(cls, settings_path: Path) -> TeitokSettings:
        """
        Load and parse a TEITOK settings.xml file.
        
        Args:
            settings_path: Path to settings.xml file
            
        Returns:
            Parsed TeitokSettings object
        """
        settings = cls()
        settings.settings_path = settings_path
        
        if not settings_path.exists():
            # Return default settings if file doesn't exist
            return settings
        
        try:
            tree = ET.parse(settings_path)
            root = tree.getroot()
        except ET.ParseError as e:
            # Return default settings if parsing fails
            return settings
        
        # Parse /cqp section for pattributes and sattributes
        settings._parse_cqp_section(root)
        
        # Parse /xmlfiles section for defaults
        settings._parse_xmlfiles_section(root)
        
        # Parse /teiheader section for metadata defaults
        settings._parse_teiheader_section(root)
        
        # Build attribute mappings from CQP attributes
        settings._build_attribute_mappings()
        
        return settings
    
    def _parse_cqp_section(self, root: ET.Element) -> None:
        """Parse /cqp section for pattributes and sattributes."""
        cqp_elem = root.find(".//cqp")
        if cqp_elem is None:
            return
        
        # Parse pattributes (positional attributes - token-level)
        pattributes_elem = cqp_elem.find("pattributes")
        if pattributes_elem is not None:
            for attr_elem in pattributes_elem.findall("attribute"):
                attr_name = attr_elem.get("name") or attr_elem.text
                if attr_name:
                    self.cqp_pattributes.append(attr_name.strip())
        
        # Parse sattributes (structural attributes - sentence/document-level)
        sattributes_elem = cqp_elem.find("sattributes")
        if sattributes_elem is not None:
            for attr_elem in sattributes_elem.findall("attribute"):
                attr_name = attr_elem.get("name") or attr_elem.text
                if attr_name:
                    self.cqp_sattributes.append(attr_name.strip())
    
    def _parse_xmlfiles_section(self, root: ET.Element) -> None:
        """Parse /xmlfiles section for XML file processing defaults."""
        xmlfiles_elem = root.find(".//xmlfiles")
        if xmlfiles_elem is None:
            return
        
        # Look for default language in various places
        # 1. Direct language attribute
        lang = xmlfiles_elem.get("language")
        if lang:
            self.default_language = lang
        
        # 2. Language element
        if not self.default_language:
            lang_elem = xmlfiles_elem.find("language")
            if lang_elem is not None:
                self.default_language = lang_elem.text or lang_elem.get("value")
        
        # Store other defaults
        for key, value in xmlfiles_elem.attrib.items():
            if key != "language":
                self.xmlfile_defaults[key] = value
        
        # Parse child elements as defaults
        for child in xmlfiles_elem:
            if child.tag not in ("language", "pattributes", "sattributes"):
                self.xmlfile_defaults[child.tag] = child.text or child.get("value", "")
    
    def _parse_teiheader_section(self, root: ET.Element) -> None:
        """Parse /teiheader section for TEI header metadata defaults."""
        teiheader_elem = root.find(".//teiheader")
        if teiheader_elem is None:
            return
        
        # Parse language from teiheader (can override xmlfiles)
        lang_elem = teiheader_elem.find("language")
        if lang_elem is not None:
            lang_value = lang_elem.text or lang_elem.get("value") or lang_elem.get("code")
            if lang_value:
                self.default_language = lang_value
        
        # Store other teiheader defaults
        for child in teiheader_elem:
            if child.tag != "language":
                self.teiheader_defaults[child.tag] = child.text or child.get("value", "")
    
    def _build_attribute_mappings(self) -> None:
        """
        Build attribute mappings from CQP pattributes.
        
        CQP pattributes define which XML attributes are used for token-level features.
        We map these to our internal attribute names.
        """
        # Start with defaults
        mappings = dict(DEFAULT_ATTRIBUTE_MAPPINGS)
        
        # Map CQP pattributes to internal attributes
        for pattr in self.cqp_pattributes:
            # Check if this CQP attribute maps to an internal attribute
            for cqp_name, internal_name in CQP_ATTRIBUTE_MAPPINGS.items():
                if pattr.lower() == cqp_name.lower():
                    # Add to the mapping list if not already present
                    if internal_name not in mappings:
                        mappings[internal_name] = []
                    if pattr not in mappings[internal_name]:
                        mappings[internal_name].insert(0, pattr)  # Prepend to prioritize
        
        self.attribute_mappings = mappings
    
    def get_attribute_mapping(self, internal_attr: str) -> List[str]:
        """
        Get the list of XML attribute names for an internal attribute.
        
        Args:
            internal_attr: Internal attribute name (e.g., "xpos", "reg", "lemma")
            
        Returns:
            List of XML attribute names to check (in priority order)
        """
        return self.attribute_mappings.get(internal_attr, [internal_attr])
    
    def get_language(self, override: Optional[str] = None) -> Optional[str]:
        """
        Get the default language, with optional override.
        
        Args:
            override: Language code to use if provided (takes precedence)
            
        Returns:
            Language code, or None if not set
        """
        return override or self.default_language


def find_settings_xml(corpus_path: Path) -> Optional[Path]:
    """
    Find settings.xml file in a TEITOK corpus directory.
    
    Looks for settings.xml in:
    1. Resources/settings.xml (standard TEITOK location)
    2. settings.xml (root of corpus)
    3. Parent directories up to 3 levels
    
    Args:
        corpus_path: Path to corpus directory or XML file
        
    Returns:
        Path to settings.xml if found, None otherwise
    """
    # If corpus_path is a file, use its directory
    if corpus_path.is_file():
        search_dir = corpus_path.parent
    else:
        search_dir = corpus_path
    
    # Standard locations
    candidates = [
        search_dir / "Resources" / "settings.xml",
        search_dir / "settings.xml",
    ]
    
    # Check parent directories (up to 3 levels)
    current = search_dir
    for _ in range(3):
        current = current.parent
        candidates.extend([
            current / "Resources" / "settings.xml",
            current / "settings.xml",
        ])
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    
    return None


def load_teitok_settings(settings_path: Optional[Path] = None, corpus_path: Optional[Path] = None) -> TeitokSettings:
    """
    Load TEITOK settings from a file or by searching for it.
    
    Args:
        settings_path: Explicit path to settings.xml (takes precedence)
        corpus_path: Path to corpus directory or XML file (used to search for settings.xml)
        
    Returns:
        TeitokSettings object (may have defaults if settings.xml not found)
    """
    if settings_path:
        return TeitokSettings.load(settings_path)
    
    if corpus_path:
        found_path = find_settings_xml(Path(corpus_path))
        if found_path:
            return TeitokSettings.load(found_path)
    
    # Return default settings
    return TeitokSettings()

