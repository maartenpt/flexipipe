from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .teitok_settings import TeitokSettings

from .doc import Document, Sentence, SubToken, Token, Entity
from .doc_utils import collect_span_entities_by_sentence
from .unicode_utils import normalize_unicode

# Import helper functions from engine
from .engine import (
    _apply_sentence_correspondence,
    _ensure_serializable,
    _infer_space_after_from_text,
    _sanitize_document,
    assign_doc_id_from_path,
)

XML_NS = "{http://www.w3.org/XML/1998/namespace}"


def _normalize_annotation_attr(value: str, unicode_normalization: Optional[str] = None) -> str:
    """
    Normalize an annotation attribute value (form, lemma, etc.) for XML output.
    
    This normalizes annotation attributes but NOT innerText content, which should
    preserve the original encoding differences (NFC vs NFD).
    
    Args:
        value: The attribute value to normalize
        unicode_normalization: Normalization form ("NFC", "NFD", or None/"none")
        
    Returns:
        Normalized value, or original if normalization is None/"none"
    """
    if not value or not unicode_normalization or unicode_normalization == "none":
        return value
    return normalize_unicode(value, unicode_normalization) or value

# Import C++ bindings for fast TEITOK I/O
try:
    from flexitag_py import (
        load_teitok as _load_teitok,
        save_teitok as _save_teitok,
        dump_teitok as _dump_teitok,
    )
except ImportError:  # pragma: no cover - handled during runtime
    # Try adding the build directory to sys.path
    import sys
    from pathlib import Path
    import os
    import importlib
    
    # Look for flexitag_py in multiple locations:
    # 1. In flexipipe package directory (when installed from wheel)
    # 2. In flexitag/build directory (development/editable install)
    resolved_file_path = Path(__file__).resolve()
    flexipipe_package_dir = resolved_file_path.parent  # flexipipe/ directory
    flexipipe_dir = resolved_file_path.parent.parent   # parent of flexipipe/
    
    # Check flexipipe package directory first (for wheel installations)
    flexitag_py_in_package = list(flexipipe_package_dir.glob("flexitag_py*.so")) + list(flexipipe_package_dir.glob("flexitag_py*.pyd"))
    
    # Check flexitag/build directory (for development installations)
    flexitag_build = flexipipe_dir / "flexitag" / "build"
    
    _import_error = None
    
    # Try importing from flexipipe package directory first
    if flexitag_py_in_package:
        try:
            # The module is in the same directory as this file, so it should be importable
            # But we need to make sure the directory is in sys.path
            package_path_str = str(flexipipe_package_dir.resolve())
            if package_path_str not in sys.path:
                sys.path.insert(0, package_path_str)
            
            # Remove flexitag_py from sys.modules if it exists (to force re-import)
            if 'flexitag_py' in sys.modules:
                del sys.modules['flexitag_py']
            
            flexitag_py_module = importlib.import_module('flexitag_py')
            _load_teitok = flexitag_py_module.load_teitok
            _save_teitok = flexitag_py_module.save_teitok
            _dump_teitok = flexitag_py_module.dump_teitok
            _import_error = None  # Success
        except (ImportError, AttributeError) as exc:
            _import_error = exc
            # Fall through to try flexitag/build directory
    
    # Fall back to flexitag/build directory if not found in package
    if _import_error is not None and flexitag_build.exists():
        build_path_str = str(flexitag_build.resolve())
        if build_path_str not in sys.path:
            sys.path.insert(0, build_path_str)
        
        # Check for Python version mismatch
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        so_files = [f for f in os.listdir(build_path_str) if f.startswith('flexitag_py') and f.endswith('.so')]
        version_mismatch = False
        if so_files:
            # Extract Python version from .so filename (e.g., cpython-311-darwin.so -> 3.11)
            so_file = so_files[0]
            if f"cpython-{python_version.replace('.', '')}" not in so_file:
                version_mismatch = True
                # Try to extract version from filename
                import re
                match = re.search(r'cpython-(\d)(\d+)', so_file)
                if match:
                    so_python_version = f"{match.group(1)}.{match.group(2)}"
                else:
                    so_python_version = "unknown"
        
        # Remove flexitag_py from sys.modules if it exists (to force re-import)
        if 'flexitag_py' in sys.modules:
            del sys.modules['flexitag_py']
        
        try:
            # Use importlib to force a fresh import
            flexitag_py_module = importlib.import_module('flexitag_py')
            _load_teitok = flexitag_py_module.load_teitok
            _save_teitok = flexitag_py_module.save_teitok
            _dump_teitok = flexitag_py_module.dump_teitok
            _import_error = None  # Success
        except (ImportError, AttributeError) as exc:
            _import_error = exc
    
    # If still not found, set up fallback functions
    if _import_error is not None:
        # Check if we have version mismatch info from the flexitag/build check
        version_mismatch = False
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        so_python_version = None
        if flexitag_build.exists():
            build_path_str = str(flexitag_build.resolve())
            so_files = [f for f in os.listdir(build_path_str) if f.startswith('flexitag_py') and f.endswith('.so')]
            if so_files:
                so_file = so_files[0]
                if f"cpython-{python_version.replace('.', '')}" not in so_file:
                    version_mismatch = True
                    import re
                    match = re.search(r'cpython-(\d)(\d+)', so_file)
                    if match:
                        so_python_version = f"{match.group(1)}.{match.group(2)}"
        
        def _missing_extension(*_: object, **__: object) -> None:
            error_msg = (
                "flexitag_py extension is not available. "
                "Build the flexitag project with pybind11 support or ensure it is on PYTHONPATH."
            )
            if version_mismatch and so_python_version:
                error_msg += (
                    f"\n\nPython version mismatch detected: "
                    f"running Python {python_version} but module is compiled for Python {so_python_version}. "
                    f"Please rebuild the flexitag module for Python {python_version} by running: "
                    f"cd flexitag/build && cmake .. && make flexitag_py"
                )
            raise RuntimeError(error_msg) from _import_error

        _load_teitok = _missing_extension  # type: ignore
        _save_teitok = _missing_extension  # type: ignore
        _dump_teitok = _missing_extension  # type: ignore
    else:
        _import_error = ImportError("flexitag_py extension not found")

        def _missing_extension(*_: object, **__: object) -> None:
            raise RuntimeError(
                "flexitag_py extension is not available. Build the flexitag project with "
                "pybind11 support or ensure it is on PYTHONPATH."
            ) from _import_error

        _load_teitok = _missing_extension  # type: ignore
        _save_teitok = _missing_extension  # type: ignore
        _dump_teitok = None  # Will be set to Python fallback below

# Python fallback for _dump_teitok when C++ extension is not available
def _dump_teitok_python(payload: Dict[str, Any], custom_attributes: List[str], pretty_print: bool = False, *, unicode_normalization: Optional[str] = None) -> str:
    """
    Python fallback implementation of dump_teitok.
    
    Converts a document dict to TEITOK XML string.
    
    Args:
        payload: Document dict (from Document.to_dict())
        custom_attributes: List of custom attribute names to include from token.attrs
        pretty_print: If True, pretty-print the XML (not used in fallback, handled by caller)
    
    Returns:
        XML string representation
    """
    from .doc import Document
    
    # Reconstruct Document from dict
    doc = Document.from_dict(payload)
    
    # Build XML using ElementTree
    # Don't set xmlns to avoid namespace prefixes (ns0:, etc.)
    root = ET.Element("TEI")
    
    # Create text element
    text_elem = ET.SubElement(root, "text")
    body_elem = ET.SubElement(text_elem, "body")
    
    # Process sentences
    for sent in doc.sentences:
        s_elem = ET.SubElement(body_elem, "s")
        if sent.id:
            s_elem.set("id", str(sent.id))
        
        # Process entities (name elements) - group tokens by entity
        entity_tokens: Dict[int, List[Token]] = {}  # entity index -> tokens
        for ent in sent.entities:
            if ent.start not in entity_tokens:
                entity_tokens[ent.start] = []
            # Find tokens in this entity range
            for tok in sent.tokens:
                if ent.start <= tok.id <= ent.end:
                    entity_tokens.setdefault(ent.start, []).append(tok)
        
        # Process tokens
        current_entity_start = None
        name_elem = None
        for tok in sent.tokens:
            # Check if this token starts a new entity
            if tok.id in entity_tokens:
                if current_entity_start is not None and name_elem is not None:
                    # Close previous entity
                    current_entity_start = None
                    name_elem = None
                if tok.id in entity_tokens:
                    # Start new entity
                    current_entity_start = tok.id
                    ent = next((e for e in sent.entities if e.start == tok.id), None)
                    if ent:
                        name_elem = ET.SubElement(s_elem, "name")
                        name_elem.set("type", ent.label)
                        if ent.text:
                            name_elem.set("text", ent.text)
                        # Add entity attributes
                        for key, value in ent.attrs.items():
                            name_elem.set(key, str(value))
            
            # Create tok element (inside name if in entity)
            parent = name_elem if name_elem is not None else s_elem
            tok_elem = ET.SubElement(parent, "tok")
            
            # Set text content to token form (innerText) - DO NOT normalize, preserve original encoding
            if tok.form:
                tok_elem.text = tok.form
            
            # Add standard attributes - normalize annotation attributes but NOT innerText
            if tok.form:
                tok_elem.set("form", _normalize_annotation_attr(tok.form, unicode_normalization))
            if tok.lemma:
                tok_elem.set("lemma", _normalize_annotation_attr(tok.lemma, unicode_normalization))
            if tok.xpos:
                tok_elem.set("xpos", tok.xpos)  # xpos/upos are tags, not text
            if tok.upos:
                tok_elem.set("upos", tok.upos)
            if tok.feats:
                tok_elem.set("feats", tok.feats)  # FEATS are tags, not text
            if tok.reg:
                tok_elem.set("reg", _normalize_annotation_attr(tok.reg, unicode_normalization))
            if tok.expan:
                tok_elem.set("expan", _normalize_annotation_attr(tok.expan, unicode_normalization))
            if tok.mod:
                tok_elem.set("mod", _normalize_annotation_attr(tok.mod, unicode_normalization))
            if tok.trslit:
                tok_elem.set("trslit", _normalize_annotation_attr(tok.trslit, unicode_normalization))
            if tok.ltrslit:
                tok_elem.set("ltrslit", _normalize_annotation_attr(tok.ltrslit, unicode_normalization))
            if tok.tokid:
                tok_elem.set("tokid", tok.tokid)
            if tok.id:
                tok_elem.set("id", str(tok.id))
            if tok.head:
                tok_elem.set("head", str(tok.head))
            if tok.deprel:
                tok_elem.set("deprel", tok.deprel)
            if tok.deps:
                tok_elem.set("deps", tok.deps)
            if tok.misc:
                tok_elem.set("misc", tok.misc)
            
            # Add custom attributes from token.attrs
            if tok.attrs:
                if custom_attributes:
                    # Only include specified custom attributes
                    for attr_name in custom_attributes:
                        if attr_name in tok.attrs:
                            tok_elem.set(attr_name, str(tok.attrs[attr_name]))
                else:
                    # Include all custom attributes
                    for attr_name, attr_value in tok.attrs.items():
                        tok_elem.set(attr_name, str(attr_value))
            
            # Add subtokens (dtok elements)
            if tok.subtokens:
                for st in tok.subtokens:
                    dtok_elem = ET.SubElement(tok_elem, "dtok")
                    if st.form:
                        dtok_elem.set("form", _normalize_annotation_attr(st.form, unicode_normalization))
                    if st.lemma:
                        dtok_elem.set("lemma", _normalize_annotation_attr(st.lemma, unicode_normalization))
                    if st.xpos:
                        dtok_elem.set("xpos", st.xpos)  # xpos/upos are tags, not text
                    if st.upos:
                        dtok_elem.set("upos", st.upos)
                    if st.feats:
                        dtok_elem.set("feats", st.feats)  # FEATS are tags, not text
                    if st.reg:
                        dtok_elem.set("reg", _normalize_annotation_attr(st.reg, unicode_normalization))
                    if st.expan:
                        dtok_elem.set("expan", _normalize_annotation_attr(st.expan, unicode_normalization))
                    # Add subtoken custom attributes
                    if st.attrs:
                        if custom_attributes:
                            for attr_name in custom_attributes:
                                if attr_name in st.attrs:
                                    dtok_elem.set(attr_name, str(st.attrs[attr_name]))
                        else:
                            for attr_name, attr_value in st.attrs.items():
                                dtok_elem.set(attr_name, str(attr_value))
            
            # Add space after token if space_after is True (matching C++ behavior)
            # Only add space if space_after is explicitly True (not False or None)
            if tok.space_after is True:
                # Add space as tail text on the tok element
                # This will appear after the closing </tok> tag
                tok_elem.tail = " "
            
            # Check if this token ends the current entity
            if current_entity_start is not None and name_elem is not None:
                ent = next((e for e in sent.entities if e.start == current_entity_start), None)
                if ent and tok.id >= ent.end:
                    # Close entity
                    current_entity_start = None
                    name_elem = None
    
    # Convert to string
    # Don't use indent as it may add namespace prefixes
    # Serialize without namespace prefixes by not setting xmlns
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    # Remove any namespace prefixes that might have been added (ns0:, etc.)
    import re
    xml_str = re.sub(r'<ns\d+:', '<', xml_str)
    xml_str = re.sub(r'</ns\d+:', '</', xml_str)
    return xml_str

# Set Python fallback if C++ extension is not available
if _dump_teitok is None:
    _dump_teitok = _dump_teitok_python


def load_teitok(
    path: str,
    *,
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
) -> Document:
    """
    Load a TEITOK XML file.
    
    Args:
        path: Path to the TEITOK XML file
        xpos_attr: Comma-separated attribute names to try for xpos (e.g., "pos,msd")
        reg_attr: Comma-separated attribute names to try for reg (e.g., "nform,fform")
        expan_attr: Comma-separated attribute names to try for expan (e.g., "fform")
        lemma_attr: Comma-separated attribute names to try for lemma
    
    Returns:
        Document object
    """
    # Always fix duplicate IDs first (before loading)
    # This ensures the backend receives deduplicated IDs
    had_duplicates = False
    temp_path = None
    try:
        # Handle files with debug output at the start (find first XML tag)
        # Note: We let the XML parser handle the declared encoding.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            xml_start = content.find("<")
        if xml_start >= 0:
            from io import StringIO
            tree = ET.parse(StringIO(content[xml_start:]))
            root = tree.getroot()
            # Check for duplicates before fixing
            seen_sent_ids: Dict[str, int] = {}
            seen_tokids: Dict[str, int] = {}
            for s_node in root.iter():
                if s_node.tag.endswith("}s") or s_node.tag == "s":
                    sent_id = s_node.get("id") or s_node.get("{http://www.w3.org/XML/1998/namespace}id")
                    if sent_id:
                        seen_sent_ids[sent_id] = seen_sent_ids.get(sent_id, 0) + 1
            for tok_node in root.iter():
                if tok_node.tag.endswith("}tok") or tok_node.tag == "tok":
                    tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                    if tokid:
                        seen_tokids[tokid] = seen_tokids.get(tokid, 0) + 1
            
            # Check if there are duplicates
                has_duplicates = any(count > 1 for count in seen_sent_ids.values()) or any(count > 1 for count in seen_tokids.values())
                
                if has_duplicates:
                    had_duplicates = True
                    _fix_duplicate_ids(root)
                    # Write fixed XML back to a temp file
                    # (We'll use this temp file for loading)
                    import tempfile
                    # Include XML declaration
                    fixed_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
                    # Write to temp file
                    temp_fd, temp_path = tempfile.mkstemp(suffix=".xml", text=True)
                    try:
                        with os.fdopen(temp_fd, "w", encoding="utf-8") as tf:
                            tf.write(fixed_content)
                    except Exception:
                        os.close(temp_fd)
                        temp_path = None
    except Exception:
        # If fixing fails, just use original file
        pass
    
    load_path = temp_path if temp_path and had_duplicates else path
    
    # If attribute mappings are provided OR we fixed duplicates, use Python-based loading
    # (Python loader can handle the fixed XML, C++ loader might not)
    if xpos_attr or reg_attr or expan_attr or lemma_attr or had_duplicates:
        doc = _load_teitok_with_mappings(
            load_path,
            xpos_attr=xpos_attr,
            reg_attr=reg_attr,
            expan_attr=expan_attr,
            lemma_attr=lemma_attr,
        )
    else:
        # Otherwise, use the fast C++ loader
        data = _load_teitok(load_path)  # type: ignore
        doc = Document.from_dict(data)
        _apply_sentence_correspondence(doc)
        # Fix space_after values by inferring from sentence text
        _infer_space_after_from_text(doc)
        doc = _sanitize_document(doc)
    
    # Clean up temp file if we created one
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    
    assign_doc_id_from_path(doc, path)
    doc.meta.setdefault("source_path", path)
    for sentence in doc.sentences:
        if sentence.id and not getattr(sentence, "source_id", ""):
            sentence.source_id = sentence.id
    return doc


def _get_attr_value_with_fallback(
    elem: ET.Element,
    attr_names: List[str],
    fallback_to_text: bool = False,
) -> str:
    """
    Get attribute value, trying multiple names in order, optionally falling back to element text.
    
    Args:
        elem: XML element
        attr_names: List of attribute names to try (in order)
        fallback_to_text: If True, fall back to element text if no attribute found
    
    Returns:
        Attribute value, or empty string if not found
    """
    for attr_name in attr_names:
        value = elem.get(attr_name)
        if value and value != "--":  # Filter out "--" (reserved value in TEITOK)
            return value
    
    if fallback_to_text:
        text = (elem.text or "").strip()
        if text and text != "--":
            return text
    
    return ""


def _build_parent_map(elem: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> None:
    """Build a map of element -> parent for all elements in the tree."""
    for child in elem:
        parent_map[child] = elem
        _build_parent_map(child, parent_map)


def teitok_has_tokens(path: str) -> bool:
    """
    Check if a TEITOK XML file contains any <tok> elements.
    
    Args:
        path: Path to the TEITOK XML file
    
    Returns:
        True if the file contains at least one <tok> element, False otherwise
    """
    try:
        # Try to find XML content (skip any leading non-XML lines like debug output)
        # Note: We let the XML parser handle the declared encoding.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Find the first XML tag
            xml_start = content.find("<")
        if xml_start < 0:
            return False
        # Try parsing from the XML start
        from io import StringIO
        tree = ET.parse(StringIO(content[xml_start:]))
        root = tree.getroot()
        # Check for any tok elements (handle namespaces)
        toks = root.findall(".//{*}tok") or root.findall(".//tok")
        return len(toks) > 0
    except (ET.ParseError, OSError, ValueError):
        # Fallback: simple text search for <tok
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return "<tok" in content or "<tok " in content or "<tok>" in content
        except OSError:
            return False


def extract_teitok_plain_text(path: str, textnode_xpath: str = ".//text", include_notes: bool = False) -> str:
    """
    Extract plain text from a TEITOK XML file using an XPath expression.
    
    This function extracts all text content from the nodes matching the XPath,
    preserving whitespace as it appears in the XML structure.
    If <p> or <div> elements are present, double newlines are added before and after each.
    By default, <note> elements are excluded from the extracted text.
    
    Args:
        path: Path to the TEITOK XML file
        textnode_xpath: XPath expression to locate the text node (default: ".//text")
        include_notes: If True, include <note> elements in extracted text (default: False)
    
    Returns:
        Plain text content as a string
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Helper function to check if an element is a note
        def is_note(elem: ET.Element) -> bool:
            """Check if element is a <note> element (handles namespaces)."""
            return (elem.tag.endswith("}note") or elem.tag == "note")
        
        # Find nodes matching the XPath
        # Handle both namespaced and non-namespaced elements
        try:
            # Try with namespaces first
            nodes = root.findall(textnode_xpath.replace(".//", ".//{*}"))
            if not nodes:
                # Fall back to non-namespaced
                nodes = root.findall(textnode_xpath)
        except (SyntaxError, ValueError):
            # If XPath is invalid, try simple findall
            nodes = root.findall(textnode_xpath)
        
        if not nodes:
            # If no nodes found, try the root itself
            nodes = [root]
        
        # Check if there are any <p> or <div> elements
        has_block_elements = False
        for node in nodes:
            # Check for <p> and <div> elements (handle namespaces)
            p_elems = node.findall(".//{*}p") or node.findall(".//p")
            div_elems = node.findall(".//{*}div") or node.findall(".//div")
            if p_elems or div_elems:
                has_block_elements = True
                break
        
        # Custom text extraction function that skips note elements but preserves their tail
        def extract_text_skipping_notes(elem: ET.Element) -> str:
            """Extract text from element, skipping note elements but preserving their tail text."""
            parts = []
            # Add element's own text
            if elem.text:
                parts.append(elem.text)
            
            # Process children
            for child in elem:
                if not include_notes and is_note(child):
                    # Skip note element, but preserve its tail text
                    if child.tail:
                        parts.append(child.tail)
                else:
                    # Recursively extract text from child
                    child_text = extract_text_skipping_notes(child)
                    if child_text:
                        parts.append(child_text)
                    # Add tail text (text after the element)
                    if child.tail:
                        parts.append(child.tail)
            
            return "".join(parts)
        
        # Extract text from all matching nodes
        text_parts = []
        for node in nodes:
            if has_block_elements:
                # Process node with special handling for <p> and <div> elements
                parts = []
                # Process direct children and text nodes
                if node.text:
                    parts.append(node.text)
                
                for child in node:
                    # Check if this is a <p> or <div> element
                    is_p = (child.tag.endswith("}p") or child.tag == "p")
                    is_div = (child.tag.endswith("}div") or child.tag == "div")
                    if is_p or is_div:
                        # Add double newline before block element content
                        parts.append("\n\n")
                        # Get text from block element, skipping notes
                        block_text = extract_text_skipping_notes(child)
                        parts.append(block_text)
                        # Add double newline after block element content
                        parts.append("\n\n")
                    else:
                        # For non-block elements, extract text skipping notes
                        child_text = extract_text_skipping_notes(child)
                        if child_text:
                            parts.append(child_text)
                    # Add tail text (text after the element)
                    if child.tail:
                        parts.append(child.tail)
                
                text = "".join(parts)
            else:
                # No block elements, use custom text extraction that skips notes
                text = extract_text_skipping_notes(node)
            
            if text:
                text_parts.append(text)
        
        result = "".join(text_parts).strip()
        # Clean up multiple consecutive newlines (more than 2) to exactly 2
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result
    except (ET.ParseError, OSError) as e:
        raise ValueError(f"Failed to extract text from TEITOK file {path}: {e}") from e


def _fix_duplicate_ids(root: ET.Element) -> None:
    """
    Fix duplicate sentence and token IDs in the XML tree by appending suffixes.
    
    This ensures that all IDs are unique, which is required for valid XML.
    Duplicate IDs are renamed with suffixes: s1 -> s1-2, s1-3, etc.
    
    Args:
        root: Root XML element
    """
    # Fix duplicate sentence IDs
    all_sentences: List[ET.Element] = []
    seen_sent_ids: Dict[str, int] = {}
    
    # First pass: collect all sentences and detect duplicates
    for s_node in root.iter():
        if s_node.tag.endswith("}s") or s_node.tag == "s":
            all_sentences.append(s_node)
            sent_id = s_node.get("id") or s_node.get("{http://www.w3.org/XML/1998/namespace}id")
            if sent_id:
                seen_sent_ids[sent_id] = seen_sent_ids.get(sent_id, 0) + 1
    
    # Second pass: fix duplicate sentence IDs
    sent_id_counts: Dict[str, int] = {}
    for s_node in all_sentences:
        sent_id = s_node.get("id") or s_node.get("{http://www.w3.org/XML/1998/namespace}id")
        if sent_id and seen_sent_ids.get(sent_id, 0) > 1:
            # This is a duplicate - need to fix it
            sent_id_counts[sent_id] = sent_id_counts.get(sent_id, 0) + 1
            if sent_id_counts[sent_id] > 1:
                # Append suffix for duplicates
                new_sent_id = f"{sent_id}-{sent_id_counts[sent_id]}"
                # Remove old id attribute (could be id or xml:id)
                if s_node.get("id"):
                    s_node.attrib.pop("id")
                if s_node.get("{http://www.w3.org/XML/1998/namespace}id"):
                    s_node.attrib.pop("{http://www.w3.org/XML/1998/namespace}id")
                # Set new id
                s_node.set("id", new_sent_id)
    
    # Fix duplicate token IDs
    all_tok_nodes: List[ET.Element] = []
    seen_tokids: Dict[str, int] = {}
    
    # First pass: collect all tokens and detect duplicates
    for tok_node in root.iter():
        if tok_node.tag.endswith("}tok") or tok_node.tag == "tok":
            all_tok_nodes.append(tok_node)
            tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
            if tokid:
                seen_tokids[tokid] = seen_tokids.get(tokid, 0) + 1
    
    # Second pass: fix duplicate token IDs
    tokid_counts: Dict[str, int] = {}
    for tok_node in all_tok_nodes:
        tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
        if tokid and seen_tokids.get(tokid, 0) > 1:
            # This is a duplicate - need to fix it
            tokid_counts[tokid] = tokid_counts.get(tokid, 0) + 1
            if tokid_counts[tokid] > 1:
                # Append suffix for duplicates (w-1 -> w-1-2, w-1-3, etc.)
                new_tokid = f"{tokid}-{tokid_counts[tokid]}"
                # Remove old id attribute (could be id or xml:id)
                if tok_node.get("id"):
                    tok_node.attrib.pop("id")
                if tok_node.get("{http://www.w3.org/XML/1998/namespace}id"):
                    tok_node.attrib.pop("{http://www.w3.org/XML/1998/namespace}id")
                # Set new id
                tok_node.set("id", new_tokid)


def _load_teitok_with_mappings(
    path: str,
    *,
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
) -> Document:
    """
    Load TEITOK XML with custom attribute mappings.
    
    This is a Python-based implementation that supports comma-separated attribute lists
    and fallback to form (innerText) for reg/expan.
    """
    # Handle files with debug output at the start (find first XML tag)
    # Note: We let the XML parser handle the declared encoding.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        xml_start = content.find("<")
    if xml_start < 0:
        raise ValueError(f"Invalid XML file {path}: No XML content found")
    # Parse from the XML start
    from io import StringIO
    try:
        tree = ET.parse(StringIO(content[xml_start:]))
    except ET.ParseError as e:
        # Provide helpful error message with file path and line number if available
        error_msg = f"Invalid XML in file {path}"
        if hasattr(e, 'position') and e.position:
            line_num = content[:xml_start + e.position[0]].count('\n') + 1
            col_num = e.position[1] if len(e.position) > 1 else 0
            error_msg += f" at line {line_num}, column {col_num}"
        error_msg += f": {str(e)}"
        raise ValueError(error_msg) from e
    root = tree.getroot()
    
    # Fix duplicate IDs before processing
    _fix_duplicate_ids(root)
    
    # Parse comma-separated attribute lists
    xpos_attrs = [a.strip() for a in xpos_attr.split(",")] if xpos_attr else []
    reg_attrs = [a.strip() for a in reg_attr.split(",")] if reg_attr else []
    expan_attrs = [a.strip() for a in expan_attr.split(",")] if expan_attr else []
    lemma_attrs = [a.strip() for a in lemma_attr.split(",")] if lemma_attr else []
    
    # Add defaults
    if xpos_attrs:
        xpos_attrs.extend(["xpos", "msd", "pos"])
    else:
        xpos_attrs = ["xpos", "msd", "pos"]
    
    if reg_attrs:
        reg_attrs.extend(["reg", "nform"])
    else:
        reg_attrs = ["reg", "nform"]
    
    if expan_attrs:
        expan_attrs.extend(["expan", "fform"])
    else:
        expan_attrs = ["expan", "fform"]
    
    if lemma_attrs:
        lemma_attrs.extend(["lemma"])
    else:
        lemma_attrs = ["lemma"]
    
    tei_attrs: Dict[str, str] = {}
    text_elem = root.find(".//text")
    doc_id = ""
    if text_elem is not None:
        doc_id = text_elem.get("id") or text_elem.get(f"{XML_NS}id", "") or ""
        for key, value in text_elem.attrib.items():
            if value:
                tei_attrs[key] = value
        xml_text_id = text_elem.get(f"{XML_NS}id")
        if xml_text_id:
            tei_attrs["xml:id"] = xml_text_id
    else:
        doc_id = root.get("id") or root.get(f"{XML_NS}id", "") or ""
    document = Document(id=doc_id, attrs=tei_attrs, meta={"source_path": path})
    assign_doc_id_from_path(document, path)
    
    # Build parent map for the entire tree
    parent_map: Dict[ET.Element, ET.Element] = {}
    _build_parent_map(root, parent_map)
    
    # Find all sentence elements
    sentence_counter = 1
    for s_elem in root.findall(".//s"):
        xml_sent_id = s_elem.get("id") or s_elem.get(f"{XML_NS}id", "") or ""
        sent_id_attr = s_elem.get("sent_id", "")
        assigned_id = xml_sent_id or sent_id_attr
        if not assigned_id:
            assigned_id = f"s-{sentence_counter}"
        # Store all attributes from <s> element in attrs (except those we use directly)
        s_attrs: Dict[str, str] = {}
        for key, value in s_elem.attrib.items():
            # Skip attributes we use directly (id, sent_id, text, corr)
            if key not in ("id", "sent_id", "text", "corr", f"{XML_NS}id"):
                s_attrs[key] = value
        
        sentence = Sentence(
            id=assigned_id,
            sent_id=sent_id_attr,
            source_id=xml_sent_id or sent_id_attr,
            text=s_elem.get("text", ""),
            tokens=[],
            attrs=s_attrs,
        )
        
        token_id = 1
        # Get all token elements recursively (including those inside <name> elements)
        # We need to find all tokens within the sentence, not just direct children
        tok_elements = s_elem.findall(".//tok")
        
        # Build a map of token elements to their containing <name> elements
        # This will help us extract entity information
        tok_to_name: Dict[ET.Element, ET.Element] = {}
        for tok_elem in tok_elements:
            # Walk up the parent chain to find enclosing <name> element
            parent = parent_map.get(tok_elem)
            while parent is not None:
                if parent.tag == "name" or (hasattr(parent.tag, '__name__') and 'name' in str(parent.tag).lower()):
                    tok_to_name[tok_elem] = parent
                    break
                parent = parent_map.get(parent)
        
        # Group tokens by their containing <name> element
        # This will help us create Entity objects
        name_to_tokens: Dict[ET.Element, List[ET.Element]] = {}
        for tok_elem, name_elem in tok_to_name.items():
            if name_elem not in name_to_tokens:
                name_to_tokens[name_elem] = []
            name_to_tokens[name_elem].append(tok_elem)
        
        # Also get all children for spacing detection
        all_children = list(s_elem)
        
        # Map token elements to their index in tok_elements (for entity creation)
        tok_elem_to_index: Dict[ET.Element, int] = {tok: idx for idx, tok in enumerate(tok_elements)}
        
        for tok_idx, tok_elem in enumerate(tok_elements):
            # Check if @form attribute is "--" (suppressed token) - skip it
            # Note: We only check the attribute, not innerText
            form_attr = tok_elem.get("form", "")
            if form_attr == "--":
                continue  # Skip suppressed tokens
            
            # Parse token ID
            tok_id_str = tok_elem.get("id", "")
            if tok_id_str:
                try:
                    dash_idx = tok_id_str.rfind("-")
                    if dash_idx >= 0:
                        token_id = int(tok_id_str[dash_idx + 1:])
                except ValueError:
                    pass
            
            # Get form (required)
            form = _get_attr_value_with_fallback(tok_elem, ["form"], fallback_to_text=True)
            if not form:
                form = (tok_elem.text or "").strip()
            
            # Infer space_after from XML structure
            # Check if there's any whitespace (XML whitespace, not spaces inside tags) between
            # the current token and the next token, regardless of where it appears in the structure.
            # This handles cases like:
            # - <tok>X</tok> <tok>Y</tok> (space in tail of first tok)
            # - <name><tok>X</tok></name> <tok>Y</tok> (space in tail of name)
            # - <name><tok>X</tok> <tok>Y</tok></name> <tok>Z</tok> (X has no space, Y has space from name's tail)
            # - <name><tok>X</tok></name><s> <tok>Y</tok></s> (space in text of <s> element)
            space_after = False
            if tok_idx < len(tok_elements) - 1:
                next_tok_elem = tok_elements[tok_idx + 1]
                
                # Helper function to check if a string contains whitespace
                def has_whitespace(s: str) -> bool:
                    return bool(s and (s.strip() or s[0].isspace()))
                
                # Check all possible locations for whitespace between current and next token:
                
                # 1. Tail of current token (text after </tok>)
                if has_whitespace(tok_elem.tail or ""):
                    space_after = True
                
                # 2. Walk up the parent chain from current token and check tails
                if not space_after:
                    p = parent_map.get(tok_elem)
                    while p is not None and p != s_elem:
                        if has_whitespace(p.tail or ""):
                            space_after = True
                            break
                        p = parent_map.get(p)
                
                # 3. Check elements between current and next token in sentence's children
                # Find the outermost parent elements that are direct children of the sentence
                if not space_after:
                    current_parent = parent_map.get(tok_elem)
                    next_parent = parent_map.get(next_tok_elem)
                    
                    # Walk up to find the outermost parent that's a direct child of the sentence
                    current_outermost = current_parent
                    while current_outermost is not None and parent_map.get(current_outermost) != s_elem:
                        current_outermost = parent_map.get(current_outermost)
                    
                    next_outermost = next_parent
                    while next_outermost is not None and parent_map.get(next_outermost) != s_elem:
                        next_outermost = parent_map.get(next_outermost)
                    
                    # If both are direct children, check elements between them
                    if current_outermost is not None and next_outermost is not None:
                        try:
                            current_pos = all_children.index(current_outermost)
                            next_pos = all_children.index(next_outermost)
                            for i in range(current_pos + 1, next_pos):
                                node = all_children[i]
                                # Skip comments
                                if callable(node.tag) or (hasattr(node.tag, '__name__') and 'Comment' in str(node.tag)):
                                    continue
                                # Check text content of intermediate elements (e.g., <s> <tok>Y</tok></s>)
                                if hasattr(node, 'text') and has_whitespace(node.text or ""):
                                    space_after = True
                                    break
                                # Check tail of intermediate elements
                                if hasattr(node, 'tail') and has_whitespace(node.tail or ""):
                                    space_after = True
                                    break
                        except (ValueError, AttributeError):
                            pass
                    
                    # Also check if next token's parent has text before the token
                    # (e.g., <s> <tok>Y</tok></s> where space is in <s>'s text)
                    if not space_after and next_parent is not None:
                        # Check if the next token is the first child of its parent
                        # and if the parent has text before it
                        parent_children = list(next_parent)
                        if parent_children and parent_children[0] == next_tok_elem:
                            if has_whitespace(next_parent.text or ""):
                                space_after = True
            
            # Get other attributes with mappings
            # Only fall back to form/innerText if "form" is explicitly in the attribute list
            lemma_has_form = "form" in lemma_attrs
            lemma = _get_attr_value_with_fallback(tok_elem, lemma_attrs, fallback_to_text=lemma_has_form)
            if not lemma and lemma_has_form:
                lemma = form  # Fallback to form for lemma only if form is in the list
            xpos = _get_attr_value_with_fallback(tok_elem, xpos_attrs)
            upos = _get_attr_value_with_fallback(tok_elem, ["upos"])
            feats = _get_attr_value_with_fallback(tok_elem, ["feats"])
            
            # Get reg - only fall back to form if "form" is explicitly in the attribute list
            reg_has_form = "form" in reg_attrs
            reg = _get_attr_value_with_fallback(tok_elem, reg_attrs, fallback_to_text=reg_has_form)
            if not reg and reg_has_form:
                reg = form  # Fallback to form only if form is in the list
            
            # Get expan - only fall back to form if "form" is explicitly in the attribute list
            expan_has_form = "form" in expan_attrs
            expan = _get_attr_value_with_fallback(tok_elem, expan_attrs, fallback_to_text=expan_has_form)
            if not expan and expan_has_form:
                expan = form  # Fallback to form only if form is in the list
            
            mod = _get_attr_value_with_fallback(tok_elem, ["mod"])
            trslit = _get_attr_value_with_fallback(tok_elem, ["trslit"])
            ltrslit = _get_attr_value_with_fallback(tok_elem, ["ltrslit"])
            # Prefer id over xml:id (unless source has xml:id)
            tokid = tok_elem.get("id")
            if not tokid:
                tokid = tok_elem.get("{http://www.w3.org/XML/1998/namespace}id") or tok_elem.get("xml:id")
            
            # Store all attributes from <tok> element in attrs (except those we use directly)
            # Initialize early so it's available for head_tokid storage
            tok_attrs: Dict[str, str] = {}
            known_attrs = {"form", "lemma", "xpos", "upos", "feats", "reg", "expan", "mod", 
                          "trslit", "ltrslit", "id", "xml:id", "head", "deprel", "deps", "misc", "tokid"}
            # Also add all attribute names from the mapping lists
            known_attrs.update(xpos_attrs)
            known_attrs.update(reg_attrs)
            known_attrs.update(expan_attrs)
            known_attrs.update(lemma_attrs)
            
            # Get head, deprel, deps, misc, ord if present
            # Head in TEITOK is a tokid, not an ord - we'll convert it later if needed
            head_str = tok_elem.get("head", "")
            # Try to parse as integer (ord) first, but if it's not a digit, it's a tokid
            if head_str and head_str.isdigit():
                head = int(head_str)
            else:
                # Head is a tokid - store it in attrs and set head to 0 for now
                # We'll need to convert tokid to ord after all tokens are loaded
                head = 0
                if head_str:
                    tok_attrs["head_tokid"] = head_str  # Store original tokid for later conversion
            deprel = tok_elem.get("deprel", "")
            deps = tok_elem.get("deps", "")
            misc = tok_elem.get("misc", "")
            ord_val = tok_elem.get("ord", "")
            known_attrs = {"form", "lemma", "xpos", "upos", "feats", "reg", "expan", "mod", 
                          "trslit", "ltrslit", "id", "xml:id", "head", "deprel", "deps", "misc", "tokid"}
            # Also add all attribute names from the mapping lists
            known_attrs.update(xpos_attrs)
            known_attrs.update(reg_attrs)
            known_attrs.update(expan_attrs)
            known_attrs.update(lemma_attrs)
            for key, value in tok_elem.attrib.items():
                if key not in known_attrs:
                    tok_attrs[key] = value
            # Store ord in attrs if present
            if ord_val:
                tok_attrs["ord"] = ord_val
            
            # Parse subtokens (dtok elements)
            subtokens: List[SubToken] = []
            for dtok_elem in tok_elem.findall("dtok"):
                # Check if @form attribute is "--" (suppressed subtoken) - skip it
                # Note: We only check the attribute, not innerText
                dtok_form_attr = dtok_elem.get("form", "")
                if dtok_form_attr == "--":
                    continue  # Skip suppressed subtokens
                
                dtok_form = _get_attr_value_with_fallback(dtok_elem, ["form"], fallback_to_text=True)
                if not dtok_form:
                    dtok_form = (dtok_elem.text or "").strip()
                
                # Only fall back to form/innerText if "form" is explicitly in the attribute list
                dtok_lemma_has_form = "form" in lemma_attrs
                dtok_lemma = _get_attr_value_with_fallback(dtok_elem, lemma_attrs, fallback_to_text=dtok_lemma_has_form)
                if not dtok_lemma and dtok_lemma_has_form:
                    dtok_lemma = dtok_form  # Fallback to form for lemma only if form is in the list
                dtok_xpos = _get_attr_value_with_fallback(dtok_elem, xpos_attrs)
                dtok_upos = _get_attr_value_with_fallback(dtok_elem, ["upos"])
                dtok_feats = _get_attr_value_with_fallback(dtok_elem, ["feats"])
                dtok_reg_has_form = "form" in reg_attrs
                dtok_reg = _get_attr_value_with_fallback(dtok_elem, reg_attrs, fallback_to_text=dtok_reg_has_form)
                if not dtok_reg and dtok_reg_has_form:
                    dtok_reg = dtok_form
                dtok_expan_has_form = "form" in expan_attrs
                dtok_expan = _get_attr_value_with_fallback(dtok_elem, expan_attrs, fallback_to_text=dtok_expan_has_form)
                if not dtok_expan and dtok_expan_has_form:
                    dtok_expan = dtok_form
                
                # Store all attributes from <dtok> element in attrs (except those we use directly)
                dtok_attrs: Dict[str, str] = {}
                dtok_known_attrs = {"form", "lemma", "xpos", "upos", "feats", "reg", "expan"}
                dtok_known_attrs.update(xpos_attrs)
                dtok_known_attrs.update(reg_attrs)
                dtok_known_attrs.update(expan_attrs)
                dtok_known_attrs.update(lemma_attrs)
                for key, value in dtok_elem.attrib.items():
                    if key not in dtok_known_attrs:
                        dtok_attrs[key] = value
                
                subtokens.append(SubToken(
                    id=len(subtokens) + 1,
                    form=dtok_form,
                    lemma=dtok_lemma,
                    xpos=dtok_xpos,
                    upos=dtok_upos,
                    feats=dtok_feats,
                    reg=dtok_reg,
                    expan=dtok_expan,
                    space_after=False,  # Will be set later
                    attrs=dtok_attrs,
                ))
            
            token = Token(
                id=token_id,
                form=form,
                lemma=lemma,
                xpos=xpos,
                upos=upos,
                feats=feats,
                reg=reg,
                expan=expan,
                mod=mod,
                trslit=trslit,
                ltrslit=ltrslit,
                tokid=tokid,
                head=head,
                deprel=deprel,
                deps=deps,
                misc=misc,
                is_mwt=len(subtokens) > 0,
                subtokens=subtokens,
                space_after=space_after,  # Inferred from XML structure
                attrs=tok_attrs,
            )
            
            if subtokens:
                token.mwt_start = token_id
                token.mwt_end = token_id + len(subtokens) - 1
                token.parts = [st.form for st in subtokens]
                # Set space_after for last subtoken only
                for i, st in enumerate(subtokens):
                    st.space_after = (i == len(subtokens) - 1)
            
            sentence.tokens.append(token)
            token_id += len(subtokens) if subtokens else 1
        
        # Set space_after for the last token to None (no SpaceAfter entry in CoNLL-U)
        if sentence.tokens:
            sentence.tokens[-1].space_after = None
        
        # Extract entities from <name> elements
        # Build a map of token indices (in tok_elements) to their token IDs (in sentence.tokens)
        # This accounts for suppressed tokens that were skipped
        tok_elem_idx_to_token_id: Dict[int, int] = {}
        token_id_counter = 1
        for tok_idx, tok_elem in enumerate(tok_elements):
            form_attr = tok_elem.get("form", "")
            if form_attr == "--":
                continue  # Skip suppressed tokens
            tok_elem_idx_to_token_id[tok_idx] = token_id_counter
            token_id_counter += 1
        
        # Create Entity objects from <name> elements
        for name_elem, name_tok_elems in name_to_tokens.items():
            if not name_tok_elems:
                continue
            
            # Get entity type from <name> element
            entity_type = name_elem.get("type", "")
            if not entity_type:
                continue  # Skip <name> elements without type
            
            # Get entity text from <name> element (if available)
            entity_text = name_elem.get("text", "")
            
            # Find the token IDs for the first and last tokens in this entity
            # Sort tokens by their position in tok_elements to ensure correct order
            name_tok_elems_sorted = sorted(name_tok_elems, key=lambda t: tok_elem_to_index.get(t, -1))
            
            first_tok_idx = tok_elem_to_index.get(name_tok_elems_sorted[0], -1)
            last_tok_idx = tok_elem_to_index.get(name_tok_elems_sorted[-1], -1)
            
            if first_tok_idx < 0 or last_tok_idx < 0:
                continue  # Skip if we can't find token indices
            
            # Map to actual token IDs (accounting for suppressed tokens)
            first_token_id = tok_elem_idx_to_token_id.get(first_tok_idx)
            last_token_id = tok_elem_idx_to_token_id.get(last_tok_idx)
            
            if first_token_id is None or last_token_id is None:
                continue  # Skip if tokens were suppressed
            
            # Extract additional attributes from <name> element
            entity_attrs: Dict[str, str] = {}
            for key, value in name_elem.attrib.items():
                if key not in ("type", "text"):  # type and text are handled separately
                    entity_attrs[key] = value
            
            # Create Entity object
            entity = Entity(
                start=first_token_id,
                end=last_token_id,
                label=entity_type,
                text=entity_text,
                attrs=entity_attrs,
            )
            sentence.entities.append(entity)
        
        # Reconstruct sentence text if missing, using actual spacing from XML
        if not sentence.text:
            parts = []
            for tok in sentence.tokens:
                parts.append(tok.form)
                if tok.space_after:
                    parts.append(" ")
            sentence.text = "".join(parts).strip()
        else:
            # If sentence.text exists (from @text attribute), use it to refine space_after values
            # This allows @text to override XML structure if it's more accurate
            temp_doc = Document(id="", sentences=[sentence])
            _infer_space_after_from_text(temp_doc)
            # But still set last token to None
            if sentence.tokens:
                sentence.tokens[-1].space_after = None
        
        document.sentences.append(sentence)
        sentence_counter += 1
    
    # Post-process: convert head from tokid to ord if needed
    # Build tokid -> ord mapping for all tokens
    tokid_to_ord: Dict[str, int] = {}
    for sentence in document.sentences:
        for token in sentence.tokens:
            if token.tokid and token.id:
                tokid_to_ord[token.tokid] = token.id
    
    # Convert head from tokid to ord for tokens that have head_tokid in attrs
    for sentence in document.sentences:
        for token in sentence.tokens:
            if "head_tokid" in token.attrs:
                head_tokid = token.attrs["head_tokid"]
                head_ord = tokid_to_ord.get(head_tokid, 0)
                if head_ord > 0:
                    token.head = head_ord
                # Remove the temporary head_tokid from attrs
                del token.attrs["head_tokid"]
    
    assign_doc_id_from_path(document, path)
    _apply_sentence_correspondence(document)
    return _sanitize_document(document)


def save_teitok(
    document: Document,
    path: str,
    custom_attributes: Optional[List[str]] = None,
    pretty_print: bool = False,
    *,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
    settings: Optional["TeitokSettings"] = None,
    unicode_normalization: Optional[str] = None,
) -> None:
    """
    Save a Document to a TEITOK XML file.
    
    Args:
        document: The Document to save
        path: Output file path
        custom_attributes: Optional list of custom attribute names to write from token.attrs.
                          If None or empty, all attributes in token.attrs will be written.
                          If provided, only attributes in this list will be written.
        pretty_print: If True, pretty-print the XML with indentation (newlines don't add whitespace between tokens)
        spaceafter_handling: How to handle SpaceAfter=No tokens when pretty_printing:
            - "preserve" (default): Remove spaces between tokens with SpaceAfter=No and next token
            - "join": Add join="right" attribute to tokens with SpaceAfter=No
        skip_spaceafter_for_breaking_elements: If True (default), skip SpaceAfter handling
            if there's a hard-breaking element (like <p>) between tokens. Only used with
            spaceafter_handling="preserve".
    """
    # Use dump_teitok and write to file to ensure consistent pretty-printing logic
    xml_str = dump_teitok(
        document,
        custom_attributes,
        pretty_print=pretty_print,
        spaceafter_handling=spaceafter_handling,
        skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
        settings=settings,
        unicode_normalization=unicode_normalization,
    )
    Path(path).write_text(xml_str, encoding="utf-8")


def dump_teitok(
    document: Document,
    custom_attributes: Optional[List[str]] = None,
    pretty_print: bool = False,
    *,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
    settings: Optional["TeitokSettings"] = None,
    unicode_normalization: Optional[str] = None,
) -> str:
    """
    Convert a Document to a TEITOK XML string.
    
    Args:
        document: The Document to convert
        custom_attributes: Optional list of custom attribute names to write from token.attrs.
                          If None or empty, all attributes in token.attrs will be written.
                          If provided, only attributes in this list will be written.
        pretty_print: If True, pretty-print the XML with indentation (newlines don't add whitespace between tokens)
        spaceafter_handling: How to handle SpaceAfter=No tokens when pretty_printing:
            - "preserve" (default): Remove spaces between tokens with SpaceAfter=No and next token
            - "join": Add join="right" attribute to tokens with SpaceAfter=No
        skip_spaceafter_for_breaking_elements: If True (default), skip SpaceAfter handling
            if there's a hard-breaking element (like <p>) between tokens. Only used with
            spaceafter_handling="preserve".
    
    Returns:
        XML string representation of the document
    """
    sanitized = _sanitize_document(document)
    # Ensure entities from spans are in sentences (defensive check)
    # _sanitize_document should already do this, but ensure it's done
    span_entities = collect_span_entities_by_sentence(sanitized, "ner")
    for idx, sent in enumerate(sanitized.sentences):
        # Add entities from spans if not already present
        existing_entity_keys = {(e.start, e.end, e.label) for e in sent.entities}
        if idx in span_entities:
            for ent in span_entities[idx]:
                key = (ent.start, ent.end, ent.label)
                if key not in existing_entity_keys:
                    sent.entities.append(ent)
                    existing_entity_keys.add(key)
    payload = _ensure_serializable(sanitized.to_dict())
    if custom_attributes is None:
        custom_attributes = []
    
    # If Unicode normalization is needed, use Python version (C++ doesn't support it)
    # Otherwise, try C++ version first for performance
    if unicode_normalization and unicode_normalization != "none":
        xml_str = _dump_teitok_python(payload, custom_attributes, pretty_print=False, unicode_normalization=unicode_normalization)
    else:
        try:
            xml_str = _dump_teitok(payload, custom_attributes, pretty_print=False)  # type: ignore
        except RuntimeError as e:
            # If the C++ extension is not available, the error message will be informative
            # The Python fallback should already be set, so this shouldn't happen, but handle gracefully
            if "flexitag_py extension is not available" in str(e):
                # This should not happen if fallback is properly set, but just in case
                raise RuntimeError(
                    "TEITOK output requires the flexitag_py C++ extension, which is not available. "
                    "The Python fallback should have been used automatically. "
                    "Please report this issue. "
                    "To build the C++ extension, see: https://github.com/ufal/flexipipe#building-native-modules"
                ) from e
            raise
    if settings:
        xml_str = _remap_teitok_attributes(xml_str, settings)
    if pretty_print:
        xml_str = pretty_print_teitok_xml(
            xml_str,
            spaceafter_handling=spaceafter_handling,
            skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
        )
    return xml_str


def _remap_teitok_attributes(xml_str: str, settings: "TeitokSettings") -> str:
    """Rename or drop token attributes according to TEITOK settings."""
    attr_map = settings.build_output_attribute_map()
    rename_map = {src: dst for src, dst in attr_map.items() if dst and dst != src}
    remove_set = {src for src, dst in attr_map.items() if dst is None}
    if not rename_map and not remove_set:
        return xml_str
    
    use_lxml = False
    parser = None
    try:
        from lxml import etree as etree_mod
        parser = etree_mod.XMLParser(remove_blank_text=False)
        root = etree_mod.fromstring(xml_str.encode("utf-8"), parser)
        use_lxml = True
        etree = etree_mod
    except ImportError:
        import xml.etree.ElementTree as etree_mod
        etree = etree_mod
        try:
            root = etree.fromstring(xml_str)
        except etree.ParseError:
            return xml_str
    except Exception:
        return xml_str
    
    target_tags = {"tok", "dtok"}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in target_tags:
            continue
        for attr_name in list(elem.attrib.keys()):
            if attr_name in remove_set:
                elem.attrib.pop(attr_name, None)
                continue
            target = rename_map.get(attr_name)
            if target:
                if target == attr_name:
                    continue
                value = elem.attrib.pop(attr_name)
                elem.set(target, value)
    
    updated = etree.tostring(
        root,
        encoding="unicode",
        pretty_print=False if use_lxml else None,
    )
    
    if xml_str.startswith("<?xml"):
        prefix, _, remainder = xml_str.partition("?>")
        newline = "\n" if remainder.startswith("\n") else ""
        return f"{prefix}?>{newline}{updated.lstrip()}"
    return updated


def pretty_print_teitok_xml(
    xml_str: str,
    *,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
) -> str:
    """
    Pretty-print TEITOK XML with proper indentation using lxml.
    
    Args:
        xml_str: The XML string to pretty-print
        spaceafter_handling: How to handle SpaceAfter=No tokens:
            - "preserve" (default): Remove spaces between tokens with SpaceAfter=No and next token
            - "join": Add join="right" attribute to tokens with SpaceAfter=No
        skip_spaceafter_for_breaking_elements: If True (default), skip SpaceAfter handling
            if there's a hard-breaking element (like <p>) between tokens. Only used with
            spaceafter_handling="preserve".
        
    Returns:
        Pretty-printed XML string
    """
    # Use lxml for proper XML pretty-printing
    try:
        from lxml import etree
    except ImportError:
        # Fall back to ElementTree - it doesn't support pretty_print
        return xml_str
    
    # Parse the XML
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.fromstring(xml_str.encode("utf-8"), parser)
    
    # Use lxml's indent function for pretty-printing first
    etree.indent(tree, space="  ")
    
    # Handle SpaceAfter=No tokens (after indent to ensure attributes are preserved)
    # In both modes, we add join="Right" as a marker so we can check for it consistently
    # (SpaceAfter=No is CoNLL-U specific, but join="Right" is a universal marker)
    for tok_elem in tree.xpath('.//tok'):
        misc = tok_elem.get("misc", "")
        if misc and "SpaceAfter=No" in misc:
            tok_elem.set("join", "Right")
    
    # Generate the pretty-printed XML
    # Note: encoding="unicode" doesn't support xml_declaration, so we add it manually if needed
    updated = etree.tostring(tree, encoding="unicode", pretty_print=True)
    
    # Post-process SpaceAfter=No handling for preserve approach (string-based)
    if spaceafter_handling == "preserve":
        import re
        
        # 1. Remove ALL whitespace (including newlines) around <dtok> elements
        # Pattern: match all whitespace before and after <dtok> tags
        def clean_dtok_spaces(m):
            before = m.group(1)
            tag = m.group(2)
            after = m.group(3)
            # Remove all whitespace (spaces, tabs, newlines)
            return tag
        
        updated = re.sub(r'(\s*)(</?dtok[^>]*>)(\s*)', clean_dtok_spaces, updated)
        
        # 2. Remove spaces between tokens with SpaceAfter=No and the next token
        # This includes spaces in the token's tail and any spaces in parent element tails
        
        # First, find all tokens with SpaceAfter=No
        # Pattern: <tok[^>]*misc="[^"]*SpaceAfter=No[^"]*"[^>]*>.*?</tok>
        # We need to match the entire token element and remove spaces after it until the next <tok>
        
        # Strategy: Find tokens with SpaceAfter=No, then remove all whitespace (except newlines for indentation)
        # between the closing </tok> and the next <tok> or </name> or other closing tag
        
        # Pattern to match: </tok> followed by optional whitespace and closing tags, then optional whitespace, then <tok
        # We want to remove spaces but preserve newlines for indentation
        
        # More precise: match </tok> with SpaceAfter=No, then remove spaces (but keep newlines) until next <tok
        def remove_spaces_after_token(match):
            """Remove spaces after a token with SpaceAfter=No, preserving newlines for indentation."""
            token_tag = match.group(0)
            # The token tag is already matched, now we need to find the closing </tok> and remove spaces after it
            return token_tag
        
        # Find all <tok> elements with SpaceAfter=No in the XML string
        # We'll use a regex to find the pattern: <tok[^>]*misc="[^"]*SpaceAfter=No[^"]*"[^>]*>...?</tok>
        # Then remove whitespace (except newlines) between </tok> and the next <tok> or closing tag
        
        # Better approach: use regex to find </tok> that follows a token with SpaceAfter=No
        # Pattern: match </tok> that comes after a token with SpaceAfter=No, then remove spaces before next <tok>
        
        # 2. Remove ALL whitespace (including newlines) between tokens with join="Right" and the next token
        # We only check for join="Right" attribute (not SpaceAfter=No, which is CoNLL-U specific)
        # This includes spaces in the token's tail and any spaces in parent element tails
        
        def remove_all_whitespace_after_join_token(m):
            """Remove all whitespace (including newlines) after a token with join="Right"."""
            tok_element = m.group(1)  # The entire <tok>...</tok> element
            whitespace1 = m.group(2)  # Whitespace after </tok>
            closing_name = m.group(3) or ''  # Optional </name>
            whitespace2 = m.group(4) or ''  # Whitespace after </name> (if present)
            next_tok = m.group(5)  # The next <tok> tag
            
            # Remove ALL whitespace (spaces, tabs, newlines) - no whitespace between tokens
            return tok_element + closing_name + next_tok
        
        # Match token with join="Right", then optional whitespace, optional </name>, optional whitespace, then <tok>
        # Pattern matches tokens that have join="Right" anywhere in the tag
        # Be careful: match the attribute value exactly to avoid false matches
        pattern = r'(<tok[^>]*join="Right"[^>]*>.*?</tok>)(\s*)(</name>)?(\s*)(<tok)'
        updated = re.sub(pattern, remove_all_whitespace_after_join_token, updated, flags=re.DOTALL)
    
    # Add XML declaration if not present
    if not updated.startswith("<?xml"):
        updated = '<?xml version="1.0" encoding="UTF-8"?>\n' + updated
    
    return updated


def _verify_character_level_alignment(
    document: Document,
    xml_root: ET.Element,
    matched_token_mapping: Dict[Tuple[int, int], ET.Element],
    from_raw_text: bool = False,
) -> tuple[bool, str]:
    """
    Verify character-level alignment between backend output and original XML.
    
    Reconstructs text from both sources and compares character-by-character.
    This ensures that token splits/merges don't introduce errors.
    
    Returns:
        (is_valid, error_message) tuple
    """
    import sys
    
    # Reconstruct text from XML tokens (original) - use @text attribute from <s> nodes if available
    xml_text_parts = []
    sentence_nodes = []
    for s_node in xml_root.iter():
        if s_node.tag.endswith("}s") or s_node.tag == "s":
            sentence_nodes.append(s_node)
    
    # Process sentences in document order
    for s_node in sentence_nodes:
        # First try to use sentence text attribute (most accurate)
        sent_text = s_node.get("text", "")
        if sent_text:
            xml_text_parts.append(sent_text)
            xml_text_parts.append(" ")
        else:
            # Fallback: reconstruct from tokens
            sent_tok_nodes = [n for n in s_node if (n.tag.endswith("}tok") or n.tag == "tok")]
            for tok_node in sent_tok_nodes:
                # Get form from text content or @form attribute
                form = (tok_node.text or "").strip() or tok_node.get("form", "")
                if form:
                    xml_text_parts.append(form)
                    # Check SpaceAfter attribute
                    space_after = tok_node.get("SpaceAfter", "Yes")
                    if space_after and space_after.lower() != "no":
                        xml_text_parts.append(" ")
    
    xml_text = "".join(xml_text_parts)
    
    # Reconstruct text from backend output (document)
    backend_text_parts = []
    for sent in document.sentences:
        for token in sent.tokens:
            if token.form:
                backend_text_parts.append(token.form)
                # Check SpaceAfter from MISC
                has_space_after = True
                if token.misc:
                    misc_parts = token.misc.split("|")
                    for part in misc_parts:
                        if part.strip().startswith("SpaceAfter=No"):
                            has_space_after = False
                            break
                if has_space_after:
                    backend_text_parts.append(" ")
    
    backend_text = "".join(backend_text_parts)
    
    # Normalize whitespace for comparison (collapse multiple spaces to single space)
    import re
    xml_text_normalized = re.sub(r'\s+', ' ', xml_text.strip())
    backend_text_normalized = re.sub(r'\s+', ' ', backend_text.strip())
    
    # Compare character-by-character
    if xml_text_normalized == backend_text_normalized:
        return (True, "")
    
    # Find the first difference
    min_len = min(len(xml_text_normalized), len(backend_text_normalized))
    diff_pos = min_len
    for i in range(min_len):
        if xml_text_normalized[i] != backend_text_normalized[i]:
            diff_pos = i
            break
    
    # Show context around the difference
    start = max(0, diff_pos - 20)
    end = min(len(xml_text_normalized), diff_pos + 20)
    xml_context = xml_text_normalized[start:end]
    backend_context = backend_text_normalized[start:end] if diff_pos < len(backend_text_normalized) else ""
    
    error_msg = (
        f"Character-level alignment mismatch at position {diff_pos}. "
        f"XML text: ...{xml_context}... "
        f"Backend text: ...{backend_context}... "
        f"(XML length: {len(xml_text_normalized)}, Backend length: {len(backend_text_normalized)})"
    )
    
    return (False, error_msg)


def _verify_character_level_alignment(
    document: Document,
    xml_root: ET.Element,
    matched_token_mapping: Dict[Tuple[int, int], ET.Element],
    from_raw_text: bool = False,
) -> tuple[bool, str]:
    """
    Verify character-level alignment between backend output and original XML.
    
    Reconstructs text from both sources and compares character-by-character.
    This ensures that token splits/merges don't introduce errors.
    
    Returns:
        (is_valid, error_message) tuple
    """
    import sys
    import re
    
    # Reconstruct text from XML tokens (original) - use @text attribute from <s> nodes if available
    # Match sentences by ID to ensure correct order
    xml_text_parts = []
    
    # Build a mapping of sentence IDs to sentence nodes
    sentid_to_xml_node = {}
    for s_node in xml_root.iter():
        if s_node.tag.endswith("}s") or s_node.tag == "s":
            sent_id = s_node.get("id") or s_node.get("{http://www.w3.org/XML/1998/namespace}id") or s_node.get("sent_id")
            if sent_id:
                sentid_to_xml_node[sent_id] = s_node
    
    # Process sentences in document order (match by sentence ID from document)
    # In raw text mode, we should compare sentence text directly (from @text attributes)
    # rather than reconstructing from tokens, since the backend retokenizes and may change spacing
    for sent_idx, sent in enumerate(document.sentences):
        sent_id = sent.sent_id or sent.id or sent.source_id
        s_node = sentid_to_xml_node.get(sent_id) if sent_id else None
        
        if s_node is not None:
            # First try to use sentence text attribute (most accurate, especially in raw text mode)
            sent_text = s_node.get("text", "")
            if sent_text:
                xml_text_parts.append(sent_text)
                # Add space between sentences (except last one)
                if sent_idx < len(document.sentences) - 1:
                    xml_text_parts.append(" ")
            else:
                # Fallback: reconstruct from tokens
                # In raw text mode, spacing may differ due to retokenization, but we still need to reconstruct
                sent_tok_nodes = [n for n in s_node if (n.tag.endswith("}tok") or n.tag == "tok")]
                for tok_idx, tok_node in enumerate(sent_tok_nodes):
                    # Get form from text content or @form attribute
                    form = (tok_node.text or "").strip() or tok_node.get("form", "")
                    if form:
                        xml_text_parts.append(form)
                        # Check SpaceAfter attribute
                        space_after = tok_node.get("SpaceAfter", "Yes")
                        # Don't add space after last token of last sentence
                        if space_after and space_after.lower() != "no":
                            if not (sent_idx == len(document.sentences) - 1 and tok_idx == len(sent_tok_nodes) - 1):
                                xml_text_parts.append(" ")
        else:
            # If no matching XML node, reconstruct from document tokens
            for token_idx, token in enumerate(sent.tokens):
                if token.form:
                    xml_text_parts.append(token.form)
                    if token.space_after and not (sent_idx == len(document.sentences) - 1 and token_idx == len(sent.tokens) - 1):
                        xml_text_parts.append(" ")
    
    xml_text = "".join(xml_text_parts)
    
    # In raw text mode, normalize whitespace more aggressively since backend may change spacing
    if from_raw_text:
        # Remove all spaces around punctuation for comparison
        xml_text_normalized = re.sub(r'\s*([.,!?;:])\s*', r'\1', xml_text)
        xml_text_normalized = re.sub(r'\s+', ' ', xml_text_normalized.strip())
    else:
        xml_text_normalized = re.sub(r'\s+', ' ', xml_text.strip())
    
    # Reconstruct text from backend output (document) - use sentence text if available
    backend_text_parts = []
    for sent_idx, sent in enumerate(document.sentences):
        # If sentence has text, use it directly (most accurate)
        if sent.text:
            backend_text_parts.append(sent.text)
            # Add space between sentences (except last one)
            if sent_idx < len(document.sentences) - 1:
                backend_text_parts.append(" ")
        else:
            # Fallback: reconstruct from tokens
            for token_idx, token in enumerate(sent.tokens):
                if token.form:
                    backend_text_parts.append(token.form)
                    # Check SpaceAfter from MISC or token.space_after
                    has_space_after = True
                    if token.space_after is False:
                        has_space_after = False
                    elif token.misc:
                        misc_parts = token.misc.split("|")
                        for part in misc_parts:
                            part_stripped = part.strip()
                            if part_stripped.startswith("SpaceAfter=No") or part_stripped == "SpaceAfter=No":
                                has_space_after = False
                                break
                    # Don't add space after last token of last sentence
                    if has_space_after and not (sent_idx == len(document.sentences) - 1 and token_idx == len(sent.tokens) - 1):
                        backend_text_parts.append(" ")
    
    backend_text = "".join(backend_text_parts)
    
    # In raw text mode, normalize whitespace more aggressively since backend may change spacing
    if from_raw_text:
        backend_text = re.sub(r'\s*([.,!?;:])\s*', r'\1', backend_text)
    
    # Normalize whitespace for comparison (collapse multiple spaces to single space, trim)
    xml_text_normalized = re.sub(r'\s+', ' ', xml_text.strip())
    backend_text_normalized = re.sub(r'\s+', ' ', backend_text.strip())
    
    # Compare character-by-character
    if xml_text_normalized == backend_text_normalized:
        return (True, "")
    
    # Find the first difference
    min_len = min(len(xml_text_normalized), len(backend_text_normalized))
    diff_pos = min_len
    for i in range(min_len):
        if xml_text_normalized[i] != backend_text_normalized[i]:
            diff_pos = i
            break
    
    # Show context around the difference
    start = max(0, diff_pos - 30)
    end = min(len(xml_text_normalized), diff_pos + 30)
    xml_context = xml_text_normalized[start:end]
    backend_context = backend_text_normalized[start:end] if diff_pos < len(backend_text_normalized) else ""
    
    # Show the actual characters at the difference
    xml_char = xml_text_normalized[diff_pos] if diff_pos < len(xml_text_normalized) else "<EOF>"
    backend_char = backend_text_normalized[diff_pos] if diff_pos < len(backend_text_normalized) else "<EOF>"
    
    xml_char_code = ord(xml_char) if len(xml_char) == 1 else 0
    backend_char_code = ord(backend_char) if len(backend_char) == 1 else 0
    
    error_msg = (
        f"Character-level alignment mismatch at position {diff_pos}. "
        f"XML char: '{xml_char}' (U+{xml_char_code:04X}), Backend char: '{backend_char}' (U+{backend_char_code:04X}). "
        f"XML text: ...{xml_context}... "
        f"Backend text: ...{backend_context}... "
        f"(XML length: {len(xml_text_normalized)}, Backend length: {len(backend_text_normalized)})"
    )
    
    return (False, error_msg)


def _add_change_to_tei_header(
    root: ET.Element,
    change_text: str,
    change_when: Optional[str] = None,
) -> None:
    """
    Add a <change> element to the <revisionDesc> in the TEI header.
    
    Args:
        root: The root element of the TEI XML tree
        change_text: The text content for the <change> element
        change_when: Optional ISO 8601 timestamp (defaults to current UTC time)
    """
    if change_when is None:
        from datetime import datetime
        change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Find or create teiHeader
    tei_header = root.find("teiHeader")
    if tei_header is None:
        tei_header = ET.Element("teiHeader")
        # Insert at the beginning (before <text>)
        root.insert(0, tei_header)
    
    # Find or create revisionDesc
    revision_desc = tei_header.find("revisionDesc")
    if revision_desc is None:
        revision_desc = ET.SubElement(tei_header, "revisionDesc")
    
    # Add change element
    change_elem = ET.SubElement(
        revision_desc,
        "change",
        {"when": change_when, "who": "flexipipe"},
    )
    change_elem.text = change_text


def update_teitok(
    document: Document,
    original_path: str,
    output_path: Optional[str] = None,
    *,
    settings: Optional["TeitokSettings"] = None,
    from_raw_text: bool = False,
    strict_alignment: bool = True,
    unicode_normalization: Optional[str] = None,
) -> None:
    """
    Update a TEITOK XML file in-place by matching nodes by ID and updating annotation attributes.
    
    This function preserves the original XML structure, comments, and metadata while only
    updating the annotation attributes (xpos, lemma, upos, feats, head, deprel, etc.) on
    nodes that match by ID.
    
    Args:
        document: Tagged Document with updated annotations
        original_path: Path to the original TEITOK XML file
        output_path: Optional output path (defaults to original_path for in-place update)
        settings: Optional TeitokSettings object to control attribute mappings
        from_raw_text: If True, document was processed from raw text (allows token splitting/merging).
                      If False, expects 1:1 token alignment (pretokenized mode).
        strict_alignment: If True, refuse to write if alignment cannot be verified (default: True).
                         If False, allow partial alignment with warnings.
    
    Raises:
        RuntimeError: If strict_alignment=True and alignment cannot be verified (too many mismatches).
    """
    try:
        from lxml import etree as ET
        preserve_comments = True
    except ImportError:
        import xml.etree.ElementTree as ET
        preserve_comments = False
    
    original_path_obj = Path(original_path)
    if not original_path_obj.exists():
        raise FileNotFoundError(f"Original TEITOK file not found: {original_path}")
    
    output_path_obj = Path(output_path) if output_path else original_path_obj
    
    # Parse original XML
    if preserve_comments:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        tree = ET.parse(str(original_path_obj), parser)
        root = tree.getroot()
    else:
        tree = ET.parse(str(original_path_obj))
        root = tree.getroot()
    
    # Build ID mappings: tokid -> XML node, sent_id -> XML node
    tokid_to_node: Dict[str, ET.Element] = {}
    sentid_to_node: Dict[str, ET.Element] = {}
    
    # Build parent map for ElementTree compatibility (lxml has getparent(), ElementTree doesn't)
    parent_map: Dict[ET.Element, ET.Element] = {}
    if not hasattr(root, 'getparent'):
        # ElementTree - build parent map
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent
    
    def get_parent(node: ET.Element) -> Optional[ET.Element]:
        """Get parent node, handling both lxml and ElementTree."""
        if hasattr(node, 'getparent'):
            return node.getparent()
        return parent_map.get(node)
    
    def _attribute_aliases(internal_attr: str) -> List[str]:
        if settings:
            aliases = settings.get_attribute_mapping(internal_attr)
        else:
            aliases = [internal_attr]
        ordered: List[str] = []
        for alias in aliases + [internal_attr]:
            if alias and alias not in ordered:
                ordered.append(alias)
        return ordered
    
    def _resolve_attr_name(internal_attr: str) -> Optional[str]:
        if settings:
            return settings.resolve_xml_attribute(internal_attr, default=internal_attr)
        return internal_attr
    
    # Find all <s>, <tok>, and <dtok> nodes
    # Handle both namespaced and non-namespaced elements
    # Note: Duplicate IDs are already fixed during loading, so we can just collect them
    for s_node in root.iter():
        if s_node.tag.endswith("}s") or s_node.tag == "s":
            sent_id = s_node.get("id") or s_node.get("{http://www.w3.org/XML/1998/namespace}id") or s_node.get("sent_id")
            if sent_id:
                sentid_to_node[sent_id] = s_node
            # Also ensure sentence has an ID if missing
            if not sent_id:
                # Generate ID from sentence index
                parent = get_parent(s_node)
                if parent is not None:
                    siblings = [c for c in parent if (c.tag.endswith("}s") or c.tag == "s")]
                    if s_node in siblings:
                        idx = siblings.index(s_node) + 1
                        sent_id = f"s-{idx}"
                        s_node.set("id", sent_id)
                        sentid_to_node[sent_id] = s_node
    
    # First pass: collect all tokens in document order for continuous numbering
    all_tok_nodes: List[ET.Element] = []
    for tok_node in root.iter():
        if tok_node.tag.endswith("}tok") or tok_node.tag == "tok":
            all_tok_nodes.append(tok_node)
    
    # Second pass: assign IDs and build mapping
    # Preserve existing IDs, only generate new ones if missing
    # Prefer id over xml:id when reading
    # Note: Duplicate IDs are already fixed during loading
    global_token_counter = 1
    for tok_node in all_tok_nodes:
        tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
        if not tokid:
            # Generate ID with continuous numbering across all sentences
            # Prefer id over xml:id when writing
            tokid = f"w-{global_token_counter}"
            tok_node.set("id", tokid)
        elif tok_node.get("{http://www.w3.org/XML/1998/namespace}id") and not tok_node.get("id"):
            # Has xml:id but not id - move to id (prefer id over xml:id)
            xml_id = tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
            tok_node.set("id", xml_id)
            del tok_node.attrib["{http://www.w3.org/XML/1998/namespace}id"]
            tokid = xml_id
        if tokid:
            tokid_to_node[tokid] = tok_node
        global_token_counter += 1
    
    for dtok_node in root.iter():
        if dtok_node.tag.endswith("}dtok") or dtok_node.tag == "dtok":
            tokid = dtok_node.get("{http://www.w3.org/XML/1998/namespace}id") or dtok_node.get("id")
            if not tokid:
                # Generate ID if missing - use parent tok ID + dtok index
                parent = get_parent(dtok_node)
                if parent is not None:
                    parent_tokid = parent.get("{http://www.w3.org/XML/1998/namespace}id") or parent.get("id")
                    siblings = [c for c in parent if (c.tag.endswith("}dtok") or c.tag == "dtok")]
                    if dtok_node in siblings:
                        idx = siblings.index(dtok_node) + 1
                        if parent_tokid:
                            tokid = f"{parent_tokid}-dtok{idx}"
                        else:
                            tokid = f"dtok-{idx}"
                        dtok_node.set("{http://www.w3.org/XML/1998/namespace}id", tokid)
            if tokid:
                tokid_to_node[tokid] = dtok_node
    
    # Update nodes from tagged Document
    sanitized = _sanitize_document(document)
    matched_tokens = 0
    skipped_tokens = 0
    matched_sentences = 0
    skipped_sentences = 0
    
    # Helper function to check if a token is likely from a CoNLL-U comment line
    # These tokens should be skipped during alignment as they don't exist in the XML
    def is_comment_line_token(token: Token) -> bool:
        """Check if token looks like it came from a CoNLL-U comment line."""
        form = token.form or ""
        # Check for common comment line patterns that UDPipe might tokenize
        comment_patterns = [
            "#", "sent_id", "text", "newdoc", "newpar", "generator", "language",
            "udpipe_model", "udpipe_model_licence", "TokId=", "SpaceAfter=",
            "http://", "www.w3.org", "namespace", "space", "preserve", "=",
            "{", "}", "|", "newdoc", "id", "toktest", "newpar"
        ]
        # Check if form matches a comment pattern exactly
        if form in comment_patterns:
            return True
        # Check if form looks like a MISC field value (contains = or |)
        if "=" in form or ("|" in form and not form.startswith("|")):
            # But allow forms that are actual words (e.g., "awesome-align")
            # Only flag if it's clearly a comment/metadata pattern
            if any(pattern in form for pattern in ["TokId", "SpaceAfter", "http", "namespace", "udpipe"]):
                return True
        # Check if form looks like a URL or namespace
        if form.startswith("http://") or form.startswith("www.") or "namespace" in form.lower():
            return True
        # Check if form is just punctuation that might be from comment lines
        if form in ["{", "}", "=", "|", "#"]:
            return True
        return False
    
    # Build list of sentence nodes in order for position-based matching
    sentence_nodes_ordered = []
    for s_node in root.iter():
        if s_node.tag.endswith("}s") or s_node.tag == "s":
            sentence_nodes_ordered.append(s_node)
    
    # Build global ord_to_tokid mapping by matching all tokens to XML nodes
    # This is needed to convert head from ord to tokid (head can point across sentences)
    # First, ensure ALL tokens have tokids assigned (even if no XML node yet)
    global_ord_to_tokid: Dict[int, str] = {}
    # Use (sent_idx, token_idx) as key instead of Token object (Token is not hashable)
    global_token_to_tok_node: Dict[Tuple[int, int], ET.Element] = {}
    global_tokid_counter = 1  # For continuous numbering when generating new tokids
    
    # First, count existing tokids to get the starting counter
    for existing_tokid in tokid_to_node.keys():
        if existing_tokid.startswith("w-"):
            try:
                num = int(existing_tokid[2:])
                global_tokid_counter = max(global_tokid_counter, num + 1)
            except ValueError:
                pass
    
    # Pre-assign tokids to all tokens that don't have them yet
    for sent in sanitized.sentences:
        for token in sent.tokens:
            if token.id and not token.tokid:
                # Generate tokid for token that doesn't have one yet
                token.tokid = f"w-{global_tokid_counter}"
                global_tokid_counter += 1
            # Build initial mapping
            if token.id and token.tokid:
                global_ord_to_tokid[token.id] = token.tokid
    
    for sent_idx, sent in enumerate(sanitized.sentences):
        sent_id = sent.sent_id or sent.id or sent.source_id
        s_node = None
        
        # Try exact match first
        if sent_id:
            s_node = sentid_to_node.get(sent_id)
        
        # If exact match failed, try position-based matching
        if s_node is None and sent_idx < len(sentence_nodes_ordered):
            s_node = sentence_nodes_ordered[sent_idx]
        
        if s_node is None:
            continue  # Skip for now, will handle in main loop
        
        # Match tokens to XML nodes and build mapping
        # Only get direct <tok> children, not all descendants (performance optimization)
        sent_tok_nodes = [n for n in s_node if (n.tag.endswith("}tok") or n.tag == "tok")]
        
        # Build form-based index for smart alignment (for raw text mode with splits/joins)
        # Track which XML nodes have been matched to avoid double-matching
        matched_xml_nodes: Set[ET.Element] = set()
        xml_form_to_nodes: Dict[str, List[ET.Element]] = {}
        
        def get_full_text_content(elem: ET.Element) -> str:
            """Get all text content from an element, including text before and after children."""
            # Get text before first child
            text_parts = [elem.text or ""]
            # Get text from children (tail text after each child)
            for child in elem:
                if child.tail:
                    text_parts.append(child.tail)
            return "".join(text_parts).strip()
        
        def is_valid_match(candidate_node: ET.Element, token_form: str, settings: Optional["TeitokSettings"] = None) -> bool:
            """
            Verify that a candidate XML node is a valid match for a token form.
            
            STRICT MATCHING: Only accepts matches where forms are EXACTLY equal.
            Returns False if:
            - Text content is longer than token form (indicates a split token)
            - Canonical form is longer than token form (indicates a split token)
            - Text content is shorter than token form (indicates a merge or wrong match)
            - Canonical form is shorter than token form (indicates a merge or wrong match)
            - Neither canonical form nor text content EXACTLY matches token form
            
            Args:
                candidate_node: XML token element to check
                token_form: Token form to match against
                settings: Optional TeitokSettings object
            
            Returns:
                True if the match is valid, False otherwise
            """
            if not token_form:
                return True  # No form to verify
            
            token_form = token_form.strip()
            if not token_form:
                return True  # Empty form, accept
            
            candidate_text = get_full_text_content(candidate_node).strip()
            candidate_canonical = gettokform(candidate_node, settings).strip()
            
            # CRITICAL: Reject if text content is longer than token form (indicates a split token)
            # Example: candidate_text="awesome-align" (13 chars) vs token_form="awesome" (7 chars) -> REJECT
            if candidate_text and len(candidate_text) > len(token_form):
                return False
            
            # CRITICAL: Reject if canonical form is longer than token form (indicates a split token)
            if candidate_canonical and len(candidate_canonical) > len(token_form):
                return False
            
            # CRITICAL: Reject if text content is shorter than token form (indicates wrong match)
            # Example: candidate_text="a" vs token_form="awesome" -> REJECT
            if candidate_text and len(candidate_text) < len(token_form):
                return False
            
            # CRITICAL: Reject if canonical form is shorter than token form (indicates wrong match)
            if candidate_canonical and len(candidate_canonical) < len(token_form):
                return False
            
            # STRICT: Only accept if canonical form or text content EXACTLY matches token form
            if candidate_canonical and candidate_canonical == token_form:
                return True
            if candidate_text and candidate_text == token_form:
                return True
            
            # If we have a form attribute that EXACTLY matches and no text/canonical, that's OK
            candidate_form = candidate_node.get("form", "").strip()
            if candidate_form == token_form and not candidate_text and not candidate_canonical:
                return True
            
            # If neither matches EXACTLY, reject
            return False
        
        def gettokform(elem: ET.Element, settings: Optional["TeitokSettings"] = None, form_attr: Optional[str] = None) -> str:
            """
            Get the canonical form of a token element based on TEITOK settings.
            
            When forms hierarchy is defined in settings (via <forms> in <pattributes>):
            - Follows inheritance chain: requested_form -> parent_form -> ... -> base_form (pform) -> innerText
            - Example: nform inherits from fform, fform from form, form from pform, pform = innerText
            - If form_attr is None, uses defaultform from settings (default: "form")
            
            When forms hierarchy is not defined:
            - Uses legacy hierarchy: @reg > @expan > @form > innerText (if reg/expan in CQP pattributes)
            - Otherwise: @form > innerText
            
            Args:
                elem: XML token element (<tok> or <dtok>)
                settings: Optional TeitokSettings object
                form_attr: Optional form attribute name to use (e.g., "nform", "fform"). 
                          If None, uses defaultform from settings or "form"
            
            Returns:
                The canonical form string
            """
            # Determine which form attribute to use
            target_form = form_attr
            if target_form is None and settings:
                target_form = settings.default_form
            if target_form is None:
                target_form = "form"
            
            # Check if forms hierarchy is defined in settings
            if settings and settings.form_hierarchy:
                # Use forms hierarchy with inheritance
                # Build the inheritance chain for the target form
                inheritance_chain = []
                current_form = target_form
                visited = set()  # Prevent infinite loops
                
                while current_form and current_form not in visited:
                    visited.add(current_form)
                    inheritance_chain.append(current_form)
                    # Get parent form from hierarchy
                    parent = settings.form_hierarchy.get(current_form)
                    if parent:
                        current_form = parent
                    else:
                        # No parent means this is a base form (like pform)
                        # Base form maps to innerText
                        break
                
                # Try each form in the inheritance chain (most specific first)
                for form_name in inheritance_chain:
                    form_val = elem.get(form_name, "")
                    if form_val and form_val.strip():
                        return form_val.strip()
                
                # If no form attribute found, base form (pform) maps to innerText
                inner_text = get_full_text_content(elem)
                if inner_text:
                    return inner_text
                return ""
            
            # Fallback: legacy hierarchy when forms not defined
            # Check if reg/expan attributes are defined in CQP pattributes
            use_legacy_hierarchy = False
            if settings:
                if "reg" in settings.defined_token_attributes or "expan" in settings.defined_token_attributes:
                    use_legacy_hierarchy = True
            
            if use_legacy_hierarchy:
                # Legacy hierarchy: @reg > @expan > @form > innerText
                # Get reg attributes
                reg_attrs = ["reg", "nform"]
                reg_mapping = settings.get_attribute_mapping("reg") if settings else []
                if reg_mapping:
                    reg_attrs = reg_mapping + reg_attrs
                
                for attr in reg_attrs:
                    reg_val = elem.get(attr)
                    if reg_val and reg_val.strip():
                        return reg_val.strip()
                
                # Get expan attributes
                expan_attrs = ["expan", "fform"]
                expan_mapping = settings.get_attribute_mapping("expan") if settings else []
                if expan_mapping:
                    expan_attrs = expan_mapping + expan_attrs
                
                for attr in expan_attrs:
                    expan_val = elem.get(attr)
                    if expan_val and expan_val.strip():
                        return expan_val.strip()
            
            # Try @form attribute
            form_attr_val = elem.get("form", "")
            if form_attr_val and form_attr_val.strip():
                return form_attr_val.strip()
            
            # Fall back to innerText
            inner_text = get_full_text_content(elem)
            if inner_text:
                return inner_text
            
            return ""
        
        # Get attribute mappings from settings to know which attributes to check for form matching
        # Check if reg or expan are used for normalization/expansion
        reg_attrs = ["reg", "nform"]
        expan_attrs = ["expan", "fform"]
        if settings:
            # Check if settings has reg/expan attribute mappings
            reg_mapping = settings.get_attribute_mapping("reg")
            if reg_mapping:
                reg_attrs = reg_mapping + reg_attrs
            expan_mapping = settings.get_attribute_mapping("expan")
            if expan_mapping:
                expan_attrs = expan_mapping + expan_attrs
        
        for node in sent_tok_nodes:
            # Use gettokform to get the canonical form based on settings hierarchy
            canonical_form = gettokform(node, settings)
            
            # Also collect all possible forms for matching (to handle cases where
            # different forms might match)
            full_text = get_full_text_content(node)
            form_attr = node.get("form", "")
            
            # Try reg/nform attributes
            reg_val = None
            for attr in reg_attrs:
                reg_val = node.get(attr)
                if reg_val:
                    break
            
            # Try expan/fform attributes
            expan_val = None
            for attr in expan_attrs:
                expan_val = node.get(attr)
                if expan_val:
                    break
            
            # Collect all possible forms for matching
            possible_forms = []
            # Add canonical form first (highest priority)
            if canonical_form:
                possible_forms.append(canonical_form)
            # Add other forms for fallback matching
            if full_text and full_text not in possible_forms:
                possible_forms.append(full_text)
            if form_attr and form_attr not in possible_forms:
                possible_forms.append(form_attr)
            if reg_val and reg_val not in possible_forms:
                possible_forms.append(reg_val)
            if expan_val and expan_val not in possible_forms:
                possible_forms.append(expan_val)
            
            # Store all forms in the mapping dictionary
            for node_form in possible_forms:
                if not node_form:
                    continue
                node_form_normalized = node_form.strip()
                if node_form_normalized:
                    if node_form_normalized not in xml_form_to_nodes:
                        xml_form_to_nodes[node_form_normalized] = []
                    xml_form_to_nodes[node_form_normalized].append(node)
        
        # Define verification function before first pass
        def verify_match(candidate_node: ET.Element, token_form: str) -> bool:
            """
            Verify that a candidate XML node exactly matches a token form. 
            Returns False if ANY mismatch - this is the CRITICAL check that prevents incorrect matches.
            """
            if not token_form:
                return False
            token_form = token_form.strip()
            if not token_form:
                return False
            
            candidate_text = get_full_text_content(candidate_node).strip()
            candidate_canonical = gettokform(candidate_node, settings).strip()
            
            # CRITICAL: Forms must match EXACTLY - same length AND same content
            # Reject if lengths don't match (prevents "con" matching "Español")
            if candidate_text and len(candidate_text) != len(token_form):
                return False
            if candidate_canonical and len(candidate_canonical) != len(token_form):
                return False
            
            # Check if candidate text exactly matches
            if candidate_text == token_form:
                return True
            # Check if candidate canonical form exactly matches
            if candidate_canonical == token_form:
                return True
            
            # NO MATCH - reject
            return False
        
        # First pass: match tokens that have TokIds (from CoNLL-U input)
        for token_idx, token in enumerate(sent.tokens):
            if not token.id:
                continue
            # Skip tokens that look like they came from comment lines
            if is_comment_line_token(token):
                skipped_tokens += 1
                continue
            if token.tokid:
                tok_node = tokid_to_node.get(token.tokid)
                if tok_node is not None and tok_node not in matched_xml_nodes:
                    # VERIFY: Even TokId-based matches must have matching forms
                    if token.form:
                        token_form_check = token.form.strip()
                        if token_form_check:
                            if not verify_match(tok_node, token_form_check):
                                # TokId matches but forms don't - skip this match
                                tok_node = None
                    
                    if tok_node is not None:
                        global_token_to_tok_node[(sent_idx, token_idx)] = tok_node
                        matched_xml_nodes.add(tok_node)
                        node_tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                        if node_tokid:
                            token.tokid = node_tokid
                            global_ord_to_tokid[token.id] = node_tokid
                        else:
                            tok_node.set("id", token.tokid)
                            tokid_to_node[token.tokid] = tok_node
                            global_ord_to_tokid[token.id] = token.tokid
        
        # Second pass (pretokenized mode only): match MWT tokens to multiple XML tokens
        # This handles cases where the backend creates MWTs - we match subtokens to individual XML tokens
        if not from_raw_text:
            for token_idx, token in enumerate(sent.tokens):
                if not token.id:
                    continue
                # Skip if already matched
                if (sent_idx, token_idx) in global_token_to_tok_node:
                    continue
                if is_comment_line_token(token):
                    continue
                
                # Only process MWT tokens in this pass
                if not (token.is_mwt and token.subtokens):
                    continue
                
                unmatched_nodes = [n for n in sent_tok_nodes if n not in matched_xml_nodes]
                
                # First, try to match the MWT token directly to an XML token that is also an MWT
                # (XML already has MWT structure from previous run)
                matched_mwt_node = None
                for candidate in unmatched_nodes:
                    # Check if candidate is an MWT (has dtok children)
                    has_dtoks = any(c.tag.endswith("}dtok") or c.tag == "dtok" for c in candidate)
                    if has_dtoks:
                        # Check if forms match
                        candidate_text = get_full_text_content(candidate).strip()
                        candidate_canonical = gettokform(candidate, settings).strip()
                        token_form = token.form.strip() if token.form else ""
                        
                        # Normalize for comparison
                        candidate_text_norm = candidate_text.lower()
                        candidate_canonical_norm = candidate_canonical.lower()
                        token_form_norm = token_form.lower()
                        
                        if (candidate_text_norm == token_form_norm or 
                            candidate_canonical_norm == token_form_norm):
                            matched_mwt_node = candidate
                            break
                
                if matched_mwt_node:
                    # XML already has MWT structure - match directly
                    tok_node = matched_mwt_node
                    matched_xml_nodes.add(tok_node)
                    global_token_to_tok_node[(sent_idx, token_idx)] = tok_node
                    # Don't set _mwt_subtoken_xml_nodes - XML already has dtoks, annotations will go there
                    continue
                
                # Second, try to match subtokens to individual XML tokens
                # (XML has individual tokens, backend has MWT)
                if len(token.subtokens) > len(unmatched_nodes):
                    continue  # Not enough unmatched nodes
                
                # Try to find a starting position where all subtokens match consecutively
                matched_nodes_for_mwt = None
                for start_pos in range(len(unmatched_nodes) - len(token.subtokens) + 1):
                    # Try matching subtokens starting at this position
                    candidate_matches = []
                    all_match = True
                    for sub_idx, subtoken in enumerate(token.subtokens):
                        candidate = unmatched_nodes[start_pos + sub_idx]
                        candidate_text = get_full_text_content(candidate).strip()
                        candidate_canonical = gettokform(candidate, settings).strip()
                        subtoken_form = subtoken.form.strip() if subtoken.form else ""
                        
                        # Normalize for comparison (case-insensitive)
                        candidate_text_norm = candidate_text.lower()
                        candidate_canonical_norm = candidate_canonical.lower()
                        subtoken_form_norm = subtoken_form.lower()
                        
                        # Check if subtoken matches this XML node
                        if (candidate_text_norm == subtoken_form_norm or 
                            candidate_canonical_norm == subtoken_form_norm):
                            candidate_matches.append(candidate)
                        else:
                            all_match = False
                            break
                    
                    if all_match and len(candidate_matches) == len(token.subtokens):
                        matched_nodes_for_mwt = candidate_matches
                        break
                
                if matched_nodes_for_mwt:
                    # Match all subtokens to XML nodes
                    # Map the MWT token to the first XML node
                    tok_node = matched_nodes_for_mwt[0]
                    # Mark ALL matched XML nodes as matched
                    for xml_node in matched_nodes_for_mwt:
                        matched_xml_nodes.add(xml_node)
                    global_token_to_tok_node[(sent_idx, token_idx)] = tok_node
                    
                    # Store mapping from subtoken index to XML node for later annotation
                    if "_mwt_subtoken_xml_nodes" not in token.attrs:
                        token.attrs["_mwt_subtoken_xml_nodes"] = {}
                    for sub_idx, xml_node in enumerate(matched_nodes_for_mwt):
                        if sub_idx > 0:  # First subtoken maps to tok_node (already set above)
                            token.attrs["_mwt_subtoken_xml_nodes"][sub_idx] = xml_node
        
        # Third pass: match remaining tokens using form-based alignment with split handling
        # COMPLETELY REWRITTEN: Simple, strict matching that NEVER accepts incorrect matches
        matched_backend_token_indices = set()
        
        for token_idx, token in enumerate(sent.tokens):
            if not token.id:
                continue
            # Skip if already matched
            if (sent_idx, token_idx) in global_token_to_tok_node:
                matched_backend_token_indices.add(token_idx)
                continue
            if token_idx in matched_backend_token_indices:
                continue
            if is_comment_line_token(token):
                skipped_tokens += 1
                continue
            
            if not token.form:
                continue
            
            token_form = token.form.strip()
            if not token_form:
                continue
            
            # Skip MWT tokens in pretokenized mode only if they were successfully matched in second pass
            if token.is_mwt and token.subtokens and not from_raw_text:
                if (sent_idx, token_idx) in global_token_to_tok_node:
                    # Already matched in second pass, skip
                    continue
            
            tok_node = None
            matched_form = None  # Track which form matched (for verification)
            unmatched_nodes = [n for n in sent_tok_nodes if n not in matched_xml_nodes]
            
            # Strategy 1: Try sequences of tokens (for split tokens like "awesome-align" -> "awesome", "-", "align")
            # Try combining current token with following tokens until we find an exact match
            max_lookahead = min(10 if from_raw_text else 5, len(sent.tokens) - token_idx - 1)
            
            for lookahead in range(max_lookahead + 1):
                if token_idx + lookahead >= len(sent.tokens):
                    continue
                # Skip if any token in sequence is already matched
                if any(token_idx + i in matched_backend_token_indices for i in range(lookahead + 1)):
                    continue
                
                # Build combined form
                combined_forms = [sent.tokens[token_idx + i].form for i in range(lookahead + 1)]
                combined_no_space = "".join(combined_forms)
                combined_with_hyphen = "-".join(combined_forms)
                
                # Try to find exact match in unmatched XML nodes
                for candidate in unmatched_nodes:
                    candidate_text = get_full_text_content(candidate).strip()
                    candidate_canonical = gettokform(candidate, settings).strip()
                    
                    # Check exact match with combined_no_space
                    if candidate_text == combined_no_space or candidate_canonical == combined_no_space:
                        # CRITICAL VERIFICATION: Forms must match EXACTLY
                        if verify_match(candidate, combined_no_space):
                            # DOUBLE-CHECK: Verify again after assignment
                            if candidate_text == combined_no_space or candidate_canonical == combined_no_space:
                                tok_node = candidate
                                matched_form = combined_no_space
                                matched_xml_nodes.add(candidate)
                                global_token_to_tok_node[(sent_idx, token_idx)] = candidate
                                matched_backend_token_indices.add(token_idx)
                                # Mark remaining tokens in sequence as matched to same node (will become dtok)
                                for i in range(1, lookahead + 1):
                                    if token_idx + i < len(sent.tokens):
                                        global_token_to_tok_node[(sent_idx, token_idx + i)] = candidate
                                        matched_backend_token_indices.add(token_idx + i)
                                break
                    # Check exact match with combined_with_hyphen
                    elif candidate_text == combined_with_hyphen or candidate_canonical == combined_with_hyphen:
                        # CRITICAL VERIFICATION: Forms must match EXACTLY
                        if verify_match(candidate, combined_with_hyphen):
                            # DOUBLE-CHECK: Verify again after assignment
                            if candidate_text == combined_with_hyphen or candidate_canonical == combined_with_hyphen:
                                tok_node = candidate
                                matched_form = combined_with_hyphen
                                matched_xml_nodes.add(candidate)
                                global_token_to_tok_node[(sent_idx, token_idx)] = candidate
                                matched_backend_token_indices.add(token_idx)
                                for i in range(1, lookahead + 1):
                                    if token_idx + i < len(sent.tokens):
                                        global_token_to_tok_node[(sent_idx, token_idx + i)] = candidate
                                        matched_backend_token_indices.add(token_idx + i)
                                break
                
                if tok_node is not None:
                    break
            
            # Strategy 2: Try exact single-token match (only if sequence matching failed)
            if tok_node is None:
                for candidate in unmatched_nodes:
                    if verify_match(candidate, token_form):
                        tok_node = candidate
                        matched_form = token_form
                        matched_xml_nodes.add(candidate)
                        global_token_to_tok_node[(sent_idx, token_idx)] = candidate
                        matched_backend_token_indices.add(token_idx)
                        break
            
            # FINAL VERIFICATION: Reject ANY match that doesn't exactly match
            # This is the absolute last check - if forms don't match, reject the match
            # CRITICAL: This check MUST run for EVERY match, no exceptions
            if tok_node is not None and token.form:
                token_form_check = token.form.strip()
                if token_form_check:  # Only verify if we have a form
                    candidate_text = get_full_text_content(tok_node).strip()
                    candidate_canonical = gettokform(tok_node, settings).strip()
                    
                    # Determine expected form
                    expected_form = matched_form if matched_form else token_form_check
                    
                    # CRITICAL: Forms must match EXACTLY - no exceptions
                    # Reject if lengths don't match
                    if candidate_text and len(candidate_text) != len(expected_form):
                        # Match failed verification - reject it
                        tok_node = None
                        matched_form = None
                        # Remove from mappings
                        if (sent_idx, token_idx) in global_token_to_tok_node:
                            matched_xml_nodes.discard(global_token_to_tok_node[(sent_idx, token_idx)])
                            del global_token_to_tok_node[(sent_idx, token_idx)]
                        matched_backend_token_indices.discard(token_idx)
                    elif candidate_canonical and len(candidate_canonical) != len(expected_form):
                        # Match failed verification - reject it
                        tok_node = None
                        matched_form = None
                        # Remove from mappings
                        if (sent_idx, token_idx) in global_token_to_tok_node:
                            matched_xml_nodes.discard(global_token_to_tok_node[(sent_idx, token_idx)])
                            del global_token_to_tok_node[(sent_idx, token_idx)]
                        matched_backend_token_indices.discard(token_idx)
                    # Reject if content doesn't match exactly
                    elif candidate_text != expected_form and candidate_canonical != expected_form:
                        # Match failed verification - reject it
                        tok_node = None
                        matched_form = None
                        # Remove from mappings
                        if (sent_idx, token_idx) in global_token_to_tok_node:
                            matched_xml_nodes.discard(global_token_to_tok_node[(sent_idx, token_idx)])
                            del global_token_to_tok_node[(sent_idx, token_idx)]
                        matched_backend_token_indices.discard(token_idx)
            
            # Only assign tokid if we have a valid match
            if tok_node is not None:
                node_tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                if node_tokid:
                    token.tokid = node_tokid
                    global_ord_to_tokid[token.id] = node_tokid
                elif not token.tokid:
                    new_tokid = f"w-{global_tokid_counter}"
                    global_tokid_counter += 1
                    token.tokid = new_tokid
                    tok_node.set("id", new_tokid)
                    tokid_to_node[new_tokid] = tok_node
                    global_ord_to_tokid[token.id] = new_tokid
            elif token.id:
                # Token has no matching XML node - still assign tokid for mapping
                if not token.tokid:
                    new_tokid = f"w-{global_tokid_counter}"
                    global_tokid_counter += 1
                    token.tokid = new_tokid
                global_ord_to_tokid[token.id] = token.tokid
    
    # Final pass: ensure ALL tokens are in the mapping (even if they weren't matched to XML nodes)
    # Also build reverse mapping (ord -> token) for efficient head lookup
    ord_to_token: Dict[int, Token] = {}
    for sent in sanitized.sentences:
        for token in sent.tokens:
            if token.id:
                ord_to_token[token.id] = token
                if token.id not in global_ord_to_tokid:
                    # Token not in mapping - ensure it has a tokid and add it
                    if not token.tokid:
                        token.tokid = f"w-{global_tokid_counter}"
                        global_tokid_counter += 1
                    global_ord_to_tokid[token.id] = token.tokid
    
    # Now process sentences for updates
    for sent_idx, sent in enumerate(sanitized.sentences):
        sent_id = sent.sent_id or sent.id or sent.source_id
        s_node = None
        
        # Try exact match first
        if sent_id:
            s_node = sentid_to_node.get(sent_id)
        
        # If exact match failed, try position-based matching
        if s_node is None and sent_idx < len(sentence_nodes_ordered):
            s_node = sentence_nodes_ordered[sent_idx]
        
        if s_node is None:
            # Sentence not found - skip
            skipped_sentences += 1
            continue
        matched_sentences += 1
        
        # Ensure sentence has an ID - assign one if missing
        if not s_node.get("id") and not s_node.get("{http://www.w3.org/XML/1998/namespace}id"):
            # Generate sentence ID
            sent_id = f"s-{sent_idx + 1}"
            s_node.set("id", sent_id)
            sentid_to_node[sent_id] = s_node
        
        # Update sentence-level attributes if needed
        if sent.text and not s_node.get("text"):
            s_node.set("text", sent.text)
        if sent.corr and not s_node.get("corr"):
            s_node.set("corr", sent.corr)
        
        # Update tokens using pre-built mapping
        # Track which XML nodes have been updated and which backend tokens are part of splits
        updated_xml_nodes = set()
        # Track which backend tokens are part of a split sequence (will become dtok elements)
        split_token_indices = set()
        for token_idx, token in enumerate(sent.tokens):
            tok_node = global_token_to_tok_node.get((sent_idx, token_idx))
            if tok_node is None:
                skipped_tokens += 1
                # Debug output for mismatched tokens
                import sys
                sent_id_str = sent.sent_id or sent.id or sent.source_id or f"s-{sent_idx+1}"
                print(f"[flexipipe] update_teitok: Token mismatch - sent {sent_idx+1} (id={sent_id_str}), token {token_idx+1} (ord={token.id}, tokid={token.tokid}, form='{token.form}') has no matching XML node", file=sys.stderr)
                continue
            
            # Check if this backend token is part of a split sequence (will be handled as dtok)
            # This can happen in both raw text mode and pretokenized mode when backends split tokens
            tokens_for_this_node = [(s, t) for (s, t), node in global_token_to_tok_node.items() 
                                   if node == tok_node and s == sent_idx]
            if len(tokens_for_this_node) > 1:
                # Multiple backend tokens map to this XML node - this is a split
                sorted_tokens = sorted(tokens_for_this_node, key=lambda x: x[1])
                # The first token updates the main <tok> element
                # The remaining tokens become <dtok> elements
                if token_idx != sorted_tokens[0][1]:
                    # This is not the first token in the sequence - it will be a dtok
                    split_token_indices.add(token_idx)
                    continue  # Skip updating this token directly, it will be handled as dtok
            
            # When multiple backend tokens map to the same XML node (token splits),
            # only update the XML node once (using the first token in the sequence)
            # Check if this XML node has already been updated
            if tok_node in updated_xml_nodes:
                # This XML node was already updated by a previous token in the sequence
                # Skip updating it again to avoid overwriting attributes
                continue
            
            matched_tokens += 1
            updated_xml_nodes.add(tok_node)
            
            # Update token attributes
            def _set_attr(node: ET.Element, internal_attr: str, value: str, default_empty: bool = False) -> None:
                """Set attribute respecting TEITOK mappings. Normalizes text attributes (form, lemma, etc.) but not tags."""
                target_attr = _resolve_attr_name(internal_attr)
                aliases = _attribute_aliases(internal_attr)
                
                # Normalize text attributes (form, lemma, reg, expan, mod, trslit, ltrslit) but not tags (xpos, upos, feats)
                text_attrs = {"form", "lemma", "reg", "expan", "mod", "trslit", "ltrslit"}
                if value and value != "_" and internal_attr.lower() in text_attrs:
                    value = _normalize_annotation_attr(value, unicode_normalization)
                
                if target_attr is None:
                    for alias in aliases:
                        node.attrib.pop(alias, None)
                    return
                if value and value != "_":
                    node.set(target_attr, value)
                    for alias in aliases:
                        if alias != target_attr:
                            node.attrib.pop(alias, None)
                elif default_empty:
                    for alias in aliases:
                        node.attrib.pop(alias, None)
                elif target_attr != internal_attr:
                    for alias in aliases:
                        if alias != target_attr:
                            node.attrib.pop(alias, None)
            
            # Update form - only write @form if it differs from inner XML text
            # This avoids redundancy when form matches the element content
            inner_text = (tok_node.text or "").strip()
            # Check if element has children (if so, form might differ from inner text)
            has_children = len(tok_node) > 0
            
            # When multiple backend tokens map to the same XML node (token splits),
            # we need to create <dtok> elements for the split tokens
            # This can happen in both raw text mode and pretokenized mode when backends split tokens
            # Check if this XML node is mapped to multiple backend tokens
            tokens_for_this_node = [(s, t) for (s, t), node in global_token_to_tok_node.items() 
                                   if node == tok_node and s == sent_idx]
            split_tokens = []
            if len(tokens_for_this_node) > 1:
                # Multiple backend tokens map to this XML node - this is a split
                sorted_tokens = sorted(tokens_for_this_node, key=lambda x: x[1])  # Sort by token index
                # The first token updates the main <tok> element
                # The remaining tokens become <dtok> elements
                split_tokens = [sent.tokens[t] for s, t in sorted_tokens[1:] if s == sent_idx and t < len(sent.tokens)]
                # Combine forms for the main token text content
                combined_form = "".join(sent.tokens[t].form for s, t in sorted_tokens if s == sent_idx and t < len(sent.tokens))
                # Try hyphenated version too (for cases like "awesome-align")
                combined_form_hyphen = "-".join(sent.tokens[t].form for s, t in sorted_tokens if s == sent_idx and t < len(sent.tokens))
                # Use the form that matches the original XML text, or the hyphenated version
                if inner_text and combined_form_hyphen == inner_text:
                    combined_form = combined_form_hyphen
                elif inner_text and combined_form == inner_text:
                    pass  # Already correct
                # Update text content with combined form
                if combined_form and combined_form != inner_text:
                    tok_node.text = combined_form
                    inner_text = combined_form
            
            # For split tokens, skip setting attributes on parent <tok> - they go on <dtok> elements
            # Check if this is a split token before setting any attributes
            is_split = len(tokens_for_this_node) > 1
            
            if not is_split:
                # Not a split token - set attributes normally on <tok>
                # Only write @form if it's different from inner text or if there are children
                if token.form:
                    if has_children or token.form != inner_text:
                        tok_node.set("form", _normalize_annotation_attr(token.form, unicode_normalization))
                    elif tok_node.get("form") and token.form == inner_text:
                        # Remove @form if it matches inner text and there are no children
                        tok_node.attrib.pop("form", None)
                
                # Set ord attribute (CoNLL-U ordinal number) - use _set_attr to respect known_tags_only
                ord_val = token.attrs.get("ord") or (str(token.id) if token.id else "")
                if ord_val:
                    _set_attr(tok_node, "ord", ord_val)
            
            # Ensure tokid is preserved in the XML
            # Prefer id over xml:id when writing (unless source explicitly had xml:id and not id)
            if token.tokid:
                current_tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                # Check if original XML had xml:id but not id (preserve that preference)
                has_xml_id = "{http://www.w3.org/XML/1998/namespace}id" in tok_node.attrib
                has_id = "id" in tok_node.attrib
                original_had_xml_id_only = has_xml_id and not has_id
                
                if current_tokid != token.tokid or (has_xml_id and not has_id and not original_had_xml_id_only):
                    # Update XML ID to match token's tokid
                    # Prefer id over xml:id, unless original had xml:id only
                    if original_had_xml_id_only:
                        # Original had xml:id only, preserve that
                        tok_node.set("{http://www.w3.org/XML/1998/namespace}id", token.tokid)
                        if "id" in tok_node.attrib:
                            del tok_node.attrib["id"]
                    else:
                        # Prefer id over xml:id
                        tok_node.set("id", token.tokid)
                        if "{http://www.w3.org/XML/1998/namespace}id" in tok_node.attrib:
                            del tok_node.attrib["{http://www.w3.org/XML/1998/namespace}id"]
                    # Update mapping if tokid changed
                    if current_tokid and current_tokid in tokid_to_node and current_tokid != token.tokid:
                        del tokid_to_node[current_tokid]
                    tokid_to_node[token.tokid] = tok_node
            
            # Handle split tokens: create <dtok> elements for ALL tokens in the split (including the first one)
            # This applies to both raw text mode and pretokenized mode when backends split tokens
            # When tokens are split, ALL annotations go on <dtok> elements, not on the parent <tok>
            if split_tokens:
                # Get all tokens in the split sequence (including the first one)
                all_split_tokens = [sent.tokens[t] for s, t in sorted_tokens if s == sent_idx and t < len(sent.tokens)]
                
                # Get existing dtok nodes (but don't count MWT dtok nodes if any)
                dtok_nodes = [c for c in tok_node if (c.tag.endswith("}dtok") or c.tag == "dtok")]
                base_tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id") or token.tokid
                
                # Create or update dtok elements for ALL tokens in the split
                # Start after any existing MWT dtok elements
                existing_mwt_dtok_count = len([c for c in tok_node if (c.tag.endswith("}dtok") or c.tag == "dtok")])
                for idx, split_token in enumerate(all_split_tokens):
                    dtok_idx = existing_mwt_dtok_count + idx
                    if dtok_idx < len(dtok_nodes):
                        dtok_node = dtok_nodes[dtok_idx]
                    else:
                        # Create new dtok element
                        dtok_node = ET.Element("dtok")
                        tok_node.append(dtok_node)
                    
                    # Set dtok ID (w-13.1, w-13.2, w-13.3, etc.)
                    dtok_tokid = f"{base_tokid}.{idx+1}"
                    dtok_node.set("id", dtok_tokid)
                    
                    # Update dtok attributes from split token using _set_attr for proper mapping
                    # Check both direct fields and attrs dict (backends may store in either)
                    lemma_val = split_token.lemma or split_token.attrs.get("lemma", "")
                    xpos_val = split_token.xpos or split_token.attrs.get("xpos", "")
                    upos_val = split_token.upos or split_token.attrs.get("upos", "")
                    feats_val = split_token.feats or split_token.attrs.get("feats", "")
                    deprel_val = split_token.deprel or split_token.attrs.get("deprel", "")
                    misc_val = split_token.misc or split_token.attrs.get("misc", "")
                    
                    # Use _set_attr to respect TEITOK attribute mappings (e.g., xpos -> pos)
                    if split_token.form:
                        dtok_node.set("form", _normalize_annotation_attr(split_token.form, unicode_normalization))
                    _set_attr(dtok_node, "lemma", lemma_val, default_empty=True)
                    _set_attr(dtok_node, "xpos", xpos_val)
                    _set_attr(dtok_node, "upos", upos_val)
                    _set_attr(dtok_node, "feats", feats_val)
                    _set_attr(dtok_node, "reg", split_token.reg)
                    _set_attr(dtok_node, "expan", split_token.expan)
                    _set_attr(dtok_node, "mod", split_token.mod)
                    _set_attr(dtok_node, "trslit", split_token.trslit)
                    _set_attr(dtok_node, "ltrslit", split_token.ltrslit)
                    
                    # Set ord attribute (CoNLL-U ordinal number) - use _set_attr to respect known_tags_only
                    ord_val = split_token.attrs.get("ord") or (str(split_token.id) if split_token.id else "")
                    if ord_val:
                        _set_attr(dtok_node, "ord", ord_val)
                    
                    # Handle head value - convert from ord to tokid
                    head_int = split_token.head
                    if head_int == 0 and "head" in split_token.attrs:
                        head_attr = split_token.attrs["head"]
                        if isinstance(head_attr, int):
                            head_int = head_attr
                        elif isinstance(head_attr, str):
                            try:
                                head_int = int(head_attr)
                            except (ValueError, TypeError):
                                head_int = 0
                    
                    if head_int > 0:
                        head_tokid = global_ord_to_tokid.get(head_int)
                        if not head_tokid:
                            head_token = ord_to_token.get(head_int)
                            if head_token and head_token.tokid:
                                head_tokid = head_token.tokid
                                global_ord_to_tokid[head_int] = head_tokid
                        if head_tokid:
                            _set_attr(dtok_node, "head", head_tokid)
                        else:
                            # Last resort: use ord if tokid not found (shouldn't happen)
                            _set_attr(dtok_node, "head", str(head_int))
                    else:
                        _set_attr(dtok_node, "head", "", default_empty=True)
                    
                    if deprel_val:
                        _set_attr(dtok_node, "deprel", deprel_val)
                    else:
                        _set_attr(dtok_node, "deprel", "", default_empty=True)
                    
                    if misc_val:
                        _set_attr(dtok_node, "misc", misc_val)
                
                # For split tokens, remove all attributes from parent <tok> except id
                # The parent <tok> should only have id and text content - all annotations go on <dtok>
                parent_id = tok_node.get("id")
                parent_xml_id = tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                # Clear all attributes
                tok_node.attrib.clear()
                # Restore only the id attribute(s)
                if parent_id:
                    tok_node.set("id", parent_id)
                if parent_xml_id and not parent_id:
                    tok_node.set("{http://www.w3.org/XML/1998/namespace}id", parent_xml_id)
                
                # Skip the rest of the attribute setting for this token
                continue
            
            # For MWT tokens where XML already has MWT structure (has dtoks)
            # Apply subtoken annotations to existing dtoks
            if token.is_mwt and token.subtokens and not from_raw_text:
                # Check if XML node already has dtoks
                xml_has_dtoks = any(c.tag.endswith("}dtok") or c.tag == "dtok" for c in tok_node)
                if xml_has_dtoks:
                    # XML already has MWT structure - apply annotations to existing dtoks
                    dtok_nodes = [c for c in tok_node if (c.tag.endswith("}dtok") or c.tag == "dtok")]
                    for sub_idx, subtoken in enumerate(token.subtokens):
                        if sub_idx >= len(dtok_nodes):
                            break
                        dtok_node = dtok_nodes[sub_idx]
                        
                        # Apply subtoken annotations to dtok
                        dtok_lemma = subtoken.lemma or subtoken.attrs.get("lemma", "")
                        dtok_xpos = subtoken.xpos or subtoken.attrs.get("xpos", "")
                        dtok_upos = subtoken.upos or subtoken.attrs.get("upos", "")
                        dtok_feats = subtoken.feats or subtoken.attrs.get("feats", "")
                        dtok_deprel = subtoken.attrs.get("deprel", "")
                        dtok_deps = subtoken.attrs.get("deps", "")
                        
                        _set_attr(dtok_node, "lemma", dtok_lemma, default_empty=True)
                        _set_attr(dtok_node, "xpos", dtok_xpos)
                        _set_attr(dtok_node, "upos", dtok_upos)
                        _set_attr(dtok_node, "feats", dtok_feats)
                        
                        # Set ord attribute
                        if subtoken.id:
                            _set_attr(dtok_node, "ord", str(subtoken.id))
                        
                        # Set head attribute (convert from ord to tokid)
                        head_int = 0
                        if "head" in subtoken.attrs:
                            head_attr = subtoken.attrs["head"]
                            if isinstance(head_attr, int):
                                head_int = head_attr
                            elif isinstance(head_attr, str):
                                try:
                                    head_int = int(head_attr)
                                except (ValueError, TypeError):
                                    head_int = 0
                        
                        if head_int > 0:
                            head_tokid = global_ord_to_tokid.get(head_int)
                            if not head_tokid:
                                head_token = ord_to_token.get(head_int)
                                if head_token and head_token.tokid:
                                    head_tokid = head_token.tokid
                                    global_ord_to_tokid[head_int] = head_tokid
                            if head_tokid:
                                _set_attr(dtok_node, "head", head_tokid)
                            else:
                                _set_attr(dtok_node, "head", str(head_int))
                        else:
                            _set_attr(dtok_node, "head", "", default_empty=True)
                        
                        if dtok_deprel:
                            _set_attr(dtok_node, "deprel", dtok_deprel)
                        if dtok_deps:
                            _set_attr(dtok_node, "deps", dtok_deps)
                        
                        misc_val = subtoken.attrs.get("misc", "")
                        if misc_val:
                            _set_attr(dtok_node, "misc", misc_val)
                    
                    # Skip the rest of the attribute setting for this token
                    continue
            
            # For MWT tokens that were matched to multiple XML tokens (pretokenized mode)
            # Apply each subtoken's annotations to its corresponding XML token
            if token.is_mwt and token.subtokens and not from_raw_text and "_mwt_subtoken_xml_nodes" in token.attrs:
                # This MWT was matched to multiple XML tokens - apply annotations directly to those tokens
                subtoken_xml_nodes = token.attrs["_mwt_subtoken_xml_nodes"]
                for sub_idx, subtoken in enumerate(token.subtokens):
                    if sub_idx == 0:
                        # First subtoken - apply to the main XML token (tok_node)
                        xml_node = tok_node
                    else:
                        # Remaining subtokens - apply to their respective XML tokens
                        xml_node = subtoken_xml_nodes.get(sub_idx)
                        if not xml_node:
                            continue
                    
                    # Apply subtoken annotations to this XML node
                    dtok_lemma = subtoken.lemma or subtoken.attrs.get("lemma", "")
                    dtok_xpos = subtoken.xpos or subtoken.attrs.get("xpos", "")
                    dtok_upos = subtoken.upos or subtoken.attrs.get("upos", "")
                    dtok_feats = subtoken.feats or subtoken.attrs.get("feats", "")
                    dtok_deprel = subtoken.attrs.get("deprel", "")
                    dtok_deps = subtoken.attrs.get("deps", "")
                    
                    _set_attr(xml_node, "lemma", dtok_lemma, default_empty=True)
                    _set_attr(xml_node, "xpos", dtok_xpos)
                    _set_attr(xml_node, "upos", dtok_upos)
                    _set_attr(xml_node, "feats", dtok_feats)
                    
                    # Set ord attribute
                    if subtoken.id:
                        _set_attr(xml_node, "ord", str(subtoken.id))
                    
                    # Set head attribute (convert from ord to tokid)
                    head_int = 0
                    if "head" in subtoken.attrs:
                        head_attr = subtoken.attrs["head"]
                        if isinstance(head_attr, int):
                            head_int = head_attr
                        elif isinstance(head_attr, str):
                            try:
                                head_int = int(head_attr)
                            except (ValueError, TypeError):
                                head_int = 0
                    
                    if head_int > 0:
                        head_tokid = global_ord_to_tokid.get(head_int)
                        if not head_tokid:
                            head_token = ord_to_token.get(head_int)
                            if head_token and head_token.tokid:
                                head_tokid = head_token.tokid
                                global_ord_to_tokid[head_int] = head_tokid
                        if head_tokid:
                            _set_attr(xml_node, "head", head_tokid)
                        else:
                            _set_attr(xml_node, "head", str(head_int))
                    else:
                        _set_attr(xml_node, "head", "", default_empty=True)
                    
                    if dtok_deprel:
                        _set_attr(xml_node, "deprel", dtok_deprel)
                    if dtok_deps:
                        _set_attr(xml_node, "deps", dtok_deps)
                    
                    misc_val = subtoken.attrs.get("misc", "")
                    if misc_val:
                        _set_attr(xml_node, "misc", misc_val)
                
                # Skip the rest of the attribute setting for this token
                continue
            
            # For MWT tokens that match a single XML token (raw text mode), update the <tok> element but annotations go on <dtok>
            if token.is_mwt and token.subtokens:
                # Update <tok> attributes (form, reg, expan, etc. but not lemma/xpos/upos)
                _set_attr(tok_node, "reg", token.reg)
                _set_attr(tok_node, "expan", token.expan)
                _set_attr(tok_node, "mod", token.mod)
                _set_attr(tok_node, "trslit", token.trslit)
                _set_attr(tok_node, "ltrslit", token.ltrslit)
                _set_attr(tok_node, "feats", token.feats)
                
                # Get the actual XML node's ID to use as base for dtok IDs (not token.tokid)
                parent_tokid = tok_node.get("id") or tok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                if not parent_tokid:
                    # Fallback to token.tokid if XML node has no ID
                    parent_tokid = token.tokid or f"w-{token.id}"
                    # Set it on the XML node
                    tok_node.set("id", parent_tokid)
                
                # Extract base tokid (remove any existing -dtok or . suffix)
                base_tokid = parent_tokid
                if "-dtok" in base_tokid:
                    # Old format like w-30-dtok2, extract just w-30
                    base_tokid = base_tokid.split("-dtok")[0]
                elif "." in base_tokid:
                    # Already has a dot, extract base (e.g., w-30.1 -> w-30)
                    base_tokid = base_tokid.split(".")[0]
                
                # Update <dtok> children - create new ones if needed
                dtok_nodes = [c for c in tok_node if (c.tag.endswith("}dtok") or c.tag == "dtok")]
                for idx, subtoken in enumerate(token.subtokens):
                    # Create dtok node if it doesn't exist
                    if idx < len(dtok_nodes):
                        dtok_node = dtok_nodes[idx]
                    else:
                        # Create new dtok element
                        dtok_node = ET.Element("dtok")
                        tok_node.append(dtok_node)
                    
                    # Generate dtok ID in format w-30.1, w-30.2 (always from parent XML node's ID)
                    dtok_tokid = f"{base_tokid}.{idx+1}"
                    
                    # Check for duplicate IDs and resolve them
                    if dtok_tokid in tokid_to_node and tokid_to_node[dtok_tokid] is not dtok_node:
                        # Duplicate ID found - modify this one
                        # Try appending a suffix until we find a unique ID
                        suffix = 1
                        while f"{base_tokid}.{idx+1}.{suffix}" in tokid_to_node:
                            suffix += 1
                        dtok_tokid = f"{base_tokid}.{idx+1}.{suffix}"
                    
                    # Always ensure dtok has the correct ID (prefer id over xml:id)
                    # Remove old ID from mapping if it exists (even if it's the old format)
                    current_dtok_id = dtok_node.get("id") or dtok_node.get("{http://www.w3.org/XML/1998/namespace}id")
                    if current_dtok_id and current_dtok_id != dtok_tokid:
                        if current_dtok_id in tokid_to_node and tokid_to_node[current_dtok_id] is dtok_node:
                            del tokid_to_node[current_dtok_id]
                        # Set new ID (prefer id over xml:id)
                        if dtok_node.get("{http://www.w3.org/XML/1998/namespace}id") and not dtok_node.get("id"):
                            dtok_node.set("{http://www.w3.org/XML/1998/namespace}id", dtok_tokid)
                        else:
                            dtok_node.set("id", dtok_tokid)
                    elif not current_dtok_id:
                        # No existing ID, set it
                        dtok_node.set("id", dtok_tokid)
                    # Always update mapping with new format
                    tokid_to_node[dtok_tokid] = dtok_node
                    
                    # Check both direct fields and attrs dict (backends may store in either)
                    dtok_lemma = subtoken.lemma or subtoken.attrs.get("lemma", "")
                    dtok_xpos = subtoken.xpos or subtoken.attrs.get("xpos", "")
                    dtok_upos = subtoken.upos or subtoken.attrs.get("upos", "")
                    dtok_feats = subtoken.feats or subtoken.attrs.get("feats", "")
                    # SubToken doesn't have deprel/deps directly, only in attrs
                    dtok_deprel = subtoken.attrs.get("deprel", "")
                    dtok_deps = subtoken.attrs.get("deps", "")
                    
                    # Always set form (required for dtok)
                    _set_attr(dtok_node, "form", subtoken.form)
                    _set_attr(dtok_node, "lemma", dtok_lemma, default_empty=True)
                    _set_attr(dtok_node, "xpos", dtok_xpos)
                    _set_attr(dtok_node, "upos", dtok_upos)
                    _set_attr(dtok_node, "feats", dtok_feats)
                    _set_attr(dtok_node, "reg", subtoken.reg)
                    _set_attr(dtok_node, "expan", subtoken.expan)
                    _set_attr(dtok_node, "mod", subtoken.mod)
                    _set_attr(dtok_node, "trslit", subtoken.trslit)
                    _set_attr(dtok_node, "ltrslit", subtoken.ltrslit)
                    
                    # Set head and deprel on dtok (convert head from ord to tokid)
                    # SubToken doesn't have head directly, check attrs
                    dtok_head_int = 0
                    if "head" in subtoken.attrs:
                        head_attr = subtoken.attrs["head"]
                        if isinstance(head_attr, int):
                            dtok_head_int = head_attr
                        elif isinstance(head_attr, str):
                            try:
                                dtok_head_int = int(head_attr)
                            except (ValueError, TypeError):
                                dtok_head_int = 0
                    
                    if dtok_head_int > 0:
                        # Convert from ord to tokid using global mapping
                        dtok_head_tokid = global_ord_to_tokid.get(dtok_head_int)
                        if not dtok_head_tokid:
                            # Fallback: use reverse mapping for efficient lookup
                            dtok_head_token = ord_to_token.get(dtok_head_int)
                            if dtok_head_token and dtok_head_token.tokid:
                                dtok_head_tokid = dtok_head_token.tokid
                                global_ord_to_tokid[dtok_head_int] = dtok_head_tokid
                        
                        if dtok_head_tokid:
                            _set_attr(dtok_node, "head", dtok_head_tokid)
                        else:
                            # Last resort: use ord if tokid not found (shouldn't happen)
                            _set_attr(dtok_node, "head", str(dtok_head_int))
                    
                    if dtok_deprel:
                        _set_attr(dtok_node, "deprel", dtok_deprel)
                    if dtok_deps:
                        _set_attr(dtok_node, "deps", dtok_deps)
            else:
                # Non-MWT token - update all attributes on <tok>
                # Check both direct fields and attrs dict (backends may store in either)
                lemma_val = token.lemma or token.attrs.get("lemma", "")
                xpos_val = token.xpos or token.attrs.get("xpos", "")
                upos_val = token.upos or token.attrs.get("upos", "")
                feats_val = token.feats or token.attrs.get("feats", "")
                deprel_val = token.deprel or token.attrs.get("deprel", "")
                deps_val = token.deps or token.attrs.get("deps", "")
                misc_val = token.misc or token.attrs.get("misc", "")
                
                _set_attr(tok_node, "lemma", lemma_val, default_empty=True)
                _set_attr(tok_node, "xpos", xpos_val)
                _set_attr(tok_node, "upos", upos_val)
                _set_attr(tok_node, "feats", feats_val)
                _set_attr(tok_node, "reg", token.reg)
                _set_attr(tok_node, "expan", token.expan)
                _set_attr(tok_node, "mod", token.mod)
                _set_attr(tok_node, "trslit", token.trslit)
                _set_attr(tok_node, "ltrslit", token.ltrslit)
                
                # Update dependency attributes if present
                # Handle head value - convert from ord (CoNLL-U) to tokid (TEITOK)
                # CoNLL-U head refers to ord (token.id), but TEITOK @head should be tokid
                head_int = token.head
                if head_int == 0 and "head" in token.attrs:
                    head_attr = token.attrs["head"]
                    if isinstance(head_attr, int):
                        head_int = head_attr
                    elif isinstance(head_attr, str):
                        try:
                            head_int = int(head_attr)
                        except (ValueError, TypeError):
                            head_int = 0
                
                if head_int > 0:
                    # Convert from ord to tokid using global mapping
                    head_tokid = global_ord_to_tokid.get(head_int)
                    if not head_tokid:
                        # Fallback: use reverse mapping for efficient lookup
                        head_token = ord_to_token.get(head_int)
                        if head_token and head_token.tokid:
                            head_tokid = head_token.tokid
                            # Update mapping for future use
                            global_ord_to_tokid[head_int] = head_tokid
                    
                    if head_tokid:
                        _set_attr(tok_node, "head", head_tokid)
                    else:
                        # Last resort: use ord if tokid not found (shouldn't happen)
                        _set_attr(tok_node, "head", str(head_int))
                else:
                    _set_attr(tok_node, "head", "", default_empty=True)
                if deprel_val:
                    _set_attr(tok_node, "deprel", deprel_val)
                else:
                    _set_attr(tok_node, "deprel", "", default_empty=True)
                if deps_val:
                    _set_attr(tok_node, "deps", deps_val)
                else:
                    _set_attr(tok_node, "deps", "", default_empty=True)
                if misc_val:
                    _set_attr(tok_node, "misc", misc_val)
    
    # Verify alignment before writing
    import sys
    total_sentences = matched_sentences + skipped_sentences
    total_tokens = matched_tokens + skipped_tokens
    
    # Calculate alignment quality metrics
    sentence_match_rate = matched_sentences / total_sentences if total_sentences > 0 else 0.0
    token_match_rate = matched_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Determine if alignment is acceptable
    # For pretokenized mode (from_raw_text=False), we require near-perfect alignment (1:1 mapping)
    # For raw text mode (from_raw_text=True), we allow some flexibility but still need reasonable alignment
    if from_raw_text:
        # Raw text mode: allow some token splitting/merging, but still need >80% alignment
        min_sentence_rate = 0.95  # 95% of sentences must match
        min_token_rate = 0.80  # 80% of tokens must match (allows for splitting/merging)
        mode_str = "raw text"
    else:
        # Pretokenized mode: require near-perfect 1:1 alignment
        min_sentence_rate = 0.99  # 99% of sentences must match
        min_token_rate = 0.99  # 99% of tokens must match (1:1 mapping expected)
        mode_str = "pretokenized"
    
    alignment_acceptable = (
        sentence_match_rate >= min_sentence_rate and
        token_match_rate >= min_token_rate
    )
    
    # Print alignment summary
    if skipped_tokens > 0 or skipped_sentences > 0 or not alignment_acceptable:
        if skipped_sentences > 0:
            # Show what IDs we have vs what we're looking for
            doc_sent_ids = [f"{s.sent_id or s.id or s.source_id or 'NO_ID'}" for s in sanitized.sentences]
            xml_sent_ids = list(sentid_to_node.keys())
            print(f"[flexipipe] update_teitok: matched {matched_sentences}/{total_sentences} sentences ({sentence_match_rate*100:.1f}%), {skipped_sentences} skipped", file=sys.stderr)
            print(f"[flexipipe] Document sentence IDs: {doc_sent_ids}", file=sys.stderr)
            print(f"[flexipipe] XML sentence IDs: {xml_sent_ids}", file=sys.stderr)
        if skipped_tokens > 0:
            print(f"[flexipipe] update_teitok: matched {matched_tokens}/{total_tokens} tokens ({token_match_rate*100:.1f}%), {skipped_tokens} skipped (mode: {mode_str})", file=sys.stderr)
    
    # Perform character-level alignment verification
    char_alignment_valid, char_error_msg = _verify_character_level_alignment(
        sanitized, root, global_token_to_tok_node, from_raw_text
    )
    
    # Refuse to write if alignment is unacceptable and strict_alignment is enabled
    if strict_alignment and not alignment_acceptable:
        error_msg = (
            f"Token alignment verification failed (mode: {mode_str}). "
            f"Matched {matched_sentences}/{total_sentences} sentences ({sentence_match_rate*100:.1f}%, required: {min_sentence_rate*100:.1f}%) "
            f"and {matched_tokens}/{total_tokens} tokens ({token_match_rate*100:.1f}%, required: {min_token_rate*100:.1f}%). "
            f"Refusing to write to prevent incorrect annotations. "
            f"This may indicate tokenization mismatch between backend output and original XML. "
            f"Check if --use-raw-text is set correctly, or if the backend is retokenizing when it shouldn't."
        )
        if not char_alignment_valid:
            error_msg += f"\nCharacter-level alignment check also failed: {char_error_msg}"
        raise RuntimeError(error_msg)
    
    # Even if token alignment passes, check character-level alignment
    if strict_alignment and not char_alignment_valid:
        error_msg = (
            f"Character-level alignment verification failed (mode: {mode_str}). "
            f"Token alignment passed ({matched_tokens}/{total_tokens} tokens, {token_match_rate*100:.1f}%), "
            f"but character-level comparison revealed a mismatch. "
            f"{char_error_msg} "
            f"Refusing to write to prevent incorrect annotations."
        )
        raise RuntimeError(error_msg)
    elif not alignment_acceptable:
        # Non-strict mode: warn but continue
        print(
            f"[flexipipe] WARNING: Token alignment is below threshold (mode: {mode_str}). "
            f"Matched {matched_sentences}/{total_sentences} sentences ({sentence_match_rate*100:.1f}%) "
            f"and {matched_tokens}/{total_tokens} tokens ({token_match_rate*100:.1f}%). "
            f"Proceeding with partial update, but results may be incorrect.",
            file=sys.stderr
        )
    
    # Add change element to TEI header
    # Extract change information from document metadata
    from datetime import datetime
    change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Build change text from document metadata
    backends_used = document.meta.get("_backends_used", [])
    if not backends_used:
        backends_used = ["flexipipe"]
    
    # Get model information
    file_level_attrs = document.meta.get("_file_level_attrs", {})
    model_keys = sorted([k for k in file_level_attrs.keys() if k.endswith("_model")])
    model_str = None
    if model_keys:
        model_str = file_level_attrs[model_keys[0]]
    
    # Build backend string
    backend_names = [b.upper() for b in backends_used]
    change_source = ", ".join(backend_names) if len(backend_names) > 1 else (model_str or backend_names[0] if backend_names else "flexipipe")
    
    # Detect performed tasks
    tasks = set()
    if document.meta.get("_tokenized", False):
        tasks.add("tokenize")
    if document.meta.get("_segmented", False):
        tasks.add("segment")
    if any(t.lemma for s in document.sentences for t in s.tokens):
        tasks.add("lemmatize")
    if any(t.xpos or t.upos for s in document.sentences for t in s.tokens):
        tasks.add("tag")
    if any(t.head for s in document.sentences for t in s.tokens):
        tasks.add("parse")
    # NER is stored in sentence.entities, not on tokens
    if getattr(document, "spans", None) and document.spans.get("ner"):
        tasks.add("ner")
    elif any(s.entities for s in document.sentences):
        tasks.add("ner")
    # Normalization is detected from various token attributes
    if any(t.reg or t.expan or t.mod or t.trslit or t.ltrslit or t.corr or t.lex or (t.misc and t.misc != "_") for s in document.sentences for t in s.tokens):
        tasks.add("normalize")
    
    tasks_summary_str = ",".join(sorted(tasks)) if tasks else "segment,tokenize"
    change_text = f"Tagged via {change_source} (tasks={tasks_summary_str})"
    
    _add_change_to_tei_header(root, change_text, change_when)
    
    # Write updated XML
    if preserve_comments:
        # lxml preserves comments and formatting better
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True, pretty_print=True)
    else:
        # ElementTree - basic write
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True)

