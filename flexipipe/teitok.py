from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from .doc import Document, Sentence, SubToken, Token, Entity
from .doc_utils import collect_span_entities_by_sentence

# Import helper functions from engine
from .engine import (
    _apply_sentence_correspondence,
    _ensure_serializable,
    _infer_space_after_from_text,
    _sanitize_document,
    assign_doc_id_from_path,
)

XML_NS = "{http://www.w3.org/XML/1998/namespace}"

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
    
    # Look for flexitag_py in the flexitag/build directory relative to this file
    # Resolve __file__ to an absolute path first to handle cases where it might be relative
    resolved_file_path = Path(__file__).resolve()
    flexipipe_dir = resolved_file_path.parent.parent
    
    flexitag_build = flexipipe_dir / "flexitag" / "build"
    
    if flexitag_build.exists():
        build_path_str = str(flexitag_build.resolve())
        if build_path_str not in sys.path:
            sys.path.insert(0, build_path_str)
        
        # Check for Python version mismatch
        import sys
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
        except (ImportError, AttributeError) as exc:
            _import_error = exc

            def _missing_extension(*_: object, **__: object) -> None:
                error_msg = (
                    "flexitag_py extension is not available. "
                    "Build the flexitag project with pybind11 support or ensure it is on PYTHONPATH."
                )
                if version_mismatch:
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
        _dump_teitok = _missing_extension  # type: ignore


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
    # If attribute mappings are provided, use Python-based loading
    if xpos_attr or reg_attr or expan_attr or lemma_attr:
        doc = _load_teitok_with_mappings(
            path,
            xpos_attr=xpos_attr,
            reg_attr=reg_attr,
            expan_attr=expan_attr,
            lemma_attr=lemma_attr,
        )
    else:
        # Otherwise, use the fast C++ loader
        data = _load_teitok(path)  # type: ignore
        doc = Document.from_dict(data)
        _apply_sentence_correspondence(doc)
        # Fix space_after values by inferring from sentence text
        _infer_space_after_from_text(doc)
        doc = _sanitize_document(doc)
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
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        xml_start = content.find("<")
        if xml_start < 0:
            raise ET.ParseError("No XML content found in file")
        # Parse from the XML start
        from io import StringIO
        tree = ET.parse(StringIO(content[xml_start:]))
    root = tree.getroot()
    
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
            assigned_id = f"s{sentence_counter}"
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
            tokid = _get_attr_value_with_fallback(tok_elem, ["id", "xml:id"])
            
            # Get head, deprel, deps, misc if present
            head_str = tok_elem.get("head", "")
            head = int(head_str) if head_str and head_str.isdigit() else 0
            deprel = tok_elem.get("deprel", "")
            deps = tok_elem.get("deps", "")
            misc = tok_elem.get("misc", "")
            
            # Store all attributes from <tok> element in attrs (except those we use directly)
            tok_attrs: Dict[str, str] = {}
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
    )
    Path(path).write_text(xml_str, encoding="utf-8")


def dump_teitok(
    document: Document,
    custom_attributes: Optional[List[str]] = None,
    pretty_print: bool = False,
    *,
    spaceafter_handling: str = "preserve",
    skip_spaceafter_for_breaking_elements: bool = True,
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
    doc_dict = sanitized.to_dict()
    payload = _ensure_serializable(sanitized.to_dict())
    if custom_attributes is None:
        custom_attributes = []
    xml_str = _dump_teitok(payload, custom_attributes, pretty_print=False)  # type: ignore
    if pretty_print:
        xml_str = pretty_print_teitok_xml(
            xml_str,
            spaceafter_handling=spaceafter_handling,
            skip_spaceafter_for_breaking_elements=skip_spaceafter_for_breaking_elements,
        )
    return xml_str


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


def update_teitok(document: Document, original_path: str, output_path: Optional[str] = None) -> None:
    """
    Update a TEITOK XML file in-place by matching nodes by ID and updating annotation attributes.
    
    This function preserves the original XML structure, comments, and metadata while only
    updating the annotation attributes (xpos, lemma, upos, feats, head, deprel, etc.) on
    nodes that match by ID.
    
    Args:
        document: Tagged Document with updated annotations
        original_path: Path to the original TEITOK XML file
        output_path: Optional output path (defaults to original_path for in-place update)
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
    
    # Find all <s>, <tok>, and <dtok> nodes
    # Handle both namespaced and non-namespaced elements
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
                        sent_id = f"s{idx}"
                        s_node.set("id", sent_id)
                        sentid_to_node[sent_id] = s_node
    
    for tok_node in root.iter():
        if tok_node.tag.endswith("}tok") or tok_node.tag == "tok":
            tokid = tok_node.get("{http://www.w3.org/XML/1998/namespace}id") or tok_node.get("id")
            if not tokid:
                # Generate ID if missing
                parent = get_parent(tok_node)
                if parent is not None:
                    siblings = [c for c in parent if (c.tag.endswith("}tok") or c.tag == "tok")]
                    if tok_node in siblings:
                        idx = siblings.index(tok_node) + 1
                        tokid = f"w-{idx}"
                        tok_node.set("{http://www.w3.org/XML/1998/namespace}id", tokid)
            if tokid:
                tokid_to_node[tokid] = tok_node
    
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
    for sent in sanitized.sentences:
        sent_id = sent.sent_id or sent.id
        if not sent_id:
            continue
        
        s_node = sentid_to_node.get(sent_id)
        if s_node is None:
            # Sentence not found - skip
            continue
        
        # Update sentence-level attributes if needed
        if sent.text and not s_node.get("text"):
            s_node.set("text", sent.text)
        if sent.corr and not s_node.get("corr"):
            s_node.set("corr", sent.corr)
        
        # Update tokens
        for token in sent.tokens:
            tokid = token.tokid
            if not tokid:
                # Try to generate tokid from token.id
                tokid = f"w-{token.id}"
            
            tok_node = tokid_to_node.get(tokid)
            if tok_node is None:
                # Token not found - skip
                continue
            
            # Update token attributes
            def _set_attr(node: ET.Element, attr: str, value: str, default_empty: bool = False) -> None:
                """Set attribute if value is non-empty or default_empty is True."""
                if value and value != "_":
                    node.set(attr, value)
                elif default_empty and not value:
                    # Remove attribute if it exists and we're setting to empty
                    if attr in node.attrib:
                        del node.attrib[attr]
            
            # Update form if it changed (shouldn't normally, but handle it)
            if token.form and token.form != tok_node.get("form"):
                tok_node.set("form", token.form)
            
            # For MWT tokens, update the <tok> element but annotations go on <dtok>
            if token.is_mwt and token.subtokens:
                # Update <tok> attributes (form, reg, expan, etc. but not lemma/xpos/upos)
                _set_attr(tok_node, "reg", token.reg)
                _set_attr(tok_node, "expan", token.expan)
                _set_attr(tok_node, "mod", token.mod)
                _set_attr(tok_node, "trslit", token.trslit)
                _set_attr(tok_node, "ltrslit", token.ltrslit)
                _set_attr(tok_node, "feats", token.feats)
                
                # Update <dtok> children
                dtok_nodes = [c for c in tok_node if (c.tag.endswith("}dtok") or c.tag == "dtok")]
                for idx, subtoken in enumerate(token.subtokens):
                    if idx < len(dtok_nodes):
                        dtok_node = dtok_nodes[idx]
                        dtok_tokid = subtoken.tokid or f"{tokid}-dtok{idx+1}"
                        if dtok_tokid not in tokid_to_node:
                            # Ensure dtok has ID
                            dtok_node.set("{http://www.w3.org/XML/1998/namespace}id", dtok_tokid)
                            tokid_to_node[dtok_tokid] = dtok_node
                        
                        _set_attr(dtok_node, "form", subtoken.form)
                        _set_attr(dtok_node, "lemma", subtoken.lemma, default_empty=True)
                        _set_attr(dtok_node, "xpos", subtoken.xpos)
                        _set_attr(dtok_node, "upos", subtoken.upos)
                        _set_attr(dtok_node, "feats", subtoken.feats)
                        _set_attr(dtok_node, "reg", subtoken.reg)
                        _set_attr(dtok_node, "expan", subtoken.expan)
                        _set_attr(dtok_node, "mod", subtoken.mod)
                        _set_attr(dtok_node, "trslit", subtoken.trslit)
                        _set_attr(dtok_node, "ltrslit", subtoken.ltrslit)
            else:
                # Non-MWT token - update all attributes on <tok>
                _set_attr(tok_node, "lemma", token.lemma, default_empty=True)
                _set_attr(tok_node, "xpos", token.xpos)
                _set_attr(tok_node, "upos", token.upos)
                _set_attr(tok_node, "feats", token.feats)
                _set_attr(tok_node, "reg", token.reg)
                _set_attr(tok_node, "expan", token.expan)
                _set_attr(tok_node, "mod", token.mod)
                _set_attr(tok_node, "trslit", token.trslit)
                _set_attr(tok_node, "ltrslit", token.ltrslit)
            
            # Update dependency attributes if present
            if token.head > 0:
                tok_node.set("head", str(token.head))
            if token.deprel:
                tok_node.set("deprel", token.deprel)
            if token.deps:
                tok_node.set("deps", token.deps)
    
    # Write updated XML
    if preserve_comments:
        # lxml preserves comments and formatting better
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True, pretty_print=True)
    else:
        # ElementTree - basic write
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True)

