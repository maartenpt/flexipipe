"""
Version 3: Build plain-text with standoff representation first.

This approach:
1. Extracts plain text from XML (similar to xml2standoff)
2. Creates a standoff representation mapping XML elements to character positions
3. Maps tokens/sentences to character positions in the plain text
4. Rebuilds XML by inserting <s> and <tok> elements at the correct positions
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
import copy
import re

try:
    from lxml import etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAS_LXML = False

# Import Document and Token types
try:
    from .doc import Document, Token, Sentence
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from flexipipe.doc import Document, Token, Sentence


def insert_tokens_into_teitok(
    document: Document,
    original_path: str,
    output_path: Optional[str] = None,
    textnode_xpath: str = ".//text",
    include_notes: bool = False,
    block_elements: Optional[List[str]] = None,
    extract_elements: Optional[List[str]] = None,
    settings: Optional[Any] = None,
) -> None:
    """
    Insert tokens and sentences into a non-tokenized TEITOK XML file.
    
    This function uses a standoff representation approach:
    1. Extract plain text with standoff markup
    2. Map tokens/sentences to character positions
    3. Rebuild XML with <s> and <tok> elements inserted at correct positions
    """
    # Set defaults
    if block_elements is None:
        block_elements = ['div', 'head', 'p', 'u', 'speaker']
    if extract_elements is None:
        extract_elements = ['note', 'desc', 'gap', 'pb', 'fw', 'rdg']
    
    original_path_obj = Path(original_path)
    if not original_path_obj.exists():
        raise FileNotFoundError(f"Original TEITOK file not found: {original_path}")
    
    output_path_obj = Path(output_path) if output_path else original_path_obj
    
    # Parse original XML
    if HAS_LXML:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        tree = ET.parse(str(original_path_obj), parser)
        root = tree.getroot()
    else:
        tree = ET.parse(str(original_path_obj))
        root = tree.getroot()
    
    # Keep a pristine copy of the original XML for later verification
    original_root_snapshot = copy.deepcopy(root)
    
    # Find text nodes
    try:
        text_nodes = root.findall(textnode_xpath.replace(".//", ".//{*}"))
        if not text_nodes:
            text_nodes = root.findall(textnode_xpath)
    except (SyntaxError, ValueError):
        text_nodes = root.findall(textnode_xpath)
    
    if not text_nodes:
        text_nodes = [root]
    
    block_elements_set = {elem.lower() for elem in block_elements}
    extract_elements_set = {elem.lower() for elem in extract_elements}
    
    # Helper functions
    def get_tag_name(elem: ET.Element) -> str:
        tag = elem.tag
        if '}' in tag:
            tag = tag.split('}')[1]
        return tag.lower()
    
    def is_extract_element(elem: ET.Element) -> bool:
        return get_tag_name(elem) in extract_elements_set
    
    def is_block_element(elem: ET.Element) -> bool:
        return get_tag_name(elem) in block_elements_set
    
    def is_self_closing_element(elem: ET.Element) -> bool:
        """Check if an element is self-closing (like <lb/>, <pb/>, etc.).
        These elements should never have content - if they do, it's malformed XML."""
        tag_name = get_tag_name(elem)
        self_closing_tags = {'lb', 'pb', 'milestone', 'anchor', 'gap', 'fw'}
        return tag_name in self_closing_tags
    
    # Extract content from extract elements (like notes)
    note_content_map: Dict[str, Tuple[str, Dict[str, str], List[ET.Element]]] = {}
    note_id_counter = 1
    
    def extract_note_content(elem: ET.Element) -> Tuple[str, Dict[str, str], List[ET.Element]]:
        text = elem.text or ""
        attrib = dict(elem.attrib)
        children = list(elem)
        return (text, attrib, children)
    
    def extract_all_extract_elements(elem: ET.Element, inside_extract: bool = False) -> None:
        nonlocal note_id_counter
        for child in list(elem):
            child_is_extract = is_extract_element(child)
            if child_is_extract and not inside_extract:
                elem_id = child.get("id") or child.get("{http://www.w3.org/XML/1998/namespace}id")
                if not elem_id:
                    tag_name = get_tag_name(child)
                    elem_id = f"{tag_name}-{note_id_counter}"
                    note_id_counter += 1
                    child.set("id", elem_id)
                note_content_map[elem_id] = extract_note_content(child)
                child.text = None
                for subchild in list(child):
                    child.remove(subchild)
                # Do not recurse into removed content (nested extract elements stay bundled)
                continue
            # Recurse into children (propagate flag if we're inside an extract)
            extract_all_extract_elements(child, inside_extract or child_is_extract)
    
    if not include_notes:
        extract_all_extract_elements(root)
    
    # Track which sentences/tokens have been used
    used_sentence_indices = set()
    used_token_count = 0
    global_tok_id = 1  # Global token ID counter across all block elements
    global_sent_idx = 0  # Global sentence index counter across all block elements
    
    # Build standoff representation for each text node
    for text_node in text_nodes:
        # Find block elements within text node
        block_elems = []
        for child in text_node:
            if is_block_element(child):
                block_elems.append(child)
        
        if not block_elems:
            # No block elements, process text_node directly
            block_elems = [text_node]
        
        for block_elem in block_elems:
            # Build standoff representation for this block element
            plaintext, markup = build_standoff_representation(
                block_elem, 
                block_elements_set, 
                extract_elements_set, 
                include_notes,
                is_self_closing_element  # Pass the function
            )
            
            # Print plain text and standoff representation
            
            # Create a subset document with only unused sentences/tokens
            # Try to match the plaintext with remaining sentences
            remaining_sentences = [sent for i, sent in enumerate(document.sentences) if i not in used_sentence_indices]
            
            # Try to find matching sentences for this block
            # Match by comparing normalized text
            import re
            plaintext_norm = re.sub(r'\s+', ' ', plaintext).strip()
            matched_sentences = []
            matched_sentence_indices = []
            
            accumulated_text = ""
            for i, sent in enumerate(remaining_sentences):
                sent_idx = len(used_sentence_indices) + i
                sent_text_norm = re.sub(r'\s+', ' ', sent.text or "").strip()
                
                # Try adding this sentence to the accumulated text
                if accumulated_text:
                    # Add sentence - allow flexible whitespace between sentences
                    # Sentences might be separated by space, period+space, etc.
                    accumulated_text_norm = re.sub(r'\s+', ' ', accumulated_text + " " + sent_text_norm).strip()
                else:
                    accumulated_text_norm = sent_text_norm
                
                # Check if the plaintext starts with the accumulated text (allowing for flexible whitespace)
                # Normalize both for comparison
                accumulated_norm = re.sub(r'\s+', ' ', accumulated_text_norm).strip()
                plaintext_norm_compare = re.sub(r'\s+', ' ', plaintext_norm).strip()
                
                if plaintext_norm_compare.startswith(accumulated_norm):
                    matched_sentences.append(sent)
                    matched_sentence_indices.append(sent_idx)
                    accumulated_text = accumulated_text_norm
                    
                    # If we've matched the entire plaintext exactly (allowing for whitespace differences), stop
                    if accumulated_norm == plaintext_norm_compare:
                        break
                    # Continue to try matching more sentences
                elif accumulated_norm.startswith(plaintext_norm_compare):
                    # The accumulated text is longer than plaintext - we've gone too far
                    # Use the sentences we've matched so far (if any)
                    if matched_sentences:
                        break
                    # If no sentences matched yet, try with just this sentence if it matches
                    sent_norm_compare = re.sub(r'\s+', ' ', sent_text_norm).strip()
                    if sent_norm_compare == plaintext_norm_compare or plaintext_norm_compare.startswith(sent_norm_compare):
                        matched_sentences = [sent]
                        matched_sentence_indices = [sent_idx]
                        break
                    # Otherwise, stop trying
                    break
                else:
                    # Can't match this sentence - if we have matched sentences, use those
                    if matched_sentences:
                        break
                    # If no sentences matched yet, try with just this sentence if it matches
                    sent_norm_compare = re.sub(r'\s+', ' ', sent_text_norm).strip()
                    if sent_norm_compare == plaintext_norm_compare or plaintext_norm_compare.startswith(sent_norm_compare):
                        matched_sentences = [sent]
                        matched_sentence_indices = [sent_idx]
                        break
                    # Otherwise, stop trying - can't match anything
                    break
            
            if not matched_sentences:
                # No match found, skip this block
                print(f"\nWARNING: Could not match plaintext for block element, skipping...")
                continue
            
            # Check if we've matched all the plaintext
            # Normalize both for comparison
            matched_text_norm = re.sub(r'\s+', ' ', accumulated_text).strip() if accumulated_text else ""
            plaintext_norm_check = re.sub(r'\s+', ' ', plaintext_norm).strip()
            
            # If we haven't matched all the plaintext, try to match more sentences
            if matched_text_norm != plaintext_norm_check and len(matched_sentences) < len(remaining_sentences):
                # We have more sentences available - try to match them
                remaining_plaintext = plaintext_norm_check[len(matched_text_norm):].strip() if len(matched_text_norm) < len(plaintext_norm_check) else ""
                if remaining_plaintext:
                    # Try to match remaining sentences to remaining plaintext
                    for i in range(len(matched_sentences), len(remaining_sentences)):
                        sent = remaining_sentences[i]
                        sent_idx = len(used_sentence_indices) + i
                        sent_text_norm = re.sub(r'\s+', ' ', sent.text or "").strip()
                        
                        # Check if remaining plaintext starts with this sentence
                        if remaining_plaintext.startswith(sent_text_norm) or sent_text_norm.startswith(remaining_plaintext):
                            matched_sentences.append(sent)
                            matched_sentence_indices.append(sent_idx)
                            # Update accumulated text
                            if accumulated_text:
                                accumulated_text = re.sub(r'\s+', ' ', accumulated_text + " " + sent_text_norm).strip()
                            else:
                                accumulated_text = sent_text_norm
                            # Update remaining plaintext
                            if remaining_plaintext.startswith(sent_text_norm):
                                remaining_plaintext = remaining_plaintext[len(sent_text_norm):].strip()
                            else:
                                remaining_plaintext = ""
                            
                            # If we've matched all remaining text, stop
                            if not remaining_plaintext:
                                break
            
            # Create a temporary document with matched sentences
            temp_document = Document(id=document.id)
            temp_document.sentences = matched_sentences
            
            # Map tokens and sentences to character positions using the matched sentences
            token_positions = map_tokens_to_positions(plaintext, temp_document)
            sentence_positions = map_sentences_to_positions(plaintext, temp_document, token_positions)
            
            adjusted_markup = _split_markup_for_sentence_boundaries(
                markup,
                sentence_positions,
                block_elements_set,
                extract_elements_set,
                plaintext,
            )
            
            # Rebuild XML with <s> and <tok> elements
            # Pass current counters and update them with the return value
            tokens_used, sentences_used = rebuild_xml_with_tokens(
                block_elem,
                plaintext,
                adjusted_markup,
                token_positions,
                sentence_positions,
                temp_document,  # Use temp_document instead of full document
                block_elements_set,
                extract_elements_set,
                note_content_map,
                start_tok_id=global_tok_id,
                start_sent_idx=global_sent_idx,
                settings=settings
            )
            # Update global counters for next block element
            global_tok_id += tokens_used
            global_sent_idx += sentences_used
            
            # Mark these sentences as used
            for sent_idx in matched_sentence_indices:
                used_sentence_indices.add(sent_idx)
    
    # Restore note content
    def restore_note_content(root: ET.Element) -> None:
        for elem in root.iter():
            if is_extract_element(elem):
                elem_id = elem.get("id") or elem.get("{http://www.w3.org/XML/1998/namespace}id")
                if elem_id and elem_id in note_content_map:
                    text, attrib, children = note_content_map[elem_id]
                    elem.text = text
                    for key, value in attrib.items():
                        if key != "id":
                            elem.set(key, value)
                    for child in children:
                        elem.append(child)
    
    restore_note_content(root)
    
    # Verify structural integrity (ignoring inserted tags/attributes)
    # If verification fails, raise an exception to prevent writing incorrect XML
    verify_structure_preserved(
        original_root_snapshot,
        root,
        ignore_tags={"s", "tok", "dtok"},  # Also ignore dtok tags (sub-tokens of MWTs)
        ignore_attrs={"id", "rpt"}
    )
    
    # Add change element to TEI header
    from .teitok import _add_change_to_tei_header
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
    
    # Write updated XML (without pretty-printing)
    if HAS_LXML:
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True, pretty_print=False)
    else:
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True)


def verify_structure_preserved(
    original_root: ET.Element,
    modified_root: ET.Element,
    ignore_tags: Set[str],
    ignore_attrs: Set[str],
) -> None:
    """
    Compare original vs modified XML by converting both to strings and stripping:
    - Any <s>/<tok> tags (or other ignore_tags)
    - Attributes listed in ignore_attrs (e.g. id, xml:id, rpt)
    - Bridge sequences like </ab><ab rpt="1"> that arise from reopening tags
    """

    def serialize(elem: ET.Element) -> str:
        return ET.tostring(elem, encoding="unicode")

    def remove_tags(text: str, tags_to_remove: Set[str]) -> str:
        if not tags_to_remove:
            return text
        tags_pattern = "|".join(re.escape(tag) for tag in tags_to_remove)
        # Remove opening tags (including attributes) and self-closing forms
        text = re.sub(rf"<(?:{tags_pattern})(\s[^<>]*?)?/?>", "", text, flags=re.IGNORECASE)
        # Remove closing tags
        text = re.sub(rf"</(?:{tags_pattern})>", "", text, flags=re.IGNORECASE)
        return text

    def remove_attributes(text: str, attrs_to_remove: Set[str]) -> str:
        for attr in attrs_to_remove:
            attr_pattern = r'\s+(?:\w+:)?' + re.escape(attr) + r'="[^"]*"'
            text = re.sub(attr_pattern, "", text, flags=re.IGNORECASE)
        return text

    def remove_repetition_bridges(text: str) -> str:
        pattern_whitespace = re.compile(
            r"<([A-Za-z0-9:_-]+)[^<>]*?\brpt=\"1\"[^<>]*?>(\s*)</\1>",
            flags=re.IGNORECASE,
        )
        pattern_bridge = re.compile(
            r"</([A-Za-z0-9:_-]+)>\s*<\1[^<>]*?\brpt=\"1\"[^<>]*?>",
            flags=re.IGNORECASE,
        )
        prev = None
        while prev != text:
            prev = text
            text = pattern_whitespace.sub(r"\2", text)
            text = pattern_bridge.sub("", text)
        return text

    original_text = serialize(original_root)
    modified_text = serialize(modified_root)

    # Apply same normalization steps to both strings
    original_text = remove_tags(original_text, ignore_tags)
    modified_text = remove_tags(modified_text, ignore_tags)

    original_text = remove_repetition_bridges(original_text)
    modified_text = remove_repetition_bridges(modified_text)
    
    original_text = remove_attributes(original_text, ignore_attrs)
    modified_text = remove_attributes(modified_text, ignore_attrs)

    if original_text != modified_text:
        diff_pos = 0
        max_len = min(len(original_text), len(modified_text))
        while diff_pos < max_len and original_text[diff_pos] == modified_text[diff_pos]:
            diff_pos += 1
        snippet_original = original_text[diff_pos:diff_pos + 120]
        snippet_modified = modified_text[diff_pos:diff_pos + 120]
        raise ValueError(
            "Modified XML differs from original once token/sentence tags and ignorable attrs are stripped.\n"
            f"First difference at position {diff_pos}:\n"
            f"Original: {snippet_original!r}\n"
            f"Modified: {snippet_modified!r}"
        )
def build_standoff_representation(
    elem: ET.Element,
    block_elements: Set[str],
    extract_elements: Set[str],
    include_notes: bool,
    is_self_closing_element_func
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build plain text and standoff representation from XML element.
    
    Returns:
        plaintext: str - extracted plain text
        markup: List[Dict] - standoff representation with:
            - name: element name
            - contents: attributes and other tag contents
            - start: character position where element starts
            - end: character position where element ends
            - level: nesting depth
            - order: order in XML tree
            - text: optional text at start of element (if not extracted)
            - tail: optional text at end of element (if not extracted)
    """
    plaintext_parts = []
    markup = []
    char_pos = 0
    order = 0
    level = 0
    open_elements = []  # Stack of open elements
    
    def get_tag_name(e: ET.Element) -> str:
        tag = e.tag
        if '}' in tag:
            tag = tag.split('}')[1]
        return tag.lower()
    
    def is_extract_element(e: ET.Element) -> bool:
        return get_tag_name(e) in extract_elements
    
    def is_block_element(e: ET.Element) -> bool:
        return get_tag_name(e) in block_elements
    
    def process_element(e: ET.Element) -> None:
        nonlocal char_pos, order, level
        
        tag_name = get_tag_name(e)
        order += 1
        
        # Build element description
        elem_desc = {
            'name': tag_name,
            'contents': build_element_contents(e),
            'start': char_pos,
            'level': level,
            'order': order,
            'element': e  # Store reference to original element
        }
        
        # Handle extract elements (like notes) - skip their content
        if not include_notes and is_extract_element(e):
            # Extract element: mark position but don't extract content
            # Store the element's text in the markup (not in plaintext)
            def collect_extract_text(elem):
                text_parts = []
                if elem.text:
                    text_parts.append(elem.text)
                for child in elem:
                    text_parts.append(collect_extract_text(child))
                    if child.tail:
                        text_parts.append(child.tail)
                return ''.join(text_parts)
            elem_desc['text'] = collect_extract_text(e)
            elem_desc['end'] = char_pos
            markup.append(elem_desc)
            # Process tail (text after the element) - this SHOULD be extracted
            # The tail is processed by the parent, so we don't process it here
            # Just return - the parent will handle the tail
            return
        
        # Handle self-closing elements (like <lb/>, <pb/>)
        # These elements should never have content - if they do, treat it as tail text
        is_self_closing = is_self_closing_element_func(e)
        if is_self_closing:
            # Self-closing element: open and close at same position
            elem_desc['end'] = char_pos
            # If the element has text content, it's malformed XML - treat it as tail
            if e.text:
                # Store the text as tail (it should be after the element, not inside)
                elem_desc['tail'] = e.text
                # Move the text to tail so parent will process it
                # The text should come before the existing tail (if any)
                e.tail = e.text + (e.tail or "")
                e.text = None
            markup.append(elem_desc)
            # Don't process tail here - let the parent handle it in the child processing loop
            return
        
        # Regular element: extract text and process children
        open_elements.append(elem_desc)
        level += 1
        
        # Process element's text
        if e.text:
            plaintext_parts.append(e.text)
            char_pos += len(e.text)
        
        # Process children
        for child in e:
            process_element(child)
            if child.tail:
                plaintext_parts.append(child.tail)
                char_pos += len(child.tail)
        
        # Close element
        level -= 1
        elem_desc = open_elements.pop()
        elem_desc['end'] = char_pos
        markup.append(elem_desc)
    
    # Process the element
    process_element(elem)
    
    plaintext = ''.join(plaintext_parts)
    return plaintext, markup


def _split_markup_for_sentence_boundaries(
    markup: List[Dict[str, Any]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    block_elements: Set[str],
    extract_elements: Set[str],
    plaintext: str,
) -> List[Dict[str, Any]]:
    """
    Split markup spans so that sentence boundaries always align with start/end positions
    for splittable tags (prevents unnecessary fallbacks).
    """
    if not sentence_positions:
        return markup

    split_positions: Set[int] = set()
    text_len = len(plaintext)
    for _, end, _ in sentence_positions:
        adjusted_end = end
        while adjusted_end < text_len and plaintext[adjusted_end].isspace():
            adjusted_end += 1
        split_positions.add(adjusted_end)

    if not split_positions:
        return markup

    never_break = set(block_elements) | set(extract_elements)
    result: List[Dict[str, Any]] = []

    for entry in markup:
        tag_name = entry.get("name")
        start = entry.get("start")
        end = entry.get("end")

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start >= end
            or tag_name in never_break
        ):
            result.append(entry)
            continue

        splits = sorted(pos for pos in split_positions if start < pos < end)
        if not splits:
            result.append(entry)
            continue

        boundaries = [start] + splits + [end]
        nonempty_segments: List[Tuple[int, int]] = []
        current_start = boundaries[0]
        for idx in range(1, len(boundaries)):
            seg_start = current_start
            seg_end = boundaries[idx]
            segment_text = plaintext[seg_start:seg_end]
            if segment_text.strip():
                nonempty_segments.append((seg_start, seg_end))
                current_start = seg_end
            else:
                current_start = seg_end
                if nonempty_segments:
                    last_start, _ = nonempty_segments[-1]
                    nonempty_segments[-1] = (last_start, seg_end)
        for idx, (seg_start, seg_end) in enumerate(nonempty_segments):
            segment_entry = dict(entry)
            segment_entry["start"] = seg_start
            segment_entry["end"] = seg_end
            if idx > 0:
                segment_entry["split_reopen"] = True
            else:
                segment_entry.pop("split_reopen", None)
            result.append(segment_entry)

    return result


def build_element_contents(elem: ET.Element) -> str:
    """Build the contents string for an element (attributes, etc.)."""
    parts = []
    for key, value in elem.attrib.items():
        # Handle namespace prefixes
        if '}' in key:
            key = key.split('}')[1]
        parts.append(f'{key}="{value}"')
    return ' ' + ' '.join(parts) if parts else ''


def map_tokens_to_positions(plaintext: str, document: Document) -> List[Tuple[int, int, Token]]:
    """
    Map tokens to character positions in plain text.
    
    For MWT tokens, only map the parent token (not subtokens) to avoid text duplication.
    
    Returns:
        List of (start_pos, end_pos, token) tuples
    """
    positions = []
    text_pos = 0
    
    # Track subtoken IDs to skip them (they're handled as part of the MWT parent)
    subtoken_ids = set()
    for sent in document.sentences:
        for token in sent.tokens:
            if token.is_mwt and token.subtokens:
                # Mark all subtokens (except the first one, which is the token itself) as subtokens to skip
                for st in token.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)
    
    for sent in document.sentences:
        for token in sent.tokens:
            # Skip subtokens of MWT tokens (they're handled as part of the MWT parent)
            if token.id and token.id in subtoken_ids:
                continue
            
            # Skip whitespace
            while text_pos < len(plaintext) and plaintext[text_pos].isspace():
                text_pos += 1
            
            if text_pos >= len(plaintext):
                break
            
            # For MWT tokens, use the combined form
            if token.is_mwt and token.subtokens:
                tok_form = token.form  # Combined form (e.g., "awesome-align")
            else:
                tok_form = token.form
            
            # Try to find token
            if plaintext[text_pos:].startswith(tok_form):
                start_pos = text_pos
                end_pos = text_pos + len(tok_form)
                positions.append((start_pos, end_pos, token))
                text_pos = end_pos
            else:
                # Try case-insensitive
                remaining = plaintext[text_pos:].lower()
                if remaining.startswith(tok_form.lower()):
                    actual_len = len(tok_form)
                    start_pos = text_pos
                    end_pos = text_pos + actual_len
                    positions.append((start_pos, end_pos, token))
                    text_pos = end_pos
                else:
                    # Token not found at expected position - try to find it anywhere in remaining text
                    # This handles cases where there's extra whitespace or the token appears later
                    remaining_text = plaintext[text_pos:]
                    # Try to find the token form in the remaining text (case-insensitive)
                    import re
                    pattern = re.escape(tok_form)
                    match = re.search(pattern, remaining_text, re.IGNORECASE)
                    if match:
                        start_pos = text_pos + match.start()
                        end_pos = text_pos + match.end()
                        positions.append((start_pos, end_pos, token))
                        text_pos = end_pos
                    else:
                        # Still can't find it - skip this token but warn
                        # This shouldn't happen often, but if it does, we continue
                        continue
    
    return positions


def map_sentences_to_positions(
    plaintext: str, 
    document: Document, 
    token_positions: List[Tuple[int, int, Token]]
) -> List[Tuple[int, int, Sentence]]:
    """
    Map sentences to character positions based on token positions.
    
    Ensures sentences always span at least from the first token's start to the last token's end.
    If sentence text is available, extends to include trailing whitespace/punctuation.
    Ensures sentences don't overlap - each sentence ends before the next one starts.
    
    Returns:
        List of (start_pos, end_pos, sentence) tuples
    """
    positions = []
    token_idx = 0
    
    for i, sent in enumerate(document.sentences):
        if token_idx >= len(token_positions):
            break
        
        first_token_start = token_positions[token_idx][0]
        
        # Find the last token that belongs to this sentence
        # We need to match tokens from token_positions to tokens in sent.tokens
        # MWT subtokens are skipped in token_positions, so we need to account for that
        last_token_idx = token_idx
        sent_token_idx = 0
        pos_idx = token_idx
        
        # Iterate through tokens in token_positions, matching them to sent.tokens
        while pos_idx < len(token_positions) and sent_token_idx < len(sent.tokens):
            token_from_pos = token_positions[pos_idx][2]  # Token object from positions
            sent_token = sent.tokens[sent_token_idx]
            
            # Check if this is the same token
            if token_from_pos == sent_token or (token_from_pos.id and sent_token.id and token_from_pos.id == sent_token.id):
                # Match found - this token belongs to this sentence
                last_token_idx = pos_idx
                pos_idx += 1
                sent_token_idx += 1
            else:
                # This sent_token is likely a subtoken (not in token_positions)
                # Skip it and continue
                sent_token_idx += 1
        
        # Calculate last token end position
        if last_token_idx >= len(token_positions):
            # Fallback: use first token position if we can't find last token
            last_token_end = first_token_start
        else:
            last_token_end = token_positions[last_token_idx][1]
        
        # Determine the maximum end position - don't go beyond next sentence's start
        # This is the STRICT upper bound - sentence must end before next sentence starts
        max_sentence_end = len(plaintext)
        if i + 1 < len(document.sentences):
            # Find where the next sentence starts
            # The next sentence starts at the token after the last token of this sentence
            next_token_idx = last_token_idx + 1
            if next_token_idx < len(token_positions):
                next_sentence_start = token_positions[next_token_idx][0]
                # CRITICAL: Ensure this sentence ends BEFORE the next one starts
                # This prevents overlap - use the next sentence's start as the hard limit
                max_sentence_end = next_sentence_start
            else:
                # No more tokens - also try to find next sentence by its text
                next_sent = document.sentences[i + 1]
                if next_sent.text:
                    next_sent_text = next_sent.text.strip()
                    # Search for next sentence's text in plaintext starting from last_token_end
                    import re
                    next_sent_pattern = re.escape(next_sent_text)
                    next_sent_pattern = next_sent_pattern.replace(r'\ ', r'\s+')
                    match = re.search(next_sent_pattern, plaintext[last_token_end:], re.IGNORECASE)
                    if match:
                        next_sentence_start = last_token_end + match.start()
                        max_sentence_end = next_sentence_start
        
        # Start with sentence end at the last token's end position
        sentence_end = last_token_end
        
        # If sentence text is available, try to extend to include trailing whitespace/punctuation
        # But STRICTLY enforce that we never exceed max_sentence_end (next sentence's start)
        if sent.text:
            sent_text_clean = sent.text.strip()
            if sent_text_clean:
                # Look for the sentence text in plaintext starting from first_token_start
                search_start = max(0, first_token_start)
                # Search only up to max_sentence_end - never beyond
                search_end = max_sentence_end
                if search_end > search_start:
                    search_text = plaintext[search_start:search_end]
                    
                    # Try to find the sentence text (allowing for whitespace normalization)
                    import re
                    # Escape special regex characters but allow flexible whitespace
                    sent_text_pattern = re.escape(sent_text_clean)
                    sent_text_pattern = sent_text_pattern.replace(r'\ ', r'\s+')
                    match = re.search(sent_text_pattern, search_text, re.IGNORECASE)
                    if match:
                        # Found the sentence text - use its end position
                        found_end = search_start + match.end()
                        # CRITICAL: Never exceed max_sentence_end (next sentence's start)
                        # If the match extends beyond, truncate it
                        sentence_end = min(found_end, max_sentence_end)
        
        # Ensure sentence_end is at least last_token_end (sentence must contain all its tokens)
        sentence_end = max(sentence_end, last_token_end)
        
        # FINAL CHECK: Ensure sentence doesn't extend beyond max_sentence_end (next sentence's start)
        # This is a hard limit - never exceed it
        sentence_end = min(sentence_end, max_sentence_end)
        
        # Ensure sentence doesn't extend beyond plaintext
        sentence_end = min(sentence_end, len(plaintext))
        
        # CRITICAL: Final check - ensure sentence doesn't extend beyond max_sentence_end
        # This is already set above based on next sentence's start, but double-check here
        if sentence_end > max_sentence_end:
            sentence_end = max_sentence_end
        
        # Ensure this sentence doesn't overlap with the previous sentence
        if positions:
            prev_start, prev_end, _ = positions[-1]
            if first_token_start < prev_end:
                # This sentence starts before the previous one ends - this shouldn't happen
                # but if it does, adjust the previous sentence's end to be before this one starts
                # Actually, we can't modify previous positions, so ensure this one starts after previous ends
                first_token_start = max(first_token_start, prev_end)
                sentence_end = max(sentence_end, first_token_start)
            # Also ensure this sentence ends before it would overlap with previous
            if sentence_end <= prev_end and first_token_start < prev_end:
                # This sentence is completely within previous - shouldn't happen, but adjust
                sentence_end = prev_end + 1
        
        # Final check: ensure sentence_end > first_token_start (sentence must have positive length)
        if sentence_end <= first_token_start:
            sentence_end = first_token_start + 1
        
        positions.append((first_token_start, sentence_end, sent))
        # Update token_idx to point to the first token of the next sentence
        # Use last_token_idx + 1 (the token after the last token of this sentence)
        token_idx = last_token_idx + 1
    
    return positions


def rebuild_xml_with_tokens(
    block_elem: ET.Element,
    plaintext: str,
    markup: List[Dict[str, Any]],
    token_positions: List[Tuple[int, int, Token]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    document: Document,
    block_elements: Set[str],
    extract_elements: Set[str],
    note_content_map: Dict[str, Tuple[str, Dict[str, str], List[ET.Element]]],
    start_tok_id: int = 1,
    start_sent_idx: int = 0,
    settings: Optional[Any] = None,
) -> Tuple[int, int]:
    """
    Rebuild XML by inserting <s> and <tok> elements at correct positions.
    
    This is the core function that uses the standoff representation to correctly
    insert tokens and sentences while preserving all original XML structure.
    """
    # Helper functions for attribute mapping (similar to teitok.py)
    def _resolve_attr_name(internal_attr: str) -> Optional[str]:
        if settings:
            return settings.resolve_xml_attribute(internal_attr, default=internal_attr)
        return internal_attr
    
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
    
    def _set_attr(node: ET.Element, internal_attr: str, value: str, default_empty: bool = False) -> None:
        """Set attribute respecting TEITOK mappings."""
        target_attr = _resolve_attr_name(internal_attr)
        aliases = _attribute_aliases(internal_attr)
        
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
    
    # Save original attributes
    original_attrs = dict(block_elem.attrib)
    original_tail = block_elem.tail
    block_elem.clear()
    block_elem.attrib.update(original_attrs)
    
    # Build maps for quick lookup
    # Map character position to list of elements that open/close there
    opens_at = {}  # pos -> list of (type, data) where type is 'markup', 'sentence', or 'token'
    closes_at = {}  # pos -> list of (type, data)
    # Map character position to the element whose tail it belongs to
    char_to_tail_elem = {}  # char_pos -> (element, start_pos, end_pos)
    
    for m in markup:
        start = m['start']
        end = m['end']
        if start not in opens_at:
            opens_at[start] = []
        opens_at[start].append(('markup', m))
        if end not in closes_at:
            closes_at[end] = []
        closes_at[end].append(('markup', m))
        
        # Track tail text for self-closing elements
        # For self-closing elements, the tail comes AFTER the element
        # The element's end position is where it "closes" (same as start for self-closing)
        # The tail starts at end and continues for len(tail) characters
        if 'tail' in m and m['tail']:
            tail_start = end
            tail_end = end + len(m['tail'])
            for tail_pos in range(tail_start, tail_end):
                char_to_tail_elem[tail_pos] = (m, tail_start, tail_end)
    
    # Add sentences and tokens to opens_at/closes_at
    for sent_start, sent_end, sent in sentence_positions:
        if sent_start not in opens_at:
            opens_at[sent_start] = []
        opens_at[sent_start].append(('sentence', (sent_start, sent_end, sent)))
        if sent_end not in closes_at:
            closes_at[sent_end] = []
        closes_at[sent_end].append(('sentence', (sent_start, sent_end, sent)))
    
    for tok_start, tok_end, token in token_positions:
        if tok_start not in opens_at:
            opens_at[tok_start] = []
        opens_at[tok_start].append(('token', (tok_start, tok_end, token)))
        if tok_end not in closes_at:
            closes_at[tok_end] = []
        closes_at[tok_end].append(('token', (tok_start, tok_end, token)))
    
    def compute_sentence_fallbacks(sentence_positions, markup, text):
        fallback = set()
        for idx, (s_start, s_end, _) in enumerate(sentence_positions):
            for m in markup:
                if m['element'] == block_elem or m['start'] == m['end']:
                    continue
                m_start = m['start']
                m_end = m['end']
                overlap_before = m_start < s_start < m_end and text[m_start:s_start].strip()
                overlap_after = m_start < s_end < m_end and text[s_end:m_end].strip()
                if overlap_before or overlap_after:
                    fallback.add(idx)
                    break
        return fallback
    
    sentence_fallbacks = compute_sentence_fallbacks(sentence_positions, markup, plaintext)
    
    # Note: Implicit MWT detection is handled in __main__.py via _create_implicit_mwt
    # before calling insert_tokens_into_teitok, so we don't need to do it here.
    # This avoids conflicts and ensures MWTs are created consistently.
    
    # Track token IDs that are subtokens of MWT tokens (these should be skipped during insertion)
    mwt_subtoken_ids = set()
    for sent_start, sent_end, sent in sentence_positions:
        sent_tokens = [(t_start, t_end, tok) for t_start, t_end, tok in token_positions 
                      if sent_start <= t_start < sent_end]
        sent_tokens.sort(key=lambda x: x[0])
        for t_start, t_end, tok in sent_tokens:
            if tok.is_mwt and tok.subtokens:
                # Mark all subtokens (except the first one, which is tok itself) as MWT subtokens to skip
                for st in tok.subtokens[1:]:
                    # Track by token ID instead of Token object (Token objects are not hashable)
                    if st.id:
                        mwt_subtoken_ids.add(st.id)
    
    # Track current state
    current_elem = block_elem
    open_markup_stack = []  # Stack of (markup_entry, element) tuples
    token_idx = 0
    sent_idx = start_sent_idx  # Start from the provided index
    global_tok_id = start_tok_id  # Start from the provided token ID
    current_tok_elem = None
    current_sent_elem = None
    current_sentence_idx = None
    current_sentence_is_fallback = False
    current_sentence_first_token = None
    current_sentence_obj = None
    sent_token_ids = []
    # Map from token ord (id) to tokid for head conversion
    ord_to_tokid: Dict[int, str] = {}
    # Track self-closing elements that were just inserted (for tail text handling)
    self_closing_elements = {}  # (start_pos, end_pos) -> element
    # Track token state when markup elements open while a token is active
    markup_token_state: Dict[ET.Element, Dict[str, Any]] = {}
    # Build parent map for xml.etree.ElementTree compatibility
    parent_map = {c: p for p in block_elem.iter() for c in p}
    
    def resolve_parent_after_token(parent):
        candidate = parent
        open_elements = {elem for _, elem in open_markup_stack}
        while candidate is not None:
            if current_sent_elem is not None and candidate == current_sent_elem:
                return candidate
            if candidate in open_elements:
                return candidate
            candidate = parent_map.get(candidate)
        return block_elem
    
    # Process each character position
    initial_tok_id = start_tok_id
    initial_sent_idx = start_sent_idx  # Use the provided start index
    for char_pos in range(len(plaintext) + 1):  # +1 to handle closing at end
        char = plaintext[char_pos] if char_pos < len(plaintext) else None
        
        # Close elements that end at this position (before opening new ones)
        # Close in reverse order: markup first (innermost), then tokens, then sentences (outermost)
        if char_pos in closes_at:
            # Sort by type: markup first (innermost), then tokens, then sentences (outermost)
            # Within each type, close those that end later first (to close innermost first)
            def get_close_key(item):
                item_type, item_data = item
                if item_type == 'markup':
                    level = item_data.get('level', 0)
                    return (0, -item_data['end'], -level)  # Markup first, deeper first
                elif item_type == 'token':
                    return (1, -item_data[1])  # Tokens second, later end first
                else:  # sentence
                    return (2, -item_data[1])  # Sentences last, later end first
            
            to_close = sorted(closes_at[char_pos], key=get_close_key)
            
            for close_type, close_data in to_close:
                if close_type == 'sentence':
                    sent_start, sent_end, sent = close_data
                    if char_pos == sent_end and current_sent_elem is not None:
                        # Close sentence
                        if not current_sent_elem.text and not list(current_sent_elem) and sent_token_ids:
                            # Empty sentence with tokens - use @corresp
                            current_sent_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                        parent_after_sentence = parent_map.get(current_sent_elem, block_elem)
                        if open_markup_stack:
                            current_elem = open_markup_stack[-1][1]
                        else:
                            current_elem = parent_after_sentence
                        current_sent_elem = None
                        sent_token_ids = []
                        sent_idx += 1
                elif close_type == 'token':
                    tok_start, tok_end, token = close_data
                    if char_pos == tok_end and current_tok_elem is not None:
                        # Close token
                        # Clear any markup state tied to this token
                        for elem in list(markup_token_state.keys()):
                            if markup_token_state[elem].get("token") == current_tok_elem:
                                markup_token_state.pop(elem, None)
                        parent_after_token = parent_map.get(current_tok_elem, block_elem)
                        current_elem = resolve_parent_after_token(parent_after_token)
                        current_tok_elem = None
                        token_idx += 1
                elif close_type == 'markup':
                    m = close_data
                    # Find this markup in the open stack and close it
                    # Compare by start/end positions and tag name
                    for i, (open_m, open_elem) in enumerate(open_markup_stack):
                        # Match by start/end positions and tag name
                        if (open_m['start'] == m['start'] and 
                            open_m['end'] == m['end'] and
                            open_m['name'] == m['name']):
                            # Close this element
                            # If we're inside a token, we need to handle crossing XML
                            # The element should close, and the token will continue outside
                            if current_tok_elem is not None:
                                # Check if token is a descendant of the element to close
                                check_elem = current_tok_elem
                                token_inside_element = False
                                while check_elem is not None:
                                    if check_elem == open_elem:
                                        token_inside_element = True
                                        break
                                    check_elem = parent_map.get(check_elem)
                                
                                if token_inside_element:
                                    # Token is inside the element and spans across its boundary
                                    # The element ends, but the token continues outside it
                                    tok_start, tok_end, token = token_positions[token_idx] if token_idx < len(token_positions) else (None, None, None)
                                    parent_of_element = parent_map.get(open_elem, block_elem)
                                    
                                    if tok_end is not None and tok_end > char_pos:
                                        # Move token to be a sibling of the element (outside)
                                        token_parent = parent_map.get(current_tok_elem)
                                        if token_parent == open_elem:
                                            open_elem.remove(current_tok_elem)
                                            parent_children = list(parent_of_element)
                                            elem_index = parent_children.index(open_elem) if open_elem in parent_children else len(parent_children) - 1
                                            parent_of_element.insert(elem_index + 1, current_tok_elem)
                                            parent_map[current_tok_elem] = parent_of_element
                                        
                                        # Determine portion of the token captured while this markup was open
                                        state = markup_token_state.get(open_elem)
                                        text_len_before = state["text_len"] if state and state.get("token") == current_tok_elem else len(current_tok_elem.text or "")
                                        children_count_before = state["children_count"] if state and state.get("token") == current_tok_elem else len(current_tok_elem)
                                        
                                        token_text = current_tok_elem.text or ""
                                        moved_text = token_text[text_len_before:]
                                        remaining_text = token_text[:text_len_before]
                                        current_tok_elem.text = remaining_text if remaining_text else None
                                        
                                        moved_children = list(current_tok_elem)[children_count_before:]
                                        for child in moved_children:
                                            current_tok_elem.remove(child)
                                        
                                        if moved_text or moved_children:
                                            reopened_elem = ET.Element(open_elem.tag, dict(open_elem.attrib))
                                            reopened_elem.set("rpt", "1")
                                            if moved_text:
                                                reopened_elem.text = moved_text
                                            for child in moved_children:
                                                reopened_elem.append(child)
                                                parent_map[child] = reopened_elem
                                            current_tok_elem.insert(children_count_before, reopened_elem)
                                            parent_map[reopened_elem] = current_tok_elem
                                        
                                        # Further characters should go to the token, not the reopened element
                                        current_elem = current_tok_elem
                                        
                                        markup_token_state.pop(open_elem, None)
                                        current_elem = parent_of_element
                                        open_markup_stack.pop(i)
                                    else:
                                        # Token doesn't continue - just close
                                        markup_token_state.pop(open_elem, None)
                                        current_elem = parent_of_element
                                        open_markup_stack.pop(i)
                                    break
                                else:
                                    # Token is not inside the element - safe to close
                                    markup_token_state.pop(open_elem, None)
                                    current_elem = parent_map.get(open_elem, block_elem)
                                    open_markup_stack.pop(i)
                                    break
                            else:
                                # Not inside a token - safe to close
                                markup_token_state.pop(open_elem, None)
                                current_elem = parent_map.get(open_elem, block_elem)
                                open_markup_stack.pop(i)
                                break
        
        # Open elements that start at this position
        if char_pos in opens_at:
            # Sort to ensure correct nesting:
            # 1. Longest elements first (sentences should be longer than tokens)
            # 2. Type as tie-breaker (sentences before tokens)
            # 3. Level/order for markup
            def get_open_key(item):
                item_type, item_data = item
                if item_type == 'sentence':
                    start_pos, end_pos = item_data[0], item_data[1]
                    length = end_pos - start_pos
                    return (-length, 0, -end_pos)  # Longest first, type 0 (sentence), then by end position
                elif item_type == 'token':
                    start_pos, end_pos = item_data[0], item_data[1]
                    length = end_pos - start_pos
                    return (-length, 1, -end_pos)  # Longest first, type 1 (token), then by end position
                else:  # markup
                    end_pos = item_data['end']
                    start_pos = item_data['start']
                    length = end_pos - start_pos
                    level = item_data.get('level', 0)
                    order = item_data.get('order', 0)
                    return (-length, 2, -end_pos, level, order)  # Longest first, type 2 (markup), then by end position, level, order
            
            to_open = sorted(opens_at[char_pos], key=get_open_key)
            
            for open_type, open_data in to_open:
                if open_type == 'sentence':
                    sent_start, sent_end, sent = open_data
                    if sent_start == char_pos:
                        # Ensure previous sentence is closed before opening a new one
                        if current_sent_elem is not None:
                            # Previous sentence should have been closed - this shouldn't happen
                            # but if it does, close it first
                            parent_after_sentence = parent_map.get(current_sent_elem, block_elem)
                            if open_markup_stack:
                                current_elem = open_markup_stack[-1][1]
                            else:
                                current_elem = parent_after_sentence
                            current_sent_elem = None
                            sent_token_ids = []
                            sent_idx += 1
                        
                        # Open sentence
                        current_sentence_idx = sent_idx
                        current_sentence_is_fallback = current_sentence_idx in sentence_fallbacks
                        current_sentence_first_token = None
                        current_sentence_obj = sent
                        sent_token_ids = []
                        
                        if current_sentence_is_fallback:
                            current_sent_elem = None
                            continue
                        
                        s_elem = ET.Element("s")
                        s_id = f"s-{sent_idx + 1}"  # sent_idx already starts at start_sent_idx
                        s_elem.set("id", s_id)
                        if sent.text:
                            s_elem.set("text", sent.text)
                        current_elem.append(s_elem)
                        parent_map[s_elem] = current_elem
                        current_elem = s_elem
                        current_sent_elem = s_elem
                elif open_type == 'token':
                    tok_start, tok_end, token = open_data
                    if tok_start == char_pos:
                        # Skip tokens that are subtokens of MWT tokens (they're handled as part of the MWT)
                        if token.id and token.id in mwt_subtoken_ids:
                            continue
                        # Check if this is an MWT token with subtokens
                        if token.is_mwt and token.subtokens:
                            # Create parent <tok> element with combined form
                            tok_elem = ET.Element("tok")
                            tok_id = f"w-{global_tok_id}"
                            global_tok_id += 1
                            tok_elem.set("id", tok_id)
                            # Set the combined form as text content
                            tok_elem.text = token.form
                            
                            # Store mapping from ord to tokid for head conversion (use first subtoken's id)
                            if token.subtokens and token.subtokens[0].id:
                                ord_to_tokid[token.subtokens[0].id] = tok_id
                            
                            # Create <dtok> elements for each subtoken
                            for dtok_idx, subtoken in enumerate(token.subtokens):
                                dtok_elem = ET.Element("dtok")
                                dtok_id = f"{tok_id}.{dtok_idx + 1}"
                                dtok_elem.set("id", dtok_id)
                                dtok_elem.set("form", subtoken.form)
                                
                                # Store mapping from subtoken ord to dtok id for head conversion
                                if subtoken.id:
                                    ord_to_tokid[subtoken.id] = dtok_id
                                
                                # Set all annotations on <dtok>
                                # Check both direct fields and attrs dict (backends may store in either)
                                dtok_lemma = subtoken.lemma or subtoken.attrs.get("lemma", "")
                                dtok_xpos = subtoken.xpos or subtoken.attrs.get("xpos", "")
                                dtok_upos = subtoken.upos or subtoken.attrs.get("upos", "")
                                dtok_feats = subtoken.feats or subtoken.attrs.get("feats", "")
                                
                                _set_attr(dtok_elem, "lemma", dtok_lemma, default_empty=True)
                                _set_attr(dtok_elem, "xpos", dtok_xpos)
                                _set_attr(dtok_elem, "upos", dtok_upos)
                                _set_attr(dtok_elem, "feats", dtok_feats)
                                
                                # Set other attributes
                                _set_attr(dtok_elem, "reg", subtoken.reg or "")
                                _set_attr(dtok_elem, "expan", subtoken.expan or "")
                                _set_attr(dtok_elem, "mod", subtoken.mod or "")
                                _set_attr(dtok_elem, "trslit", subtoken.trslit or "")
                                _set_attr(dtok_elem, "ltrslit", subtoken.ltrslit or "")
                                
                                # Set ord attribute
                                if subtoken.id:
                                    _set_attr(dtok_elem, "ord", str(subtoken.id))
                                
                                # Set head attribute (convert from ord to tokid)
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
                                    head_tokid = ord_to_tokid.get(dtok_head_int)
                                    if head_tokid:
                                        _set_attr(dtok_elem, "head", head_tokid)
                                    else:
                                        # Head not found yet, use ord for now
                                        _set_attr(dtok_elem, "head", str(dtok_head_int))
                                else:
                                    _set_attr(dtok_elem, "head", "", default_empty=True)
                                
                                # Set deprel attribute (check attrs dict)
                                dtok_deprel = subtoken.attrs.get("deprel", "")
                                _set_attr(dtok_elem, "deprel", dtok_deprel, default_empty=True)
                                
                                # Set deps attribute (check attrs dict)
                                dtok_deps = subtoken.attrs.get("deps", "")
                                if dtok_deps:
                                    _set_attr(dtok_elem, "deps", dtok_deps)
                                
                                # Set misc attribute (check attrs dict)
                                dtok_misc = subtoken.attrs.get("misc", "")
                                if dtok_misc:
                                    _set_attr(dtok_elem, "misc", dtok_misc)
                                
                                tok_elem.append(dtok_elem)
                            
                            current_elem.append(tok_elem)
                            # Update parent map
                            parent_map[tok_elem] = current_elem
                            # Don't change current_elem - MWT tokens don't contain other elements
                            current_tok_elem = tok_elem
                        else:
                            # Regular (non-MWT) token
                            tok_elem = ET.Element("tok")
                            tok_id = f"w-{global_tok_id}"
                            global_tok_id += 1
                            tok_elem.set("id", tok_id)  # id is always "id", not mapped
                            tok_elem.set("form", token.form)  # form is always "form", not mapped
                            
                            # Store mapping from ord to tokid for head conversion
                            if token.id:
                                ord_to_tokid[token.id] = tok_id
                            
                            _set_attr(tok_elem, "lemma", token.lemma or "", default_empty=True)
                            _set_attr(tok_elem, "xpos", token.xpos or "")
                            _set_attr(tok_elem, "upos", token.upos or "")
                            _set_attr(tok_elem, "feats", token.feats or "")
                            
                            # Set ord attribute
                            if token.id:
                                _set_attr(tok_elem, "ord", str(token.id))
                            
                            # Set head attribute (convert from ord to tokid)
                            head_ord = token.head
                            if head_ord and head_ord > 0:
                                head_tokid = ord_to_tokid.get(head_ord)
                                if head_tokid:
                                    _set_attr(tok_elem, "head", head_tokid)
                                else:
                                    # Head not found yet (might be later in document), use ord for now
                                    _set_attr(tok_elem, "head", str(head_ord))
                            else:
                                _set_attr(tok_elem, "head", "", default_empty=True)
                            
                            # Set deprel attribute
                            _set_attr(tok_elem, "deprel", token.deprel or "", default_empty=True)
                            
                            # Set misc attribute
                            if token.misc:
                                _set_attr(tok_elem, "misc", token.misc)
                            
                            current_elem.append(tok_elem)
                            # Update parent map
                            parent_map[tok_elem] = current_elem
                            current_elem = tok_elem
                            current_tok_elem = tok_elem
                        
                        if current_sentence_first_token is None:
                            current_sentence_first_token = tok_elem
                        
                        if current_sent_elem is not None:
                            sent_token_ids.append(tok_id)
                        elif current_sentence_is_fallback:
                            sent_token_ids.append(tok_id)
                        # Record starting state for any open markup elements
                        for _, open_elem in open_markup_stack:
                            markup_token_state[open_elem] = {
                                "token": tok_elem,
                                "text_len": len(tok_elem.text or ""),
                                "children_count": len(tok_elem),
                            }
                elif open_type == 'markup':
                    m = open_data
                    # Skip if it's the block element itself
                    if m['element'] == block_elem:
                        continue
                    
                    # Check if it's an extract element (like note)
                    tag_name = m['name']
                    self_closing_tags = {'lb', 'pb', 'milestone', 'anchor', 'gap', 'fw'}
                    is_self_closing = tag_name in self_closing_tags
                    
                    if tag_name in extract_elements:
                        # Insert extract element (empty, will be restored later)
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        current_elem.append(new_elem)
                        # Update parent map
                        parent_map[new_elem] = current_elem
                        # Don't change current_elem for extract elements
                    elif is_self_closing:
                        # Self-closing element: insert as empty, handle tail separately
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        current_elem.append(new_elem)
                        # Update parent map
                        parent_map[new_elem] = current_elem
                        # Track this element for tail text handling
                        # Use the actual element we just created, not the original
                        self_closing_elements[(m['start'], m['end'])] = new_elem
                        # Don't add to stack - self-closing elements don't need to be closed
                        # Don't change current_elem - we want to continue adding to the parent
                    else:
                        # Regular element
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        if m.get("split_reopen"):
                            new_elem.set("rpt", "1")
                        current_elem.append(new_elem)
                        # Update parent map
                        parent_map[new_elem] = current_elem
                        current_elem = new_elem
                        open_markup_stack.append((m, new_elem))
                        if current_tok_elem is not None:
                            markup_token_state[new_elem] = {
                                "token": current_tok_elem,
                                "text_len": len(current_tok_elem.text or ""),
                                "children_count": len(current_tok_elem),
                            }
        
        # Add character to current element (if we're inside a token)
        if char is not None:
            # Check if this character belongs to a self-closing element's tail
            # This is character-by-character routing based on position
            # First, check if we're inside a token and if char_pos is in any tail range
            if token_idx < len(token_positions):
                tok_start, tok_end, token = token_positions[token_idx]
                if tok_start <= char_pos < tok_end and current_tok_elem is not None:
                    # Check if char_pos is in the tail range of any self-closing element
                    # that's a child of current_tok_elem
                    tail_elem_info = char_to_tail_elem.get(char_pos)
                    if tail_elem_info:
                        m, tail_start, tail_end = tail_elem_info
                        # Check if char_pos is in the tail range
                        if tail_start <= char_pos < tail_end:
                            # Find the corresponding element
                            tail_elem = None
                            # First try to find it in self_closing_elements
                            for (start, end), elem in self_closing_elements.items():
                                if start == m['start'] and end == m['end'] and elem.tag == m['name']:
                                    # Verify it's inside current_tok_elem
                                    if parent_map.get(elem) == current_tok_elem:
                                        tail_elem = elem
                                        break
                            # If not found, look in current_tok_elem's children
                            if tail_elem is None:
                                for child in current_tok_elem:
                                    if child.tag == m['name']:
                                        # Check attributes match (if any)
                                        if not m.get('contents') or all(child.get(k) == v for k, v in m.get('contents', {}).items()):
                                            tail_elem = child
                                            # Store it in self_closing_elements for future lookups
                                            self_closing_elements[(m['start'], m['end'])] = child
                                            break
                            
                            # If we found the element, route the character to its tail
                            if tail_elem is not None:
                                if tail_elem.tail:
                                    tail_elem.tail += char
                                else:
                                    tail_elem.tail = char
                                # Skip the rest - we've handled this character
                                continue
                    
                    # If no explicit tail range, check if there's a self-closing element
                    # that was inserted at a position before or at char_pos and is a child of the token
                    # For self-closing elements without explicit tail, all subsequent characters
                    # (within the token) should go to the element's tail
                    # Find the most recently inserted self-closing element (highest start position <= char_pos)
                    # that's a child of the token
                    tail_elem = None
                    best_start = -1
                    for (start, end), elem in self_closing_elements.items():
                        # Check if this element is a child of current_tok_elem
                        if parent_map.get(elem) == current_tok_elem:
                            # Check if the element was inserted at or before char_pos
                            if start <= char_pos and start > best_start:
                                tail_elem = elem
                                best_start = start
                    
                    # If we found a self-closing element, route the character to its tail
                    if tail_elem is not None:
                        if tail_elem.tail:
                            tail_elem.tail += char
                        else:
                            tail_elem.tail = char
                        # Skip the rest - we've handled this character
                        continue
            
            if token_idx < len(token_positions):
                tok_start, tok_end, token = token_positions[token_idx]
                if tok_start <= char_pos < tok_end:
                    # Inside token - add to token element
                    if current_tok_elem is not None:
                        check_elem = current_elem
                        is_inside_tok = False
                        while check_elem is not None:
                            if check_elem == current_tok_elem:
                                is_inside_tok = True
                                break
                            check_elem = parent_map.get(check_elem)
                        
                        if not is_inside_tok:
                            current_elem = current_tok_elem
                            is_inside_tok = True
                        
                        if current_elem == current_tok_elem:
                            # Check if this is an MWT token (has <dtok> children)
                            # MWT tokens already have their text set, so don't add characters
                            has_dtoks = any(c.tag.endswith("}dtok") or c.tag == "dtok" for c in current_tok_elem)
                            if not has_dtoks:
                                # Not an MWT token - append text as usual
                                # Append after any existing children to preserve order
                                if len(current_tok_elem):
                                    last_child = current_tok_elem[-1]
                                    if last_child.tail:
                                        last_child.tail += char
                                    else:
                                        last_child.tail = char
                                else:
                                    if current_tok_elem.text:
                                        current_tok_elem.text += char
                                    else:
                                        current_tok_elem.text = char
                            # If it's an MWT token, skip adding text (already set when token was created)
                        else:
                            if not current_elem.text:
                                current_elem.text = char
                            else:
                                current_elem.text += char
                    else:
                        # Fallback
                        if tail_elem is not None:
                            if tail_elem.tail:
                                tail_elem.tail += char
                            else:
                                tail_elem.tail = char
                        else:
                            if not current_elem.text:
                                current_elem.text = char
                            else:
                                current_elem.text += char
                elif char_pos == tok_end:
                    # Close token
                    current_elem = parent_map.get(current_tok_elem, block_elem)
                    current_tok_elem = None
                    token_idx += 1
                    
                    # Add character after token
                    if char_pos < len(plaintext):
                        if tail_elem is not None:
                            if tail_elem.tail:
                                tail_elem.tail += char
                            else:
                                tail_elem.tail = char
                        else:
                            if not current_elem.text and not list(current_elem):
                                current_elem.text = char
                            elif list(current_elem):
                                if current_elem[-1].tail:
                                    current_elem[-1].tail += char
                                else:
                                    current_elem[-1].tail = char
                            else:
                                if current_elem.text:
                                    current_elem.text += char
                                else:
                                    current_elem.text = char
                else:
                    # Before token - add to current element
                    if tail_elem is not None:
                        if tail_elem.tail:
                            tail_elem.tail += char
                        else:
                            tail_elem.tail = char
                    else:
                        # Normal placement
                        if not current_elem.text and not list(current_elem):
                            current_elem.text = char
                        elif list(current_elem):
                            if current_elem[-1].tail:
                                current_elem[-1].tail += char
                            else:
                                current_elem[-1].tail = char
                        else:
                            if current_elem.text:
                                current_elem.text += char
                            else:
                                current_elem.text = char
            else:
                # No more tokens - add to current element
                if tail_elem is not None:
                    if tail_elem.tail:
                        tail_elem.tail += char
                    else:
                        tail_elem.tail = char
                else:
                    # Normal placement
                    if not current_elem.text and not list(current_elem):
                        current_elem.text = char
                    elif list(current_elem):
                        if current_elem[-1].tail:
                            current_elem[-1].tail += char
                        else:
                            current_elem[-1].tail = char
                    else:
                        if current_elem.text:
                            current_elem.text += char
                        else:
                            current_elem.text = char
        
        # Close sentence if needed
        if sent_idx < len(sentence_positions):
            sent_start, sent_end, sent = sentence_positions[sent_idx]
            if char_pos == sent_end:
                # Check if sentence is empty
                if current_sent_elem is not None:
                    has_text = current_sent_elem.text and current_sent_elem.text.strip()
                    has_children = len(list(current_sent_elem)) > 0
                    if not has_text and not has_children and sent_token_ids:
                        current_sent_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                    
                    # Close sentence
                    current_elem = parent_map.get(current_sent_elem, block_elem)
                    current_sent_elem = None
                elif current_sentence_is_fallback and sent_token_ids:
                    s_elem = ET.Element("s")
                    s_id = f"s-{sent_idx + 1}"  # sent_idx already starts at start_sent_idx
                    
                    s_elem.set("id", s_id)
                    if current_sentence_obj and current_sentence_obj.text:
                        s_elem.set("text", current_sentence_obj.text)
                    s_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                    
                    target_parent = parent_map.get(current_sentence_first_token, block_elem)
                    if target_parent is not None:
                        children = list(target_parent)
                        try:
                            insert_idx = children.index(current_sentence_first_token)
                        except ValueError:
                            insert_idx = 0
                        target_parent.insert(insert_idx, s_elem)
                    else:
                        block_elem.append(s_elem)
                        target_parent = block_elem
                    parent_map[s_elem] = target_parent
                
                sent_token_ids = []
                current_sentence_idx = None
                current_sentence_is_fallback = False
                current_sentence_first_token = None
                current_sentence_obj = None
                sent_idx += 1
    
    # Restore original tail whitespace for this block element
    if original_tail is not None:
        block_elem.tail = original_tail
    
    # Second pass: Update all head values from ord to tokid
    # Now that all tokens are created, we can convert head ord values to tokids
    for tok_elem in block_elem.iter():
        if tok_elem.tag.endswith("}tok") or tok_elem.tag == "tok":
            # Get current head value
            head_attr = _resolve_attr_name("head")
            if head_attr:
                head_value = tok_elem.get(head_attr)
                if head_value and head_value.isdigit():
                    # It's an ord value, try to convert to tokid
                    head_ord = int(head_value)
                    head_tokid = ord_to_tokid.get(head_ord)
                    if head_tokid:
                        _set_attr(tok_elem, "head", head_tokid)
                    elif head_ord == 0:
                        # Root token, remove head attribute
                        _set_attr(tok_elem, "head", "", default_empty=True)
    
    # Return number of tokens and sentences used
    tokens_used = global_tok_id - initial_tok_id
    sentences_used = sent_idx - initial_sent_idx
    return (tokens_used, sentences_used)


def rebuild_xml_with_tokens_string_based(
    block_elem: ET.Element,
    plaintext: str,
    markup: List[Dict[str, Any]],
    token_positions: List[Tuple[int, int, Token]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    document: Document,
) -> None:
    """
    Placeholder for a fallback approach that rebuilds TEITOK blocks purely from
    the standoff representation, without mutating an ElementTree in place.
    
    Intended flow:
        1. Collect every start/end offset from `markup`, `token_positions`, and
           `sentence_positions`. Split the original XML string at those offsets
           so no remaining span intersects another (cf. `tmp/standoff2xml`).
        2. Convert the split structure into a linear stream of text fragments
           plus explicit open/close tag tokens. Because splitting happened up
           front, inserting new tags cannot create crossing markup.
        3. Walk that stream, emitting characters into a string builder and
           injecting `<s>` / `<tok>` tags when their offsets are reached. All
           whitespace from `plaintext` is copied verbatim.
        4. After finishing a block (or periodically), validate the accumulated
           string with `ET.fromstring` to ensure well‑formed XML before replacing
           the original block element.
        5. Once the new XML fragment is parsed, swap it back into the document
           and reapply the saved note/gap contents just like the current code.
    
    This function is not yet implemented; it serves as a skeleton so we can add
    a reliable fallback if the in-place builder ever fails the verification
    check.
    """
    raise NotImplementedError("string-based rebuild not implemented yet")

