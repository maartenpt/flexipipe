"""
Tagset definition parser for FlexiPipe.

Reads TEITOK-style tagset XML files and creates mappings from XPOS to UPOS and FEATS.
Supports position-based tagsets (EAGLES style).
"""
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict


# Mapping from EAGLES main categories to UD UPOS
EAGLES_TO_UPOS = {
    'A': 'ADJ',      # Adjective
    'R': 'ADV',      # Adverb
    'D': 'DET',      # Determiner
    'N': 'NOUN',     # Noun (common)
    'NP': 'PROPN',   # Proper Noun
    'V': 'VERB',     # Verb
    'P': 'PRON',     # Pronoun
    'C': 'CCONJ',   # Conjunction
    'I': 'INTJ',     # Interjection
    'S': 'ADP',      # Preposition
    'F': 'PUNCT',    # Punctuation
    'Z': 'NUM',      # Numeral
    'X': 'X',        # Other
}


def parse_teitok_tagset(tagset_file: Path) -> Dict:
    """
    Parse a TEITOK-style tagset XML file.
    
    Args:
        tagset_file: Path to tagset XML file
        
    Returns:
        Dictionary with:
        - 'main_categories': Main category mappings
        - 'position_defs': Position definitions for feature extraction
        - 'tagset_file': Path to tagset file
    """
    tree = ET.parse(tagset_file)
    root = tree.getroot()
    
    # Find tagset element
    tagset_elem = root.find('.//tagset') or root
    if tagset_elem is None:
        raise ValueError(f"Could not find <tagset> element in {tagset_file}")
    
    positions_elem = tagset_elem.find('positions')
    if positions_elem is None:
        raise ValueError(f"Could not find <positions> element in {tagset_file}")
    
    # Parse position-based tagset
    main_categories = {}
    position_defs = {}  # main_cat -> {pos_index -> {char_key: {ud_feat, ud_value, display}}}
    
    for item in positions_elem.findall('item'):
        key = item.get('key')
        maintag = item.get('maintag', '0')
        display = item.get('display', '')
        
        if key:
            main_categories[key] = {
                'maintag': maintag,
                'display': display,
                'upos': EAGLES_TO_UPOS.get(key, 'X')
            }
            
            # Initialize position definitions for this main category
            if key not in position_defs:
                position_defs[key] = {}
            
            # Handle multi-category items (e.g., NP for proper nouns)
            multi_elem = item.find('multi')
            if multi_elem is not None:
                for multi_item in multi_elem.findall('item'):
                    multi_key = multi_item.get('key')
                    if multi_key:
                        main_categories[multi_key] = {
                            'maintag': maintag,
                            'display': multi_item.get('display', ''),
                            'upos': EAGLES_TO_UPOS.get(multi_key, EAGLES_TO_UPOS.get(key, 'X'))
                        }
                        # Share position definitions with parent category
                        if multi_key not in position_defs:
                            position_defs[multi_key] = {}
            
            # Parse position definitions for this main category
            for pos_item in item.findall('item[@pos]'):
                pos_index = int(pos_item.get('pos'))
                if pos_index not in position_defs[key]:
                    position_defs[key][pos_index] = {}
                
                pos_display = pos_item.get('display', '').lower()
                
                # Map common feature names to UD FEATS
                ud_feat_name = _map_feature_name_to_ud(pos_display)
                
                # Parse value definitions - store by value key (the character in the XPOS tag)
                for value_item in pos_item.findall('item'):
                    value_key = value_item.get('key', '')
                    value_display = value_item.get('display', '').strip()
                    
                    # Skip items with key="0" and empty display (non-relevant features)
                    if value_key == '0' and not value_display:
                        continue
                    
                    value_display_lower = value_display.lower()
                    ud_value = _map_feature_value_to_ud(ud_feat_name, value_display_lower, value_key)
                    
                    if ud_value:  # Only store if we have a valid UD value
                        position_defs[key][pos_index][value_key] = {
                            'display': value_display,
                            'ud_value': ud_value,
                            'ud_feat': ud_feat_name
                        }
                    elif value_key:  # Store even if no UD value (for lookup purposes)
                        position_defs[key][pos_index][value_key] = {
                            'display': value_display,
                            'ud_value': None,
                            'ud_feat': ud_feat_name
                        }
            
            # Also copy position definitions to multi-category items
            if multi_elem is not None:
                for multi_item in multi_elem.findall('item'):
                    multi_key = multi_item.get('key')
                    if multi_key and key in position_defs:
                        position_defs[multi_key] = position_defs[key].copy()
    
    return {
        'main_categories': main_categories,
        'position_defs': position_defs,
        'tagset_file': str(tagset_file)
    }


def _map_feature_name_to_ud(feature_name: str) -> Optional[str]:
    """Map feature name to UD FEATS name."""
    feature_lower = feature_name.lower()
    
    # Common mappings (including abbreviations and multiple languages)
    mappings = {
        'gender': 'Gender',
        'género': 'Gender',
        'gen': 'Gender',  # Abbreviation (Czech, Portuguese)
        'number': 'Number',
        'número': 'Number',
        'num': 'Number',  # Abbreviation (Czech, Portuguese)
        'person': 'Person',
        'persona': 'Person',
        'mood': 'Mood',
        'modo': 'Mood',
        'tense': 'Tense',
        'tiempo': 'Tense',
        'case': 'Case',
        'caso': 'Case',
        'degree': 'Degree',
        'grado': 'Degree',
        'type': None,  # Type is usually not a UD feature
        'tipo': None,
        'form': 'Form',
        'forma': 'Form',
    }
    
    return mappings.get(feature_lower)


def _map_feature_value_to_ud(feat_name: Optional[str], value_display: str, value_key: str) -> Optional[str]:
    """Map feature value to UD FEATS value."""
    if feat_name is None:
        return None
    
    value_lower = value_display.lower()
    
    # Gender mappings
    if feat_name == 'Gender':
        if value_key == '0':
            return None
        mappings = {
            'masculine': 'Masc', 'masculino': 'Masc', 'm': 'Masc',
            'feminine': 'Fem', 'femenino': 'Fem', 'f': 'Fem',
            'common': 'Com', 'común': 'Com', 'comum': 'Com', 'c': 'Com',
            'neutral': 'Neut', 'neutro': 'Neut', 'n': 'Neut',
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    # Number mappings
    if feat_name == 'Number':
        if value_key == '0':
            return None
        mappings = {
            'singular': 'Sing', 'singular': 'Sing', 's': 'Sing',
            'plural': 'Plur', 'plural': 'Plur', 'p': 'Plur',
            'invariable': None, 'invariável': None, 'invariável': None, 'n': None,
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    # Person mappings
    if feat_name == 'Person':
        if value_key == '0':
            return None
        mappings = {
            'first': '1', 'primera': '1', 'primeira': '1', '1': '1',
            'second': '2', 'segunda': '2', '2': '2',
            'third': '3', 'tercera': '3', 'terceira': '3', '3': '3',
        }
        return mappings.get(value_lower) or (value_key if value_key and value_key != '0' else None)
    
    # Mood mappings
    if feat_name == 'Mood':
        if value_key == '0':
            return None
        mappings = {
            'indicative': 'Ind', 'indicativo': 'Ind', 'i': 'Ind',
            'subjunctive': 'Sub', 'subjuntivo': 'Sub', 'conjuntivo': 'Sub', 's': 'Sub',
            'imperative': 'Imp', 'imperativo': 'Imp', 'm': 'Imp',
            'infinitive': None, 'infinitivo': 'Inf', 'n': None,
            'gerund': 'Ger', 'gerundio': 'Ger', 'gerúndio': 'Ger', 'g': 'Ger',
            'participle': 'Part', 'participio': 'Part', 'particípio': 'Part', 'p': 'Part',
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    # Tense mappings
    if feat_name == 'Tense':
        if value_key == '0':
            return None
        mappings = {
            'present': 'Pres', 'presente': 'Pres', 'p': 'Pres',
            'imperfective': 'Imp', 'pretérito imperfecto': 'Imp', 'pretérito imperfeito': 'Imp', 'i': 'Imp',
            'future': 'Fut', 'futuro': 'Fut', 'f': 'Fut',
            'past': 'Past', 'pretérito perfecto': 'Past', 'pretérito perfeito': 'Past', 's': 'Past',
            'conditional': 'Cond', 'condicional': 'Cond', 'c': 'Cond',
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    # Case mappings
    if feat_name == 'Case':
        if value_key == '0':
            return None
        mappings = {
            'nominative': 'Nom', 'nominativo': 'Nom', 'n': 'Nom',
            'accusative': 'Acc', 'acusativo': 'Acc', 'a': 'Acc',
            'dative': 'Dat', 'dativo': 'Dat', 'd': 'Dat',
            'oblique': 'Obl', 'oblicuo': 'Obl', 'oblíquo': 'Obl', 'o': 'Obl',
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    # Degree mappings
    if feat_name == 'Degree':
        if value_key == '0':
            return None
        mappings = {
            'comparative': 'Cmp', 'comparativo': 'Cmp', 'c': 'Cmp',
            'superlative': 'Sup', 'superlativo': 'Sup', 'suerlativo': 'Sup', 's': 'Sup',
            'augmentative': None, 'aumentativo': None, 'a': None,
            'diminutive': None, 'diminutivo': None, 'd': None,
        }
        return mappings.get(value_lower) or (value_key.upper() if value_key and value_key != '0' else None)
    
    return None


def xpos_to_upos_feats(xpos: str, tagset_def: Dict) -> Tuple[str, str]:
    """
    Convert XPOS tag to UPOS and FEATS using tagset definition.
    
    Args:
        xpos: XPOS tag (e.g., "NCMS000" for Spanish)
        tagset_def: Tagset definition from parse_teitok_tagset()
        
    Returns:
        Tuple of (upos, feats) where feats is in UD format (e.g., "Gender=Masc|Number=Sing")
    """
    if not xpos or xpos == '_':
        return '_', '_'
    
    main_categories = tagset_def.get('main_categories', {})
    position_defs = tagset_def.get('position_defs', {})
    
    # Determine main category (first character or first two for multi-char categories)
    main_cat = None
    remaining = ''
    if len(xpos) >= 2 and xpos[:2] in main_categories:
        # Check if this multi-char category shares position definitions with a single-char parent
        multi_cat = xpos[:2]
        single_cat = xpos[0]
        multi_upos = main_categories.get(multi_cat, {}).get('upos')
        single_upos = main_categories.get(single_cat, {}).get('upos')
        
        if (single_cat in main_categories and 
            multi_cat in position_defs and 
            single_cat in position_defs and
            position_defs[multi_cat] == position_defs[single_cat] and
            multi_upos == single_upos):
            # Multi-category shares position defs AND UPOS with single - treat second char as position 1
            main_cat = single_cat
            remaining = xpos[1:]  # Include the second character in remaining
        else:
            # True multi-char category (different position defs or different UPOS)
            main_cat = multi_cat
            remaining = xpos[2:]
    elif len(xpos) >= 1:
        main_cat = xpos[0]
        remaining = xpos[1:]
    else:
        return 'X', '_'
    
    # Get UPOS from main category
    cat_info = main_categories.get(main_cat)
    if not cat_info:
        return 'X', '_'
    
    upos = cat_info.get('upos', 'X')
    
    # Extract features from remaining positions
    feats_dict = {}
    
    # Get position definitions for this main category
    cat_pos_defs = position_defs.get(main_cat, {})
    
    # Extract features from remaining positions
    for char_index, char in enumerate(remaining, start=1):
        if char_index in cat_pos_defs:
            pos_def = cat_pos_defs[char_index]
            if char in pos_def:
                char_info = pos_def[char]
                ud_feat = char_info.get('ud_feat')
                ud_value = char_info.get('ud_value')
                # Only add if it's a valid UD feature with a value
                if ud_feat and ud_value:
                    feats_dict[ud_feat] = ud_value
    
    # Convert feats_dict to UD format string
    if feats_dict:
        feats = '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items()))
    else:
        feats = '_'
    
    return upos, feats

