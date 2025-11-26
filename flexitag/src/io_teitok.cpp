#include "flexitag/io_teitok.h"

#include <pugixml.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace flexitag {

namespace {

bool is_whitespace(const char* value) {
    if (!value) {
        return false;
    }
    for (const char* ptr = value; *ptr != '\0'; ++ptr) {
        if (!std::isspace(static_cast<unsigned char>(*ptr))) {
            return false;
        }
    }
    return true;
}

// Helper to check if an attribute value should be written
// Returns true if the value should be written, false otherwise
// Special case: lemma and form can have "_" as a legitimate value (representing actual underscore)
bool should_write_attr(const std::string& value, const std::string& attr_name) {
    if (value.empty()) {
        return false;
    }
    // For lemma and form, "_" is a valid value (represents actual underscore)
    if (value == "_" && (attr_name == "lemma" || attr_name == "form")) {
        return true;
    }
    // For all other attributes, "_" means "not available" and should not be written
    if (value == "_") {
        return false;
    }
    return true;
}

bool has_space_after(const pugi::xml_node& node) {
    pugi::xml_node sibling = node.next_sibling();
    while (sibling && sibling.type() == pugi::node_comment) {
        sibling = sibling.next_sibling();
        }
    if (sibling && sibling.type() == pugi::node_pcdata) {
        return is_whitespace(sibling.value());
    }
    return false;
}

void build_teitok_document(const Document& doc, pugi::xml_document& out, 
                           const std::set<std::string>& custom_attributes) {
    out.reset();
    auto decl = out.append_child(pugi::node_declaration);
    decl.append_attribute("version") = "1.0";
    decl.append_attribute("encoding") = "UTF-8";

    auto tei = out.append_child("TEI");
    auto text = tei.append_child("text");

    // Step 1: Create all sentences as direct children of <text>
    // Store a map of sentence ID to sentence node for later reference
    std::map<std::string, pugi::xml_node> sentence_id_to_node;
    int sentence_index = 0;
    
    for (const auto& sentence : doc.sentences) {
        // Create sentence as direct child of text
        // Create sentence as direct child of text
        auto s_node = text.append_child("s");
        std::string sentence_id = sentence.id.empty() ? sentence.sent_id : sentence.id;
        if (sentence_id.empty()) {
            sentence_id = "s" + std::to_string(sentence_index + 1);
        }
        s_node.append_attribute("id") = sentence_id.c_str();
        sentence_id_to_node[sentence_id] = s_node;  // Store for later reference
        // Never write @sent_id - we only use @id
        if (!sentence.text.empty()) {
            s_node.append_attribute("text") = sentence.text.c_str();
        }
        if (!sentence.corr.empty()) {
            s_node.append_attribute("corr") = sentence.corr.c_str();
        }
        
        // Build entity tracking maps
        std::map<int, std::vector<const Entity*>> entities_by_start;  // token_id -> entities starting here
        std::map<int, std::vector<const Entity*>> entities_by_end;    // token_id -> entities ending here
        
        for (const auto& entity : sentence.entities) {
            entities_by_start[entity.start].push_back(&entity);
            entities_by_end[entity.end].push_back(&entity);
        }
        
        // Track currently open <name> elements (stack-based approach)
        std::vector<pugi::xml_node> open_name_nodes;
        
        int prev_token_id = 0;
        for (const auto& token : sentence.tokens) {
            int token_id = token.id;
            
            // Close entities that end at the previous token (using token.id, not position)
            if (prev_token_id > 0 && entities_by_end.find(prev_token_id) != entities_by_end.end()) {
                for (const auto* entity : entities_by_end[prev_token_id]) {
                    if (!open_name_nodes.empty()) {
                        open_name_nodes.pop_back();
                    }
                }
            }
            
            // Open entities that start at this token (using token.id, not position)
            pugi::xml_node current_parent = s_node;
            if (entities_by_start.find(token_id) != entities_by_start.end()) {
                for (const auto* entity : entities_by_start[token_id]) {
                    auto name_node = current_parent.append_child("name");
                    name_node.append_attribute("type") = entity->label.c_str();
                    if (!entity->text.empty()) {
                        name_node.append_attribute("text") = entity->text.c_str();
                    }
                    // Add additional attributes
                    for (const auto& [key, value] : entity->attrs) {
                        name_node.append_attribute(key.c_str()) = value.c_str();
                    }
                    open_name_nodes.push_back(name_node);
                    current_parent = name_node;
                }
            } else if (!open_name_nodes.empty()) {
                current_parent = open_name_nodes.back();
            }
            
            auto tok_node = current_parent.append_child("tok");
            // Prefer @id over @xml:id to avoid duplication - only write @id, never @xml:id
            if (!token.tokid.empty()) {
                // Use tokid for @id (not @xml:id)
                tok_node.append_attribute("id") = token.tokid.c_str();
            } else {
                // Generate @id from numeric token.id
            std::ostringstream id_stream;
            id_stream << "w-" << token.id;
            tok_node.append_attribute("id") = id_stream.str().c_str();
            }
            // Never write @xml:id - we only use @id
            tok_node.append_attribute("form") = token.form.c_str();
            
            // For MWTs, don't add lemma, xpos, or upos to the <tok> element
            // (only the dtok children have these attributes)
            bool is_mwt = token.is_mwt && !token.subtokens.empty();
            
            if (!is_mwt) {
                std::string head_str;
                // For non-MWT tokens, add lemma, xpos, upos if available
                if (should_write_attr(token.lemma, "lemma")) {
                    tok_node.append_attribute("lemma") = token.lemma.c_str();
                }
                if (should_write_attr(token.xpos, "xpos")) {
                    tok_node.append_attribute("xpos") = token.xpos.c_str();
                }
                if (should_write_attr(token.upos, "upos")) {
                    tok_node.append_attribute("upos") = token.upos.c_str();
                }
                if (token.head > 0) {
                    head_str = std::to_string(token.head);
                    tok_node.append_attribute("head") = head_str.c_str();
                }
                if (should_write_attr(token.deprel, "deprel")) {
                    tok_node.append_attribute("deprel") = token.deprel.c_str();
                }
                if (should_write_attr(token.deps, "deps")) {
                    tok_node.append_attribute("deps") = token.deps.c_str();
                }
                if (should_write_attr(token.misc, "misc")) {
                    // Filter out SpaceAfter=No from MISC field for the last token
                    // (last token should not have SpaceAfter=No in TEI output)
                    std::string misc_value = token.misc;
                    bool is_last_token = (&token == &sentence.tokens.back());
                    if (is_last_token) {
                        // Remove SpaceAfter=No from MISC field
                        std::string filtered_misc;
                        std::istringstream misc_stream(misc_value);
                        std::string part;
                        bool first = true;
                        while (std::getline(misc_stream, part, '|')) {
                            // Skip SpaceAfter=No entries
                            if (part.find("SpaceAfter=No") == std::string::npos) {
                                if (!first) {
                                    filtered_misc += "|";
                                }
                                filtered_misc += part;
                                first = false;
                            }
                        }
                        if (!filtered_misc.empty() && filtered_misc != "_") {
                            tok_node.append_attribute("misc") = filtered_misc.c_str();
                        }
                        // If filtered_misc is empty, don't write misc attribute
                    } else {
                        tok_node.append_attribute("misc") = token.misc.c_str();
                    }
                }
            }
            
            // Other attributes (always check for "_" except form which is already set)
            if (should_write_attr(token.reg, "reg")) {
                tok_node.append_attribute("reg") = token.reg.c_str();
            }
            if (should_write_attr(token.expan, "expan")) {
                tok_node.append_attribute("expan") = token.expan.c_str();
            }
            if (should_write_attr(token.mod, "mod")) {
                tok_node.append_attribute("mod") = token.mod.c_str();
            }
            if (should_write_attr(token.trslit, "trslit")) {
                tok_node.append_attribute("trslit") = token.trslit.c_str();
            }
            if (should_write_attr(token.ltrslit, "ltrslit")) {
                tok_node.append_attribute("ltrslit") = token.ltrslit.c_str();
            }
            if (should_write_attr(token.feats, "feats")) {
                tok_node.append_attribute("feats") = token.feats.c_str();
            }
            
            // Write custom attributes from token.attrs
            // If custom_attributes is empty, write all attributes; otherwise, only write those in the list
            for (const auto& [attr_name, attr_value] : token.attrs) {
                if (!custom_attributes.empty() && custom_attributes.find(attr_name) == custom_attributes.end()) {
                    continue;  // Skip if not in the allowed list (only when list is non-empty)
                }
                if (should_write_attr(attr_value, attr_name)) {
                    tok_node.append_attribute(attr_name.c_str()) = attr_value.c_str();
                }
            }

            if (is_mwt) {
                // Set text content first (the actual surface form, which may differ from dtok forms)
                tok_node.text().set(token.form.c_str());
                // Then append dtok children - pugixml will keep them adjacent without spaces
                for (const auto& sub : token.subtokens) {
                    auto dtok = tok_node.append_child("dtok");
                    dtok.append_attribute("form") = sub.form.c_str();
                    
                    if (should_write_attr(sub.lemma, "lemma")) {
                        dtok.append_attribute("lemma") = sub.lemma.c_str();
                    }
                    if (should_write_attr(sub.xpos, "xpos")) {
                        dtok.append_attribute("xpos") = sub.xpos.c_str();
                    }
                    if (should_write_attr(sub.upos, "upos")) {
                        dtok.append_attribute("upos") = sub.upos.c_str();
                    }
                    if (should_write_attr(sub.feats, "feats")) {
                        dtok.append_attribute("feats") = sub.feats.c_str();
                    }
                    if (should_write_attr(sub.reg, "reg")) {
                        dtok.append_attribute("reg") = sub.reg.c_str();
                    }
                    if (should_write_attr(sub.expan, "expan")) {
                        dtok.append_attribute("expan") = sub.expan.c_str();
                    }
                    if (should_write_attr(sub.mod, "mod")) {
                        dtok.append_attribute("mod") = sub.mod.c_str();
                    }
                    if (should_write_attr(sub.trslit, "trslit")) {
                        dtok.append_attribute("trslit") = sub.trslit.c_str();
                    }
                    if (should_write_attr(sub.ltrslit, "ltrslit")) {
                        dtok.append_attribute("ltrslit") = sub.ltrslit.c_str();
                    }
                    
                    // Write custom attributes from subtoken.attrs if they're in the custom_attributes list
                    for (const auto& [attr_name, attr_value] : sub.attrs) {
                        if (!custom_attributes.empty() && custom_attributes.find(attr_name) == custom_attributes.end()) {
                            continue;  // Skip if not in the allowed list
                        }
                        if (should_write_attr(attr_value, attr_name)) {
                            dtok.append_attribute(attr_name.c_str()) = attr_value.c_str();
                        }
                    }
                }
            } else {
                tok_node.text().set(token.form.c_str());
            }

            // Only add space if space_after is explicitly true
            // If space_after is false or nullopt, don't add space
            if (token.space_after.has_value() && token.space_after.value()) {
                auto space_parent = (!open_name_nodes.empty()) ? open_name_nodes.back() : s_node;
                auto space = space_parent.append_child(pugi::node_pcdata);
                space.set_value(" ");
            }
            
            // Close entities that end at this token (using token.id, not position)
            if (entities_by_end.find(token_id) != entities_by_end.end()) {
                for (const auto* entity : entities_by_end[token_id]) {
                    if (!open_name_nodes.empty()) {
                        open_name_nodes.pop_back();
                    }
                }
            }
            
            prev_token_id = token_id;
        }
    }
    
    // Step 2: Insert <p> elements before the first sentence of each paragraph span
    // and add @corresp attribute with all sentence IDs in that span
    for (const auto& par_span : doc.paragraph_spans) {
        if (par_span.start >= static_cast<int>(doc.sentences.size())) {
            continue;  // Skip invalid spans
        }
        
        // Get the first sentence's ID
        const auto& first_sentence = doc.sentences[par_span.start];
        std::string first_sentence_id = first_sentence.id.empty() ? first_sentence.sent_id : first_sentence.id;
        if (first_sentence_id.empty()) {
            first_sentence_id = "s" + std::to_string(par_span.start + 1);
        }
        
        // Find the sentence node
        auto first_sentence_node_it = sentence_id_to_node.find(first_sentence_id);
        if (first_sentence_node_it == sentence_id_to_node.end()) {
            continue;  // Skip if sentence not found
        }
        pugi::xml_node first_sentence_node = first_sentence_node_it->second;
        
        // Create <p> element and insert it before the first sentence
        pugi::xml_node p_node = text.insert_child_before("p", first_sentence_node);
        
        // Add paragraph attributes
        for (const auto& [key, value] : par_span.attrs) {
            p_node.append_attribute(key.c_str()) = value.c_str();
        }
        
        // Build @corresp attribute with all sentence IDs in this span
        std::string corresp_value;
        for (int i = par_span.start; i < par_span.end && i < static_cast<int>(doc.sentences.size()); ++i) {
            const auto& sent = doc.sentences[i];
            std::string sent_id = sent.id.empty() ? sent.sent_id : sent.id;
            if (sent_id.empty()) {
                sent_id = "s" + std::to_string(i + 1);
            }
            if (!corresp_value.empty()) {
                corresp_value += " ";
            }
            corresp_value += "#" + sent_id;
        }
        if (!corresp_value.empty()) {
            p_node.append_attribute("corresp") = corresp_value.c_str();
        }
    }
    
    // Step 3: Move all sentences listed in @corresp into their respective <p> elements
    for (pugi::xml_node p_node = text.first_child(); p_node; p_node = p_node.next_sibling()) {
        const char* node_name = p_node.name();
        if (!node_name || std::strcmp(node_name, "p") != 0) {
            continue;  // Skip non-<p> elements
        }
        
        // Get @corresp attribute
        pugi::xml_attribute corresp_attr = p_node.attribute("corresp");
        if (!corresp_attr) {
            continue;  // Skip if no @corresp
        }
        
        std::string corresp_value = corresp_attr.value();
        if (corresp_value.empty()) {
            continue;
        }
        
        // Parse sentence IDs from @corresp (format: "#s1 #s2 #s3")
        std::istringstream corresp_stream(corresp_value);
        std::string sent_id_ref;
        std::vector<pugi::xml_node> sentences_to_move;
        
        while (corresp_stream >> sent_id_ref) {
            // Remove leading '#' if present
            if (!sent_id_ref.empty() && sent_id_ref[0] == '#') {
                sent_id_ref = sent_id_ref.substr(1);
            }
            
            // Find the sentence node
            auto sent_node_it = sentence_id_to_node.find(sent_id_ref);
            if (sent_node_it != sentence_id_to_node.end()) {
                sentences_to_move.push_back(sent_node_it->second);
            }
        }
        
        // Move all sentences (and any following text nodes) into the <p> element
        // Move in reverse order to maintain document order
        for (auto it = sentences_to_move.rbegin(); it != sentences_to_move.rend(); ++it) {
            pugi::xml_node sent_node = *it;
            // Move the sentence node and any following text/whitespace nodes
            pugi::xml_node next_node = sent_node.next_sibling();
            p_node.prepend_move(sent_node);
            
            // Also move any text nodes that follow the sentence
            while (next_node && next_node.type() == pugi::node_pcdata) {
                pugi::xml_node text_node = next_node;
                next_node = next_node.next_sibling();
                p_node.prepend_move(text_node);
            }
        }
    }
}

std::string rebuild_sentence_text(const Sentence& sentence) {
    std::string text;
    const auto& toks = sentence.tokens;
    const std::size_t n = toks.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& token = toks[i];
        text += token.form;
        // When @text is missing, prefer deterministic spacing heuristics rather than relying
        // on XML pretty-print whitespace. Default to a space between tokens, then remove it
        // in punctuation-specific cases below.
        bool add_space = (i + 1 < n);
        // Heuristics to avoid spaces before closing punctuation and around hyphens
        const std::string next_form = (i + 1 < n) ? toks[i + 1].form : std::string();
        const std::string cur_form = token.form;
        auto starts_with_any = [](const std::string& s, std::initializer_list<const char*> opts) {
            if (s.empty()) return false;
            for (const char* o : opts) {
                if (!o) continue;
                const std::size_t len = std::char_traits<char>::length(o);
                if (s.size() >= len && s.compare(0, len, o) == 0) {
                    return true;
                }
            }
            return false;
        };
        if (!next_form.empty() && starts_with_any(next_form, {",", ".", ";", ":", "!", "?", ")", "]", "»"})) {
            add_space = false;
        }
        if (!cur_form.empty() && starts_with_any(cur_form, {"(", "[", "«"})) {
            add_space = false;
        }
        // No spaces immediately around hyphen in compounds (e.g., CDU-Obmann)
        if (cur_form == "-" || (!next_form.empty() && next_form == "-")) {
            add_space = false;
        }
        if (add_space) {
            text += ' ';
        }
    }
    if (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
    return text;
}

// Helper to get attribute value, checking multiple possible names
std::string get_attr_value(const pugi::xml_node& node, const std::vector<std::string>& attr_names) {
    for (const auto& name : attr_names) {
        if (node.attribute(name.c_str())) {
            return node.attribute(name.c_str()).value();
        }
    }
    return "";
}

Token parse_token(const pugi::xml_node& node, 
                  const std::unordered_map<std::string, std::string>& attr_mappings = {}) {
    Token token;
    
    // Get attribute name mappings (with defaults)
    std::string reg_attr = attr_mappings.count("reg") ? attr_mappings.at("reg") : "reg";
    std::string xpos_attr = attr_mappings.count("xpos") ? attr_mappings.at("xpos") : "xpos";
    std::string tokid_attr = attr_mappings.count("tokid") ? attr_mappings.at("tokid") : "id";
    
    // Parse id (for token.id)
    if (node.attribute("id")) {
        std::string id_attr = node.attribute("id").value();
        try {
            std::size_t dash = id_attr.find_last_of('-');
            token.id = std::stoi(id_attr.substr(dash + 1));
        } catch (...) {
            token.id = 0;
        }
    }
    
    // Parse tokid (from @id, @xml:id, or custom attribute)
    std::vector<std::string> tokid_attrs = {"id", "xml:id"};
    if (attr_mappings.count("tokid")) {
        tokid_attrs.insert(tokid_attrs.begin(), attr_mappings.at("tokid"));
    }
    token.tokid = get_attr_value(node, tokid_attrs);
    
    if (node.attribute("form")) token.form = node.attribute("form").value();
    if (token.form.empty()) token.form = node.child_value();
    
    // Parse reg (check nform first if mapped, then reg)
    std::vector<std::string> reg_attrs = {reg_attr, "reg", "nform"};
    token.reg = get_attr_value(node, reg_attrs);
    
    if (node.attribute("expan")) token.expan = node.attribute("expan").value();
    if (node.attribute("mod")) token.mod = node.attribute("mod").value();
    if (node.attribute("trslit")) token.trslit = node.attribute("trslit").value();
    if (node.attribute("ltrslit")) token.ltrslit = node.attribute("ltrslit").value();
    if (node.attribute("lemma")) token.lemma = node.attribute("lemma").value();
    
    // Parse xpos (check msd/pos first if mapped, then xpos)
    std::vector<std::string> xpos_attrs = {xpos_attr, "xpos", "msd", "pos"};
    token.xpos = get_attr_value(node, xpos_attrs);
    
    if (node.attribute("upos")) token.upos = node.attribute("upos").value();
    if (node.attribute("feats")) token.feats = node.attribute("feats").value();
    token.space_after = has_space_after(node);

    for (auto dtok : node.children("dtok")) {
        SubToken st;
        st.form = dtok.attribute("form") ? dtok.attribute("form").value() : dtok.child_value();
        if (dtok.attribute("lemma")) st.lemma = dtok.attribute("lemma").value();
        
        // Parse xpos with attribute mapping
        std::string xpos_attr = attr_mappings.count("xpos") ? attr_mappings.at("xpos") : "xpos";
        std::vector<std::string> xpos_attrs = {xpos_attr, "xpos", "msd", "pos"};
        st.xpos = get_attr_value(dtok, xpos_attrs);
        
        if (dtok.attribute("upos")) st.upos = dtok.attribute("upos").value();
        if (dtok.attribute("feats")) st.feats = dtok.attribute("feats").value();
        if (dtok.attribute("reg")) st.reg = dtok.attribute("reg").value();
        if (dtok.attribute("expan")) st.expan = dtok.attribute("expan").value();
        if (dtok.attribute("mod")) st.mod = dtok.attribute("mod").value();
        if (dtok.attribute("trslit")) st.trslit = dtok.attribute("trslit").value();
        if (dtok.attribute("ltrslit")) st.ltrslit = dtok.attribute("ltrslit").value();
        token.subtokens.push_back(std::move(st));
    }

    if (!token.subtokens.empty()) {
        token.is_mwt = true;
        token.mwt_start = token.id;
        token.mwt_end = token.id + static_cast<int>(token.subtokens.size()) - 1;
        for (const auto& st : token.subtokens) {
            token.parts.push_back(st.form);
        }
        for (std::size_t idx = 0; idx < token.subtokens.size(); ++idx) {
            token.subtokens[idx].space_after = (idx + 1 ==
                                                token.subtokens.size())
                                                   ? token.space_after.value_or(true)
                                                   : false;
        }
    }

    return token;
}

} // namespace

Document load_teitok(const std::string& path) {
    pugi::xml_document doc;
    unsigned int parse_flags = pugi::parse_default | pugi::parse_ws_pcdata | pugi::parse_ws_pcdata_single;
    if (!doc.load_file(path.c_str(), parse_flags)) {
        throw std::runtime_error("Failed to load TEI document: " + path);
    }

    Document output;
    output.id = path;

    pugi::xpath_node_set sentences = doc.select_nodes("//s");
    for (const auto& node : sentences) {
        Sentence sentence;
        pugi::xml_node sn = node.node();
        if (sn.attribute("id")) {
            sentence.id = sn.attribute("id").value();
        }
        if (sn.attribute("sent_id")) {
            sentence.sent_id = sn.attribute("sent_id").value();
        }
        if (sn.attribute("text")) {
            sentence.text = sn.attribute("text").value();
        }
        int running_id = 1;
        // Find all tokens recursively (including those inside <name> elements)
        pugi::xpath_node_set tokens = sn.select_nodes(".//tok");
        for (const auto& xp_node : tokens) {
            pugi::xml_node tok_node = xp_node.node();
            Token token = parse_token(tok_node);  // TODO: pass attr_mappings when available
            if (token.id == 0) {
                token.id = running_id;
            }
            running_id = token.id + 1;
            sentence.tokens.push_back(std::move(token));
        }
        if (sentence.text.empty()) {
            sentence.text = rebuild_sentence_text(sentence);
        }
        output.sentences.push_back(std::move(sentence));
    }
    return output;
}

// Helper function to check if a character is whitespace
bool is_whitespace_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Helper function to check if a string starts with a given prefix
bool starts_with(const char* str, const char* prefix, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (str[i] != prefix[i]) {
            return false;
        }
    }
    return true;
}

// Helper function to remove spaces/tabs between tokens (but preserve newlines and all other content)
// This is used to prevent whitespace from being added between token text content when pretty-printing
std::string remove_token_whitespace(const std::string& xml_str) {
    std::string result;
    result.reserve(xml_str.size());
    
    const char* ptr = xml_str.c_str();
    const char* end = ptr + xml_str.size();
    
    while (ptr < end) {
        // Look for patterns like ">...<tok" where we need to remove spaces/tabs but preserve newlines
        // This handles: </tok>...<tok, </name>...<tok, etc.
        if (ptr + 1 < end && *ptr == '>' && (ptr[1] == ' ' || ptr[1] == '\t')) {
            const char* check = ptr + 1;
            const char* whitespace_start = check;
            
            // Skip spaces/tabs (but not newlines)
            while (check < end && (*check == ' ' || *check == '\t')) {
                check++;
            }
            
            // Look ahead to see if there's a <tok tag (possibly after other XML elements)
            const char* lookahead = check;
            bool found_tok = false;
            
            // Skip any XML elements between the current position and <tok>
            while (lookahead < end) {
                if (lookahead + 4 <= end && starts_with(lookahead, "<tok", 4)) {
                    found_tok = true;
                    break;
                }
                // Skip XML tags: <...> or </...>
                if (*lookahead == '<') {
                    const char* tag_end = lookahead + 1;
                    while (tag_end < end && *tag_end != '>') {
                        tag_end++;
                    }
                    if (tag_end < end) {
                        lookahead = tag_end + 1;
                        // Skip spaces/tabs after the tag (but preserve newlines)
                        while (lookahead < end && (*lookahead == ' ' || *lookahead == '\t')) {
                            lookahead++;
                        }
                        continue;
                    }
                }
                // If we hit something that's not whitespace or a tag, stop looking
                if (*lookahead != ' ' && *lookahead != '\t' && *lookahead != '\n' && *lookahead != '\r' && *lookahead != '<') {
                    break;
                }
                lookahead++;
            }
            
            if (found_tok) {
                // Found pattern like ">...<tok" - remove spaces/tabs but preserve newlines
                result += '>';
                // Copy everything from whitespace_start to lookahead, but skip spaces/tabs
                bool has_newline = false;
                const char* pos = whitespace_start;
                while (pos < lookahead) {
                    if (*pos == '\n' || *pos == '\r') {
                        has_newline = true;
                        result += *pos;
                        if (*pos == '\r' && pos + 1 < lookahead && pos[1] == '\n') {
                            result += '\n';
                            pos++;
                        }
                    } else if (*pos != ' ' && *pos != '\t') {
                        // Preserve all other characters (like XML tags)
                        result += *pos;
                    }
                    // Skip spaces/tabs
                    pos++;
                }
                // If there was no newline, add one for pretty-printing
                if (!has_newline) {
                    result += '\n';
                }
                // Skip to the <tok tag
                ptr = lookahead;
                continue;
            }
        }
        
        // Also handle "><dtok" pattern
        if (ptr + 1 < end && *ptr == '>' && (ptr[1] == ' ' || ptr[1] == '\t')) {
            const char* check = ptr + 1;
            while (check < end && (*check == ' ' || *check == '\t')) {
                check++;
            }
            if (check + 5 <= end && starts_with(check, "<dtok", 5)) {
                // Found ">...<dtok", remove spaces/tabs
                result += '>';
                ptr = check;
                continue;
            }
        }
        
        // Normal character, copy it
        result += *ptr;
        ptr++;
    }
    
    return result;
}

void save_teitok(const Document& doc, const std::string& path, 
                 const std::set<std::string>& custom_attributes,
                 bool pretty_print) {
    pugi::xml_document out;
    build_teitok_document(doc, out, custom_attributes);
    // Save to string first, then post-process to remove whitespace around dtok elements and between tokens
    std::ostringstream stream;
    if (pretty_print) {
        // Save with indentation - this includes the root element
        out.save(stream, "", pugi::format_indent, pugi::encoding_utf8);
    } else {
        out.save(stream, "", pugi::format_raw, pugi::encoding_utf8);
    }
    std::string xml_str = stream.str();
    // If pretty_print is enabled, remove spaces/tabs between tokens (but preserve newlines and all other content)
    if (pretty_print && !xml_str.empty()) {
        xml_str = remove_token_whitespace(xml_str);
    }
    
    // Write the processed XML to file
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << xml_str;
    file.close();
}

std::string dump_teitok(const Document& doc, const std::set<std::string>& custom_attributes,
                        bool pretty_print) {
    pugi::xml_document out;
    build_teitok_document(doc, out, custom_attributes);
    std::ostringstream stream;
    if (pretty_print) {
        // Save with indentation - this includes the root element
        out.save(stream, "", pugi::format_indent, pugi::encoding_utf8);
    } else {
        out.save(stream, "", pugi::format_raw, pugi::encoding_utf8);
    }
    std::string xml_str = stream.str();
    // If pretty_print is enabled, remove spaces/tabs between tokens (but preserve newlines and all other content)
    if (pretty_print && !xml_str.empty()) {
        xml_str = remove_token_whitespace(xml_str);
    }
    return xml_str;
}

} // namespace flexitag

