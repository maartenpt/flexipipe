#include "io_conllu.h"
#include "types.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cctype>

bool CoNLLUReader::parse_line(const std::string& line, Token& token) {
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    trimmed.erase(trimmed.find_last_not_of(" \t") + 1);
    
    if (trimmed.empty() || trimmed[0] == '#') {
        return false;
    }
    
    std::vector<std::string> parts;
    std::istringstream iss(trimmed);
    std::string part;
    
    while (std::getline(iss, part, '\t')) {
        parts.push_back(part);
    }
    
    if (parts.empty()) {
        return false;
    }
    
    // Handle MWT (Multi-Word Token) - e.g., "1-2"
    std::string tid_str = parts[0];
    if (tid_str.find('-') != std::string::npos) {
        // MWT line
        size_t dash_pos = tid_str.find('-');
        try {
            token.mwt_start = std::stoi(tid_str.substr(0, dash_pos));
            token.mwt_end = std::stoi(tid_str.substr(dash_pos + 1));
            token.is_mwt = true;
            if (parts.size() > 1) {
                token.form = parts[1];
            }
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Regular token
    try {
        token.id = std::stoi(tid_str);
        token.is_mwt = false;
    } catch (...) {
        return false;
    }
    
    // VRT format (1-3 columns)
    if (parts.size() >= 1) {
        token.form = parts[0];
    }
    if (parts.size() >= 2) {
        token.lemma = parts[1];
    }
    if (parts.size() >= 3) {
        token.upos = parts[2];
    }
    
    // Full CoNLL-U format (10 columns)
    if (parts.size() >= 10) {
        token.form = parts[1];
        token.lemma = parts[2];
        token.upos = parts[3];
        token.xpos = parts[4];
        token.feats = parts[5];
        token.head = parts[6];
        token.deprel = parts[7];
        token.misc = parts[9];
    }
    
    return true;
}

std::vector<Sentence> CoNLLUReader::load_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open CoNLL-U file: " << file_path << std::endl;
        return {};
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();
    
    return load_string(content);
}

std::vector<Sentence> CoNLLUReader::load_string(const std::string& content) {
    std::vector<Sentence> sentences;
    Sentence current_sentence;
    std::string current_text;
    std::unordered_map<std::string, std::pair<std::string, std::pair<int, int>>> mwt_info;
    
    std::istringstream iss(content);
    std::string line;
    
    while (std::getline(iss, line)) {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        
        // Check for # text = comment
        if (trimmed.find("# text =") == 0) {
            current_text = trimmed.substr(8);
            // Trim whitespace
            current_text.erase(0, current_text.find_first_not_of(" \t"));
            continue;
        }
        
        // Check for # sent_id = comment
        if (trimmed.find("# sent_id =") == 0) {
            current_sentence.sent_id = trimmed.substr(11);
            current_sentence.sent_id.erase(0, current_sentence.sent_id.find_first_not_of(" \t"));
            continue;
        }
        
        // Empty line = sentence boundary
        if (trimmed.empty()) {
            if (!current_sentence.tokens.empty()) {
                if (!current_text.empty()) {
                    current_sentence.text = current_text;
                }
                sentences.push_back(current_sentence);
                current_sentence = Sentence();
                current_text.clear();
            }
            continue;
        }
        
        Token token;
        if (parse_line(line, token)) {
            if (token.is_mwt) {
                // MWT token - add it to the sentence so it's preserved
                // The MWT line represents the orthographic form (e.g., "im")
                current_sentence.tokens.push_back(token);
            } else {
                // Regular token - check if it's part of an MWT
                // If this token's id is within an MWT range, mark it
                for (const auto& mwt_pair : mwt_info) {
                    int mwt_start = mwt_pair.second.second.first;
                    int mwt_end = mwt_pair.second.second.second;
                    if (token.id >= mwt_start && token.id <= mwt_end) {
                        // This token is part of an MWT
                        // The MWT token should already be in the list
                        // Just add this split token
                    }
                }
                current_sentence.tokens.push_back(token);
            }
        }
    }
    
    // Add last sentence if any
    if (!current_sentence.tokens.empty()) {
        if (!current_text.empty()) {
            current_sentence.text = current_text;
        }
        sentences.push_back(current_sentence);
    }
    
    return sentences;
}

void CoNLLUWriter::write(const std::vector<Sentence>& sentences, std::ostream& out,
                          const std::string& generator, const std::string& model) {
    // Write document-level metadata (only once, before first sentence)
    if (!sentences.empty()) {
        if (!generator.empty()) {
            out << "# generator = " << generator << "\n";
        }
        if (!model.empty()) {
            out << "# model = " << model << "\n";
        }
    }
    
    for (const auto& sentence : sentences) {
        // Write sentence metadata
        if (!sentence.sent_id.empty()) {
            out << "# sent_id = " << sentence.sent_id << "\n";
        }
        if (!sentence.text.empty()) {
            out << "# text = " << sentence.text << "\n";
        }
        
        // Derive SpaceAfter from original text if available (same logic as Python)
        std::vector<bool> space_after_flags;
        if (!sentence.text.empty()) {
            // Build MWT mapping: for split tokens, map to their MWT form
            std::unordered_map<int, std::string> mwt_forms;  // Maps token ID -> MWT form (e.g., {19: "im", 20: "im"})
            std::unordered_set<int> split_token_ids;  // IDs that are split parts of MWTs
            
            for (const auto& token : sentence.tokens) {
                if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
                    // Map all split token IDs to this MWT form
                    for (int split_id = token.mwt_start; split_id <= token.mwt_end; split_id++) {
                        mwt_forms[split_id] = token.form;
                        split_token_ids.insert(split_id);
                    }
                }
            }
            
            // Match tokens to original text to derive SpaceAfter
            size_t text_pos = 0;
            std::unordered_map<int, bool> mwt_space_after;  // Cache SpaceAfter for MWT split tokens
            
            for (const auto& token : sentence.tokens) {
                // Skip MWT tokens themselves
                if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
                    space_after_flags.push_back(true);  // Default for MWT line
                    continue;
                }
                
                std::string form = token.form;
                if (form.empty() || form == "_") {
                    space_after_flags.push_back(true);
                    continue;
                }
                
                // If this token is a split part of an MWT, use the MWT form for matching
                std::string form_to_match = form;
                bool is_split_token = false;
                if (token.id > 0 && mwt_forms.count(token.id)) {
                    form_to_match = mwt_forms[token.id];
                    is_split_token = true;
                    
                    // Check if we've already computed SpaceAfter for this MWT
                    if (mwt_space_after.count(token.id)) {
                        space_after_flags.push_back(mwt_space_after[token.id]);
                        continue;
                    }
                }
                
                // Try to find the token in the original text
                size_t found_pos = sentence.text.find(form_to_match, text_pos);
                
                if (found_pos == std::string::npos) {
                    // Try case-insensitive match
                    std::string text_lower = sentence.text;
                    std::string form_lower = form_to_match;
                    std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
                    std::transform(form_lower.begin(), form_lower.end(), form_lower.begin(), ::tolower);
                    found_pos = text_lower.find(form_lower, text_pos);
                }
                
                bool space_after = true;  // Default
                if (found_pos != std::string::npos && found_pos >= text_pos) {
                    size_t end_pos = found_pos + form_to_match.length();
                    if (end_pos < sentence.text.length()) {
                        space_after = std::isspace(static_cast<unsigned char>(sentence.text[end_pos]));
                    }
                    text_pos = end_pos;
                    // Skip whitespace for next token
                    if (space_after) {
                        while (text_pos < sentence.text.length() && 
                               std::isspace(static_cast<unsigned char>(sentence.text[text_pos]))) {
                            text_pos++;
                        }
                    }
                }
                
                // If this is a split token, cache the SpaceAfter value for all tokens in this MWT range
                if (is_split_token && token.id > 0) {
                    // Find the MWT range this token belongs to
                    for (const auto& mwt_token : sentence.tokens) {
                        if (mwt_token.is_mwt && mwt_token.mwt_start > 0 && mwt_token.mwt_end > 0) {
                            if (mwt_token.mwt_start <= token.id && token.id <= mwt_token.mwt_end) {
                                // Cache SpaceAfter for all split tokens in this MWT
                                for (int split_id = mwt_token.mwt_start; split_id <= mwt_token.mwt_end; split_id++) {
                                    mwt_space_after[split_id] = space_after;
                                }
                                break;
                            }
                        }
                    }
                }
                
                space_after_flags.push_back(space_after);
            }
        } else {
            // No original text, use defaults
            for (const auto& token : sentence.tokens) {
                space_after_flags.push_back(true);
            }
        }
        
        // Write tokens
        size_t space_after_idx = 0;
        for (const auto& token : sentence.tokens) {
            // Handle MWT (multi-word tokens)
            if (token.is_mwt && token.mwt_start > 0 && token.mwt_end > 0) {
                out << token.mwt_start << "-" << token.mwt_end << "\t"
                    << token.form << "\t"
                    << "_\t"  // lemma for MWT
                    << "_\t"  // upos for MWT
                    << "_\t"  // xpos for MWT
                    << "_\t"  // feats for MWT
                    << "_\t"  // head for MWT
                    << "_\t"  // deprel for MWT
                    << "_\t"  // deps
                    << "_\n";  // misc
                space_after_idx++;
                continue;
            }
            
            // Build misc field with SpaceAfter if needed
            std::string misc = token.misc;
            if (space_after_idx < space_after_flags.size()) {
                bool space_after = space_after_flags[space_after_idx];
                if (!space_after) {
                    if (misc.empty() || misc == "_") {
                        misc = "SpaceAfter=No";
                    } else {
                        // Check if SpaceAfter=No is already in misc
                        if (misc.find("SpaceAfter=No") == std::string::npos) {
                            misc += "|SpaceAfter=No";
                        }
                    }
                }
            }
            
            out << token.id << "\t"
                << token.form << "\t"
                << (token.lemma.empty() ? "_" : token.lemma) << "\t"
                << (token.upos.empty() ? "_" : token.upos) << "\t"
                << (token.xpos.empty() ? "_" : token.xpos) << "\t"
                << (token.feats.empty() ? "_" : token.feats) << "\t"
                << (token.head.empty() ? "_" : token.head) << "\t"
                << (token.deprel.empty() ? "_" : token.deprel) << "\t"
                << "_\t"  // deps (usually empty)
                << (misc.empty() ? "_" : misc) << "\n";
            
            space_after_idx++;
        }
        
        // Empty line between sentences
        out << "\n";
    }
}

bool CoNLLUWriter::write_file(const std::vector<Sentence>& sentences, const std::string& file_path,
                               const std::string& generator, const std::string& model) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << file_path << std::endl;
        return false;
    }
    
    write(sentences, file, generator, model);
    file.close();
    return true;
}

