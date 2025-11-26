#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <map>
#include <set>

namespace flexitag {

struct AttributeSet {
    std::string form;
    std::string reg;
    std::string expan;
    std::string mod;      // ModernForm
    std::string trslit;   // Translit
    std::string ltrslit;  // LTransLit
    std::string tokid;    // TokId
    std::string lemma;
    std::string xpos;
    std::string upos;
    std::string feats;
    std::string corr;     // Morphological/grammatical correction
    std::string lex;      // Lexical correction

    AttributeSet() = default;
};

struct SubToken : AttributeSet {
    int id = 0;
    std::string source;
    bool space_after = true;
    std::map<std::string, std::string> attrs;  // Additional custom attributes
};

struct Token : AttributeSet {
    int id = 0;
    bool is_mwt = false;
    int mwt_start = 0;
    int mwt_end = 0;
    std::vector<std::string> parts;
    std::vector<SubToken> subtokens;
    std::string source;
    int head = 0;
    std::string deprel;
    std::string deps;
    std::string misc;
    std::optional<bool> space_after;
    std::map<std::string, std::string> attrs;  // Additional custom attributes
};

struct Entity {
    int start;  // Token index (1-based) of first token in entity
    int end;    // Token index (1-based) of last token in entity (inclusive)
    std::string label;  // Entity type (e.g., "PERSON", "ORG", "GPE")
    std::string text;   // Optional: text of the entity
    std::map<std::string, std::string> attrs;  // Additional attributes
};

struct Span {
    std::string label;  // Span label (e.g., "p", "text")
    int start;  // Sentence index (0-based) where span starts
    int end;    // Sentence index (0-based) where span ends (exclusive)
    std::map<std::string, std::string> attrs;  // Additional attributes
};

struct Sentence {
    std::string id;
    std::string sent_id;
    std::string text;
    std::string corr;
    std::vector<Token> tokens;
    std::vector<Entity> entities;  // Named entities
};

struct Document {
    std::string id;
    std::vector<Sentence> sentences;
    std::vector<Span> paragraph_spans;  // Paragraph spans (label="p")
};

struct WordCandidate {
    std::string form;
    std::string lemma;
    std::string tag;
    std::string source;
    std::string wcase;
    float prob = 0.f;
    int freq = 0;
    std::unordered_map<std::string, int> lemmatizations;
    std::vector<std::shared_ptr<WordCandidate>> dtoks;
    std::shared_ptr<WordCandidate> lexitem;
    std::map<std::string, int> lexprobs;
    std::optional<AttributeSet> lex_attributes;
    Token* token = nullptr;
};

enum class InputFormat {
    TeiXml,
    Conllu,
    PlainText
};

enum class OutputFormat {
    TeiXml,
    Conllu
};

} // namespace flexitag

