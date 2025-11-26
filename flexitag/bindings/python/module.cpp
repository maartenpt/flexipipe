#include "flexitag/tagger.h"
#include "flexitag/io_teitok.h"
#include "flexitag/settings.h"
#include "flexitag/lexicon.h"
#include "flexitag/unicode_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>

namespace py = pybind11;

namespace {
// Using ICU-based Unicode sanitization
using flexitag::unicode::sanitize_utf8;
}

using flexitag::AttributeSet;
using flexitag::Document;
using flexitag::Lexicon;
using flexitag::FlexitagTagger;
using flexitag::Sentence;
using flexitag::SubToken;
using flexitag::TaggerSettings;
using flexitag::TaggerStats;
using flexitag::Token;
using flexitag::dump_teitok;
using flexitag::load_teitok;
using flexitag::save_teitok;

namespace {

std::string to_string_any(const py::handle& value) {
    if (value.is_none()) {
        return "";
    }
    return py::cast<std::string>(py::str(value));
}

void apply_attribute_set(AttributeSet& target, const py::dict& src) {
    if (auto it = src.attr("get")("form"); !it.is_none()) target.form = py::cast<std::string>(it);
    if (auto it = src.attr("get")("reg"); !it.is_none()) target.reg = py::cast<std::string>(it);
    if (auto it = src.attr("get")("expan"); !it.is_none()) target.expan = py::cast<std::string>(it);
    if (auto it = src.attr("get")("mod"); !it.is_none()) target.mod = py::cast<std::string>(it);
    if (auto it = src.attr("get")("trslit"); !it.is_none()) target.trslit = py::cast<std::string>(it);
    if (auto it = src.attr("get")("ltrslit"); !it.is_none()) target.ltrslit = py::cast<std::string>(it);
    if (auto it = src.attr("get")("tokid"); !it.is_none()) target.tokid = py::cast<std::string>(it);
    if (auto it = src.attr("get")("lemma"); !it.is_none()) target.lemma = py::cast<std::string>(it);
    if (auto it = src.attr("get")("xpos"); !it.is_none()) target.xpos = py::cast<std::string>(it);
    if (auto it = src.attr("get")("upos"); !it.is_none()) target.upos = py::cast<std::string>(it);
    if (auto it = src.attr("get")("feats"); !it.is_none()) target.feats = py::cast<std::string>(it);
    if (auto it = src.attr("get")("corr"); !it.is_none()) target.corr = py::cast<std::string>(it);
    if (auto it = src.attr("get")("lex"); !it.is_none()) target.lex = py::cast<std::string>(it);
}

SubToken subtok_from_py(const py::dict& token_dict) {
    SubToken st;
    apply_attribute_set(st, token_dict);
    if (auto it = token_dict.attr("get")("id"); !it.is_none()) st.id = py::cast<int>(it);
    if (auto it = token_dict.attr("get")("source"); !it.is_none()) st.source = py::cast<std::string>(it);
    if (auto it = token_dict.attr("get")("space_after"); !it.is_none()) st.space_after = py::cast<bool>(it);
    return st;
}

py::dict subtok_to_py(const SubToken& st) {
    py::dict out;
    out["id"] = st.id;
    out["form"] = sanitize_utf8(st.form);
    out["reg"] = sanitize_utf8(st.reg);
    out["expan"] = sanitize_utf8(st.expan);
    out["mod"] = sanitize_utf8(st.mod);
    out["trslit"] = sanitize_utf8(st.trslit);
    out["ltrslit"] = sanitize_utf8(st.ltrslit);
    out["tokid"] = sanitize_utf8(st.tokid);
    out["lemma"] = sanitize_utf8(st.lemma);
    out["xpos"] = sanitize_utf8(st.xpos);
    out["upos"] = sanitize_utf8(st.upos);
    out["feats"] = sanitize_utf8(st.feats);
    out["corr"] = sanitize_utf8(st.corr);
    out["lex"] = sanitize_utf8(st.lex);
    if (!st.source.empty()) out["source"] = sanitize_utf8(st.source);
    out["space_after"] = st.space_after;
    
    // Add custom attributes from attrs map
    if (!st.attrs.empty()) {
        py::dict attrs_dict;
        for (const auto& [key, value] : st.attrs) {
            attrs_dict[key.c_str()] = sanitize_utf8(value);
        }
        out["attrs"] = attrs_dict;
    }
    
    return out;
}

Token token_from_py(const py::dict& token_dict) {
    Token token;
    apply_attribute_set(token, token_dict);
    if (auto it = token_dict.attr("get")("id"); !it.is_none()) token.id = py::cast<int>(it);
    if (auto it = token_dict.attr("get")("is_mwt"); !it.is_none()) token.is_mwt = py::cast<bool>(it);
    if (auto it = token_dict.attr("get")("mwt_start"); !it.is_none()) token.mwt_start = py::cast<int>(it);
    if (auto it = token_dict.attr("get")("mwt_end"); !it.is_none()) token.mwt_end = py::cast<int>(it);
    if (auto it = token_dict.attr("get")("source"); !it.is_none()) token.source = py::cast<std::string>(it);
    if (auto it = token_dict.attr("get")("head"); !it.is_none()) token.head = py::cast<int>(it);
    if (auto it = token_dict.attr("get")("deprel"); !it.is_none()) token.deprel = py::cast<std::string>(it);
    if (auto it = token_dict.attr("get")("deps"); !it.is_none()) token.deps = py::cast<std::string>(it);
    if (auto it = token_dict.attr("get")("misc"); !it.is_none()) token.misc = py::cast<std::string>(it);
    if (auto it = token_dict.attr("get")("space_after"); !it.is_none()) token.space_after = py::cast<bool>(it);

    if (auto parts_obj = token_dict.attr("get")("parts"); !parts_obj.is_none()) {
        token.parts = py::cast<std::vector<std::string>>(parts_obj);
    }

    if (auto subtoks_obj = token_dict.attr("get")("subtokens"); !subtoks_obj.is_none()) {
        for (const auto& item : py::cast<py::list>(subtoks_obj)) {
            token.subtokens.push_back(subtok_from_py(py::cast<py::dict>(item)));
        }
    }
    
    // Handle custom attributes from attrs dict
    if (auto attrs_obj = token_dict.attr("get")("attrs"); !attrs_obj.is_none()) {
        auto attrs_dict = py::cast<py::dict>(attrs_obj);
        for (const auto& item : attrs_dict) {
            std::string key = py::cast<std::string>(item.first);
            std::string value = py::cast<std::string>(item.second);
            token.attrs[key] = value;
        }
    }

    return token;
}

py::dict token_to_py(const Token& token) {
    py::dict out;
    out["id"] = token.id;
    out["form"] = sanitize_utf8(token.form);
    out["reg"] = sanitize_utf8(token.reg);
    out["expan"] = sanitize_utf8(token.expan);
    out["mod"] = sanitize_utf8(token.mod);
    out["trslit"] = sanitize_utf8(token.trslit);
    out["ltrslit"] = sanitize_utf8(token.ltrslit);
    out["tokid"] = sanitize_utf8(token.tokid);
    out["corr"] = sanitize_utf8(token.corr);
    out["lex"] = sanitize_utf8(token.lex);
    out["lemma"] = sanitize_utf8(token.lemma);
    out["xpos"] = sanitize_utf8(token.xpos);
    out["upos"] = sanitize_utf8(token.upos);
    out["feats"] = sanitize_utf8(token.feats);
    out["is_mwt"] = token.is_mwt;
    out["mwt_start"] = token.mwt_start;
    out["mwt_end"] = token.mwt_end;
    std::vector<std::string> sanitized_parts;
    for (const auto& p : token.parts) {
        sanitized_parts.push_back(sanitize_utf8(p));
    }
    out["parts"] = sanitized_parts;
    out["source"] = sanitize_utf8(token.source);
    out["head"] = token.head;
    out["deprel"] = sanitize_utf8(token.deprel);
    out["deps"] = sanitize_utf8(token.deps);
    out["misc"] = sanitize_utf8(token.misc);
    out["space_after"] = token.space_after;

    py::list subtok_list;
    for (const auto& st : token.subtokens) {
        subtok_list.append(subtok_to_py(st));
    }
    out["subtokens"] = std::move(subtok_list);
    return out;
}

flexitag::Entity entity_from_py(const py::dict& entity_dict) {
    flexitag::Entity entity;
    if (auto it = entity_dict.attr("get")("start"); !it.is_none()) entity.start = py::cast<int>(it);
    if (auto it = entity_dict.attr("get")("end"); !it.is_none()) entity.end = py::cast<int>(it);
    if (auto it = entity_dict.attr("get")("label"); !it.is_none()) entity.label = py::cast<std::string>(it);
    if (auto it = entity_dict.attr("get")("text"); !it.is_none()) entity.text = py::cast<std::string>(it);
    if (auto attrs_obj = entity_dict.attr("get")("attrs"); !attrs_obj.is_none()) {
        for (const auto& item : py::cast<py::dict>(attrs_obj)) {
            auto key = py::cast<std::string>(item.first);
            auto value = to_string_any(item.second);
            entity.attrs[key] = value;
        }
    }
    return entity;
}

py::dict entity_to_py(const flexitag::Entity& entity) {
    py::dict out;
    out["start"] = entity.start;
    out["end"] = entity.end;
    out["label"] = sanitize_utf8(entity.label);
    out["text"] = sanitize_utf8(entity.text);
    py::dict attrs;
    for (const auto& [key, value] : entity.attrs) {
        attrs[py::str(sanitize_utf8(key))] = py::str(sanitize_utf8(value));
    }
    out["attrs"] = std::move(attrs);
    return out;
}

Sentence sentence_from_py(const py::dict& sentence_dict) {
    Sentence sentence;
    if (auto it = sentence_dict.attr("get")("id"); !it.is_none()) sentence.id = py::cast<std::string>(it);
    if (auto it = sentence_dict.attr("get")("sent_id"); !it.is_none()) sentence.sent_id = py::cast<std::string>(it);
    if (auto it = sentence_dict.attr("get")("text"); !it.is_none()) sentence.text = py::cast<std::string>(it);
    if (auto it = sentence_dict.attr("get")("corr"); !it.is_none()) sentence.corr = py::cast<std::string>(it);
    if (auto tokens_obj = sentence_dict.attr("get")("tokens"); !tokens_obj.is_none()) {
        for (const auto& item : py::cast<py::list>(tokens_obj)) {
            sentence.tokens.push_back(token_from_py(py::cast<py::dict>(item)));
        }
    }
    if (auto entities_obj = sentence_dict.attr("get")("entities"); !entities_obj.is_none()) {
        for (const auto& item : py::cast<py::list>(entities_obj)) {
            sentence.entities.push_back(entity_from_py(py::cast<py::dict>(item)));
        }
    }
    return sentence;
}

py::dict sentence_to_py(const Sentence& sentence) {
    py::dict out;
    out["id"] = sanitize_utf8(sentence.id);
    out["sent_id"] = sanitize_utf8(sentence.sent_id);
    out["text"] = sanitize_utf8(sentence.text);
    out["corr"] = sanitize_utf8(sentence.corr);
    py::list token_list;
    for (const auto& token : sentence.tokens) {
        token_list.append(token_to_py(token));
    }
    out["tokens"] = std::move(token_list);
    if (!sentence.entities.empty()) {
        py::list entity_list;
        for (const auto& entity : sentence.entities) {
            entity_list.append(entity_to_py(entity));
        }
        out["entities"] = std::move(entity_list);
    }
    return out;
}

Document document_from_py(const py::dict& doc_dict) {
    Document doc;
    if (auto it = doc_dict.attr("get")("id"); !it.is_none()) doc.id = py::cast<std::string>(it);
    if (auto sentences_obj = doc_dict.attr("get")("sentences"); !sentences_obj.is_none()) {
        for (const auto& item : py::cast<py::list>(sentences_obj)) {
            doc.sentences.push_back(sentence_from_py(py::cast<py::dict>(item)));
        }
    }
    // Extract paragraph spans (label="p")
    if (doc_dict.contains("spans")) {
        auto spans_obj = doc_dict["spans"];
        if (!spans_obj.is_none() && py::isinstance<py::dict>(spans_obj)) {
            try {
                auto spans_dict = py::cast<py::dict>(spans_obj);
                if (auto p_spans_obj = spans_dict.attr("get")("p"); !p_spans_obj.is_none()) {
                    auto p_spans_list = py::cast<py::list>(p_spans_obj);
                    for (const auto& item : p_spans_list) {
                        auto span_dict = py::cast<py::dict>(item);
                        flexitag::Span span;
                        if (auto label_obj = span_dict.attr("get")("label"); !label_obj.is_none()) {
                            span.label = py::cast<std::string>(label_obj);
                        }
                        if (auto start_obj = span_dict.attr("get")("start"); !start_obj.is_none()) {
                            span.start = py::cast<int>(start_obj);
                        }
                        if (auto end_obj = span_dict.attr("get")("end"); !end_obj.is_none()) {
                            span.end = py::cast<int>(end_obj);
                        }
                        if (auto attrs_obj = span_dict.attr("get")("attrs"); !attrs_obj.is_none()) {
                            auto attrs_dict = py::cast<py::dict>(attrs_obj);
                            for (const auto& [key, value] : attrs_dict) {
                                span.attrs[py::cast<std::string>(key)] = py::cast<std::string>(value);
                            }
                        }
                        doc.paragraph_spans.push_back(span);
                    }
                }
            } catch (const std::exception& e) {
                // If spans can't be parsed, just continue without them
            }
        }
    }
    return doc;
}

py::dict document_to_py(const Document& doc) {
    py::dict out;
    out["id"] = sanitize_utf8(doc.id);
    py::list sentences;
    for (const auto& sent : doc.sentences) {
        sentences.append(sentence_to_py(sent));
    }
    out["sentences"] = std::move(sentences);
    return out;
}

py::dict stats_to_py(const TaggerStats& stats) {
    py::dict out;
    out["word_count"] = stats.word_count;
    out["oov_count"] = stats.oov_count;
    out["elapsed_seconds"] = stats.elapsed_seconds;
    return out;
}

py::dict load_teitok_py(const std::string& path) {
    Document doc = load_teitok(path);
    return document_to_py(doc);
}

void save_teitok_py(const py::dict& doc_dict, const std::string& path, 
                    const py::object& custom_attributes = py::none(),
                    bool pretty_print = false) {
    Document doc = document_from_py(doc_dict);
    std::set<std::string> custom_attrs_set;
    if (!custom_attributes.is_none()) {
        if (py::isinstance<py::list>(custom_attributes)) {
            for (const auto& item : py::cast<py::list>(custom_attributes)) {
                custom_attrs_set.insert(py::cast<std::string>(item));
            }
        } else if (py::isinstance<py::set>(custom_attributes)) {
            for (const auto& item : py::cast<py::set>(custom_attributes)) {
                custom_attrs_set.insert(py::cast<std::string>(item));
            }
        }
    }
    save_teitok(doc, path, custom_attrs_set, pretty_print);
}

std::string dump_teitok_py(const py::dict& doc_dict, const py::object& custom_attributes = py::none(),
                           bool pretty_print = false) {
    Document doc = document_from_py(doc_dict);
    std::set<std::string> custom_attrs_set;
    if (!custom_attributes.is_none()) {
        if (py::isinstance<py::list>(custom_attributes)) {
            for (const auto& item : py::cast<py::list>(custom_attributes)) {
                custom_attrs_set.insert(py::cast<std::string>(item));
            }
        } else if (py::isinstance<py::set>(custom_attributes)) {
            for (const auto& item : py::cast<py::set>(custom_attributes)) {
                custom_attrs_set.insert(py::cast<std::string>(item));
            }
        }
    }
    return dump_teitok(doc, custom_attrs_set, pretty_print);
}

TaggerSettings settings_from_py(const py::dict& options) {
    TaggerSettings settings;
    for (const auto& item : options) {
        auto key = py::cast<std::string>(item.first);
        const auto& value = item.second;
        std::string value_str = to_string_any(value);
        settings.options[key] = value_str;

        if (key == "pid") {
            settings.pid = value_str;
        } else if (key == "xmlfile") {
            settings.xmlfile = value_str;
        } else if (key == "outfile") {
            settings.outfile = value_str;
        } else if (key == "settings") {
            settings.settings_file = value_str;
        } else if (key == "lexicon") {
            settings.lexicon_file = value_str;
        } else if (key == "extra-vocab" || key == "extra_vocab") {
            // Support both list and single string
            if (py::isinstance<py::list>(value)) {
                for (const auto& item : py::cast<py::list>(value)) {
                    settings.extra_vocab_files.push_back(py::cast<std::string>(item));
                }
            } else {
                settings.extra_vocab_files.push_back(value_str);
            }
        } else if (key == "training") {
            settings.training_folder = value_str;
        } else if (key == "verbose") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.verbose = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.verbose = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.verbose = py::cast<bool>(value);
            }
        } else if (key == "debug") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.debug = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.debug = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.debug = py::cast<bool>(value);
            }
        } else if (key == "test") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.test = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.test = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.test = py::cast<bool>(value);
            }
        } else if (key == "overwrite") {
            // Handle overwrite as boolean - convert to "1" or "0" for get_bool()
            bool overwrite_val = false;
            if (py::isinstance<py::bool_>(value)) {
                overwrite_val = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                overwrite_val = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                overwrite_val = py::cast<bool>(value);
            }
            settings.options[key] = overwrite_val ? "1" : "0";
        }
    }
    return settings;
}

void apply_overrides(TaggerSettings& settings, const py::dict& overrides) {
    for (const auto& item : overrides) {
        auto key = py::cast<std::string>(item.first);
        const auto& value = item.second;
        std::string value_str = to_string_any(value);
        settings.options[key] = value_str;

        if (key == "pid") {
            settings.pid = value_str;
        } else if (key == "xmlfile") {
            settings.xmlfile = value_str;
        } else if (key == "outfile") {
            settings.outfile = value_str;
        } else if (key == "settings") {
            settings.settings_file = value_str;
        } else if (key == "lexicon") {
            settings.lexicon_file = value_str;
        } else if (key == "extra-vocab" || key == "extra_vocab") {
            // Support both list and single string
            if (py::isinstance<py::list>(value)) {
                for (const auto& item : py::cast<py::list>(value)) {
                    settings.extra_vocab_files.push_back(py::cast<std::string>(item));
                }
            } else {
                settings.extra_vocab_files.push_back(value_str);
            }
        } else if (key == "training") {
            settings.training_folder = value_str;
        } else if (key == "verbose") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.verbose = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.verbose = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.verbose = py::cast<bool>(value);
            }
        } else if (key == "debug") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.debug = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.debug = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.debug = py::cast<bool>(value);
            }
        } else if (key == "test") {
            // Handle boolean values that might come as strings from JSON or _prepare_flexitag_options
            if (py::isinstance<py::bool_>(value)) {
                settings.test = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                settings.test = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                settings.test = py::cast<bool>(value);
            }
        } else if (key == "overwrite") {
            // Handle overwrite as boolean - convert to "1" or "0" for get_bool()
            bool overwrite_val = false;
            if (py::isinstance<py::bool_>(value)) {
                overwrite_val = py::cast<bool>(value);
            } else if (py::isinstance<py::str>(value)) {
                std::string val_str = py::cast<std::string>(value);
                overwrite_val = (val_str == "true" || val_str == "True" || val_str == "1");
            } else {
                overwrite_val = py::cast<bool>(value);
            }
            settings.options[key] = overwrite_val ? "1" : "0";
        }
    }
}

class FlexitagEngine {
public:
    FlexitagEngine(const std::string& params_file, const py::dict& base_options = py::dict()) {
        base_settings_ = settings_from_py(base_options);
        base_settings_.options["params"] = params_file;

        lexicon_ = std::make_shared<Lexicon>();
        lexicon_->set_endlen(base_settings_.get_int("endlen", 6));
        if (!lexicon_->load(params_file)) {
            throw std::runtime_error("Failed to load flexitag parameters: " + params_file);
        }
        
        // Load additional vocab files (merge mode)
        for (const auto& extra_vocab : base_settings_.extra_vocab_files) {
            if (!lexicon_->load(extra_vocab, true)) {  // true = merge mode
                throw std::runtime_error("Failed to load extra vocabulary: " + extra_vocab);
            }
        }
        
        // Apply default settings from the model (if available), but don't override provided options
        for (const auto& [key, value] : lexicon_->default_settings()) {
            if (base_settings_.get(key).empty()) {
                base_settings_.options[key] = value;
            }
        }
    }

    py::tuple tag(const py::dict& doc_dict, const py::dict& overrides = py::dict()) const {
        TaggerSettings run_settings = base_settings_;
        if (!overrides.empty()) {
            apply_overrides(run_settings, overrides);
        }

        FlexitagTagger tagger;
        tagger.configure(run_settings);
        tagger.set_lexicon(lexicon_);

        Document input_doc = document_from_py(doc_dict);
        TaggerStats stats;
        Document output_doc = tagger.tag(input_doc, &stats);

        py::dict result_doc = document_to_py(output_doc);
        py::dict result_stats = stats_to_py(stats);
        return py::make_tuple(result_doc, result_stats);
    }

private:
    TaggerSettings base_settings_;
    std::shared_ptr<Lexicon> lexicon_;
};

py::tuple tag_document(const py::dict& doc_dict,
                       const std::string& params_file,
                       const py::dict& options = py::dict()) {
    FlexitagEngine engine(params_file, options);
    return engine.tag(doc_dict);
}

} // namespace

PYBIND11_MODULE(flexitag_py, m) {
    m.doc() = "Python bindings for the flexitag Viterbi tagger";

    py::class_<FlexitagEngine>(m, "FlexitagEngine")
        .def(py::init<const std::string&, const py::dict&>(),
             py::arg("params_file"), py::arg("options") = py::dict())
        .def("tag", &FlexitagEngine::tag, py::arg("document"), py::arg("overrides") = py::dict(),
             "Tag a document and return the updated document plus stats");

    m.def("tag_document", &tag_document, py::arg("document"), py::arg("params_file"),
          py::arg("options") = py::dict(),
          "Tag a document represented as nested dictionaries using the specified parameter file");

    m.def("load_teitok", &load_teitok_py, py::arg("path"),
          "Load a TEI/TEITOK XML file into a document dictionary");
    m.def("save_teitok", &save_teitok_py, py::arg("document"), py::arg("path"), 
          py::arg("custom_attributes") = py::none(),
          py::arg("pretty_print") = false,
          "Write a document dictionary to TEI/TEITOK XML");
    m.def("dump_teitok", &dump_teitok_py, 
          py::arg("document"), 
          py::arg("custom_attributes") = py::none(),
          py::arg("pretty_print") = false,
          "Serialize a document dictionary to TEI/TEITOK XML string");
}
