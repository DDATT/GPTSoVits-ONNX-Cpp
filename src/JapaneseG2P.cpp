#include "JapaneseG2P.h"

std::vector<std::string> extract_punctuation(const std::string& text) {
    std::vector<std::string> marks;
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        if (c < 128) {
            if (c == ',' || c == '.' || c == '!' || c == '?') {
                marks.push_back(std::string(1, c));
            }
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < text.length()) {
                std::string sub = text.substr(i, 3);
                if (sub == "、" || sub == "，") marks.push_back(",");
                else if (sub == "。" || sub == "．") marks.push_back(".");
                else if (sub == "！") marks.push_back("!");
                else if (sub == "？") marks.push_back("?");
                else if (sub == "…") marks.push_back("…");
            }
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            i += 4;
        } else {
            i++;
        }
    }
    return marks;
}

std::vector<std::string> get_phoneme_vector(const char* raw_phonemes) {
    std::vector<std::string> list;
    if (!raw_phonemes) return list;
    
    std::stringstream ss(raw_phonemes);
    std::string p;
    while (ss >> p) {
        list.push_back(p);
    }
    return list;
}

JapaneseG2P::JapaneseG2P(std::string dict_path) {
    this->dict_path = dict_path;
}

JapaneseG2P::~JapaneseG2P() {
}

std::vector<int64_t> JapaneseG2P::g2p(std::string text) {
    std::vector<std::string> phones;
    void* handle = openjtalk_native_create(dict_path.c_str());

    // 2. Convert Japanese text to phonemes
    OpenJTalkNativeProsodyResult* prosody = openjtalk_native_phonemize_with_prosody(handle, text.c_str());
    std::vector<std::string> phoneme_list = get_phoneme_vector(prosody->phonemes);

    for (int n = 0; n < prosody->phoneme_count; ++n) {
        if (n >= phoneme_list.size()) break;

        std::string p3 = phoneme_list[n];
        if (p3 == "A" || p3 == "E" || p3 == "I" || p3 == "O" || p3 == "U") {
            p3[0] = std::tolower(p3[0]);
        }

        if (p3 == "sil") {
            if (n == 0) {
                phones.push_back("^");
            }
            continue;
        } else if (p3 == "pau") {
            phones.push_back("_");
            continue;
        } else {
            phones.push_back(p3);
        }

        int a1 = prosody->prosody_a1[n];
        int a2 = prosody->prosody_a2[n];
        int a3 = prosody->prosody_a3[n];

        int a2_next = (n + 1 < phoneme_list.size()) ? prosody->prosody_a2[n + 1] : -1;

        bool is_vowel_or_cl = (p3 == "a" || p3 == "e" || p3 == "i" || p3 == "o" || p3 == "u" || 
                            p3 == "A" || p3 == "E" || p3 == "I" || p3 == "O" || p3 == "U" || 
                            p3 == "N" || p3 == "c" || p3 == "l");

        if (a3 == 1 && a2_next == 1 && is_vowel_or_cl) {
            phones.push_back("#");
        } 
        else if (a1 == 0 && a2_next == a2 + 1) {
            phones.push_back("]");
        } 
        else if (a2 == 1 && a2_next == 2) {
            phones.push_back("[");
        }
    }
    openjtalk_native_free_prosody_result(prosody);
    openjtalk_native_destroy(handle);
    phones.erase(phones.begin());
    
    std::vector<std::string> marks = extract_punctuation(text);
    size_t mark_idx = 0;
    for (auto& ph : phones) {
        if (ph == "_") {
            if (mark_idx < marks.size()) {
                ph = marks[mark_idx];
                mark_idx++;
            }
        }
    }
    while (mark_idx < marks.size()) {
        phones.push_back(marks[mark_idx]);
        mark_idx++;
    }

    std::vector<int64_t> phone_ids;

    // 2. Lọc và chuyển đổi sang ID
    for (const std::string& ph : phones) {
        // Kiểm tra ph có trong symbols_v2 (thông qua map để tối ưu tốc độ)
        auto it = SymbolsV2::symbol_to_id_v2.find(ph);
        if (it != SymbolsV2::symbol_to_id_v2.end()) {
            phone_ids.push_back(it->second);
        }
    }
    return phone_ids;
}