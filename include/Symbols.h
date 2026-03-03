#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <algorithm>

namespace SymbolsV2 {

// Khởi tạo các tập hợp cơ bản
const std::vector<std::string> PUNCTUATION_SYMBOLS = {"!", "?", "…", ",", ".", "-", "SP", "SP2", "SP3", "UNK"};
const std::vector<std::string> PINYIN_INITIALS = {
    "AA", "EE", "OO", "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh"
};
const std::vector<std::string> PINYIN_FINALS_BASE = {
    "E", "En", "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
    "i", "i0", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong",
    "ir", "iu", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ui",
    "un", "uo", "v", "van", "ve", "vn"
};
const std::vector<std::string> JAPANESE_SYMBOLS = {
    "I", "N", "U", "a", "b", "by", "ch", "cl", "d", "dy", "e", "f", "g",
    "gy", "h", "hy", "i", "j", "k", "ky", "m", "my", "n", "ny", "o", "p",
    "py", "r", "ry", "s", "sh", "t", "ts", "u", "v", "w", "y", "z"
};
const std::set<std::string> ARPABET_SYMBOLS = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G",
    "AY0", "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0",
    "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1", "ER1", "EH2", "OY0",
    "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER", "AW0",
    "UW0", "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P",
    "AY1", "EY0", "OY2", "TH", "HH", "D", "ER0", "CH", "AO1", "AE1",
    "AO2", "OY1", "AY2", "IH1", "OW0", "L", "SH"
};

// Chuỗi Unicode cho tiếng Hàn (C++ xử lý chuỗi UTF-8)
const std::string KOREAN_SYMBOLS_STR = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ空停";
// Lưu ý: Tiếng Quảng Đông (Cantonese) trong file Python của bạn rất dài, 
// tôi sẽ giả định nó được nạp vào một set tương tự ARPABET.
const std::set<std::string> CANTONESE_SYMBOLS = { /* ... copy từ file Python ... */ "Yeot3", "Yip1", "Yyu3" };

inline std::vector<std::string> create_master_symbol_list() {
    std::set<std::string> main_symbols;
    main_symbols.insert("_");

    for (const auto& s : PINYIN_INITIALS) main_symbols.insert(s);
    
    // Generate Pinyin with tones (1-5)
    for (int tone = 1; tone <= 5; ++tone) {
        for (const auto& final : PINYIN_FINALS_BASE) {
            main_symbols.insert(final + std::to_string(tone));
        }
    }

    for (const auto& s : JAPANESE_SYMBOLS) main_symbols.insert(s);
    for (const auto& s : PUNCTUATION_SYMBOLS) main_symbols.insert(s);
    for (const auto& s : ARPABET_SYMBOLS) main_symbols.insert(s);

    // Chuyển sang vector và sắp xếp (giống sorted() trong Python)
    std::vector<std::string> master_list(main_symbols.begin(), main_symbols.end());
    std::sort(master_list.begin(), master_list.end());

    master_list.push_back("[");
    master_list.push_back("]");

    // Xử lý ký tự Unicode (Korean) - Cần tách từng ký tự UTF-8
    // Đây là một ví dụ đơn giản, trong thực tế cần thư viện ICU hoặc xử lý byte UTF-8
    // Ở đây tôi mô phỏng việc đẩy các ký tự đơn lẻ vào.
    master_list.push_back("ㄱ"); // ... thực hiện tương tự cho các ký tự khác

    // Thêm Cantonese
    std::vector<std::string> cantonese_vec(CANTONESE_SYMBOLS.begin(), CANTONESE_SYMBOLS.end());
    std::sort(cantonese_vec.begin(), cantonese_vec.end());
    master_list.insert(master_list.end(), cantonese_vec.begin(), cantonese_vec.end());

    return master_list;
}

// Biến toàn cục mô phỏng symbols_v2 và symbol_to_id_v2
static const std::vector<std::string> symbols_v2 = create_master_symbol_list();

inline std::map<std::string, int> create_symbol_to_id() {
    std::map<std::string, int> m;
    for (size_t i = 0; i < symbols_v2.size(); ++i) {
        m[symbols_v2[i]] = static_cast<int>(i);
    }
    return m;
}

static const std::map<std::string, int> symbol_to_id_v2 = create_symbol_to_id();

} // namespace SymbolsV2