#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "openjtalk_native.h"
#include <sstream>
#include "Symbols.h"

class JapaneseG2P {
public:
    JapaneseG2P() = delete;
    JapaneseG2P(std::string dict_path);
    virtual ~JapaneseG2P();
    std::vector<int64_t> g2p(std::string text);
private:
    std::string dict_path;
};