#include <iostream>
#include "GPTSoVits.h"
#include "JapaneseG2P.h"
#include "wavfile.hpp"

int main() {
    GPTSoVits gptSoVits("/path_to_trained_models", false);
    gptSoVits.LoadStyle("/path_to_audio_embeddings");
    JapaneseG2P japanese_g2p("path_to_openjtalk/open_jtalk_dic_utf_8-1.11");

    std::vector<int64_t> ref_seg = japanese_g2p.g2p("ハーイ、久しぶりね！2人きりの素敵な時間を、あなたはどう過ごしたいかしら？");
    std::vector<int64_t> text_seq = japanese_g2p.g2p("ハーイ、久しぶりね！2人きりの素敵な時間を、あなたはどう過ごしたいかしら？");

    std::vector<int16_t> audio = gptSoVits.synthesize(ref_seg, text_seq);
    std::ofstream audioFile("output.wav", std::ios::binary);
    writeWavHeader(32000, 2, 1, (int32_t)audio.size(), audioFile);
    audioFile.write((const char *)audio.data(), sizeof(int16_t) * audio.size());
    audioFile.close();
    std::cout<<"Audio file saved to output.wav";
    return 0;
}