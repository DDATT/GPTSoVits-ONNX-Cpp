# GPTSoVits-ONNX-Cpp

A C++ inference implementation of GPT-SoVITS for Japanese voice synthesis. This project leverages **ONNXRuntime** for fast model execution and the `openjtalk-native` C library for accurate Japanese text-to-phoneme conversion.

## Features

- Fast and efficient Japanese text-to-speech inference directly in C++.
- Utilizes ONNX models for optimal inference speed and cross-platform compatibility.
- Integrated Japanese G2P (Grapheme-to-Phoneme) using OpenJTalk.

## Prerequisites

Ensure you have the following dependencies installed before building the project:

- **CMake** (3.10 or higher)
- **C++17** compatible compiler (GCC, Clang, etc.)
- **ONNXRuntime** (C++ API): can use pre-built binary from [ONNXRuntime](https://github.com/microsoft/onnxruntime) or build from source.
- **openjtalk-native**: A native shared library for OpenJTalk. Please follow the instructions in the [openjtalk-native](https://github.com/ayutaz/openjtalk-native) repository to build and install it.

## Build Instructions

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd GPTSoVits-ONNX-Cpp
   ```

2. Create a build directory and compile the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Upon successful compilation, the executable `vits` will be generated in the `build` directory.

## Usage

You can use the compiled `vits` executable to synthesize audio. Below is the primary example of how the underlying C++ classes (`GPTSoVits` and `JapaneseG2P`) are utilized to generate a WAV file from Japanese text.

```cpp
#include <iostream>
#include "GPTSoVits.h"
#include "JapaneseG2P.h"
#include "wavfile.hpp"

int main() {
    // 1. Initialize the GPT-SoVITS model and OpenJTalk G2P
    GPTSoVits gptSoVits("/path_to_trained_models", false);
    gptSoVits.LoadStyle("/path_to_audio_embeddings");
    JapaneseG2P japanese_g2p("path_to_openjtalk/open_jtalk_dic_utf_8-1.11");

    // 2. Convert Japanese text to phoneme sequences
    std::string text = "ハーイ、久しぶりね！2人きりの素敵な時間を、あなたはどう過ごしたいかしら？";
    std::vector<int64_t> ref_seg = japanese_g2p.g2p(text);
    std::vector<int64_t> text_seq = japanese_g2p.g2p(text);

    // 3. Synthesize audio from the phonemes
    std::vector<int16_t> audio = gptSoVits.synthesize(ref_seg, text_seq);

    // 4. Save the generated audio to a WAV file
    std::ofstream audioFile("output.wav", std::ios::binary);
    writeWavHeader(32000, 2, 1, (int32_t)audio.size(), audioFile);
    audioFile.write((const char *)audio.data(), sizeof(int16_t) * audio.size());
    audioFile.close();

    std::cout << "Audio file saved to output.wav" << std::endl;
    return 0;
}
```

*Note: Update the paths to your trained ONNX models, audio style embeddings, and OpenJTalk dictionary as needed.*

## Acknowledgments

Special thanks to:
- **[RVC-Boss](https://github.com/RVC-Boss)** - Original [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main) implementation and research.
- **[High-Logic](https://github.com/High-Logic)** - [GENIE](https://github.com/High-Logic/Genie-TTS) optimizes the original model for outstanding CPU performance.
- **[ayutaz](https://github.com/ayutaz)** - [openjtalk-native](https://github.com/ayutaz/openjtalk-native) a cross-platform native shared library for OpenJTalk.
