#include "GPTSoVits.h"

GPTSoVits::GPTSoVits(std::string modelDir, bool useCuda) 
	:t2sEncoder(nullptr), t2sFirstStageDecoder(nullptr), t2sStageDecoder(nullptr), vocoder(nullptr)
{
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "GPTSoVits");
    env_.DisableTelemetryEvents();

    if (useCuda) {
        // Use CUDA provider
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
    }

    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    sessionOptions_.DisableCpuMemArena();
    sessionOptions_.DisableMemPattern();
    sessionOptions_.DisableProfiling();

	std::string t2sEncoderPathString = modelDir + "/t2s_encoder.onnx";
	std::string t2sFirstStageDecoderPathString = modelDir + "/t2s_first_stage_decoder.onnx";
	std::string t2sStageDecoderPathString = modelDir + "/t2s_stage_decoder.onnx";
	std::string vocoderPathString = modelDir + "/vits.onnx";

    #ifdef _WIN32
		std::wstring t2sEncoderPath_wstr = std::wstring(t2sEncoderPathString.begin(), t2sEncoderPathString.end());
		std::wstring t2sFirstStageDecoderPath_wstr = std::wstring(t2sFirstStageDecoderPathString.begin(), t2sFirstStageDecoderPathString.end());
		std::wstring t2sStageDecoderPath_wstr = std::wstring(t2sStageDecoderPathString.begin(), t2sStageDecoderPathString.end());
		std::wstring vocoderPath_wstr = std::wstring(vocoderPathString.begin(), vocoderPathString.end());

		auto t2sEncoderPath = t2sEncoderPath_wstr.c_str();
		auto t2sFirstStageDecoderPath = t2sFirstStageDecoderPath_wstr.c_str();
		auto t2sStageDecoderPath = t2sStageDecoderPath_wstr.c_str();
		auto vocoderPath = vocoderPath_wstr.c_str();
    #else
        auto t2sEncoderPath = t2sEncoderPathString.c_str();
        auto t2sFirstStageDecoderPath = t2sFirstStageDecoderPathString.c_str();
        auto t2sStageDecoderPath = t2sStageDecoderPathString.c_str();
        auto vocoderPath = vocoderPathString.c_str();
    #endif

	t2sEncoder = Ort::Session(env_, t2sEncoderPath, sessionOptions_);
	t2sFirstStageDecoder = Ort::Session(env_, t2sFirstStageDecoderPath, sessionOptions_);
	t2sStageDecoder = Ort::Session(env_, t2sStageDecoderPath, sessionOptions_);
	vocoder = Ort::Session(env_, vocoderPath, sessionOptions_);
}

GPTSoVits::~GPTSoVits() 
{
}

std::vector<float> GPTSoVits::LoadBinaryFile(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Cannot open bin file!!" << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));

    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }

    return {};
}

void GPTSoVits::LoadStyle(std::string styleDir) {
    std::string sslPath = styleDir + "/ssl_content.bin";
    std::string globalEmbPath = styleDir + "/global_emb.bin";
    std::string globalEmbAdvancedPath = styleDir + "/global_emb_advanced.bin";
    ssl_values = LoadBinaryFile(sslPath);
    global_emb_values = LoadBinaryFile(globalEmbPath);
    global_emb_advanced_values = LoadBinaryFile(globalEmbAdvancedPath);
}

std::vector<int16_t> GPTSoVits::synthesize(std::vector<int64_t> ref_seg, std::vector<int64_t> text_seq, float speed)
{
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    int ref_length = ref_seg.size();
    std::vector<int64_t> ref_bert_dims = {ref_length, 1024};
    size_t ref_bert_size = ref_length*1024;
    std::vector<float> ref_bert(ref_bert_size);
    for (size_t i = 0; i < ref_bert_size; ++i) {
        ref_bert[i] = 0.0f;
    }

    int text_length = text_seq.size();
    std::vector<int64_t> text_bert_dims = {text_length, 1024};
    size_t text_bert_size = text_length*1024;
    std::vector<float> text_bert(text_bert_size);
    for (size_t i = 0; i < text_bert_size; ++i) {
        text_bert[i] = 0.0f;
    }

    std::vector<int64_t> ssl_dims = { 1, 768, 402 };
    size_t ssl_size = 1 * 768 * 402;

    std::vector<Ort::Value> encoderInputTensors;

    encoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, ref_seg.data(), ref_seg.size(),
        std::vector<int64_t>{1, static_cast<int64_t>(ref_seg.size())}.data(), 2));
    encoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, text_seq.data(), text_seq.size(),
        std::vector<int64_t>{1, static_cast<int64_t>(text_seq.size())}.data(), 2));
    encoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, ref_bert.data(), ref_bert.size(),
        ref_bert_dims.data(), ref_bert_dims.size()));
    encoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, text_bert.data(), text_bert.size(),
        text_bert_dims.data(), text_bert_dims.size()));
    encoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, ssl_values.data(), ssl_values.size(),
        ssl_dims.data(), ssl_dims.size()));

    auto encoderOutput = t2sEncoder.Run(Ort::RunOptions{ nullptr },
        encoderInputNames.data(),
        encoderInputTensors.data(),
        encoderInputTensors.size(),
        encoderOutputNames.data(),
        encoderOutputNames.size());

    //
    //======================Running T2S first stage decoder==============================
    //y, y_emb, *present_key_values = t2s_first_stage_decoder.run(
    //    None, {"x": x, "prompts": prompts}
    //)
    //

    std::vector<Ort::Value> t2sFirstStageDecoderInput;
    t2sFirstStageDecoderInput.push_back(std::move(encoderOutput[0]));
    t2sFirstStageDecoderInput.push_back(std::move(encoderOutput[1]));

    auto t2sFirstStageDecoderOutput = t2sFirstStageDecoder.Run(
        Ort::RunOptions{ nullptr },
        t2sFirstStageDecoderInputNames.data(), t2sFirstStageDecoderInput.data(), t2sFirstStageDecoderInput.size(),
        t2sFirstStageDecoderOutputNames.data(), t2sFirstStageDecoderOutputNames.size()
    );

    //====================T2S Stage Decoder=======================
    
    std::vector<Ort::Value> t2sStageDecoderInput;
    for (int i = 0; i < 50; i++)
        t2sStageDecoderInput.push_back(std::move(t2sFirstStageDecoderOutput[i]));
    const int64_t* yOutput = nullptr;
    size_t ySize;
    int i = 0;
    for (i; i < 500; i++) {
        auto t2sStageDecoderOutput = t2sStageDecoder.Run(
            Ort::RunOptions{ nullptr },
            t2sStageDecoderInputNames.data(), t2sStageDecoderInput.data(), t2sStageDecoderInput.size(),
            t2sStageDecoderOutputNames.data(), t2sStageDecoderOutputNames.size()
        );
        t2sStageDecoderInput.clear();
        Ort::Value& stopSensor = t2sStageDecoderOutput[2];
        bool* stopData = stopSensor.GetTensorMutableData<bool>();
        if (stopData[0]) {
            yOutput = t2sStageDecoderOutput[0].GetTensorMutableData<int64_t>();
            ySize = t2sStageDecoderOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
            break;
        }
        t2sStageDecoderInput.push_back(std::move(t2sStageDecoderOutput[0]));
        t2sStageDecoderInput.push_back(std::move(t2sStageDecoderOutput[1]));

        for (size_t k = 3; k < t2sStageDecoderOutput.size(); k++) {
            t2sStageDecoderInput.push_back(std::move(t2sStageDecoderOutput[k]));
        }

    }
    std::vector<int64_t> y(yOutput, yOutput + ySize);
    if (y.size() > 0) y[y.size() - 1] = 0;
    std::vector<int64_t> semantic_tokens(y.end() - i, y.end());

    auto it = std::find_if(semantic_tokens.begin(), semantic_tokens.end(),
        [](int val) { return val >= 1024; });
    if (it != semantic_tokens.end()) {
        semantic_tokens.erase(it, semantic_tokens.end());
    }
    std::vector<int64_t> global_emb_dims = { 1, 1024, 1 };
    size_t global_emb_size = 1 * 1024 * 1;

    std::vector<int64_t> global_emb_advanced_dims = { 1, 512, 1 };
    size_t global_emb_advanced_size = 1 * 512 * 1;
    std::vector<Ort::Value> vocoderInputs;
    vocoderInputs.push_back(
        Ort::Value::CreateTensor<int64_t>(
            memoryInfo,
            text_seq.data(), text_seq.size(),
            std::vector<int64_t>{1, static_cast<int64_t>(text_seq.size())}.data(), 2
        )
    );
    vocoderInputs.push_back(
        Ort::Value::CreateTensor<int64_t>(
            memoryInfo,
            semantic_tokens.data(), semantic_tokens.size(),
            std::vector<int64_t>{1, 1, static_cast<int64_t>(semantic_tokens.size())}.data(), 3
        )
    );

    vocoderInputs.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            global_emb_values.data(), global_emb_values.size(),
            global_emb_dims.data(), global_emb_dims.size()
        )
    );

    vocoderInputs.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            global_emb_advanced_values.data(), global_emb_advanced_values.size(),
            global_emb_advanced_dims.data(), global_emb_advanced_dims.size()
        )
    );
    auto audioOutput = vocoder.Run(
        Ort::RunOptions{ nullptr },
        vocoderInputNames.data(), vocoderInputs.data(), vocoderInputs.size(),
        vocoderOutputNames.data(), vocoderOutputNames.size()
    );
    std::vector<int16_t> audioBuffer;
    const float* audioOutputData = audioOutput.front().GetTensorData<float>();

    std::vector<int64_t> audioOutputShape = audioOutput.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioOutputCount = audioOutputShape[audioOutputShape.size() - 1];
    audioBuffer.reserve(audioOutputCount);

    // Convert float audio to int16
    for (int64_t i = 0; i < audioOutputCount; i++) {
        int16_t intAudioValue = static_cast<int16_t>(
            std::clamp(audioOutputData[i] * MAX_WAV_VALUE,
                static_cast<float>(std::numeric_limits<int16_t>::min()),
                static_cast<float>(std::numeric_limits<int16_t>::max())));
        audioBuffer.push_back(intAudioValue);
    }
	return audioBuffer;
}