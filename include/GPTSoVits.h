#pragma once

#include <iostream>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <onnxruntime_cxx_api.h>

class GPTSoVits
{
public:
	GPTSoVits() = delete;
	GPTSoVits(std::string modelDir, bool useCuda);
	virtual ~GPTSoVits();
	std::vector<int16_t> synthesize(std::vector<int64_t> ref_seg, std::vector<int64_t> text_seq, float speed = 1.0f);
	std::vector<float> LoadBinaryFile(std::string filename);
	void LoadStyle(std::string styleDir);
	std::vector<float> ssl_values;
    std::vector<float> global_emb_values;
    std::vector<float> global_emb_advanced_values;
private:
	Ort::Env env_;
	Ort::SessionOptions sessionOptions_;
	Ort::Session t2sEncoder, t2sFirstStageDecoder, t2sStageDecoder, vocoder;

	const float MAX_WAV_VALUE = 32767.0f;
	//Encoder input and output names
	std::array<const char*, 5> encoderInputNames = {
										"ref_seq",
										"text_seq",
										"ref_bert",
										"text_bert",
										"ssl_content"
	};
	std::array<const char*, 2> encoderOutputNames = { "x", "prompts" };

	//T2SFirstStageDecoder input and output names
    std::array<const char*, 2> t2sFirstStageDecoderInputNames = {
                                    "x",
                                    "prompts"
    };
    std::array<const char*, 50> t2sFirstStageDecoderOutputNames = {
                                    "y", "y_emb",
                                    "present_k_layer_0",
                                    "present_v_layer_0",
                                    "present_k_layer_1",
                                    "present_v_layer_1",
                                    "present_k_layer_2",
                                    "present_v_layer_2",
                                    "present_k_layer_3",
                                    "present_v_layer_3",
                                    "present_k_layer_4",
                                    "present_v_layer_4",
                                    "present_k_layer_5",
                                    "present_v_layer_5",
                                    "present_k_layer_6",
                                    "present_v_layer_6",
                                    "present_k_layer_7",
                                    "present_v_layer_7",
                                    "present_k_layer_8",
                                    "present_v_layer_8",
                                    "present_k_layer_9",
                                    "present_v_layer_9",
                                    "present_k_layer_10",
                                    "present_v_layer_10",
                                    "present_k_layer_11",
                                    "present_v_layer_11",
                                    "present_k_layer_12",
                                    "present_v_layer_12",
                                    "present_k_layer_13",
                                    "present_v_layer_13",
                                    "present_k_layer_14",
                                    "present_v_layer_14",
                                    "present_k_layer_15",
                                    "present_v_layer_15",
                                    "present_k_layer_16",
                                    "present_v_layer_16",
                                    "present_k_layer_17",
                                    "present_v_layer_17",
                                    "present_k_layer_18",
                                    "present_v_layer_18",
                                    "present_k_layer_19",
                                    "present_v_layer_19",
                                    "present_k_layer_20",
                                    "present_v_layer_20",
                                    "present_k_layer_21",
                                    "present_v_layer_21",
                                    "present_k_layer_22",
                                    "present_v_layer_22",
                                    "present_k_layer_23",
                                    "present_v_layer_23",
    };

	//t2sStageDecoder input and output names
    std::array<const char*, 50> t2sStageDecoderInputNames = {
        "iy",
        "iy_emb",
        "past_k_layer_0",
        "past_v_layer_0",
        "past_k_layer_1",
        "past_v_layer_1",
        "past_k_layer_2",
        "past_v_layer_2",
        "past_k_layer_3",
        "past_v_layer_3",
        "past_k_layer_4",
        "past_v_layer_4",
        "past_k_layer_5",
        "past_v_layer_5",
        "past_k_layer_6",
        "past_v_layer_6",
        "past_k_layer_7",
        "past_v_layer_7",
        "past_k_layer_8",
        "past_v_layer_8",
        "past_k_layer_9",
        "past_v_layer_9",
        "past_k_layer_10",
        "past_v_layer_10",
        "past_k_layer_11",
        "past_v_layer_11",
        "past_k_layer_12",
        "past_v_layer_12",
        "past_k_layer_13",
        "past_v_layer_13",
        "past_k_layer_14",
        "past_v_layer_14",
        "past_k_layer_15",
        "past_v_layer_15",
        "past_k_layer_16",
        "past_v_layer_16",
        "past_k_layer_17",
        "past_v_layer_17",
        "past_k_layer_18",
        "past_v_layer_18",
        "past_k_layer_19",
        "past_v_layer_19",
        "past_k_layer_20",
        "past_v_layer_20",
        "past_k_layer_21",
        "past_v_layer_21",
        "past_k_layer_22",
        "past_v_layer_22",
        "past_k_layer_23",
        "past_v_layer_23",
    };

    std::array<const char*, 51> t2sStageDecoderOutputNames = {
        "y", "y_emb", "stop_condition_tensor",
        "present_k_layer_0",
        "present_v_layer_0",
        "present_k_layer_1",
        "present_v_layer_1",
        "present_k_layer_2",
        "present_v_layer_2",
        "present_k_layer_3",
        "present_v_layer_3",
        "present_k_layer_4",
        "present_v_layer_4",
        "present_k_layer_5",
        "present_v_layer_5",
        "present_k_layer_6",
        "present_v_layer_6",
        "present_k_layer_7",
        "present_v_layer_7",
        "present_k_layer_8",
        "present_v_layer_8",
        "present_k_layer_9",
        "present_v_layer_9",
        "present_k_layer_10",
        "present_v_layer_10",
        "present_k_layer_11",
        "present_v_layer_11",
        "present_k_layer_12",
        "present_v_layer_12",
        "present_k_layer_13",
        "present_v_layer_13",
        "present_k_layer_14",
        "present_v_layer_14",
        "present_k_layer_15",
        "present_v_layer_15",
        "present_k_layer_16",
        "present_v_layer_16",
        "present_k_layer_17",
        "present_v_layer_17",
        "present_k_layer_18",
        "present_v_layer_18",
        "present_k_layer_19",
        "present_v_layer_19",
        "present_k_layer_20",
        "present_v_layer_20",
        "present_k_layer_21",
        "present_v_layer_21",
        "present_k_layer_22",
        "present_v_layer_22",
        "present_k_layer_23",
        "present_v_layer_23",
    };
    std::array<const char*, 4> vocoderInputNames = {
    "text_seq",
    "pred_semantic",
    "ge",
    "ge_advanced"
    };
    std::array<const char*, 1> vocoderOutputNames = {
        "audio",
    };
};

