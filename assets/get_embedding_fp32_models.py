import onnxruntime
import numpy as np
import soxr
from scipy.io import wavfile

from Audio import load_audio

# Base path cho models đã convert
MODELS_DIR = "models_fp32"

# Load tất cả models trực tiếp (không cần FP16 conversion)
print("Loading models...")

sess_options = onnxruntime.SessionOptions()
providers = ["CPUExecutionProvider"]

prompt_encoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/prompt_encoder.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - prompt_encoder loaded")
print("    Inputs:")
for inp in prompt_encoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in prompt_encoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

t2s_encoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/t2s_encoder.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - t2s_encoder loaded")
print("    Inputs:")
for inp in t2s_encoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in t2s_encoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

t2s_first_stage_decoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/t2s_first_stage_decoder.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - t2s_first_stage_decoder loaded")
print("    Inputs:")
for inp in t2s_first_stage_decoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in t2s_first_stage_decoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

t2s_stage_decoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/t2s_stage_decoder.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - t2s_stage_decoder loaded")
print("    Inputs:")
for inp in t2s_stage_decoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in t2s_stage_decoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

vocoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/vits.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - vocoder loaded")
print("    Inputs:")
for inp in vocoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in vocoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

cn_hubert = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/chinese-hubert-base.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - cn_hubert loaded")
print("    Inputs:")
for inp in cn_hubert.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in cn_hubert.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

speaker_encoder = onnxruntime.InferenceSession(
    f"{MODELS_DIR}/speaker_encoder.onnx",
    providers=providers,
    sess_options=sess_options
)
print("  - speaker_encoder loaded")
print("    Inputs:")
for inp in speaker_encoder.get_inputs():
    print(f"      {inp.name}: {inp.shape} ({inp.type})")
print("    Outputs:")
for out in speaker_encoder.get_outputs():
    print(f"      {out.name}: {out.shape} ({out.type})")

print("\nAll models loaded successfully!")

# Load audio
print("\nLoading audio...")
audio_32k = load_audio(
    audio_path="VO_JA_Archive_Cyrene_2.wav",
    target_sampling_rate=32000
)
audio_16k = soxr.resample(audio_32k, 32000, 16000, quality='hq')

audio_32k = np.expand_dims(audio_32k, axis=0)
audio_16k = np.expand_dims(audio_16k, axis=0)

# Extract SSL content
print("Extracting SSL content...")
ssl_content = cn_hubert.run(
    None, {'input_values': audio_16k}
)[0]

# Save ssl_content to binary file
print(f"Saving ssl_content to file... Shape: {ssl_content.shape}, dtype: {ssl_content.dtype}")
ssl_content.tofile("ssl_content.bin")

# Text processing
BERT_FEATURE_DIM = 1024
from Text.JapaneseG2P import japanese_to_phones
phones = japanese_to_phones("ハーイ、久しぶりね！2人きりの素敵な時間を、あなたはどう過ごしたいかしら？")
# phones = [250,  96, 322, 318, 229, 229, 227,  96, 248,  96,   3, 316,  96, 322, 252,  96, 251, 160,
#         227, 229,  86, 254, 322, 252,  96, 323, 156,  96,  86,  96, 322, 227,  96, 323, 252,  96,
#         227, 160,  86, 254, 322, 253, 254, 222, 254, 251, 160, 323, 160,  86, 318, 254, 322, 225,
#         129, 323, 229,  86, 225, 229, 322, 252,  96, 248,  96, 251, 160, 225,  96, 323, 250, 254,
#         86, 318, 229, 323, 229, 227, 160,   3]
text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

ref_seq = np.array([phones], dtype=np.int64)
ref_bert = text_bert
text_seq = np.array([phones], dtype=np.int64)
text_bert = text_bert

# T2S Encoder
print("Running T2S encoder...")
x, prompts = t2s_encoder.run(
    None,
    {
        "ref_seq": ref_seq,
        "text_seq": text_seq,
        "ref_bert": ref_bert,
        "text_bert": text_bert,
        "ssl_content": ssl_content,
    },
)

# Speaker encoder & Prompt encoder
print("Running speaker & prompt encoder...")
sv_embedding = speaker_encoder.run(None, {'waveform': audio_16k})[0]
global_emb, global_emb_advanced = prompt_encoder.run(None, {
    'ref_audio': audio_32k,
    'sv_emb': sv_embedding,
})

# Save embeddings to binary files
print(f"Saving sv_embedding to file... Shape: {sv_embedding.shape}, dtype: {sv_embedding.dtype}")
sv_embedding.tofile("sv_embedding.bin")

print(f"Saving global_emb to file... Shape: {global_emb.shape}, dtype: {global_emb.dtype}")
global_emb.tofile("global_emb.bin")

print(f"Saving global_emb_advanced to file... Shape: {global_emb_advanced.shape}, dtype: {global_emb_advanced.dtype}")
global_emb_advanced.tofile("global_emb_advanced.bin")

# First Stage Decoder
print("Running T2S first stage decoder...")
y, y_emb, *present_key_values = t2s_first_stage_decoder.run(
    None, {"x": x, "prompts": prompts}
)

# Stage Decoder (autoregressive)
print("Running T2S stage decoder...")
input_names = [inp.name for inp in t2s_stage_decoder.get_inputs()]
idx = 0
for idx in range(0, 500):
    input_feed = {
        name: data
        for name, data in zip(input_names, [y, y_emb, *present_key_values])
    }
    outputs = t2s_stage_decoder.run(None, input_feed)
    y, y_emb, stop_condition_tensor, *present_key_values = outputs

    if stop_condition_tensor:
        break

print(f"  - Generated {idx} tokens")

# Post-process semantic tokens
y[0, -1] = 0
semantic_tokens = np.expand_dims(y[:, -idx:], axis=0)
eos_indices = np.where(semantic_tokens >= 1024)
if len(eos_indices[0]) > 0:
    first_eos_index = eos_indices[-1][0]
    semantic_tokens = semantic_tokens[..., :first_eos_index]

# Vocoder
print("Running vocoder...")
audio_chunk = vocoder.run(None, {
    "text_seq": text_seq,
    "pred_semantic": semantic_tokens,
    "ge": global_emb,
    "ge_advanced": global_emb_advanced,
})[0]

# Save output
output_file = "test_fp32.wav"
print(f"\nSaving audio to {output_file}...")
print(f"  - Audio shape: {audio_chunk.shape}")
wavfile.write(output_file, 32000, audio_chunk)

print("\nDone!")
