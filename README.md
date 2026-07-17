# respeak

Provides the modules required for conversation and is released as a Python SDK.

## Install

```bash
pip install respeak-ai

# local development
pip install -e .
```

Optional extras:

```bash
pip install "respeak-ai[vad]"
pip install "respeak-ai[tts]"
pip install "respeak-ai[llm]"
pip install "respeak-ai[a2f]"
pip install "respeak-ai[dev]"
```

## Supported models

| Module | Class | Recommended weights | Source |
|--------|-------|---------------------|--------|
| ASR | `StreamingParaformerAsr` | `paraformer-zh-streaming` (`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`) | [ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) · [Hugging Face](https://huggingface.co/funasr/paraformer-zh-streaming) · [FunASR](https://github.com/modelscope/FunASR) |
| ASR (punc) | used via `punc_model=` | `ct-punc` (`iic/punc_ct-transformer_cn-en-common-vocab471067-large`) | [ModelScope](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large) · [Hugging Face](https://huggingface.co/funasr/ct-punc) |
| TTS | `CosyVoice3Tts` | `Fun-CosyVoice3-0.5B-2512` | [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) · [Hugging Face](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) · [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| VAD | `SileroVad` | Silero VAD (auto-download via `silero-vad`) | [GitHub](https://github.com/snakers4/silero-vad) · [PyPI](https://pypi.org/project/silero-vad/) |
| LLM | `Qwen3LLM` | `Qwen3-8B` (or other Qwen3 / Qwen2.5 Instruct checkpoints) | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B) · [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) · [Qwen](https://github.com/QwenLM/Qwen3) |
| Audio2Face | `NvidiaAudio2Face3D` | `Audio2Face-3D-v2.3.1-Claire` | [Hugging Face](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire) · [Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D) |

Download examples:

```bash
# ASR + punctuation (ModelScope)
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --local_dir ./models/paraformer-zh-streaming
modelscope download --model iic/punc_ct-transformer_cn-en-common-vocab471067-large \
  --local_dir ./models/punc_ct-transformer_cn-en-common-vocab471067-large

# TTS (ModelScope or Hugging Face)
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local_dir ./models/Fun-CosyVoice3-0.5B-2512

# LLM
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B

# Audio2Face 3D (Hugging Face)
huggingface-cli download nvidia/Audio2Face-3D-v2.3.1-Claire \
  --local-dir ./models/audio2face-3d-v2.3.1-claire
```

Silero VAD needs no manual download: `pip install "respeak-ai[vad]"` then `SileroVad.from_pretrained()`.

## Usage

```python
from respeak import StreamingParaformerAsr, CosyVoice3Tts, SileroVad, Qwen3LLM, NvidiaAudio2Face3D

# ASR
asr = StreamingParaformerAsr.from_pretrained(asr="path/or/model_id", punc_model="path/or/model_id")
text = asr.generate(input=audio_chunk, is_final=False, cache={})

# CosyVoice3 TTS (vLLM)
tts = CosyVoice3Tts.from_pretrained(
    model_dir="path/to/TTS_Model",
    prompt_text="...",
    prompt_wav="path/to/prompt.wav",
    load_vllm=True,
)
for wav in tts.generate("你好，欢迎使用本系统。", stream=True):
    ...

# VAD
vad = SileroVad.from_pretrained()
speech_prob = vad.generate(audio_chunk)

# LLM sentence-level streaming
llm = Qwen3LLM.from_pretrained(
    "path/to/LLM_Model_ft",
    base_prompt_file="path/to/base_prompt.md",
    persona_setting="你叫小睿，是一个温和的中文助手。",
    strategy_prompt="回答要简洁自然。",
)
for sentence in llm.generate("你好，介绍一下你自己。", stream=True):
    ...

# Audio2Face 3D (requires .[a2f])
a2f = NvidiaAudio2Face3D.from_pretrained("path/to/audio2face-3d-model", use_cuda=True)
weights = a2f.generate(audio_window)          # one 520ms window -> [51]
frames = a2f.generate(audio, stream=True, is_final=True)  # sliding window -> list[[51]]
```

New model types should live under `src/respeak/models/<name>/` (subclass `BaseModel`, expose `from_pretrained` / `generate`). Add matching tests under `tests/models/<name>/`.

## Examples

```bash
# ASR
python examples/paraformer_asr_streaming.py \
  --asr path/to/ASR_Model \
  --punc path/to/PUNC_Model \
  --audio path/to/test.wav

# TTS (requires .[tts])
python examples/cosyvoice3_tts_streaming.py \
  --model-dir path/to/Fun-CosyVoice3 \
  --prompt-wav path/to/prompt.wav \
  --prompt-text "..." \
  --text "你好，欢迎使用本系统。" \
  --output output.wav

# VAD (requires .[vad])
python examples/silerovad_real_inference.py \
  --audio path/to/test.wav \
  --threshold 0.5

# LLM (requires .[llm])
python examples/qwen3_llm_streaming.py \
  --model-path path/to/LLM_Model_ft \
  --base-prompt path/to/base_prompt.md \
  --text "你好，介绍一下你自己。"

# Audio2Face 3D (requires .[a2f])
python examples/nvidia_audio2face_3d_streaming.py \
  --model-dir path/to/audio2face-3d-v2.3.1-claire \
  --audio path/to/test.wav \
  --output arkit_weights.npy
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/models -q
```

