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
pip install "respeak[vad]"
pip install "respeak[tts]"
pip install "respeak[llm]"
pip install "respeak[dev]"
```

## Usage

```python
from respeak import StreamingParaformerAsr, CosyVoice3Tts, SileroVad, Qwen3LLM

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
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/models -q
```

## Publish to PyPI

CI runs on every push/PR to `main`. Publishing is triggered by pushing a version tag.

### One-time PyPI setup

1. Create a project on [PyPI](https://pypi.org/) named `respeak`.
2. In PyPI → **Publishing** → **Add a new pending publisher**:
   - Owner: `shanguanma`
   - Repository: `respeak`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`

### Release flow

```bash
# 1. Bump version in pyproject.toml
# 2. Commit and tag
git tag v0.1.1
git push origin main
git push origin v0.1.1
```

The tag must match `pyproject.toml` (`v0.1.1` ↔ `version = "0.1.1"`).
