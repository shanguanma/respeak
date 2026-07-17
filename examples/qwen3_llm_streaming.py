"""Streaming Qwen3LLM example.

Example:
    python examples/qwen3_llm_streaming.py \
      --model-path ../model/LLM_Model_ft \
      --base-prompt ../configs/base_prompt.md \
      --persona "你叫小睿，是一个温和的中文助手。" \
      --strategy "回答要简洁自然。" \
      --text "你好，介绍一下你自己。"
"""

from __future__ import annotations

import argparse

from respeak import Qwen3LLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3LLM sentence-level streaming demo")
    parser.add_argument("--model-path", required=True, help="Local model path or ModelScope id")
    parser.add_argument("--text", required=True, help="User utterance")
    parser.add_argument("--base-prompt", help="Base system prompt markdown file")
    parser.add_argument("--persona", default="", help="Persona setting appended to base prompt")
    parser.add_argument("--strategy", default="", help="Reply strategy appended to base prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    args = parser.parse_args()

    llm = Qwen3LLM.from_pretrained(
        args.model_path,
        base_prompt_file=args.base_prompt,
        persona_setting=args.persona,
        strategy_prompt=args.strategy,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    print("[LLM] Start generating response...")
    for sentence in llm.generate(args.text, stream=True):
        print(sentence, flush=True)


if __name__ == "__main__":
    main()
