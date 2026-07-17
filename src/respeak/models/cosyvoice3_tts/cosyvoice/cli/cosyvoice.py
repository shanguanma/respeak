# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Slimmed for CosyVoice3 + vLLM inference only.
import os
import time
from typing import Generator

import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.utils.file_utils import logging

# Keys present in stock Fun-CosyVoice3 yaml that are only needed for training.
_INFER_YAML_NULLS = {
    "hifigan": None,
    "mel_spec_transform1": None,
    "parquet_opener": None,
    "tokenize": None,
    "filter": None,
    "resample": None,
    "truncate": None,
    "compute_fbank": None,
    "compute_f0": None,
    "parse_embedding": None,
    "shuffle": None,
    "sort": None,
    "batch": None,
    "padding": None,
    "data_pipeline": None,
    "data_pipeline_gan": None,
    "train_conf": None,
    "train_conf_gan": None,
}


class CosyVoice3:
    """CosyVoice3 runtime (zero-shot TTS, optional vLLM LLM decode)."""

    def __init__(self, model_dir, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
            self.model_dir = model_dir

        hyper_yaml_path = "{}/cosyvoice3.yaml".format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError("{} not found!".format(hyper_yaml_path))

        overrides = {
            "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
            **_INFER_YAML_NULLS,
        }
        with open(hyper_yaml_path, "r") as f:
            configs = load_hyperpyyaml(f, overrides=overrides)
        assert get_model_type(configs) == CosyVoice3Model, (
            "do not use {} for CosyVoice3 initialization!".format(model_dir)
        )

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v3.onnx".format(model_dir),
            "{}/spk2info.pt".format(model_dir),
            configs["allowed_special"],
        )
        self.sample_rate = configs["sample_rate"]

        if torch.cuda.is_available() is False and (load_trt is True or load_vllm is True or fp16 is True):
            load_trt, load_vllm, fp16 = False, False, False
            self.fp16 = False
            logging.warning("no cuda device, set load_trt/load_vllm/fp16 to False")

        self.model = CosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16)
        self.model.load(
            "{}/llm.pt".format(model_dir),
            "{}/flow.pt".format(model_dir),
            "{}/hift.pt".format(model_dir),
        )
        if load_vllm:
            self.model.load_vllm("{}/vllm".format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning("DiT tensorRT fp16 engine have some performance issue, use at caution!")
            self.model.load_trt(
                "{}/flow.decoder.estimator.{}.mygpu.plan".format(
                    model_dir, "fp16" if self.fp16 is True else "fp32"
                ),
                "{}/flow.decoder.estimator.fp32.onnx".format(model_dir),
                trt_concurrent,
                self.fp16,
            )
        del configs

    def list_available_spks(self):
        return list(self.frontend.spk2info.keys())

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        assert zero_shot_spk_id != "", "do not use empty zero_shot_spk_id"
        model_input = self.frontend.frontend_zero_shot(
            "", prompt_text, prompt_wav, self.sample_rate, ""
        )
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, "{}/spk2info.pt".format(self.model_dir))

    def inference_zero_shot(
        self,
        tts_text,
        prompt_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend
        )
        for i in tqdm(
            self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)
        ):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning(
                    "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
                        i, prompt_text
                    )
                )
            model_input = self.frontend.frontend_zero_shot(
                i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            start_time = time.time()
            logging.info("synthesis text {}".format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(
                    "yield speech len {}, rtf {}".format(
                        speech_len, (time.time() - start_time) / speech_len
                    )
                )
                yield model_output
                start_time = time.time()


def AutoModel(**kwargs):
    model_dir = kwargs.get("model_dir")
    if model_dir is None:
        raise TypeError("model_dir is required")
    if not os.path.exists(model_dir):
        kwargs["model_dir"] = snapshot_download(model_dir)
        model_dir = kwargs["model_dir"]
    if os.path.exists("{}/cosyvoice3.yaml".format(model_dir)):
        return CosyVoice3(**kwargs)
    raise TypeError("No CosyVoice3 model found under {} (missing cosyvoice3.yaml)".format(model_dir))
