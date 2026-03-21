
"""
Convert LLaVA-Onevision-style model with Qwen3 backbone to HF version. 
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/convert_llava_onevision_weights_to_hf.py
"""

import argparse
import gc
import glob
import json
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionImageProcessor,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
    SiglipVisionConfig,
)

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.vision_tower": "model.vision_tower",
    "model.mm_projector.0": "model.multi_modal_projector.linear_1",
    "model.mm_projector.2": "model.multi_modal_projector.linear_2",
    "model.layers": "model.language_model.layers",
    "model.embed_tokens": "model.language_model.embed_tokens",
    "model.norm": "model.language_model.norm",
    "model.image_newline": "model.image_newline",
}

chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # if "lm_head.weight" not in original_state_dict:
    #     original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict

def convert_llava_to_hf(model_id, text_model_id, pytorch_dump_folder_path):
    # load original config
    filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    vision_model_id = data["mm_vision_tower"]
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=True)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    video_token_index = tokenizer.convert_tokens_to_ids("<video>")

    image_processor = LlavaOnevisionImageProcessor.from_pretrained(vision_model_id)
    video_processor = LlavaOnevisionVideoProcessor.from_pretrained(vision_model_id)
    processor = LlavaOnevisionProcessor(
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_processor=image_processor,
        num_image_tokens=729,
        vision_feature_select_strategy="full",
        chat_template=chat_template,
    )

    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        image_size=384,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=26,  # drop the last layer
        patch_size=14,
        vision_use_head=False,  # no head
    ).to_dict()

    config = LlavaOnevisionConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        use_image_newline_parameter=True,
        tie_word_embeddings=text_config.tie_word_embeddings,
        image_token_index=image_token_index,
        video_token_index=video_token_index,
    )

    with init_empty_weights():
        model = LlavaOnevisionForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True, strict=False)
    model.tie_weights()
    model.eval()

    pre_expansion_embeddings = model.language_model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    # Qwen-based models have extra unused space in the vocab size already, so no need to resize
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    num_tokens = vocab_size + 2
    model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
    model.language_model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )

    print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        choices=[
            "markendo/llava-extract-qwen3-0.6B",
            "markendo/llava-extract-qwen3-1.7B",
        ],
        required=True,
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the base text model used",
        choices=[
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
        ],
        required=True,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=False, default="/pasteur/u/markendo/perception-alignment-mllm-summer2025/converted_llava_onevision_qwen3_final", help="Path to the output PyTorch model directory."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.text_model_id, args.pytorch_dump_folder_path)