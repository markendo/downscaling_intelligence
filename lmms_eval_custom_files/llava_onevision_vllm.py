import os
import json
import torch
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm

from accelerate import Accelerator
from transformers import AutoTokenizer

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality
)

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval import utils

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import base64
from accelerate import DistributedType

import logging

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN

NUM_SECONDS_TO_SLEEP = int(os.getenv("NUM_SECONDS_TO_SLEEP", "5"))
WORKERS = int(os.getenv("WORKERS", "32"))

# --- 2. Model Class ---
@register_model("llava_onevision_vllm")
class Llava_OneVision_VLLM(lmms):
    def __init__(
        self,
        model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        batch_size: Optional[Union[int, str]] = 1,
        conv_template: Optional[str] = "qwen_1_5",
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__()

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Accelerator Setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        if data_parallel_size > 1:
            assert tensor_parallel_size == 1, "Data parallelism is not supported with tensor parallelism. For current vllm version"
        if accelerator.num_processes > 1:
            kwargs["distributed_executor_backend"] = "external_launcher"


        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # --- INITIALIZE VLLM WITH CUSTOM PROCESSOR ---
        # Method 3: Pass Class Object to constructor

        self._model = LLM(
            model=model_path, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=1,
            **kwargs,
        )

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template

    @property
    def config(self):
        return self._model.llm_engine.model_config.hf_config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input):
        new_list = []
        for i in input:
            if isinstance(i, (list, tuple)):
                new_list.extend(i)
            else:
                new_list.append(i)
        return new_list
    
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy().convert("RGB")

        return img

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "temperature": gen_kwargs["temperature"],
                    "top_p": gen_kwargs["top_p"],
                }
                sampling_params = SamplingParams(**params)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []  # multiple images or frames for video
                    all_tasks = []
                    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                        for visual in visuals:
                            if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                                all_tasks.append(executor.submit(self.encode_video, visual))
                            elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                                all_tasks.append(executor.submit(self.encode_image, visual))
                            elif isinstance(visual, Image.Image):
                                all_tasks.append(executor.submit(self.encode_image, visual))

                        for task in all_tasks:
                            imgs.append(task.result())
                
                if len(imgs) > 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                    image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(imgs))
                    question = image_tokens + "\n" + contexts
                else:
                    question = contexts

                conv = conv_templates[self.conv_template].copy()
                if utils.is_json(question):
                    data = json.loads(question)
                    for item in data:
                        conv.append_message(conv.roles[0] if item['from']=='human' else conv.roles[1], item['value'])
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                
                final_prompt = conv.get_prompt()

                input_item = {
                    "prompt": final_prompt,
                    "multi_modal_data": {}
                }
                
                if len(imgs) > 0:
                    input_item["multi_modal_data"]["image"] = imgs

                batched_messages.append(input_item)

            sampling_params = SamplingParams(**params)

            response = self._model.generate(batched_messages, sampling_params=sampling_params, use_tqdm=False)
            response_text = [o.outputs[0].text for o in response]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented")