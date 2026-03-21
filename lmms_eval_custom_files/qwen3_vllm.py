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

# --- Constants ---
REFLEXIVE_IDS = [13824, 32771, 11780, 32263, 59401, 24077, 46093, 1549, 80922, 46619, 28188, 71196, 7196, 29215, 7200, 59936, 1575, 97836, 28722, 58936, 30267, 15934, 37442, 78917, 25157, 54853, 80456, 38478, 15954, 21076, 86103, 60503, 98904, 4695, 64092, 41569, 97381, 48750, 95342, 54390, 53882, 88190, 29316, 11908, 56456, 92812, 17037, 46223, 46224, 53905, 11409, 72335, 41109, 80022, 18071, 4248, 4764, 63646, 12446, 48288, 58017, 10915, 69796, 47781, 46761, 93361, 30385, 47288, 86205, 24765, 40129, 76995, 14019, 17094, 3783, 51911, 714, 34516, 14037, 2267, 28384, 37601, 11489, 56034, 29923, 56038, 82666, 56042, 96498, 1779, 98549, 49397, 30456, 68863, 3328, 4354, 73989, 72472, 64285, 6944, 68387, 81187, 18214, 5937, 35638, 22327, 10555, 99131, 80198, 18760, 69960, 13644, 20813, 44878, 49999, 66894, 99153, 26450, 10067, 10065, 25429, 90963, 13657, 86874, 32091, 69472, 10600, 86890, 14190, 92014, 32627, 71032, 52601, 12666, 78204, 7549, 7039, 27520, 81283, 88964, 3973, 9093, 22407, 2441, 62345, 41868, 46988, 21390, 3983, 89996, 93587, 85395, 1431, 8088, 36760, 43929, 10146, 98212, 16294, 29101, 64945, 72118, 97209, 1466, 13244, 13759, 1988, 41413, 31684, 53191, 10696, 11209, 10700, 87501, 55766, 62427, 8670, 48618, 41963, 2028, 13293, 16366, 13295, 1008, 66034, 75763, 82424, 57854, 52223]

# --- 1. Custom Logits Processor Implementation ---
class ThinkingLogitsProcessor(LogitsProcessor):
    """
    V1-Compatible Custom Logits Processor for Thinking Budget.
    Reads 'thinking_budget' and 'block_reflexive' from SamplingParams.extra_args.
    """
    
    @classmethod
    def validate_params(cls, params: SamplingParams):
        # Validate that if args are present, they are correct types
        if params.extra_args:
            budget = params.extra_args.get("thinking_budget")
            if budget is not None and not isinstance(budget, int):
                raise ValueError("thinking_budget must be an int")
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.device = device
        
        self.think_end_token_id = 151668 # </think>
        self.nl_token_id = 198         # \n
        
        self.reflexive_indices = torch.tensor(REFLEXIVE_IDS, device=device, dtype=torch.long)
        self.neg_inf = float("-inf")
        
        self.req_states: Dict[int, List[Any]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if not batch_update:
            return

        for index, params, _, output_tok_ids in batch_update.added:
            self.validate_params(params)
            budget = params.extra_args.get("thinking_budget") if params.extra_args else None
            
            if budget is not None:
                self.req_states[index] = [budget, output_tok_ids, False]
            else:
                self.req_states.pop(index, None)

        for index in batch_update.removed:
            self.req_states.pop(index, None)

        for from_idx, to_idx, direction in batch_update.moved:
            from_val = self.req_states.pop(from_idx, None)
            to_val = self.req_states.pop(to_idx, None)
            
            if from_val is not None:
                self.req_states[to_idx] = from_val
            
            if direction == MoveDirectionality.SWAP and to_val is not None:
                self.req_states[from_idx] = to_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        active_rows = []
        force_nl_rows = []
        force_end_rows = []
        boost_rows = []
        boost_vals = []

        for idx, (budget, output_ids, stopped) in self.req_states.items():
            if stopped:
                continue
            
            active_rows.append(idx)
            
            tokens_generated = len(output_ids)
            
            if tokens_generated == budget - 1:
                force_nl_rows.append(idx)
            elif tokens_generated >= budget:
                force_end_rows.append(idx)
                self.req_states[idx][2] = True 
            elif (tokens_generated / budget) > 0.95:
                boost_rows.append(idx)
                boost_vals.append(1.0 + (tokens_generated / budget))

        if active_rows:
            rows_tensor = torch.tensor(active_rows, device=self.device, dtype=torch.long)
            logits[rows_tensor.unsqueeze(1), self.reflexive_indices] = self.neg_inf

        if boost_rows:
            rows_tensor = torch.tensor(boost_rows, device=self.device, dtype=torch.long)
            vals_tensor = torch.tensor(boost_vals, device=self.device, dtype=torch.float32)
            
            logits[rows_tensor, self.nl_token_id] *= vals_tensor
            logits[rows_tensor, self.think_end_token_id] *= vals_tensor

        if force_nl_rows:
            rows_tensor = torch.tensor(force_nl_rows, device=self.device, dtype=torch.long)
            logits[rows_tensor, :] = self.neg_inf
            logits[rows_tensor, self.nl_token_id] = 0.0

        if force_end_rows:
            rows_tensor = torch.tensor(force_end_rows, device=self.device, dtype=torch.long)
            logits[rows_tensor, :] = self.neg_inf
            logits[rows_tensor, self.think_end_token_id] = 0.0

        return logits


@register_model("qwen3_vllm")
class Qwen3_VLLM(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-14B",
        batch_size: Optional[Union[int, str]] = 1,
        stage_1_path: Optional[str] = None,
        conv_template: Optional[str] = "qwen_1_5",
        **kwargs,
    ) -> None:
        super().__init__()

        # Accelerator Setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            kwargs["distributed_executor_backend"] = "external_launcher"
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        perception_model_size, model_path, enable_thinking = pretrained.split(';')
        
        self.enable_thinking = enable_thinking.lower() == "true"

        self.non_thinking_budget = 1024
        self.thinking_budget = 4096
        self.block_reflexive_tokens = True
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self._model = LLM(
            model=model_path, 
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(), 
            gpu_memory_utilization=0.9,
            logits_processors=[ThinkingLogitsProcessor],
            **kwargs 
        )
        
        self.conv_template = conv_template
        self.batch_size_per_gpu = int(batch_size)
        
        if stage_1_path is not None:
            self.stage_1_results_cache = {}
            with open(stage_1_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.stage_1_results_cache[str(data["doc_id"])] = data["filtered_resps"][0]
        else:
            self.stage_1_results_cache = None

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

    def build_prism_stage_2_prompt(self, question, description, task):
        if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
            question = 'Question: ' + question 
        des = description
        if not des.endswith('\n'):
            des += '\n'
        description = 'Description: ' + des
        role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'
        
        if not self.enable_thinking:
            if 'mmstar' in task or 'vmcbench' in task or 'coin_nights' in task: 
                post_prompt = '\nAnswer directly with the option\'s letter in the format of "Answer:". Do not add anything other than the letter answer after "Answer:".'
            else:
                assert False, "Task not supported"
        else:
            if 'mmstar' in task or 'vmcbench' in task or 'coin_nights' in task:
                post_prompt = '\nPlease reason step by step, and give the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the letter answer after "Answer:".'
            else:
                assert False, "Task not supported" 
                
        prompt = role + description + question + post_prompt
        return prompt

    def generate_until(self, requests) -> List[str]:
        res = []
        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        pbar = tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")

        for batch_requests in batched_requests:
            prompts = []
            
            sampling_params_list = []
            
            for idx, req in enumerate(batch_requests):
                contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
                question = contexts
                if self.stage_1_results_cache is not None:
                    description = self.stage_1_results_cache[str(doc_id)]
                    question = self.build_prism_stage_2_prompt(question, description, task)
                
                question_input = [{"role": "user", "content": question}]
                try:
                    text = self.tokenizer.apply_chat_template(
                        question_input, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
                    )
                except TypeError:
                    text = self.tokenizer.apply_chat_template(
                        question_input, tokenize=False, add_generation_prompt=True
                    )
                
                prompts.append(text)
                
            if self.enable_thinking:
                sp = SamplingParams(
                    max_tokens=self.thinking_budget + self.non_thinking_budget,
                    temperature=0.6,
                    top_p=0.95,
                    extra_args={
                        "thinking_budget": self.thinking_budget
                    }
                )
            else:
                sp = SamplingParams(
                    max_tokens=self.non_thinking_budget,
                    temperature=0.7,
                    top_p=0.8
                )

            outputs = self._model.generate(prompts, sp, use_tqdm=False)
            for output in outputs:
                res.append(output.outputs[0].text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented")