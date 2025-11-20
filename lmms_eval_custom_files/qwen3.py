from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union, List, Tuple
import torch
import json
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from llava.conversation import SeparatorStyle, conv_templates
from lmms_eval.api.instance import Instance

import re

# custom logits processor for budget forcing and no reflexive tokens

reflexive_token_ids = [13824, 32771, 11780, 32263, 59401, 24077, 46093, 1549, 80922, 46619, 28188, 71196, 7196, 29215, 7200, 59936, 1575, 97836, 28722, 58936, 30267, 15934, 37442, 78917, 25157, 54853, 80456, 38478, 15954, 21076, 86103, 60503, 98904, 4695, 64092, 41569, 97381, 48750, 95342, 54390, 53882, 88190, 29316, 11908, 56456, 92812, 17037, 46223, 46224, 53905, 11409, 72335, 41109, 80022, 18071, 4248, 4764, 63646, 12446, 48288, 58017, 10915, 69796, 47781, 46761, 93361, 30385, 47288, 86205, 24765, 40129, 76995, 14019, 17094, 3783, 51911, 714, 34516, 14037, 2267, 28384, 37601, 11489, 56034, 29923, 56038, 82666, 56042, 96498, 1779, 98549, 49397, 30456, 68863, 3328, 4354, 73989, 72472, 64285, 6944, 68387, 81187, 18214, 5937, 35638, 22327, 10555, 99131, 80198, 18760, 69960, 13644, 20813, 44878, 49999, 66894, 99153, 26450, 10067, 10065, 25429, 90963, 13657, 86874, 32091, 69472, 10600, 86890, 14190, 92014, 32627, 71032, 52601, 12666, 78204, 7549, 7039, 27520, 81283, 88964, 3973, 9093, 22407, 2441, 62345, 41868, 46988, 21390, 3983, 89996, 93587, 85395, 1431, 8088, 36760, 43929, 10146, 98212, 16294, 29101, 64945, 72118, 97209, 1466, 13244, 13759, 1988, 41413, 31684, 53191, 10696, 11209, 10700, 87501, 55766, 62427, 8670, 48618, 41963, 2028, 13293, 16366, 13295, 1008, 66034, 75763, 82424, 57854, 52223]

from transformers import LogitsProcessor
# based on https://muellerzr.github.io/til/end_thinking.html
class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, thinking_budget, block_reflexive_tokens=False):
        self.thinking_budget = thinking_budget
        self.think_end_token =tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float('-inf')

        self.block_reflexive_tokens = block_reflexive_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1
        if self.thinking_budget is not None and not self.stopped_thinking:
            if (self.tokens_generated / self.thinking_budget) > .95:
                scores[0][self.nl_token] = scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.thinking_budget))
                scores[0][self.think_end_token] = (
                    scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.thinking_budget))
                )

            if self.tokens_generated >= (self.thinking_budget - 1):
                if self.tokens_generated == self.thinking_budget-1:
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True
        
        if self.block_reflexive_tokens:
            scores[:, reflexive_token_ids] = self.neg_inf

        return scores

@register_model("qwen3")
class Qwen3(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-14B",
        batch_size: Optional[Union[int, str]] = 1,
        stage_1_path: Optional[str] = None,
        conv_template: Optional[str] = "qwen_1_5",
        **kwargs,
    ) -> None:
        super().__init__()

        perception_model_size, pretrained, enable_thinking = pretrained.split(';')

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map="auto"
        )
        self.enable_thinking = enable_thinking.lower() == "true"

        self.non_thinking_budget = 1024
        self.thinking_budget = 4096
        self.block_reflexive_tokens = True

        self.conv_template = conv_template
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1 # TODO: Support batch size > 1

        self._rank = 0
        self._world_size = 1

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
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
        
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def build_prism_stage_2_prompt(self, question, description, task):
        if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
            question = 'Question: ' + question 
        des = description
        if not des.endswith('\n'):
            des += '\n'
        description = 'Description: ' + des
        role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'
        if not self.enable_thinking:
            if 'mmstar' in task or 'vmcbench' in task or 'coin_nights' in task: # multiple choice
                post_prompt = '\nAnswer directly with the option\'s letter in the format of "Answer:". Do not add anything other than the letter answer after "Answer:".'
            else:
                assert False, "Task not supported"
        else:
            if 'mmstar' in task or 'vmcbench' in task or 'coin_nights' in task:
                post_prompt = '\nPlease reason step by step, and give the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the letter answer after "Answer:".'
            else:
                assert False, "Task not supported"
        prompt =  role + description + question + post_prompt

        return prompt

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            visual = None
            task_type = "text"
            placeholder_count = 0
            image_tensor = None

            question = contexts

            if self.stage_1_results_cache is not None:
                description = self.stage_1_results_cache[str(doc_id)]
                question = self.build_prism_stage_2_prompt(question, description, task)

            question_input = [
                    {"role": "user", "content": question}
                ]

            text = self.tokenizer.apply_chat_template(
                question_input,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            if self.enable_thinking:
                logits_processor = CustomLogitsProcessor(self.tokenizer, self.thinking_budget, block_reflexive_tokens=self.block_reflexive_tokens)
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.thinking_budget + self.non_thinking_budget,
                    temperature=0.6,
                    top_p=0.95,
                    logits_processor=[logits_processor],
                )
            else:
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.non_thinking_budget,
                    temperature=0.7,
                    top_p=0.8,
                )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            res.append(content)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood")