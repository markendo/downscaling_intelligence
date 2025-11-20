import datetime
import json
import os
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmstar.utils import mmstar_process_results as mmstar_process_results_original
import re

dir_name = os.path.dirname(os.path.abspath(__file__))

replace_prompt = " Please answer yes or no."

def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    
    return question


def mmstar_process_results(doc, results):
    ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
    for i in range(len(results)):
        result = results[i]
        result = result.split("\n")[-1]
        match = re.search(ANSWER_PATTERN, result)
        results[i] = match.group(1) if match else result

    return mmstar_process_results_original(doc, results)