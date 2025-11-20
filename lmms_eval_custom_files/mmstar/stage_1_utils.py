import json

prompt_prefix = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present. ' + 'Especially, pay attention to '

suffix_prompt_path = '../lmms_eval_custom_files/mmstar/question_specific_info_qwen3_8B.json'
with open(suffix_prompt_path, 'r') as f:
    suffix_prompts = json.load(f)

def mmstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_index = str(doc['index'])
    suffix_prompt = suffix_prompts[question_index]
    question = prompt_prefix + suffix_prompt
    if not question.endswith('.'):
        question += '.'
    return question