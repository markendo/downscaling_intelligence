import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse
import os

from PIL import Image
import ast
import re

# we downloaded images from https://github.com/zackschen/CoIN
IMAGE_FOLDER = '/path/to/image/folder'

si_paths = {'ScienceQA': '/path/to/scienceqa/train.json',
            'TextVQA': '/path/to/textvqa/train.json',
            'GQA': '/path/to/gqa/train.json',
            'VizWiz': '/path/to/vizwiz/train.json',
            'VQAv2': '/path/to/vqa_v2/train.json',
            'OCRVQA': '/path/to/ocrvqa/train.json',
            }

save_dir = '/path/to/save/directory/'

def load_existing_jsonl(save_path):
    num_processed = 0

    if not os.path.exists(save_path):
        return num_processed

    with open(save_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                num_processed += 1
            except json.JSONDecodeError:
                print("Found corrupted or partial line")
        
    return num_processed

# issue with Qwen3 batching where it tries to combine batched inputs as a single input
def process_inputs(inputs, device, processor):
    def left_pad_sequence(sequences, padding_value, batch_first=True):
        # sequences: list of 1D tensors
        reversed_seq = [torch.flip(x, dims=[0]) for x in sequences]  # reverse each sequence
        padded = torch.nn.utils.rnn.pad_sequence(reversed_seq, batch_first=batch_first, padding_value=padding_value)
        return torch.flip(padded, dims=[1] if batch_first else [0])  # flip back

    input_ids_list = [x["input_ids"].squeeze(0) for x in inputs]
    attention_mask_list = [x["attention_mask"].squeeze(0) for x in inputs]

    batch_input_ids = left_pad_sequence(input_ids_list, padding_value=processor.tokenizer.pad_token_id).to(device)
    batch_attention_mask = left_pad_sequence(attention_mask_list, padding_value=0).to(device)
    
    if "pixel_values" not in inputs[0]:
        return {'input_ids': batch_input_ids,
              'attention_mask': batch_attention_mask}

    batch_pixel_values = torch.stack([x["pixel_values"].squeeze(0) for x in inputs]).to(device)
    batch_image_grid_thw = torch.stack([x["image_grid_thw"].squeeze(0) for x in inputs]).to(device)

    return {'input_ids': batch_input_ids,
              'attention_mask': batch_attention_mask,
              'pixel_values': batch_pixel_values,
              'image_grid_thw': batch_image_grid_thw}

def build_qa_statement_prompt(conversations):
    task_prompt = "Your task is to convert each question–answer pair about an image into a concise, fully self-contained declarative statement. The resulting statements should be understandable on their own, without requiring the reader to refer to the original question."
    for i in range(0, len(conversations), 2):
        task_prompt += f"\nQuestion #{int(i/2)+1}: {conversations[i]['value']}"
        task_prompt += f"\nAnswer: {conversations[i+1]['value']}"
    if len(conversations) != 2:
        task_prompt += f"\nAs there are {int(len(conversations) / 2)} questions, you should respond with {int(len(conversations) / 2)} statements. Include each statement on its own line"
    task_prompt += f"\nDeclarative Statement(s):"
    return task_prompt


def main(dataset_name, start_index, end_index, batch_size):
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    with open(si_paths[dataset_name], 'r') as f:
        data = json.load(f)
    
    if end_index > len(data):
        end_index = len(data)
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    # os.makedirs(dataset_save_dir, exist_ok=True)
    save_path = os.path.join(dataset_save_dir, f"{start_index}-{end_index}_captions.jsonl")

    num_processed = load_existing_jsonl(save_path)
    print(f"Resuming from {num_processed} saved examples")

    for i in tqdm(range(start_index + num_processed, end_index, batch_size)):

        if i+batch_size > end_index:
            batch = data[i:end_index]
        else:
            batch = data[i:i+batch_size]
        
        new_batch = []
        batch_indices = [None] * len(batch)
        for ex_idx, ex in enumerate(batch):
            if 'image' in ex:
                new_batch.append(ex)
                batch_indices[ex_idx] = len(new_batch) - 1
        batch = new_batch


        if len(batch) == 0:
            entries = []
            for ex_idx, batch_idx in enumerate(batch_indices):
                entries.append({"index": ex_idx + i, "caption": 'SKIP'})
            with open(save_path, 'a') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
            continue

        prompts = [build_qa_statement_prompt(ex['conversations']) for ex in batch]

        processed_inputs = []
        for prompt in prompts: 
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            input_tensor = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=False  # pad manually later
            )
            processed_inputs.append(input_tensor)
        inputs = process_inputs(processed_inputs, model.device, processor)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        batch_statements = output_text

        processed_inputs = []
        batch_images = [Image.open(os.path.join(IMAGE_FOLDER, ex['image'])).convert("RGB").resize((512,512)) for ex in batch]
        for ex_idx, statements in enumerate(batch_statements):
            prompt = "Your task is to describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present."
            prompt += f"\nAs part of your description, you should incorporate the following information about the image.\n"
            prompt += statements
            prompt += "\nDescription:"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": batch_images[ex_idx],
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            input_tensor = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=False  # pad manually later
            )
            processed_inputs.append(input_tensor)
        inputs = process_inputs(processed_inputs, model.device, processor)

        generated_ids = model.generate(**inputs,max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        entries = []

        for ex_idx, batch_idx in enumerate(batch_indices):
            if batch_idx is not None:
                entries.append({"index": ex_idx + i, "caption": output_text[batch_idx]})
            else:
                entries.append({"index": ex_idx + i, "caption": 'SKIP'})

        with open(save_path, 'a') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    main(args.dataset_name, args.start_index, args.end_index, args.batch_size)