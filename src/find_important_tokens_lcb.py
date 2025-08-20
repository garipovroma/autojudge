import argparse
from typing import Union, Sequence, Tuple
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
import time
from tqdm.auto import tqdm
from termcolor import colored
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import transformers

import itertools
import logging

from utils import color_replaced_tokens
from livecodebench_v5 import load_code_generation_dataset, extract_code, apply_llama_lcb_prompt_format, codegen_metrics, unpack_lcb_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def verify_args(args):
    assert args.process_id < args.world_size, "--process_id must be < --world_size"


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")

    parser.add_argument('--draft_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Path or identifier of the draft model.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='float32',
                        help='Data type for torch tensors.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--world_size', type=int, default=1,
                        help='world size')
    parser.add_argument('--process_id', type=int, default=0,
                        help='Process ID')
    parser.add_argument('--dump_freq', type=int, default=64,
                        help='Dump frequency.')
    parser.add_argument('--write_index_offset', type=int, default=0,
                        help='Write index offset.')
    parser.add_argument('--keep_responses', type=bool, default=False,
                        help='Keep all responses in find_important_tokens algorithm output.')
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--num_process_evaluate', type=int, default=16)
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=8)
    parser.add_argument('--lcb_path', type=str, default='none')

    args = parser.parse_args()
    verify_args(args)

    return args

def test_program(program_tokens, eval_samples, tokenizer):
    code = extract_code(tokenizer.batch_decode(program_tokens)[0])
    result = codegen_metrics([eval_samples], [[code]], num_process_evaluate=args.num_process_evaluate)
    score = int(result[0]['pass@1'])
    return dict(score=score, code=code, gen=tokenizer.batch_decode(program_tokens)[0], result=result)

@torch.no_grad()
def find_important_tokens(
        prompt: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizer,
        draft_model: transformers.AutoModelForCausalLM,
        target_model: transformers.AutoModelForCausalLM,
        eval_samples: list,
        sample: dict
):
    batch_size, prompt_max_length = prompt.shape
    assert batch_size == 1

    device = prompt.device

    batch_dict = {
        'input_ids': prompt,
        'attention_mask': torch.ones_like(prompt),
    }

    target_response = target_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_length=prompt_max_length + args.max_new_tokens
    )

    draft_response = draft_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_length=prompt_max_length + args.max_new_tokens
    )

    target_score = test_program(target_response, eval_samples, tokenizer)['score']
    draft_score = test_program(draft_response, eval_samples, tokenizer)['score']

    logger.info(f'{target_score=}, {draft_score=}, difficulty={sample.difficulty}')

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = prompt_max_length

    responses = [(current_response.cpu().clone())]

    for iteration in itertools.count():
        draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device),
                                         draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(
                                             current_response)).logits.argmax(-1)[:, :-1]], dim=1)

        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted(
            [idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= left_border])
        if not mismatch_indices:
            break

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, prompt_max_length,
                                               changed_token_indices, tokenizer)

        mismatch_index = mismatch_indices[0]

        prefix_with_draft_token = torch.cat([
            current_response[:, :mismatch_index],
            torch.tensor([[draft_argmax_tokens[:, mismatch_index]]], device=device)
        ], dim=1)

        alternative_batch = {
            'input_ids': prefix_with_draft_token,
            'attention_mask': torch.ones_like(prefix_with_draft_token),
        }
        alternative_response = target_model.generate(
            **alternative_batch,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_length=prompt_max_length + args.max_new_tokens,
        )
        alternative_score = test_program(alternative_response, eval_samples, tokenizer)['score']
        if target_score == alternative_score:
            changed_token_indices.append((mismatch_index, False, current_response[0, mismatch_index].item(),
                                          alternative_response[0, mismatch_index].item()))
            current_response = alternative_response
        else:
            changed_token_indices.append((mismatch_index, True, current_response[0, mismatch_index].item(),
                                          alternative_response[0, mismatch_index].item()))
        left_border = mismatch_index + 1
        responses.append((current_response.cpu().clone(), alternative_score))

    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens, draft_score=draft_score,
                target_score=target_score, current_response=current_response, responses=responses)

if __name__ == '__main__':
    args = get_args()
    print(f'The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    np.random.seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device, low_cpu_mem_usage=True)
    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='left')
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = load_code_generation_dataset(args)

    livecodebench_v5_dataset, livecodebench_v5_inputs_outputs = unpack_lcb_data(dataset, tokenizer)

    n_samples = args.n_tasks
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(f'{n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_id}.pt')
    if os.path.exists(output_file_path):
        mined_data = torch.load(output_file_path)
        logger.info(f'Loaded {len(mined_data)} important tokens from {output_file_path}, starting from {shard_start + len(mined_data)}-th sample')
    else:   
        mined_data = []
        logger.info(f'No important tokens found in {output_file_path}, will generate them')

    mined_data = []
    sample_idx = 0
    with tqdm(total=shard_end - shard_start) as pbar:
        pbar.update(len(mined_data))
        for sample_idx in range(shard_start + len(mined_data), shard_end):            
            task = dataset[sample_idx]
            task_dict = livecodebench_v5_dataset[sample_idx]
            eval_samples = livecodebench_v5_inputs_outputs[sample_idx]
            batch = tokenizer(task_dict['prompt'], add_special_tokens=False, return_tensors='pt').to(device)

            mined_dict = find_important_tokens(batch['input_ids'], tokenizer, draft_model, target_model, eval_samples, dataset[sample_idx])

            del task_dict['tests']
            mined_dict.update(task_dict)
            mined_data.append(mined_dict)

            if ((sample_idx % args.dump_freq == (args.dump_freq - 1)) or (sample_idx == shard_end - 1)):
                torch.save(mined_data, output_file_path)
            pbar.update(1)
            pbar.set_description(f'Process {args.process_id}')

    done_file = os.path.join(args.output_folder, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")
    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == args.process_saver_id and args.world_size > 1:
        logger.info(f"Process {args.process_saver_id} is waiting for all other processes to finish...")
        while True:
            done_files = [f"done_{i}.txt" for i in range(args.process_saver_id + 1, args.process_saver_id + args.total_gpus) if i * shard_len < shard_end]
            all_done = all(os.path.exists(os.path.join(args.output_folder, f)) for f in done_files)

            if all_done:
                logger.info("All processes finished. Concatenating important tokens data...")
                all_data = []
                for i in range(args.process_saver_id, args.process_saver_id + args.total_gpus):
                    if i * shard_len >= shard_end:
                        continue
                    process_output_file = os.path.join(args.output_folder, f"{args.output_file}_{i}.pt")
                    all_data.extend(torch.load(process_output_file))
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    done_file = os.path.join(args.output_folder, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                final_output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_saver_id}' + '.pt')
                logger.info(f"Saving important tokens data to {final_output_file_path}...")
                torch.save(all_data, final_output_file_path)
                logger.info("Done.")
                break

            time.sleep(5)