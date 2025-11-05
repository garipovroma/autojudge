import argparse
from typing import Union, Sequence, Tuple
import json
import os
import re

import time
from tqdm.auto import tqdm
from termcolor import colored
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import transformers
import itertools
import logging

from time import perf_counter

from core_utils import color_replaced_tokens
from mmlu_pro_utils import extract_answer

import sys
sys.path.append('.')

from lm_eval_utils import stop_sequences_criteria

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

LAST_DUMP = perf_counter()
DUMP_FREQ_SECS = 20 * 60  # mins


@torch.no_grad()
def find_important_tokens(
        prompt: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizer,
        draft_model: transformers.AutoModelForCausalLM,
        target_model: transformers.AutoModelForCausalLM,
):
    global LAST_DUMP, DUMP_FREQ_SECS

    batch_size, n_input_tokens = prompt.shape
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
        max_new_tokens=args.max_new_tokens,
        stop_strings=["Question:"],
        tokenizer=tokenizer
    )

    stopping_criteria = stop_sequences_criteria(tokenizer, ["Question:"], initial_decoder_input_length=n_input_tokens, batch_size=batch_size)
    draft_response = draft_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=args.max_new_tokens,
        stop_strings=["Question:"],
        tokenizer=tokenizer
    )

    target_gen_str = tokenizer.batch_decode(target_response[:, n_input_tokens:])[0]
    target_ans = extract_answer(target_gen_str)

    draft_gen_str = tokenizer.batch_decode(draft_response[:, n_input_tokens:])[0]
    draft_ans = extract_answer(draft_gen_str)
    logger.info(f'{target_ans=}, {draft_ans=}')

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = n_input_tokens

    responses = [current_response.cpu().clone()]

    for iteration in itertools.count():
        draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device),
                                         draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(
                                             current_response)).logits.argmax(-1)[:, :-1]], dim=1)

        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted(
            [idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= left_border])
        if not mismatch_indices:
            break

        logger.info(f'Iteration {iteration}: {len(mismatch_indices)} mismatches')
        logger.info(f'\t{mismatch_indices[:20]}...')

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, n_input_tokens, changed_token_indices,
                                               tokenizer)

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
            max_new_tokens=max(1, args.max_new_tokens - (alternative_batch['input_ids'].shape[1] - n_input_tokens)),
            stop_strings=["Question:"],
            tokenizer=tokenizer
        )

        if tokenizer.eos_token_id in alternative_response[:, n_input_tokens:mismatch_index + 1]:
            eos_pos = (alternative_response[:, n_input_tokens:mismatch_index + 1] == tokenizer.eos_token_id).nonzero(as_tuple=True)[1][-1].item() + n_input_tokens
            logger.info(f'\t\t\t EOS found at {eos_pos=}')
            alternative_response = alternative_response[:, :eos_pos + 1]

        alternative_gen_str = tokenizer.batch_decode(alternative_response[:, n_input_tokens:])[0]
        alt_ans = extract_answer(alternative_gen_str)
        answers_match = alt_ans == target_ans

        logger.info(f'\t{target_ans=}, {alt_ans=}, {answers_match=}')

        if answers_match:
            changed_token_indices.append((mismatch_index, False, current_response[0, mismatch_index].item(),
                                          alternative_response[0, mismatch_index].item()))
            current_response = alternative_response
        else:
            changed_token_indices.append((mismatch_index, True, current_response[0, mismatch_index].item(),
                                          alternative_response[0, mismatch_index].item()))
        left_border = mismatch_index + 1
        responses.append(current_response.cpu().clone())

        if tokenizer.eos_token_id in alternative_response[0, n_input_tokens:mismatch_index + 1].cpu().tolist():
            logger.info(f'\t\t\t EOS found in alternative_response, terminating')
            logger.info(f'\t\t\t alternative_response prefix: {tokenizer.decode(alternative_response[:, n_input_tokens:n_input_tokens + 16][0])}')
            logger.info(f'\t\t\t alternative_response suffix: {tokenizer.decode(alternative_response[:, -16:][0])}')
            break

    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens, draft_answer=draft_ans,
                target_answer=target_ans, current_response=current_response, responses=responses)


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
    parser.add_argument('--mmlu_pro_train_path', type=str,
                        help='Path to the mmlu_pro train set file.')
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
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=8)

    args = parser.parse_args()
    verify_args(args)

    return args


def load_questions(args):
    data = torch.load(args.mmlu_pro_train_path)
    return data


if __name__ == "__main__":
    args = get_args()
    print('The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    if args.torch_dtype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info(f'{torch.backends.cuda.matmul.allow_tf32=}, {torch.backends.cudnn.allow_tf32=}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device

    np.random.seed(args.random_seed)
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)

    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='left')
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    mmlu_pro_questions = load_questions(args)
    logger.info(f'Read {len(mmlu_pro_questions)} questions from {args.mmlu_pro_train_path}')

    n_samples = len(mmlu_pro_questions)
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(
        f'Process {args.process_id} {n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(
        f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_id}.pt')
    if os.path.exists(output_file_path):
        important_tokens_data = torch.load(output_file_path)
        logger.info(
            f'Loaded {len(important_tokens_data)} important tokens from {output_file_path}, starting from {shard_start + len(important_tokens_data)}-th sample')
    else:
        important_tokens_data = []
        logger.info(f'No important tokens found in {output_file_path}, will generate them')

    if 'NV_YT_OPERATION_ID' in os.environ:
        logger.info('NY_YT_OPERATION_ID found in os.environ')
    else:
        logger.info('NY_YT_OPERATION_ID not found in os.environ')

    if shard_end > shard_start:
        with tqdm(total=shard_end - shard_start) as pbar:
            pbar.update(len(important_tokens_data))
            for sample_idx in range(shard_start + len(important_tokens_data), shard_end):
                question_sample = mmlu_pro_questions[sample_idx]
                answer = question_sample['answer']
                prompt = question_sample['prompt']
                batch_input_ids = \
                tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, return_tensors='pt',
                                              return_dict=True)['input_ids'].to(device)

                important_tokens_dict = find_important_tokens(
                    batch_input_ids, tokenizer=tokenizer, draft_model=draft_model, target_model=target_model)

                if not args.keep_responses:
                    del important_tokens_dict['responses']

                important_tokens_dict['id'] = sample_idx + args.write_index_offset

                important_tokens_data.append(important_tokens_dict)

                if ((sample_idx % args.dump_freq == (args.dump_freq - 1)) or (sample_idx == shard_end - 1)):
                    torch.save(important_tokens_data, output_file_path)

                pbar.update(1)
                pbar.set_description(f'Process {args.process_id}')
    else:
        torch.save(important_tokens_data, output_file_path)

    done_file = os.path.join(args.output_folder, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")
    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == args.process_saver_id and args.world_size > 1:
        logger.info("Process 0 is waiting for all other processes to finish...")
        while True:
            print("Process-saver tries to save snapshot:")
            done_files = [f"done_{i}.txt" for i in
                          range(args.process_saver_id + 1, args.process_saver_id + args.local_world_size)]
            print(f"{done_files=}")
            print(f"Done files:\n{[os.path.exists(os.path.join(args.output_folder, f)) for f in done_files]}")
            all_done = all(os.path.exists(os.path.join(args.output_folder, f)) for f in done_files)

            if all_done:
                logger.info("All processes finished. Concatenating important tokens data...")
                all_data = []
                for i in range(args.process_saver_id, args.process_saver_id + args.local_world_size):
                    # if i * shard_len >= shard_end:
                    #     continue
                    process_output_file = os.path.join(args.output_folder, f"{args.output_file}_{i}.pt")
                    all_data.extend(torch.load(process_output_file))
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    done_file = os.path.join(args.output_folder, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                final_output_file_path = os.path.join(args.output_folder,
                                                      args.output_file + f'_{args.process_saver_id}' + '.pt')
                logger.info(f"Saving important tokens data to {final_output_file_path}...")
                torch.save(all_data, final_output_file_path)
                logger.info("Done.")
                break

            time.sleep(5)
