import argparse
from typing import Union, Sequence, Tuple, Dict
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
import yaml

from time import perf_counter

from core_utils import color_replaced_tokens, make_config, pairwise_judgment

import sys
sys.path.append('.')

LAST_DUMP = perf_counter()
DUMP_FREQ_SECS = 30 * 60  # mins
    
@torch.no_grad()
def find_important_tokens(
    prompt: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    draft_model: transformers.AutoModelForCausalLM,
    target_model: transformers.AutoModelForCausalLM,
    logger: logging.Logger,
    question: Dict,
    config: Dict,
    args: argparse.Namespace
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
        max_new_tokens=args.max_new_tokens
    )

    draft_response = draft_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=args.max_new_tokens
    )

    target_gen_str = tokenizer.batch_decode(target_response[:, n_input_tokens:], skip_special_tokens=True)[0]
    draft_gen_str = tokenizer.batch_decode(draft_response[:, n_input_tokens:], skip_special_tokens=True)[0]

    # logger.info(f"target_gen_str={target_gen_str}")
    # logger.info(f"draft_gen_str={draft_gen_str}")

    draft_vs_target_judgment = pairwise_judgment(
        question=question, 
        answer_a=draft_gen_str, 
        answer_b=target_gen_str,
        config=config,
        model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        best_of_n=args.best_of_n
    )
    logger.info(f"judgment['score']={draft_vs_target_judgment['score']}, A is draft, B is target")

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = n_input_tokens

    responses = [current_response.cpu().clone()]
    
    gen_lens = []

    target_token_is_important_verdicts = set([
        "A<B",
        "A<<B",
        "B>A",
        "B>>A",
        "None"
    ])

    judgments = []

    for iteration in itertools.count():
        draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device), 
                                  draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(current_response)).logits.argmax(-1)[:, :-1]], dim=1)
        logger.info(f'{current_response.shape=}, {draft_argmax_tokens.shape=}')
        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted([idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= left_border])
        if len(mismatch_indices) > 230:
            logger.info(f'Skipping, too much mismatches = {len(mismatch_indices)}')
            return dict(skip=True)
        if not mismatch_indices:
            break

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, n_input_tokens, changed_token_indices, tokenizer)
        mismatch_index = mismatch_indices[0]
        
        logging.info(f'Iteration {iteration}, {mismatch_index=}')

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
        )
        if tokenizer.eos_token_id in alternative_response[:, n_input_tokens:mismatch_index + 1]:
            eos_pos = (alternative_response[:, n_input_tokens:mismatch_index + 1] == tokenizer.eos_token_id).nonzero(as_tuple=True)[1][-1].item() + n_input_tokens
            logger.info(f'\t\t\t EOS found at {eos_pos=}')
            alternative_response = alternative_response[:, :eos_pos + 1]

        alternative_gen_str = tokenizer.batch_decode(alternative_response[:, n_input_tokens:], skip_special_tokens=True)[0]
        current_gen_str = tokenizer.batch_decode(current_response[:, n_input_tokens:], skip_special_tokens=True)[0]

        judgment = pairwise_judgment(
            question=question, 
            answer_a=alternative_gen_str, 
            answer_b=current_gen_str,
            config=config,
            model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            best_of_n=args.best_of_n
        )
        judgments.append(judgment)

        gen_lens.append(current_response.shape[1])
        gen_lens.append(alternative_response.shape[1])
        
        logger.info(f'\t\tmismatches in total: {len(mismatch_indices)}, mismatches = {mismatch_indices[:15]}...')
        
        if judgment['score'] not in target_token_is_important_verdicts:
            changed_token_indices.append((mismatch_index, False, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
            current_response = alternative_response
            logger.info('\t\tNot Important')
        else:
            changed_token_indices.append((mismatch_index, True, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
            logger.info('\t\tImportant')
        left_border = mismatch_index + 1
        responses.append(current_response.cpu().clone())

    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens, draft_gen=draft_response.cpu(), target_gen=target_response.cpu(), current_response=current_response, responses=responses, gen_lens=gen_lens, draft_vs_target_judgment=draft_vs_target_judgment, target_gen_str=target_gen_str, draft_gen_str=draft_gen_str, judgments=judgments)


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
    parser.add_argument('--arena_hard_questions_path', type=str, default='arena_hard_auto/data/arena-hard-v2.0/question.jsonl',
                        help='Path to the GSM8K train dataset JSON file.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=4096 * 2,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder name.')
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
    parser.add_argument('--judge_model', type=str, default='gpt-4.1-mini')
    parser.add_argument('--judge_temperature', type=float, default=0.0)
    parser.add_argument('--judge_max_tokens', type=int, default=16000)
    parser.add_argument('--shard_start', type=int, default=-1)
    parser.add_argument('--shard_end', type=int, default=-1)
    parser.add_argument('--best_of_n', type=int, default=3)
    parser.add_argument('--saver', action='store_true')

    args = parser.parse_args()
    verify_args(args)

    return args


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(
        level=logging.INFO, 
        format=f"[Process {args.process_id}] %(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)


    print('The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

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
    tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    questions = load_questions(args.arena_hard_questions_path)
    config = make_config('arena_hard_auto/config/arena-hard-v2.0.yaml')

    logger.info(f'Read {len(questions)} questions from {args.arena_hard_questions_path}')

    n_samples = len(questions)
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)

    if args.shard_start != -1 and args.shard_end != -1:
        n_samples = args.shard_end - args.shard_start
        shard_len = (n_samples + 8 - 1) // 8
        shard_start = args.shard_start + (args.process_id % 8) * shard_len
        shard_end = args.shard_start + min((((args.process_id % 8) + 1)) * shard_len, n_samples)

    logger.info(f'Process {args.process_id} {n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    if 'NV_YT_OPERATION_ID' in os.environ:
        logger.info('NY_YT_OPERATION_ID found in os.environ')
    else:
        logger.info('NY_YT_OPERATION_ID not found in os.environ')

    ##### FAST REBUTTAL QWEN PATCH START #####

    if 'qwen' in args.target_model.lower():
        tokenizer.bos_token_id = tokenizer.eos_token_id # based on: https://huggingface.co/Qwen/Qwen2-7B-Instruct/discussions/15
        with open("data/qwens/gsm8k-cot-llama.yaml", "r") as f:
            config = yaml.safe_load(f)

        fewshot_samples = config.get("fewshot_config", {}).get("samples", [])

        format_prompt = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            "Problem: {question}\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        )

        few_shot_turns = []
        for sample in fewshot_samples:
            question = sample["question"]
            target = sample["target"]

            formatted_question = format_prompt.format(question=question)
            
            few_shot_turns.append({
                "role": "user",
                "content": formatted_question,
            })
            few_shot_turns.append({
                "role": "assistant",
                "content": target,
            })
    ##### FAST REBUTTAL QWEN PATCH END #####

    if shard_end > shard_start:
        with tqdm(total=shard_end - shard_start) as pbar:
            for sample_idx in range(shard_start, shard_end):     
                file_path = os.path.join(args.output_folder, f'Task_{sample_idx}.pt')

                if os.path.exists(file_path):
                    logging.info(f'{file_path} exists, skipping {sample_idx} sample')
                    pbar.update(1)
                    continue

                prompt = questions[sample_idx]['prompt']

                batch_input_ids = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': prompt},
                ], return_tensors='pt', tokenize=True, add_generation_prompt=True).to(device)
                
                ##### FAST REBUTTAL QWEN PATCH START #####
                if 'qwen' in args.target_model.lower():
                    raise ValueError("Qwen isn't supported for arena_hard_auto for now")
                ##### FAST REBUTTAL QWEN PATCH END #####


                important_tokens_dict = find_important_tokens(
                    batch_input_ids, 
                    tokenizer=tokenizer, 
                    draft_model=draft_model, 
                    target_model=target_model, 
                    logger=logger,
                    question=questions[sample_idx],
                    config=config,
                    args=args
                )

                important_tokens_dict['uid'] = questions[sample_idx]['uid']

                if not args.keep_responses and 'responses' in important_tokens_dict:
                    del important_tokens_dict['responses']

                important_tokens_dict['id'] = sample_idx + args.write_index_offset

                torch.save(important_tokens_dict, file_path)

                pbar.update(1)
                pbar.set_description(f'Process {args.process_id}')
    else:
        logger.info(f'Process {args.process_id} has no samples to process')


