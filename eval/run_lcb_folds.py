"""For use with gpu_parallel.py, see README.md"""

import sys

sys.path.append('..')
sys.path.append('.')


import argparse
import os
import torch
from tqdm import tqdm
from gpu_parallel import get_worker_rank, init_worker_logger, TaskQueue
import pandas as pd
import json
import transformers
import numpy as np

from joblib import dump
import logging

from src.livecodebench_v5 import load_code_generation_dataset, extract_code, apply_llama_lcb_prompt_format, codegen_metrics, unpack_lcb_data

from generation_utils import run_spec_dec

from time import perf_counter

LAST_DUMP = perf_counter()
DUMP_FREQ_SECS = 30 * 60  # mins

def parse_args():
    parser = argparse.ArgumentParser(description="Eval baselines")
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Endpoint for a zmq task dispenser that dispenses task indices. Provide *either* this or start & end"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First task to be processed by script inclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last task to be processed by script exclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default='.',
        help='Results will be written to "args.eval_folder/evals_data/limo/exp_name".'
    )
    parser.add_argument(
        "--dump_snapshot_freq",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--total_tasks",
        type=int,
        default=-1,
        help="For --queue!=None only!"
    )


    parser.add_argument("--draft_model", default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument("--target_model", default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--torch_dtype", default='float32')

    parser.add_argument("--head_path")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--window_size", default=16, type=int)
    parser.add_argument("--head_threshold_idx", type=int)
    parser.add_argument('--n_tasks', type=int, default=880)
    parser.add_argument('--num_process_evaluate', type=int, default=64)
    parser.add_argument('--lcb_path', type=str, default='none')
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument("--split_path", type=str)
    parser.add_argument("--setup", default='DD-DT', choices=['DD-DT', 'DT'])

    args = parser.parse_args()
    args.bench_name = 'LCB'

    return args


def load_head(args):
    checkpoints = pd.read_pickle(args.head_path)
    checkpoint_dict = checkpoints[args.fold_id]
    head = checkpoint_dict['model']
    scaler = checkpoint_dict['scaler']
    # head_thresholds = checkpoint_dict['thresholds']
    head_thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1175, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5]

    print(f'{head_thresholds}')

    return head, scaler, head_thresholds



def load_models(args, device: torch.device):
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map='auto', low_cpu_mem_usage=True)

    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map='auto', low_cpu_mem_usage=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='left')
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return draft_model, target_model, tokenizer


def test_program(program_tokens, eval_samples, tokenizer):
    code = extract_code(tokenizer.batch_decode(program_tokens)[0])
    result = codegen_metrics([eval_samples], [[code]], num_process_evaluate=args.num_process_evaluate)
    score = int(result[0]['pass@1'])
    return dict(score=score, code=code, gen=tokenizer.batch_decode(program_tokens)[0], result=result)

def run_spec_dec_collect_stats(sample_idx, question_sample, device, target_model, draft_model, tokenizer, scaler, head, args, eval_sample):
    batch_input_ids = tokenizer(question_sample, add_special_tokens=False, return_tensors='pt').to(device)['input_ids']

    run_stats = run_spec_dec(batch_input_ids, target_model, draft_model, tokenizer, scaler, head, args)

    generation = run_stats['current_gen']

    generation_str = tokenizer.batch_decode(generation)[0]

    gen_tokens = generation.shape[-1] - batch_input_ids.shape[-1]
    verdict = test_program(generation, eval_sample, tokenizer)['score']

    target_calls = run_stats['target_calls']
    draft_calls = run_stats['draft_calls']
    accepts = run_stats['accepts']
    mismatches = run_stats['mismatches']
    mean_accept = np.mean(accepts)
    accepts_levi = run_stats['accepts_levi']
    mean_accept_levi = np.mean(accepts_levi)

    stats_dict = {
            'idx': sample_idx,
            'question': question_sample,
            'input_tokens': batch_input_ids.cpu(),
            'tp': verdict,
            'gen_tokens': gen_tokens,
            't_ratio': round(target_calls / gen_tokens, 4) if gen_tokens else 0,
            'd_ratio': round(draft_calls / gen_tokens, 4) if gen_tokens else 0,
            'generation': generation.cpu(),
            't_calls': target_calls,
            'd_calls': draft_calls,
            'mean_accept': mean_accept,
            'raw_accepts': accepts,
            'mean_accept_levi': mean_accept_levi,
            'raw_accepts_levi': accepts_levi,
            'mismatches': mismatches,  # keeping only first mismatch in the drafting window,
            'generation_str': generation_str,
            'thr': args.head_threshold,
        }
    return stats_dict
    
def load_folds(args):
    folds = torch.load(args.split_path)
    folds = [set(i) for i in folds]
    return folds

def filter_by_ids(dataset, ids):
    filtered_dataset = [i for i in dataset if i.question_id in ids]
    return filtered_dataset

def main(args):
    rank = get_worker_rank()
    device = torch.device('cuda')  # gpu_parallel already sets CUDA_VISIBLE_DEVICES for you
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items() if k != 'bench_name'))
    logger.info(f'Output directory: {args.save_folder}')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
        logger.info(f'Created directory {args.save_folder}')
    else:
        logger.info(f'Directory {args.save_folder} already exists')

    logger.info('Loading model and tokenizer')
    draft_model, target_model, tokenizer = load_models(args, device)
    logger.info('Loading dataset')
    dataset = load_code_generation_dataset(args)
    logger.info(f'Loading {args.split_path}')
    split_ids = load_folds(args)[args.fold_id]

    dataset = filter_by_ids(dataset, split_ids)

    livecodebench_v5_dataset, livecodebench_v5_inputs_outputs = unpack_lcb_data(dataset, tokenizer)

    local_tasks_solved = 0

    head, scaler, head_thresholds = load_head(args)
    np.random.seed(args.random_seed)
    args.head_threshold = head_thresholds[args.head_threshold_idx]
    print(f'thr = {args.head_threshold}')

    def _run_task(idx: int):
        global LAST_DUMP, DUMP_FREQ_SECS
        nonlocal local_tasks_solved
        
        task_output_path = f'{args.save_folder}/Task_{idx}_{args.fold_id}.pkl'
        if os.path.exists(task_output_path):
            return  # already solved by previous attempt and saved in snapshot

        ######### EXAMPLE CODE ###########
        question_sample = livecodebench_v5_dataset[idx]['prompt']
        eval_sample = livecodebench_v5_inputs_outputs[idx]

        task_report = run_spec_dec_collect_stats(
            sample_idx=idx,
            question_sample=question_sample,
            device=device,
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            scaler=scaler,
            head=head,
            args=args,
            eval_sample=eval_sample
        )

        with open(task_output_path, "wb") as f_write:
            dump(task_report, filename=f_write)

        local_tasks_solved += 1

        ######### END OF EXAMPLE CODE ###########

    if args.start is not None and args.end is not None and args.queue is None:
        logger.info(f'Generating tasks [{args.start}; {args.end})')
        for idx in tqdm(range(args.start, args.end), desc=f'Process {rank}', total=args.end-args.start):
            _run_task(idx)
    elif args.queue is not None:
        logger.info(f'Generating tasks from {args.queue}')
        total = args.total_tasks if args.total_tasks != -1 else None
        for idx in tqdm(TaskQueue.iterate_tasks_from_queue(endpoint=args.queue), desc=f"Process {rank}", total=total):
            _run_task(idx)
    else:
        raise NotImplementedError("Please specify either --queue or both --start and --end")
    logger.info(f'Process {rank} has finished.')


if __name__ == "__main__":
    args = parse_args()

    main(args)