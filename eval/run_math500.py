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

from src.core_utils import extract_answer, extract_float
from generation_utils import run_spec_dec

import sys
sys.path.append('.')

from lm_eval_utils import MATH500Parser, MATH500Evaluator
from prompts import MATH500Prompts, llama_assistant_turn_end

# TODO add args
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
    parser.add_argument("--num_shots", default=0, type=int, choices=[0, 8])

    parser.add_argument("--math500_test_path")
    parser.add_argument("--head_path")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--window_size", default=16, type=int)
    parser.add_argument("--head_threshold_idx", type=int)
    parser.add_argument("--head_threshold", default=None, type=float)
    parser.add_argument("--setup", default='DD-DT', choices=['DD-DT', 'DT'])

    args = parser.parse_args()
    args.bench_name = 'MATH500'

    return args


def load_head(args):
    checkpoint_dict = pd.read_pickle(args.head_path)
    head = checkpoint_dict['model']
    scaler = checkpoint_dict['scaler']
    head_thresholds = checkpoint_dict['thresholds']

    return head, scaler, head_thresholds



def load_models(args, device: torch.device):
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map = 'auto', low_cpu_mem_usage=True)

    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map = 'auto', low_cpu_mem_usage=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='left')
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return draft_model, target_model, tokenizer


def load_questions(args):
    with open(args.math500_test_path) as f:
        questions = [json.loads(line) for line in f]

    questions = [
        {
            'question': question_dict['question'],
            'answer': question_dict['answer'],
            'solution': question_dict['solution'],
        }
        for question_dict in questions
    ]

    return questions

def sample_to_tokens(question_sample, tokenizer, device, args):
    question = question_sample['question']
    prompt_with_shots = MATH500Prompts.prompt_with_0_shots if args.num_shots == 0 else MATH500Prompts.prompt_with_8_shots
    prompt = prompt_with_shots + question + MATH500Prompts.formatting_prompt + llama_assistant_turn_end + "\n\n"
    batch_input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)

    return batch_input_ids


def run_spec_dec_collect_stats(sample_idx, question_sample, device, target_model, draft_model, tokenizer, scaler, head, args):
    batch_input_ids = sample_to_tokens(question_sample, tokenizer, device, args)
    run_stats = run_spec_dec(batch_input_ids, target_model, draft_model, tokenizer, scaler, head, args)

    generation = run_stats['current_gen']

    generations = tokenizer.batch_decode(generation) # 1-length list
    generation_str = generations[0]

    parser = MATH500Parser()
    evaluator = MATH500Evaluator()
    answer = [question_sample['answer']]

    verdict = bool(evaluator(generations=generations, references=answer, doc=[question_sample["solution"]]) == 1.0)

    answer_float = question_sample['answer']
    raw_pred = parser(generations)
    pred_float = raw_pred

    gen_tokens = generation.shape[-1] - batch_input_ids.shape[-1]

    target_calls = run_stats['target_calls']
    draft_calls = run_stats['draft_calls']
    accepts = run_stats['accepts']
    mismatches = run_stats['mismatches']
    mean_accept = np.mean(accepts)
    accepts_levi = run_stats['accepts_levi']
    mean_accept_levi = np.mean(accepts_levi)

    stats_dict = {
            'idx': sample_idx,
            'question': question_sample['question'],
            'raw_answer': question_sample['answer'],
            'input_tokens': batch_input_ids.cpu(),
            'answer': answer_float,
            # 'raw_pred': raw_pred,
            # 'pred': pred_float,
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
    dataset = load_questions(args)
    local_tasks_solved = 0

    head, scaler, head_thresholds = load_head(args)
    np.random.seed(args.random_seed)

    if args.head_threshold is None:
        # TODO: add this logic to all head evals 
        args.head_threshold = head_thresholds[args.head_threshold_idx]

    logger.info(f'head_threshold={args.head_threshold}')

    def _run_task(idx: int):
        nonlocal local_tasks_solved
        task_output_path = f'{args.save_folder}/Task_{idx}.pkl'
        if os.path.exists(task_output_path):
            return  # already solved by previous attempt and saved in snapshot

        ######### EXAMPLE CODE ###########
        question_sample = dataset[idx]

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