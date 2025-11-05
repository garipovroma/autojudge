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

from generation_utils import run_spec_dec

from src.core_utils import color_replaced_tokens, make_config, pairwise_judgment

import sys
sys.path.append('.')


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

    parser.add_argument("--arena_hard_questions_path")
    parser.add_argument("--arena_target_generations_path", default='data/arena_hard_auto/target_generations.json')
    
    parser.add_argument("--head_path")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--window_size", default=16, type=int)
    parser.add_argument("--head_threshold_idx", type=int)
    parser.add_argument("--head_threshold", default=None, type=float)
    parser.add_argument("--setup", default='DD-DT', choices=['DD-DT', 'DT'])
    parser.add_argument('--judge_model', type=str, default='gpt-4.1-mini')
    parser.add_argument('--judge_temperature', type=float, default=0.0)
    parser.add_argument('--judge_max_tokens', type=int, default=16000)

    args = parser.parse_args()
    args.bench_name = 'Arena-Hard-Auto'

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


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def load_target_generations(target_generations_file: str):
    """Load target generations from a file."""
    with open(target_generations_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results

def sample_to_tokens(question_sample, tokenizer, device, args):
    prompt = question_sample['prompt']

    batch_input_ids = tokenizer.apply_chat_template([
        {'role': 'user', 'content': prompt},
    ], return_tensors='pt', tokenize=True, add_generation_prompt=True).to(device)

    return batch_input_ids


draft_geq = set([
    "A>B",
    "A>>B",
    "A=B",
    "B=A",
    "B<A",
    "B<<A"   
])

def run_spec_dec_collect_stats(sample_idx, question_sample, device, target_model, draft_model, tokenizer, scaler, head, args, config):
    batch_input_ids = sample_to_tokens(question_sample, tokenizer, device, args)
    
    # args.max_new_tokens is patched here since we do it for arena's tokens mining 
    args.max_new_tokens = args.max_new_tokens_ - batch_input_ids.shape[1]

    run_stats = run_spec_dec(batch_input_ids, target_model, draft_model, tokenizer, scaler, head, args)

    generation = run_stats['current_gen']

    generation_str = tokenizer.batch_decode(generation[:, batch_input_ids.shape[1]:], skip_special_tokens=True)[0]
    target_generation = target_model.generate(
        input_ids=batch_input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None
    )
    target_generation_str = tokenizer.batch_decode(target_generation[:, batch_input_ids.shape[1]:], skip_special_tokens=True)[0]
    for try_idx in range(1):
        print(f'------------------ try# {try_idx} ------------------')
        judgment = pairwise_judgment(
            question=question_sample, 
            answer_a=generation_str, 
            answer_b=target_generation_str,
            config=config,
            model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
        tp = judgment['score'] in ('A=B', 'A>B')

        print(f"{judgment['score']=}, {tp=}")
        print()
        print()
        # del judgment['answer']
        # print(f"{judgment=}")

    gen_tokens = generation.shape[-1] - batch_input_ids.shape[-1]

    target_calls = run_stats['target_calls']
    draft_calls = run_stats['draft_calls']
    accepts = run_stats['accepts']
    mismatches = run_stats['mismatches']
    mean_accept = np.mean(accepts)
    accepts_levi = run_stats['accepts_levi']
    mean_accept_levi = np.mean(accepts_levi)

    stats_dict = {
            'idx': sample_idx if 'id' not in question_sample else question_sample['id'],
            'subcategory': question_sample['subcategory'],
            'question': question_sample['prompt'],
            'input_tokens': batch_input_ids.cpu(),
            'score': judgment['score'],
            'tp': tp,
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
            'target_generation_str': target_generation_str,
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
    dataset = load_questions(args.arena_hard_questions_path)
    # target_generations = load_target_generations(args.arena_target_generations_path)
    config = make_config('arena_hard_auto/config/arena-hard-v2.0.yaml')
    local_tasks_solved = 0

    # for idx in range(len(dataset)):
    #     dataset[idx]['target_generation'] = target_generations[dataset[idx]['id']]['target_completion']

    head, scaler, head_thresholds = load_head(args)
    np.random.seed(args.random_seed)

    if args.head_threshold is None:
        # TODO: add this logic to all head evals 
        args.head_threshold = head_thresholds[args.head_threshold_idx]

    logger.info(f'head_threshold={args.head_threshold}')

    # args.max_new_tokens is patched here since we do it for arena's tokens mining
    args.max_new_tokens_ = args.max_new_tokens

    def _run_task(idx: int):
        nonlocal local_tasks_solved
        task_output_path = f'{args.save_folder}/Task_{idx}_{args.head_threshold_idx}.pkl'
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
            config=config
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