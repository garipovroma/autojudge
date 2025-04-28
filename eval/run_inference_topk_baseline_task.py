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

from src.utils import extract_answer, extract_float
from generation_utils import run_spec_dec_top_k

prompt_with_8_shots = "Given the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
prompt_with_0_shots = "Given the following problem, reason and give a final answer to the problem.\n"
formatting_prompt = "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."


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

    parser.add_argument("--gsm8k_test_path")
    parser.add_argument("--head_path")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--window_size", default=16, type=int)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--setup", default='DD-DT')

    return parser.parse_args()


def load_head(args: 'args_dataclass'):
    checkpoint_dict = pd.read_pickle(args.head_path)
    head = checkpoint_dict['model']
    scaler = checkpoint_dict['scaler']

    # NOTE for the case with 2 tokens
    if args.setup == "DD-DT":
        scaler.mean_ = scaler.mean_[:2048 * 3]
        scaler.scale_ = scaler.scale_[:2048 * 3]
        scaler.n_features_in_ = 2048 * 3
    elif args.setup == "DD-DT-TD":
        scaler.mean_ = scaler.mean_[:2048 * 3 + 2048]
        scaler.scale_ = scaler.scale_[:2048 * 3 + 2048]
        scaler.n_features_in_ = 2048 * 3 + 2048
    else:
        raise NotImplementedError(f"Setup {args.setup} isn't currently supported")

    return head, scaler



def load_models(args: 'args_dataclass', device: torch.device):
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device, low_cpu_mem_usage=True)

    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map=device, low_cpu_mem_usage=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='left')
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return draft_model, target_model, tokenizer


def load_questions(args: 'args_dataclass'):
    with open(args.gsm8k_test_path) as f:
        gsm_questions = [json.loads(line) for line in f]

    gsm_questions = [
        {
            'question': question_dict['question'],
            'answer': question_dict['answer'][question_dict['answer'].rfind('#### ') + 5:]
        }
        for question_dict in gsm_questions
    ]

    return gsm_questions

def format_question_extract_answer_return_input_for_models(question_sample: dict[str, str], device: torch.device, tokenizer):
    answer = question_sample['answer']
    question = question_sample['question']

    formatted_zero_shot_prompt = prompt_with_0_shots + question + "\n" + formatting_prompt

    batch_input_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': formatted_zero_shot_prompt}],
        tokenize=True, return_tensors='pt', padding=True, continue_final_message=False,  # <--- TODO @romagaripov
    ).to(device)

    answer_float = extract_float(answer)

    return batch_input_ids, answer_float


def run_spec_dec_collect_stats(sample_idx, question_sample, device, target_model, draft_model, tokenizer, scaler, K, args: 'args_dataclass'):
    batch_input_ids, answer_float = format_question_extract_answer_return_input_for_models(question_sample=question_sample, device=device, tokenizer=tokenizer)
    run_stats = run_spec_dec_top_k(batch_input_ids, target_model, draft_model, tokenizer, scaler, K, args, setup=args.setup)

    generation = run_stats['current_gen']

    generation_str = tokenizer.batch_decode(generation)[0]

    raw_pred = extract_answer(generation_str)
    pred_float = extract_float(raw_pred)

    gen_tokens = generation.shape[-1] - batch_input_ids.shape[-1]
    verdict = int(answer_float == pred_float)

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
            'raw_pred': raw_pred,
            'pred': pred_float,
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
            'k': args.K,
        }
    return stats_dict


def main():
    rank = get_worker_rank()
    device = torch.device('cuda')  # gpu_parallel already sets CUDA_VISIBLE_DEVICES for you
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))
    logger.info(f'Output directory: {args.save_folder}')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
        logger.info(f'Created directory {args.save_folder}')
    else:
        logger.info(f'Directory {args.save_folder} already exists')

    logger.info('Loading model and tokenizer')
    draft_model, target_model, tokenizer = load_models(args, device)
    dataset = load_questions(args_dataclass)
    local_tasks_solved = 0

    head, scaler = load_head(args)
    np.random.seed(args_dataclass.random_seed)

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
            K=args_dataclass.K,
            args=args_dataclass,
        )

        with open(task_output_path, "wb") as f_write:
            dump(task_report, filename=f_write)

        local_tasks_solved += 1

        ######### END OF EXAMPLE CODE ###########

    if args.start is not None and args.end is not None and args.queue is None:
        logger.info(f'Generating tasks [{args.start}; {args.end})')
        for idx in tqdm(range(args.start, args.end), desc=f'Process {rank}', total=args.end-args.start+1):
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

    class args_dataclass:
        draft_model = args.draft_model
        target_model = args.target_model
        torch_dtype = args.torch_dtype
        gsm8k_test_path = args.gsm8k_test_path
        random_seed = args.random_seed
        max_new_tokens = args.max_new_tokens
        window_size = args.window_size
        K = args.K
        head_path = args.head_path
        setup = args.setup

    main()