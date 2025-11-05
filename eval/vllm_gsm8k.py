import json
import vllm

import torch
from tqdm import tqdm
import re
import os
import pickle
import time
import argparse
import random
import math

import sys

sys.path.append('..')
sys.path.append('.')



from lm_eval_utils import GSM8KEvaluator
from prompts import GSM8KPrompts, llama_assistant_turn_end

GSM8K_STOP_SEQUENCES = [
    "<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>",
    "Q:",
    "</s>",
    "<|im_end|>",
]
FORMATTING_PROMPT = 'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="out_file", required=True)
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--draft_model", required=True)
    parser.add_argument("--shots", type=int, choices=[0, 8], required=True)
    parser.add_argument("--no_spec_dec", action="store_true")
    parser.add_argument("--no_judge", action="store_true")
    parser.add_argument("--data_size", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--judge_path", required=False)
    parser.add_argument("--judge_threshold", type=float, required=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--force_overwrite", action="store_true")
    parser.add_argument("data_file")
    return parser.parse_args()


def load_questions(gsm8k_test_path, data_size):
    with open(gsm8k_test_path) as f:
        gsm_questions = [json.loads(line) for line in f]

    gsm_questions = [
        {
            "question": i["question"],
            "answer": i["answer"][i["answer"].rfind("#### ") + 5 :],
        }
        for i in gsm_questions
    ]
    if data_size == 1.0:
        return gsm_questions
    selected_questions = random.sample(
        gsm_questions, math.ceil(data_size * len(gsm_questions))
    )
    return selected_questions


def eval(
    gsm8k_test_path,
    target_model,
    draft_model,
    eval_shots,
    use_spec_dec,
    data_size,
    window_size,
    judge_config,
    temperature=0.6, 
    top_p=0.9,
):
    sampling_params = vllm.SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=2048, stop=GSM8K_STOP_SEQUENCES
    )
    print(f"@@@@@@@@@@@@@@@@@@@@ {torch.cuda.device_count()=}")
    if use_spec_dec:
        if judge_config is not None:
            speculative_config = {
                "model": draft_model,
                "num_speculative_tokens": window_size,
                "judge_config": judge_config,
                "quantization": None

            }
        else:
            speculative_config = {
                "model": draft_model,
                "num_speculative_tokens": window_size,
                "quantization": None,
            }
    else:
        speculative_config = None
    llm = vllm.LLM(
        model=target_model,
        tensor_parallel_size=torch.cuda.device_count(),
        quantization="modelopt",
        speculative_config=speculative_config,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=40000,
    )

    problems = load_questions(gsm8k_test_path, data_size)

    correct = 0
    all = 0

    outputs = []
    total_time = 0.0
    if eval_shots == 0:
        meta_prompt = GSM8KPrompts.prompt_with_0_shots
    elif eval_shots == 8:
        meta_prompt = GSM8KPrompts.prompt_with_8_shots

    for problem in tqdm(problems):
        question = problem["question"]
        formatted_prompt = meta_prompt + question + "\n" + GSM8KPrompts.formatting_prompt + llama_assistant_turn_end + "\n"

        start_time = time.time()
        cur_outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        end_time = time.time()
        torch.cuda.empty_cache()
        outputs += cur_outputs
        elapsed = end_time - start_time
        total_time += elapsed

    prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
    output_tokens = sum(
        len(out.token_ids) for output in outputs for out in output.outputs
    )

    evaluator = GSM8KEvaluator()

    for problem, output in zip(problems, outputs):
        answer = problem["answer"]

        verdict = (
            evaluator(generations=[output.outputs[0].text], references=[answer]) == 1.0
        )
        correct += verdict
        all += 1

    return correct / all, total_time, prompt_tokens, output_tokens

def get_judge_config(args):
    try:
        from vllm.config import JudgeConfig
        with open(args.judge_path, "rb") as f:
            data = pickle.load(f)
            # print(data)
            judge_config = JudgeConfig(
                weights=data["weights"],
                mean=data["mean"],
                scale=data["scale"],
                bias=data["bias"],
                threshold=args.judge_threshold,            
            )
        return judge_config
    except ImportError:
        return None

if __name__ == "__main__":
    args = parse_args()
    gsm8k_test_path = args.data_file
    out_file = args.out_file
    if not os.path.exists(out_file) or args.force_overwrite:
        if not args.no_judge:
            judge_config = get_judge_config(args)
        else:
            judge_config = None
        acc, total_time, prompt_tokens, output_tokens = eval(
            gsm8k_test_path,
            args.target_model,
            args.draft_model,
            args.shots,
            not args.no_spec_dec,
            args.data_size,
            args.window_size,
            judge_config,
            args.temperature,
            args.top_p,
        )

        autojudge_threshold = args.judge_threshold
        result = dict(
            thr=args.judge_threshold,
            acc=acc,
            total_time=total_time,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            window_size=args.window_size,
        )
        torch.save(result, out_file)
    else:
        print(f"out_file ({out_file}) exists and `--force_overwrite` flag wasn't provided. Skipping evaluation")
