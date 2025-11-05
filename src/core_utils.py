import re

import numpy as np
import torch

from typing import Tuple
from termcolor import colored

# arena_hard_auto imports
import openai
import yaml
import os
import time
import json
from collections import Counter

def find_answer(response: torch.Tensor, tokenizer, patterns=('The final answer is', ' The final answer is', '\n\nThe final answer is'), end_tokens=[128009, 271]) -> Tuple:
    """find start and end index of what comes after 'the final answer is ...'"""
    answer_segments = []
    rightmost_seg = -1
    rightmost_seg_idx = -1

    # print(f'response={tokenizer.batch_decode(response)}')
    for pattern_idx, pattern in enumerate(patterns):
        seq_len = response.shape[-1]
        answer_tokens = tokenizer(pattern, add_special_tokens=False, return_tensors='pt')['input_ids']
        answer_tokens = answer_tokens.to(response.device)
        n_ans_tokens = answer_tokens.numel()
        
        n_intervals = seq_len - n_ans_tokens + 1
        rows = torch.arange(n_intervals, dtype=torch.int)[..., None]
        cols = torch.arange(n_ans_tokens, dtype=torch.int)
        indices = rows + cols
    
        answer_positions = torch.nonzero((response.repeat((n_intervals, 1))[rows, indices] == answer_tokens).prod(dim=-1))
        answer_positions = answer_positions.flatten().tolist()

        if len(answer_positions) == 0:
            answer_segments.append(None)
        else:
            left = answer_positions[-1]
            right = left
            while right < seq_len:
                if response[0, right] in end_tokens: # 271 -> '\n\n':
                    right += 1
                    break
                right += 1
            answer_segments.append((left, right))
            if left > rightmost_seg:
                rightmost_seg = left
                rightmost_seg_idx = pattern_idx
    return answer_segments[rightmost_seg_idx]

def extract_answer_v2(s, tokenizer):
    left, right = find_answer(s, tokenizer)
    answer_str = tokenizer.batch_decode(s[:, left:right], skip_special_tokens=True)[0]
    extracted_answer = extract_answer(answer_str)
    
    return extracted_answer


def extract_answer(s, suffix='<|eot_id|>'):
    s = s.lower().replace(suffix, '').replace('the final answer is', '=')
    idx = s.rfind("=")
    if idx != - 1:
        return s[idx + 1:].strip()
    

def extract_float(num_str):
    try:
        num_str = re.sub(r'[^0-9.-]', '', num_str).strip(".")
        return float(num_str)
    except (ValueError, TypeError):
        return

def color_replaced_tokens(current_response, argmax_tokens, prompt_max_length, changed_token_indices, tokenizer):
    seq_len = current_response.shape[-1]
    is_mismatch = current_response != argmax_tokens  # [batch_size=1, max_length]
    mismatch_indices = sorted([idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= prompt_max_length])

    token_colors = ['black' for _ in range(seq_len)]
    for mismatch_idx in mismatch_indices:
        token_colors[mismatch_idx] = 'red'

    for changed_token_idx, is_important, orig_token, alternative_token in changed_token_indices:
        if is_important:
            token_colors[changed_token_idx] = 'yellow'
        else:
            token_colors[changed_token_idx] = 'green'

    colored_tokens = []
    
    for token, color in zip(current_response.flatten().tolist(), token_colors):
        colored_tokens.append(colored(tokenizer.decode(token), color))
        
    for token_pos, is_important, orig_token, alternative_token in changed_token_indices:
        alternative = f"[{tokenizer.decode(alternative_token if is_important else orig_token)}]".replace('\n', '\\n').replace(' ', '_')
        colored_tokens[token_pos] = f'{colored_tokens[token_pos]} {colored(alternative, "light_grey")}'

    return "".join(colored_tokens)

MODEL_TO_HIDDEN_DIM = {
    'meta-llama/Llama-3.1-405B-Instruct': 2048 * 8,
    'meta-llama/Llama-3.1-70B-Instruct': 2048 * 4,
    'meta-llama/Llama-3.1-8B-Instruct': 2048 * 2,
    'meta-llama/Llama-3.2-1B-Instruct': 2048,
    'Qwen/Qwen2.5-0.5B-Instruct': 896, # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json
    'Qwen/Qwen2.5-7B-Instruct': 3584, # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json,
    'Qwen/Qwen2.5-32B-Instruct': 5120, # https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json,
    'nvidia/Llama-3.1-405B-Instruct-FP8': 16384,
}

def make_setup_slice_mapper(draft_model, target_model):
    draft_hidden_dim = MODEL_TO_HIDDEN_DIM[draft_model]
    target_hidden_dim = MODEL_TO_HIDDEN_DIM[target_model]

    return {
        'DD-DT': slice(0, draft_hidden_dim + target_hidden_dim),
        'DT': slice(draft_hidden_dim, draft_hidden_dim + target_hidden_dim)
    }

### arena_hard_auto utils

OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

JUDGE_SETTINGS = {
    "hard_prompt": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    },
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}

def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs
    
try:
    config = make_config('arena_hard_auto/config/arena-hard-v2.0.yaml')
except FileNotFoundError as e:
    config = make_config('../arena_hard_auto/config/arena-hard-v2.0.yaml')

global api_key_index
api_key_index = 0

api_keys = [
    ("YOUR_API_KEY", "NAME_YOUR_KEY_SOMEHOW")
]

BASE_URL = 'https://api.anthropic.com/v1/'

import httpx
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

global client
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=api_keys[api_key_index][0],
    http_client = httpx.Client(verify=False)
)

API_MAX_RETRY = 15
API_RETRY_SLEEP = 5
API_ERROR_OUTPUT = None

NONE_RESPONSE_RETRY = 3
API_BEST_OF_N_SLEEP = 0.5

def api_completion_func(model, **kwargs):
    output: None | Literal[1] = API_ERROR_OUTPUT
    global api_key_index
    global client
    for _ in range(API_MAX_RETRY):

        kwargs.update({
            'model': model
        })

        response = requests.post(
            BASE_URL,
            headers={
                "authorization": f"OAuth {api_keys[api_key_index][0]}",
                "content-type": "application/json",
            },
            verify=False,
            json=kwargs
        )

        completion = response.json()
        if 'error' in completion:
            print(completion, completion['error'])
            if 'quota exceeded' in completion['error'].lower():
                new_index = (api_key_index + 1) % len(api_keys)
                print(f"Switching to {api_keys[new_index][1]}'s API key, current index: {api_key_index} / {len(api_keys)}, new index: {new_index} / {len(api_keys)}")
                api_key_index = new_index

                client = openai.OpenAI(
                    base_url=BASE_URL,
                    api_key=api_keys[api_key_index][0],
                    http_client = httpx.Client(verify=False)
                )

                time.sleep(API_RETRY_SLEEP)
        else:
            try:
                answer = completion["response"]["content"][0]["text"]
            except Exception:
                return None

            output = {
                "answer": answer,
                "completion": completion,
            }
            break
        
    
    return output


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None

from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_best_of_n(
    kwargs,
    best_of_n,
    regex_patterns,
):
    scores = []
    outputs = []
    unique_scores = set()

    def one_trial_():
        score = None
        for _ in range(NONE_RESPONSE_RETRY):
            if score is None:
                output = api_completion_func(**kwargs)
                if output is None:
                    score = None
                else:
                    score = get_score(output["answer"], regex_patterns)
            else:
                break
            time.sleep(API_BEST_OF_N_SLEEP)
        return score, output

    workers = best_of_n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one_trial_) for _ in range(best_of_n)]
        for fut in as_completed(futures):
            result = fut.result()
            s, output = result
            scores.append(s)
            unique_scores.add(s)
            outputs.append(output)
            if Counter(scores).most_common(1)[0][1] * 2 >= best_of_n:
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    return scores, outputs

def one_trial(kwargs, regex_patterns):
    score = None
    for _ in range(NONE_RESPONSE_RETRY):
        if score is None:
            output: None | Literal[1] = api_completion_func(**kwargs)
            if output is None:
                score = None
            else:
                score = get_score(output["answer"], regex_patterns)
        else:
            break
        time.sleep(API_BEST_OF_N_SLEEP)
    return score, output

def sequential_best_of_n(
    kwargs,
    best_of_n,
    regex_patterns,
    early_stop=False
):
    scores = []
    outputs = []
    unique_scores = set()

    for try_idx in range(best_of_n):
        score, output = one_trial(kwargs=kwargs, regex_patterns=regex_patterns)
        scores.append(score)
        unique_scores.add(score)
        outputs.append(output)

        if early_stop and Counter(scores).most_common(1)[0][1] * 2 >= best_of_n:
            break

        
    
    return scores, outputs

dump_idx = 0
def pairwise_judgment(question, answer_a, answer_b, config, model='gpt-4.1-mini', temperature=0.0, max_tokens=16000, dump=False, best_of_n=3):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": answer_a,
        "ANSWER_B": answer_b,
    }

    user_prompt = config['prompt_template'].format(**prompt_args)
    system = [
        {
            "type": "text",
            "text": JUDGE_SETTINGS[question['category']]["system_prompt"],
        }
    ]
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    kwargs = {
        "model": model,
        "system": system,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if dump:
        global dump_idx
        os.mkdir('dumps')
        torch.save(kwargs, f"dumps/dump_{question['uid']}_{dump_idx}.pt")
        dump_idx += 1

    scores, outputs = sequential_best_of_n(kwargs=kwargs, best_of_n=best_of_n, regex_patterns=config['regex_patterns'], early_stop=False)

    score_mapper = {
        "A>B": "A>B",
        "A>>B": "A>B",
        "B<A": "A>B",
        "B<<A": "A>B",

        'A=B': 'A=B',
        'B=A': 'A=B',

        "B>A": "B>A",
        "B>>A": "B>A",
        "A<B": "B>A",
        "A<<B": "B>A",

        'None': 'None',
        None: 'None'
    }

    mapped_scores = [score_mapper[score] for score in scores]
    if 'B>A' not in mapped_scores and 'A>B' in mapped_scores:
        final_score = 'A>B'
    elif 'A>B' not in mapped_scores and 'B>A' in mapped_scores:
        final_score = 'B>A'
    else:
        final_score = 'A=B'

    result = {
        "score": final_score,
        "scores": scores,
        "mapped_scores": mapped_scores,
        "messages": messages,
        "output": outputs,
    }

    return result