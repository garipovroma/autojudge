import re

import numpy as np
import torch

from typing import Tuple
from termcolor import colored

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