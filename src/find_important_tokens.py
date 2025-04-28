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

from utils import color_replaced_tokens

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

prompt_with_8_shots = "Given the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
prompt_with_0_shots = "Given the following problem, reason and give a final answer to the problem.\n"
formatting_prompt = "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."


def find_answer(response: torch.Tensor, tokenizer: transformers.AutoTokenizer, patterns=('The final answer is', ' The final answer is', '\n\nThe final answer is'), end_tokens=[128009, 271]) -> Tuple:
    """find start and end index of what comes after 'the final answer is ...'"""
    answer_segments = []
    rightmost_seg = -1
    rightmost_seg_idx = -1
    
    eos_positions = (response == tokenizer.eos_token_id).nonzero(as_tuple=True)[1].tolist()
    eos_cutoff = eos_positions[2] if len(eos_positions) >= 3 else 999999999999999

    # print(f'response={tokenizer.batch_decode(response)}')
    for pattern_idx, pattern in enumerate(patterns):
        seq_len = response.shape[-1]
        answer_tokens = tokenizer(pattern, add_special_tokens=False, return_tensors='pt')['input_ids']
        answer_tokens = answer_tokens.to(device)
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
                
            if left < eos_cutoff:
                answer_segments.append((left, right))
                if left > rightmost_seg:
                    rightmost_seg = left
                    rightmost_seg_idx = pattern_idx
            else:
                answer_segments.append(None)
    return answer_segments[rightmost_seg_idx]
    
@torch.no_grad()
def find_important_tokens(
    prompt: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    draft_model: transformers.AutoModelForCausalLM,
    target_model: transformers.AutoModelForCausalLM,
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
    draft_ans_start, draft_ans_end = find_answer(draft_response, tokenizer) 
    draft_answer = draft_response[:, draft_ans_start: draft_ans_end]
    
    ref_start, ref_end = find_answer(target_response, tokenizer) 
    target_answer = target_response[:, ref_start: ref_end]  # make sure this include end toke (\n\n or eos)

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = prompt_max_length

    responses = [current_response.cpu().clone()]
    
    for iteration in itertools.count():
        start, end = find_answer(current_response, tokenizer)
        draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device), 
                                  draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(current_response)).logits.argmax(-1)[:, :-1]], dim=1)

        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted([idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= left_border])
        if not mismatch_indices:
            break

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, prompt_max_length, changed_token_indices, tokenizer)

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
        alt_start_end_tuple = find_answer(alternative_response, tokenizer)
        if alt_start_end_tuple is not None:
            alt_start, alt_end = alt_start_end_tuple
            alternative_answer = alternative_response[:, alt_start: alt_end]
        if alt_start_end_tuple is not None and alternative_answer.numel() == target_answer.numel() and torch.all(alternative_answer == target_answer):
            changed_token_indices.append((mismatch_index, False, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
            current_response = alternative_response
        else:
            changed_token_indices.append((mismatch_index, True, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
        left_border = mismatch_index + 1
        responses.append(current_response.cpu().clone())

    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens, draft_answer=draft_answer, target_answer=target_answer, current_response=current_response, responses=responses)


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


def verify_args(args):
    assert args.num_shots in (0, 8), "--num_shots must be 0 or 8"
    assert args.process_id < args.world_size, "--process_id must be < --world_size"

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")
    
    parser.add_argument('--draft_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Path or identifier of the draft model.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='float32',
                        help='Data type for torch tensors.')
    parser.add_argument('--gsm8k_train_path', type=str, default='/home/garipovroma/code/speculation_for_reasoning/data/gsm8k_train.json',
                        help='Path to the GSM8K train dataset JSON file.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--num_shots', type=int, default=8,
                        help='Number of shots to use.')
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

    args = parser.parse_args()
    verify_args(args)

    return args

def load_questions(args):
    with open(args.gsm8k_train_path) as f:
        gsm_questions = [json.loads(line) for line in f]

    gsm_questions = [
        {
            'question': i['question'],
            'answer': i['answer'][i['answer'].rfind('#### ') + 5:]
        }
        for i in gsm_questions
    ]

    return gsm_questions

if __name__ == "__main__":
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
    tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    gsm_questions = load_questions(args)
    logger.info(f'Read {len(gsm_questions)} questions from {args.gsm8k_train_path}')

    n_samples = len(gsm_questions)
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(f'{n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if args.num_shots == 0:
        prompt_with_shots = prompt_with_0_shots
        logger.info(f'{args.num_shots=}, using prompt_with_0_shots')
    else:
        prompt_with_shots = prompt_with_8_shots
        logger.info(f'{args.num_shots=}, using prompt_with_8_shots')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_id}.pt')
    if os.path.exists(output_file_path):
        important_tokens_data = torch.load(output_file_path)
        logger.info(f'Loaded {len(important_tokens_data)} important tokens from {output_file_path}, starting from {shard_start + len(important_tokens_data)}-th sample')
    else:
        important_tokens_data = []
        logger.info(f'No important tokens found in {output_file_path}, will generate them')

    if 'NV_YT_OPERATION_ID' in os.environ:
        logger.info(f'NY_YT_OPERATION_ID found in os.environ')
    else:
        logger.info(f'NY_YT_OPERATION_ID not found in os.environ')

    with tqdm(total=shard_end - shard_start) as pbar:
        pbar.update(len(important_tokens_data))
        for sample_idx in range(shard_start + len(important_tokens_data), shard_end):            
            question_sample = gsm_questions[sample_idx]
            answer = question_sample['answer']
            question = question_sample['question']
            prompt = prompt_with_shots + question + "\n" + formatting_prompt
            batch_input_ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=True, return_tensors='pt', padding=True, continue_final_message=False # <--- MIGHT BE A BUG
            ).to(device)

            important_tokens_dict = find_important_tokens(
                batch_input_ids, tokenizer=tokenizer, draft_model=draft_model, target_model=target_model)
            
            if not args.keep_responses:
                del important_tokens_dict['responses']

            target_float_ans = extract_float(extract_answer_v2(important_tokens_dict['target_answer'], tokenizer))
            draft_float_ans = extract_float(extract_answer_v2(important_tokens_dict['draft_answer'], tokenizer))
            important_tokens_dict['target_float_ans'] = target_float_ans
            important_tokens_dict['draft_float_ans'] = draft_float_ans
            important_tokens_dict['target_ans_is_none'] = target_float_ans is None
            important_tokens_dict['draft_ans_is_none'] = draft_float_ans is None
            important_tokens_dict['id'] = sample_idx + args.write_index_offset

            important_tokens_data.append(important_tokens_dict)

            if ((sample_idx % args.dump_freq == (args.dump_freq - 1)) or (sample_idx == shard_end - 1)):
                torch.save(important_tokens_data, output_file_path)
            pbar.update(1)
            pbar.set_description(f'Process {args.process_id}')

    done_file = os.path.join(args.output_folder, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")
    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == 0 and args.world_size > 1:
        logger.info("Process 0 is waiting for all other processes to finish...")
        while True:
            done_files = [f"done_{i}.txt" for i in range(1, args.world_size)]
            all_done = all(os.path.exists(os.path.join(args.output_folder, f)) for f in done_files)

            if all_done:
                logger.info("All processes finished. Concatenating important tokens data...")
                all_data = []
                for i in range(args.world_size):
                    process_output_file = os.path.join(args.output_folder, f"{args.output_file}_{i}.pt")
                    all_data.extend(torch.load(process_output_file))
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    done_file = os.path.join(args.output_folder, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                final_output_file_path = os.path.join(args.output_folder, args.output_file + '.pt')
                logger.info(f"Saving important tokens data to {final_output_file_path}...")
                torch.save(all_data, final_output_file_path)
                logger.info("Done.")
                break

            time.sleep(5)
