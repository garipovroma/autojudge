import argparse
from typing import Dict, Any, List, Callable
import json
import os


import time
from tqdm.auto import tqdm

import numpy as np


import torch

import transformers
import itertools
import logging

import together
from together import Together

from time import perf_counter

from core_utils import color_replaced_tokens

import sys
sys.path.append('.')


from lm_eval_utils import GSM8KParser, GSM8KEvaluator
from prompts import build_gsm8k_0shot_messages, build_gsm8k_8shot_messages


logger = None  # NOTE will be initialized later


LAST_DUMP = perf_counter()
DUMP_FREQ_SECS = 20 * 60  # mins


def apply_chat_template(
    messages: List[Dict[str, Any]],
    tokenizer: transformers.AutoTokenizer,
) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        continue_final_message=False,
        add_generation_prompt=True,
    )


class TogetherGenerator:
    def __init__(
        self,
        model_id: str,
        tokenizer: transformers.AutoTokenizer,
    ) -> None:
        self.model_id = model_id
        self._tokenizer = tokenizer
        self._client = Together()

    def generate_greedy_from_chat(
        self,
        messages: List[Dict[str, Any]],
        max_length: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0,
            top_p=0,
            seed=0,
            top_k=1,
            max_tokens=max_length,
        )

        completion_dict = dict(
            role="assistant",
            content=completion.choices[0].message.content,
        )
        prompt = apply_chat_template(
            messages + [completion_dict], self._tokenizer
        )

        if device is not None:
            prompt = prompt.to(device)
        return prompt

    def complete_from_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        def _get_completion_sync(prompt_text):
            completion = self._client.completions.create(
                model=self.model_id,
                prompt=prompt_text,
                temperature=0,
                top_p=0,
                seed=0,
                top_k=1,
                max_tokens=max_length,
            )
            return completion

        assert input_ids.ndim == 2
        assert input_ids.size(0) == 1
        if device is None:
            device = input_ids.device
        prompt_text = self._tokenizer.batch_decode(input_ids, clean_up_tokenization_spaces=False)[0]
        num_retries = 0
        while True:
            try:
                completion = _get_completion_sync(prompt_text)
                break
            except together.error.ServiceUnavailableError as e:
                if num_retries >= 15:
                    print("Max retries reached, raising ServiceUnavailableError")
                    raise e
                print(f"Caught ServiceUnavailableError: {e} - retrying")
                num_retries += 1
                time.sleep(60)

        completion_text = completion.choices[0].text
        full_conversation = prompt_text + completion_text + "<|eot_id|>"
        input_ids = self._tokenizer(
            full_conversation,
            return_tensors="pt",
        )["input_ids"][:, 1:]
        if device is not None:
            input_ids = input_ids.to(device)
        return input_ids

    
@torch.no_grad()
def find_important_tokens(
    messages: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    draft_model: transformers.AutoModelForCausalLM,
    target_model: TogetherGenerator,
    device: torch.device,
):
    prompt = apply_chat_template(messages, tokenizer).to(device)

    global LAST_DUMP, DUMP_FREQ_SECS

    parser = GSM8KParser()
    evaluator = GSM8KEvaluator()

    batch_size, n_input_tokens = prompt.shape
    assert batch_size == 1

    target_response = target_model.generate_greedy_from_chat(
        messages,
        max_length=args.max_new_tokens,
        device=device,
    )

    batch_dict = {
        'input_ids': prompt,
        'attention_mask': torch.ones_like(prompt),
    }

    draft_response = draft_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_length=n_input_tokens + args.max_new_tokens
    )

    target_gen_str = tokenizer.batch_decode(target_response)
    target_ref_ans = parser(target_gen_str)

    draft_gen_str = tokenizer.batch_decode(draft_response)

    draft_answer = parser(draft_gen_str)

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = n_input_tokens

    responses = [current_response.cpu().clone()]
    
    for iteration in itertools.count():
        draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device), 
                                  draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(current_response)).logits.argmax(-1)[:, :-1]], dim=1)

        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted([idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if idx >= left_border])
        if not mismatch_indices:
            logger.info('\t\tDidn`t find any mismatches, leaving current task')
            break

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, n_input_tokens, changed_token_indices, tokenizer)

        mismatch_index = mismatch_indices[0]

        prefix_with_draft_token = torch.cat([
              current_response[:, :mismatch_index],
              torch.tensor([[draft_argmax_tokens[:, mismatch_index]]], device=device)
            ], dim=1)

        alternative_batch = {
            'input_ids': prefix_with_draft_token,
            'attention_mask': torch.ones_like(prefix_with_draft_token),
        }

        target_max_len_to_generate = args.max_new_tokens - (alternative_batch['input_ids'].shape[1] - n_input_tokens)
        logger.info(f"\t[iteration={iteration}], generated_tokens={(alternative_batch['input_ids'].shape[1] - n_input_tokens)}, remaining_tokens={target_max_len_to_generate}")

        if target_max_len_to_generate < 1:
            logger.info('\tGeneration limit exceeded, leaving current task')
            break

        alternative_response = target_model.complete_from_tokens(
            alternative_batch["input_ids"],
            max_length=target_max_len_to_generate,
        )

        if tokenizer.eos_token_id in alternative_response[:, n_input_tokens:mismatch_index + 1]:
            eos_pos = (alternative_response[:, n_input_tokens:mismatch_index + 1] == tokenizer.eos_token_id).nonzero(as_tuple=True)[1][-1].item() + n_input_tokens
            logger.info(f'\t\t\t EOS found at {eos_pos=}')
            alternative_response = alternative_response[:, :eos_pos + 1]


        alternative_gen_str = tokenizer.batch_decode(alternative_response)
        answers_match = evaluator(generations=alternative_gen_str, references=target_ref_ans) == 1.0
        alt_ans = parser(alternative_gen_str)
        draft_argmax_token = draft_argmax_tokens[:, mismatch_index].item()
        target_token = current_response[0, mismatch_index].item()
        logger.info(f'\t\t\t{target_max_len_to_generate=}, {mismatch_index=}, {len(mismatch_indices)=}, {draft_argmax_token=}, {target_token=},  {answers_match=}, {alt_ans=}, {target_ref_ans=}')

        if answers_match:
            changed_token_indices.append((mismatch_index, False, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
            current_response = alternative_response
        else:
            changed_token_indices.append((mismatch_index, True, current_response[0, mismatch_index].item(), alternative_response[0, mismatch_index].item()))
        left_border = mismatch_index + 1
        responses.append(current_response.cpu().clone())

        if tokenizer.eos_token_id in alternative_response[0, n_input_tokens:mismatch_index + 1].cpu().tolist():
            logger.info(f'\t\t\t EOS found in alternative_response, terminating')
            logger.info(f'\t\t\t alternative_response prefix: {tokenizer.decode(alternative_response[:, n_input_tokens:n_input_tokens + 16][0])}')
            logger.info(f'\t\t\t alternative_response suffix: {tokenizer.decode(alternative_response[:, -16:][0])}')
            break


    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens, draft_answer=draft_answer, target_answer=target_ref_ans, current_response=current_response, responses=responses)



def verify_args(args):
    assert args.process_id < args.world_size, "--process_id must be < --world_size"

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")
    
    parser.add_argument('--draft_model', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct',
                        help='Path or identifier of the draft model.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='float32',
                        help='Data type for torch tensors.')
    parser.add_argument('--gsm8k_train_path', type=str,
                        help='Path to the GSM8K train dataset JSON file.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--num_shots', type=int, default=0, choices=[0, 8],
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
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=8)

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
    print('The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    logging.basicConfig(
        level=logging.INFO, 
        format=f"[Process {args.process_id}] "+ "%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Initialized Logger")
    assert logger is not None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    together_client = Together()

    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device

    np.random.seed(args.random_seed)
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.draft_model, padding_side='left')
    target_model = TogetherGenerator(args.target_model, tokenizer)
    tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
    # draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id
    draft_model.generation_config.pad_token_id = tokenizer.pad_token_id

    gsm_questions = load_questions(args)
    logger.info(f'Read {len(gsm_questions)} questions from {args.gsm8k_train_path}')

    n_samples = len(gsm_questions)
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(f'Process {args.process_id} {n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if args.num_shots == 0:
        messages_prompt_builder: Callable[[str], list[dict[str, str]]] = build_gsm8k_0shot_messages
        logger.info(f'{args.num_shots=}, using build_gsm8k_0shot_messages messages builder')
    else:
        messages_prompt_builder = build_gsm8k_8shot_messages
        logger.info(f'{args.num_shots=}, using build_gsm8k_8shot_messages messages builder')

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
        logger.info('NY_YT_OPERATION_ID found in os.environ')
    else:
        logger.info('NY_YT_OPERATION_ID not found in os.environ')

    try:
        if shard_end >= shard_start:
            with tqdm(total=shard_end - shard_start) as pbar:
                pbar.update(len(important_tokens_data))
                for sample_idx in range(shard_start + len(important_tokens_data), shard_end):            
                    question_sample = gsm_questions[sample_idx]
                    answer = question_sample['answer']
                    question = question_sample['question']

                    messages = messages_prompt_builder(question)[1:]  # NOTE first message is system prompt of cutting knowledge date, it will be duplicated ==> we skip it

                    important_tokens_dict = find_important_tokens(
                        messages, tokenizer=tokenizer, draft_model=draft_model, target_model=target_model, device=device)
                    
                    if not args.keep_responses:
                        del important_tokens_dict['responses']

                    important_tokens_dict['id'] = sample_idx + args.write_index_offset

                    important_tokens_data.append(important_tokens_dict)

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
                done_files = [f"done_{i}.txt" for i in range(args.process_saver_id + 1, args.process_saver_id + args.local_world_size)]
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

                    final_output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_saver_id}' + '.pt')
                    logger.info(f"Saving important tokens data to {final_output_file_path}...")
                    torch.save(all_data, final_output_file_path)
                    logger.info("Done.")
                    break

                time.sleep(5)
    except Exception as e:
        print(f"Process {args.process_id} dies because of the exception, but it gracefully saves checkpoint before that. Error: {e}")
        torch.save(important_tokens_data, output_file_path)
        raise e
