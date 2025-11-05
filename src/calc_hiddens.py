import argparse
from typing import Union, Sequence, Tuple
import json
import os

from tqdm.auto import tqdm
from termcolor import colored
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import transformers
import logging
import time

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def verify_args(args):
    assert args.process_id < args.n_processes, "--process_id must be < --n_processes"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--torch_dtype', type=str, default='float32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--n_processes', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--use_corrupted_data', action='store_true')
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()


    verify_args(args)

    return args

def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':
    args = get_args()
    print(f'The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    if args.torch_dtype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info(f'{torch.backends.cuda.matmul.allow_tf32=}, {torch.backends.cudnn.allow_tf32=}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device, low_cpu_mem_usage=True)
    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model, padding_side='right')
    tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    try:
        data = torch.load(args.data_file)
    except FileNotFoundError as e:
        file_found = False
        for i in range(500):
            fixed_path = args.data_file.split('.pt')[0] + f'_{i}.pt'
            try:
                data = torch.load(fixed_path)
                file_found = True
                break
            except FileNotFoundError:
                continue
        if not file_found:
            raise e
            
    if args.start != -1 and args.end != -1:
        data = data[args.start:args.end]


    if args.n_samples != -1:
        data = data[:args.n_samples]
        logger.info(f'But args.n_samples={args.n_samples}')
    n = len(data)


    if args.use_corrupted_data:
        for sample_dict in data:
            for mismatch_idx, (changed_token_pos, important, target_token, draft_token) in enumerate(sample_dict['changed_token_indices']):
                if not important: # draft token leads to the same answer as the target model gives
                    # so in the previous implementation of calc_hiddens.py we set draft_token at this position, but it has to be target token
                    sample_dict['current_response'][0, changed_token_pos] = draft_token
            del sample_dict['id']
            del sample_dict['hiddens']
            del sample_dict['prev_hiddens']

    block_size = (n + args.n_processes - 1) // args.n_processes
    start = args.process_id * block_size
    end = min((args.process_id + 1) * block_size, n)

    data = data[start:end]
    logger.info(f'Size before filtering: {len(data)}')
    data = [i for i in data if 'skip' not in i]
    logger.info(f'Size after filtering: {len(data)}')

    logger.info(f'Process {args.process_id} has {end - start} samples, [{start}:{end})')

    if end > start:
        for idx, sample_dict in enumerate(data):
            sample_dict['hiddens'] = []
            sample_dict['prev_hiddens'] = []

        tokens_to_encode = []
        text_sample_ids = []
        mismatch_ids = []

        n_samples = len(data)

        for sample_idx in tqdm(range(n_samples), total=n_samples, desc=f'Process {args.process_id}/{args.n_processes}'):
            sample_dict = data[sample_idx]

            for mismatch_idx, (changed_token_pos, importance, target_token, draft_token) in enumerate(data[sample_idx]['changed_token_indices']):
                token_ids = sample_dict['current_response'].clone()
                orig_token = token_ids[:, changed_token_pos].item()
                replacement_token = draft_token

                token_ids[:, changed_token_pos] = draft_token
                tokens_to_encode.append(token_ids.clone())
                text_sample_ids.append(sample_idx)
                mismatch_ids.append(mismatch_idx)

                token_ids = sample_dict['current_response'].clone()                
                token_ids[:, changed_token_pos] = target_token
                tokens_to_encode.append(token_ids.clone())
                text_sample_ids.append(sample_idx)
                mismatch_ids.append(mismatch_idx)

        n_seqs_to_encode = len(tokens_to_encode)
        n_tokens = 0

        orig_output_path = args.output_path
        args.output_path = f'{args.output_path}_{args.process_id}.pt'

        # check if args.output_path exists
        loaded_from_checkpoint = False
        loaded_checkpoint_batch_end = None
        if os.path.exists(args.output_path):
            checkpoint = torch.load(args.output_path)
            data = checkpoint['data']
            logger.info(f'File {args.output_path} exist')
            logger.info(f'Loaded {len(data)} samples from {args.output_path}')
            loaded_from_checkpoint = True
            loaded_checkpoint_batch_end = checkpoint['last_batch_end']
            logger.info(f'Loaded checkpoint batch end: {loaded_checkpoint_batch_end}')
        else:
            logger.info(f'File {args.output_path} does not exist, starting from scratch')

        for i in data:
            if 'responses' in i:
                del i['responses']


        iter_id = 0
        for batch_start in tqdm(range(0, n_seqs_to_encode, args.batch_size), desc=f'Process {args.process_id}/{args.n_processes}'):
            if loaded_from_checkpoint:
                if batch_start < loaded_checkpoint_batch_end:
                    continue
            iter_id += 1
            batch_end = min(n_seqs_to_encode, batch_start + args.batch_size)
            max_seq_len = max(i.numel() for i in tokens_to_encode[batch_start:batch_end])
            
            padded_token_ids_list = []
            # input_ids padding
            
            tokens_to_pad = [tokens_to_encode[sample_idx].flatten() for sample_idx in range(batch_start, batch_end)]

            inputs = tokenizer.pad(dict(input_ids=tokens_to_pad), return_tensors='pt')
            inputs = inputs.to(device)

            # draft & target forward to get hiddens
            with torch.no_grad():
                draft_hiddens = draft_model.model(**inputs).last_hidden_state.cpu()       
                target_hiddens = target_model.model(**inputs).last_hidden_state.cpu()

            del inputs

            for sample_in_batch_idx in range(0, batch_end - batch_start):  
                sample_idx = batch_start + sample_in_batch_idx
                mismatch_idx = mismatch_ids[sample_idx]
                text_sample_idx = text_sample_ids[sample_idx]

                changed_token_pos, _, _, _ = data[text_sample_idx]['changed_token_indices'][mismatch_idx]

                draft_hidden = draft_hiddens[sample_in_batch_idx, changed_token_pos]
                target_hidden = target_hiddens[sample_in_batch_idx, changed_token_pos]

                draft_prev_hidden = draft_hiddens[sample_in_batch_idx, changed_token_pos - 1]
                target_prev_hidden = target_hiddens[sample_in_batch_idx, changed_token_pos - 1]
                
                concated_hidden = torch.cat([draft_hidden, target_hidden], dim=0).to(torch.float32)
                concated_prev_hidden = torch.cat([draft_prev_hidden, target_prev_hidden], dim=0).to(torch.float32)
                
                if sample_in_batch_idx % 2 == 0:
                    data[text_sample_ids[sample_idx]]['hiddens'].append(concated_hidden)
                    data[text_sample_ids[sample_idx]]['prev_hiddens'].append(concated_prev_hidden)
                else:
                    data[text_sample_ids[sample_idx]]['hiddens'][-1] = torch.cat((data[text_sample_ids[sample_idx]]['hiddens'][-1], concated_hidden))
                    data[text_sample_ids[sample_idx]]['prev_hiddens'][-1] = torch.cat((data[text_sample_ids[sample_idx]]['prev_hiddens'][-1], concated_prev_hidden))
                
            del draft_hiddens, target_hiddens
            torch.cuda.empty_cache()

            if iter_id % args.save_freq == 0:
                checkpoint = dict(last_batch_end=batch_end, data=data)
                save_checkpoint(checkpoint, args.output_path)

        checkpoint = dict(last_batch_end=batch_end, data=data)
        save_checkpoint(checkpoint, args.output_path)

    else:
        checkpoint = dict(last_batch_end=0, data=[])
        save_checkpoint(checkpoint, args.output_path)

    folder_path = os.path.dirname(args.output_path)
    done_file = os.path.join(folder_path, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")

    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == 0:
        logger.info("Process 0 is waiting for all other processes to finish...")
        args.output_path = orig_output_path
        while True:
            done_files = [f"done_{i}.txt" for i in range(1, args.n_processes)]
            all_done = all(os.path.exists(os.path.join(folder_path, f)) for f in done_files) or args.n_processes == 1

            if all_done:
                logger.info("All processes have finished.")
                all_processes_data = []
                for i in range(args.n_processes):
                    process_output_file = f'{args.output_path}_{i}.pt'
                    try:
                        checkpoint = torch.load(process_output_file)
                        all_processes_data.extend(checkpoint['data'])
                        assert np.all([len(j['changed_token_indices']) == len(j['hiddens']) for j in checkpoint['data']])
                        logger.info(f"Removing {process_output_file}")
                        os.remove(process_output_file)
                    except FileNotFoundError:
                        print(f'Checkpoint {process_output_file} is missing. Skipping')

                    #remove done file
                    done_file = os.path.join(folder_path, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                all_processes_data_path = f'{args.output_path}.pt'
                save_checkpoint(all_processes_data, all_processes_data_path)
                logger.info(f"Saved all processes data to {all_processes_data_path}")
                break

            time.sleep(5)
