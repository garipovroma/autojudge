import argparse
import numpy as np
import torch
import logging
import os

import sys
sys.path.append('.')

from src.livecodebench_v5 import load_code_generation_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_size', type=int, default=128)
    parser.add_argument('--filter_by_difficulty', type=str, default='none')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--train_file', type=str, default='train.pt')
    parser.add_argument('--val_file', type=str, default='val.pt')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--lcb_path', type=str, default='none')
    parser.add_argument('--n_tasks', type=int, default=880)

    args = parser.parse_args()
    assert 1 <= args.val_size <= 879

    return args

def make_split(ids, args):
    np.random.shuffle(ids)
    first_val_sample_idx = -args.val_size
    train, val = ids[:first_val_sample_idx], ids[first_val_sample_idx:]
    return train, val

def save_file(args, data, file_name):
    full_path = os.path.join(args.output_folder, file_name)
    logger.info(f'Saving split to {full_path}')
    torch.save(data, full_path)

if __name__ == '__main__':
    args = parse_args()
    print(f'The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    np.random.seed(args.random_seed)

    logger.info('Loading dataset')
    dataset = load_code_generation_dataset(args, shuffle=False)
    ids = [i.question_id for i in dataset]

    logger.info(f'Creating {args.output_folder}')
    os.makedirs(args.output_folder, exist_ok=True)

    train, val = make_split(ids, args)
    logger.info(f'train_size = {len(train)}')
    save_file(args, train, args.train_file)
    logger.info(f'val_size = {len(val)}')
    save_file(args, val, args.val_file)

    logger.info('Done')