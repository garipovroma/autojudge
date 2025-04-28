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
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--filter_by_difficulty', type=str, default='none')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--folds_file', type=str, default='folds.pt')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--lcb_path', type=str, default='none')
    parser.add_argument('--n_tasks', type=int, default=880)

    args = parser.parse_args()
    assert 1 <= args.n_folds <= 880

    return args

def make_split(ids, args):
    np.random.shuffle(ids)
    n = len(ids)
    fold_len = (n + args.n_folds - 1) // args.n_folds
    folds = [ids[fold_len * i: min(fold_len * (i + 1), n)] for i in range(args.n_folds)]
    for idx, fold in enumerate(folds):
        logger.info(f'fold {idx} size = {len(fold)}')
    return folds

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

    folds = make_split(ids, args)
    save_file(args, folds, args.folds_file)

    logger.info('Done')