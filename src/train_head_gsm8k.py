from typing import Union, Sequence, Tuple, Dict, Literal
import json
import os

from argparse import ArgumentParser
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import pickle

import sys
sys.path.append('.')

from core_utils import make_setup_slice_mapper

def make_X_y(data: list[Dict]) -> Tuple[np.array, np.array]: 
    X = []
    y = []

    n_skip = 0
    skip_indexes = set()
    for sample_idx, sample_dict in enumerate(data):
        if 'hiddens' in sample_dict and 'changed_token_indices' in sample_dict:
            assert len(sample_dict['changed_token_indices']) == len(sample_dict['hiddens'])
            x_ = [i.numpy() if hasattr(i, "numpy") else np.array(i) for i in sample_dict['hiddens']]
            y_ = [int(i[1]) for i in sample_dict['changed_token_indices']]
            X.extend(x_)
            y.extend(y_)
        else:
            n_skip += 1
            skip_indexes.add(sample_idx)
            print(f"n_skips+=1 ==> {n_skip=}")

        
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_head_and_search_best_hparam(C: float | None = None):

    C_grid = [C] if C is not None else (10**0.5) ** -np.arange(1, 15)[::-1]
    C_grid = C_grid.tolist() + [1, 3, 10, 50, 100]
    # C_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    dataframes = []

    best_C = None
    best_metric_roc_auc = -1
    best_dataframe = None

    best_model = None
    for c_idx, C in enumerate(tqdm(C_grid)):
        # TODO: remove this
        # if c_idx >= 3:
        #     break
        print('#####' * 6, end='')
        print(f' {C=:.5f}, {c_idx=} ', end='')
        print('#####' * 6)    

        model_ = LogisticRegression(C=C)

        model_.fit(X_train, y_train)

        train_probs = model_.predict_proba(X_train)[:, 1]
        val_probs = model_.predict_proba(X_val)[:, 1]
        quantiles =  np.linspace(0, 1, 26)

        roc_auc_score_train = 0.0 if len(set(y_val)) == 1 else roc_auc_score(y_train, train_probs)
        roc_auc_score_val = 0.0 if len(set(y_val)) == 1 else roc_auc_score(y_val, val_probs)
        metrics = {
            'roc_auc_train': [],
            'roc_auc_val': [],

            'acc_train': [],
            'recall_train': [],
            'precision_train': [],

            'acc_val': [],
            'recall_val': [],
            'precision_val': [],

            'neg_rate_val': [],

            'q': [],
            'thr': []
        }
        quantiles = list(quantiles)
        quantiles = sorted(quantiles)
        for quantile in quantiles:
            thr = np.quantile(val_probs, quantile)
            
            train_pred = train_probs > thr
            val_pred = val_probs > thr

            metrics['acc_train'].append(accuracy_score(y_train, train_pred))
            metrics['recall_train'].append(recall_score(y_train, train_pred))
            metrics['precision_train'].append(precision_score(y_train, train_pred))

            metrics['acc_val'].append(accuracy_score(y_val, val_pred))
            metrics['recall_val'].append(recall_score(y_val, val_pred))
            metrics['precision_val'].append(precision_score(y_val, val_pred))

            metrics['roc_auc_train'].append(roc_auc_score_train)
            metrics['roc_auc_val'].append(roc_auc_score_val)

            metrics['q'].append(quantile)
            metrics['thr'].append(thr)

            metrics['neg_rate_val'].append(1 - (val_probs > thr).mean())

        metrics_df = pd.DataFrame(metrics).set_index('q')
        metrics_df['C'] = C
        metrics_df['c_idx'] = c_idx
    
        print(metrics_df[['roc_auc_train', 'roc_auc_val', 'C']].iloc[0])
        print()
        if roc_auc_score_val > best_metric_roc_auc:
            best_metric_roc_auc = roc_auc_score_val
            best_C = C
            best_dataframe = metrics_df
            best_model = model_

        dataframes.append(metrics_df)

    dataframes = pd.concat(dataframes)

    print(f"{best_C=} {best_metric_roc_auc=}")

    return dataframes, best_dataframe, best_model, best_C

if __name__ == '__main__':

    parser = ArgumentParser()
    # TODO
    parser.add_argument('--random_seed', type=int, default=52)
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--resulting_table_path', type=str, default=None)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--draft_model', type=str)
    parser.add_argument("--setup", default='DD-DT', choices=['DD-DT', 'DT'])
    parser.add_argument("--remove_target_nones", action='store_true')
    parser.add_argument("--remove_draft_nones", action='store_true')
    parser.add_argument("--convert_to_vllm", action='store_true')
    parser.add_argument("--train_on_all_data", action='store_true')
    parser.add_argument("--uniform_thresholds", action='store_true')
    parser.add_argument("--remove_arena_duplicates", action='store_true')
    parser.add_argument("--concat_kostyl", action='store_true')
    parser.add_argument("--val_fold", type=int, default=-1)
    parser.add_argument("--judge_decoding_setup", action="store_true", help="We use this step to train classifier in order to implementa manual annotation Judge Decoding setup")

    args = parser.parse_args()

    if args.convert_to_vllm:
        assert args.setup == 'DT', "autojudge vllm inference works in DT setup only"


    data = torch.load(args.data_path)
    print(type(data))

    if args.concat_kostyl:
        data = sum(data, [])

    if isinstance(data, dict) and 'data' in data:  # NOTE handle interrupted merging of processes
        data = data['data']
    size_before_filtering = len(data)
    print(f'Size of data: {len(data)}')
    data = [i for i in data if type(i) == dict]
    print(f'Size of data: {len(data)}, ∆size = {len(data) - size_before_filtering}')

    if args.remove_target_nones:
        print(f'Size before target nones filtering: {len(data)}')
        data = [i for i in data if i['target_answer'] is not None and \
            type(i['target_answer']) == list and \
            type(i['target_answer'][0]) == str and \
            'none' not in i['target_answer'][0].lower() \
        ]
        print(f'Size after target nones filtering: {len(data)}')

    if args.remove_draft_nones:
        print(f'Size before draft nones filtering: {len(data)}')
        data = [i for i in data if i['draft_answer'] is not None and \
            type(i['draft_answer']) == list and \
            type(i['draft_answer'][0]) == str and \
            'none' not in i['draft_answer'][0].lower() \
        ]
        print(f'Size after draft nones filtering: {len(data)}')

    if args.remove_arena_duplicates:
        print(f'Size before arena duplicates filtering: {len(data)}')
        already_appeared = set()
        samples = []

        for i in data:
            if 'draft_gen_str' in i and i['draft_gen_str'] in already_appeared:
                continue
            already_appeared.add(i.get('draft_gen_str', "PLACEHOLDER"))
            samples.append(i)
        data = samples
        del already_appeared

        print(f'Size after arena duplicates filtering: {len(data)}')

    np.random.seed(args.random_seed)
    np.random.shuffle(data)

    first_test_sample_idx = int(len(data) * args.train_size) + 1

    train_data = data[:first_test_sample_idx]
    test_data = data[first_test_sample_idx:]

    if args.val_fold != -1:
        assert args.val_fold in range(5)
        assert not args.train_on_all_data
        folds = []
        root_dir = os.path.dirname(os.path.dirname(__file__))
        train_ids = []
        val_ids = []
        for fold_id in range(5):
            fold = torch.load(f"data/arena_hard_auto/creative_writing_fold_{fold_id}.pt")
            folds.append(fold)
            if fold_id == args.val_fold:
                val_ids.extend(fold)
            else:
                train_ids.extend(fold)
        
        train_ids = set(train_ids)
        val_ids = set(val_ids)
        intersection_sz = len(train_ids & val_ids)

        assert intersection_sz == 0, f"Data contamination found, train/val {intesection_sz=}"

        train_data = [i for i in data if i['id'] in train_ids]
        test_data = [i for i in data if i['id'] in val_ids]
        print(f'FOLD is specified, {args.val_fold=}')
        print(f'train_size = {len(train_data)}, val_size = {len(test_data)}')

    X_train, y_train = make_X_y(train_data)
    X_val, y_val = make_X_y(test_data)
    if not args.judge_decoding_setup:

        SETUP_SLICE_MAP = make_setup_slice_mapper(args.draft_model, args.target_model)

        setup_slice = SETUP_SLICE_MAP[args.setup]
        X_train = X_train[:, setup_slice]
        X_val = X_val[:, setup_slice]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(X_train.shape, X_val.shape)
    print(y_train.mean(), y_val.mean())

    all_dataframes_1_token, best_datarame_1_token, best_head, best_c = train_head_and_search_best_hparam()

    if args.uniform_thresholds:
        thresholds = np.linspace(0, 1, 26)
    else:
        thresholds = best_datarame_1_token['thr'].values.tolist()

    best_datarame_1_token_in_dicts = best_datarame_1_token.to_dict('records')

    if bool(args.train_on_all_data):
        print('Attention! Training on all data, previous metrics are obtained using 90/10 splits')
        X_train, y_train = make_X_y(data)
        if not args.judge_decoding_setup:
            X_train = X_train[:, setup_slice]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        best_head = LogisticRegression(C=best_c).fit(X_train, y_train) 

    print('Head is trained, here are the results: TODO')
    dump_dict = dict(model=best_head, scaler=scaler, thresholds=thresholds)
    if args.convert_to_vllm:

        target_hidden_size = setup_slice.stop - setup_slice.start
        head_dict = dict(
            mean=scaler.mean_[-target_hidden_size:],
            scale=scaler.scale_[-target_hidden_size:],
            weights=best_head.coef_[0][-target_hidden_size:],
            bias=best_head.intercept_[-target_hidden_size:],
            thr=0.25, # placeholder, we don't use this further,
            thresholds=thresholds,  # needed for pareto-curve
        )

        with open(args.checkpoint_path, 'wb') as f:
            pickle.dump(head_dict, f)

    else:
        with open(args.checkpoint_path, 'wb') as f:
            pickle.dump(dump_dict, f)

