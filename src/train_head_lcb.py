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
sys.path.append('..')
from core_utils import make_setup_slice_mapper

from src.livecodebench_v5 import load_code_generation_dataset

def make_X_y(data: list[Dict]) -> Tuple[np.array, np.array]: 
    X = []
    y = []

    n_skip = 0
    skip_indexes = set()
    for sample_idx, sample_dict in enumerate(data):
        if 'hiddens' in sample_dict:
            assert len(sample_dict['changed_token_indices']) == len(sample_dict['hiddens'])
            x_ = [i.numpy() for i in sample_dict['hiddens']]
            y_ = [int(i[1]) for i in sample_dict['changed_token_indices']]
            X.extend(x_)
            y.extend(y_)
        else:
            n_skip += 1
            skip_indexes.add(sample_idx)

        
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_head_and_search_best_hparam(X_train, y_train, X_val, y_val, C: float | None = None):

    C_grid = [C] if C is not None else (10**0.5) ** -np.arange(1, 15)[::-1]

    dataframes = []

    best_C = None
    best_metric_roc_auc = -1
    best_dataframe = None

    best_model = None
    for c_idx, C in enumerate(tqdm(C_grid)):
        
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

    return dataframes, best_dataframe, best_model

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=52)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--folds_path', type=str)    
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--resulting_table_path', type=str, default=None)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--draft_model', type=str)
    parser.add_argument("--setup", default='DD-DT', choices=['DD-DT', 'DT'])
    parser.add_argument("--convert_to_vllm", action='store_true')

    args = parser.parse_args()

    if args.convert_to_vllm:
        assert args.setup == 'DT', "autojudge vllm inference works in DT setup only"

    folds = torch.load(args.folds_path)
    folds = [set(i) for i in folds]

    data = torch.load(args.data_path)

    print(f'Size of data: {len(data)}')

    train_sets = []
    val_sets = []
    scalers = []


    SETUP_SLICE_MAP = make_setup_slice_mapper(args.draft_model, args.target_model)

    setup_slice = SETUP_SLICE_MAP[args.setup]

    for val_fold_id, val_fold in enumerate(folds):
        test_data = [i for i in data if i['question_id'] in val_fold]
        train_data = [i for i in data if i['question_id'] not in val_fold]
        print(f'{val_fold_id=}')
        print(list(val_fold)[:10])
        

        X_train, y_train = make_X_y(train_data)
        X_val, y_val = make_X_y(test_data)

        X_train = X_train[:, setup_slice]
        X_val = X_val[:, setup_slice]

        print(X_train.shape, X_val.shape)
        print(y_train.mean(), y_val.mean())
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        train_sets.append((X_train, y_train))
        val_sets.append((X_val, y_val))
        scalers.append(scaler)



    checkpoints = []

    for val_fold_id, ((X_train, y_train), (X_val, y_val), scaler) in enumerate(zip(train_sets, val_sets, scalers)):

        all_dataframes_1_token, best_datarame_1_token, best_head_1_token = train_head_and_search_best_hparam(X_train, y_train, X_val, y_val)
        thresholds = best_datarame_1_token['thr'].values.tolist()

        best_head = best_head_1_token

        target_hidden_size = setup_slice.stop - setup_slice.start
        head_dict = dict(
            mean=scaler.mean_[-target_hidden_size:],
            scale=scaler.scale_[-target_hidden_size:],
            weights=best_head.coef_[0][-target_hidden_size:],
            bias=best_head.intercept_[-target_hidden_size:],
            thr=0.25 # placeholder, we don't use this further
        )

        if args.convert_to_vllm:
            checkpoints.append(head_dict)
        else:
            checkpoints.append(
                dict(model=best_head_1_token, scaler=scaler, thresholds=thresholds)
            )

    with open(args.checkpoint_path, 'wb') as f:
        pickle.dump(checkpoints, f)
