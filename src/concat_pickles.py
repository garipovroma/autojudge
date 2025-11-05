import argparse
import torch
from pathlib import Path
from joblib import load
import numpy as np

import json
import os
from multiprocessing import Pool

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input directory', type=Path)
    parser.add_argument('--out_filename', help='Output filename')
    parser.add_argument('--out_json_path', help='Output for the table')
    return parser.parse_args()


# def is_serializable(value):
#     return isinstance(value int, str, float, )

# key='idx' type(value)=<class 'int'>
# key='question' type(value)=<class 'str'>
# key='raw_answer' type(value)=<class 'str'>
# key='input_tokens' type(value)=<class 'torch.Tensor'>
# key='answer' type(value)=<class 'float'>
# key='raw_pred' type(value)=<class 'str'>
# key='pred' type(value)=<class 'float'>
# key='tp' type(value)=<class 'int'>
# key='gen_tokens' type(value)=<class 'int'>
# key='t_ratio' type(value)=<class 'float'>
# key='d_ratio' type(value)=<class 'float'>
# key='generation' type(value)=<class 'torch.Tensor'>
# key='t_calls' type(value)=<class 'int'>
# key='d_calls' type(value)=<class 'int'>
# key='mean_accept' type(value)=<class 'numpy.float64'>
# key='raw_accepts' type(value)=<class 'list'>
# key='mean_accept_levi' type(value)=<class 'numpy.float64'>
# key='raw_accepts_levi' type(value)=<class 'list'>
# key='mismatches' type(value)=<class 'list'>
# key='generation_str' type(value)=<class 'str'>
# key='thr' type(value)=<class 'float'>



if __name__ == "__main__":
    args = _parse_args()

    tasks_files = list(args.input.glob("**/Task*.pkl"))

    with Pool(processes=min(os.cpu_count(), len(tasks_files))) as pool:
        results_to_reduce = pool.map(load, tasks_files)

    torch.save(obj=results_to_reduce, f=args.out_filename)

    data_to_save_in_json = []
    for result_dict in results_to_reduce:

        dict_to_save = {}

        for key, value in result_dict.items():
            if isinstance(value, torch.Tensor):
                continue
            elif isinstance(value, np.float64):
                dict_to_save[key] = value.item()
            else:
                dict_to_save[key] = value
        data_to_save_in_json.append(dict_to_save)

    with open(args.out_json_path, 'w') as f_write:
        json.dump(data_to_save_in_json, f_write)


    print("Reduced map results!")
