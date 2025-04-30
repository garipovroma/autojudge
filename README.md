# AutoJudge: Judge Decoding Without Manual Annotation

<a href='https://arxiv.org/abs/2504.20039'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 

Official PyTorch implementation for the paper `AutoJudge: Judge Decoding Without Manual Annotation`

## 🚀 Running the code

Our work proposes an algorithm to mine important token mismatches dataset. We calculate hidden states for them, train a lightweight classifier on them and then do evaluations.

To obtain results using our approach you need to:
1. Run dataset mining script
2. Run hidden states collection script
3. Train a classifier
4. Run evaluations

** 📦 Mined datasets will be published on HuggingFace shortly, so you can skip first two steps once released! **

### 🛠️ Getting started

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### ⛏️ Dataset mining 💎

Here we provide a small snippet of how to run dataset mining for GSM8K and LiveCodeBench, for the detailed instructions including multiple-gpu run please refer to the [`find_important_tokens_gsm8k.sh`](https://github.com/garipovroma/autojudge/blob/master/scripts/find_important_tokens_gsm8k.sh) and [`find_important_tokens_lcb.sh`](https://github.com/garipovroma/autojudge/blob/master/scripts/find_important_tokens_lcb.sh) scripts.

#### 📐 GSM8K 🔢


```bash

export TORCH_DTYPE=auto
export GSM8K_TRAIN=data/train_small.jsonl # replace by data/train.jsonl for full run
export RANDOM_SEED=42
export MAX_NEW_TOKENS=2048
export OUTPUT_FOLDER=output
export OUTPUT_FILE=important_tokens
export DUMP_FREQ=64

mkdir $OUTPUT_FOLDER

# one-gpu run

CUDA_VISIBLE_DEVICES=0 python3 src/find_important_tokens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --gsm8k_train_path $GSM8K_TRAIN \
    --random_seed $RANDOM_SEED \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_folder $OUTPUT_FOLDER \
    --output_file $OUTPUT_FILE \
    --dump_freq $DUMP_FREQ

rm output/done*
```

#### 💻 LiveCodeBench 📄

```bash
export TORCH_DTYPE=auto
export GSM8K_TRAIN=data/train_small.jsonl # replace by data/train.jsonl for full run
export RANDOM_SEED=42
export MAX_NEW_TOKENS=2048
export OUTPUT_FOLDER=output
export OUTPUT_FILE=important_tokens_lcb
export DUMP_FREQ=64
export NUM_PROCESS_EVALUATE=64
export N_TASKS=2 # will use 2 tasks for short demo, set 880 for full lcb release_v5 dataset
export TOTAL_GPUS=1

mkdir $OUTPUT_FOLDER

# one-gpu run

CUDA_VISIBLE_DEVICES=0 python3 src/find_important_tokens_lcb.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --random_seed $RANDOM_SEED \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_folder $OUTPUT_FOLDER \
    --output_file $OUTPUT_FILE \
    --dump_freq $DUMP_FREQ \
    --n_tasks $N_TASKS \
    --num_process_evaluate $NUM_PROCESS_EVALUATE \
    --total_gpus $TOTAL_GPUS
```    

### 🧮 Calculating hidden states ⚙️

For the full script including multiple-gpus run please refer to the [`calc_hiddens.sh`](https://github.com/garipovroma/autojudge/blob/master/scripts/calc_hiddens.sh) script.

```bash
export MODEL0="meta-llama/Llama-3.2-1B-Instruct"
export MODEL1="meta-llama/Llama-3.1-8B-Instruct"
export TORCH_DTYPE=auto
export BATCH_SIZE=8
export DATA_FILE=output/important_tokens.pt
export OUTPUT_PATH=output/important_tokens_with_hiddens
export SAVE_FREQ=128
export N_PROCESSES=1

# single gpu run
CUDA_VISIBLE_DEVICES=0 python src/calc_hiddens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --batch_size $BATCH_SIZE \
    --data_file $DATA_FILE \
    --output_path $OUTPUT_PATH \
    --save_freq $SAVE_FREQ \
    --n_processes $N_PROCESSES \
    --process_id 0 
```

### 🧠 Training a classifier 🎯

Classifier training snippet can be found in [`train.ipynb`📒 ](https://github.com/garipovroma/autojudge/blob/master/notebooks/train.ipynb).

Coming soon.

### 📊 Evaluations 📝

Coming soon.

## Citing us

If you found this work useful, please consider citing:

```
@misc{garipov2025autojudgejudgedecodingmanual,
      title={AutoJudge: Judge Decoding Without Manual Annotation}, 
      author={Roman Garipov and Fedor Velikonivtsev and Ruslan Svirschevski and Vage Egiazarian and Max Ryabinin},
      year={2025},
      eprint={2504.20039},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.20039}, 
}
```

