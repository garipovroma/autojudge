export GSM_TRAIN_LINK=https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/train.jsonl
export GSM_TEST_LINK=https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/test.jsonl

export MODEL0="meta-llama/Llama-3.2-1B-Instruct"
export MODEL1="meta-llama/Llama-3.1-8B-Instruct"

echo "Find important tokens script"

# Data
echo "Creating data dir"
mkdir data

echo "Downloading GSM8k Train"
wget $GSM_TRAIN_LINK --no-check-certificate && mv train.jsonl data/train.jsonl

echo "Downloading GSM8k Test"
wget $GSM_TEST_LINK --no-check-certificate && mv test.jsonl data/test.jsonl

# Will use small gsm subset for short demo
cat data/train.jsonl | head -n 4 > data/train_small.jsonl


# Running the script
echo "Running imporant tokens mining script"

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

# multiple-gpus run

export WORLD_SIZE=2

CUDA_VISIBLE_DEVICES=0 python3 src/find_important_tokens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --gsm8k_train_path $GSM8K_TRAIN \
    --random_seed $RANDOM_SEED \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_folder $OUTPUT_FOLDER \
    --output_file $OUTPUT_FILE \
    --dump_freq $DUMP_FREQ \
    --world_size $WORLD_SIZE \
    --process_id 0 &
CUDA_VISIBLE_DEVICES=1 python3 src/find_important_tokens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --gsm8k_train_path $GSM8K_TRAIN \
    --random_seed $RANDOM_SEED \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_folder $OUTPUT_FOLDER \
    --output_file $OUTPUT_FILE \
    --dump_freq $DUMP_FREQ \
    --world_size $WORLD_SIZE \
    --process_id 1 &
wait

rm output/done*

# Done, result can be found in $OUTPUT_FOLDER/$OUTPUT_FILE_0.pt if one gpu run, or $OUTPUT_FOLDER/$OUTPUT_FILE.pt if multiple gpus run