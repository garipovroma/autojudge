export MODEL0="meta-llama/Llama-3.2-1B-Instruct"
export MODEL1="meta-llama/Llama-3.1-8B-Instruct"

echo "Find important tokens script"

# Running the script
echo "Running imporant tokens mining script"

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

# multiple-gpus run

export WORLD_SIZE=2
export TOTAL_GPUS=$WORLD_SIZE

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
    --total_gpus $TOTAL_GPUS \
    --process_id 0 \
    --world_size $WORLD_SIZE &
CUDA_VISIBLE_DEVICES=1 python3 src/find_important_tokens_lcb.py \
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
    --total_gpus $TOTAL_GPUS \
    --process_id 1 \
    --world_size $WORLD_SIZE &
wait

rm output/done*

echo "Done."

# Done, result can be found in $OUTPUT_FOLDER/$OUTPUT_FILE_0.pt