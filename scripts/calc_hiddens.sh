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


# multiple gpus run
export N_PROCESSES=2

CUDA_VISIBLE_DEVICES=0 python src/calc_hiddens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --batch_size $BATCH_SIZE \
    --data_file $DATA_FILE \
    --output_path $OUTPUT_PATH \
    --save_freq $SAVE_FREQ \
    --n_processes $N_PROCESSES \
    --process_id 0 &
CUDA_VISIBLE_DEVICES=0 python src/calc_hiddens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --batch_size $BATCH_SIZE \
    --data_file $DATA_FILE \
    --output_path $OUTPUT_PATH \
    --save_freq $SAVE_FREQ \
    --n_processes $N_PROCESSES \
    --process_id 1 &
wait