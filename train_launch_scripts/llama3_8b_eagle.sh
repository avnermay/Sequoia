# Need to download shareGPT

# Avner MK8 paths
# REPO_DIR=/work/avner/git2/Sequoia
# DATA_DIR=/work/avner/data/share_gpt
# WORK_DIR=/work/avner/results/spec_decoding

# Avner MK1 paths
REPO_DIR=/var/cr05_data/avner/git/Sequoia 
DATA_DIR=/var/cr05_data/avner/data/share_gpt
WORK_DIR=/var/cr05_data/avner/results/spec_decoding

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --config_file "${REPO_DIR}/src/fsdp_config_zero3.yaml" \
    ${REPO_DIR}/train.py \
    --bf16 \
    --streaming \
    --draft_type 'eagle' \
    --convert_standalone \
    --train_data_files "${DATA_DIR}/share_gpt_prompt_response_train.jsonl" \
    --valid_data_files "${DATA_DIR}/share_gpt_prompt_response_valid.jsonl" \
    --draft_model_path "${WORK_DIR}/standalone-llama-3-8b-chat-2-layers-0" \
    --target_model_path 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --output_dir "${WORK_DIR}/eagle-llama-3-8b-chat-2-layers-ShareGPT" \
    --shuffle_buffer 5000 \
    --learning_rate 2.0e-5 \
    --seq_length 2048 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --ce_loss_weight 0.0 \
    --distill_loss_weight 0.1 \
    --l1_loss_weight 1.0 \
    --max_steps 10000 \
    --eval_freq 50 \
    --save_freq 1000 \
    --num_warmup_steps 100


/work/avner/results/spec_decoding/standalone-llama-3-8b-chat-2-layers-0