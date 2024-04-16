# models: Llama-2-7b/70b-chat 16 bit
# dataset: oasst
#     - /work/avner/git/speculative/data/oasst_prompts.json
#     - [0:100] - test; [100: 540] - validation
#     - length range 140 - 1206 tokens (test)
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# models: Llama-2-7b/70b 16bit  NON-CHAT!
# dataset: Wikitext [considered C4 but prefer to stick with WT] 
#   - /work/avner/git/speculative/data/wikitext_prompts.json
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# models: Vicuna-33B + SL1.3B
# dataset: oasst, length range (200 - 1200+)
#     - /work/avner/git/speculative/data/oasst_prompts.json
#     - [0:100] - test; [100: 540] - validation
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# [Optional: ]
# models: Llama-2-7b/70b-chat 16 bit
# dataset: mtbench  (80 entries)
#     - sequoia/tests/dataset/mt_bench.jsonl"    
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# draft_models=(
#     "meta-llama/Llama-2-7b-hf"
#     "meta-llama/Llama-2-7b-chat-hf"
#     "princeton-nlp/Sheared-LLaMA-1.3B"
# )
# target_models=(
#     "meta-llama/Llama-2-70b-hf"
#     "meta-llama/Llama-2-70b-chat-hf"
#     "lmsys/vicuna-33b-v1.3"
# )


CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft meta-llama/Llama-2-7b-chat-hf \
    --target meta-llama/Llama-2-70b-chat-hf \
    --dataset oasst \
    --temp 0.6 \
    --top_p 0.9 \
    --algorithm stochastic \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft meta-llama/Llama-2-7b-chat-hf \
    --target meta-llama/Llama-2-70b-chat-hf \
    --dataset oasst \
    --algorithm greedy \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft meta-llama/Llama-2-7b-hf \
    --target meta-llama/Llama-2-70b-hf \
    --dataset wikitext \
    --temp 0.6 \
    --top_p 0.9 \
    --algorithm stochastic \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft meta-llama/Llama-2-7b-hf \
    --target meta-llama/Llama-2-70b-hf \
    --dataset wikitext \
    --algorithm greedy \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft princeton-nlp/Sheared-LLaMA-1.3B \
    --target lmsys/vicuna-33b-v1.3 \
    --dataset oasst \
    --temp 0.6 \
    --top_p 0.9 \
    --algorithm stochastic \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm


CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates.py \
    --draft princeton-nlp/Sheared-LLaMA-1.3B \
    --target lmsys/vicuna-33b-v1.3 \
    --dataset oasst \
    --algorithm greedy \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm


# CMD="/home/avner/anaconda3/envs/env2/bin/accelerate launch \
#     --num_machines 1 \
#     --num_processes 8 \
#     --mixed_precision bf16 \
#     --config_file '/work/avner/git/spec-dec/config/fsdp_config_zero3.yaml' \
#     /work/avner/git/spec-dec/train.py \
#     --bf16 \
#     --streaming \
#     --draft_type 'eagle' \
#     --train_data_files '/work/user_data/extracted/data_2024-01-*_Job.jsonl-mixtral-instruct.jsonl.gz,/work/user_data/extracted/data_2024-02-0*_Job.jsonl-mixtral-instruct.jsonl.gz' \
#     --valid_data_files '/work/user_data/extracted/data_2024-02-10_Job.jsonl-mixtral-instruct.valid.jsonl' \
#     --draft_model_path 'yuhuili/EAGLE-mixtral-instruct-8x7B' \
#     --target_model_path 'mistralai/Mixtral-8x7B-Instruct-v0.1' \
#     --output_dir '/scratch/avner/results/spec_decoding/eagle-mixtral-user-data' \
#     --learning_rate 2.0e-5 \
#     --seq_length 4096 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --max_steps 10000 \
#     --eval_freq 10 \
#     --save_freq 100 \
#     --num_warmup_steps 100"
# sbatch --gpus=8 \
#   --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
#   /work/avner/git/spec-dec/launch_scripts/launch_job.slurm



    # parser.add_argument('--draft', type=str, help='draft model')
    # parser.add_argument('--target', type=str, help='target model')
    # parser.add_argument('--dataset', type=str, default='../dataset/c4_small.json', help='dataset name or path')
    # parser.add_argument('--start', type=int, default=0, help='start')
    # parser.add_argument('--end', type=int, default=200, help='end')
    # parser.add_argument('--temp', type=float, default=0.6, help='temperature')  # T
    # parser.add_argument('--top_p', type=float, default=0.9, help='top_p')  # P
    # parser.add_argument('--algorithm', type=str, default='stochastic',
    #                     choices=['stochastic', 'greedy'], help='algorithm')
    # parser.add_argument('--max_width', type=int, default=16, help='max width')  # W
    # parser.add_argument('--max_length', type=int, default=256, help='max length')  # M
    # parser.add_argument('--Mode', type=str, default='greedy', help='tree mode')
    # parser.add_argument('--offloading', action='store_true')
    # parser.add_argument('--output_dir /work/avner/results/sequoia', type=str, default='../acceptance-rate-vector.pt', help='destination for accepetance rate vector')

