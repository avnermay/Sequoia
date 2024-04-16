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

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft meta-llama/Llama-2-7b-chat-hf \
    --target meta-llama/Llama-2-70b-chat-hf \
    --dataset oasst \
    --temp 0.6 \
    --top_p 0.9 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-01 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft meta-llama/Llama-2-7b-chat-hf \
    --target meta-llama/Llama-2-70b-chat-hf \
    --dataset oasst \
    --temp 0.05 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-01 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft meta-llama/Llama-2-7b-hf \
    --target meta-llama/Llama-2-70b-hf \
    --dataset wikitext \
    --temp 0.6 \
    --top_p 0.9 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-06 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft meta-llama/Llama-2-7b-hf \
    --target meta-llama/Llama-2-70b-hf \
    --dataset wikitext \
    --temp 0.05 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-06 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft princeton-nlp/Sheared-LLaMA-1.3B \
    --target lmsys/vicuna-33b-v1.3 \
    --dataset oasst \
    --temp 0.6 \
    --top_p 0.9 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-09 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm


CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/measure_acceptance_rates_fast.py \
    --draft princeton-nlp/Sheared-LLaMA-1.3B \
    --target lmsys/vicuna-33b-v1.3 \
    --dataset oasst \
    --temp 0.05 \
    --end 412 \
    --offloading \
    --output_dir /work/avner/results/sequoia"
sbatch --gpus=8 --nodelist=mk-viii-09 \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm
