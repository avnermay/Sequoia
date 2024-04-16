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

export ACC_RATE_DIR=/work/avner/results/sequoia/acceptance_rates
export SPEED_DIR=/work/avner/results/sequoia/forward_pass_times
export OUTPUT_DIR=/work/avner/results/sequoia/growmaps

# -ntasks=1 --cpus-per-task=4 --mem=20G

# models: Llama-2-7b/70b-chat 16 bit
# dataset: oasst
CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/oasst_target_meta-llama--Llama-2-70b-chat-hf_draft_meta-llama--Llama-2-7b-chat-hf_temp_0.6_top_p_0.9_growmap \
    --acceptance_rates $ACC_RATE_DIR/oasst_target_meta-llama--Llama-2-70b-chat-hf_draft_meta-llama--Llama-2-7b-chat-hf_temp_0.6_top_p_0.9_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/meta-llama--Llama-2-70b-chat-hf_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/meta-llama--Llama-2-7b-chat-hf_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/oasst_target_meta-llama--Llama-2-70b-chat-hf_draft_meta-llama--Llama-2-7b-chat-hf_temp_0.05_top_p_1.0_growmap \
    --acceptance_rates $ACC_RATE_DIR/oasst_target_meta-llama--Llama-2-70b-chat-hf_draft_meta-llama--Llama-2-7b-chat-hf_temp_0.05_top_p_1.0_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/meta-llama--Llama-2-70b-chat-hf_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/meta-llama--Llama-2-7b-chat-hf_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm


# models: Llama-2-7b/70b 16bit  NON-CHAT!
# dataset: Wikitext 
CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/wikitext_target_meta-llama--Llama-2-70b-hf_draft_meta-llama--Llama-2-7b-hf_temp_0.6_top_p_0.9_growmap \
    --acceptance_rates $ACC_RATE_DIR/wikitext_target_meta-llama--Llama-2-70b-hf_draft_meta-llama--Llama-2-7b-hf_temp_0.6_top_p_0.9_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/meta-llama--Llama-2-70b-hf_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/meta-llama--Llama-2-7b-hf_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/wikitext_target_meta-llama--Llama-2-70b-hf_draft_meta-llama--Llama-2-7b-hf_temp_0.05_top_p_1.0_growmap \
    --acceptance_rates $ACC_RATE_DIR/wikitext_target_meta-llama--Llama-2-70b-hf_draft_meta-llama--Llama-2-7b-hf_temp_0.05_top_p_1.0_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/meta-llama--Llama-2-70b-hf_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/meta-llama--Llama-2-7b-hf_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

# models: Vicuna-33B + SL1.3B
# dataset: oasst
CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/oasst_target_lmsys--vicuna-33b-v1.3_draft_princeton-nlp--Sheared-LLaMA-1.3B_temp_0.6_top_p_0.9_growmap \
    --acceptance_rates $ACC_RATE_DIR/oasst_target_lmsys--vicuna-33b-v1.3_draft_princeton-nlp--Sheared-LLaMA-1.3B_temp_0.6_top_p_0.9_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/lmsys--vicuna-33b-v1.3_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/princeton-nlp--Sheared-LLaMA-1.3B_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm

CMD="/home/avner/anaconda3/envs/env2/bin/python /work/avner/git2/Sequoia/tree_search.py \
    --output_path $OUTPUT_DIR/oasst_target_lmsys--vicuna-33b-v1.3_draft_princeton-nlp--Sheared-LLaMA-1.3B_temp_0.05_top_p_1.0_growmap \
    --acceptance_rates $ACC_RATE_DIR/oasst_target_lmsys--vicuna-33b-v1.3_draft_princeton-nlp--Sheared-LLaMA-1.3B_temp_0.05_top_p_1.0_stochastic_acc_rates_fast.json \
    --target_model_speeds $SPEED_DIR/lmsys--vicuna-33b-v1.3_forward_pass_times.json \
    --draft_model_speeds $SPEED_DIR/princeton-nlp--Sheared-LLaMA-1.3B_forward_pass_times.json"
sbatch \
  --export=HF_HOME,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,DATA_DIR,WANDB_CACHE_DIR,WANDB_DATA_DIR,WANDB_ARTIFACT_DIR,CMD_TO_RUN="$CMD" \
  /work/avner/git/spec-dec/launch_scripts/launch_job.slurm
