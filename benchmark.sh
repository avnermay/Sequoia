# models: Llama-2-7b/70b-chat 16 bit
# dataset: oasst
#     - https://github.com/poedator/speculative/blob/main/data/oasst_prompts.json
#     - [0:100] - test; [100: 540] - validation
#     - length range 140 - 1206 tokens (test)
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# models: Llama-2-7b/70b 16bit  NON-CHAT!
# dataset: Wikitext [considered C4 but prefer to stick with WT] 
#   - https://github.com/poedator/speculative/blob/main/data/wikitext_prompts.json
# temperatures:
#     - t 0.6, top_p 0.9
#     - t 0.0

# models: Vicuna-33B + SL1.3B
# dataset: oasst, length range (200 - 1200+)
#     - https://github.com/poedator/speculative/blob/main/data/oasst_prompts.json
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

models=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    "lmsys/vicuna-33b-v1.3"
    "princeton-nlp/Sheared-LLaMA-1.3B"
)

SEQUOIA_PATH=/work/avner/git/Sequoia
OUTPUT_DIR=/home/avner/sequoia

for model in "${models[@]}"; do
    python $SEQUOIA_PATH/benchmark_inference.py --model $model --output_dir $OUTPUT_DIR
done
