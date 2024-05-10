"""Functions for initializing draft models from the layers and embeddings of a target model."""

import numpy as np
import pathlib

# from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def init(output_path, target_model, layer_nums=None, save_tokenizer=True):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    layer_nums = layer_nums or [0, -1]

    config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(target_model, config=config, trust_remote_code=True)
    
    config.num_hidden_layers = len(layer_nums)
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in layer_nums])
    model.save_pretrained(output_path)

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

    # Note: Saving config must come last, otherwise config gets overwritten when the model is saved.
    config.save_pretrained(output_path)
    return model


def init_llama3_8B_2_layer(output_path):
    return init(
        output_path,
        target_model='meta-llama/Meta-Llama-3-8B-Instruct',
        layer_nums=np.linspace(0, 31, num=2, dtype=np.int32).tolist(),
    )


def init_llama3_8B_4_layer(output_path):
    return init(
        output_path,
        target_model='meta-llama/Meta-Llama-3-8B-Instruct',
        layer_nums=np.linspace(0, 31, num=4, dtype=np.int32).tolist(),
    )


if __name__ == '__main__':
    init_llama3_8B_2_layer('/work/avner/results/spec_decoding/standalone-llama-3-8b-chat-2-layers-0')
    init_llama3_8B_2_layer('/work/avner/results/spec_decoding/standalone-llama-3-8b-chat-4-layers-0')
