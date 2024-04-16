import argparse
import json
import os
from tqdm import tqdm

from accelerate import Accelerator
from datasets import load_from_disk
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval, convert_jsonl_file, convert_dataset
from Tree.SpecTree import SpecTreeTest
from Tree.GreedyTree import GreedyTreeTest
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=str, help='draft model')
    parser.add_argument('--target', type=str, help='target model')
    parser.add_argument('--dataset', type=str, default='dataset/c4_small.json', help='dataset name or path')
    parser.add_argument('--start', type=int, default=0, help='start')
    parser.add_argument('--end', type=int, default=200, help='end')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')  # T
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p')  # P
    parser.add_argument('--algorithm', type=str, default='stochastic',
                        choices=['stochastic', 'greedy'], help='algorithm')
    parser.add_argument('--max_width', type=int, default=16, help='max width')  # W
    parser.add_argument('--max_length', type=int, default=256, help='max length')  # M
    parser.add_argument('--offloading', action='store_true')
    parser.add_argument('--output_dir', type=str, default='.', help='destination directory for acceptance rate vector')
    parser.add_argument('--output_file', type=str, default=None, help='filename for acceptance rate vector')
    args = parser.parse_args()
    print(f'{args}=')
    return args


def get_output_filename(args):
    output_dir = args.output_dir or '.'
    if args.output_file is None:
        target_str = f'target_{args.target.replace('/', '--')}'
        draft_str = f'draft_{args.draft.replace('/', '--')}'
        if args.algorithm == 'stochastic':
            temp_str = f'temp_{args.temp}'
            top_p_str = f'top_p_{args.top_p}'
            output_file = f'{args.dataset}_{target_str}_{draft_str}_{temp_str}_{top_p_str}_stochastic_acc_rates.json'
        else:
            output_file = f'{args.dataset}_{target_str}_{draft_str}_greedy_acc_rates.json'
    else:
        output_file = args.output_file
    return f'{output_dir}/{output_file}'


def save_results(branch_prob, output_branch_prob, num_decoding_steps, num_large_model_steps, step, output_file):
    output_branch_prob[1:] = branch_prob / branch_prob.sum(dim=-1)
    results = {
        'branch_prob': branch_prob.cpu().numpy().tolist(),
        'acceptance_rates': output_branch_prob.cpu().numpy().tolist(),
        'num_decoding_steps': num_decoding_steps,
        'num_large_model_steps': num_large_model_steps,
        'step': step,
    }
    print(f'{results=}')
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        torch.save(output_branch_prob, output_file)


def simulation_stochastic(
        target_model : GraphInferenceEngineTG,
        draft_model: GraphInferenceEngine,
        dataloader: DataLoader,
        output_file: str,
        temp=0.6,
        top_p=0.9,
        max_width=4, 
        max_length=256,
    ):
    max_input_length = max_length + max_width
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    dtype = torch.float16
    attn_mask = torch.full((max_input_length, max_input_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_input_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_input_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_input_length).long().to('cuda:0')
    position_ids = torch.zeros(max_input_length).long().to('cuda:0')
    branch_prob = torch.zeros(max_width + 1).to('cuda:0')
    output_branch_prob = torch.zeros(max_width + 2).to('cuda:0')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            while input_ids.shape[1] < max_length and not terminate:
                attn_mask.fill_(torch.finfo(dtype).min)
                spectree = SpecTreeTest(
                    prefix=input_ids.squeeze(0), device='cuda:0', temperature=temp,
                    top_p=top_p,
                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_input_length,
                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer,
                    parents_buffer = parents_buffer,
                    position_ids = position_ids, max_width=max_width,
                )
                
                
                valid_tokens, draft_kv_len, target_kv_len,  b, terminate = spectree.verify(benchmark=True)
                branch_prob[b] += 1
                
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true():
                    terminate = True

            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)

            # We save intermediate results after every example.
            save_results(branch_prob, output_branch_prob, num_decoding_steps, num_large_model_steps, step, output_file)

    return num_decoding_steps / num_large_model_steps


def simulation_greedy(
        target_model : GraphInferenceEngineTG,
        draft_model: GraphInferenceEngine, 
        dataloader: DataLoader,
        output_file: str,
        max_width=4,
        max_length=256,
    ):
    max_input_length = max_length + max_width
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    dtype = torch.float16
    attn_mask = torch.full((max_input_length, max_input_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_input_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_input_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_input_length).long().to('cuda:0')
    position_ids = torch.zeros(max_input_length).long().to('cuda:0')
    branch_prob = torch.zeros(max_width + 1).to('cuda:0')
    output_branch_prob = torch.zeros(max_width + 2).to('cuda:0')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            while input_ids.shape[1] < max_length and terminate == False:
                attn_mask.fill_(torch.finfo(dtype).min)
                spectree = GreedyTreeTest(
                    prefix=input_ids.squeeze(0), device='cuda:0',
                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_input_length,
                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer,
                    parents_buffer = parents_buffer,
                    position_ids = position_ids, max_width=max_width,
                )
                
                valid_tokens, draft_kv_len, target_kv_len,  b, terminate = spectree.verify(benchmark=True)
                initial_size = input_ids.shape[1]
                input_ids = valid_tokens.unsqueeze(0)
                
                
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true(): terminate = True
                if not terminate:
                    branch_prob[b] += 1
                    num_decoding_steps += (valid_tokens.shape[0] - initial_size)
                    num_large_model_steps += 1

            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)

            # We save intermediate results after every example.
            save_results(branch_prob, output_branch_prob, num_decoding_steps, num_large_model_steps, step, output_file)

    return num_decoding_steps / num_large_model_steps


def get_tokenized_dataloader(dataset, start, end):
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset == 'openwebtext':
        tokenized_dataset_eval = load_from_disk(f'{current_dir}/dataset/openwebtext_eval')
    elif dataset == 'wiki':
        tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer)
    elif dataset == 'cnn':
        tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer)
    elif dataset == 'wikitext':
        tokenized_dataset_eval = convert_jsonl_file(tokenizer, f'{current_dir}/dataset/wikitext_dev.jsonl')
    elif dataset == 'oasst':
        tokenized_dataset_eval = convert_jsonl_file(tokenizer, f'{current_dir}/dataset/oasst_dev.jsonl')
    elif dataset == 'c4_small':
        tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer, file_path=f'{current_dir}/dataset/c4_small.json')
    else:
        raise ValueError(f'Unsupported dataset: {dataset}.')
    
    tokenized_dataset_eval = tokenized_dataset_eval.select(list(range(start, end)))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)
    return dataloader


if __name__ == '__main__':
    args = get_args()
    dataloader = get_tokenized_dataloader(args.dataset, args.start, args.end)
    max_input_length = args.max_length + args.max_width
    draft_model = GraphInferenceEngine(max_length=max_input_length, model_name_or_path=args.draft, dtype=torch.float16, device='cuda:0')
    target_model = GraphInferenceEngineTG(max_length=max_input_length, model_name_or_path=args.target, dtype=torch.float16, device='cuda:0', offloading=args.offloading)
    graph_capture_list = list(range(1, 129))
    draft_model.initialize_cuda_graph(graph_capture_list)

    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)

    output_file = get_output_filename(args)
    if args.algorithm == 'stochastic':
        simulation_stochastic(
            target_model=target_model, draft_model=draft_model, dataloader=dataloader, output_file=output_file,
            temp=args.temp, top_p=args.top_p, max_width=args.max_width, max_length=args.max_length,
        )

    elif args.algorithm == 'greedy':
        simulation_greedy(
            target_model=target_model, draft_model=draft_model, dataloader=dataloader, output_file=output_file,
            max_width=args.max_width, max_length=args.max_length,
        )
