import argparse
import json
import os
from tqdm import tqdm

from accelerate import Accelerator
from transformers import LlamaForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import softmax

from data_converter import convert_dataset, convert_cnn_dataset, convert_jsonl_file, convert_wiki_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=str, help='draft model')
    parser.add_argument('--target', type=str, help='target model')
    parser.add_argument('--dataset', type=str, default='dataset/c4_small.json', help='dataset name or path')
    parser.add_argument('--start', type=int, default=0, help='start')
    parser.add_argument('--end', type=int, default=200, help='end')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')  # T
    parser.add_argument('--top_p', type=float, default=1.0, help='top_p')  # P
    parser.add_argument('--max_width', type=int, default=16, help='max width')  # W
    parser.add_argument('--max_length', type=int, default=256, help='max length')  # M
    # TODO: Delete offloading param. It is just here for compatibility with 
    parser.add_argument('--offloading', action='store_true')
    parser.add_argument('--output_dir', type=str, default='.', help='destination directory for acceptance rate vector')
    parser.add_argument('--output_file', type=str, default=None, help='filename for acceptance rate vector')
    args = parser.parse_args()
    print(f'{args}=')
    return args


def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)    
    return residual


def evaluate(target_model : LlamaForCausalLM, draft_model: LlamaForCausalLM, dataloader: DataLoader, k:int, T=0.6, top_p=0.9, draft_top_p=0.99):
    num_eval_steps = len(dataloader)
    acceptance_rate = torch.zeros(k)
    num_samples = 0
    draft_model_prob = []
    token_accept_rate = []
    sampled_token_sets = []
    real_budget = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            target_logits : torch.Tensor = target_model(**batch).logits
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                target_logits[indices_to_remove] = float('-inf')

            
            draft_logits : torch.Tensor = draft_model(**batch).logits
            target_prob = softmax(target_logits / T, dim=-1).squeeze(0)
            q = softmax(draft_logits / T, dim=-1).squeeze(0)
            
            for i in range(128, target_prob.shape[0]):
                token_acceptance_rate = torch.zeros(k)
                draft_tokens = []
                if batch['labels'][0][i] == -100 or batch['labels'][0][i] == 0: continue
                num_samples = num_samples + 1
                token_target_prob = target_prob[i]
                # token_draft_prob = q[i]
                #draft_model_prob.append(q[i].cpu())
                token_draft_logits = draft_logits[0][i]

                if draft_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_draft_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                    filter = cumulative_probs > draft_top_p
                    filter[..., 1:] = filter[..., :-1].clone()
                    filter[..., 0] = 0
                    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                    token_draft_logits[indices_to_remove] = float('-inf')

                token_draft_prob = softmax(token_draft_logits / T, dim=-1).squeeze(0)
                sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                draft_tokens.append(sampled_token.item())
                real_budget = real_budget + 1
                token_acceptance_rate[0] = min(1.0, (token_target_prob[sampled_token]/ token_draft_prob[sampled_token]))

                token_target_prob = get_residual(token_target_prob, token_draft_prob)
                
                
                for j in range(k-1):
                    token_draft_logits[sampled_token] = - torch.inf
                    token_draft_prob = softmax(token_draft_logits / (T), dim=-1).squeeze(0)
                    if torch.isnan(token_draft_prob).long().sum() >= 1:
                        break
                    token_draft_prob = token_draft_prob / token_draft_prob.sum(-1)
                    sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                    draft_tokens.append(sampled_token.item())
                    real_budget = real_budget + 1
                    branch_token_acceptance_rate = min(1, token_target_prob[sampled_token]/ token_draft_prob[sampled_token])
                    token_acceptance_rate[j+1] = (1 - token_acceptance_rate.sum()) * branch_token_acceptance_rate
                    
                    token_target_prob = get_residual(token_target_prob, token_draft_prob)
                acceptance_rate = acceptance_rate + token_acceptance_rate
                token_accept_rate.append(token_acceptance_rate.cpu())
                sampled_token_sets.append(draft_tokens)
                draft_model_prob.append(q[i][draft_tokens].cpu()) 
    return acceptance_rate / num_samples


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


def get_output_filename(args):
    output_dir = args.output_dir or '.'
    if args.output_file is None:
        target_str = f'target_{args.target.replace('/', '--')}'
        draft_str = f'draft_{args.draft.replace('/', '--')}'
        temp_str = f'temp_{args.temp}'
        top_p_str = f'top_p_{args.top_p}'
        output_file = f'{args.dataset}_{target_str}_{draft_str}_{temp_str}_{top_p_str}_acc_rates_fast.json'
    else:
        output_file = args.output_file
    return f'{output_dir}/{output_file}'


def save_results(acceptance_rate, output_file):
    x = torch.zeros(len(acceptance_rate) + 1)
    x[1:] = acceptance_rate
    results = {
        'acceptance_rates': x.cpu().numpy().tolist(),
    }
    print(f'{results=}')
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        torch.save(x, output_file)



if __name__ == '__main__':
    args = get_args()
    dataloader = get_tokenized_dataloader(args.dataset, args.start, args.end)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, device_map="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16, device_map="cuda:0")
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)

    acceptance_rate_list = [0]
    branch_acceptance_rate_list = [0]

    acceptance_rate = evaluate(target_model, draft_model, dataloader, k=args.max_width, T=args.temp, top_p=args.top_p, draft_top_p=args.top_p)
    output_filename = get_output_filename(args)
    save_results(acceptance_rate, output_filename)
