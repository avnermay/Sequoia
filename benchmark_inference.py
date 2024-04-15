import argparse
import os
import json
import time
import torch

from Engine.Engine import GraphInferenceEngineTG
from utils import _make_causal_mask
from tqdm import tqdm, trange


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf", help='model')
    parser.add_argument('--max_length', type=int, default=1152, help='max length')
    parser.add_argument('--prefix_length', type=int, default=128, help='prefix length')
    parser.add_argument('--decode_lengths', type=str, default='1,2,4,8,16,32,64,128,256,512,768,1024', help='dec length')
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup forward passes')
    parser.add_argument('--num_repeats', type=int, default=10, help='number of forward passes to run')
    parser.add_argument('--offloading', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output', help='output directory')
    args = parser.parse_args()
    return args

def benchmark(args):
    device = 'cuda:0'
    dtype = torch.float16
    prefix = torch.randint(low=3, high=30000, size=(1, args.prefix_length), device=device)
    prefix_storage_ids = torch.arange(args.prefix_length, device=device)
    attn_mask = _make_causal_mask((args.max_length, args.max_length), dtype=dtype, device=device)
    attn_mask = attn_mask[None, None, :, :]
    prefix_position_ids = torch.arange(args.prefix_length, device=device).unsqueeze(0)

    decode_lengths = [int(s) for s in args.decode_lengths.split(',')]
    assert max(decode_lengths) + args.prefix_length <= args.max_length
    
    graph_engine = GraphInferenceEngineTG(max_length=args.max_length, model_name_or_path=args.model, dtype=dtype, device=device, offloading=args.offloading)
    graph_engine.inference(input_ids=prefix, storage_ids=prefix_storage_ids, position_ids=prefix_position_ids, attn_mask=attn_mask[..., :args.prefix_length,:args.prefix_length])

    avg_forward_pass_times = []
    for decode_length in decode_lengths:
        input_ids = torch.randint(low=3, high=30000, size=(1, decode_length), device=device)
        storage_ids = torch.arange(decode_length, device=device) + args.prefix_length
        position_ids = storage_ids.clone().unsqueeze(0)
        curr_attn_mask = attn_mask[..., args.prefix_length: args.prefix_length + decode_length,:args.prefix_length + decode_length].clone()

        for _ in trange(args.warmup, desc=f"warmup, {decode_length=}", leave=False):
            graph_engine.inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attn_mask=curr_attn_mask)
            graph_engine.set_kv_len(args.prefix_length)

        torch.cuda.synchronize()
        t1 = time.time()

        for _ in trange(args.num_repeats, desc=f"measuring, {decode_length=}", leave=False):
            graph_engine.inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attn_mask=curr_attn_mask)
            graph_engine.set_kv_len(args.prefix_length)

        torch.cuda.synchronize()
        t2 = time.time()

        avg_forward_pass_time = round((t2 - t1) / args.num_repeats, 6)
        avg_forward_pass_times.append(avg_forward_pass_time)
        print("Max Length :{}, Prefix Length :{}, Decode Length :{}, inference time:{}s"\
              .format(args.max_length, args.prefix_length, decode_length, avg_forward_pass_time))

    print('=============SUMMARY=============')
    print(f'{decode_lengths=}')
    print(f'{avg_forward_pass_times=}')
    result_dict = {
        'decode_lengths': decode_lengths,
        'avg_forward_pass_times': avg_forward_pass_times,
        'args': vars(args),
    }
    print('=================================')
    return result_dict


if __name__ == '__main__':
    args = get_args()
    
    print(f"STARTING TEST: ", args)
    result_dict = benchmark(args)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.model.replace('/','--')}_forward_pass_times.json"
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=4)
