import argparse
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='/work/avner/results/spec_decoding/grow_map.pt')
    parser.add_argument('--acceptance_rate_vector', type=str, default=None,
                        help='Pytorch tensor with acceptance rates as function of tree branch width')
    # Hacky way of taking output of spec-dec/eval.py as input to this script.
    parser.add_argument('--eval_results_json', type=str,
                        default='/work/avner/results/spec_decoding/evals/march_eval_unpacked_target_mixtral_draft_eagle-ft-8B-userdata.json')
    parser.add_argument('--max_depth', type=int, default=16)
    parser.add_argument('--max_budget', type=int, default=256)
    parser.add_argument('--draft_time', type=float, default=2.0,
                        help='Time for a forward pass of draft model, batch size=1')
    parser.add_argument('--target_times', type=float, nargs='+',
                        default=[47.0, 80.0, 85.0, 90.0, 94.0, 98.0, 100.0, 102.0, 115.0],
                        help='Target verification time for each tree size in `valid_budgets`.')
    parser.add_argument('--valid_budgets', type=int, nargs='+',
                        default=[ 1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help='List of budgets to consider for optimal tree size')
    args = parser.parse_args()
    assert len(args.valid_budgets) == len(args.target_times)
    return args


def get_alpha(path):
    with open(path, 'r') as f:
        eagle_ft_eval_results = json.load(f)
    acceptance_rates = eagle_ft_eval_results['acceptance_rate']
    alpha = np.array([0,0] + acceptance_rates[2])
    alpha = alpha[1:] - alpha[:-1]
    return alpha


def dynamic_program(alpha, max_budget, max_depth, max_branch):
    T = torch.zeros((max_budget + 1, max_depth + 1, max_branch + 1)).fill_(-torch.inf)
    branch_map = {}
    for l in range(1, max_depth + 1):
        for b in range(0, max_branch + 1):
            if b == 0:
                T[1, l, b] = 1.0
                branch_map[(1,l,b)] = []

    for m in tqdm(range(2, max_budget + 1)):
        for l in range(2, max_depth + 1):
            T[m, l, 1] = 1 + alpha[1] * T[m-1, l-1].max()
            if T[m, l, 1] > 0:
                branch_map[(m,l,1)] = [(m-1, l-1, T[m-1, l-1].argmax(dim=0).item())]
            for b in range(2, max_branch + 1):
                max_value = -torch.inf
                for y in range(1, m):
                    new_value = T[y, l, b-1] + alpha[b] * T[m-y, l-1].max()
                    if new_value > max_value:
                        max_value = new_value
                        new_y = y
                    max_value = max(max_value, new_value)
                T[m, l, b] = max_value
                if max_value >= 0:
                    new_branch = T[m-new_y, l-1].argmax(dim=0).item()
                    new_list :list = deepcopy(branch_map[(new_y, l, b-1)])
                    new_list.append((m-new_y, l-1, new_branch))
                    branch_map[(m,l,b)] = new_list
    results = T.max(dim=2).values
    return results, T, branch_map


def get_optimal_tree_size_and_depth(draft_inference_time, target_verify_time, valid_budget, results):
    best_speedup = 0.0
    best_tree_size = 1
    best_tree_depth = 1
    best_acc_length = 1
    c = draft_inference_time / target_verify_time[0]
    t = target_verify_time / target_verify_time[0]
    for i, b in enumerate(valid_budget):
        for d, ac_len in enumerate(results[b]):
            if ac_len < 0:
                continue
            x = ac_len / (c * d + t[i])
            if x > best_speedup:
                best_speedup = x
                best_tree_size, best_tree_depth = b, d
                best_acc_length = ac_len
    
    best_time_per_token = target_verify_time[0] / best_speedup
    print(f'{best_tree_size=}, {best_tree_depth=}, {best_time_per_token=}, {best_acc_length=}, {best_speedup=}')
    return best_tree_size, best_tree_depth


def get_grow_map(branch_map, tree_size, tree_depth, max_branches):
    m, l, b = tree_size, tree_depth, max_branches
    positions = [0]
    states = [(m,l,b)]
    active = [True]
    depth = [0]
    Successors = [[]]
    attention_mask = torch.zeros(m,m).long()
    parents = [-1]
    expand_lists = []
    expand_branches = []
    num_nodes = 1
    while True:
        expand = []
        expand_branch = []
        for i, act in enumerate(active):
            if act: 
                if parents[i] != -1:
                    attention_mask[i] = attention_mask[parents[i]]
                attention_mask[i, i] = 1
                expand.append(i)
                active[i] = False
                (x,y,z) = states[i]
                expand_branch.append(z)
                positions.extend(list(range(num_nodes, num_nodes + z)))
                Successors[i].extend(list(range(num_nodes, num_nodes + z)))
                Successors.extend([[] for _ in range(z)])
                parents.extend([i for _ in range(z)])
                depth.extend([depth[i] + 1 for _ in range(z)])
                states.extend(branch_map[(x,y,z)])
                assert len(branch_map[(x,y,z)]) == z
                num_nodes = num_nodes + z
        if len(expand) == 0:
            break
        expand_lists.append(expand)
        expand_branches.append(expand_branch)
        active.extend([True for _ in range(sum(expand_branch))])

    assert num_nodes == m
    assert len(positions) == m
    assert len(depth) == m
    grow_map = {
        'roots': expand_lists,
        'branches': expand_branches,
        'Successors':Successors,
        'mask': attention_mask,
        'depth': torch.LongTensor(depth),
        'size': num_nodes
    }
    return grow_map


class Node:
    def __init__(self):
        self.children = []
        self.path = []
        self.name = ''


def create_tree(all_nodes):
    for i, node in enumerate(all_nodes):
        node.name = str(i)
    tree = {}
    for node in all_nodes:
        tree[node.name] = [n.name for n in node.children]   
    return tree


def convert_expand_branches(expand_branches):
    root = Node()
    all_nodes = [root]
    curr_layer_nodes = [root]  # root node
    for d, layer_child_nums in enumerate(expand_branches):
        next_layer_nodes = []
        for node, num_children in zip(curr_layer_nodes, layer_child_nums):
            for i in range(num_children):
                child = Node()
                child.path = node.path + [i]
                node.children.append(child)
                all_nodes.append(child)
                next_layer_nodes.append(child)
        curr_layer_nodes = next_layer_nodes
    return all_nodes


def convert_eagle_format(eagle_format):
    root = Node()
    all_nodes = {str([]): root}
    for node_path in eagle_format:
        parent = all_nodes[str(node_path[:-1])]
        child = Node()
        child.path=node_path
        parent.children.append(child)
        all_nodes[str(node_path)] = child
    return [node for node in all_nodes.values()]


def plot_tree(tree):
    # Convert the tree structure to a directed graph
    G = nx.DiGraph(tree)

    # Compute the hierarchical layout for the tree
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')

    # Draw the tree
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='lightblue', font_size=12, font_weight='bold')
    plt.show()


if __name__ == '__main__':
    args = get_args()
    alpha = get_alpha(args.eval_results_json)
    max_depth = args.max_depth
    max_budget = args.max_budget
    max_branch = alpha.shape[0] - 1
    draft_inference_time = args.draft_time
    target_verify_time = np.array(args.target_times)
    valid_budget = args.valid_budgets

    results, T, branch_map = dynamic_program(alpha, max_budget, max_depth, max_branch)
    best_tree_size, best_tree_depth = get_optimal_tree_size_and_depth(draft_inference_time, target_verify_time, valid_budget, results)
    max_branches = T[best_tree_size, best_tree_depth].argmax(dim=0).item()
    grow_map = get_grow_map(branch_map, best_tree_size, best_tree_depth, max_branches)

    # Create the tree representation string in the same format as the `mc_sim_7b_63` eagle tree.
    all_nodes = convert_expand_branches(grow_map['branches'])
    full_str = '[' + ', '.join([str(n.path) for n in all_nodes[1:]]) + ']'
    print(full_str)

    # Code for plotting the tree
    # tree = create_tree(all_nodes)
    # plot_tree(tree)
    torch.save(grow_map, args.output_path)
