from datetime import datetime
import glob
from loguru import logger
import numpy as np
import os
import re
from typing import Any, Dict, List

from datasets import interleave_datasets, load_dataset
import einops
import torch
import torch.distributed.checkpoint as dist_cp
from torch.utils.data import IterableDataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import RepetitionPenaltyLogitsProcessor
from transformers.models.llama.configuration_llama import LlamaConfig
from trl.trainer import ConstantLengthDataset

from src.eagle import EagleDraftModel


class StandaloneModel(nn.Module):

    def __init__(
        self, target_model, draft_model, seq_length,
        ce_loss_weight=1.0, distill_loss_weight=0.0, lookahead=4,
        freeze_embeds_and_lm_head=False,
    ):
        super().__init__()
        if distill_loss_weight and target_model is None:
            raise ValueError('Cannot do distillation with an empty target model')

        if target_model:
            target_model.eval()

        self.target_model = target_model
        self.draft_model = draft_model
        self.seq_length = seq_length
        self.ce_loss_weight = ce_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.lookahead = lookahead

        # Freeze parameters in target model.
        if target_model:
            for param in self.target_model.parameters():
                param.requires_grad = False

        if freeze_embeds_and_lm_head:
            self.draft_model.model.embed_tokens.weight.requires_grad_(False)
            self.draft_model.lm_head.weight.requires_grad_(False)

    def forward(self, input_ids, labels=None, attention_mask=None, loss_mask=None, temp=1.0, top_p=1.0, rep_penalty=1.0, max_branch=1):
        del labels

        # Pass input through the draft model.
        draft_outputs = self.draft_model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            output_hidden_states=True,
            use_cache=False,
        )
        draft_logits = draft_outputs.logits
        d = draft_logits.device

        if self.target_model:
            self.target_model.eval()    # Is there a nicer way to do this? Context manager?
            with torch.no_grad():
                # Pass input through the base model.
                target_outputs = self.target_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    output_hidden_states=True,
                    use_cache=False,
                )
                target_logits = target_outputs.logits.to(device=d)

        target_labels = input_ids[:, 1:].to(device=d)

        # Compute loss.
        loss_mask = loss_mask[:, :-1].to(device=d) if loss_mask is not None else None
        loss, ce_loss, distill_loss = torch.zeros([], device=d), torch.zeros([], device=d), torch.zeros([], device=d)
        if self.ce_loss_weight:
            ce_loss = cross_entropy(draft_logits, target_labels, loss_mask=loss_mask)
            loss += self.ce_loss_weight * ce_loss
        if self.distill_loss_weight:
            distill_loss = distill_cross_entropy(draft_logits, target_logits, loss_mask=loss_mask)
            loss += self.distill_loss_weight * distill_loss

        if not self.training and self.target_model:
            alpha = acceptance_rate(
                target_logits, draft_logits, temp=temp, top_p=top_p, rep_penalty=rep_penalty, input_ids=input_ids[:, :-1],
                max_branch=max_branch,
            )

            if max_branch == 1:
                alpha = alpha.unsqueeze(dim=-1)
                if loss_mask is not None:
                    loss_mask = loss_mask.unsqueeze(dim=-1)

            if loss_mask is not None:
                # Each element of batch is weighted by how many generated tokens (with loss) it has.
                alpha_weighted_avg = einops.reduce(alpha * loss_mask, 'b t k -> k', 'sum') / loss_mask.sum()
                # Each element of batch is weighted equally.
                alpha_avg = torch.mean(
                    torch.sum(alpha * loss_mask, dim=1) / loss_mask.sum(dim=1),
                    dim=0,
                )
            else:
                alpha_avg = einops.reduce(alpha, 'b t k -> k', 'mean')
                alpha_weighted_avg = alpha_avg

            if max_branch == 1:
                a, l = alpha_avg, self.lookahead
                # pulsar_acceptance_rate:
                # = (1/L) * avg_num_accepted_tokens
                # = (1/L) * (avg_num_generated_tokens - 1)
                # = (1/L) * ( (1 - a^(L + 1)) / (1 - a) - 1)
                pulsar_acceptance_rate = (1/l) * ((1 - pow(a, l + 1)) / (1 - a) - 1)
                alpha_avg = alpha_avg.squeeze()
                alpha_weighted_avg = alpha_weighted_avg.squeeze()
            else:
                pulsar_acceptance_rate = torch.zeros([], device=d)
        else:
            alpha_avg, alpha_weighted_avg, pulsar_acceptance_rate = [torch.zeros([], device=d)] * 3

        # Prepare output
        output = {
            'loss': loss,
            'ce_loss': ce_loss,
            'distill_loss': distill_loss,
            'acceptance_rate': alpha_avg,
            'acceptance_rate_weighted': alpha_weighted_avg,
            'pulsar_acceptance_rate': pulsar_acceptance_rate,
            'avg_input_length': attention_mask.sum(dim=1).to(torch.float32).mean() if attention_mask is not None else torch.tensor(0.0, device=d),
            'avg_generation_length': loss_mask.sum(dim=1).to(torch.float32).mean() if loss_mask is not None else torch.tensor(0.0, device=d),
            'max_input_length': attention_mask.sum(dim=1).to(torch.float32).max() if attention_mask is not None else torch.tensor(0.0, device=d),
            'max_generation_length': loss_mask.sum(dim=1).to(torch.float32).max() if loss_mask is not None else torch.tensor(0.0, device=d),
            'min_input_length': attention_mask.sum(dim=1).to(torch.float32).min() if attention_mask is not None else torch.tensor(0.0, device=d),
            'min_generation_length': loss_mask.sum(dim=1).to(torch.float32).min() if loss_mask is not None else torch.tensor(0.0, device=d),
        }
        assert list(output.keys()) == self.output_keys(), (
            f'Mismatch between expected ({self.output_keys()}) and actual ({list(output.keys())}) output keys')
        return output

    def output_keys(self):
        return [
            'loss', 'ce_loss', 'distill_loss', 'acceptance_rate', 'acceptance_rate_weighted', 'pulsar_acceptance_rate',
            'avg_input_length', 'avg_generation_length',    'max_input_length', 'max_generation_length', 'min_input_length', 'min_generation_length',
        ]


class EagleModel(nn.Module):

    def __init__(
        self, target_model, draft_model, seq_length,
        ce_loss_weight=1.0, distill_loss_weight=0.0, l1_loss_weight=0.0,
        lookahead=4, add_noise=True,
    ):
        super().__init__()
        target_model.eval()
        self.target_model = target_model
        self.draft_model = draft_model
        self.seq_length = seq_length
        self.ce_loss_weight = ce_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.lookahead = lookahead
        self.add_noise = add_noise
        self.l1loss = nn.SmoothL1Loss(reduction='none')
        # Freeze parameters in target model.
        for param in self.target_model.parameters():
            param.requires_grad = False


    def forward(self, input_ids, labels=None, attention_mask=None, loss_mask=None, temp=1.0, top_p=1.0, rep_penalty=1.0, max_branch=1):
        del labels
        self.target_model.eval()    # Is there a nicer way to do this? Context manager?
        with torch.no_grad():
            # Pass input through the base model.
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            input_embeds = target_outputs.hidden_states[0]
            hidden_states = target_outputs.hidden_states[-1]
            target_logits = target_outputs.logits[:, :-1]    # s2, s3
            target_labels = input_ids[:, 1:]
            target_hidden_states = hidden_states[:, :-1]     # h2, h3
            if self.add_noise and self.training:
                # Currently we hardcode gaussian noise with 0 mean and 0.2 standard deviation.
                hidden_states = hidden_states.clone() + torch.randn(hidden_states.size()) * 0.2

            b, _, d = hidden_states.shape
            z = torch.zeros((b, 1, d), device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = torch.cat([z, hidden_states[:, :-2]], dim=1)    # h1, h2

        hidden_states.requires_grad_(True)
        # Pass input through the draft model.
        draft_input_embeds = input_embeds[:, :-1]    # x2, x3
        draft_hidden_states = self.draft_model(    # h2', h3'
            hidden_states,        # h1, h2
            draft_input_embeds,    # x2, x3
            attention_mask=attention_mask[:, :-1],
            use_cache=False,
        )
        draft_logits = self.target_model.lm_head(draft_hidden_states)    # s2', s3'

        # Compute loss.
        d = draft_logits.device
        target_logits, target_labels, target_hidden_states = (
            target_logits.to(device=d), target_labels.to(device=d), target_hidden_states.to(device=d))
        loss_mask = loss_mask[:, :-1].to(device=d) if loss_mask is not None else None
        loss, ce_loss, distill_loss, l1_loss = torch.zeros([], device=d), torch.zeros([], device=d), torch.zeros([], device=d), torch.zeros([], device=d)
        if self.ce_loss_weight:
            ce_loss = cross_entropy(draft_logits, target_labels, loss_mask=loss_mask)
            loss += self.ce_loss_weight * ce_loss
        if self.distill_loss_weight:
            distill_loss = distill_cross_entropy(draft_logits, target_logits, loss_mask=loss_mask)
            loss += self.distill_loss_weight * distill_loss
        if self.l1_loss_weight:
            l1_loss = self.l1loss(draft_hidden_states, target_hidden_states)    # [b, l, d]
            if loss_mask is not None:
                l1_loss = torch.sum(loss_mask * torch.mean(l1_loss, -1)) / loss_mask.sum()
            else:
                l1_loss = torch.mean(l1_loss)

            loss += self.l1_loss_weight * l1_loss

        if not self.training and self.target_model:
            alpha = acceptance_rate(
                target_logits, draft_logits, temp=temp, top_p=top_p, rep_penalty=rep_penalty,
                input_ids=input_ids[:, :-1], max_branch=max_branch,
            )

            if max_branch == 1:
                alpha = alpha.unsqueeze(dim=-1)
                if loss_mask is not None:
                    loss_mask = loss_mask.unsqueeze(dim=-1)

            if loss_mask is not None:
                # Each element of batch is weighted by how many generated tokens (with loss) it has.
                alpha_weighted_avg = einops.reduce(alpha * loss_mask, 'b t k -> k', 'sum') / loss_mask.sum()
                # Each element of batch is weighted equally.
                alpha_avg = torch.mean(
                    torch.sum(alpha * loss_mask, dim=1) / loss_mask.sum(dim=1),
                    dim=0,
                )
            else:
                alpha_avg = einops.reduce(alpha, 'b t k -> k', 'mean')
                alpha_weighted_avg = alpha_avg

            if max_branch == 1:
                a, l = alpha_avg, self.lookahead
                # pulsar_acceptance_rate:
                # = (1/L) * avg_num_accepted_tokens
                # = (1/L) * (avg_num_generated_tokens - 1)
                # = (1/L) * ( (1 - a^(L + 1)) / (1 - a) - 1)
                pulsar_acceptance_rate = (1/l) * ((1 - pow(a, l + 1)) / (1 - a) - 1)
            else:
                pulsar_acceptance_rate = torch.zeros([], device=d)
        else:
            alpha_avg, alpha_weighted_avg, pulsar_acceptance_rate = [torch.zeros([], device=d)] * 3

        # Prepare output
        output = {
            'loss': loss,
            'ce_loss': ce_loss,
            'distill_loss': distill_loss,
            'l1_loss': l1_loss,
            'acceptance_rate': alpha_avg,
            'acceptance_rate_weighted': alpha_weighted_avg,
            'pulsar_acceptance_rate': pulsar_acceptance_rate,
            'avg_input_length': attention_mask.sum(dim=1).to(torch.float32).mean() if attention_mask is not None else torch.tensor(0.0, device=d),
            'avg_generation_length': loss_mask.sum(dim=1).to(torch.float32).mean() if loss_mask is not None else torch.tensor(0.0, device=d),
            'max_input_length': attention_mask.sum(dim=1).to(torch.float32).max() if attention_mask is not None else torch.tensor(0.0, device=d),
            'max_generation_length': loss_mask.sum(dim=1).to(torch.float32).max() if loss_mask is not None else torch.tensor(0.0, device=d),
            'min_input_length': attention_mask.sum(dim=1).to(torch.float32).min() if attention_mask is not None else torch.tensor(0.0, device=d),
            'min_generation_length': loss_mask.sum(dim=1).to(torch.float32).min() if loss_mask is not None else torch.tensor(0.0, device=d),
        }
        assert list(output.keys()) == self.output_keys(), (
                f'Mismatch between expected ({self.output_keys()}) and actual ({list(output.keys())}) output keys')
        return output

    def output_keys(self):
        return [
                'loss', 'ce_loss', 'distill_loss', 'l1_loss', 'acceptance_rate', 'acceptance_rate_weighted', 'pulsar_acceptance_rate',
                'avg_input_length', 'avg_generation_length',    'max_input_length', 'max_generation_length', 'min_input_length', 'min_generation_length',
        ]

    def forward_n(self, input_ids, start_idx=0, n=10, labels=None, attention_mask=None, loss_mask=None, temp=1.0, top_p=1.0, rep_penalty=1.0, k=32):
        # Note: We ignore `attention_mask` and `loss_mask` for now, as we assume we are just using this function during eval.
        del labels, attention_mask, loss_mask

        self.target_model.eval()    # Is there a nicer way to do this? Context manager?
        with torch.no_grad():
            # Pass input through the base model.
            target_outputs = self.target_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            input_embeds = target_outputs.hidden_states[0]
            hidden_states = target_outputs.hidden_states[-1]
            target_logits = target_outputs.logits
            draft_hidden_states = hidden_states[:, :start_idx]    # h1, h2

        outputs = {}
        past_key_values = None
        draft_input_embeds = input_embeds[:, 1:start_idx + 1]    # x2, x3
        all_draft_hidden_states = []
        for i in range(n):
            draft_hidden_states, past_key_values = self.draft_model(    # h2', h3'
                draft_hidden_states,        # h1, h2
                draft_input_embeds,    # x2, x3
                use_cache=True,
                past_key_values=past_key_values,
            )
            all_draft_hidden_states.append(draft_hidden_states[:, -1:])
            if i != n - 1:
                draft_hidden_states = draft_hidden_states[:, -1:]
                draft_input_embeds = input_embeds[:, start_idx + i + 1: start_idx + i + 2]

        all_draft_hidden_states = torch.cat(all_draft_hidden_states, dim=1)
        draft_logits = self.target_model.lm_head(all_draft_hidden_states)
        curr_target_logits = target_logits[:, start_idx: start_idx + n].to(device=draft_logits.device)
        alpha = specinfer_acceptance(curr_target_logits, draft_logits, k=k)    # [1, look-ahead, branch-factor]
        outputs['acceptance_rate'] = alpha.squeeze()
        return outputs


def date_str():
    return datetime.now().strftime("%Y_%m_%d")


def load_eagle_ft_ckpt_old(model, path):
    # state_dict = {'model': model.draft_model.state_dict(prefix='draft_model.')}
    # We skip `embed_tokens` because it is no longer part of the Eagle `Model` class.
    state_dict = {'model': {k:v for k,v in model.state_dict().items() if 'draft_model' in k and 'embed_tokens' not in k}}
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(path),
        no_dist=True,
    )
    return model


# load_standalone_ft_ckpt -> load_draft_ft_ckpt
# load_draft_ft_ckpt -> load_eagle_ft_ckpt_old

def load_draft_ft_ckpt(model, path):
    state_dict = {'model': {k:v for k,v in model.state_dict().items() if 'draft_model' in k}}
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(path),
        no_dist=True,
    )
    return model


def get_full_model_path(model_path, hf_model_cache=None):
    if os.path.isfile(f'{model_path}/config.json'):
        return model_path

    orig_model_path = model_path
    model_path = model_path.replace('/', '--')
    model_path = f'{hf_model_cache}/models--{model_path}/snapshots/*'
    globbed_model_paths = glob.glob(model_path)
    assert len(globbed_model_paths) == 1, f'There should be exactly one model path at {model_path}, got: {globbed_model_paths}'
    model_path = globbed_model_paths[0]
    if os.path.isfile(f'{model_path}/config.json'):
        return model_path

    raise ValueError(f'Model not found at {model_path}. Please download {orig_model_path} and re-launch training.')


def load_standalone_model(
        target_model_path, draft_model_path, seq_length,
        no_flash_attn_2=False, ce_loss_weight=0.0, distill_loss_weight=0.1, lookahead=4,
        freeze_embeds_and_lm_head=False, target_device_map=None,
    ):
    if target_model_path:
        # We allow training standalone draft models without a target model.
        # But will not get acceptance rates during training in this case.
        target_model = load_pretrained_model(target_model_path, no_flash_attn_2=no_flash_attn_2, device_map=target_device_map)
    else:
        target_model = None
    draft_model = load_pretrained_model(draft_model_path, no_flash_attn_2=no_flash_attn_2, device_map=None)
    return StandaloneModel(
        target_model, draft_model, seq_length,
        ce_loss_weight=ce_loss_weight, distill_loss_weight=distill_loss_weight,
        lookahead=lookahead, freeze_embeds_and_lm_head=freeze_embeds_and_lm_head,
    )


def load_eagle_model(
        target_model_path, draft_model_path, seq_length,
        no_flash_attn_2=False, ce_loss_weight=1.0, distill_loss_weight=0.0, l1_loss_weight=0.0, lookahead=4,
        add_noise=True, target_device_map=None, draft_device_map=None, draft_random_init=False,
        convert_standalone=False,
):
    target_model = load_pretrained_model(
        target_model_path, no_flash_attn_2=no_flash_attn_2, device_map=target_device_map,
    )
    draft_model = load_pretrained_model(
        draft_model_path, no_flash_attn_2=no_flash_attn_2, device_map=draft_device_map, random_init=draft_random_init,
        load_cls=EagleDraftModel, convert_standalone=convert_standalone,
    )
    return EagleModel(
        target_model, draft_model, seq_length,
        ce_loss_weight=ce_loss_weight, distill_loss_weight=distill_loss_weight, l1_loss_weight=l1_loss_weight,
        lookahead=lookahead, add_noise=add_noise,
    )


def load_eagle_model_old(
    target_model_path, draft_model_path, seq_length,
    no_flash_attn_2=False, ce_loss_weight=1.0, distill_loss_weight=0.0, l1_loss_weight=0.0, lookahead=4,
    add_noise=True, target_device_map=None,
):
    target_model = load_pretrained_model(target_model_path, no_flash_attn_2=no_flash_attn_2, device_map=target_device_map)
    config = LlamaConfig.from_pretrained(f'{draft_model_path}/config.json')
    bias = config.bias if 'bias' in vars(config) else True
    draft_model = EagleDraftModel(config, bias=bias)
    draft_model.load_state_dict(
        torch.load(f'{draft_model_path}/pytorch_model.bin'),
        # We load with `strict=False` because `embed_tokens` are no longer part of the Eagle `Model` class.
        strict=False,
    )
    return EagleModel(
        target_model, draft_model, seq_length,
        ce_loss_weight=ce_loss_weight, distill_loss_weight=distill_loss_weight, l1_loss_weight=l1_loss_weight,
        lookahead=lookahead, add_noise=add_noise,
    )


def load_pretrained_model(
        model_path, no_flash_attn_2=False, device_map=None, random_init=False, load_cls=AutoModelForCausalLM,
        convert_standalone=False
    ):
    if not model_path:
        return None

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not no_flash_attn_2:
        config._flash_attn_2_enabled = True
        config._attn_implementation = 'flash_attention_2'
    if random_init:
        model = load_cls(config)
        model.init_weights()
    else:
        kwargs = {'convert_standalone': convert_standalone} if convert_standalone else {}
        model = load_cls.from_pretrained(
            model_path,
            # config=config,
            # torch_dtype=torch.bfloat16 if bf16 else torch.float16,    # We let accelerate take care of this
            trust_remote_code=True,
            device_map=device_map,
            **kwargs,
        )
    return model


def compute_metrics(eval_preds, keys):
    metrics_tuple = eval_preds.predictions
    assert len(metrics_tuple) == len(keys)
    metrics_dict = {key: metric.mean() for key, metric in zip(keys, metrics_tuple)}
    return metrics_dict


def cross_entropy(draft_logits, target_labels, loss_mask=None):
    draft_logp = nn.LogSoftmax(dim=-1)(draft_logits)
    gathered_logp = torch.gather(draft_logp, index=target_labels.unsqueeze(-1), dim=-1).squeeze(-1)
    if loss_mask is not None:
        neg_logp = torch.sum(-gathered_logp * loss_mask) / loss_mask.sum()
    else:
        neg_logp = torch.mean(-gathered_logp)
    return neg_logp


def distill_cross_entropy(draft_logits, target_logits, loss_mask=None):
    draft_logp = nn.LogSoftmax(dim=-1)(draft_logits)
    target_p = nn.Softmax(dim=-1)(target_logits)
    neg_plogp = torch.sum(-target_p * draft_logp, dim=-1)
    if loss_mask is not None:
        neg_plogp = torch.sum(neg_plogp * loss_mask) / loss_mask.sum()
    else:
        neg_plogp = torch.mean(neg_plogp)
    return neg_plogp


def acceptance_rate(target_logits, draft_logits, temp=1.0, top_p=1.0, rep_penalty=1.0, input_ids=None, max_branch=1):
    # `target_logits` and `draft_logits` should have shape `(batch_size, seq_len, vocab_size)`.
    if rep_penalty != 1.0:
        processor = RepetitionPenaltyLogitsProcessor(rep_penalty)
        for i in range(1, target_logits.size(1)):
            processor(input_ids[:, :i], target_logits[:, i])
            processor(input_ids[:, :i], draft_logits[:, i])
        
    if top_p != 1.0:
        target_logits = top_p_logit_warper(target_logits, top_p)
        draft_logits = top_p_logit_warper(draft_logits, top_p)

    if temp != 1.0:
        target_logits = target_logits / temp
        draft_logits = draft_logits / temp


    if max_branch == 1:
        target_p = torch.softmax(target_logits, dim=-1)
        draft_p = torch.softmax(draft_logits, dim=-1)
        return 1 - (target_p - draft_p).abs().sum(dim=-1) / 2    # `(batch_size, seq_len)`

    else:
        return specinfer_acceptance(target_logits, draft_logits, k=max_branch)


def get_residual(target_p, draft_p):
    x = F.relu(target_p - draft_p)    # (b, t, v)
    x_norm = torch.norm(x, p=1, dim=-1, keepdim=True)    # (b, t, 1)
    vocab_size = target_p.shape[-1]
    acceptance_probs = (1 - x_norm / 2.0).squeeze(-1)    # (b, t)
    residual = torch.where(x_norm > 0, x / x_norm, 1 / vocab_size)    # (b, t, v)
    return residual, acceptance_probs


def specinfer_acceptance(target_logits, draft_logits, replacement=False, k=2):
    draft_logits = draft_logits.to(dtype=torch.float)
    target_logits = target_logits.to(dtype=torch.float)
    target_p = torch.softmax(target_logits, dim=-1)    # `(b, t, v)`
    b, t, v = target_p.shape
    accepted = torch.zeros((b, t, k), device=target_logits.device)
    residual_p = target_p    # `(b, t, v)`
    for i in range(k):
        draft_p = torch.softmax(draft_logits, dim=-1)    # `(b, t, v)`
        samples = einops.rearrange(
            torch.multinomial(einops.rearrange(draft_p, 'b t v -> (b t) v'), 1),
            '(b t) 1 -> b t 1', b=b, t=t)
        draft_sample_probs = torch.gather(draft_p, -1, samples)    # `(b, t, 1)`
        residual_sample_probs = torch.gather(residual_p, -1, samples)    # `(b, t, 1)`
        r = torch.rand((b, t, 1), device=target_logits.device)
        accepted[:, :, i] = (r <= (residual_sample_probs / draft_sample_probs)).squeeze(-1)    # `(b, t, 1)`
        residual_p, _ = get_residual(residual_p, draft_p)
        if not replacement:
            # We use -50 instead of `float('inf')` to avoid nan issues.
            draft_logits = torch.scatter(draft_logits, -1, samples, -50)

    # `accepted` contains a 1 at position [i,j,k] if one of the first k speculated tokens
    # for batch element i token j was accepted.
    accepted = (torch.cumsum(accepted, dim=-1) >= 1).to(dtype=torch.float)    # (b, t, k)
    return accepted


def top_p_logit_warper(logits, top_p, filter_value=-100, min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(logits, descending=False, stable=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0
    
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def prepare_sample_text(example, fix_chat_format=False, tokenizer=None):
    if 'text' in example:
        text = example['text']
    elif 'content' in example:
        text = example['content']
    elif 'data' in example:
        text = example['data']
    else:
        raise Exception('Unknown dataset format.')

    if fix_chat_format:
        pattern = r".*\[INST\] (.*) \[/INST\](.*)"
        match = re.search(pattern, text)

        if match and len(match.groups()) == 2:
            user, assistant = match.groups()
            user = user.replace('<<SYS>>', '').replace('<</SYS>>', '')
            messages = [
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return text


def get_tokenizer(model_path, no_special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = '<|endoftext|>'

    if no_special_tokens:
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
    return tokenizer


def fix_chat_format_fn(example, tokenizer):
        text = example['text']
        prompt = example['prompt']
        pattern = r".*\[INST\] (.*) \[/INST\](.*)"
        match = re.search(pattern, text)

        if match and len(match.groups()) == 2:
            user, assistant = match.groups()
            user = user.replace('<<SYS>>', '').replace('<</SYS>>', '')
            messages = [
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False)
            return text, prompt

        return text, prompt


def tokenize(example, tokenizer, seq_length, fix_chat_format=False):
    if fix_chat_format:
        text, prompt = fix_chat_format_fn(example, tokenizer)
    else:
        text = example['text']
        prompt = example['prompt']

    all_tokens = tokenizer(text, return_tensors='pt')
    prompt_tokens = tokenizer(prompt, return_tensors='pt')
    prompt_len = prompt_tokens['input_ids'].shape[1]
    attn_mask = all_tokens['attention_mask']
    loss_mask = attn_mask.clone()
    loss_mask[:, :prompt_len] = 0
    return {
        'input_ids': all_tokens['input_ids'][:, :seq_length],
        'labels': all_tokens['input_ids'][:, :seq_length],
        'attention_mask': attn_mask[:, :seq_length],
        'loss_mask': loss_mask[:, :seq_length],
    }


def get_dataset(
        data_files, tokenizer, seq_length, min_completion_tokens,
        is_train=True, streaming=False, pack=False, shuffle_buffer=5000, seed=0,
        fix_chat_format=False,
):
    data_files = data_files.split(',')
    all_data_files = []
    for data_file in data_files:
        all_data_files.extend(glob.glob(data_file))

    split = 'train' if is_train else 'test'
    datasets = []
    for data_file in all_data_files:
        try:
            dataset = load_dataset('json', data_files=[data_file], split=split, streaming=streaming)
        except:
            # Hacky way to load validation datasets where the split is 'train'.
            dataset = load_dataset('json', data_files=[data_file], split='train', streaming=streaming)

        if is_train:
            if streaming:
                dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
            else:
                dataset = dataset.shuffle(seed=seed)
        datasets.append(dataset)
    probabilities = [1/len(datasets) for _ in range(len(datasets))] if is_train else None
    dataset = interleave_datasets(datasets, stopping_strategy='all_exhausted', probabilities=probabilities, seed=seed)

    if pack:
        dataset = ConstantLengthDataset(
            tokenizer,
            dataset,
            formatting_func=lambda x: prepare_sample_text(x, fix_chat_format=fix_chat_format, tokenizer=tokenizer),
            infinite=is_train,
            seq_length=seq_length,
        )
    else:
        dataset = dataset.map(
            lambda example: tokenize(example, tokenizer, seq_length, fix_chat_format=fix_chat_format),
            remove_columns=['text', 'prompt', 'completion', 'args', 'uid'],
        )
        dataset = dataset.filter(
            lambda example: example['loss_mask'].sum().item() > min_completion_tokens,
        )
        if is_train:
            dataset = InfiniteDataset(dataset)

    return dataset


class InfiniteDataset(IterableDataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                print('WARNING: The dataset reached end and the iterator is reset to the start.')
                iterator = iter(self.dataset)


class DataCollatorWithPadding:

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def paddingtensor2D(self, in_tensors):
        b, l = in_tensors.shape
        padding_tensor = torch.zeros(b, self.seq_length - l, dtype=in_tensors.dtype)
        out_tensors = torch.cat((in_tensors, padding_tensor), dim=1)
        return out_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids']) for item in features])
        batch_labels = torch.cat([self.paddingtensor2D(item['labels']) for item in features])
        batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask']) for item in features])
        batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask']) for item in features])
        batch = {
            'input_ids': batch_input_ids,
            'labels': batch_labels,
            'attention_mask': batch_attention_mask,
            'loss_mask': batch_loss_mask,
        }
        return batch
