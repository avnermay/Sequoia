import argparse
from collections import OrderedDict
from datetime import datetime
import glob
from loguru import logger
import numpy as np
import os
import random
import socket

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from transformers import RepetitionPenaltyLogitsProcessor
from trl import SFTTrainer

import src.utils as utils


def _get_model_state_dict(model, adapter_only=False):
    logger.info('in _get_model_state_dict')
    # state_dict = model.draft_model.state_dict(prefix='draft_model.')
    state_dict = {k:v for k,v in model.state_dict().items() if 'draft_model' in k}
    return state_dict


# This is a very hacky way to only save the trainable parameters.
accelerate.utils.fsdp_utils._get_model_state_dict = _get_model_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--project_name', type=str, default='train_specdec')
    parser.add_argument('--draft_type', choices=['standalone', 'eagle'], default='standalone',
                        help='The type of draft model to use.')
    parser.add_argument('--target_model_path', type=str, default=None)
    parser.add_argument('--draft_model_path', type=str, default=None)
    parser.add_argument('--draft_ft_path', type=str, default=None)
    parser.add_argument('--draft_random_init', action='store_true', default=False,
                        help='Whether to randomly initialize the draft model (instead of using pretrained weights).')
    parser.add_argument('--freeze_embeds_and_lm_head', action='store_true', default=False,
                        help='Whether to freeze the standalone models embeddings and LM head.')
    parser.add_argument('--convert_standalone', action='store_true', default=False,
                        help='Whether to train an eagle model by converting a standalone model).')
    parser.add_argument('--train_data_files', type=str, default='/data/refined_web/train*.filtered.json',
                        help='Regex or comma separated list of json(l) files (potentially compressed) for training set.')
    parser.add_argument('--valid_data_files', type=str, default='/data/data-pool/wiki_valid_single.jsonl',
                        help='Regex or comma separated list of json(l) files (potentially compressed) for validation set.')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--shuffle_buffer', type=int, default=5000,
                        help='Size of shuffle buffer. Only used when streaming is True.')
    parser.add_argument('--seq_length', type=int, default=4096)
    parser.add_argument('--min_completion_tokens', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--no_gradient_checkpointing', action='store_false', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--hf_model_cache', type=str, default='/data/.hf_cache/transformers/hub')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=200, type=int)
    parser.add_argument('--deepspeed_config', default=None, type=str)
    parser.add_argument('--do_eval_before_training', action='store_true')
    parser.add_argument('--no_special_tokens', action='store_true')
    parser.add_argument('--no_flash_attn_2', action='store_true')
    parser.add_argument('--fix_chat_format', action='store_true')
    parser.add_argument('--pack', action='store_true', help='Whether to pack the input data.')
    parser.add_argument('--lookahead', type=int, default=4)
    parser.add_argument('--coefficient', type=float, default=1/16)
    parser.add_argument('--distill_loss_weight', type=float, default=0.0)
    parser.add_argument('--ce_loss_weight', type=float, default=1.0)
    parser.add_argument('--l1_loss_weight', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true', default=False,
                        help='Whether to add noise to target model hidden states during Eagle training.')
    parser.add_argument('--mem_profile_path', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    logger.info(socket.gethostname())
    args = get_args()
    if args.mem_profile_path:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
    if args.run_name is None:
        args.run_name = f'{utils.date_str()}_{args.draft_type}_{args.output_dir.replace("/", "_")}'

    os.environ['WANDB_PROJECT'] = args.project_name
    os.environ['WANDB_CODE_DIR'] = os.path.dirname(os.path.abspath(__file__))
    os.environ['WANDB_DIR'] = '/home/avner/wandb'
    os.environ['WANDB__SERVICE_WAIT'] = '300'

    # Load tokenizer
    tokenizer = utils.get_tokenizer(args.draft_model_path, args.no_special_tokens)

    # Load datasets
    train_dataset = utils.get_dataset(
        args.train_data_files,
        tokenizer,
        args.seq_length,
        min_completion_tokens=args.min_completion_tokens,
        is_train=True,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        pack=args.pack,
        fix_chat_format=args.fix_chat_format,
    )
    valid_dataset = utils.get_dataset(
        args.valid_data_files,
        tokenizer,
        args.seq_length,
        min_completion_tokens=args.min_completion_tokens,
        is_train=False,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        pack=args.pack,
        fix_chat_format=args.fix_chat_format,
    )

    # Load training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy='steps',
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # TODO: enable gradient checkpoint on models?
        # gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=args.run_name,
        report_to='wandb',
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed_config,
    )

    # Eagle code requires exact path to model folder.
    # target_model_path = (args.target_model_path if args.draft_type != 'eagle' else
    #                                         utils.get_full_model_path(args.target_model_path, hf_model_cache=args.hf_model_cache))
    # draft_model_path = (args.draft_model_path if args.draft_type != 'eagle' else
    #                                         utils.get_full_model_path(args.draft_model_path, hf_model_cache=args.hf_model_cache))
    target_model_path = args.target_model_path
    draft_model_path = args.draft_model_path
    if args.draft_type == 'standalone':
        model = utils.load_standalone_model(
            target_model_path, draft_model_path, args.seq_length,
            no_flash_attn_2=args.no_flash_attn_2,
            ce_loss_weight=args.ce_loss_weight, distill_loss_weight=args.distill_loss_weight,
            lookahead=args.lookahead, freeze_embeds_and_lm_head=args.freeze_embeds_and_lm_head,
        )
    elif args.draft_type == 'eagle':
        model = utils.load_eagle_model(
            target_model_path, draft_model_path, args.seq_length,
            no_flash_attn_2=args.no_flash_attn_2,
            ce_loss_weight=args.ce_loss_weight, distill_loss_weight=args.distill_loss_weight, l1_loss_weight=args.l1_loss_weight,
            lookahead=args.lookahead, add_noise=args.add_noise, draft_random_init=args.draft_random_init,
            convert_standalone=args.convert_standalone,
        )
    if args.draft_ft_path:
        model = utils.load_draft_ft_ckpt(model, args.draft_ft_path)

    # Prepare trainer
    # In order for the keys in the metrics dictionary returned by compute_metrics to be correct,
    # we must be consistent with the way the metrics are extracted by the Trainer in this code:
    # https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/trainer.py#L3614
    output_keys = model.output_keys()
    output_keys.remove('loss')
    compute_metrics_fn = lambda eval_preds: utils.compute_metrics(eval_preds, output_keys)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        max_seq_length=args.seq_length,
        data_collator=utils.DataCollatorWithPadding(args.seq_length) if not args.pack else None,
        compute_metrics=compute_metrics_fn,
        packing=True,
    )

    # train
    if args.resume_from_checkpoint:
        trainer.train(args.resume_from_checkpoint)
    else:
        trainer.train()

    if args.mem_profile_path:
        torch.cuda.memory._dump_snapshot(args.mem_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    print('Done training!')
