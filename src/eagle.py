""" PyTorch Eagle model implementation."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from transformers import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaPreTrainedModel, LlamaRMSNorm


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EagleLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config, index, convert_standalone=False):
        # In the Eagle codebase, they don't use layer norm for the first Eagle decoder layer,
        # so we copy that approach here, so that checkpoints are compatible.
        super().__init__(config, index)
        if index == 0 and not convert_standalone:
            self.input_layernorm = Identity()


class EagleDraftModel(LlamaPreTrainedModel):

    def __init__(self, config, bias=True, convert_standalone=False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.convert_standalone = convert_standalone

        # We currently hardcode SDPA attention if not doing convert_standalone.
        config._attn_implementation = 'sdpa' if not convert_standalone else config._attn_implementation
        self.layers = nn.ModuleList([EagleLlamaDecoderLayer(config, index, convert_standalone=convert_standalone)
                                     for index in range(config.num_hidden_layers)])

        # The Eagle repo does not apply LlamaRMSNorm after all the decoder layers, but the LlamaModel used by
        # Llama3 does. So when we are converting standalone models we must apply this normalization.
        if convert_standalone:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = Identity()

        if convert_standalone:
            # For now we hardcode bias=False for this setting. We could also simply initialize the
            # bias to a very small value.
            bias = False
        elif 'bias' in vars(config):
            bias = config.bias

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.gradient_checkpointing = False


    def _init_weights(self, module):
        super()._init_weights(module)
        if module == self.fc and self.convert_standalone:
            # We initialize the projection layer so that at the beginning of training it basically
            # functions as an identity layer.

            # 0.001 (58.0 -> 52.6)
            # 0.0003 (58.0 -> 57.4)
            # 0.0001 (58.0, no degradation)
            self.fc.weight.data *= 0.0003

            out_dim = self.fc.weight.data.shape[0]
            self.fc.weight.data[:out_dim, :out_dim] += torch.eye(
                out_dim, dtype=self.fc.weight.data.dtype, device=self.fc.weight.data.device)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = AttentionMaskConverter._make_causal_mask(
                input_shape,
                torch.float32, # [MODIFIED BY EAGLE REPO] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = AttentionMaskConverter._expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1],
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        next_decoder_cache = () if use_cache else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask.to(hidden_states.dtype),
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask.to(hidden_states.dtype),
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = self.norm(hidden_states)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            return hidden_states, next_cache

        return hidden_states
