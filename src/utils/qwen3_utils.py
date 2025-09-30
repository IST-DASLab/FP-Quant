from typing import Tuple, Optional, Callable, Dict, Any

import torch
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, eager_attention_forward
from transformers.activations import ACT2FN

from ..quantization.qlinear import QLinear
from ..quantization.quantizer import Quantizer
from ..transforms.transforms import BaseTransform, IdentityTransform


class QuantizedQwen3MLP(nn.Module):

    def __init__(
        self, 
        config: Qwen3Config,
        weight_quantizer_kwargs: Dict[str, Any] | None = None,
        act_quantizer_kwargs: Dict[str, Any] | None = None,
        gate_up_in_transform: BaseTransform = IdentityTransform(),
        down_in_transform: BaseTransform = IdentityTransform()
    ):
        super().__init__()
        # gate, up accept the same input
        gate_up_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        # Init layers   
        self.up_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=gate_up_act_quantizer
        )
        self.gate_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=gate_up_act_quantizer
        )
        self.down_proj = QLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        )
        self.act_fn = ACT2FN[config.hidden_act] 

        self.gate_up_in_transform = gate_up_in_transform
        self.down_in_transform = down_in_transform

        self._train_mode = True

    def forward(self, x: torch.Tensor):
        # Rotate input
        x = self.gate_up_in_transform(x)
        # Get up and gate projection outputs
        up = self.up_proj(x, self.gate_up_in_transform)
        gate = self.gate_proj(x, self.gate_up_in_transform)
        # Apply activation function
        x = self.act_fn(gate) * up
        # Get down projection output
        x = self.down_in_transform(x)
        down = self.down_proj(x, self.down_in_transform)
        return down

    def fix_parametrization(self):
        # Fix layer parametrizations
        self.up_proj.fix_parametrization(self.gate_up_in_transform)
        self.gate_proj.fix_parametrization(self.gate_up_in_transform)
        self.down_proj.fix_parametrization(self.down_in_transform)

        self._train_mode = False


class QuantizedQwen3Attention(nn.Module):

    def __init__(
        self, 
        config: Qwen3Config, 
        layer_idx: int,
        weight_quantizer_kwargs: Dict[str, Any] | None = None,
        act_quantizer_kwargs: Dict[str, Any] | None = None,
        qkv_in_transform: BaseTransform = IdentityTransform(),
        o_in_transform: BaseTransform = IdentityTransform()
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # q, k, v accept the same input
        qkv_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        
        self.q_proj = QLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.k_proj = QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.v_proj = QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.o_proj = QLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        
        # Init transformations
        self.qkv_in_transform = qkv_in_transform
        self.o_in_transform = o_in_transform

        self._train_mode = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Rotate input
        hidden_states = self.qkv_in_transform(hidden_states)

        query_states =  self.q_norm(self.q_proj(hidden_states, self.qkv_in_transform).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states, self.qkv_in_transform).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states, self.qkv_in_transform).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                ValueError(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # Rotate attn output
        attn_output = self.o_in_transform(attn_output)
        attn_output = self.o_proj(attn_output, self.o_in_transform)
        return attn_output, attn_weights

    def fix_parametrization(self):
        # Fix layer parametrizations
        self.q_proj.fix_parametrization(self.qkv_in_transform)
        self.k_proj.fix_parametrization(self.qkv_in_transform)
        self.v_proj.fix_parametrization(self.qkv_in_transform)
        self.o_proj.fix_parametrization(self.o_in_transform)

        self._train_mode = False
