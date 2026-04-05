import glob
import inspect
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class LlamaArgs:
    """Configuration dataclass for a Llama-style model.

    Follows https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py
    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        """Construct from a config dict, ignoring unknown keys.

        Credit: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py#L12
        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class LoRAInfusedLinear(nn.Module):
    """Linear layer augmented with a low-rank LoRA adapter.

    Inspired by:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/tuner/lora.py#L11
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        r=8,
        dropout=0.0,
        scale=20.0,
        bias=False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros((r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(linear, rank=8, dropout=0.0, scale=20.0):
        """Convert a base nn.Linear into a LoRAInfusedLinear."""
        dim_out, dim_in = linear.weight.shape

        if isinstance(linear, nn.QuantizedLinear):
            # Recover true input dims from quantization bit-width
            dim_in = dim_in * 32 // linear.bits

        layer = LoRAInfusedLinear(
            input_dims=dim_in,
            output_dims=dim_out,
            r=rank,
            dropout=dropout,
            scale=scale,
        )
        layer.linear = linear
        return layer


class Attention(nn.Module):
    """Multi-head grouped-query attention with RoPE.

    Heavily inspired by:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
    """

    def __init__(self, args):
        super().__init__()
        d_model = args.hidden_size

        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (
            args.hidden_size // args.num_attention_heads
        )
        # Scale factor from "Attention Is All You Need" (1 / sqrt(d_k))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            d_model, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, d_model, bias=args.attention_bias
        )

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None):
        batch_size, max_seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(batch_size, max_seq_len, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, max_seq_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, max_seq_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, max_seq_len, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward network (Llama-style).

    Inspired by:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
    """

    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden, dim, bias=args.mlp_bias)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections.

    Inspired by:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
    """

    def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LlamaModel(nn.Module):
    """Llama backbone (embedding + transformer stack + norm).

    Heavily inspired by:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py#L13
    """

    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids):
        h = self.embed_tokens(input_ids)
        # Apply causal mask only when processing more than one token
        mask = "causal" if h.shape[1] > 1 else None
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class LlamaForCausalLM(nn.Module):
    """Llama model with a causal language-modelling head.

    Reference:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/models/llama.py
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=False
            )

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, input_ids):
        out = self.model(input_ids)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Remove unused buffers (e.g. rotary freq cache) from weights."""
        weights = {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights


def load_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pretrained_model(model_dir):
    """Load a pretrained (optionally quantized) Llama model from disk.

    Reference:
    https://github.com/ml-explore/mlx-lm/blob/834fac934c4e04de9b3d723e2b9287a2c60cfd4a/mlx_lm/utils.py
    """
    config = load_config(model_dir)
    args = LlamaArgs.from_dict(config)
    model = LlamaForCausalLM(args)

    weight_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    weights = model.sanitize(weights)

    quantization = config.get("quantization") or config.get(
        "quantization_config"
    )

    if quantization is not None:

        def class_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    # MLX uses lazy evaluation; force materialisation of all parameters
    mx.eval(model.parameters())
    return model, config


def linear_to_lora_layers(model, num_layers, rank, scale, dropout=0.0):
    """Replace the last num_layers linear sub-layers with LoRA equivalents.

    Adapted and simplified from:
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    """
    keys = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    def to_lora(layer):
        return LoRAInfusedLinear.from_base(
            layer, rank=rank, scale=scale, dropout=dropout
        )

    for block in model.layers[-max(num_layers, 0):]:
        updates = [
            (k, to_lora(m))
            for k, m in block.named_modules()
            if k in keys
        ]
        if updates:
            block.update_modules(tree_unflatten(updates))


def save_lora_adapters(model, path):
    """Save trainable LoRA adapter weights to a .safetensors file.

    Inspired by:
    https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adapters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(path, adapters)


def causal_lm_loss(logits, labels, ignore_index=-100):
    """Next-token prediction cross-entropy loss with prompt masking.

    Positions where labels == ignore_index are excluded from the loss.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != ignore_index
    # Replace ignore_index with a valid vocab index before indexing;
    # the mask zeroes out the contribution afterwards.
    safe_targets = mx.where(mask, shift_labels, 0)

    ce = nn.losses.cross_entropy(shift_logits, safe_targets)
    ce = ce * mask.astype(ce.dtype)

    ntoks = mx.maximum(mask.sum(), 1)
    return ce.sum() / ntoks
