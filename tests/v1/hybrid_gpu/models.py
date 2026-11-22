# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny models, one per KV cache layout the CPU tiers serve.

Each family is a Hugging Face config small enough to boot in seconds, with
the head counts divisible by 8 so any tensor-parallel size up to 8 shards it.
Weights are random: saved by `transformers` from a seeded init (logits far
enough apart for greedy comparisons), or, where `transformers` has no module
for a part (the MTP drafter), vLLM's dummy loader.

SimpleCPU, and so the tiered connector, needs a full-attention group, so a
pure SSM or pure sliding-window model is not a family here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LINEAR = "linear_attention"
FULL = "full_attention"
SLIDING = "sliding_attention"


def _every_fourth_full(num_layers: int) -> list[str]:
    """Qwen3.5's layout: three GDN layers, then a full-attention one."""
    return [FULL if i % 4 == 3 else LINEAR for i in range(num_layers)]


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_type: str
    architecture: str
    config: dict[str, Any] = field(hash=False, compare=False)
    # Recurrent-state (GDN) layers: Mamba-style cache groups, exact-end
    # states, frontier checkpoints.
    hybrid: bool = False
    # An MTP drafter rides along (speculative decoding, k=2).
    mtp: bool = False
    weights: Literal["random", "dummy"] = "random"

    @property
    def vocab_size(self) -> int:
        return self.config["vocab_size"]

    @property
    def layouts(self) -> set[str]:
        """The kinds of layers: one means a single KV cache group."""
        return set(self.config.get("layer_types", [FULL]))

    def write(self, root: Path, seed: int = 0) -> Path:
        """The model directory under `root`, built once."""
        path = root / self.name
        if (path / "config.json").exists():
            return path
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.for_model(self.model_type, **self.config)
        config.architectures = [self.architecture]
        path.mkdir(parents=True, exist_ok=True)
        if self.weights == "dummy":
            config.save_pretrained(path)
            return path
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        model.save_pretrained(path)
        return path


_QWEN3_5_BASE: dict[str, Any] = dict(
    vocab_size=2048,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=32,
    linear_num_key_heads=8,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    layer_types=_every_fourth_full(8),
    max_position_embeddings=65536,
    initializer_range=0.08,
    tie_word_embeddings=False,
)

# Production's code path (Qwen3_5ForCausalLM): one full-attention group and
# three GDN groups. A block is a few hundred KB.
GDN = ModelFamily(
    "gdn", "qwen3_5_text", "Qwen3_5ForCausalLM", _QWEN3_5_BASE, hybrid=True
)

# Large GDN states make blocks of about 17 MiB at TP=1 (production's are
# 55.75 MB), so copies take real PCIe time and the lazy walk can fall behind.
GDN_FAT = ModelFamily(
    "gdn_fat",
    "qwen3_5_text",
    "Qwen3_5ForCausalLM",
    _QWEN3_5_BASE
    | dict(
        num_hidden_layers=32,
        layer_types=_every_fourth_full(32),
        head_dim=128,
        linear_num_key_heads=32,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    ),
    hybrid=True,
)

# Speculative decoding with Qwen3.5's own MTP module, as production runs it.
GDN_MTP = ModelFamily(
    "gdn_mtp",
    "qwen3_5_text",
    "Qwen3_5ForCausalLM",
    _QWEN3_5_BASE | dict(mtp_num_hidden_layers=1),
    hybrid=True,
    mtp=True,
    weights="dummy",
)

# Attention only: the tiers without Mamba groups.
DENSE = ModelFamily(
    "dense",
    "qwen3",
    "Qwen3ForCausalLM",
    dict(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=32,
        max_position_embeddings=65536,
        initializer_range=0.08,
        tie_word_embeddings=False,
    ),
)

# Full attention interleaved with a 64-token sliding window.
SWA = ModelFamily(
    "swa",
    "gemma3_text",
    "Gemma3ForCausalLM",
    dict(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=32,
        query_pre_attn_scalar=32,
        sliding_window=64,
        layer_types=[SLIDING, SLIDING, FULL] * 2,
        max_position_embeddings=65536,
        initializer_range=0.08,
    ),
)

FAMILIES: dict[str, ModelFamily] = {
    f.name: f for f in (GDN, GDN_FAT, GDN_MTP, DENSE, SWA)
}
