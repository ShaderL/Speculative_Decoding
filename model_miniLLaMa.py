import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 128

    num_layers: int = 6
    d_model: int = 384
    d_ffn: int = 1024
    num_heads: int = 4

    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        self.d_head = self.d_model // self.num_heads

        if self.d_head % 2 != 0:
            raise ValueError(
                f"d_head ({self.d_head}) must be even for RoPE"
            )


def apply_rope(x):
    B, seq_len, d_head = x.shape
    assert d_head % 2 == 0

    position = torch.arange(
        seq_len,
        device=x.device
    )

    freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                d_head,
                2,
                device=x.device
            ) / d_head
        )
    )

    theta = position[:, None] * freq[None, :]

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    x_even = x[:, :, 0::2]
    x_odd = x[:, :, 1::2]

    x_even_new = x_even * cos - x_odd * sin
    x_odd_new = x_even * sin + x_odd * cos

    x_rope = torch.stack(
        [x_even_new, x_odd_new],
        dim=-1
    ).flatten(-2)

    return x_rope


class MaskMultiHeadAttentionRoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_head

        # Q / K / V projection
        #
        # [B, L, d_model]
        #       ↓
        # [B, L, d_model]
        #
        # 后面再 reshape 成多个 head
        self.wq = nn.Parameter(
            torch.randn(self.d_model, self.d_model)
        )

        self.wk = nn.Parameter(
            torch.randn(self.d_model, self.d_model)
        )

        self.wv = nn.Parameter(
            torch.randn(self.d_model, self.d_model)
        )

        # Output projection
        self.wo = nn.Parameter(
            torch.randn(self.d_model, self.d_model)
        )

        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)
        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wo)

    def forward(self, x):
        B, L, _ = x.shape

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        q = q.view(
            B,
            L,
            self.num_heads,
            self.d_head
        )

        k = k.view(
            B,
            L,
            self.num_heads,
            self.d_head
        )

        v = v.view(
            B,
            L,
            self.num_heads,
            self.d_head
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.reshape(
            B * self.num_heads,
            L,
            self.d_head
        )

        k = k.reshape(
            B * self.num_heads,
            L,
            self.d_head
        )

        q = apply_rope(q)
        k = apply_rope(k)

        q = q.view(
            B,
            self.num_heads,
            L,
            self.d_head
        )

        k = k.view(
            B,
            self.num_heads,
            L,
            self.d_head
        )

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        mask = torch.triu(
            torch.ones(
                L,
                L,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        attention_scores = attention_scores.masked_fill(
            mask,
            float("-inf")
        )

        attention_probs = torch.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_probs @ v
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            B,
            L,
            self.d_model
        )
        output = output @ self.wo

        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w1 = nn.Parameter(
            torch.randn(d_model, d_ffn)
        )

        self.w2 = nn.Parameter(
            torch.randn(d_ffn, d_model)
        )

        self.w3 = nn.Parameter(
            torch.randn(d_model, d_ffn)
        )

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x):
        x1 = F.silu(x @ self.w1)
        y = x @ self.w3
        z = x1 * y
        x = z @ self.w2
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x):
        rms = torch.sqrt(
            (x ** 2).mean(
                dim=-1,
                keepdim=True
            )
            + self.eps
        )
        x = (x / rms) * self.gamma
        return x


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.wlm = nn.Parameter(
            torch.randn(d_model, vocab_size)
        )

        nn.init.xavier_uniform_(self.wlm)

    def forward(self, x):
        x = x @ self.wlm
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.rmsnorm1 = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.rmsnorm2 = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.mmha = MaskMultiHeadAttentionRoPE(
            config
        )

        self.swiglu = SwiGLU(
            config.d_model,
            config.d_ffn
        )

    def forward(self, x):
        x = x + self.mmha(
            self.rmsnorm1(x)
        )
        x = x + self.swiglu(
            self.rmsnorm2(x)
        )
        return x


class MiniLLaMA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token Embedding
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.num_layers)
        ])

        # Final RMSNorm
        self.rmsnorm = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        # LM Head
        self.lm_head = LMHead(
            config.d_model,
            config.vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.rmsnorm(x)
        x = self.lm_head(x)
        return x

