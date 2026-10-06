"""A decoder-only Transformer, with the attention math written explicitly."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if (
            not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1).")


class CausalSelfAttention(nn.Module):
    """Let each position read only itself and earlier positions."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        # Each head independently compares queries with keys, then uses the
        # resulting probabilities to average value vectors.
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_head, self.head_size
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(
            ~self.causal_mask[:sequence_length, :sequence_length], float("-inf")
        )
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_size
        )
        return self.output_dropout(self.projection(output))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class GPT(nn.Module):
    """Predict the next character at every input position.

    All weights start randomly. During training, targets should be the input
    text shifted forward by one character.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.n_layer))
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("Input must have shape (batch, sequence) with both sizes positive.")
        sequence_length = idx.shape[1]
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Input length {sequence_length} exceeds block_size {self.config.block_size}."
            )
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("Targets must have the same shape as the input.")
        positions = torch.arange(sequence_length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(self.embedding_dropout(x))
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("Generation needs a nonempty prompt with shape (batch, sequence).")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a nonnegative integer.")
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number.")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None.")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Keep the entire output but show only the most recent context
                # to the model when the text exceeds its learned window.
                logits, _ = self(idx[:, -self.config.block_size :])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(
                        logits, min(top_k, self.config.vocab_size), dim=-1
                    ).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)
        return idx
