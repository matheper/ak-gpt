from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        embed_size: int,
        block_size: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # added to model's state_dict but not the model's parameters
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * c**-0.5  # scaled dot-product
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        v = self.value(x)  # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        block_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embed_size, block_size, embed_size // num_heads, dropout
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        block_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_heads = MultiHeadAttention(
            embed_size, block_size, num_heads, dropout
        )
        self.feed_forward = FeedForward(embed_size, dropout)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_heads(self.layernorm1(x))
        x = x + self.feed_forward(self.layernorm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        block_size: int,
        num_att_blocks: int,
        num_heads: int,
        dropout: float,
    ):
        """Language model based on the Transformer architecture.
        The model predicts the next token given the current token.
        B = batch size, T = block size, C = vocab size
        """
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        att_blocks = [
            AttentionBlock(embed_size, block_size, num_heads, dropout)
            for _ in range(num_att_blocks)
        ]
        self.attention_blocks = nn.Sequential(*att_blocks)
        self.feed_forward = FeedForward(embed_size, dropout)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape  # batch size, block size
        token_embeddings = self.token_embedding(idx)  # (B, T, C)
        positional_embeddings = self.positional_embedding(
            torch.arange(t, device="cpu")  # FIXME: parameterize device
        )  # (T, C)
        # (B, T, C) where C = embed size
        x = token_embeddings + positional_embeddings
        x = self.attention_blocks(x)  # (B, T, C) where C = head size
        logits = self.linear(x)  # (B, T, C) where C = vocab size

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx shape (B, T)
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size :]  # crop to block size
            logits, loss = self(idx_crop)  # get the predictions
            logits = logits[
                :, -1, :
            ]  # focus only on the last time step, becomes (B, C)
            probs = F.softmax(
                logits, dim=-1
            )  # softmax to get probabilities, (B, C)
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # sample from the distribution (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # append sampled index to the running sequence, (B, T + 1)
        return idx


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        """Simple bigram language model with a token embedding table.
        The model predicts the next token given the current token.
        B = batch size, T = block size, C = vocab size
        """
        super().__init__()
        # lookup table of shape (vocab_size, vocab_size) with logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx shape (B, T)
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # get the predictions
            logits = logits[
                :, -1, :
            ]  # focus only on the last time step, becomes (B, C)
            probs = F.softmax(
                logits, dim=-1
            )  # softmax to get probabilities, (B, C)
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # sample from the distribution (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # append sampled index to the running sequence, (B, T + 1)
        return idx
