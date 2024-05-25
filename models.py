from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
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
