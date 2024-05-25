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

    def forward(self, idx, targets=None):
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
