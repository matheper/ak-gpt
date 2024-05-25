from typing import List


class Tokenizer:
    def __init__(self):
        self.str_to_int = {}
        self.int_to_str = {}
        self.vocabulary_size = 0

    def train(self, text: List[str]) -> None:
        """Train the tokenizer with a list of strings."""
        corpus = "".join(text)
        unique_tokens = sorted(list(set(corpus)))
        for idx, token in enumerate(unique_tokens):
            self.str_to_int[token] = idx
            self.int_to_str[idx] = token
        self.vocabulary = unique_tokens
        self.vocabulary_size = len(unique_tokens)

    def tokenize(self, text: str) -> List[str]:
        """Split a string into a list of tokens (characters).
        Raises a ValueError if a token is not in the vocabulary.
        """
        tokens = []
        for char in text:
            if char not in self.str_to_int:
                raise ValueError(f"Token desconhecido: {char}")
            tokens.append(char)
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode (including tokenization) a string into a list of integers.
        If a single token is encoded, return the integer directly.
        """
        encoded_tokens = []
        for token in self.tokenize(text):
            encoded_token = self.str_to_int[token]
            encoded_tokens.append(encoded_token)
        if len(encoded_tokens) == 1:
            encoded_tokens = encoded_tokens[0]
        return encoded_tokens
