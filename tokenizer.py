from typing import List


class Tokenizer:
    def __init__(self):
        self.str_to_int = {}
        self.int_to_str = {}
        self.vocabulary_size = 0

    def train(self, dataset: List[str]) -> None:
        """Train the tokenizer with a list of strings."""
        corpus = "".join(dataset)
        unique_tokens = sorted(list(set(corpus)))
        for idx, token in enumerate(unique_tokens):
            self.str_to_int[token] = idx
            self.int_to_str[idx] = token
        self.vocabulary = unique_tokens
        self.vocabulary_size = len(unique_tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a string and return a list of tokens.
        Raises a ValueError if a token is not in the vocabulary.
        """
        tokens = []
        for char in text:
            if char not in self.str_to_int:
                raise ValueError(f"Token desconhecido: {char}")
            tokens.append(char)
        return tokens
