import re
from functools import partial
from typing import Any, Optional, Tuple

import torch
from datasets import Dataset, load_dataset


def _gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def load_hf_dataset(
    dataset: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    max_size: int = 0,
) -> Dataset:
    """Load a Hugging Face Dataset with the option to partially download it.
    https://stackoverflow.com/a/76662982/1704191
    """
    if max_size > 0:
        iter_ds = load_dataset(
            dataset, subset, split=split, streaming=True
        ).take(max_size)
        hf_dataset = Dataset.from_generator(
            partial(_gen_from_iterable_dataset, iter_ds),
            features=iter_ds.features,
        )
    else:
        hf_dataset = load_dataset(dataset, subset, split=split)

    return hf_dataset


def latin_clean_text(text: str) -> str:
    """Clean text by removing characters that are not in
    the range from ' '(space) to 'ÿ' or are '\n' or '\t'.
    Basic Latin + Latin-1 Supplement characters from Unicode.
    https://en.wikipedia.org/wiki/List_of_Unicode_characters
    """
    pattern = r"[ -ÿ\t\n]+"
    cleaned_text = "".join(re.findall(pattern, text))
    return cleaned_text


def concat_and_clean_text(dataset: Dataset) -> str:
    """Concatenate and clean text from a Hugging Face Dataset."""
    concatenated_text = "\n".join([d["text"] for d in dataset])
    return latin_clean_text(concatenated_text)


def split_dataset(
    dataset: torch.Tensor, split_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a dataset into two parts given a split ratio. Default is 90/10."""
    split_idx = int(len(dataset) * split_ratio)
    return dataset[:split_idx], dataset[split_idx:]


class SimpleDataLoader:
    def __init__(
        self,
        dataset: torch.Tensor,
        batch_size: int,
        block_size: int,
        generator: torch.Generator = None,
    ):
        """Initialize a DataLoader for next token prediction.
        Batch size is the number of independent sequences of text.
        Block size is the number of tokens in each sequence of text.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.generator = generator

    def get_batch(self):
        """Get a batch of data for next token prediction.
        X, Y pairs are created by shifting the data by one token.
        Ex.: dataset = [0, 1, 2, 3, 4], batch_size = 2, block_size = 2
        X = [[0, 1], [1, 2]]
        Y = [[1, 2], [2, 3]]
        """
        idx = torch.randint(
            len(self.dataset) - self.block_size,
            (self.batch_size,),
            generator=self.generator,
        )
        x = torch.stack([self.dataset[i : i + self.block_size] for i in idx])
        y = torch.stack(
            [self.dataset[i + 1 : i + self.block_size + 1] for i in idx]
        )
        return x, y
