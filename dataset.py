import re
from functools import partial
from typing import Optional

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


def concat_clean_text(dataset: Dataset) -> str:
    """Concatenate and clean text from a Hugging Face Dataset."""
    concatenated_text = "\n".join([d["text"] for d in dataset])
    return latin_clean_text(concatenated_text)
