from functools import partial

from datasets import Dataset, load_dataset


# https://stackoverflow.com/a/76662982/1704191
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def load_hf_dataset(dataset, subset=None, split=None, max_size=0):
    if max_size > 0:
        iter_ds = load_dataset(
            dataset, subset, split=split, streaming=True
        ).take(max_size)
        hf_dataset = Dataset.from_generator(
            partial(gen_from_iterable_dataset, iter_ds),
            features=iter_ds.features,
        )
    else:
        hf_dataset = load_dataset(dataset, subset, split=split)

    return hf_dataset
