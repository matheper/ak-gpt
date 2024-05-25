from functools import partial

from datasets import Dataset, load_dataset


# https://stackoverflow.com/a/76662982/1704191
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


# https://huggingface.co/datasets/wikimedia/wikipedia
iter_ds = load_dataset(
    "wikimedia/wikipedia", "20231101.pt", split="train", streaming=True
).take(200)

dataset = Dataset.from_generator(
    partial(gen_from_iterable_dataset, iter_ds),
    features=iter_ds.features
)

print(dataset)
print(dataset[0])