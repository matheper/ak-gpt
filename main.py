import torch

from dataset import (
    SimpleDataLoader,
    concat_and_clean_text,
    load_hf_dataset,
    split_dataset,
)
from tokenizer import Tokenizer


def main():
    # load a slice of wikipedia pt
    hf_dataset = load_hf_dataset(
        dataset="wikimedia/wikipedia",
        subset="20231101.pt",
        split="train",
        max_size=200,
    )

    print(hf_dataset)

    # concatenate and clean text leaving only latin characters
    dataset = concat_and_clean_text(hf_dataset)

    # train tokenizer with the entire dataset
    tokenizer = Tokenizer()
    tokenizer.train(dataset)
    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print(f"Vocabulary mapping:\n{tokenizer.str_to_int}\n")

    # test tokenizer with a slice of the dataset
    text = dataset[:25]
    print(f"Text: {text}")
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # load the entire encoded dataset into a torch tensor
    data = torch.tensor(tokenizer.encode(dataset), dtype=torch.long)
    print(f"Data shape: {data.shape}")

    # split dataset into train and validation
    split_ratio = 0.9
    train_data, val_data = split_dataset(data, split_ratio=split_ratio)
    print(f"Train shape: {train_data.shape}")
    print(f"Validation shape: {val_data.shape}")

    train_gen = torch.Generator()
    train_gen.manual_seed(42)
    val_gen = torch.Generator()
    val_gen.manual_seed(42)
    batch_size = 4  # number of independent sequences processed in parallel
    block_size = 8  # number of tokens in each sequence

    train_dl = SimpleDataLoader(
        train_data,
        batch_size=batch_size,
        block_size=block_size,
        generator=train_gen,
    )
    val_dl = SimpleDataLoader(
        val_data,
        batch_size=batch_size,
        block_size=block_size,
        generator=val_gen,
    )


if __name__ == "__main__":
    main()
