import torch
import logging
from tokenizer import Tokenizer

from dataset import (
    SimpleDataLoader,
    concat_and_clean_text,
    load_hf_dataset,
    split_dataset,
)


logging.basicConfig(level=logging.INFO)


def main():
    # load a slice of wikipedia pt
    hf_dataset = load_hf_dataset(
        dataset="wikimedia/wikipedia",
        subset="20231101.pt",
        split="train",
        max_size=200,
    )

    logging.info(hf_dataset)

    # concatenate and clean text leaving only latin characters
    dataset = concat_and_clean_text(hf_dataset)

    # train tokenizer with the entire dataset
    tokenizer = Tokenizer()
    tokenizer.train(dataset)
    logging.info(f"Vocabulary size: {tokenizer.vocabulary_size}")
    logging.info(f"Vocabulary mapping:\n{tokenizer.str_to_int}\n")

    # test tokenizer with a slice of the dataset
    text = dataset[:25]
    logging.info(f"Text: {text}")
    tokens = tokenizer.tokenize(text)
    logging.info(f"Tokens: {tokens}")
    encoded = tokenizer.encode(text)
    logging.info(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    logging.info(f"Decoded: {decoded}")

    # load the entire encoded dataset into a torch tensor
    data = torch.tensor(tokenizer.encode(dataset), dtype=torch.long)
    logging.info(f"Data shape: {data.shape}")

    # split dataset into train and validation
    split_ratio = 0.9
    train_data, val_data = split_dataset(data, split_ratio=split_ratio)
    logging.info(f"Train shape: {train_data.shape}")
    logging.info(f"Validation shape: {val_data.shape}")

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

    x_batch, y_batch = train_dl.get_batch()
    logging.info(f"X batch: {x_batch}")
    logging.info(f"Y batch: {y_batch}")


if __name__ == "__main__":
    main()
