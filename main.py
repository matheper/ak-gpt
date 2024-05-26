import logging

import torch

from dataset import (
    SimpleDataLoader,
    concat_and_clean_text,
    load_hf_dataset,
    split_dataset,
)
from models import BigramLanguageModel, TransformerModel
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataloaders: dict[str, SimpleDataLoader],
    eval_iters: int,
) -> dict[str, float]:
    """Estimate the loss of a model on a dict of {split: dataloader}.
    The loss is estimated by averaging the loss over `eval_iters` batches.
    """
    out = {}
    model.eval()
    for split, dataloader in dataloaders.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = dataloader.get_batch()
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    # hyperparameters
    seed = 42  # random seed for reproducibility
    split_ratio = 0.9  # train/validation split ratio
    model = "transformer"  # model type: bigram or transformer
    batch_size = 4  # number of independent sequences processed in parallel
    block_size = 8  # number of tokens in each sequence
    learning_rate = 1e-2  # learning rate
    train_iters = 10000  # number of training iterations
    eval_interval = 100  # evaluate the model every 100 steps
    eval_iters = 100  # number of iterations to estimate the loss
    embed_size = 32  # embedding size for the transformer model
    num_heads = 4  # number of attention heads for the transformer model

    torch.manual_seed(seed)

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
    # data = torch.tensor(tokenizer.encode(dataset), dtype=torch.long)
    data = tokenizer(dataset, return_tensors="pt")
    logging.info(f"Data shape: {data.shape}")

    # split dataset into train and validation
    train_data, val_data = split_dataset(data, split_ratio=split_ratio)
    logging.info(f"Train shape: {train_data.shape}")
    logging.info(f"Validation shape: {val_data.shape}")

    dataloaders = dict()
    dataloaders["train"] = SimpleDataLoader(
        train_data,
        batch_size=batch_size,
        block_size=block_size,
    )
    dataloaders["validation"] = SimpleDataLoader(
        val_data,
        batch_size=batch_size,
        block_size=block_size,
    )

    x_batch, y_batch = dataloaders["train"].get_batch()
    logging.info(f"X batch: {x_batch}")
    logging.info(f"Y batch: {y_batch}")

    if model == "bigram":
        model = BigramLanguageModel(tokenizer.vocabulary_size)
    elif model == "transformer":
        model = TransformerModel(
            vocab_size=tokenizer.vocabulary_size,
            embed_size=embed_size,
            block_size=block_size,
            num_heads=num_heads,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    logits, loss = model(x_batch, y_batch)
    logging.info(f"Logits shape: {logits.shape}")
    # expected loss with no training is -log(1 / vocab_size)
    # -log(1 / 158) = 5.06259503303
    logging.info(f"Loss: {loss}")

    # Generate some text from the bigram model. Start with a single token "A".
    inputs = tokenizer(["A"], return_tensors="pt").reshape(1, 1)
    new_tokens = model.generate(inputs, max_new_tokens=100)[0].tolist()
    logging.info(f"Generated text: {tokenizer.decode(new_tokens)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(train_iters):
        x_batch, y_batch = dataloaders["train"].get_batch()
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_interval == 0:
            losses = estimate_loss(model, dataloaders, eval_iters)
            logging.info(
                f"Step: {step:4d} Train Loss: {losses["train"]:.16f} "
                f"Validation Loss: {losses["validation"]:.16f}"
            )

    # Generate some text from the bigram model. Start with a single token "A".
    inputs = tokenizer(["A"], return_tensors="pt").reshape(1, 1)
    new_tokens = model.generate(inputs, max_new_tokens=100)[0].tolist()
    logging.info(tokenizer.decode(new_tokens))


if __name__ == "__main__":
    main()

# Train Loss: 2.4773683547973633 Validation Loss: 2.4583020210266113 - Bigram
# Train Loss: 2.5519707202911377 Validation Loss: 2.5332398414611816 - Transformers Single Head
# Train Loss: 2.4278442859649658 Validation Loss: 2.4253225326538086 - Transformers 4 Multi Head
# Train Loss: 2.4477648735046387 Validation Loss: 2.3981783390045166 - Transformers Multi Head + ff
