from dataset import concat_clean_text, load_hf_dataset
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
    dataset = concat_clean_text(hf_dataset)

    # train tokenizer with the entire dataset
    tokenizer = Tokenizer()
    tokenizer.train(dataset)
    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print(f"Vocabulary mapping: {tokenizer.str_to_int}")

    # test tokenizer with a slice of the dataset
    text = dataset[:100]
    print(f"Text: {text}")
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")


if __name__ == "__main__":
    main()
