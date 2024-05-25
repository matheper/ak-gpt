from dataset import load_hf_dataset


# https://huggingface.co/datasets/wikimedia/wikipedia
dataset = load_hf_dataset(
    dataset="wikimedia/wikipedia",
    subset="20231101.pt",
    split="train",
    max_size=200,
)

print(dataset)
print(dataset[0])
