from dataset import concat_clean_text, load_hf_dataset


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
print(sorted(list(set(dataset))))
