import pytest
import torch

from dataset import SimpleDataLoader, split_dataset
from tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    corpus = "abc"
    tokenizer = Tokenizer()
    tokenizer.train(corpus)
    return tokenizer


def test_encode_single_token(tokenizer):
    assert tokenizer.encode("a") == 0


def test_encode_multiple_tokens(tokenizer):
    assert tokenizer.encode("abc") == [0, 1, 2]


def test_encode_unknown_token(tokenizer):
    with pytest.raises(ValueError, match="Unknown token: x"):
        tokenizer.encode("xyz")


def test_encode_empty_string(tokenizer):
    assert tokenizer.encode("") == []


def test_decode_single_token(tokenizer):
    assert tokenizer.decode(0) == "a"


def test_decode_multiple_tokens(tokenizer):
    assert tokenizer.decode([0, 1, 2]) == "abc"


def test_decode_unknown_token(tokenizer):
    with pytest.raises(ValueError, match="Unknown encoded token: 3"):
        tokenizer.decode(3)


def test_decode_empty_list(tokenizer):
    assert tokenizer.decode([]) == ""


def test_split_dataset():
    dataset = list(range(10))

    train, test = split_dataset(dataset)
    assert train == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert test == [9]

    train, test = split_dataset(dataset, split_ratio=0.7)
    assert train == [0, 1, 2, 3, 4, 5, 6]
    assert test == [7, 8, 9]


def test_get_batch():
    dataset = torch.arange(10)
    batch_size = 2
    block_size = 3
    generator = torch.Generator()
    generator.manual_seed(42)

    dataloader = SimpleDataLoader(dataset, batch_size, block_size, generator)
    x, y = dataloader.get_batch()

    expected_x = torch.tensor([[1, 2, 3], [2, 3, 4]])
    expected_y = torch.tensor([[2, 3, 4], [3, 4, 5]])

    assert torch.all(torch.eq(x, expected_x))
    assert torch.all(torch.eq(y, expected_y))
