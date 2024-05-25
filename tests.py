import pytest

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
