import numpy as np
import pytest

from llmzero.tokenizer import CharTokenizer

TINY_CORPUS = ["rio", "recife", "natal", "belém", "palmas", "macapá", "aracaju", "manaus"]


@pytest.fixture
def tiny_lines() -> list[str]:
    return list(TINY_CORPUS)


@pytest.fixture
def tok(tiny_lines) -> CharTokenizer:
    return CharTokenizer.fit(tiny_lines)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
