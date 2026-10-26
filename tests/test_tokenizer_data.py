import numpy as np
import pytest

from llmzero.data import build_windows, load_lines, split_lines
from llmzero.tokenizer import BOUNDARY, CharTokenizer, TokenizerError


def test_vocab_starts_with_boundary_and_is_sorted(tok):
    assert tok.itos[0] == BOUNDARY
    assert tok.itos[1:] == sorted(tok.itos[1:])


def test_encode_decode_round_trip_with_accents(tok):
    for word in ["belém", "macapá", "rio"]:
        assert tok.decode(tok.encode(word)) == word


def test_unknown_char_is_an_explicit_error(tok):
    with pytest.raises(TokenizerError, match="fora do vocabulário"):
        tok.encode("xyz")


def test_corpus_with_boundary_char_is_rejected():
    with pytest.raises(TokenizerError):
        CharTokenizer.fit(["rio.doce"])


def test_json_round_trip(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    assert CharTokenizer.load(tmp_path / "tok.json").itos == tok.itos


def test_windows_match_hand_computed_example(tok):
    x, y = build_windows(["rio"], tok, block_size=3)
    contexts = [tok.decode(row) for row in x]
    targets = tok.decode(y)
    assert contexts == ["...", "..r", ".ri", "rio"]
    assert targets == "rio."


def test_window_count_is_len_plus_one_per_line(tiny_lines, tok):
    x, y = build_windows(tiny_lines, tok, block_size=4)
    assert len(x) == len(y) == sum(len(s) + 1 for s in tiny_lines)
    assert x.shape[1] == 4


def test_split_is_deterministic_disjoint_and_complete(tiny_lines):
    a = split_lines(tiny_lines * 5, seed=1, val_frac=0.2, test_frac=0.2)
    b = split_lines(tiny_lines * 5, seed=1, val_frac=0.2, test_frac=0.2)
    assert a == b
    assert len(a.train) + len(a.val) + len(a.test) == 40
    assert split_lines(tiny_lines * 5, seed=2, val_frac=0.2, test_frac=0.2) != a


def test_split_rejects_invalid_fractions(tiny_lines):
    with pytest.raises(ValueError):
        split_lines(tiny_lines, seed=0, val_frac=0.6, test_frac=0.5)


def test_bundled_corpus_is_clean():
    lines = load_lines()
    assert len(lines) > 5000
    assert len(set(lines)) == len(lines), "nomes duplicados no corpus"
    assert all(BOUNDARY not in s for s in lines)
    assert all(s == s.lower() == s.strip() for s in lines)


def test_split_has_no_overlap_on_real_corpus():
    s = split_lines(load_lines(), seed=1234)
    assert not set(s.train) & set(s.val)
    assert not set(s.train) & set(s.test)
    assert np.isclose(len(s.val) / (len(s.train) + len(s.val) + len(s.test)), 0.1, atol=0.01)
