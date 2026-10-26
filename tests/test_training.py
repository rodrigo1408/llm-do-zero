import math

import numpy as np
import pytest

from llmzero.bigram import BigramModel
from llmzero.data import Split, build_windows
from llmzero.generate import novelty, sample
from llmzero.model import MLPLanguageModel, ModelConfig
from llmzero.optim import SGD, AdamW, make_optimizer
from llmzero.train import TrainConfig, evaluate, train


def tiny_split(lines):
    return Split(train=lines[:6], val=lines[6:], test=[])


def test_initial_loss_is_close_to_uniform(tiny_lines, tok):
    model = MLPLanguageModel(ModelConfig(tok.vocab_size, block_size=3), seed=0)
    x, y = build_windows(tiny_lines, tok, 3)
    assert abs(evaluate(model, x, y) - math.log(tok.vocab_size)) < 0.05


def test_model_can_memorize_a_tiny_corpus(tiny_lines, tok):
    """Sanidade clássica: com capacidade sobrando, a perda de treino tem de ir a quase zero."""
    cfg = TrainConfig(block_size=3, hidden_dim=64, steps=1500, eval_every=500, batch_size=32, lr=1e-2)
    res = train(cfg, Split(train=tiny_lines, val=tiny_lines, test=[]), tok)
    assert res.history.steps[0] == 0
    assert abs(res.history.train_loss[0] - math.log(tok.vocab_size)) < 0.05
    # Após "..." (e após prefixos comuns como "ma") há mais de um próximo caractere
    # correto, então o piso teórico não é zero; abaixo de 0.5 o resto foi memorizado.
    assert res.final_train_loss < 0.5


def test_training_is_reproducible_with_same_seed(tiny_lines, tok):
    cfg = TrainConfig(block_size=2, hidden_dim=16, steps=60, eval_every=20)
    a = train(cfg, tiny_split(tiny_lines), tok)
    b = train(cfg, tiny_split(tiny_lines), tok)
    assert a.history.train_loss == b.history.train_loss
    for (_, pa, _), (_, pb, _) in zip(a.model.named_parameters(), b.model.named_parameters(), strict=True):
        np.testing.assert_array_equal(pa, pb)


def test_huge_learning_rate_is_reported_as_divergence(tiny_lines, tok):
    cfg = TrainConfig(block_size=2, hidden_dim=16, steps=300, eval_every=10, optimizer="sgd", lr=1e6)
    res = train(cfg, tiny_split(tiny_lines), tok)
    assert res.diverged
    assert math.isnan(res.final_train_loss) and math.isnan(res.final_val_loss)


def test_adamw_and_sgd_reduce_a_quadratic():
    for opt_cls, lr in ((SGD, 0.1), (AdamW, 0.1)):
        p = np.array([3.0, -2.0])
        g = np.zeros(2)
        opt = opt_cls([("p", p, g)], lr=lr)
        for _ in range(200):
            g[...] = 2 * p  # gradiente de sum(p^2)
            opt.step()
        assert np.all(np.abs(p) < 1e-2), opt_cls.__name__


def test_weight_decay_shrinks_parameters_without_gradient():
    p = np.ones(3)
    opt = AdamW([("p", p, np.zeros(3))], lr=0.1, weight_decay=0.5)
    opt.step()
    np.testing.assert_allclose(p, 0.95)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="desconhecido"):
        make_optimizer("rmsprop", [], lr=0.1)


def test_checkpoint_round_trip(tiny_lines, tok, tmp_path):
    cfg = TrainConfig(block_size=3, hidden_dim=16, steps=50, eval_every=25)
    model = train(cfg, tiny_split(tiny_lines), tok).model
    model.save(tmp_path / "m.npz")
    loaded = MLPLanguageModel.load(tmp_path / "m.npz")
    x, _ = build_windows(tiny_lines, tok, 3)
    np.testing.assert_array_equal(model.forward(x), loaded.forward(x))


def test_generation_is_seeded_and_uses_vocabulary(tiny_lines, tok):
    cfg = TrainConfig(block_size=3, hidden_dim=32, steps=400, eval_every=200)
    model = train(cfg, tiny_split(tiny_lines), tok).model
    a = sample(model, tok, n=5, seed=11)
    assert a == sample(model, tok, n=5, seed=11)
    assert all(set(s) <= set(tok.itos[1:]) for s in a)
    with pytest.raises(ValueError):
        sample(model, tok, n=1, seed=0, temperature=0)


def test_novelty_fractions():
    out = novelty(["rio", "natal", "xique"], train=["rio"], val=["natal"])
    assert out == pytest.approx({"copied_from_train": 1 / 3, "real_unseen_val": 1 / 3, "new": 1 / 3})


def test_bigram_beats_uniform_and_counts_boundaries(tiny_lines, tok):
    bg = BigramModel(tok.vocab_size).fit(tiny_lines, tok)
    assert bg.counts[0].sum() == len(tiny_lines)  # uma transição a partir do início por nome
    assert bg.nll(tiny_lines, tok) < math.log(tok.vocab_size)
    np.testing.assert_allclose(bg.probs.sum(axis=1), 1.0)
