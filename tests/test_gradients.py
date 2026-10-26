"""Verificação numérica de gradientes.

Para cada parâmetro p, compara o gradiente analítico (backward) com a
diferença finita central (L(p+h) - L(p-h)) / 2h. Se a derivação à mão estiver
errada, o erro relativo fica ordens de grandeza acima da tolerância.
"""

import numpy as np
import pytest

from llmzero.layers import Embedding, Linear, Tanh
from llmzero.losses import cross_entropy, log_softmax
from llmzero.model import MLPLanguageModel, ModelConfig

H = 1e-6
TOL = 1e-6


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def numeric_grad(f, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + H
        up = f()
        x[i] = old - H
        down = f()
        x[i] = old
        grad[i] = (up - down) / (2 * H)
    return grad


def scalar_loss(out: np.ndarray, weights: np.ndarray) -> float:
    # Perda escalar arbitrária L = sum(out * w): dL/dout = w.
    return float((out * weights).sum())


def test_linear_gradients(rng):
    layer = Linear(5, 4, rng)
    layer.params["bias"][:] = rng.standard_normal(4)
    x = rng.standard_normal((3, 5))
    w = rng.standard_normal((3, 4))
    layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(w)
    assert rel_error(dx, numeric_grad(lambda: scalar_loss(layer.forward(x), w), x)) < TOL
    for name in ("weight", "bias"):
        num = numeric_grad(lambda: scalar_loss(layer.forward(x), w), layer.params[name])
        assert rel_error(layer.grads[name], num) < TOL, name


def test_tanh_gradient(rng):
    layer = Tanh()
    x = rng.standard_normal((4, 6))
    w = rng.standard_normal((4, 6))
    layer.forward(x)
    dx = layer.backward(w)
    assert rel_error(dx, numeric_grad(lambda: scalar_loss(layer.forward(x), w), x)) < TOL


def test_embedding_gradient_accumulates_repeated_ids(rng):
    layer = Embedding(6, 3, rng)
    ids = np.array([[1, 1, 2], [5, 1, 0]])  # id 1 aparece três vezes
    w = rng.standard_normal((2, 3, 3))
    layer.zero_grad()
    layer.forward(ids)
    layer.backward(w)
    num = numeric_grad(lambda: scalar_loss(layer.forward(ids), w), layer.params["weight"])
    assert rel_error(layer.grads["weight"], num) < TOL
    assert np.all(layer.grads["weight"][[3, 4]] == 0), "ids ausentes não recebem gradiente"


def test_cross_entropy_gradient(rng):
    logits = rng.standard_normal((5, 7))
    targets = rng.integers(0, 7, size=5)
    _, dlogits = cross_entropy(logits, targets)
    num = numeric_grad(lambda: cross_entropy(logits, targets)[0], logits)
    assert rel_error(dlogits, num) < TOL


def test_log_softmax_is_stable_for_huge_logits():
    logits = np.array([[1000.0, 0.0, -1000.0]])
    out = log_softmax(logits)
    assert np.all(np.isfinite(out))
    assert np.isclose(np.exp(out).sum(), 1.0)


@pytest.mark.parametrize("block_size", [1, 3])
def test_full_model_gradients(block_size):
    model = MLPLanguageModel(ModelConfig(vocab_size=9, block_size=block_size, embed_dim=4, hidden_dim=6), seed=3)
    # Saída maior que a inicialização padrão para que o gradiente não seja minúsculo.
    model.head.params["weight"] *= 50
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 9, size=(7, block_size))
    targets = rng.integers(0, 9, size=7)
    model.train_step_grads(ids, targets)
    for name, param, grad in model.named_parameters():
        num = numeric_grad(lambda: model.loss(ids, targets)[0], param)
        assert rel_error(grad, num) < 1e-5, name


def test_backward_before_forward_is_an_error(rng):
    with pytest.raises(RuntimeError, match="antes de forward"):
        Linear(2, 2, rng).backward(np.zeros((1, 2)))
