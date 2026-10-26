"""Função de perda: entropia cruzada sobre a distribuição do próximo token."""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    # Subtrair o máximo evita overflow em exp() sem alterar o resultado.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Média de -log p(alvo) e o gradiente dL/dlogits.

    Para softmax seguida de entropia cruzada o gradiente tem forma fechada:
    ``(softmax(logits) - one_hot(alvo)) / N``.
    """
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError(f"formatos incompatíveis: logits {logits.shape}, alvos {targets.shape}")
    n = logits.shape[0]
    logp = log_softmax(logits)
    loss = float(-logp[np.arange(n), targets].mean())
    dlogits = np.exp(logp)
    dlogits[np.arange(n), targets] -= 1.0
    return loss, dlogits / n
