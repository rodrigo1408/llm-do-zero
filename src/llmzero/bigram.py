"""Linha de base sem rede neural: bigramas por contagem.

p(próximo | atual) = (contagem(atual, próximo) + alpha) / (contagem(atual) + alpha·V)

Serve de régua: se o MLP não supera este modelo na validação, ele não está
aproveitando o contexto mais longo.
"""

from __future__ import annotations

import numpy as np

from .tokenizer import CharTokenizer


class BigramModel:
    def __init__(self, vocab_size: int, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha deve ser > 0 (sem suavização, pares não vistos têm perda infinita)")
        self.alpha = alpha
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    def fit(self, lines: list[str], tok: CharTokenizer) -> BigramModel:
        for line in lines:
            ids = [0, *tok.encode(line), 0]
            for a, b in zip(ids, ids[1:], strict=False):
                self.counts[a, b] += 1
        return self

    @property
    def probs(self) -> np.ndarray:
        smoothed = self.counts + self.alpha
        return smoothed / smoothed.sum(axis=1, keepdims=True)

    def nll(self, lines: list[str], tok: CharTokenizer) -> float:
        """Perda média (nats por token) — comparável com a do MLP."""
        logp = np.log(self.probs)
        total, n = 0.0, 0
        for line in lines:
            ids = [0, *tok.encode(line), 0]
            for a, b in zip(ids, ids[1:], strict=False):
                total -= logp[a, b]
                n += 1
        return total / n
