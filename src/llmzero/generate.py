"""Geração de texto por amostragem autorregressiva."""

from __future__ import annotations

import numpy as np

from .losses import softmax
from .model import MLPLanguageModel
from .tokenizer import CharTokenizer


def sample(
    model: MLPLanguageModel,
    tok: CharTokenizer,
    n: int,
    seed: int,
    temperature: float = 1.0,
    max_len: int = 40,
) -> list[str]:
    """Gera ``n`` sequências, caractere a caractere, até o token de fronteira.

    ``temperature`` < 1 concentra a distribuição (saídas mais conservadoras);
    > 1 a achata (mais variedade, mais erros).
    """
    if temperature <= 0:
        raise ValueError("temperature deve ser > 0")
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for _ in range(n):
        context = [0] * model.config.block_size
        ids: list[int] = []
        while len(ids) < max_len:
            logits = model.forward(np.array([context])) / temperature
            nxt = int(rng.choice(tok.vocab_size, p=softmax(logits)[0]))
            if nxt == 0:
                break
            ids.append(nxt)
            context = [*context[1:], nxt]
        out.append(tok.decode(ids))
    return out


def novelty(samples: list[str], train: list[str], val: list[str]) -> dict[str, float]:
    """Que fração das amostras copia o treino, acerta um nome real não visto, ou é inédita."""
    if not samples:
        raise ValueError("lista de amostras vazia")
    tr, va = set(train), set(val)
    n = len(samples)
    return {
        "copied_from_train": sum(s in tr for s in samples) / n,
        "real_unseen_val": sum(s in va and s not in tr for s in samples) / n,
        "new": sum(s not in tr and s not in va for s in samples) / n,
    }
