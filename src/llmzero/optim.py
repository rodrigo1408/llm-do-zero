"""Otimizadores: SGD e AdamW, atualizando os parâmetros no lugar."""

from __future__ import annotations

import numpy as np

Params = list[tuple[str, np.ndarray, np.ndarray]]


class SGD:
    def __init__(self, params: Params, lr: float, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for _, p, g in self.params:
            if self.weight_decay:
                p -= self.lr * self.weight_decay * p
            p -= self.lr * g


class AdamW:
    """Adam com weight decay desacoplado (Loshchilov & Hutter, 2019)."""

    def __init__(
        self,
        params: Params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for _, p, _ in params]
        self.v = [np.zeros_like(p) for _, p, _ in params]

    def step(self) -> None:
        self.t += 1
        for (_, p, g), m, v in zip(self.params, self.m, self.v, strict=True):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            if self.weight_decay:
                p -= self.lr * self.weight_decay * p
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, params: Params, lr: float, weight_decay: float = 0.0) -> SGD | AdamW:
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"otimizador desconhecido: {name!r} (use 'sgd' ou 'adamw')")
