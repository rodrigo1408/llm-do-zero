"""Camadas com forward e backward escritos à mão.

Contrato de cada camada:

* ``forward(x)`` calcula a saída e guarda em cache o necessário para o backward;
* ``backward(dout)`` recebe dL/d(saída), *acumula* dL/d(parâmetro) em ``grads``
  e devolve dL/d(entrada);
* ``params`` e ``grads`` são dicionários com as mesmas chaves e formatos.

Os testes em ``tests/test_gradients.py`` comparam cada backward com
diferenças finitas centrais.
"""

from __future__ import annotations

import numpy as np


class Layer:
    params: dict[str, np.ndarray]
    grads: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.params = {}
        self.grads = {}

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g.fill(0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError


def _require_cache(cache: object | None, layer: str) -> None:
    if cache is None:
        raise RuntimeError(f"{layer}.backward chamado antes de forward")


class Embedding(Layer):
    """Tabela de consulta: id inteiro -> vetor denso aprendido."""

    def __init__(self, vocab_size: int, dim: int, rng: np.random.Generator):
        super().__init__()
        self.params["weight"] = rng.standard_normal((vocab_size, dim))
        self.grads["weight"] = np.zeros_like(self.params["weight"])
        self._ids: np.ndarray | None = None

    def forward(self, ids: np.ndarray) -> np.ndarray:
        self._ids = ids
        return self.params["weight"][ids]

    def backward(self, dout: np.ndarray) -> None:
        _require_cache(self._ids, "Embedding")
        # O mesmo id pode aparecer várias vezes no lote: os gradientes somam.
        np.add.at(self.grads["weight"], self._ids, dout)
        return None  # ids são discretos: não há gradiente para a entrada


class Linear(Layer):
    """y = x W + b"""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator, gain: float = 1.0, bias: bool = True):
        super().__init__()
        # Escala de Kaiming: mantém a variância das ativações estável entre camadas.
        self.params["weight"] = rng.standard_normal((fan_in, fan_out)) * gain / np.sqrt(fan_in)
        self.grads["weight"] = np.zeros_like(self.params["weight"])
        if bias:
            self.params["bias"] = np.zeros(fan_out)
            self.grads["bias"] = np.zeros(fan_out)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        out = x @ self.params["weight"]
        if "bias" in self.params:
            out = out + self.params["bias"]
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        _require_cache(self._x, "Linear")
        self.grads["weight"] += self._x.T @ dout
        if "bias" in self.params:
            self.grads["bias"] += dout.sum(axis=0)
        return dout @ self.params["weight"].T


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        _require_cache(self._out, "Tanh")
        return dout * (1.0 - self._out**2)
