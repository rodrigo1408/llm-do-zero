"""Modelo de linguagem MLP (no estilo de Bengio et al., 2003).

::

    ids (B, T) ─► Embedding ─► (B, T, E) ─► concatena ─► (B, T·E)
               ─► Linear ─► tanh ─► (B, H) ─► Linear ─► logits (B, V)

O modelo vê os ``T = block_size`` caracteres anteriores e produz uma
distribuição sobre o próximo caractere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .layers import Embedding, Layer, Linear, Tanh
from .losses import cross_entropy

TANH_GAIN = 5.0 / 3.0  # ganho recomendado para tanh (inicialização de Kaiming)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 4
    embed_dim: int = 12
    hidden_dim: int = 200

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} deve ser um inteiro >= 1 (recebido {value!r})")


class MLPLanguageModel:
    def __init__(self, config: ModelConfig, seed: int = 0):
        self.config = config
        rng = np.random.default_rng(seed)
        c = config
        self.embedding = Embedding(c.vocab_size, c.embed_dim, rng)
        self.hidden = Linear(c.block_size * c.embed_dim, c.hidden_dim, rng, gain=TANH_GAIN)
        self.act = Tanh()
        # Saída quase nula no início: a previsão inicial fica perto da uniforme,
        # e a perda inicial perto de ln(V) em vez de um valor arbitrariamente alto.
        self.head = Linear(c.hidden_dim, c.vocab_size, rng, gain=0.01)
        self.layers: dict[str, Layer] = {"embedding": self.embedding, "hidden": self.hidden, "head": self.head}

    # -- parâmetros ---------------------------------------------------------------
    def named_parameters(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(nome, parâmetro, gradiente) em ordem estável."""
        return [
            (f"{lname}.{pname}", layer.params[pname], layer.grads[pname])
            for lname, layer in self.layers.items()
            for pname in layer.params
        ]

    def num_parameters(self) -> int:
        return sum(p.size for _, p, _ in self.named_parameters())

    def zero_grad(self) -> None:
        for layer in self.layers.values():
            layer.zero_grad()

    # -- forward / backward ---------------------------------------------------------
    def forward(self, ids: np.ndarray) -> np.ndarray:
        if ids.ndim != 2 or ids.shape[1] != self.config.block_size:
            raise ValueError(f"esperado (B, {self.config.block_size}), recebido {ids.shape}")
        emb = self.embedding.forward(ids)  # (B, T, E)
        flat = emb.reshape(ids.shape[0], -1)  # (B, T·E)
        h = self.act.forward(self.hidden.forward(flat))
        return self.head.forward(h)

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        return cross_entropy(self.forward(ids), targets)

    def backward(self, dlogits: np.ndarray) -> None:
        """Regra da cadeia, da saída para a entrada (ordem inversa do forward)."""
        dh = self.head.backward(dlogits)
        dpre = self.act.backward(dh)
        dflat = self.hidden.backward(dpre)
        b, t = dflat.shape[0], self.config.block_size
        self.embedding.backward(dflat.reshape(b, t, self.config.embed_dim))

    def train_step_grads(self, ids: np.ndarray, targets: np.ndarray) -> float:
        """zero_grad + forward + loss + backward. Devolve a perda do lote."""
        self.zero_grad()
        loss, dlogits = self.loss(ids, targets)
        self.backward(dlogits)
        return loss

    # -- persistência -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Pesos em .npz (sem pickle) e configuração em JSON dentro do mesmo arquivo."""
        arrays = {name: p for name, p, _ in self.named_parameters()}
        arrays["__config__"] = np.array(json.dumps(asdict(self.config)))
        np.savez(Path(path), **arrays)

    @classmethod
    def load(cls, path: Path) -> MLPLanguageModel:
        with np.load(Path(path), allow_pickle=False) as data:
            config = ModelConfig(**json.loads(str(data["__config__"])))
            model = cls(config)
            for name, param, _ in model.named_parameters():
                if data[name].shape != param.shape:
                    raise ValueError(f"checkpoint incompatível em {name}: {data[name].shape} != {param.shape}")
                param[...] = data[name]
        return model
