"""Preparação dos dados: leitura do corpus, divisão e janelas de contexto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import CharTokenizer

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "municipios.txt"


def load_lines(path: Path = DEFAULT_CORPUS) -> list[str]:
    """Uma sequência por linha; linhas vazias são descartadas."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"corpus vazio: {path}")
    return lines


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def split_lines(lines: list[str], seed: int, val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    """Embaralha com semente fixa e divide em treino/validação/teste.

    A divisão é por *sequência*, não por janela: um nome inteiro nunca aparece
    em dois conjuntos, então a perda de validação mede generalização real.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("frações de validação/teste inválidas")
    order = np.random.default_rng(seed).permutation(len(lines))
    shuffled = [lines[i] for i in order]
    n_val = int(len(lines) * val_frac)
    n_test = int(len(lines) * test_frac)
    return Split(
        train=shuffled[n_val + n_test :],
        val=shuffled[:n_val],
        test=shuffled[n_val : n_val + n_test],
    )


def build_windows(lines: list[str], tok: CharTokenizer, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Converte sequências em pares (contexto, próximo token).

    ``"rio"`` com ``block_size=3`` vira::

        ...  -> r
        ..r  -> i
        .ri  -> o
        rio  -> .
    """
    if block_size < 1:
        raise ValueError("block_size deve ser >= 1")
    xs: list[list[int]] = []
    ys: list[int] = []
    for line in lines:
        context = [0] * block_size
        for idx in [*tok.encode(line), 0]:
            xs.append(context)
            ys.append(idx)
            context = [*context[1:], idx]
    return np.array(xs, dtype=np.int64).reshape(-1, block_size), np.array(ys, dtype=np.int64)
