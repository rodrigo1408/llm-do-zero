"""Laço de treinamento e avaliação."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import Split, build_windows
from .model import MLPLanguageModel, ModelConfig
from .optim import make_optimizer
from .tokenizer import CharTokenizer

log = logging.getLogger(__name__)

# Uma perda de lote acima de DIVERGENCE_FACTOR * ln(V) (dez vezes pior que chutar
# uniformemente) é tratada como divergência, mesmo que ainda seja um número finito.
DIVERGENCE_FACTOR = 10.0


@dataclass(frozen=True)
class TrainConfig:
    block_size: int = 4
    embed_dim: int = 12
    hidden_dim: int = 200
    optimizer: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    steps: int = 6000
    eval_every: int = 250
    seed: int = 42
    train_limit: int | None = None  # usa só os N primeiros nomes de treino (experimento de overfitting)

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError("steps, batch_size e eval_every devem ser >= 1")
        if not self.lr > 0:
            raise ValueError("lr deve ser > 0")


@dataclass
class History:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    model: MLPLanguageModel
    history: History
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_val_step: int
    num_parameters: int
    train_sequences: int
    seconds: float
    diverged: bool


def evaluate(model: MLPLanguageModel, x: np.ndarray, y: np.ndarray, chunk: int = 4096) -> float:
    """Perda média sobre *todas* as janelas (não uma amostra), em blocos para limitar memória."""
    total = 0.0
    for i in range(0, len(x), chunk):
        loss, _ = model.loss(x[i : i + chunk], y[i : i + chunk])
        total += loss * len(x[i : i + chunk])
    return total / len(x)


def train(cfg: TrainConfig, split: Split, tok: CharTokenizer) -> TrainResult:
    train_lines = split.train[: cfg.train_limit] if cfg.train_limit else split.train
    xtr, ytr = build_windows(train_lines, tok, cfg.block_size)
    xva, yva = build_windows(split.val, tok, cfg.block_size)

    model = MLPLanguageModel(ModelConfig(tok.vocab_size, cfg.block_size, cfg.embed_dim, cfg.hidden_dim), seed=cfg.seed)
    params = model.named_parameters()
    opt = make_optimizer(cfg.optimizer, params, cfg.lr, cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed + 1)  # fluxo separado do da inicialização
    hist = History()
    diverged = False
    started = time.perf_counter()

    log.info("treino: %d janelas de %d nomes, %d parâmetros", len(xtr), len(train_lines), model.num_parameters())
    limit = DIVERGENCE_FACTOR * math.log(tok.vocab_size)
    # Passo 0: a perda antes de qualquer atualização (deve ficar perto de ln(V)).
    hist.steps.append(0)
    hist.train_loss.append(evaluate(model, xtr, ytr))
    hist.val_loss.append(evaluate(model, xva, yva))
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(xtr), size=cfg.batch_size)
        loss = model.train_step_grads(xtr[idx], ytr[idx])
        if not math.isfinite(loss) or loss > limit:
            log.warning("divergência no passo %d (perda do lote %s): treino interrompido", step, loss)
            diverged = True
            break
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            tr, va = evaluate(model, xtr, ytr), evaluate(model, xva, yva)
            hist.steps.append(step)
            hist.train_loss.append(tr)
            hist.val_loss.append(va)
            log.info("passo %5d  treino %.4f  validação %.4f", step, tr, va)

    # Num treino divergente o último ponto registrado é anterior à explosão: não reportá-lo como final.
    final_tr = float("nan") if diverged else hist.train_loss[-1]
    final_va = float("nan") if diverged else hist.val_loss[-1]
    best = int(np.argmin(hist.val_loss))
    return TrainResult(
        model=model,
        history=hist,
        final_train_loss=final_tr,
        final_val_loss=final_va,
        best_val_loss=hist.val_loss[best],
        best_val_step=hist.steps[best],
        num_parameters=model.num_parameters(),
        train_sequences=len(train_lines),
        seconds=time.perf_counter() - started,
        diverged=diverged,
    )
