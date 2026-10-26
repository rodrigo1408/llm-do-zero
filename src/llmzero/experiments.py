"""Experimentos reproduzíveis.

Cada experimento é uma lista de execuções (``TrainConfig``) sobre a mesma
divisão de dados. ``run_all`` grava, para cada experimento:

* ``results/<nome>.json`` — configurações, histórico de perdas e métricas finais;
* ``docs/img/<nome>.svg`` — curvas de perda;

e ``results/SUMMARY.md``, uma tabela gerada a partir dos JSON (nunca editada à mão).
"""

from __future__ import annotations

import json
import logging
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .bigram import BigramModel
from .data import Split, build_windows, load_lines, split_lines
from .generate import novelty, sample
from .plot import Series, line_chart
from .tokenizer import CharTokenizer
from .train import TrainConfig, TrainResult, evaluate, train

log = logging.getLogger(__name__)

SPLIT_SEED = 1234
BASE = TrainConfig()


def experiments() -> dict[str, dict]:
    """Definição declarativa dos experimentos: o que muda em relação à BASE."""
    return {
        "baseline": {
            "title": "MLP de referência: treino vs. validação",
            "runs": {"mlp": BASE},
            "plot": "train_val",
            # O conjunto de teste é usado uma única vez, só para a configuração fixada de antemão.
            "report_test": True,
        },
        "learning_rate": {
            "title": "Taxa de aprendizado (AdamW) — perda de validação",
            "runs": {f"lr={lr:g}": replace(BASE, lr=lr) for lr in (3e-4, 3e-3, 3e-2, 3e-1)},
            "plot": "val",
        },
        "context": {
            "title": "Tamanho do contexto (block_size) — perda de validação",
            "runs": {f"block={b}": replace(BASE, block_size=b) for b in (1, 2, 4, 8)},
            "plot": "val",
        },
        "overfitting": {
            "title": "Overfitting: 300 nomes de treino, rede larga",
            "runs": {
                "sem regularização": replace(BASE, train_limit=300, hidden_dim=400, lr=3e-3),
                "weight decay 0.1": replace(BASE, train_limit=300, hidden_dim=400, lr=3e-3, weight_decay=0.1),
            },
            "plot": "train_val",
        },
        "optimizer": {
            "title": "SGD vs. AdamW — perda de validação",
            "runs": {"sgd lr=0.1": replace(BASE, optimizer="sgd", lr=0.1), "adamw lr=0.003": BASE},
            "plot": "val",
        },
    }


def _finite(v: float) -> float | None:
    return v if math.isfinite(v) else None


def _run_record(
    name: str, cfg: TrainConfig, res: TrainResult, split: Split, tok: CharTokenizer, report_test: bool
) -> dict:
    samples = sample(res.model, tok, n=200, seed=7, temperature=1.0) if not res.diverged else []
    test_loss = None
    if report_test and not res.diverged:
        test_loss = evaluate(res.model, *build_windows(split.test, tok, cfg.block_size))
    return {
        "name": name,
        "config": asdict(cfg),
        "num_parameters": res.num_parameters,
        "train_sequences": res.train_sequences,
        "final_train_loss": _finite(res.final_train_loss),
        "final_val_loss": _finite(res.final_val_loss),
        "best_val_loss": _finite(res.best_val_loss),
        "best_val_step": res.best_val_step,
        "test_loss": test_loss,
        "diverged": res.diverged,
        "seconds": round(res.seconds, 2),
        "history": res.history.to_dict(),
        "samples_preview": samples[:12],
        # "copiado" é medido contra os nomes que o modelo realmente viu no treino.
        "novelty": novelty(samples, split.train[: cfg.train_limit], split.val) if samples else None,
    }


def _plot(name: str, spec: dict, records: list[dict], img_dir: Path, y_max: float) -> None:
    series: list[Series] = []
    for rec in records:
        h = rec["history"]
        if not h["steps"]:
            continue
        if spec["plot"] == "train_val":
            prefix = "" if len(records) == 1 else f"{rec['name']} · "
            series.append(Series(f"{prefix}treino", h["steps"], h["train_loss"]))
            series.append(Series(f"{prefix}validação", h["steps"], h["val_loss"]))
        else:
            label = f"{rec['name']} (divergiu)" if rec["diverged"] else rec["name"]
            series.append(Series(label, h["steps"], h["val_loss"]))
    if series:
        line_chart(
            series, spec["title"], "passo de otimização", "perda (nats/token)", img_dir / f"{name}.svg", y_max=y_max
        )


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


def run_all(root: Path, only: list[str] | None = None, steps: int | None = None) -> dict:
    """Executa os experimentos e grava resultados. ``steps`` sobrescreve o número de passos (testes rápidos)."""
    results_dir, img_dir = root / "results", root / "docs" / "img"
    results_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    lines = load_lines()
    split = split_lines(lines, seed=SPLIT_SEED)
    tok = CharTokenizer.fit(lines)
    reference = {
        "uniform_loss": math.log(tok.vocab_size),
        "bigram_val_loss": BigramModel(tok.vocab_size).fit(split.train, tok).nll(split.val, tok),
        "vocab_size": tok.vocab_size,
        "sequences": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "split_seed": SPLIT_SEED,
    }

    summary: dict[str, list[dict]] = {}
    for name, spec in experiments().items():
        if only and name not in only:
            continue
        records = []
        for run_name, cfg in spec["runs"].items():
            if steps is not None:
                cfg = replace(cfg, steps=steps, eval_every=max(1, steps // 10))
            log.info("experimento %s / %s", name, run_name)
            res = train(cfg, split, tok)
            records.append(_run_record(run_name, cfg, res, split, tok, spec.get("report_test", False)))
        payload = {
            "experiment": name,
            "title": spec["title"],
            "reference": reference,
            "environment": environment(),
            "runs": records,
        }
        (results_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Acima de ln(V) o modelo é pior que chutar: o eixo para aí para não achatar as demais curvas.
        _plot(name, spec, records, img_dir, y_max=reference["uniform_loss"])
        summary[name] = records

    (results_dir / "SUMMARY.md").write_text(render_summary(reference, summary), encoding="utf-8")
    return {"reference": reference, "experiments": summary}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _pct(nov: dict | None, key: str) -> str:
    return "—" if nov is None else f"{nov[key]:.0%}"


def render_summary(reference: dict, summary: dict[str, list[dict]]) -> str:
    env = environment()
    out = [
        "# Resultados dos experimentos",
        "",
        "> Arquivo gerado por `python -m llmzero experiments`. Não edite à mão.",
        "",
        f"Ambiente: Python {env['python']}, NumPy {env['numpy']}, {env['platform']}.",
        "",
        "## Referências",
        "",
        f"- Vocabulário: {reference['vocab_size']} tokens (caracteres + fronteira)",
        f"- Sequências: {reference['sequences']['train']} treino / {reference['sequences']['val']} validação / "
        f"{reference['sequences']['test']} teste (semente {reference['split_seed']})",
        f"- Perda de um modelo uniforme, ln(V): **{reference['uniform_loss']:.4f}**",
        f"- Perda de validação do bigrama por contagem (alpha=1): **{reference['bigram_val_loss']:.4f}**",
        "",
    ]
    for name, records in summary.items():
        out += [
            f"## {name}",
            "",
            f"![{name}](../docs/img/{name}.svg)",
            "",
            "| execução | parâmetros | nomes de treino | treino final | validação final | melhor validação (passo) "
            "| inéditos | copiados do treino | tempo (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            best = "—" if r["best_val_loss"] is None else f"{r['best_val_loss']:.4f} ({r['best_val_step']})"
            flag = " ⚠ divergiu" if r["diverged"] else ""
            out.append(
                f"| {r['name']}{flag} | {r['num_parameters']} | {r['train_sequences']} | {_fmt(r['final_train_loss'])} "
                f"| {_fmt(r['final_val_loss'])} | {best} | {_pct(r['novelty'], 'new')} "
                f"| {_pct(r['novelty'], 'copied_from_train')} | {r['seconds']} |"
            )
        out.append("")
        for r in records:
            if r["test_loss"] is not None:
                out += [
                    f"Perda no conjunto de **teste** (`{r['name']}`, avaliada uma única vez): **{r['test_loss']:.4f}**",
                    "",
                ]
        previews = [f"`{r['name']}`: " + ", ".join(r["samples_preview"][:8]) for r in records if r["samples_preview"]]
        if previews:
            out += ["Amostras (temperatura 1.0, semente 7):", ""] + [f"- {p}" for p in previews] + [""]
    return "\n".join(out)
