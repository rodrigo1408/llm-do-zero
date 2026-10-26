"""Linha de comando: ``python -m llmzero <comando>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from .data import DEFAULT_CORPUS, load_lines, split_lines
from .experiments import SPLIT_SEED, experiments, run_all
from .generate import sample
from .model import MLPLanguageModel
from .tokenizer import CharTokenizer
from .train import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _train_config(args: argparse.Namespace) -> TrainConfig:
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name, None) is not None}
    return TrainConfig(**overrides)


def cmd_train(args: argparse.Namespace) -> int:
    lines = load_lines(Path(args.corpus))
    tok = CharTokenizer.fit(lines)
    split = split_lines(lines, seed=SPLIT_SEED)
    cfg = _train_config(args)
    res = train(cfg, split, tok)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.model.save(out / "model.npz")
    tok.save(out / "tokenizer.json")
    report = {
        "num_parameters": res.num_parameters,
        "final_train_loss": res.final_train_loss,
        "final_val_loss": res.final_val_loss,
        "best_val_loss": res.best_val_loss,
        "best_val_step": res.best_val_step,
        "diverged": res.diverged,
        "seconds": round(res.seconds, 2),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if res.diverged else 0


def cmd_generate(args: argparse.Namespace) -> int:
    run = Path(args.run)
    model = MLPLanguageModel.load(run / "model.npz")
    tok = CharTokenizer.load(run / "tokenizer.json")
    for name in sample(model, tok, n=args.n, seed=args.seed, temperature=args.temperature):
        print(name)
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    run_all(Path(args.root), only=args.only or None, steps=args.steps)
    print(f"resultados em {Path(args.root) / 'results'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmzero", description="Modelo de linguagem de caracteres do zero (NumPy).")
    p.add_argument("-v", "--verbose", action="store_true", help="mostra o progresso do treino")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="treina um modelo e salva checkpoint + tokenizador")
    t.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    t.add_argument("--out", default="runs/default")
    for f in fields(TrainConfig):
        kind = {"int": int, "float": float, "str": str}.get(str(f.type).split(" ")[0], int)
        t.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=kind, default=None)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="gera nomes a partir de um checkpoint")
    g.add_argument("--run", default="runs/default", help="diretório criado por `train`")
    g.add_argument("-n", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--temperature", type=float, default=1.0)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("experiments", help="roda os experimentos e regrava results/ e docs/img/")
    e.add_argument("--root", default=str(PROJECT_ROOT))
    e.add_argument("--only", nargs="*", choices=sorted(experiments()), help="subconjunto de experimentos")
    e.add_argument("--steps", type=int, default=None, help="sobrescreve o número de passos (execução rápida)")
    e.set_defaults(func=cmd_experiments)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
