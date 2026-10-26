import json
import xml.etree.ElementTree as ET

from llmzero.cli import main
from llmzero.experiments import run_all
from llmzero.plot import Series, line_chart


def test_train_then_generate_via_cli(tmp_path, capsys, tiny_lines):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(tiny_lines * 3), encoding="utf-8")
    run = tmp_path / "run"
    code = main(
        [
            "train",
            "--corpus",
            str(corpus),
            "--out",
            str(run),
            "--steps",
            "40",
            "--eval-every",
            "20",
            "--hidden-dim",
            "16",
        ]
    )
    assert code == 0
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["num_parameters"] > 0 and not report["diverged"]
    capsys.readouterr()
    assert main(["generate", "--run", str(run), "-n", "3", "--seed", "1"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3


def test_missing_checkpoint_is_a_clean_error(tmp_path, capsys):
    assert main(["generate", "--run", str(tmp_path / "nada")]) == 2
    assert "erro:" in capsys.readouterr().err


def test_quick_experiment_writes_json_svg_and_summary(tmp_path):
    run_all(tmp_path, only=["baseline"], steps=30)
    payload = json.loads((tmp_path / "results" / "baseline.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["history"]["steps"][-1] == 30
    assert payload["reference"]["bigram_val_loss"] < payload["reference"]["uniform_loss"]
    ET.parse(tmp_path / "docs" / "img" / "baseline.svg")  # SVG bem formado
    assert "Não edite à mão" in (tmp_path / "results" / "SUMMARY.md").read_text(encoding="utf-8")


def test_line_chart_escapes_labels(tmp_path):
    line_chart([Series("a<b>&", [0, 1], [2.0, 1.0])], "t & <x>", "x", "y", tmp_path / "c.svg")
    ET.parse(tmp_path / "c.svg")


def test_summary_does_not_invent_metrics_for_diverged_runs():
    from llmzero.experiments import render_summary

    ref = {
        "vocab_size": 5,
        "sequences": {"train": 1, "val": 1, "test": 1},
        "split_seed": 0,
        "uniform_loss": 1.6,
        "bigram_val_loss": 1.2,
    }
    run = {
        "name": "lr=9",
        "num_parameters": 10,
        "train_sequences": 1,
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": 1.6,
        "best_val_step": 0,
        "test_loss": None,
        "diverged": True,
        "seconds": 0.1,
        "novelty": None,
        "samples_preview": [],
    }
    row = next(line for line in render_summary(ref, {"x": [run]}).splitlines() if line.startswith("| lr=9"))
    assert "divergiu" in row
    assert "%" not in row  # sem amostras, nada de "0% inéditos"
    assert "teste" not in render_summary(ref, {"x": [run]}).split("## x")[1].split("|")[0]


def test_ticks_cover_the_whole_data_range():
    from llmzero.plot import _nice_ticks

    for lo, hi in [(0.77, 3.74), (1.8, 1.9), (0.0, 6000.0), (2.5, 2.5)]:
        ticks = _nice_ticks(lo, hi)
        assert ticks[0] <= lo and ticks[-1] >= hi, (lo, hi, ticks)


def test_chart_clips_and_says_so(tmp_path):
    line_chart([Series("a", [0, 1, 2], [3.0, 50.0, 2.0])], "t", "x", "y", tmp_path / "c.svg", y_max=4.0)
    svg = (tmp_path / "c.svg").read_text(encoding="utf-8")
    assert "cortados" in svg and 'clip-path="url(#plot)"' in svg
