import json
import os
from pathlib import Path
import subprocess
import sys

from pytorch_foundations.cli import main, run_experiment


def test_run_experiment_returns_report_and_writes_matching_json(tmp_path: Path):
    output = tmp_path / "metrics.json"

    result = run_experiment(epochs=2, seed=5, output=output)

    assert "dataset" in result
    assert "image_level_split" in result
    assert "session_aware_split" in result

    for split_name in ("image_level_split", "session_aware_split"):
        split_metrics = result[split_name]
        assert "train_accuracy" in split_metrics
        assert "test_accuracy" in split_metrics
        assert "final_train_loss" in split_metrics

    saved = json.loads(output.read_text())
    assert saved == result


def test_main_prints_both_split_strategies(monkeypatch, capsys, tmp_path: Path):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["pytorch-foundations", "--epochs", "2", "--seed", "5", "--output", str(output)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Image-Level Split" in captured.out
    assert "Session-Aware Split" in captured.out


def test_module_execution_writes_json_report(tmp_path: Path):
    output = tmp_path / "module-metrics.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytorch_foundations.cli",
            "--epochs",
            "2",
            "--seed",
            "5",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert output.exists()
    assert "RuntimeWarning" not in completed.stderr
