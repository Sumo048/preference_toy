import importlib
import subprocess
import sys

MODULES = ["toyrepair", "toyrepair.cell", "toyrepair.policy", "toyrepair.flowstate", "toyrepair.miner",
           "toyrepair.pools", "toyrepair.oracle", "toyrepair.oracle.labels", "toyrepair.oracle.metrics",
           "toyrepair.gate", "toyrepair.edits", "toyrepair.patches", "toyrepair.train", "toyrepair.sandbox",
           "toyrepair.quick_check", "toyrepair.ledger", "toyrepair.budget", "toyrepair.cli",
           "toyrepair.dossier", "toyrepair.orchestrate", "toyrepair.arms", "toyrepair.eval_final"]


def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(m)
    from toyrepair.arms import CODE_ARMS, get
    for a in CODE_ARMS:
        assert callable(get(a))


def test_cli_does_not_import_toypref_or_oracle():
    code = ("import sys, toyrepair.cli; "
            "bad=[m for m in sys.modules if m.startswith('toypref') or m.startswith('toyrepair.oracle') or m=='torch']; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_help_and_stub():
    from toyrepair import cli
    assert cli.main([]) == 0
    assert cli.main(["atlas", "--token", "x"]) == 2
    r = subprocess.run(["toyrepair-tool", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "submit_edit" in r.stdout


def test_cost():
    from toyrepair.budget import cost_usd, gate_check_cap
    assert abs(cost_usd("opus", input=1_000_000, output=1_000_000) - 24.0) < 1e-9
    assert abs(cost_usd("claude-sonnet-5-5", cache_read=1_000_000) - 0.10) < 1e-9
    assert gate_check_cap(4) == 18
