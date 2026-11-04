"""Tests for scripts/jobs.py: run, skip-if-done, failure records, timeout, kill + ensure resume."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

JOBS = Path(__file__).resolve().parents[2] / "scripts" / "jobs.py"


def jobs(*args):
    r = subprocess.run([sys.executable, str(JOBS), *args], capture_output=True, text=True, timeout=120)
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def status(m):
    return json.loads(jobs("status", str(m)))


def wait(m, cond, t=60):
    end = time.time() + t
    while time.time() < end:
        s = status(m)
        if cond(s):
            return s
        time.sleep(0.5)
    raise AssertionError(status(m))


def write(m, units):
    m.write_text("\n".join(json.dumps(u) for u in units) + "\n")


def test_run_fail_timeout_env(tmp_path):
    m = tmp_path / "man.jsonl"
    marker = tmp_path / "out_b.txt"
    write(m, [
        {"id": "a", "cmd": [sys.executable, "-c", "import os; print(os.environ['OMP_NUM_THREADS'], os.environ['TOYPREF_RUNS'])"]},
        {"id": "b", "cmd": f"echo hi > {marker}", "out": str(marker)},
        {"id": "bad", "cmd": [sys.executable, "-c", "raise SystemExit(3)"]},
        {"id": "slow", "cmd": ["sleep", "30"]},
    ])
    subprocess.run([sys.executable, str(JOBS), "run", str(m), "--workers", "2", "--timeout", "2"], check=True)
    s = wait(m, lambda s: s["supervisor"] is None and s["running"] == 0)
    assert s["done"] == 2 and s["failed"] == 2, s
    assert (tmp_path / "man.logs" / "a.log").read_text().splitlines()[1].startswith("1 ")
    assert "timeout" in json.loads((tmp_path / "man.failed" / "slow.json").read_text())["reason"]
    assert json.loads((tmp_path / "man.failed" / "bad.json").read_text())["rc"] == 3
    # rerun: nothing to do, failed not retried
    jobs("run", str(m))
    s = wait(m, lambda s: s["supervisor"] is None)
    assert s["done"] == 2 and s["failed"] == 2
    # retry-failed reruns them (bad fails again)
    jobs("run", str(m), "--retry-failed", "--timeout", "1")
    s = wait(m, lambda s: s["supervisor"] is None and s["running"] == 0)
    assert s["failed"] == 2 and s["done"] == 2


def test_kill_supervisor_then_ensure(tmp_path):
    m = tmp_path / "k.jsonl"
    cnt = tmp_path / "count"
    cnt.mkdir()
    write(m, [{"id": f"u{i}", "cmd": f"touch {cnt}/u{i}.$$; sleep 2"} for i in range(6)])
    jobs("run", str(m), "--workers", "2")
    s = wait(m, lambda s: s["running"] == 2)
    pid = s["supervisor"]
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    assert status(m)["supervisor"] is None
    out = jobs("ensure", str(m), "--workers", "2")
    s = wait(m, lambda s: s["done"] == 6 and s["supervisor"] is None, t=90)
    # every unit ran exactly once, even the two that were running when the supervisor died
    names = [p.name.split(".")[0] for p in cnt.iterdir()]
    assert sorted(names) == [f"u{i}" for i in range(6)], names
    # ensure on a finished manifest does nothing
    assert json.loads(jobs("ensure", str(m)))["done"] == 6
