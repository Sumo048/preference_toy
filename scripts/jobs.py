#!/usr/bin/env python3
"""Resumable manifest runner for CPU jobs (SESSION_PROMPTS.md rule 6). Stdlib only.

Manifest: JSONL, one unit per line:
    {"id": "s0_k4_n4_B0", "cmd": ["python", "scripts/cond_cell.py", ...] | "shell string",
     "out": "optional/path/that/marks/completion"}

Files next to the manifest <dir>/<stem>.jsonl:
    <stem>.done/<id>.json     done marker {id, rc, seconds, end} (written atomically, rc == 0)
    <stem>.failed/<id>.json   failure record {id, rc, seconds, end, reason}
    <stem>.locks/<id>.lock    claim lock {pid, host, start}; created with O_EXCL
    <stem>.logs/<id>.log      stdout + stderr of the unit
    <stem>.pid                supervisor pid

A unit is complete if its "out" path exists (when given) or its done marker exists.
Each unit runs with OMP_NUM_THREADS=MKL_NUM_THREADS=1, TOYPREF_RUNS=<repo>/runs if unset,
cwd = repo root, and a 20-minute timeout (killed and recorded as failed).

    python scripts/jobs.py run <manifest> [--workers 4] [--retry-failed] [--timeout 1200] [--foreground]
    python scripts/jobs.py ensure <manifest> [--workers 4] [--retry-failed]
    python scripts/jobs.py status <manifest>

`run` detaches a supervisor (setsid) that survives the caller. `ensure` restarts the supervisor
if it died while units remain, clears stale locks of dead pids, and prints status.
Failed units are not retried unless --retry-failed is given.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIMEOUT_S = 20 * 60


class Paths:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        d, stem = self.manifest.parent, self.manifest.stem
        self.done = d / f"{stem}.done"
        self.failed = d / f"{stem}.failed"
        self.locks = d / f"{stem}.locks"
        self.logs = d / f"{stem}.logs"
        self.pid = d / f"{stem}.pid"
        self.suplog = d / f"{stem}.supervisor.log"

    def mkdirs(self):
        for p in (self.done, self.failed, self.locks, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def load_units(manifest: Path):
    units, seen = [], set()
    for n, line in enumerate(open(manifest), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        u = json.loads(line)
        if "id" not in u or "cmd" not in u:
            raise SystemExit(f"{manifest}:{n}: unit needs 'id' and 'cmd'")
        if u["id"] in seen:
            raise SystemExit(f"{manifest}:{n}: duplicate id {u['id']}")
        if "/" in u["id"]:
            raise SystemExit(f"{manifest}:{n}: id may not contain '/'")
        seen.add(u["id"])
        units.append(u)
    return units


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:   # zombie counts as dead
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def is_done(P: Paths, u) -> bool:
    if (P.done / f"{u['id']}.json").exists():
        return True
    out = u.get("out")
    if not out:
        return False
    return (Path(out) if os.path.isabs(out) else REPO / out).exists()


def read_lock(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def lock_live(path: Path) -> bool:
    info = read_lock(path)
    if info is None:   # being written right now, or garbage; treat young files as live
        try:
            return time.time() - path.stat().st_mtime < 10
        except OSError:
            return False
    return info.get("host") == socket.gethostname() and pid_alive(int(info.get("pid", -1)))


def clear_stale_locks(P: Paths) -> int:
    n = 0
    if P.locks.exists():
        for f in P.locks.glob("*.lock"):
            if not lock_live(f):
                try:
                    f.unlink()
                    n += 1
                except FileNotFoundError:
                    pass
    return n


def try_claim(P: Paths, uid: str) -> bool:
    path = P.locks / f"{uid}.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if lock_live(path):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as fh:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "start": time.time()}, fh)
        return True
    return False


def pending_units(P: Paths, units, retry_failed: bool):
    out = []
    for u in units:
        if is_done(P, u):
            continue
        if not retry_failed and (P.failed / f"{u['id']}.json").exists():
            continue
        out.append(u)
    return out


def unit_env():
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("TOYPREF_RUNS", str(REPO / "runs"))
    return env


CLAIM_LOST = 75


def run_unit(P: Paths, uid: str, retry_failed: bool, timeout: float) -> int:
    """Wrapper process for one unit: claim its lock, run it with a timeout, write the done or
    failed record, release the lock. It survives a supervisor crash, so no unit runs twice."""
    P.mkdirs()
    u = next((x for x in load_units(P.manifest) if x["id"] == uid), None)
    if u is None or not try_claim(P, uid):
        return CLAIM_LOST
    lock = P.locks / f"{uid}.lock"
    try:
        if is_done(P, u) or (not retry_failed and (P.failed / f"{uid}.json").exists()):
            return CLAIM_LOST
        with open(P.logs / f"{uid}.log", "a") as fh:
            fh.write(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} cmd={u['cmd']}\n")
            fh.flush()
            t0 = time.time()
            cmd = u["cmd"]
            pr = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=str(REPO), env=unit_env(),
                                  stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)

            def kill_child(*_):
                try:
                    os.killpg(pr.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            signal.signal(signal.SIGTERM, lambda *a: (kill_child(), sys.exit(143)))
            try:
                rc = pr.wait(timeout=timeout)
                reason = "" if rc == 0 else f"exit code {rc}"
            except subprocess.TimeoutExpired:
                kill_child()
                pr.wait()
                rc, reason = -9, f"timeout after {timeout:.0f}s"
            el = time.time() - t0
            fh.write(f"=== end rc={rc} {el:.1f}s {reason}\n")
        rec = {"id": uid, "rc": rc, "seconds": round(el, 1), "end": time.time()}
        if rc == 0:
            atomic_write(P.done / f"{uid}.json", rec)
            try:
                (P.failed / f"{uid}.json").unlink()
            except FileNotFoundError:
                pass
        else:
            atomic_write(P.failed / f"{uid}.json", {**rec, "reason": reason})
        return 0
    finally:
        try:
            if (read_lock(lock) or {}).get("pid") == os.getpid():
                lock.unlink()
        except FileNotFoundError:
            pass


def supervise(P: Paths, workers: int, retry_failed: bool, timeout: float) -> None:
    """Keep `workers` unit wrappers running until nothing claimable is left. Wrappers started
    by an earlier (dead) supervisor count against the worker limit while their locks live."""
    P.mkdirs()
    log = open(P.suplog, "a", buffering=1)

    def say(msg):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {msg}\n")
    mine = {}
    attempted = set()     # each unit is launched at most once per supervisor
    say(f"start workers={workers} retry_failed={retry_failed}")
    while True:
        for uid, pr in list(mine.items()):
            rc = pr.poll()
            if rc is not None:
                say(f"{uid} wrapper exit {rc}")
                del mine[uid]
        live = {f.stem for f in P.locks.glob("*.lock") if lock_live(f)}
        busy = live | set(mine)
        units = load_units(P.manifest)       # re-read: units may be appended
        todo = [u for u in pending_units(P, units, retry_failed) if u["id"] not in busy | attempted]
        for u in todo:
            if len(busy) >= workers:
                break
            cmd = [sys.executable, str(Path(__file__).resolve()), "_unit", str(P.manifest), u["id"],
                   "--timeout", str(timeout)] + (["--retry-failed"] if retry_failed else [])
            mine[u["id"]] = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                             start_new_session=True, cwd=str(REPO))
            busy.add(u["id"])
            attempted.add(u["id"])
            say(f"{u['id']} launched")
        if not busy and not todo:
            break
        time.sleep(1.0)
    say("exit")
    try:
        if int(P.pid.read_text().strip() or -1) == os.getpid():
            P.pid.unlink()
    except (OSError, ValueError):
        pass


def supervisor_pid(P: Paths) -> int:
    try:
        return int(P.pid.read_text().strip())
    except (OSError, ValueError):
        return -1


def start_detached(P: Paths, workers: int, retry_failed: bool, timeout: float) -> int:
    P.mkdirs()
    pid = supervisor_pid(P)
    if pid_alive(pid):
        print(f"supervisor already running (pid {pid})")
        return pid
    cmd = [sys.executable, str(Path(__file__).resolve()), "_supervise", str(P.manifest),
           "--workers", str(workers), "--timeout", str(timeout)] + (["--retry-failed"] if retry_failed else [])
    pr = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(P.suplog, "a"),
                          start_new_session=True, close_fds=True, cwd=str(REPO))
    P.pid.write_text(str(pr.pid))
    print(f"supervisor started (pid {pr.pid})")
    return pr.pid


def status(P: Paths, quiet=False):
    units = load_units(P.manifest)
    done, failed, running, pending = [], [], [], []
    for u in units:
        uid = u["id"]
        if is_done(P, u):
            done.append(uid)
        elif (P.locks / f"{uid}.lock").exists() and lock_live(P.locks / f"{uid}.lock"):
            running.append(uid)
        elif (P.failed / f"{uid}.json").exists():
            failed.append(uid)
        else:
            pending.append(uid)
    pid = supervisor_pid(P)
    s = {"total": len(units), "done": len(done), "failed": len(failed), "running": len(running),
         "pending": len(pending), "supervisor": pid if pid_alive(pid) else None,
         "running_ids": running, "failed_ids": failed[:20]}
    if not quiet:
        print(json.dumps(s))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "ensure", "status", "_supervise", "_unit"])
    ap.add_argument("manifest")
    ap.add_argument("unit_id", nargs="?")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--foreground", action="store_true", help="run: supervise in this process")
    a = ap.parse_args(argv)
    P = Paths(a.manifest)
    if a.cmd == "_unit":
        sys.exit(run_unit(P, a.unit_id, a.retry_failed, a.timeout))
    if a.cmd == "_supervise":
        P.pid.write_text(str(os.getpid()))
        supervise(P, a.workers, a.retry_failed, a.timeout)
    elif a.cmd == "run":
        if a.foreground:
            P.mkdirs()
            P.pid.write_text(str(os.getpid()))
            supervise(P, a.workers, a.retry_failed, a.timeout)
            status(P)
        else:
            start_detached(P, a.workers, a.retry_failed, a.timeout)
    elif a.cmd == "ensure":
        P.mkdirs()
        if not pid_alive(supervisor_pid(P)):
            n = clear_stale_locks(P)
            left = pending_units(P, load_units(P.manifest), a.retry_failed)
            if left:
                print(f"supervisor dead; cleared {n} stale locks; restarting for {len(left)} units")
                start_detached(P, a.workers, a.retry_failed, a.timeout)
                time.sleep(0.5)
        status(P)
    else:
        status(P)


if __name__ == "__main__":
    main()
