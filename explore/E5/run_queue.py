"""Resumable E5 queue: 4 workers x 1 thread; each unit is one < 20 min call of e5.py.

Order (prereg): k4, alt on seeds 5-9 first, then k1, contig, pairs, b3p1, k2. Per cell: the npm-4 fit
(re-launched from its checkpoint until it finishes), then the npm-16 evaluation. Skip-if-done; safe to
restart at any time (a killed chunk resumes from the last checkpoint).
Usage: nohup python explore/E5/run_queue.py > explore/E5/logs/queue.log 2>&1 &
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
P = int(os.environ.get("P", 4))
ORDER = ["k4", "alt", "k1", "contig", "pairs", "b3p1", "k2"]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "5 6 7 8 9").split()]
RES = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)

units = []
for S in ORDER:
    for s in SEEDS:
        units.append(("fit", S, s, 4))
    for s in SEEDS:
        units.append(("eval", S, s, 16))


def done(u):
    return os.path.exists(os.path.join(RES, f"graft_K10c_{u[1]}_s{u[2]}_n{u[3]}.json"))


def ready(u):
    return u[0] == "fit" or done(("fit", u[1], u[2], 4))


running, fails = {}, {}
while True:
    for u, p in list(running.items()):
        if p.poll() is not None:
            del running[u]
            if p.returncode != 0:
                fails[u] = fails.get(u, 0) + 1
                print(time.strftime("%H:%M:%S"), "FAIL", u, "rc", p.returncode, "count", fails[u], flush=True)
    todo = [u for u in units if not done(u) and u not in running and fails.get(u, 0) < 3]
    if not running and not any(ready(u) for u in todo):
        break
    for u in todo:
        if len(running) >= P:
            break
        if not ready(u):
            continue
        what, S, s, n = u
        cmd = [sys.executable, os.path.join(HERE, "e5.py"), what, "--setting", S, "--seed", str(s), "--npm", str(n)]
        log = open(os.path.join(LOGS, f"{what}_{S}_s{s}_n{n}.log"), "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[1:])}\n"); log.flush()
        running[u] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"})
        print(time.strftime("%H:%M:%S"), "start", u, flush=True)
    time.sleep(5)
print(time.strftime("%H:%M:%S"), "queue empty; done", sum(done(u) for u in units), "of", len(units),
      "failed", [u for u in units if not done(u)], flush=True)
