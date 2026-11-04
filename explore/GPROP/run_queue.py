"""4-worker queue (1 torch thread each). Jobs: lines of explore/GPROP/jobs.txt, each the argument list of
gprop.py (or 'e4 <args>' for explore/E4/e4.py). A job is rerun (resumable unit) until its output says DONE
or it fails 3 times. jobs.txt is re-read whenever a worker frees, so jobs can be appended while it runs.
State: explore/GPROP/logs/state.json; per-job logs in explore/GPROP/logs/.
"""
import json, os, subprocess, sys, time, threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JOBS = os.path.join(HERE, "jobs.txt")
LOGS = os.path.join(HERE, "logs")
NW = int(os.environ.get("GP_WORKERS", 4))
os.makedirs(LOGS, exist_ok=True)
lock = threading.Lock()
state = {"done": [], "failed": [], "running": []}
SF = os.path.join(LOGS, "state.json")
if os.path.exists(SF):
    state = json.load(open(SF)); state["running"] = []
fails = {}


def save():
    json.dump(state, open(SF + ".tmp", "w"), indent=1); os.replace(SF + ".tmp", SF)


def next_job():
    with lock:
        jobs = [l.strip() for l in open(JOBS) if l.strip() and not l.startswith("#")]
        for j in jobs:
            if j not in state["done"] and j not in state["failed"] and j not in state["running"]:
                state["running"].append(j); save()
                return j
    return None


def run(job):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOYPREF_RUNS=os.path.join(ROOT, "runs"))
    a = job.split()
    cmd = [sys.executable, os.path.join(ROOT, "explore", "E4", "e4.py")] + a[1:] if a[0] == "e4" else \
        [sys.executable, os.path.join(HERE, "gprop.py")] + a
    log = os.path.join(LOGS, job.replace(" ", "_").replace("--", "").replace("/", "_") + ".log")
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)}\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    tail = open(log).read()[-3000:]
    if a[0] == "e4":
        return "done" if r.returncode == 0 else "fail"
    if r.returncode != 0:
        return "fail"
    return "done" if "DONE" in tail.split("=== ")[-1] else "again"


def worker(i):
    while True:
        j = next_job()
        if j is None:
            time.sleep(30)
            continue
        while True:
            res = run(j)
            if res == "again":
                continue
            with lock:
                state["running"].remove(j)
                if res == "done":
                    state["done"].append(j)
                else:
                    fails[j] = fails.get(j, 0) + 1
                    if fails[j] >= 3:
                        state["failed"].append(j)
                save()
            print(time.strftime("%T"), res, j, flush=True)
            break


def next_job_peek():
    jobs = [l.strip() for l in open(JOBS) if l.strip() and not l.startswith("#")]
    for j in jobs:
        if j not in state["done"] and j not in state["failed"] and j not in state["running"]:
            return j
    return None


if __name__ == "__main__":
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(NW)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("queue empty", flush=True)
