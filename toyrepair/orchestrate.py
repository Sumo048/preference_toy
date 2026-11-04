"""The code half of the loop (docs/PLAN.md 6.10, 7.4), plus the run/session state the CLI shares.

``python -m toyrepair.orchestrate next --run <id>`` prints the next action, one line:
  TRIAGE triage <token>       invoke the ``triage`` subagent with this token
  SESSION <agent> <token>     invoke ``surgeon`` or ``surgeon-sonnet`` with this token
  PENDING <agent> <token>     a token was issued and not recorded yet: invoke it / record it
  CODE <what>                 the arm is a code arm; run it with scripts/run_arm.py
  DONE <stop rule>            the run stopped
``python -m toyrepair.orchestrate record --token <token> [--summary "<line>"] [--transcript P]``
reads the subagent transcript, sums tokens and cost, audits it and writes the ledger record.
``python -m toyrepair.orchestrate status`` prints one line per run.
``python -m toyrepair.orchestrate init --run <id> --arm <arm> --grid cond --setting k4 --seed 0 --npm 4
[--backend real|mock] [--phase4]`` creates the run directory.

Stop rules: no open cluster; 3 sessions in a row without an accept; gate checks reach
3 x (first-atlas clusters + 2); cost cap spent. A transcript-audit hit marks the run invalid and
stops it (``invalid_audit``).

Run directory ``$TOYPREF_RUNS/repair/<run id>/``::

    run.json          Run state (this module)            budget.json   toyrepair.budget.Budget
    ledger.jsonl      toyrepair.ledger                    versions.json version id -> parent, patch ids
    patches/          toyrepair.patches                   scratch/      the surgeon's scratch dir
    atlas/<vid>.pkl   pickled miner.Atlas (+ .txt)        branch/<sid>.pkl  pickled BranchTable (+ .txt)
    sessions/<sid>.json  SessionToken state
Token index: ``$TOYPREF_RUNS/repair/_tokens/<token>.json`` -> {run, session}.

Up to 8 runs can have sessions open at once; every state change holds the run's file lock
(``fcntl.flock`` on ``<run dir>/.lock``). Stdlib only at import time (the CLI imports this).

Owner: W1-D.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import fcntl
import json
import os
import pickle
import re
import secrets
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import RUNS

ARMS = ("B0", "B1a", "B1b", "B4", "B5", "R0", "D1", "A1", "A2", "A3",
        "A3-nobranch", "A3-sonnet", "A3-nogate")
LLM_ARMS = ("A1", "A2", "A3", "A3-nobranch", "A3-sonnet", "A3-nogate")
STOP_RULES = ("no_open_cluster", "no_accept_streak", "gate_cap", "cost_cap")
MAX_CONCURRENT_SESSIONS = 8
# Mirrors toyrepair.budget (W1-A2); repeated here so this module stays import-light.
COOLDOWN_ITERATIONS = 2
NO_ACCEPT_STREAK_STOP = 3
RUN_COST_CAP_USD = 8.0


def repair_root(runs_root: Optional[str] = None) -> str:
    return os.path.join(runs_root or os.environ.get("TOYPREF_RUNS", RUNS), "repair")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionToken:
    """token: random url-safe string (letters, digits, - _); run: run id; session: session id
    (e.g. ``s003``); role: "surgeon" | "triage"; agent: subagent name; cluster: target cluster
    id (None for triage); calls / submissions: counters; closed: True after record."""
    token: str
    run: str
    session: str
    role: str
    agent: str
    cluster: Optional[int] = None
    calls: int = 0
    submissions: int = 0
    closed: bool = False
    accepted: Optional[str] = None          # accepted version id
    versions: List[str] = field(default_factory=list)   # candidate versions made in the session
    created: str = ""
    version_at_start: str = "v0"


@dataclass
class Run:
    """run id; dir ``$TOYPREF_RUNS/repair/<id>`` (ledger.jsonl, budget.json, patches/, scratch/); arm; cell key (grid, setting, seed, npm); current version id;
    first-atlas cluster count; cool-down {cluster: iterations left}; scratch dir."""
    id: str
    dir: str
    arm: str
    grid: str
    setting: str
    seed: int
    npm: int
    current: str = "v0"
    n_first_clusters: Optional[int] = None
    cooldown: Dict[int, int] = field(default_factory=dict)
    # --- W1-D additions (all optional) ---
    backend: str = "real"             # "real" | "mock"
    phase4_tools: bool = False        # contrast / unit_card available
    gate_cfg: Dict = field(default_factory=dict)   # overrides of gate.GateConfig fields
    cost_cap_usd: float = RUN_COST_CAP_USD
    iteration: int = 0
    next_session: int = 1
    next_version: int = 1
    next_patch: int = 1
    pending: Optional[str] = None     # token issued and not yet recorded
    target: Optional[int] = None      # cluster chosen by triage, awaiting a session
    stop: Optional[str] = None
    invalid: Optional[str] = None
    created: str = ""

    @property
    def scratch(self) -> str:
        return self.dir + "/scratch"

    @property
    def cell_args(self):
        return (self.grid, self.setting, self.seed, self.npm)


# ----------------------------------------------------------------------------- persistence

def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_dir(run_id: str, runs_root: Optional[str] = None) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", run_id or ""):
        raise ValueError(f"bad run id {run_id!r}")
    return os.path.join(repair_root(runs_root), run_id)


def save_run(run: Run) -> None:
    d = asdict(run)
    d["cooldown"] = {str(k): v for k, v in run.cooldown.items()}
    _atomic_write(os.path.join(run.dir, "run.json"), json.dumps(d, indent=1))


def load_run(run_id: str, runs_root: Optional[str] = None) -> Run:
    path = os.path.join(run_dir(run_id, runs_root), "run.json")
    with open(path) as fh:
        d = json.load(fh)
    d["cooldown"] = {int(k): int(v) for k, v in (d.get("cooldown") or {}).items()}
    known = {f.name for f in dataclasses.fields(Run)}
    return Run(**{k: v for k, v in d.items() if k in known})


@contextlib.contextmanager
def run_lock(rdir: str):
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, ".lock"), "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _token_dir(runs_root: Optional[str] = None) -> str:
    return os.path.join(repair_root(runs_root), "_tokens")


def save_session(run: Run, s: SessionToken) -> None:
    os.makedirs(os.path.join(run.dir, "sessions"), exist_ok=True)
    _atomic_write(os.path.join(run.dir, "sessions", f"{s.session}.json"), json.dumps(asdict(s), indent=1))


def load_session(run: Run, session: str) -> SessionToken:
    with open(os.path.join(run.dir, "sessions", f"{session}.json")) as fh:
        return SessionToken(**json.load(fh))


def lookup_token(token: str, runs_root: Optional[str] = None):
    """token -> (Run, SessionToken). Raises KeyError for an unknown token."""
    if not token or not re.fullmatch(r"[A-Za-z0-9_\-]{8,64}", token):
        raise KeyError("unknown token")
    path = os.path.join(_token_dir(runs_root), f"{token}.json")
    if not os.path.exists(path):
        raise KeyError("unknown token")
    with open(path) as fh:
        ref = json.load(fh)
    run = load_run(ref["run"], runs_root)
    return run, load_session(run, ref["session"])


TOKEN_ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def new_token(n: int = 16) -> str:
    """Letters and digits only: a token never starts with '-' (argparse would read it as an option)."""
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(n))


def issue_token(run: Run, role: str, agent: str, cluster: Optional[int] = None) -> SessionToken:
    sid = f"s{run.next_session:03d}"
    run.next_session += 1
    s = SessionToken(token=new_token(), run=run.id, session=sid, role=role, agent=agent,
                     cluster=cluster, created=now(), version_at_start=run.current)
    save_session(run, s)
    os.makedirs(_token_dir(os.path.dirname(os.path.dirname(run.dir))), exist_ok=True)
    _atomic_write(os.path.join(_token_dir(os.path.dirname(os.path.dirname(run.dir))), f"{s.token}.json"),
                  json.dumps({"run": run.id, "session": sid}))
    run.pending = s.token
    return s


# ----------------------------------------------------------------------------- versions registry

def versions(run: Run) -> Dict[str, Dict]:
    path = os.path.join(run.dir, "versions.json")
    if not os.path.exists(path):
        return {"v0": {"parent": None, "patches": [], "session": None, "call": None}}
    with open(path) as fh:
        return json.load(fh)


def register_version(run: Run, parent: str, patch_id: Optional[str], session: Optional[str],
                     call: Optional[str]) -> str:
    vs = versions(run)
    if parent not in vs:
        raise KeyError(f"unknown version {parent}")
    vid = f"v{run.next_version}"
    run.next_version += 1
    vs[vid] = {"parent": parent, "patches": vs[parent]["patches"] + ([patch_id] if patch_id else []),
               "session": session, "call": call}
    _atomic_write(os.path.join(run.dir, "versions.json"), json.dumps(vs, indent=1))
    return vid


def new_patch_id(run: Run) -> str:
    pid = f"p{run.next_patch:04d}"
    run.next_patch += 1
    return pid


# ----------------------------------------------------------------------------- ledger / budget adapters

def ledger_append(run: Run, kind: str, data: Dict, session: Optional[str] = None,
                  version: Optional[str] = None) -> None:
    """Append through ``toyrepair.ledger.Ledger``; until W1-A2's ledger exists, write the same
    LedgerRecord as one JSON line (local adapter, docs/INTERFACE_REQUESTS_W1-D.md)."""
    from .ledger import Ledger, LedgerRecord
    rec = LedgerRecord(kind=kind, run=run.id, ts=now(), data=_jsonable(data), session=session, version=version)
    path = os.path.join(run.dir, "ledger.jsonl")
    try:
        Ledger(path).append(rec)
    except NotImplementedError:
        with open(path, "a") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def ledger_records(run: Run, kind: Optional[str] = None) -> List[Dict]:
    path = os.path.join(run.dir, "ledger.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or r.get("kind") == kind:
                out.append(r)
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "__dict__"):
        return _jsonable(vars(x))
    return str(x)


BUDGET_FIELDS = ("decodes", "gate_checks", "gate_cap", "cost_usd", "cost_cap_usd", "sessions", "no_accept_streak")


def load_budget(run: Run) -> Dict:
    """Budget counters as a dict (``toyrepair.budget.Budget`` fields). Local adapter until
    W1-A2's ``Budget.load`` exists."""
    path = os.path.join(run.dir, "budget.json")
    try:
        from .budget import Budget
        return asdict(Budget.load(path))
    except (NotImplementedError, ImportError):
        pass
    except FileNotFoundError:
        return {"decodes": 0.0, "gate_checks": 0, "gate_cap": 0, "cost_usd": 0.0,
                "cost_cap_usd": run.cost_cap_usd, "sessions": 0, "no_accept_streak": 0}
    if not os.path.exists(path):
        return {"decodes": 0.0, "gate_checks": 0, "gate_cap": 0, "cost_usd": 0.0,
                "cost_cap_usd": run.cost_cap_usd, "sessions": 0, "no_accept_streak": 0}
    with open(path) as fh:
        d = json.load(fh)
    return {k: d.get(k, 0) for k in BUDGET_FIELDS}


def save_budget(run: Run, b: Dict) -> None:
    path = os.path.join(run.dir, "budget.json")
    try:
        from .budget import Budget
        Budget(**{k: b[k] for k in BUDGET_FIELDS}).save(path)
        return
    except NotImplementedError:
        pass
    _atomic_write(path, json.dumps({k: b[k] for k in BUDGET_FIELDS}, indent=1))


def gate_check_cap(n: int) -> int:
    return 3 * (int(n) + 2)


# ----------------------------------------------------------------------------- atlas / branch storage

def save_pickle(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def atlas_path(run: Run, vid: str) -> str:
    return os.path.join(run.dir, "atlas", f"{vid}.pkl")


def branch_path(run: Run, sid: str) -> str:
    return os.path.join(run.dir, "branch", f"{sid}.pkl")


def get_atlas(run: Run, vid: Optional[str] = None):
    """The miner's Atlas for ``vid`` (default current), mining it on first use."""
    vid = vid or run.current
    p = atlas_path(run, vid)
    if os.path.exists(p):
        return load_pickle(p)
    from .cli import get_backend
    atlas = get_backend(run).mine(run, vid)
    save_pickle(p, atlas)
    from .dossier import render_atlas
    _atomic_write(p[:-4] + ".txt", render_atlas(atlas))
    return atlas


# ----------------------------------------------------------------------------- the loop

def init_run(run_id: str, arm: str, grid: str, setting: str, seed: int, npm: int,
             backend: str = "real", phase4_tools: bool = False, gate_cfg: Optional[Dict] = None,
             runs_root: Optional[str] = None, cost_cap_usd: float = RUN_COST_CAP_USD) -> Run:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if backend not in ("real", "mock"):
        raise ValueError("backend must be real or mock")
    d = run_dir(run_id, runs_root)
    if os.path.exists(os.path.join(d, "run.json")):
        raise FileExistsError(f"run {run_id} exists")
    for sub in ("patches", "scratch", "atlas", "branch", "sessions"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    run = Run(id=run_id, dir=d, arm=arm, grid=grid, setting=setting, seed=int(seed), npm=int(npm),
              backend=backend, phase4_tools=phase4_tools, gate_cfg=dict(gate_cfg or {}),
              cost_cap_usd=cost_cap_usd, created=now())
    with run_lock(d):
        save_run(run)
        _atomic_write(os.path.join(d, "versions.json"), json.dumps(versions(run), indent=1))
        save_budget(run, load_budget(run))
        ledger_append(run, "run", {"event": "init", "arm": arm, "cell": [grid, setting, seed, npm],
                                   "backend": backend, "phase4_tools": phase4_tools, "gate_cfg": run.gate_cfg})
    return run


def surgeon_agent(arm: str) -> str:
    return "surgeon-sonnet" if arm == "A3-sonnet" else "surgeon"


def _stop(run: Run, rule: str) -> str:
    run.stop = rule
    ledger_append(run, "note", {"event": "stop", "rule": rule})
    save_run(run)
    return f"DONE {rule}"


def _eligible(run: Run, atlas) -> List[int]:
    return [c for c in atlas.open_ids() if run.cooldown.get(c, 0) <= 0]


def _tick_cooldown(run: Run) -> None:
    run.cooldown = {c: n - 1 for c, n in run.cooldown.items() if n - 1 > 0}


def fallback_cluster(run: Run, atlas) -> Optional[int]:
    """Open cluster with the largest residual basin mass that is not on cool-down."""
    el = set(_eligible(run, atlas))
    cards = [c for c in atlas.cards if c.cluster in el]
    if not cards:
        return None
    return max(cards, key=lambda c: (c.r_cur, -c.cluster)).cluster


def next_action(run: Run) -> str:
    """One step of the loop (PLAN 6.10). Must be called with the run lock held."""
    if run.stop:
        return f"DONE {run.stop}"
    if run.invalid:
        return _stop(run, "invalid_audit")
    if run.pending:
        try:
            s = load_session(run, _pending_session(run))
            return f"PENDING {s.agent} {s.token}"
        except Exception:
            run.pending = None
    if run.arm not in LLM_ARMS:
        save_run(run)
        return (f"CODE python scripts/run_arm.py --arm {run.arm} --setting {run.setting} "
                f"--seed {run.seed} --npm {run.npm}")
    b = load_budget(run)
    # stop rules that do not need the atlas
    if b["cost_usd"] >= run.cost_cap_usd:
        return _stop(run, "cost_cap")
    if b["no_accept_streak"] >= NO_ACCEPT_STREAK_STOP:
        return _stop(run, "no_accept_streak")
    atlas = get_atlas(run)
    if run.n_first_clusters is None:
        run.n_first_clusters = len(atlas.cards)
        b["gate_cap"] = gate_check_cap(run.n_first_clusters)
        save_budget(run, b)
        ledger_append(run, "atlas", {"event": "first_atlas", "n_open": len(atlas.cards),
                                     "gate_cap": b["gate_cap"]}, version=run.current)
    if not atlas.open_ids():
        return _stop(run, "no_open_cluster")
    if b["gate_cap"] and b["gate_checks"] >= b["gate_cap"]:
        return _stop(run, "gate_cap")
    if run.target is None:
        # every open cluster on cool-down: let iterations pass (no session is spent)
        while not _eligible(run, atlas):
            _tick_cooldown(run)
        s = issue_token(run, "triage", "triage")
        ledger_append(run, "triage", {"event": "issued", "token_session": s.session}, session=s.session)
        save_run(run)
        return f"TRIAGE triage {s.token}"
    # triage chose: branch tool, then a surgeon session
    cluster = run.target
    s = issue_token(run, "surgeon", surgeon_agent(run.arm), cluster=cluster)
    run.iteration += 1
    if run.arm != "A3-nobranch":
        from .cli import get_backend
        table = get_backend(run).branch(run, run.current, cluster)
        save_pickle(branch_path(run, s.session), table)
        from .dossier import render_branch
        _atomic_write(branch_path(run, s.session)[:-4] + ".txt", render_branch(table))
        ledger_append(run, "branch", {"cluster": cluster, "k_commit": table.k_commit,
                                      "failure_class": table.failure_class,
                                      "n_decodes": table.n_decodes, "by": "orchestrator"},
                      session=s.session, version=run.current)
        b = load_budget(run)
        b["decodes"] += float(table.n_decodes)
        save_budget(run, b)
        # the card shows the branch result from now on
        atlas = get_atlas(run)
        atlas.cards = [dataclasses.replace(c, k_commit=table.k_commit, failure_class=table.failure_class)
                       if c.cluster == cluster else c for c in atlas.cards]
        save_pickle(atlas_path(run, run.current), atlas)
    ledger_append(run, "session", {"event": "start", "cluster": cluster, "agent": s.agent,
                                   "iteration": run.iteration}, session=s.session, version=run.current)
    save_run(run)
    return f"SESSION {s.agent} {s.token}"


def _pending_session(run: Run) -> str:
    with open(os.path.join(_token_dir(os.path.dirname(os.path.dirname(run.dir))), f"{run.pending}.json")) as fh:
        return json.load(fh)["session"]


def parse_triage(text: str) -> Optional[Dict]:
    """The triage's one-line JSON, or None. Accepts the JSON anywhere in the last message."""
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and isinstance(d.get("cluster"), int):
            return d
    return None


def record(token: str, summary: str = "", transcript: Optional[str] = None,
           runs_root: Optional[str] = None) -> Dict:
    """Close a session: transcript usage and cost, isolation audit, ledger record, loop state."""
    from . import transcripts as tr
    run, s = lookup_token(token, runs_root)
    with run_lock(run.dir):
        run = load_run(run.id, runs_root)
        s = load_session(run, s.session)
        if s.closed:
            return {"error": "already recorded", "session": s.session}
        path = transcript or tr.find_subagent_for_token(token)
        info = tr.session_report(path, run_id=run.id, other_runs=_other_runs(run, runs_root),
                                 expected_agent=s.agent) if path else {"transcript": None,
                                                                         "cost_usd": 0.0,
                                                                         "audit_hits": [],
                                                                         "final_text": ""}
        final = summary or info.get("final_text", "")
        if s.role == "surgeon":     # keep the RESULT line when the subagent wrote more than one line
            res_line = next((ln.strip() for ln in final.splitlines() if ln.strip().startswith("RESULT")), None)
            final = res_line or final
        b = load_budget(run)
        b["cost_usd"] += float(info.get("cost_usd", 0.0))
        out = {"run": run.id, "session": s.session, "role": s.role, "cost_usd": info.get("cost_usd", 0.0),
               "transcript": info.get("transcript")}
        if info.get("audit_hits"):
            run.invalid = f"audit hit in {s.session}"
            ledger_append(run, "invalidate", {"reason": "transcript audit hit", "hits": info["audit_hits"][:20]},
                          session=s.session)
        if s.role == "triage":
            choice = parse_triage(final)
            atlas = get_atlas(run)
            el = _eligible(run, atlas)
            fallback = choice is None or choice.get("cluster") not in el
            cluster = fallback_cluster(run, atlas) if fallback else choice["cluster"]
            run.target = cluster
            _tick_cooldown(run)
            ledger_append(run, "triage", {"event": "record", "choice": choice, "cluster": cluster,
                                          "fallback": fallback, "usage": info}, session=s.session)
            out.update(cluster=cluster, fallback=fallback)
        else:
            accepted = s.accepted
            b["sessions"] += 1
            if accepted:
                b["no_accept_streak"] = 0
            else:
                b["no_accept_streak"] += 1
                run.cooldown[s.cluster] = COOLDOWN_ITERATIONS
            run.target = None
            ledger_append(run, "session", {"event": "end", "cluster": s.cluster, "accepted": accepted,
                                           "calls": s.calls, "submissions": s.submissions,
                                           "summary": final[:300], "usage": info},
                          session=s.session, version=run.current)
            out.update(accepted=accepted, calls=s.calls, submissions=s.submissions,
                       summary=final[:200], context_exceeded=info.get("context_exceeded", False))
        out["audit_hits"] = len(info.get("audit_hits", []))
        s.closed = True
        save_session(run, s)
        save_budget(run, b)
        run.pending = None
        save_run(run)
    return out


def _other_runs(run: Run, runs_root: Optional[str] = None) -> List[str]:
    root = repair_root(runs_root)
    try:
        return [d for d in os.listdir(root) if d != run.id and not d.startswith("_")
                and os.path.isdir(os.path.join(root, d))]
    except FileNotFoundError:
        return []


def status(runs_root: Optional[str] = None) -> List[str]:
    """One line per run under ``$TOYPREF_RUNS/repair``: id, arm, cell, current version,
    sessions, gate checks, cost, stop rule or "running"."""
    root = repair_root(runs_root)
    lines = []
    if not os.path.isdir(root):
        return lines
    for rid in sorted(os.listdir(root)):
        if rid.startswith("_") or not os.path.exists(os.path.join(root, rid, "run.json")):
            continue
        r = load_run(rid, runs_root)
        b = load_budget(r)
        state = r.stop or ("invalid" if r.invalid else ("pending" if r.pending else "running"))
        lines.append(f"{r.id}\t{r.arm}\t{r.grid}_{r.setting}_s{r.seed}_n{r.npm}\tcur={r.current}\t"
                     f"sessions={b['sessions']}\tgate={b['gate_checks']}/{b['gate_cap'] or '?'}\t"
                     f"cost=${b['cost_usd']:.2f}\tdecodes={int(b['decodes'])}\t{state}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toyrepair.orchestrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    for a in ("--run", "--arm", "--grid", "--setting"):
        p.add_argument(a, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--npm", type=int, required=True)
    p.add_argument("--backend", default="real", choices=("real", "mock"))
    p.add_argument("--phase4", action="store_true")
    p.add_argument("--gate_cfg", default="", help="JSON overrides of gate.GateConfig")
    p.add_argument("--cost_cap", type=float, default=RUN_COST_CAP_USD)
    sub.add_parser("next").add_argument("--run", required=True)
    sub.add_parser("status")
    p = sub.add_parser("record")
    p.add_argument("--token", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--transcript", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "init":
        r = init_run(args.run, args.arm, args.grid, args.setting, args.seed, args.npm, backend=args.backend,
                     phase4_tools=args.phase4, gate_cfg=json.loads(args.gate_cfg) if args.gate_cfg else None,
                     cost_cap_usd=args.cost_cap)
        print(f"INIT {r.id} {r.dir}")
    elif args.cmd == "next":
        d = run_dir(args.run)
        with run_lock(d):
            print(next_action(load_run(args.run)))
    elif args.cmd == "record":
        print(json.dumps(record(args.token, args.summary, args.transcript)))
    elif args.cmd == "status":
        lines = status()
        print("\n".join(lines) if lines else "no runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
