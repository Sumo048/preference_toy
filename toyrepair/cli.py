"""``toyrepair-tool``: the only interface of the surgeon and triage roles (docs/PLAN.md 7.2-7.5).

Import rule: this module imports neither ``toypref``, ``toyrepair.oracle`` nor torch at import
time. Commands load what they need lazily, and never the oracle (gate results come back through
``toyrepair.gate.GateResult.surgeon_view``).

Every command takes ``--token``; the CLI maps it to (run, session) and enforces:
  * the session limits: 20 calls (every call counts, including ``prompt`` and refused ones),
    3 gate submissions; a session closes at its first accepted edit;
  * the role: a triage token runs only ``triage-input``; a surgeon token never runs it;
  * the arm's command set (Section 9.2): a removed command returns an error;
  * path checks: every file argument resolves inside the run's scratch directory;
  * a ``tool`` ledger record for every call (command, options, ok, error, output size, decodes);
  * output caps: tables are cut to their row caps and every cut states how many rows it removed;
    any output beyond OUTPUT_TOKEN_CAP is cut with a note.

Backends: ``RealBackend`` calls the harness (policy, flowstate, miner, quick_check, sandbox,
edits, train, patches, gate). ``MockBackend`` fakes every tool output with no torch and no
oracle; it tests the mechanics of the CLI, the orchestrator and the LLM roles end to end. The
backend is chosen per run (``orchestrate init --backend``).

Owner: W1-D.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Sequence

SESSION_CALLS = 20
SESSION_SUBMISSIONS = 3
OUTPUT_TOKEN_CAP = 2000
#: commands with their own caps (the dossier parts are capped piecewise in toyrepair.dossier)
COMMAND_TOKEN_CAPS = {"prompt": 4000 + 6000 + 50, "dossier": 6000 + 50, "triage-input": 1000 + 1500 + 2000 + 50}

#: command -> one-line help
COMMANDS = {
    "prompt": "print the static prompt and your dossier (first call of a session)",
    "dossier": "print the variable part: failure card, branch table, ledger summary, budget",
    "atlas": "print the failure cards of the current version",
    "triage-input": "print the atlas and the ledger summary (triage only)",
    "branch": "run the branch tool on a cluster (--cluster, --steps, --M, --primitives)",
    "inspect": "one decode of one negative, <= 21 rows (--neg, --units)",
    "sandbox": "run Python on the export (--stdin heredoc or --file)",
    "quick_check": "oracle-free check of a candidate version (--version)",
    "graft_gated_path": "edit: gated hidden units beside a layer",
    "rank1": "edit: W <- W + s u v^T",
    "ablate_direction": "edit: project a direction out of a layer (side pre|post)",
    "set_units": "edit: write selected units' weights",
    "targeted_train": "edit: targeted training with the locality penalty",
    "submit_edit": "send a candidate version to the gate (max 3 per session)",
    "contrast": "activation contrast at step k for a cluster (Phase 4)",
    "unit_card": "correlates of one unit (Phase 4)",
}

#: command -> option syntax (shown in the surgeon's static prompt)
USAGE = {
    "prompt": ": this text and your dossier.",
    "dossier": ": the dossier part again.",
    "atlas": ": cards of all open clusters.",
    "branch": "`[--cluster ID] [--steps 0,2,4] [--M 32]`: branch table of a cluster (default yours).",
    "inspect": "`--neg UID [--version V] [--units LAYER:i,j]`: one decode of a negative: step, t, x, x_hat, |v|, x_hat in B_i, units.",
    "sandbox": "`--stdin` or `--file F.py`, `[--version V]`: Python (numpy, torch) in scratch; `export/` has weights.pt, model.py (net, decode(xi, c)), negatives.npz, basins.npz.",
    "quick_check": "`--version V`: still_bad (share of logged noises still in their basins), residual, drift (share of 4096 actions moved > median basin radius).",
    "graft_gated_path": "`--layer L --n_units N --w_in F --w_out F [--b_in F] [--src S] [--act silu|relu|linear] --det_source S --det_w F [--det_b B] [--det_thr T] [--det_k LO:HI] [--parent V]`",
    "rank1": "`--layer L --u F --v F --s S [--parent V]`",
    "ablate_direction": "`--layer L --v F --side pre|post [--parent V]`",
    "set_units": "`--layer L --idx 3,17 --w_in F --b F --w_out F [--parent V]`",
    "targeted_train": "`--params blocks.2,out --objective bc|npo|kto --steps N [--anchor dist|map|local|local_map|none] [--beta B] [--lr LR] [--data toypref:hopB:bsup | rescue:F.npy:bsup] [--parent V]`; bc needs --data; F.npy: one target row per member.",
    "submit_edit": "`--version V`: run the gate on a candidate.",
    "contrast": "`--layer L --k K [--cluster ID]`: top units, probe direction and AUC, velocity difference.",
    "unit_card": "`--layer L --idx I`: correlates of one unit.",
}

DIAG = ("prompt", "dossier", "atlas", "branch", "inspect", "sandbox", "quick_check", "submit_edit")
WEIGHT_CALLS = ("graft_gated_path", "rank1", "ablate_direction", "set_units")
TRAIN_CALLS = ("targeted_train",)
PHASE4 = ("contrast", "unit_card")
ARM_COMMANDS = {
    "A1": DIAG + TRAIN_CALLS,
    "A2": DIAG + WEIGHT_CALLS,
    "A3": DIAG + WEIGHT_CALLS + TRAIN_CALLS,
    "A3-nobranch": tuple(c for c in DIAG if c != "branch") + WEIGHT_CALLS + TRAIN_CALLS,
    "A3-sonnet": DIAG + WEIGHT_CALLS + TRAIN_CALLS,
    "A3-nogate": DIAG + WEIGHT_CALLS + TRAIN_CALLS,
}
EDIT_COMMANDS = WEIGHT_CALLS + TRAIN_CALLS
FILE_OPTIONS = ("file", "w_in", "w_out", "b_in", "det_w", "u", "v", "b")


def allowed_commands(arm: str, role: str, phase4: bool = False) -> tuple:
    if role == "triage":
        return ("triage-input",)
    cmds = ARM_COMMANDS.get(arm, ())
    return cmds + (PHASE4 if phase4 else ())


class ToolError(Exception):
    """A refused or failed call: the message goes to the caller, the call still counts."""


# ----------------------------------------------------------------------------- parser

def _int_list(s: str) -> List[int]:
    try:
        return [int(x) for x in s.split(",") if x != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {s!r}")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ToolError(f"bad arguments: {message}")


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="toyrepair-tool",
                 description="Tool CLI of the toyrepair harness. One command per call; pass --token on every call.")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    P = {}
    for name, helptext in COMMANDS.items():
        p = sub.add_parser(name, help=helptext, description=helptext)
        p.add_argument("--token", help="session token from your task message")
        P[name] = p
    P["branch"].add_argument("--cluster", type=int)
    P["branch"].add_argument("--steps", type=_int_list)
    P["branch"].add_argument("--M", type=int)
    P["branch"].add_argument("--primitives", default="default", choices=("default",))
    P["inspect"].add_argument("--neg", type=int, required=True)
    P["inspect"].add_argument("--version")
    P["inspect"].add_argument("--units", default="")
    P["sandbox"].add_argument("--stdin", action="store_true")
    P["sandbox"].add_argument("--file")
    P["sandbox"].add_argument("--version")
    P["quick_check"].add_argument("--version", required=True)
    P["submit_edit"].add_argument("--version", required=True)
    for e in EDIT_COMMANDS:
        P[e].add_argument("--parent")
    g = P["graft_gated_path"]
    g.add_argument("--layer", required=True)
    g.add_argument("--n_units", type=int, required=True)
    g.add_argument("--w_in", required=True)
    g.add_argument("--w_out", required=True)
    g.add_argument("--b_in")
    g.add_argument("--src")
    g.add_argument("--act", default="silu", choices=("silu", "relu", "linear"))
    g.add_argument("--det_source", required=True)
    g.add_argument("--det_w", required=True)
    g.add_argument("--det_b", type=float, default=0.0)
    g.add_argument("--det_thr", type=float, default=0.0)
    g.add_argument("--det_k")
    r = P["rank1"]
    r.add_argument("--layer", required=True)
    r.add_argument("--u", required=True)
    r.add_argument("--v", required=True)
    r.add_argument("--s", type=float, required=True)
    a = P["ablate_direction"]
    a.add_argument("--layer", required=True)
    a.add_argument("--v", required=True)
    a.add_argument("--side", required=True, choices=("pre", "post"))
    s = P["set_units"]
    s.add_argument("--layer", required=True)
    s.add_argument("--idx", type=_int_list, required=True)
    s.add_argument("--w_in", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--w_out", required=True)
    t = P["targeted_train"]
    t.add_argument("--params", required=True)
    t.add_argument("--objective", required=True, choices=("bc", "npo", "kto"))
    t.add_argument("--steps", type=int, required=True)
    # the surgeon hook allows no '+', so "local+map" is spelled local_map on the command line
    t.add_argument("--anchor", default="dist", choices=("dist", "map", "local", "local_map", "local+map", "none"))
    t.add_argument("--beta", type=float, default=10.0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--data", default="")
    c = P["contrast"]
    c.add_argument("--layer", required=True)
    c.add_argument("--k", type=int, required=True)
    c.add_argument("--cluster", type=int)
    u = P["unit_card"]
    u.add_argument("--layer", required=True)
    u.add_argument("--idx", type=int, required=True)
    return ap


# ----------------------------------------------------------------------------- path checks

def check_path(path: str, scratch: str) -> str:
    """Resolve ``path`` (relative paths: relative to scratch) and require it inside scratch."""
    if not path:
        raise ToolError("empty path")
    full = path if os.path.isabs(path) else os.path.join(scratch, path)
    real = os.path.realpath(full)
    root = os.path.realpath(scratch)
    if os.path.commonpath([real, root]) != root:
        raise ToolError(f"path {path!r} is outside your scratch directory")
    return real


def check_paths(args: argparse.Namespace, scratch: str) -> Dict[str, str]:
    """Every file option, plus the rescue-target file of --data, checked against scratch."""
    out = {}
    for name in FILE_OPTIONS:
        v = getattr(args, name, None)
        if v:
            out[name] = check_path(v, scratch)
    data = getattr(args, "data", "") or ""
    if data.startswith("rescue:"):
        parts = data.split(":")
        if len(parts) < 2 or not parts[1]:
            raise ToolError("--data rescue:<file.npy>[:support]")
        out["targets"] = check_path(parts[1], scratch)
    return out


# ----------------------------------------------------------------------------- output caps

def cap_output(text: str, max_tokens: int = OUTPUT_TOKEN_CAP) -> str:
    from .dossier import cap_lines
    return "\n".join(cap_lines(text.rstrip("\n").split("\n"), max_tokens=max_tokens, what="output lines"))


def _load_npy(path: str):
    import numpy as np
    if not os.path.exists(path):
        raise ToolError(f"file not found in scratch: {os.path.basename(path)}")
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        raise ToolError(f"cannot load {os.path.basename(path)}: {type(e).__name__}")
    return arr.astype("float32")


# ----------------------------------------------------------------------------- backends

def cached_cell(run, cache_dir: Optional[str] = None):
    """``load_cell(*run.cell_args)`` with the basins/vote built, cached in
    ``<cache_dir>/cell_cache.pt`` (default ``$TOYPREF_RUNS/cells/<cell key>/``; W2a): each CLI
    call is a new process, and a fresh cond cell costs ~40 s (pool decode + basins). The file
    sits in the mode-700 repo, so the sandbox user cannot read it; the surgeon reaches it only
    through this CLI. The cell is deterministic, so all runs on a cell share one file."""
    import torch
    from .cell import cell_key, load_cell
    if cache_dir is None:
        cache_dir = os.path.join(os.environ["TOYPREF_RUNS"], "cells", cell_key(*run.cell_args))
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "cell_cache.pt")
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            pass
    cell = load_cell(*run.cell_args)
    cell.vote                                    # build basins + vote once
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        torch.save(cell, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
    return cell


class RealBackend:
    """Calls the harness modules (W1-A1, W1-A2, W1-B). Every method returns (text, decodes)."""

    def __init__(self):
        self._cell = None

    # -- shared
    def cell(self, run):
        if self._cell is None:
            self._cell = cached_cell(run)
        return self._cell

    def version(self, run, vid: str):
        from .orchestrate import versions
        vs = versions(run)
        if vid not in vs:
            raise ToolError(f"unknown version {vid}")
        cell = self.cell(run)
        from . import policy
        if vid == "v0":
            return policy.base_version(cell)
        from .patches import rebuild
        return rebuild(cell, run.dir, list(vs[vid]["patches"]), vid)

    def cluster(self, run, cid: int, vid: Optional[str] = None):
        from .orchestrate import get_atlas
        atlas = get_atlas(run, vid)
        for c in atlas.clusters:
            if c.id == cid:
                return c
        raise ToolError(f"unknown cluster {cid}")

    # -- orchestrator hooks
    def mine(self, run, vid: str):
        from .miner import mine, log_purity
        cell = self.cell(run)
        atlas = mine(cell, self.version(run, vid))
        try:
            log_purity(cell, atlas, os.path.join(run.dir, "purity.csv"))
        except NotImplementedError:
            pass
        return atlas

    def branch(self, run, vid: str, cid: int, steps=None, M=None):
        from . import flowstate
        kw = {}
        if steps:
            kw["steps"] = tuple(steps)
        if M:
            kw["M"] = int(M)
        return flowstate.branch(self.cell(run), self.version(run, vid), self.cluster(run, cid), **kw)

    # -- tools
    def inspect(self, run, vid, uid, units):
        from . import flowstate
        fn = getattr(flowstate, "inspect", None)
        if fn is None:
            raise ToolError("inspect is not available yet (flowstate.inspect missing)")
        pairs = []                       # "LAYER:i,j" -> [(LAYER, i), (LAYER, j)] (flowstate.inspect's form)
        for part in filter(None, (units or "").split(";")):
            layer, _, idx = part.partition(":")
            try:
                pairs += [(layer.strip(), int(j)) for j in idx.split(",") if j.strip()]
            except ValueError:
                raise ToolError(f"bad --units {part!r}; use LAYER:i,j")
        try:
            res = fn(self.cell(run), self.version(run, vid), uid, pairs)
        except (ValueError, KeyError) as e:
            raise ToolError(f"inspect: {e}")
        text = res.render() if hasattr(res, "render") else str(res)
        return text, float(getattr(res, "n_decodes", 1))

    def sandbox(self, run, vid, code):
        from . import sandbox
        sandbox.export(self.cell(run), self.version(run, vid), run.scratch)
        r = sandbox.run(code, run.scratch)
        text = r.stdout
        if r.stderr:
            text += ("\n" if text else "") + "[stderr]\n" + r.stderr
        if r.timed_out:
            text += "\n[timed out]"
        if r.truncated:
            text += "\n[output truncated at the sandbox cap]"
        text += f"\n[exit {r.returncode}, decodes {r.decodes}]"
        return text, float(r.decodes)

    def quick_check(self, run, vid, cid):
        from .quick_check import quick_check
        r = quick_check(self.cell(run), self.version(run, run.current), self.version(run, vid),
                        self.cluster(run, cid))
        return (f"quick_check {vid}: still_bad={r.still_bad:.3f} residual={r.residual:.3f} drift={r.drift:.4f}",
                float(r.decodes))

    def edit(self, run, call, parent_vid, new_vid, args, paths):
        """-> (EditResult-like with .version/.patch/.record)."""
        from . import edits
        parent = self.version(run, parent_vid)
        pid = getattr(args, "_patch_id", None)
        if call == "graft_gated_path":
            w = _load_npy(paths["det_w"]).reshape(-1)
            k_range = None
            if args.det_k:
                lo, hi = args.det_k.split(":")
                k_range = (int(lo), int(hi))
            det = edits.Detector(source=args.det_source, w=tuple(float(x) for x in w), b=args.det_b,
                                 threshold=args.det_thr, k_range=k_range)
            return edits.graft_gated_path(parent, new_vid, args.layer, det, args.n_units, _load_npy(paths["w_in"]),
                                          _load_npy(paths["w_out"]),
                                          b_in=_load_npy(paths["b_in"]) if "b_in" in paths else None,
                                          src=args.src, act=args.act, patch_id=pid)
        if call == "rank1":
            return edits.rank1(parent, new_vid, args.layer, _load_npy(paths["u"]), _load_npy(paths["v"]), args.s,
                               patch_id=pid)
        if call == "ablate_direction":
            return edits.ablate_direction(parent, new_vid, args.layer, _load_npy(paths["v"]), args.side,
                                          patch_id=pid)
        if call == "set_units":
            return edits.set_units(parent, new_vid, args.layer, args.idx, _load_npy(paths["w_in"]),
                                   _load_npy(paths["b"]), _load_npy(paths["w_out"]), patch_id=pid)
        if call == "targeted_train":
            import numpy as np
            import torch
            from . import train
            cell = self.cell(run)
            cl = self.cluster(run, run_session_cluster(run, args))
            uids = np.asarray(cl.uids)
            if args.objective == "bc":
                if args.data.startswith("toypref:"):
                    _, rule, *sup = args.data.split(":")
                    data = train.toypref_rows(cell, uids, rule=rule, support=(sup[0] if sup else "bsup"))
                elif args.data.startswith("rescue:"):
                    parts = args.data.split(":")
                    tg = _load_npy(paths["targets"]).reshape(-1, 2)
                    if tg.shape[0] != len(uids):
                        raise ToolError(f"rescue targets need {len(uids)} rows (one per member), got {tg.shape[0]}")
                    data = train.rescue_rows(cell, uids, torch.from_numpy(tg),
                                             support=(parts[2] if len(parts) > 2 else "bsup"))
                else:
                    raise ToolError("objective bc needs --data toypref:<rule>:<support> or rescue:<file.npy>:<support>")
            else:
                data = train.TrainData(neg_uids=uids, source=f"cluster:{cl.id}")
            spec = train.TrainSpec(params=tuple(p for p in args.params.split(",") if p), objective=args.objective,
                                   steps=int(args.steps), anchor=args.anchor.replace("local_map", "local+map"),
                                   beta=args.beta, lr=args.lr)
            return train.targeted_train(cell, parent, new_vid, spec, data, patch_id=pid)
        raise ToolError(f"unknown edit call {call}")

    def save_patch(self, run, res, pid):
        from .patches import save_patch
        res.patch.id = pid
        save_patch(run.dir, res.patch)

    def gate(self, run, vid, cid, check_index):
        from . import gate
        cfg = gate.GateConfig(**{**run.gate_cfg, **({"accept_all": True} if run.arm == "A3-nogate" else {})})
        cl = self.cluster(run, cid)
        res = gate.gate_check(self.cell(run), self.version(run, "v0"), self.version(run, run.current),
                              self.version(run, vid), list(int(u) for u in cl.uids), check_index,
                              scope="cluster", cfg=cfg)
        return res

    def contrast(self, run, args):
        raise ToolError("contrast is not available until Phase 4")

    def unit_card(self, run, args):
        raise ToolError("unit_card is not available until Phase 4")


def run_session_cluster(run, args) -> int:
    return getattr(args, "_session_cluster")


class MockBackend:
    """Fake tool outputs: deterministic, no torch, no oracle, no toypref. For mechanics tests
    of the CLI, the orchestrator, the dossier and the LLM roles. Mock gate: a submission is
    accepted iff it is the session's 2nd or later and its candidate passed quick_check with
    still_bad < 0.5 (so a sensible surgeon can reach an accept)."""

    N_CLUSTERS = 3

    @staticmethod
    def _h(*parts) -> float:
        d = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
        return int(d[:8], 16) / 0xFFFFFFFF

    def mine(self, run, vid: str):
        import numpy as np
        from .miner import Atlas, Cluster, FailureCard
        from .orchestrate import ledger_records
        accepted = {(r.get("data") or {}).get("cluster") for r in ledger_records(run, "gate")
                    if ((r.get("data") or {}).get("surgeon_view") or {}).get("accepted")}
        clusters, cards = [], []
        uid0 = 1000
        for j in range(self.N_CLUSTERS):
            n = 4 + 2 * j
            uids = np.arange(uid0, uid0 + n, dtype=np.int64)
            uid0 += 100
            clusters.append(Cluster(id=j, members=np.arange(n), uids=uids))
            r_ref = round(0.6 - 0.15 * j, 3)
            r_cur = 0.01 if j in accepted else r_ref
            if r_cur >= 0.02:
                cards.append(FailureCard(cluster=j, n_members=n, r_ref=r_ref, r_cur=r_cur,
                                         c_range=(round(0.1 + 0.25 * j, 2), round(0.3 + 0.25 * j, 2)),
                                         cohesion=round(0.8 - 0.1 * j, 2), k_commit=None, failure_class="?",
                                         exemplars=tuple(int(u) for u in uids[:3])))
        cards.sort(key=lambda c: -c.r_cur)
        return Atlas(version_id=vid, clusters=clusters, cards=cards, affinity=np.eye(1), residual=np.zeros(1),
                     meta={"mock": True})

    def branch(self, run, vid, cid, steps=None, M=None):
        from .flowstate import BranchRow, BranchTable, Primitive
        steps = tuple(steps or (0, 2, 4, 6, 8, 10, 12, 14, 16))
        M = int(M or 32)
        rows = []
        for k in steps:
            p = max(0.0, 0.9 - 0.11 * k + 0.05 * self._h(cid, k))
            best = ()
            if k in (2, 4, 6):
                best = ((f"push(d{cid + 1},1.0r,k{k},h3)", round(0.7 - 0.05 * k, 2)),
                        (f"push(d{cid + 5},0.5r,k{k},h3)", round(0.4 - 0.03 * k, 2)))
            rows.append(BranchRow(k=k, t=round(1 - k / 20, 2), p=round(p, 3), ci=(round(max(p - 0.08, 0), 3),
                                  round(min(p + 0.08, 1), 3)), n=M, best=best))
        return BranchTable(cluster=cid, rows=rows, k_commit=6, failure_class="R",
                           best_primitive=Primitive(kind="push", u=(1.0, 0.0), alpha=1.0, k=4,
                                                    name=f"push(d{cid + 1},1.0r,k4,h3)"),
                           n_decodes=M * len(rows) * 6)

    def inspect(self, run, vid, uid, units):
        lines = ["step  t     x                 x_hat             |v|    x_hat_in_B" + ("  units" if units else "")]
        for k in range(0, 21, 2):
            a = self._h(uid, k)
            lines.append(f"{k:>4}  {1 - k / 20:.2f}  ({a * 2 - 1:+.3f},{a - 0.5:+.3f})  "
                         f"({2.5 + a:+.3f},{0.5 * a:+.3f})  {1 + a:.3f}  {int(k >= 6)}"
                         + (f"  {units}=[{a:.3f}]" if units else ""))
        return "\n".join(lines), 1.0

    def sandbox(self, run, vid, code):
        export = os.path.join(run.scratch, "export")
        os.makedirs(export, exist_ok=True)
        model = os.path.join(export, "model.py")
        if not os.path.exists(model):
            with open(model, "w") as fh:
                fh.write(_MOCK_MODEL)
            import numpy as np
            g = np.random.default_rng(0)
            np.savez(os.path.join(export, "negatives.npz"), uid=np.arange(1000, 1004), c=g.random(4).astype("f4"),
                     xi=g.standard_normal((4, 2)).astype("f4"), a=g.standard_normal((4, 2)).astype("f4"))
        script = os.path.join(run.scratch, f"_sandbox_{int(time.time() * 1000)}.py")
        with open(script, "w") as fh:
            fh.write(_MOCK_AUDIT + "\n" + code)
        try:
            r = subprocess.run([sys.executable, "-I", script], cwd=run.scratch, capture_output=True, text=True,
                               timeout=120)
            text = r.stdout + (("\n[stderr]\n" + r.stderr[-1500:]) if r.stderr else "") + f"\n[exit {r.returncode}, mock decodes 0]"
        except subprocess.TimeoutExpired:
            text = "[timed out after 120 s]"
        return text, 0.0

    def quick_check(self, run, vid, cid):
        sb = round(0.2 + 0.6 * self._h(vid, "sb"), 3)
        return (f"quick_check {vid}: still_bad={sb:.3f} residual={sb * 0.8:.3f} drift={0.02 + 0.05 * self._h(vid, 'd'):.4f}",
                256.0)

    def edit(self, run, call, parent_vid, new_vid, args, paths):
        for name, p in paths.items():
            if name != "file" and not os.path.exists(p):
                raise ToolError(f"file not found in scratch: {os.path.basename(p)}")
        from types import SimpleNamespace
        rec = {"call": call, "parent": parent_vid, "version": new_vid, "frac_weights_changed": 0.001,
               "units_added": getattr(args, "n_units", 0) or 0, "frob": 0.1, "decodes": 0.0, "ok": True, "error": ""}
        return SimpleNamespace(version=new_vid, patch=None, record=SimpleNamespace(**rec))

    def save_patch(self, run, res, pid):
        return None

    def gate(self, run, vid, cid, check_index):
        from .gate import GateResult, TEST_NAMES
        from .orchestrate import ledger_records
        qc = [r for r in ledger_records(run, "quick_check") if r.get("version") == vid]
        good = bool(qc) and float((qc[-1].get("data") or {}).get("still_bad", 1)) < 0.5
        sid = getattr(self, "_session", None)
        nth = sum(1 for r in ledger_records(run, "gate") if r.get("session") == sid) + 1
        ok = good and nth >= 2
        passed = {t: True for t in TEST_NAMES}
        if not ok:
            passed["target"] = False
            if not good:
                passed["drift"] = False
        fall = 0.03 if ok else 0.004
        return GateResult(accepted=ok or run.arm == "A3-nogate", passed=passed, target_fall=fall,
                          target_fall_ci=(fall - 0.006, fall + 0.006), numbers={"mock": 1.0},
                          check_index=check_index, n_g=16384, scope="cluster", cpu_s=0.0)

    def contrast(self, run, args):
        return f"(mock) contrast {args.layer} k={args.k}: top units 17 (d=1.2), 40 (d=0.9); probe AUC 0.91", 64.0

    def unit_card(self, run, args):
        return f"(mock) unit {args.layer}[{args.idx}]: corr t -0.3, c 0.1, x_hat in basin 0.6", 64.0


_MOCK_MODEL = '''"""Mock export: a numpy-only stand-in for the policy (mechanics tests only)."""
import numpy as np
def velocity(x, t, c):
    return x - np.stack([np.cos(6.28 * c), np.sin(6.28 * c)], -1) * 2.5
def decode(xi, c, steps=20):
    x = np.asarray(xi, dtype=np.float32).copy(); c = np.asarray(c, dtype=np.float32)
    for k in range(steps):
        x = x - velocity(x, 1 - k / steps, c) / steps
    return x
'''

_MOCK_AUDIT = '''import sys as _s
try:
    import numpy, torch          # torch calls ldconfig through subprocess at import: import before the hook
except Exception:
    pass
def _hook(ev, a):
    if ev == "import" and a and str(a[0]).split(".")[0] in ("toypref", "toyrepair"):
        raise RuntimeError("blocked import")
    if ev in ("subprocess.Popen", "os.system", "socket.connect", "os.exec", "os.posix_spawn", "os.fork"):
        raise RuntimeError("blocked: " + ev)
_s.addaudithook(_hook)
del _hook'''

_BACKENDS: Dict[str, object] = {}


def get_backend(run):
    if run.backend == "mock":
        return MockBackend()
    key = f"real:{run.id}"
    if key not in _BACKENDS:
        _BACKENDS[key] = RealBackend()
    return _BACKENDS[key]


# ----------------------------------------------------------------------------- commands

#: gate test names as the surgeon sees them (the internal name of test 4 mentions forgotten modes)
GATE_TEST_LABELS = {"target": "1_target", "progress": "2_progress", "drift": "3_drift",
                    "other_forgotten": "4_other", "between": "5_between", "unit": "6_unit"}


def _fmt_gate(res, remaining: int) -> str:
    sv = res.surgeon_view()
    tests = " ".join(f"{GATE_TEST_LABELS.get(t, t)}={'pass' if ok else 'FAIL'}" for t, ok in sv["passed"].items())
    lo, hi = sv["target_fall_ci"]
    return (f"gate: {'ACCEPT' if sv['accepted'] else 'REJECT'}\n{tests}\n"
            f"target_fall={sv['target_fall']:.4f} 95% CI [{lo:.4f}, {hi:.4f}]\n"
            f"submissions left this session: {remaining}")


def execute(run, sess, args, stdin_text: str = "") -> (str, Dict):
    """Run one allowed command. Returns (text, ledger data). Raises ToolError."""
    from . import dossier as ds
    from . import orchestrate as orc
    be = get_backend(run)
    if isinstance(be, MockBackend):
        be._session = sess.session
    cmd = args.command
    info: Dict = {"decodes": 0.0}
    paths = check_paths(args, run.scratch)
    args._session_cluster = sess.cluster

    if cmd == "triage-input":
        return ds.triage_input(run), info
    if cmd == "prompt":
        d = ds.build(run, sess)
        info.update(tokens_static=d.tokens_static, tokens_variable=d.tokens_variable)
        return d.text(), info
    if cmd == "dossier":
        return ds.variable_part(run, sess), info
    if cmd == "atlas":
        return ds.render_atlas(orc.get_atlas(run)), info
    if cmd == "branch":
        cid = args.cluster if args.cluster is not None else sess.cluster
        table = be.branch(run, run.current, cid, steps=args.steps, M=args.M)
        info["decodes"] = float(table.n_decodes)
        info.update(cluster=cid, k_commit=table.k_commit, failure_class=table.failure_class)
        return ds.render_branch(table), info
    if cmd == "inspect":
        text, dec = be.inspect(run, _version_arg(run, sess, args.version), args.neg, args.units)
        info["decodes"] = dec
        return "\n".join(ds.cap_lines(text.split("\n"), max_rows=22, what="rows")), info
    if cmd == "sandbox":
        if args.stdin == bool(args.file):
            raise ToolError("sandbox needs exactly one of --stdin or --file")
        if args.file:
            with open(paths["file"]) as fh:
                code = fh.read()
        else:
            code = stdin_text
        if not code.strip():
            raise ToolError("no code given")
        text, dec = be.sandbox(run, _version_arg(run, sess, args.version), code)
        info["decodes"] = dec
        info["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        info["code"] = code[:4000]
        return text, info
    if cmd == "quick_check":
        vid = _version_arg(run, sess, args.version)
        text, dec = be.quick_check(run, vid, sess.cluster)
        info["decodes"] = dec
        nums = dict(kv.split("=") for kv in text.split(": ", 1)[1].split())
        orc.ledger_append(run, "quick_check", {k: float(v) for k, v in nums.items()}, session=sess.session,
                          version=vid)
        return text, info
    if cmd in EDIT_COMMANDS:
        parent = _version_arg(run, sess, args.parent)
        new_vid = f"v{run.next_version}"
        # the patch id is fixed before the edit: a graft's id is part of the version hash, so
        # renaming the patch afterwards would break the rebuild check (W2a fix)
        args._patch_id = f"p{run.next_patch:04d}"
        res = be.edit(run, cmd, parent, new_vid, args, paths)
        rec = res.record
        if not getattr(rec, "ok", True) or res.version is None:
            raise ToolError(f"edit rejected: {getattr(rec, 'error', '') or 'unknown reason'}")
        data = dict(orc._jsonable(rec))
        pid = orc.new_patch_id(run)
        assert pid == args._patch_id
        be.save_patch(run, res, pid)
        vid = orc.register_version(run, parent, pid, sess.session, cmd)
        assert vid == new_vid
        sess.versions.append(vid)
        data.update(patch_id=pid, cluster=sess.cluster)
        orc.ledger_append(run, "edit", data, session=sess.session, version=vid)
        info["decodes"] = float(getattr(rec, "decodes", 0.0) or 0.0)
        return (f"created candidate {vid} (parent {parent}, {cmd}): frac_weights_changed="
                f"{getattr(rec, 'frac_weights_changed', 0):.5f} units_added={getattr(rec, 'units_added', 0)} "
                f"frob={getattr(rec, 'frob', 0):.4f}"), info
    if cmd == "submit_edit":
        vid = args.version
        if vid not in sess.versions:
            raise ToolError(f"{vid} is not a candidate made in this session")
        if sess.submissions >= SESSION_SUBMISSIONS:
            raise ToolError("submission limit reached for this session")
        b = orc.load_budget(run)
        if b["gate_cap"] and b["gate_checks"] >= b["gate_cap"]:
            raise ToolError("the run has no gate checks left")
        sess.submissions += 1
        res = be.gate(run, vid, sess.cluster, b["gate_checks"])
        b["gate_checks"] += 1
        b["decodes"] += 3.0 * float(getattr(res, "n_g", 0))
        orc.save_budget(run, b)
        orc.ledger_append(run, "gate", {"cluster": sess.cluster, "call": orc.versions(run)[vid]["call"],
                                        "surgeon_view": res.surgeon_view(), "result": orc._jsonable(res)},
                          session=sess.session, version=vid)
        if res.accepted:
            sess.accepted = vid
            run.current = vid
        return _fmt_gate(res, SESSION_SUBMISSIONS - sess.submissions), info
    if cmd == "contrast":
        text, dec = be.contrast(run, args)
        info["decodes"] = dec
        return text, info
    if cmd == "unit_card":
        text, dec = be.unit_card(run, args)
        info["decodes"] = dec
        return text, info
    raise ToolError(f"unknown command {cmd}")


def _version_arg(run, sess, vid: Optional[str]) -> str:
    """A version the session may use: the current version, an ancestor, or its own candidates."""
    from .orchestrate import versions
    if not vid:
        return run.current
    vs = versions(run)
    if vid not in vs:
        raise ToolError(f"unknown version {vid}")
    allowed = set(sess.versions)
    v = run.current
    while v is not None:
        allowed.add(v)
        v = vs[v]["parent"]
    if vid not in allowed:
        raise ToolError(f"version {vid} is not available in this session")
    return vid


def _redact_args(argv: Sequence[str]) -> List[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            out.append("<token>")
            skip = False
        elif a == "--token":
            out.append(a)
            skip = True
        elif a.startswith("--token="):
            out.append("--token=<token>")
        else:
            out.append(a)
    return out


def main(argv: Optional[List[str]] = None, stdin=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    if not argv or argv[0] in ("-h", "--help"):
        ap.print_help()
        return 0
    try:
        args = ap.parse_args(argv)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:          # --help of a subcommand
        return int(e.code or 0)
    if args.command is None:
        ap.print_help()
        return 0
    from . import orchestrate as orc
    try:
        run, sess = orc.lookup_token(args.token or "")
    except (KeyError, FileNotFoundError):
        print("error: unknown or missing --token", file=sys.stderr)
        return 2
    stdin_text = ""
    if args.command == "sandbox" and getattr(args, "stdin", False):
        stdin_text = (stdin or sys.stdin).read()
    t0 = time.time()
    with orc.run_lock(run.dir):
        run = orc.load_run(run.id)
        sess = orc.load_session(run, sess.session)
        rc, text, err, info = 0, "", "", {}
        refused = None
        if sess.closed:
            refused = "this session is closed"
        elif sess.accepted:
            refused = f"the session ended: the gate accepted {sess.accepted}. Give your RESULT line."
        elif sess.calls >= SESSION_CALLS:
            refused = f"session call limit reached ({SESSION_CALLS}). Give your RESULT line."
        elif args.command not in allowed_commands(run.arm, sess.role, run.phase4_tools):
            refused = f"command {args.command} is not available in this session"
        elif run.stop:
            refused = f"the run has stopped ({run.stop})"
        sess.calls += 1
        if refused:
            rc, err = 2, refused
        else:
            try:
                text, info = execute(run, sess, args, stdin_text)
                text = cap_output(text, COMMAND_TOKEN_CAPS.get(args.command, OUTPUT_TOKEN_CAP))
            except ToolError as e:
                rc, err = 2, str(e)
            except NotImplementedError as e:
                rc, err = 2, f"{args.command} is not available yet (harness module not implemented: {e or 'stub'})"
            except Exception as e:      # report, never leak a traceback with repo paths
                rc, err = 2, f"{args.command} failed: {type(e).__name__}: {str(e)[:300]}"
        if not refused and info.get("decodes"):
            b = orc.load_budget(run)
            b["decodes"] += float(info["decodes"])
            orc.save_budget(run, b)
        left = SESSION_CALLS - sess.calls
        footer = f"[calls left: {left}, submissions left: {SESSION_SUBMISSIONS - sess.submissions}]"
        orc.ledger_append(run, "tool", {"cmd": args.command, "argv": _redact_args(argv), "ok": rc == 0,
                                        "error": err, "out_chars": len(text), "wall_s": round(time.time() - t0, 3),
                                        **{k: v for k, v in info.items() if k != "code"},
                                        **({"code": info["code"]} if "code" in info else {})},
                          session=sess.session, version=run.current)
        orc.save_session(run, sess)
        orc.save_run(run)
    if rc == 0:
        print(text)
        if sess.role != "triage":
            print(footer)
    else:
        print(f"error: {err}")
        if sess.role != "triage":
            print(footer)
    return rc


if __name__ == "__main__":
    sys.exit(main())
