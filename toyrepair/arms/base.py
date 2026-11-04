"""Shared arm types, the code-arm registry, and the machinery the code arms share. Owner: W1-C.

Every arm module exposes ``run(cell, run, svc=None) -> ArmOutput``. ``run`` is a
``toyrepair.orchestrate.Run`` (only ``.id`` and ``.dir`` are read). ``svc`` is a ``Services``
bundle of the harness functions the arm calls (policy, miner, branch tool, gate, edits,
patches). It defaults to the real modules; tests pass fakes.

Shared pieces:

* ``RunIO``: run directory, version ids (``v0`` = pi_ref, ``v1``, ...), ledger records (through
  ``toyrepair.ledger`` when implemented, else a JSONL adapter with the same fields), patches.
* ``toypref_arm``: train one toypref config on all negatives exactly as ``scripts/cond_cell.py``
  (B1a, B1b, B5) and wrap the result as a ``policy.Version`` with a full-tensor patch.
* ``iterate``: the A3 loop schedule with code in place of the LLM (B4, R0, D1; PLAN 6.10):
  mine -> pick the open cluster with the largest residual mass not on cool-down (class R first
  when the arm needs a primitive) -> up to 3 candidates through the gate (cluster scope) -> on
  accept the candidate becomes current and the miner reruns; stop rules: no open cluster,
  3 cluster attempts in a row without an accept, gate checks reach 3 x (first-atlas clusters + 2),
  or ``max_rounds`` attempts (the pilot uses 1: one repair, no recursion).
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

#: arm name -> module under toyrepair.arms
CODE_ARMS: Dict[str, str] = {"B0": "b0", "B1a": "b1a", "B1b": "b1b", "B4": "b4", "B5": "b5",
                             "R0": "r0", "D1": "d1"}
#: candidates a code arm may send to the gate for one cluster (same as an LLM session's submissions)
MAX_CANDIDATES_PER_CLUSTER = 3
COOLDOWN = 2
NO_ACCEPT_STREAK_STOP = 3


@dataclass
class ArmOutput:
    """arm; final version id; accepted version ids in order; gate_checks used; decodes used;
    stop rule (or "single" for one-shot arms); meta (e.g. the toypref config chosen).
    final: the final ``policy.Version`` object (not serialised; for evaluation by the caller)."""
    arm: str
    version: str
    accepted: List[str] = field(default_factory=list)
    gate_checks: int = 0
    decodes: float = 0.0
    stop: str = "single"
    meta: Dict = field(default_factory=dict)
    final: Optional[object] = field(default=None, repr=False)

    def to_json(self) -> Dict:
        return {"arm": self.arm, "version": self.version, "accepted": list(self.accepted),
                "gate_checks": self.gate_checks, "decodes": float(self.decodes), "stop": self.stop,
                "meta": _json_safe(self.meta)}


def get(name: str) -> Callable:
    """The ``run`` function of a code arm."""
    return importlib.import_module(f"toyrepair.arms.{CODE_ARMS[name]}").run


# ---------------------------------------------------------------------------------------- services
@dataclass
class Services:
    """Harness functions the code arms call (signatures as in docs/INTERFACES.md)."""
    base_version: Callable          # policy.base_version(cell) -> Version
    decode: Callable                # policy.decode(version, xi, c, save_states, capture, start_step)
    velocity: Callable              # policy.velocity(version, x, t, c, capture) -> (v, acts)
    mine: Callable                  # miner.mine(cell, version, branch_tables=None) -> Atlas
    branch: Callable                # flowstate.branch(cell, version, cluster, ...) -> BranchTable
    restore_and_decode: Callable    # flowstate.restore_and_decode(version, state, primitive)
    flow_state: Callable            # flowstate.FlowState(x, k, c)
    gate_check: Callable            # gate.gate_check(cell, base, current, new, uids, check_index, scope)
    targeted_train: Callable        # train.targeted_train(cell, parent, new_id, spec, data) -> EditResult
    toypref_rows: Callable          # train.toypref_rows(cell, uids, rule, support) -> TrainData
    rescue_rows: Callable           # train.rescue_rows(cell, uids, targets, support) -> TrainData
    train_spec: Callable            # train.TrainSpec(...)
    save_patch: Callable            # patches.save_patch(run_dir, patch) -> path
    make_patch: Callable            # patches.Patch(...)
    make_version: Callable          # policy.Version(...)


def _base_version_adapter(cell):
    """Local adapter for ``policy.base_version`` until W1-A1 merges: v0 = deep copy of pi_ref's net."""
    import copy
    from toyrepair import policy
    try:
        return policy.base_version(cell)
    except NotImplementedError:
        net = copy.deepcopy(cell.ref.net).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return policy.Version(id="v0", parent=None, patches=(), net=net, grid=cell.grid, meta={"adapter": True})


def real_services() -> Services:
    from toyrepair import flowstate, gate, miner, patches, policy, train
    return Services(base_version=_base_version_adapter, decode=policy.decode, velocity=policy.velocity,
                    mine=miner.mine, branch=flowstate.branch,
                    restore_and_decode=flowstate.restore_and_decode, flow_state=flowstate.FlowState,
                    gate_check=gate.gate_check, targeted_train=train.targeted_train,
                    toypref_rows=train.toypref_rows, rescue_rows=train.rescue_rows,
                    train_spec=train.TrainSpec, save_patch=patches.save_patch, make_patch=patches.Patch,
                    make_version=policy.Version)


# ---------------------------------------------------------------------------------------- run io
def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "item") and callable(x.item) and getattr(x, "ndim", 1) == 0:
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "__dataclass_fields__"):
        return _json_safe({k: getattr(x, k) for k in x.__dataclass_fields__})
    return repr(x)


class RunIO:
    """Run directory, version ids, ledger and patches for one code-arm run."""

    def __init__(self, run, svc: Services):
        self.run, self.svc = run, svc
        self.dir = run.dir
        os.makedirs(os.path.join(self.dir, "patches"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "scratch"), exist_ok=True)
        self.ledger_path = os.path.join(self.dir, "ledger.jsonl")
        self._n_versions = 0
        self._n_patches = 0
        self._ledger = None
        try:
            from toyrepair.ledger import Ledger, LedgerRecord
            self._ledger, self._Rec = Ledger(self.ledger_path), LedgerRecord
        except Exception:      # pragma: no cover - module missing
            self._ledger = None

    def new_version_id(self) -> str:
        self._n_versions += 1
        return f"v{self._n_versions}"

    def new_patch_id(self) -> str:
        self._n_patches += 1
        return f"p{self._n_patches:04d}"

    def log(self, kind: str, data: Dict, version: Optional[str] = None, session: Optional[str] = None) -> None:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        data = _json_safe(data)
        if self._ledger is not None:
            try:
                self._ledger.append(self._Rec(kind=kind, run=self.run.id, ts=ts, data=data,
                                              session=session, version=version))
                return
            except NotImplementedError:
                self._ledger = None          # W1-A2 not merged yet: local adapter below
        rec = {"kind": kind, "run": self.run.id, "ts": ts, "session": session, "version": version, "data": data}
        with open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def save_patch(self, patch) -> Optional[str]:
        if patch is None:
            return None
        try:
            return self.svc.save_patch(self.dir, patch)
        except NotImplementedError:          # W1-B not merged yet: local adapter
            d = os.path.join(self.dir, "patches")
            np.savez(os.path.join(d, f"{patch.id}.npz"), **{k: np.asarray(v) for k, v in patch.arrays.items()})
            meta = {"id": patch.id, "call": patch.call, "args": _json_safe(patch.args),
                    "parent_version": patch.parent_version, "version": patch.version, "sha256": patch.sha256}
            p = os.path.join(d, f"{patch.id}.json")
            with open(p, "w") as fh:
                json.dump(meta, fh)
            return p


# ---------------------------------------------------------------------------------------- toypref arms
def toypref_train(cell, cfg: str, steps: int = 5000):
    """Train toypref config ``cfg`` on all of the cell's negatives exactly as
    ``scripts/cond_cell.py`` (same Ctx, same streams). Returns (CondFlow, info)."""
    from toypref.cond import experiment as E
    arm = E.ARMS[cfg]
    ctx = cell.toypref_ctx(steps=steps)
    if arm.rule == "pu":
        upool = E.make_pool(cell.ref, cell.seed, 20_000, purpose=E.P_PU + 1)
        ctx.pu = E.train_pu(ctx.ed, upool, cell.seed, prior=cell.n_forget / E.MIX.n_modes)
    pol, info = E.train_arm(arm, ctx)
    return pol, info


def version_from_net(io: RunIO, base, net, call: str, args: Dict, meta: Optional[Dict] = None):
    """Wrap a trained full network as a Version on top of ``base`` with one patch that stores
    every changed parameter tensor in full, keyed by parameter name of ``base.net``. Use
    ``call="targeted_train"``: ``patches.apply_patch`` applies such a patch as a full-tensor
    overwrite (W1-B), so ``patches.rebuild`` rebuilds B1a/B1b/B5 versions."""
    import torch
    vid, pid = io.new_version_id(), io.new_patch_id()
    base_sd = dict(base.net.named_parameters())
    new_sd = dict(net.named_parameters())
    if set(new_sd) != set(base_sd):
        raise ValueError("version_from_net: parameter names differ from the base net")
    arrays, n_changed, n_total, frob2 = {}, 0, 0, 0.0
    for k, v in new_sd.items():
        b = base_sd[k]
        n_total += b.numel()
        if not torch.equal(v, b):
            arrays[k] = v.detach().cpu().numpy().copy()
            n_changed += int((v != b).sum())
            frob2 += float(((v.double() - b.double()) ** 2).sum())
    stats = {"frac_weights_changed": n_changed / max(n_total, 1), "frob": frob2 ** 0.5, "units_added": 0}
    patch = io.svc.make_patch(id=pid, call=call, args=dict(args), arrays=arrays,
                              parent_version=base.id, version=vid)
    io.save_patch(patch)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    ver = io.svc.make_version(id=vid, parent=base.id, patches=tuple(base.patches) + (pid,), net=net,
                              grid=getattr(base, "grid", "cond"), meta={"call": call, **args, **(meta or {}), **stats})
    io.log("edit", {"call": call, "args": args, "parent": base.id, "patch_id": pid, **stats}, version=vid)
    return ver


def first_atlas_uids(cell, base, svc: Services, io: RunIO) -> List[List[int]]:
    """Uid lists of the first atlas's open clusters (global gate scope)."""
    atlas = svc.mine(cell, base)
    io.log("atlas", {"version": base.id, "open": atlas.open_ids(), "n_clusters": len(atlas.clusters)},
           version=base.id)
    by_id = {c.id: c for c in atlas.clusters}
    return [list(map(int, by_id[cid].uids)) for cid in atlas.open_ids()]


def gate_record(res) -> Dict:
    return {"accepted": bool(res.accepted), "passed": dict(res.passed), "target_fall": res.target_fall,
            "target_fall_ci": list(res.target_fall_ci), "numbers": dict(res.numbers),
            "check_index": res.check_index, "scope": res.scope, "cpu_s": res.cpu_s}


def toypref_arm(label: str, cfg: str, cell, run, svc: Optional[Services] = None,
                measure_gate: bool = False) -> ArmOutput:
    """B1a / B1b: one toypref config on all negatives, one training run, no gate decision.
    With ``measure_gate`` one global-scope gate check is run and logged (not decisive)."""
    svc = svc or real_services()
    io = RunIO(run, svc)
    base = svc.base_version(cell)
    io.log("run", {"arm": label, "cell": cell.key, "toypref_config": cfg})
    pol, info = toypref_train(cell, cfg)
    ver = version_from_net(io, base, pol.net, "targeted_train", {"toypref_config": cfg, "source": "toypref.train_arm"},
                           meta={k: v for k, v in info.items() if k != "train_history"})
    out = ArmOutput(arm=label, version=ver.id, accepted=[ver.id], meta={"toypref_config": cfg}, final=ver)
    if measure_gate:
        uids = first_atlas_uids(cell, base, svc, io)
        res = svc.gate_check(cell, base, base, ver, uids, check_index=0, scope="global")
        io.log("gate", gate_record(res), version=ver.id)
        out.gate_checks, out.meta["gate_accepted"] = 1, bool(res.accepted)
    io.log("note", {"stop": out.stop, **out.to_json()}, version=ver.id)
    return out


# ---------------------------------------------------------------------------------------- iterative loop
@dataclass
class RepairContext:
    """What a per-cluster repair function gets. ``table`` is the cluster's branch table under
    ``current`` (None when the arm does not need one)."""
    cell: object
    base: object
    current: object
    cluster: object
    card: object
    table: Optional[object]
    io: RunIO
    svc: Services


def iterate(arm: str, cell, run, svc: Optional[Services],
            repair: Callable[[RepairContext], Iterable], needs_branch: bool,
            max_rounds: Optional[int] = None, max_candidates: int = MAX_CANDIDATES_PER_CLUSTER) -> ArmOutput:
    """The code loop shared by B4, R0 and D1 (see module doc).

    ``repair(ctx)`` yields up to ``max_candidates`` candidate Versions (lazily, so a later
    variant is only built if the earlier one was rejected). ``needs_branch``: the arm needs a
    class-R cluster's best primitive; the branch tool runs on open clusters in order of residual
    mass until one is class R (class-U clusters go on cool-down)."""
    svc = svc or real_services()
    io = RunIO(run, svc)
    base = svc.base_version(cell)
    current = base
    io.log("run", {"arm": arm, "cell": cell.key, "needs_branch": needs_branch, "max_rounds": max_rounds})
    atlas = svc.mine(cell, current)
    n_first = len(atlas.cards)
    cap = 3 * (n_first + 2)
    io.log("atlas", {"version": current.id, "open": atlas.open_ids(), "n_clusters": len(atlas.clusters),
                     "gate_cap": cap}, version=current.id)
    out = ArmOutput(arm=arm, version=base.id, final=base, meta={"n_first_clusters": n_first, "gate_cap": cap,
                                                                   "rounds": []})
    cooldown: Dict[int, int] = {}
    streak, rounds = 0, 0
    tables: Dict[int, object] = {}        # branch tables under the current version

    def tick():
        for k in list(cooldown):
            cooldown[k] -= 1
            if cooldown[k] <= 0:
                del cooldown[k]

    while True:
        if not atlas.cards:
            out.stop = "no_open_cluster"
            break
        if max_rounds is not None and rounds >= max_rounds:
            out.stop = "max_rounds"
            break
        if out.gate_checks >= cap:
            out.stop = "gate_cap"
            break
        cands = [c for c in atlas.cards if c.cluster not in cooldown]
        if not cands:
            tick()
            continue
        by_id = {c.id: c for c in atlas.clusters}
        card, table = None, None
        for c in cands:
            if not needs_branch:
                card = c
                break
            if c.cluster not in tables:
                tables[c.cluster] = svc.branch(cell, current, by_id[c.cluster])
                out.decodes += float(getattr(tables[c.cluster], "n_decodes", 0) or 0)
                t = tables[c.cluster]
                io.log("branch", {"cluster": c.cluster, "k_commit": t.k_commit, "class": t.failure_class,
                                  "best": getattr(t.best_primitive, "name", None)}, version=current.id)
            if tables[c.cluster].failure_class == "R" and tables[c.cluster].best_primitive is not None:
                card, table = c, tables[c.cluster]
                break
            cooldown[c.cluster] = COOLDOWN       # class U: nothing to do for a primitive-based arm
        if card is None:
            if needs_branch and all(tables.get(c.cluster) is not None and tables[c.cluster].failure_class != "R"
                                    for c in atlas.cards):
                out.stop = "no_class_R_cluster"
                break
            tick()
            continue
        cluster = by_id[card.cluster]
        rounds += 1
        ctx = RepairContext(cell=cell, base=base, current=current, cluster=cluster, card=card, table=table,
                            io=io, svc=svc)
        accepted, tried = None, []
        gen = iter(repair(ctx))
        for _ in range(max_candidates):
            if out.gate_checks >= cap:
                break
            try:
                cand = next(gen)
            except StopIteration:
                break
            if cand is None:
                continue
            res = svc.gate_check(cell, base, current, cand, list(map(int, cluster.uids)),
                                 check_index=out.gate_checks, scope="cluster")
            out.gate_checks += 1
            io.log("gate", {"cluster": card.cluster, **gate_record(res)}, version=cand.id)
            tried.append({"version": cand.id, "variant": cand.meta.get("variant"), "accepted": bool(res.accepted)})
            if res.accepted:
                accepted = cand
                break
        out.meta["rounds"].append({"cluster": int(card.cluster), "r_cur": float(card.r_cur), "tried": tried,
                                   "accepted": accepted.id if accepted is not None else None,
                                   "k_commit": getattr(table, "k_commit", None),
                                   "primitive": getattr(getattr(table, "best_primitive", None), "name", None)})
        tick()
        if accepted is not None:
            current, streak = accepted, 0
            out.accepted.append(accepted.id)
            tables = {}
            atlas = svc.mine(cell, current)
            io.log("atlas", {"version": current.id, "open": atlas.open_ids()}, version=current.id)
        else:
            streak += 1
            cooldown[card.cluster] = COOLDOWN
            if streak >= NO_ACCEPT_STREAK_STOP:
                out.stop = "no_accept_streak"
                break
    out.version, out.final = current.id, current
    io.log("note", {"stop": out.stop, **out.to_json()}, version=current.id)
    return out
