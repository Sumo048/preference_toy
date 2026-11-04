"""Patch files and rebuild (docs/PLAN.md 4.7, 6.7). Owner: W1-B.

Each edit is stored as ``runs/<run id>/patches/<patch id>.json`` (call, args, parent version,
hashes) plus its arrays in ``runs/<run id>/patches/<patch id>.npz``. Any version is rebuilt
from pi_ref and its patch stack, bit for bit.

``apply_patch`` is the single implementation of every edit: the edit calls in
``toyrepair.edits`` build their new version with it, so the live version and the rebuild run
the same float32 operations on the same inputs (tested bit for bit).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PATCH_DIR = "patches"


@dataclass
class Patch:
    """id: patch id (e.g. ``p0003``); call: edit call name; args: JSON-safe args; arrays: name ->
    ndarray (saved to .npz); parent_version / version: ids; sha256 of the resulting state dict."""
    id: str
    call: str
    args: Dict
    arrays: Dict[str, np.ndarray]
    parent_version: str
    version: str
    sha256: str = ""


def _paths(run_dir: str, patch_id: str) -> Tuple[str, str]:
    d = os.path.join(run_dir, PATCH_DIR)
    return os.path.join(d, f"{patch_id}.json"), os.path.join(d, f"{patch_id}.npz")


def save_patch(run_dir: str, patch: Patch) -> str:
    """Write the patch JSON + NPZ atomically. Returns the JSON path."""
    jp, zp = _paths(run_dir, patch.id)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    tmpz = zp + ".tmp.npz"
    np.savez(tmpz, **{k: np.asarray(v) for k, v in patch.arrays.items()})
    with open(tmpz, "rb") as f:
        zsha = hashlib.sha256(f.read()).hexdigest()
    os.replace(tmpz, zp)
    rec = {"id": patch.id, "call": patch.call, "args": patch.args, "parent_version": patch.parent_version,
           "version": patch.version, "sha256": patch.sha256, "npz_sha256": zsha,
           "arrays": sorted(patch.arrays)}
    tmpj = jp + ".tmp"
    with open(tmpj, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(tmpj, jp)
    return jp


def load_patch(run_dir: str, patch_id: str) -> Patch:
    jp, zp = _paths(run_dir, patch_id)
    with open(jp) as f:
        rec = json.load(f)
    with open(zp, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != rec.get("npz_sha256"):
            raise ValueError(f"patch {patch_id}: npz checksum mismatch")
    with np.load(zp, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return Patch(id=rec["id"], call=rec["call"], args=rec["args"], arrays=arrays,
                 parent_version=rec["parent_version"], version=rec["version"], sha256=rec.get("sha256", ""))


def apply_patch(version, patch: Patch):
    """``policy.Version`` -> a new Version = ``version`` with ``patch`` applied (deep copy; input unchanged)."""
    from . import edits
    net = edits.apply_call(copy.deepcopy(version.net), patch.call, patch.args, patch.arrays, gid=patch.id)
    return edits.make_version(version, patch.version, net, patch.id, meta={"call": patch.call})


def rebuild(cell, run_dir: str, patch_ids: List[str], version_id: str):
    """``policy.base_version(cell)`` + patches in order -> ``policy.Version`` with id
    ``version_id``; checks each patch's sha256. Equals the live version bit for bit."""
    from . import edits
    v = edits.base_version(cell)
    for pid in patch_ids:
        p = load_patch(run_dir, pid)
        v = apply_patch(v, p)          # parent ids are informative; the sha check is the guard
        if p.sha256 and state_sha256(v) != p.sha256:
            raise ValueError(f"rebuild: patch {pid} gives sha {state_sha256(v)[:12]}, recorded {p.sha256[:12]}")
    v.id = version_id
    return v


def state_sha256(version) -> str:
    """sha256 over the version's parameters and graft arrays (deterministic order)."""
    from . import edits
    net = version.net if hasattr(version, "net") else version
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().contiguous().cpu()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    for g in edits.grafts_of(net):
        h.update(json.dumps(g.spec(), sort_keys=True).encode())
    return h.hexdigest()
