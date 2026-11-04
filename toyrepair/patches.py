"""Patch files and rebuild (docs/PLAN.md 4.7, 6.7). Owner: W1-B.

Each edit is stored as ``runs/<run id>/patches/<patch id>.json`` (call, args, parent version,
hashes) plus its arrays in ``runs/<run id>/patches/<patch id>.npz``. Any version is rebuilt
from pi_ref and its patch stack, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def save_patch(run_dir: str, patch: Patch) -> str:
    """Write the patch JSON + NPZ atomically. Returns the JSON path."""
    raise NotImplementedError


def load_patch(run_dir: str, patch_id: str) -> Patch:
    raise NotImplementedError


def apply_patch(version, patch: Patch):
    """``policy.Version`` -> a new Version = ``version`` with ``patch`` applied (deep copy; input unchanged)."""
    raise NotImplementedError


def rebuild(cell, run_dir: str, patch_ids: List[str], version_id: str):
    """``policy.base_version(cell)`` + patches in order -> ``policy.Version`` with id
    ``version_id``; checks each patch's sha256. Equals the live version bit for bit."""
    raise NotImplementedError


def state_sha256(version) -> str:
    """sha256 over the version's parameters and graft arrays (deterministic order)."""
    raise NotImplementedError
