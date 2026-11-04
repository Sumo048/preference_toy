"""Code arms against small fakes (W1-C). The real harness modules are not needed: every arm takes
a ``Services`` bundle, and these tests pass fakes with the documented signatures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from toyrepair.arms import CODE_ARMS, get
from toyrepair.arms import b4, b5, d1, r0
from toyrepair.arms.base import ArmOutput, RunIO, Services, iterate, version_from_net


# ------------------------------------------------------------------------------------------ fakes
@dataclass
class FVersion:
    id: str
    parent: Optional[str]
    patches: Tuple[str, ...]
    net: nn.Module
    grid: str = "cond"
    meta: Dict = field(default_factory=dict)


@dataclass
class FPatch:
    id: str
    call: str
    args: Dict
    arrays: Dict
    parent_version: str
    version: str
    sha256: str = ""


@dataclass
class FCluster:
    id: int
    members: np.ndarray
    uids: np.ndarray


@dataclass
class FCard:
    cluster: int
    r_cur: float


@dataclass
class FAtlas:
    clusters: List[FCluster]
    cards: List[FCard]

    def open_ids(self):
        return [c.cluster for c in self.cards]


@dataclass
class FPrim:
    kind: str
    u: Tuple[float, float]
    alpha: float
    k: int
    h: int = 3
    name: str = "push(test)"


@dataclass
class FTable:
    k_commit: int
    failure_class: str
    best_primitive: Optional[FPrim]
    n_decodes: int = 10


@dataclass
class FGate:
    accepted: bool
    passed: Dict = field(default_factory=dict)
    target_fall: float = 0.0
    target_fall_ci: Tuple[float, float] = (0.0, 0.0)
    numbers: Dict = field(default_factory=dict)
    check_index: int = 0
    scope: str = "cluster"
    cpu_s: float = 0.0


class TinyNet(nn.Module):
    """v = W x + small c/t terms; deterministic."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(4, 2)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(2, 4, generator=g) * 0.3)
            self.lin.bias.copy_(torch.randn(2, generator=g) * 0.1)

    def forward(self, x, t, c):
        return self.lin(torch.cat([x, t.reshape(-1, 1), c.reshape(-1, 1)], -1))


def euler(net, xi, c, steps=20):
    x = xi.clone()
    for k in range(steps):
        t = torch.full((x.shape[0],), 1.0 - k / 20)
        x = x - (1 / 20) * net(x, t, c)
    return x


def fake_velocity(version, x, t, c, capture=()):
    v = version.net(x, t, c)
    return v, {name: torch.cat([x, c.reshape(-1, 1)], -1) for name in capture}


def fake_decode(version, xi, c, save_states=False, capture=(), start_step=0, batch=65536):
    x, acts = xi.clone(), {n: [] for n in capture}
    for k in range(20):
        t = torch.full((x.shape[0],), 1.0 - k / 20)
        v, a = fake_velocity(version, x, t, c, capture)
        for n in capture:
            acts[n].append(a[n])
        x = x - (1 / 20) * v
    return SimpleNamespace(action=x, acts={n: torch.stack(v) for n, v in acts.items()}, n_decodes=xi.shape[0])


class FakeCell:
    def __init__(self, n=12, setting="k4"):
        g = torch.Generator().manual_seed(1)
        self.key, self.setting, self.seed, self.index, self.npm, self.n_forget = f"cond_{setting}_s0_n4", setting, 0, 210, 4, 4
        self.neg = SimpleNamespace(uid=np.arange(100, 100 + n), c=torch.rand(n, generator=g),
                                   xi=torch.randn(n, 2, generator=g))
        self.ref = SimpleNamespace(net=TinyNet())

    def in_bad_basin(self, i, x):
        return (x[:, 0] > 0).numpy()          # "bad basin": right half-plane


def make_services(gate_fn, atlas_fn=None, branch_fn=None, log=None):
    log = log if log is not None else []

    def base_version(cell):
        return FVersion(id="v0", parent=None, patches=(), net=cell.ref.net)

    def mine(cell, version, branch_tables=None):
        log.append(("mine", version.id))
        return atlas_fn(version) if atlas_fn else FAtlas([], [])

    def branch(cell, version, cluster, **kw):
        log.append(("branch", cluster.id))
        return branch_fn(cluster) if branch_fn else FTable(6, "R", FPrim("push", (1.0, 0.0), 1.0, 6))

    def gate_check(cell, base, current, new, uids, check_index, scope="cluster"):
        log.append(("gate", new.id, check_index, scope))
        return gate_fn(new, check_index, scope)

    def targeted_train(cell, parent, new_id, spec, data):
        log.append(("train", new_id, spec.params, spec.steps, spec.anchor))
        net = TinyNet(seed=int(new_id[1:]))
        ver = FVersion(id=new_id, parent=parent.id, patches=tuple(parent.patches) + (f"p_{new_id}",), net=net)
        rec = SimpleNamespace(error="", ok=True, frac_weights_changed=1.0)
        return SimpleNamespace(version=ver, patch=FPatch(f"p_{new_id}", "targeted_train", {}, {}, parent.id, new_id),
                               record=rec)

    def toypref_rows(cell, uids, rule="hopB", support="bsup"):
        log.append(("toypref_rows", tuple(int(u) for u in uids), rule, support))
        return SimpleNamespace(source="toypref")

    def rescue_rows(cell, uids, targets, support="bsup"):
        log.append(("rescue_rows", tuple(int(u) for u in uids), tuple(targets.shape), support))
        return SimpleNamespace(source="rescue")

    def restore_and_decode(version, state, primitive=None):
        x = euler(version.net, state.x, state.c)
        return x - 10.0 * torch.tensor([1.0, 0.0]) if primitive is not None else x   # push left = rescue

    def save_patch(run_dir, patch):
        p = os.path.join(run_dir, "patches", f"{patch.id}.json")
        json.dump({"id": patch.id, "call": patch.call}, open(p, "w"))
        return p

    return Services(base_version=base_version, decode=fake_decode, velocity=fake_velocity, mine=mine,
                    branch=branch, restore_and_decode=restore_and_decode,
                    flow_state=lambda x, k, c: SimpleNamespace(x=x, k=k, c=c), gate_check=gate_check,
                    targeted_train=targeted_train, toypref_rows=toypref_rows, rescue_rows=rescue_rows,
                    train_spec=lambda **kw: SimpleNamespace(**kw), save_patch=save_patch, make_patch=FPatch,
                    make_version=FVersion), log


def three_cluster_atlas(closed_after: Dict[str, List[int]] = None):
    """Clusters 0, 1, 2 (rows 0-3, 4-7, 8-11); r_cur 0.3, 0.2, 0.1. ``closed_after[vid]`` lists
    clusters closed once version vid is current."""
    closed_after = closed_after or {}

    def atlas(version):
        clusters = [FCluster(i, np.arange(4 * i, 4 * i + 4), np.arange(100 + 4 * i, 104 + 4 * i)) for i in range(3)]
        closed = set()
        for vid, cl in closed_after.items():
            if version.id >= vid:      # v1 < v2 < ... lexicographically for single digits
                closed |= set(cl)
        cards = [FCard(i, r) for i, r in ((0, 0.3), (1, 0.2), (2, 0.1)) if i not in closed]
        return FAtlas(clusters, cards)
    return atlas


def run_ctx(tmp_path, name="r"):
    return SimpleNamespace(id=name, dir=str(tmp_path / name))


def read_ledger(run):
    return [json.loads(l) for l in open(os.path.join(run.dir, "ledger.jsonl"))]


# ------------------------------------------------------------------------------------------ registry
def test_registry_covers_all_code_arms():
    assert set(CODE_ARMS) == {"B0", "B1a", "B1b", "B4", "B5", "R0", "D1"}
    for name in CODE_ARMS:
        assert callable(get(name))


def test_d1_variant_list_is_fixed_and_has_three():
    assert len(d1.VARIANTS) == 3
    assert [v.name for v in d1.VARIANTS] == ["all_5000", "all_1500", "late_5000"]
    assert all(v.anchor == "dist" and v.support == "bsup" for v in d1.VARIANTS)


def test_b5_configs_cover_all_settings():
    assert set(b5.CONFIGS) == {"k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"}
    from toypref.cond.experiment import ARMS
    for cfgs in b5.CONFIGS.values():
        assert len(cfgs) == 3 and all(c in ARMS for c in cfgs)


def test_best_configs_match_method_map():
    from toyrepair.arms.b1b import best_configs
    assert best_configs("k4") == ["R_hopB_bsup_mine_neg"]
    assert best_configs("alt") == ["R_hop_bsup_neg"]
    for s in ("k4", "alt", "pairs", "b3p1", "contig", "k1", "k2"):
        assert best_configs(s, 3) == b5.CONFIGS[s], s


# ------------------------------------------------------------------------------------------ loop
def test_b4_accepts_every_cluster_until_none_open(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(True),
                             three_cluster_atlas({"v1": [0], "v2": [1], "v3": [2]}))
    run = run_ctx(tmp_path)
    out = b4.run(FakeCell(), run, svc)
    assert out.stop == "no_open_cluster"
    assert out.accepted == ["v1", "v2", "v3"] and out.version == "v3" and out.gate_checks == 3
    rows = [e for e in log if e[0] == "toypref_rows"]
    assert [r[1] for r in rows] == [tuple(range(100, 104)), tuple(range(104, 108)), tuple(range(108, 112))]
    assert all(r[2:] == ("hopB", "bsup") for r in rows)
    trains = [e for e in log if e[0] == "train"]
    assert all(t[2:] == (("all",), 5000, "dist") for t in trains)
    assert all(e[3] == "cluster" for e in log if e[0] == "gate")
    assert not any(e[0] == "branch" for e in log)          # B4 needs no branch tool
    kinds = [r["kind"] for r in read_ledger(run)]
    assert kinds[0] == "run" and kinds.count("gate") == 3 and kinds[-1] == "note"


def test_no_accept_streak_and_cooldown(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(False), three_cluster_atlas())
    out = b4.run(FakeCell(), run_ctx(tmp_path), svc)
    assert out.stop == "no_accept_streak" and out.accepted == [] and out.version == "v0"
    # cool-down: three rejected attempts go to three different clusters
    assert [r["cluster"] for r in out.meta["rounds"]] == [0, 1, 2]


def test_gate_cap(tmp_path):
    # one open cluster that never closes, accepts always: cap = 3 * (1 + 2) = 9
    def atlas(version):
        return FAtlas([FCluster(0, np.arange(4), np.arange(100, 104))], [FCard(0, 0.5)])
    svc, log = make_services(lambda new, i, s: FGate(True), atlas)
    out = b4.run(FakeCell(), run_ctx(tmp_path), svc)
    assert out.stop == "gate_cap" and out.gate_checks == 9 and len(out.accepted) == 9


def test_pilot_single_round(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(False), three_cluster_atlas())
    out = d1.run(FakeCell(), run_ctx(tmp_path), svc, max_rounds=1)
    assert out.stop == "max_rounds" and len(out.meta["rounds"]) == 1
    assert out.gate_checks == 3                                # the 3 D1 variants, all rejected
    assert [t["variant"] for t in out.meta["rounds"][0]["tried"]] == ["all_5000", "all_1500", "late_5000"]


def test_d1_stops_at_first_accept_and_uses_rescue_targets(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(new.meta.get("variant") == "all_1500"),
                             three_cluster_atlas({"v2": [0, 1, 2]}))
    out = d1.run(FakeCell(), run_ctx(tmp_path), svc)
    assert out.accepted == ["v2"] and out.stop == "no_open_cluster" and out.gate_checks == 2
    trains = [e for e in log if e[0] == "train"]
    assert [t[2:4] for t in trains] == [(("all",), 5000), (("all",), 1500)]
    rr = [e for e in log if e[0] == "rescue_rows"]
    assert rr and rr[0][3] == "bsup" and rr[0][2][1] == 2
    assert set(rr[0][1]) <= set(range(100, 104))


def test_class_u_clusters_are_skipped(tmp_path):
    def branch(cluster):
        if cluster.id == 0:
            return FTable(8, "U", None)
        return FTable(6, "R", FPrim("push", (1.0, 0.0), 1.0, 6))
    svc, log = make_services(lambda new, i, s: FGate(True), three_cluster_atlas({"v1": [1, 2]}), branch)
    out = d1.run(FakeCell(), run_ctx(tmp_path), svc, max_rounds=1)
    assert out.meta["rounds"][0]["cluster"] == 1


def test_all_class_u_stops(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(True), three_cluster_atlas(),
                             lambda cl: FTable(8, "U", None))
    out = r0.run(FakeCell(), run_ctx(tmp_path), svc)
    assert out.stop == "no_class_R_cluster" and out.gate_checks == 0


# ------------------------------------------------------------------------------------------ B5
def test_b5_keeps_best_accepted_global_scope(tmp_path):
    D = {"v1": 0.30, "v2": 0.25, "v3": 0.20}
    acc = {"v1": True, "v2": True, "v3": False}
    svc, log = make_services(lambda new, i, s: FGate(acc[new.id], numbers={"D_new": D[new.id]}, scope=s),
                             three_cluster_atlas())
    trained = []

    def train_fn(cell, cfg):
        trained.append(cfg)
        return SimpleNamespace(net=TinyNet(seed=len(trained))), {}
    out = b5.run(FakeCell(setting="alt"), run_ctx(tmp_path), svc, train_fn=train_fn)
    assert trained == b5.CONFIGS["alt"]
    assert out.version == "v2" and out.meta["toypref_config"] == "KTO_b10_neg_AMmap" and out.gate_checks == 3
    assert all(e[3] == "global" for e in log if e[0] == "gate")


def test_version_from_net_stores_only_changed_tensors(tmp_path):
    svc, _ = make_services(lambda *a: FGate(True))
    io = RunIO(run_ctx(tmp_path), svc)
    base = FVersion("v0", None, (), TinyNet(0))
    net = TinyNet(0)
    with torch.no_grad():
        net.lin.bias.add_(1.0)
    ver = version_from_net(io, base, net, "toypref_train", {"toypref_config": "X"})
    assert ver.id == "v1" and ver.patches == ("p0001",)
    assert os.path.exists(os.path.join(io.dir, "patches", "p0001.json"))
    assert ver.meta["frac_weights_changed"] == pytest.approx(2 / 10)


# ------------------------------------------------------------------------------------------ R0 controller
def _ctrl(inner, kind, thr, k=6, h=3, alpha=0.5):
    w = np.array([0.0, 0.0, 1.0], dtype=np.float32)        # feature = (x0, x1, c): fires where c > thr
    return r0.ControlledNet(inner, "L", w, 0.0, thr, kind, (1.0, 0.0), alpha, k, h, fake_velocity, FVersion)


def test_controller_off_is_bit_identical():
    inner = TinyNet(3)
    xi, c = torch.randn(256, 2), torch.rand(256)
    for kind in ("push", "hop"):
        net = _ctrl(inner, kind, thr=10.0)
        assert torch.equal(euler(net, xi, c), euler(inner, xi, c))


def test_controller_push_and_hop_semantics():
    inner = TinyNet(3)
    xi, c = torch.randn(512, 2), torch.rand(512)
    fire = c > 0.5
    # push: displacement alpha*u/h added after each Euler update of steps k..k+h-1
    net = _ctrl(inner, "push", thr=0.5, k=6, h=3, alpha=0.6)
    x = xi.clone()
    for k in range(20):
        t = torch.full((x.shape[0],), 1.0 - k / 20)
        x = x - (1 / 20) * inner(x, t, c)
        if 6 <= k < 9:
            x = x + (0.6 / 3) * torch.tensor([1.0, 0.0]) * fire.reshape(-1, 1)
    assert torch.allclose(euler(net, xi, c), x, atol=1e-5)
    assert not torch.allclose(euler(net, xi, c), euler(inner, xi, c))
    # hop: xi <- xi + tau*u where the detector fires at step 0
    net = _ctrl(inner, "hop", thr=0.5, k=0, h=1, alpha=0.4)
    ref = euler(inner, xi + 0.4 * torch.tensor([1.0, 0.0]) * fire.reshape(-1, 1), c)
    assert torch.allclose(euler(net, xi, c), ref, atol=1e-5)
    unfired = ~fire
    assert torch.equal(euler(net, xi, c)[unfired], euler(inner, xi, c)[unfired])


def test_r0_fits_detector_and_submits_thresholds(tmp_path):
    svc, log = make_services(lambda new, i, s: FGate(False), three_cluster_atlas())
    out = r0.run(FakeCell(), run_ctx(tmp_path), svc, max_rounds=1)
    tried = out.meta["rounds"][0]["tried"]
    assert [t["variant"] for t in tried] == ["p0.5", "p0.7", "p0.3"]
    assert os.path.exists(os.path.join(tmp_path, "r", "r0", "v1.npz"))
