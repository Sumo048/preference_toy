"""robo2d unit tests: seeding / nesting, label construction, env collision / jerk detection, and
that no method reads a privileged label. Run: python -m pytest tests/test_robo.py -q"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from toypref.robo import data as DT, env as E, evaluate as EV, methods as MT, seeds as S, targets as TG
from toypref.robo.policy import ChunkFlow, frozen


def tiny_policy(seed=0):
    torch.manual_seed(seed)
    p = ChunkFlow(hidden=32, n_layers=1, t_dim=16, o_dim=16)
    with torch.no_grad():
        p.net.out.weight.normal_(0, 0.3)
    return frozen(p)


class GoalPolicy:
    """Deterministic scripted 'policy': heads straight to the goal; noise dim 0 > 1 adds zig-zag.
    Straight lines through obstacles collide, so both failure types occur."""

    def sample(self, xi, o):
        o = o.numpy()
        d = o[:, 2:4]
        u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        v = np.repeat(u[:, None, :] * 1.2, E.H, 1)
        nrm = np.stack([-u[:, 1], u[:, 0]], -1)
        zz = (xi[:, 0].numpy() > 1.0)[:, None, None] * 0.8 * nrm[:, None, :] * ((-1.0) ** np.arange(E.H))[None, :, None]
        return torch.from_numpy((v + zz).reshape(len(o), -1).astype(np.float32))


class BimodalPolicy:
    """Scripted sampler with two well-separated behaviours split by the sign of noise dim 0."""

    def sample(self, xi, o):
        return torch.sign(xi[:, :1]) * 2.0 * torch.ones(1, E.ACT_DIM) + 0.05 * xi


# ------------------------------------------------------------------ env
def one_setup(centre=(0.0, 0.0), r=0.2, start=(-0.8, 0.0), goal=(0.8, 0.0)):
    C = np.full((1, 3, 2), E.FAR); R = np.zeros((1, 3))
    C[0, 0], R[0, 0] = centre, r
    return E.Setups(np.array([0]), np.array([start], float), np.array([goal], float), C, R, np.array([1]))


def test_seg_disc_hit():
    C = np.array([[[0.0, 0.0], [E.FAR, E.FAR], [E.FAR, E.FAR]]]); R = np.array([[0.2, 0.0, 0.0]])
    assert E.seg_disc_hit(np.array([[-0.5, 0.0]]), np.array([[0.5, 0.0]]), C, R)[0]          # crosses
    assert not E.seg_disc_hit(np.array([[-0.5, 0.3]]), np.array([[0.5, 0.3]]), C, R)[0]      # passes above
    assert E.seg_disc_hit(np.array([[-0.5, 0.1]]), np.array([[-0.1, 0.1]]), C, R)[0]         # ends inside
    assert not E.seg_disc_hit(np.array([[-0.5, 0.0]]), np.array([[-0.3, 0.0]]), C, R)[0]     # stops short
    # padded obstacles (radius 0) never collide
    assert not E.seg_disc_hit(np.array([[E.FAR - 1, E.FAR]]), np.array([[E.FAR + 1, E.FAR]]), C, R)[0]


def test_execute_collision_success_and_observation():
    st = one_setup()
    ep = E.Episodes(st, [0, 0])
    act = np.zeros((2, E.ACT_DIM), np.float32)
    act[0] = np.tile([1.2, 0.0], E.H)                 # straight into the disc
    act[1] = np.tile([1.2, 1.2], E.H)                 # diagonal away
    for c in range(E.N_CHUNKS):
        ep.execute(act, c)
    assert ep.collided[0] and ep.col_chunk[0] >= 0 and not ep.success[0]
    assert not ep.collided[1]
    st2 = one_setup(centre=(0.0, 0.6))
    ep2 = E.Episodes(st2, [0])
    for c in range(E.N_CHUNKS):
        ep2.execute(np.tile([1.4, 0.0], E.H)[None].astype(np.float32), c)
    assert ep2.success[0] and not ep2.collided[0]
    # observation: absent obstacles zero-padded, nearest first
    o = E.observe(np.array([[0.0, -0.5]]), st2.goal, st2.centre, st2.radius)
    assert o.shape == (1, E.OBS_DIM)
    assert np.allclose(o[0, 4:7], [0.0, 1.1, 0.2]) and np.allclose(o[0, 7:], 0.0)


def test_jerk_detection():
    # smooth arc (curvature 2) vs the same arc with zig-zag
    th = np.linspace(0, 1.2, E.T_MAX + 1)
    arc = np.stack([0.5 * np.sin(th), 0.5 * (1 - np.cos(th))], -1)[None]
    assert E.jerk_score(arc)[0] < E.JERK_THR
    zz = arc.copy()
    zz[0, 10:19, 1] += 0.03 * (-1.0) ** np.arange(9)
    assert E.jerk_score(zz)[0] > E.JERK_THR
    # NaN tails (terminated episodes) are ignored
    tail = arc.copy(); tail[0, 20:] = np.nan
    assert np.isfinite(E.jerk_score(tail)[0]) and E.jerk_score(tail)[0] < E.JERK_THR
    # chunk-level detector agrees on single chunks
    v = np.tile([1.2, 0.0], E.H).reshape(E.H, 2)
    zzc = v + 0.7 * np.array([0, 1.0]) * ((-1.0) ** np.arange(E.H))[:, None]
    assert E.chunk_jerk(v.reshape(1, -1))[0] < E.JERK_THR < E.chunk_jerk(zzc.reshape(1, -1))[0]


def test_label_is_or_of_collision_and_jerk():
    st = one_setup(start=(-0.5, 0.0))
    ep = E.Episodes(st, [0, 0, 0])
    a = np.zeros((3, E.ACT_DIM), np.float32)
    a[0] = np.tile([1.2, 0.0], E.H)                                         # collide
    base = np.tile([1.0, 1.0], E.H).reshape(E.H, 2)
    a[1] = (base + 0.8 * np.array([1.0, -1.0]) / np.sqrt(2) * ((-1.0) ** np.arange(E.H))[:, None]).reshape(-1)   # jerk
    a[2] = np.tile([1.0, 1.0], E.H)                                         # clean
    ep.execute(a, 0)
    neg, col, jer, _ = E.label(ep)
    assert list(col) == [True, False, False] and list(jer) == [False, True, False] and list(neg) == [True, True, False]


# ------------------------------------------------------------------ seeding / nesting
def test_streams_deterministic_and_distinct():
    assert torch.equal(torch.randn(3, generator=S.gen(1, S.P_EVAL)), torch.randn(3, generator=S.gen(1, S.P_EVAL)))
    assert not torch.equal(torch.randn(3, generator=S.gen(1, S.P_EVAL)), torch.randn(3, generator=S.gen(2, S.P_EVAL)))
    a = torch.randn(4, generator=S.row_gen(0, 123, S.ROW_CAND))
    assert torch.equal(a, torch.randn(4, generator=S.row_gen(0, 123, S.ROW_CAND)))
    for other in (S.row_gen(0, 124, S.ROW_CAND), S.row_gen(0, 123, S.ROW_SUPPORT), S.row_gen(1, 123, S.ROW_CAND)):
        assert not torch.equal(a, torch.randn(4, generator=other))


@pytest.fixture(scope="module")
def setups():
    return DT.experiment_setups(0)


def test_setups_and_noneg(setups):
    tr, he = setups
    tr2, he2 = DT.experiment_setups(0)
    assert np.array_equal(tr.centre, tr2.centre) and np.array_equal(he.goal, he2.goal)
    assert len(tr) == DT.N_TRAIN_SETUPS and len(he) == DT.N_HELDOUT_SETUPS
    assert not set(tr.sid) & set(he.sid)
    nn = DT.noneg_setups(0)
    assert len(nn) == round(0.2 * DT.N_TRAIN_SETUPS) and np.array_equal(nn, DT.noneg_setups(0))
    k = DT.n_keep(0)
    assert (k[nn] == 0).all() and set(np.unique(np.delete(k, nn))) <= {1, 2, 3}


def test_logging_labels_and_eval_crn(setups):
    tr, he = setups
    pol = GoalPolicy()
    log = DT.log_negatives(pol, 0, tr)
    keep = DT.n_keep(0)
    R = log.rows
    # no rows from the no-negative setups; at most keep[s] episodes per setup
    assert not np.isin(R.setup, DT.noneg_setups(0)).any()
    for s in np.unique(R.setup):
        assert len(np.unique(R.episode[R.setup == s])) <= keep[s]
    # every executed chunk of every kept episode is a row; replaying the logged noise reproduces a negative
    rep = DT.rollout(pol, tr, log.episodes["setup"], torch.from_numpy(log.episodes["noise"]))
    assert rep.negative.all()
    assert len(R) == int(rep.ran.sum())
    for e in range(len(log.episodes["setup"])):
        m = R.episode == e
        assert np.array_equal(np.sort(R.chunk[m]), np.flatnonzero(rep.ran[e]))
        assert torch.allclose(R.obs[m], torch.from_numpy(rep.obs[e][rep.ran[e]]))
    # rows identical across two runs (seeded)
    log2 = DT.log_negatives(pol, 0, tr)
    assert np.array_equal(log2.rows.uid, R.uid) and torch.equal(log2.rows.xi, R.xi)
    # CRN: two eval sets share noises, and the frozen base scores identically
    ev1 = EV.EvalSet(pol, 0, tr, he, log, M=4)
    ev2 = EV.EvalSet(pol, 0, tr, he, log, M=4)
    assert torch.equal(ev1.noise, ev2.noise)
    assert ev1.compute(pol)["neg_bad"] == ev2.compute(pol)["neg_bad"]


def _rows(n_setups=3):
    tr, _ = DT.experiment_setups(0)
    log = DT.log_negatives(GoalPolicy(), 0, tr)
    keep = np.unique(log.rows.setup)[:n_setups]
    return log.rows.take(np.isin(log.rows.setup, keep)), log


def test_cache_nesting():
    """A row's targets / support / anchor depend only on (seed, row uid): the cache built on a
    subset equals the subset of the cache built on a superset."""
    ref = BimodalPolicy()
    rows, _ = _rows(3)
    rows = rows.take(np.arange(min(len(rows), 12)))
    rows.a = ref.sample(rows.xi, rows.obs)
    sub = rows.take(np.arange(0, len(rows), 2))
    full = TG.build_hop_cache(ref, 0, rows)
    part = TG.build_hop_cache(ref, 0, sub)
    idx = np.arange(0, len(rows), 2)
    assert torch.allclose(full.target[idx], part.target, atol=1e-6)
    for n, i in enumerate(idx):
        assert torch.allclose(full.support[i], part.support[n], atol=1e-6)
    assert np.array_equal(full.n_clusters[idx], part.n_clusters)
    a = TG.fresh_decodes(ref, 0, rows, TG.N_SDPO_POS, S.ROW_SDPO)[idx]
    b = TG.fresh_decodes(ref, 0, sub, TG.N_SDPO_POS, S.ROW_SDPO)
    assert torch.allclose(a, b, atol=1e-6)


def test_hop_leaves_the_bad_basin():
    ref = BimodalPolicy()
    rows, _ = _rows(3)
    rows = rows.take(np.arange(min(len(rows), 8)))
    rows.a = ref.sample(rows.xi, rows.obs)
    c = TG.build_hop_cache(ref, 0, rows)
    assert (c.n_clusters == 2).all()
    assert all(d["kind"] == "hop" for d in c.info)
    # target in the other behaviour; support noises still decode into the negative's behaviour
    assert (torch.sign(c.target[:, 0]) != torch.sign(rows.a[:, 0])).all()
    for i in range(len(rows)):
        assert (torch.sign(c.support[i][:, 0]) == torch.sign(rows.xi[i, 0])).all()
    lo, lz, la, cnt = c.local
    assert (cnt > 0).all()
    owner = np.repeat(np.arange(len(rows)), cnt)
    assert (torch.sign(la[:, 0]) != torch.sign(rows.a[owner, 0])).all()


# ------------------------------------------------------------------ no privileged labels
def test_methods_never_read_privileged_labels(monkeypatch):
    ref = tiny_policy()
    rows, _ = _rows(2)
    rows = rows.take(np.arange(min(len(rows), 8)))

    def boom(*a, **k):
        raise AssertionError("a method touched the env / labeller")
    for name in ("label", "jerk_score", "chunk_jerk", "homotopy", "seg_disc_hit", "Episodes", "observe", "path_length"):
        monkeypatch.setattr(E, name, boom)
    import toypref.robo.diagnostics as DG
    for name in ("oracle_chunk_bad", "separability"):
        monkeypatch.setattr(DG, name, boom)
    ctx = MT.Ctx(ref=ref, seed=0, rows=rows, steps=3, batch=16)
    for n, arm in MT.ARMS.items():
        pol, info = MT.train_arm(arm, ctx)
        assert pol is not None
    # the method API has no slot for privileged facts
    import inspect
    assert "diag" not in inspect.signature(MT.Ctx).parameters
    assert set(DT.NegRows.__dataclass_fields__) == {"obs", "xi", "a", "uid", "setup", "episode", "chunk"}


def test_arms_share_draws():
    """Every arm consumes the same minibatch stream key and the same per-row draws."""
    ref = tiny_policy()
    rows, _ = _rows(2)
    rows = rows.take(np.arange(min(len(rows), 6)))
    c1 = MT.Ctx(ref=ref, seed=0, rows=rows, steps=2, batch=8)
    c2 = MT.Ctx(ref=ref, seed=0, rows=rows, steps=2, batch=8)
    p1, _ = MT.train_arm(MT.ARMS["SDPO_pin_b1"], c1)
    p2, _ = MT.train_arm(MT.ARMS["SDPO_pin_b1"], c2)
    for a, b in zip(p1.parameters(), p2.parameters()):
        assert torch.equal(a, b)
