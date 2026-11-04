"""W2b Phase 4: behaviour tracks, contrast and unit_card (PLAN 6.5)."""
import math
import os

import numpy as np
import pytest
import torch

import toyrepair  # noqa: F401  (sets TOYPREF_RUNS)
from toyrepair import act_tools, edits, policy, tracks
from toyrepair.cell import load_cell
from toyrepair.dossier import CHARS_PER_TOKEN

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    return load_cell("cond", "k4", 0, 4)


@pytest.fixture(scope="module")
def tr(cell, tmp_path_factory):
    return tracks.get(cell, cache_dir=str(tmp_path_factory.mktemp("tracks")))


def _tok(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def test_tracks_deterministic_and_marked(cell, tr, tmp_path):
    tr2 = tracks.build(cell)
    assert tr2.n_tracks == tr.n_tracks
    assert np.array_equal(tr2.track, tr.track)
    # every logged action lies in some track at the marking radius
    t = tr.assign(cell.neg.a, cell.neg.c, eps_mult=tracks.MARK_EPS)
    assert (t >= 0).mean() > 0.9
    marked = tr.marked(cell, cell.neg.uid)
    assert 1 <= len(marked) < tr.n_tracks


def test_contrast_output_and_detector(cell, tr, tmp_path, monkeypatch):
    monkeypatch.setattr(tracks, "get", lambda c, cache_dir=None: tr)
    v0 = policy.base_version(cell)
    uids = list(map(int, cell.neg.uid[:4]))
    text, dec = act_tools.contrast(cell, v0, uids, 0, "blocks.1.act", 6, str(tmp_path))
    assert dec == act_tools.N_SAMPLES
    assert _tok(text) <= act_tools.TOKEN_CAP, _tok(text)
    assert "probe held-out AUC by step" in text
    f = tmp_path / "contrast_c0_blocks.1.act_k6.npy"
    w = np.load(f)
    assert w.shape == (128,) and w.dtype == np.float32
    b = float(text.split("b=")[1].split(")")[0])
    # the saved probe is a legal graft detector and fires on the same states as reported
    det = edits.Detector("blocks.1.act", tuple(float(x) for x in w), b, 0.0, (0, 7))
    g = edits.graft_gated_path(v0, "vt", "out", det, 2, np.zeros((2, 128), np.float32), np.zeros((2, 2), np.float32),
                               src="out_act", act="linear")
    assert g.record.ok, g.record.error


def test_contrast_rejects_bad_layer(cell, tmp_path):
    v0 = policy.base_version(cell)
    with pytest.raises(act_tools.ToolInputError):
        act_tools.contrast(cell, v0, [0], 0, "nope", 3, str(tmp_path))
    with pytest.raises(act_tools.ToolInputError):
        act_tools.contrast(cell, v0, [0], 0, "blocks.1.act", 20, str(tmp_path))


def test_unit_card_output(cell, tr, monkeypatch):
    monkeypatch.setattr(tracks, "get", lambda c, cache_dir=None: tr)
    v0 = policy.base_version(cell)
    text, dec = act_tools.unit_card(cell, v0, list(map(int, cell.neg.uid[:4])), 0, "blocks.2.act", 5)
    assert dec == act_tools.N_SAMPLES
    assert _tok(text) <= act_tools.TOKEN_CAP, _tok(text)
    assert len(text.splitlines()) == 2 + 10 + 1
    with pytest.raises(act_tools.ToolInputError):
        act_tools.unit_card(cell, v0, [0], 0, "blocks.2.act", 128)
