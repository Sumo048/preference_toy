"""W1-B: sandbox export, isolation as user ``sandbox``, audit hook, output cap (PLAN 6.6)."""
import os
import pwd
import shutil

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair import edits as E
from toyrepair import sandbox as S
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
try:
    pwd.getpwnam(S.SANDBOX_USER)
    have_user = os.getuid() == 0
except KeyError:
    have_user = False
need_user = pytest.mark.skipif(not have_user, reason="needs root and the sandbox user (scripts/cloud_setup.sh)")
TASK = os.path.join(toyrepair.REPO, "src", "toypref", "cond", "task.py")


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0")
    return load_cell("cond", "k4", 0, 1)


@pytest.fixture(scope="module")
def grafted(cell):
    v0 = E.base_version(cell)
    rng = np.random.default_rng(0)
    d = E.Detector("blocks.0.act", tuple(rng.standard_normal(128).astype(np.float32)), 0.0, 0.5)
    r = E.graft_gated_path(v0, "v1", "blocks.1.fc2", d, 8, rng.standard_normal((8, 128)) * 0.3,
                           rng.standard_normal((128, 8)) * 0.3)
    return v0, r.version


@pytest.fixture(scope="module")
def scratch(cell, grafted):
    d = os.path.join(RUNS, "repair", "_pytest_sandbox", "scratch")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    S.export(cell, grafted[1], d)
    yield d
    shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def run(code, scratch, **kw):
    return S.run(code, scratch, **kw)


def test_export_contents(cell, scratch):
    ex = os.path.join(scratch, "export")
    assert sorted(os.listdir(ex)) == ["README.txt", "basins.npz", "model.py", "negatives.npz", "weights.pt"]
    src = open(os.path.join(ex, "model.py")).read()
    assert "toypref" not in src and "toyrepair" not in src and "CondMixture" not in src and "assign" not in src
    neg = np.load(os.path.join(ex, "negatives.npz"))
    assert np.array_equal(neg["uid"], np.sort(cell.neg.uid))
    assert sorted(neg.files) == ["a", "c", "uid", "xi"]


@need_user
def test_exported_model_bit_identical_to_harness(cell, grafted, scratch):
    xi = torch.randn(512, 2, generator=torch.Generator().manual_seed(4))
    c = torch.rand(512, generator=torch.Generator().manual_seed(5))
    np.save(os.path.join(scratch, "xi.npy"), xi.numpy())
    np.save(os.path.join(scratch, "c.npy"), c.numpy())
    code = (
        "import numpy as np, torch, model\n"
        "net = model.load()\n"
        "a = model.decode(net, torch.from_numpy(np.load('xi.npy')), torch.from_numpy(np.load('c.npy')))\n"
        "np.save('a_sbx.npy', a.numpy())\n"
        "neg = model.negatives()\n"
        "print('in_basin', bool(model.in_basin(0, torch.from_numpy(neg['a'][:1]))[0]))\n")
    r = run(code, scratch)
    assert r.returncode == 0, r.stderr
    assert "in_basin True" in r.stdout and r.decodes == 512
    a = np.load(os.path.join(scratch, "a_sbx.npy"))
    assert np.array_equal(a, E.decode_actions(grafted[1].net, xi, c).numpy())
    assert not np.array_equal(a, E.decode_actions(grafted[0].net, xi, c).numpy())


@need_user
def test_runs_as_sandbox_user(scratch):
    r = run("import os; print(os.getuid(), os.getgroups())", scratch)
    assert r.stdout.split()[0] == str(pwd.getpwnam("sandbox").pw_uid)


@need_user
@pytest.mark.parametrize("code", [
    f"print(open({TASK!r}).read()[:40])",                                   # hook + permissions
    "p = '/home/user/preference_' + 'toy/src/toy' + 'pref/cond/task.py'\nprint(open(p).read()[:40])",
    "import os; print(os.listdir(" + repr(toyrepair.REPO) + "))",
])
def test_reading_task_py_fails(code, scratch):
    r = run(code, scratch)
    assert r.returncode != 0 and ("PermissionError" in r.stderr or "blocked" in r.stderr)
    assert "CondMixture" not in r.stdout and "import" not in r.stdout


@need_user
def test_permissions_alone_block_the_repo(scratch):
    """Second layer: a repo path the audit hook does not match is refused by the OS (mode 700)."""
    for name in ("README.md", "pyproject.toml"):
        path = os.path.join(toyrepair.REPO, name)
        assert "toypref" not in path and "toyrepair" not in path
        r = run(f"print(open({path!r}).read()[:40])", scratch)
        assert r.returncode != 0 and "Errno 13" in r.stderr, r.stderr


@need_user
@pytest.mark.parametrize("code", [
    "import toypref",
    "import toypref.cond.task",
    "import toyrepair",
    "import toyrepair.oracle",
    "from toyrepair import oracle",
    "import importlib; importlib.import_module('toypref.cond.task')",
    "import importlib; importlib.import_module('toyrepair.oracle')",
    "import sys; sys.path.insert(0, " + repr(os.path.join(toyrepair.REPO, "src")) + "); import toypref.cond.task",
])
def test_imports_fail(code, scratch):
    r = run(code, scratch)
    assert r.returncode != 0 and ("ImportError" in r.stderr or "ModuleNotFoundError" in r.stderr), r.stderr


@need_user
@pytest.mark.parametrize("code", [
    "import subprocess; subprocess.run(['id'])",
    "import os; os.system('id')",
    "import os; os.execv('/bin/sh', ['sh'])",
    "import os; os.fork()",
    "import os; os.posix_spawn('/bin/id', ['id'], {})",
    "import socket; socket.socket().connect(('1.1.1.1', 80))",
    "import socket; socket.create_connection(('example.com', 80))",
    "import urllib.request; urllib.request.urlopen('http://example.com')",
    "import ctypes; ctypes.CDLL(None)",
])
def test_subprocess_and_socket_fail(code, scratch):
    r = run(code, scratch)
    assert r.returncode != 0 and ("blocked" in r.stderr or "PermissionError" in r.stderr), r.stderr


@need_user
def test_least_squares_fit_of_graft_w_out(cell, grafted, scratch):
    """The surgeon's workflow: fit w_out of a graft in the sandbox, read it back, graft it."""
    v1 = grafted[1]
    S.export(cell, v1, scratch)
    rng = np.random.default_rng(1)
    np.save(os.path.join(scratch, "w_in.npy"), (rng.standard_normal((16, 128)) * 0.5).astype(np.float32))
    code = """
import numpy as np, torch, model
net = model.load()
neg = model.negatives()
xi, c = torch.from_numpy(neg['xi']), torch.from_numpy(neg['c'])
a, states, acts = model.decode(net, xi, c, save_states=True, capture=('out_act',))
k = 6
h = acts['out_act'][k]                                      # (N, 128) at step k
w_in = np.load('w_in.npy')
z = torch.nn.functional.silu(h @ torch.from_numpy(w_in).T)  # (N, 16)
target = -0.5 * torch.from_numpy(neg['a'])                  # desired velocity change (N, 2)
sol = np.linalg.lstsq(z.numpy().astype(np.float64), target.numpy().astype(np.float64), rcond=None)[0]
np.save('w_out.npy', sol.T.astype(np.float32))              # (2, 16)
print('fit', sol.shape, float(np.abs(z.numpy() @ sol - target.numpy()).mean()))
"""
    r = run(code, scratch)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("fit (16, 2)") and r.decodes == len(cell.neg)
    w_out = np.load(os.path.join(scratch, "w_out.npy"))
    d = E.Detector("t", (1.0,), 0.0, -1.0, k_range=(6, 7))
    g = E.graft_gated_path(v1, "v2", "out", d, 16, np.load(os.path.join(scratch, "w_in.npy")), w_out)
    assert g.record.ok and E.unit_tests(g.version, v1).passed


@need_user
def test_export_is_read_only_and_copy_back_rules(scratch):
    code = """
import os
try:
    open(os.path.join('export', 'weights.pt'), 'ab').write(b'x')
    print('WROTE')
except PermissionError:
    print('readonly')
os.symlink('/etc/passwd', 'link.txt')
open('ok.npy', 'wb').write(b'0')
open('evil.so', 'wb').write(b'0')
"""
    before = open(os.path.join(scratch, "export", "weights.pt"), "rb").read()
    r = run(code, scratch)
    assert "readonly" in r.stdout, (r.stdout, r.stderr)
    assert open(os.path.join(scratch, "export", "weights.pt"), "rb").read() == before
    assert os.path.exists(os.path.join(scratch, "ok.npy"))
    assert not os.path.lexists(os.path.join(scratch, "link.txt")) and not os.path.exists(os.path.join(scratch, "evil.so"))
    assert os.listdir(S.SANDBOX_ROOT) == [] or all(not d.startswith(".") for d in os.listdir(S.SANDBOX_ROOT))


@need_user
def test_output_cap_and_timeout(scratch):
    r = run("print('x' * 50000)", scratch)
    assert r.truncated and len(r.stdout) < S.OUTPUT_TOKEN_CAP * S.CHARS_PER_TOKEN + 200 and "cut" in r.stdout
    r = run("import time; time.sleep(30)", scratch, timeout_s=3)
    assert r.timed_out and r.returncode != 0


def test_runner_without_user(scratch):
    """Mechanics only (user=None runs as the caller; used where no sandbox user exists)."""
    r = run("import numpy as np; np.save('z.npy', np.arange(3)); print('ok')", scratch, user=None)
    assert r.returncode == 0 and r.stdout.strip() == "ok"
    assert np.array_equal(np.load(os.path.join(scratch, "z.npy")), np.arange(3))
    r = run("import toypref", scratch, user=None)
    assert r.returncode != 0 and "blocked" in r.stderr
