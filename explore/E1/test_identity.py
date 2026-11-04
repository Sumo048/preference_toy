"""Graft identity: a gated path with zero output weights, or with its detector off, gives
bit-identical decodes (PLAN Sec. 6.7 / 10.3), on 10,000 noises."""
import torch, numpy as np
from common import *
from graft import GCfg, Graft, make_hooks
ref = load_ref(0); net = ref.net
g = torch.Generator().manual_seed(0)
xi = torch.randn(10000, 2, generator=g); c = torch.rand(10000, generator=g)
a0 = decode_batched(net, xi, c)
cfg = GCfg(n_units=16)
d = 128 + 2 + 16
mk = lambda w, b: Graft(cfg, d, d, w, b, torch.zeros(d), torch.ones(d), torch.zeros(d), torch.ones(d), 2)
gr = mk(torch.randn(d), torch.tensor(0.0))                    # zero w_out, detector fires often
a1 = decode_batched(net, xi, c, hooks=make_hooks(net, gr))
gr2 = mk(torch.zeros(d), torch.tensor(-1.0)); gr2.w_out.data.normal_()   # detector never fires, w_out random
a2 = decode_batched(net, xi, c, hooks=make_hooks(net, gr2))
print("zero-output graft identical:", torch.equal(a0, a1), "| detector-off graft identical:", torch.equal(a0, a2))
assert torch.equal(a0, a1) and torch.equal(a0, a2)
