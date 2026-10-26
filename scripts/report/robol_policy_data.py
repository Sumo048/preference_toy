"""Data for robol_rollouts.png and robol_chunkfan.png from the saved seed-0 policies (traj labels, N_s = 512)."""
import json, pickle
from robol_common import *

POL = os.path.join(OUT, "policies")
ARMS = [("B0", None), ("NPO", "npo"), ("KTO", "kto"), ("SDPO fresh", "fresh"), ("SDPO pinned", "pin"),
        ("SDPO pinned, 5000 steps", "pin5000"), ("SDPO fresh + re-collected negatives, S=8", "sn8"),
        ("SDPO fresh + re-collected negatives, S=64", "sn64")]
ref, lg, st, sel, idx, noise = setup()
pols = {"B0": ref}
meta = {"B0": {}}
for nm, f in ARMS[1:]:
    if os.path.exists(os.path.join(POL, f + ".pt")):
        pols[nm] = load(os.path.join(POL, f + ".pt")); meta[nm] = json.load(open(os.path.join(POL, f + ".json")))
    else:
        print("MISSING", nm)
base = DT.rollout_chunked(ref, st, idx, noise)
boc = outcome(base)
courses = pick_courses(boc, sel, 3)
print("courses", courses)

# ---------------- figure 2 data: all held episodes per arm for rates, 16 episodes on 3 courses for paths
res = {}
for nm, p in pols.items():
    ro = base if nm == "B0" else DT.rollout_chunked(p, st, idx, noise)
    oc = outcome(ro)
    res[nm] = dict(rates=dict(clean=float((oc == 0).mean()), neg=float((oc == 1).mean()), timeout=float((oc == 2).mean())), meta=meta[nm],
                   panels=[dict(pos=[ro.pos[i][~np.isnan(ro.pos[i, :, 0])] for i in range(j * M, j * M + 16)], oc=oc[j * M:j * M + 16]) for j, _ in courses])
    print(nm, res[nm]["rates"], [int((x["oc"] == 1).sum()) for x in res[nm]["panels"]], flush=True)
courses_geo = [pack_course(st, c) for _, c in courses]

# ---------------- figure 3: chunk fan before a typical contact
N = 64


def replay_fan(pol, c, hi, prefix, a_first, rest):
    n = len(a_first)
    ep = E.Episodes(st, np.full(n, c))
    for ch in range(E.N_CHUNKS):
        live = np.flatnonzero(ep.alive)
        if not len(live):
            break
        o = ep.obs()
        a = np.zeros((n, E.ACT_DIM), np.float32)
        if ch < hi:
            a[live] = prefix[ch]
        elif ch == hi:
            a[live] = a_first[live]
        else:
            a[live] = pol.sample(rest[torch.from_numpy(live), ch], torch.from_numpy(o[live])).numpy()
        ep.execute(a, ch)
    neg, col, jerky, js = E.label(ep)
    oc = np.where(neg, 1, np.where(ep.success, 0, 2))
    return oc, ep.pos[:, hi * E.H:(hi + 1) * E.H + 1].copy()


def obs_at(c, hi, prefix):
    ep = E.Episodes(st, np.array([c]))
    for ch in range(hi):
        ep.execute(prefix[ch][None], ch)
    return ep.obs()[0], ep.p.copy(), ep.pos[0, :hi * E.H + 1].copy()


def fan_pack(ro, setups, tag, seed):
    """ro: base rollouts (the logged episodes); fan at chunks whose episode's first event is a contact."""
    ev = E.events(ro.pos, ro.contact)
    cands = [(i, min(e[0][1] // E.H, 17)) for i, e in enumerate(ev) if ro.negative[i] and len(e) and e[0][0] == "contact" and not ro.jerky[i]
             and min(e[0][1] // E.H, 17) >= 3]
    np.random.default_rng(seed).shuffle(cands)
    cands = cands[:80]

    def one(i, hi, pol_list):
        c = int(setups[i]); prefix = ro.act[i, :hi]
        o, p, ppos = obs_at(c, hi, prefix)
        xi = torch.randn(N, E.ACT_DIM, generator=torch.Generator().manual_seed(11)); xi[0] = torch.from_numpy(ro.noise[i, hi])
        rest = torch.randn(N, E.N_CHUNKS, E.ACT_DIM, generator=torch.Generator().manual_seed(12))
        out = {}
        for nm, pol in pol_list.items():
            a = pol.sample(xi, torch.from_numpy(np.repeat(o[None], N, 0)))
            oc, seg = replay_fan(pol, c, hi, prefix, a.numpy(), rest)
            out[nm] = dict(oc=oc, seg=seg, shares=[float((oc == k).mean()) for k in range(3)])
        return out, ppos

    best = None
    agg = {nm: np.zeros(3) for nm in pols}
    for i, hi in cands:
        out, _ = one(i, hi, pols)
        for nm in pols:
            agg[nm] += np.array(out[nm]["shares"]) / len(cands)
        sh = np.array(out["B0"]["shares"])
        score = abs(sh[1] - 0.4) + 0.5 * abs(sh[2] - 0.1)
        if best is None or score < best[0]:
            best = (score, i, hi, out)
    _, i, hi, out = best
    _, ppos = one(i, hi, {"B0": ref})
    print(tag, "fan state: episode", i, "chunk", hi, "B0 shares", np.round(out["B0"]["shares"], 3), flush=True)
    print(tag, "aggregate over", len(cands), "contact states", {k: np.round(v, 3).tolist() for k, v in agg.items()}, flush=True)
    for nm in out:
        print(tag, "fan", nm, np.round(out[nm]["shares"], 3).tolist())
    return dict(fan=out, course=pack_course(st, int(setups[i])), prefix=ppos, logged=ro.pos[i, hi * E.H:(hi + 1) * E.H + 1], hi=hi, ep=int(i),
                agg={k: v.tolist() for k, v in agg.items()}, nstates=len(cands))


# (a) states the methods were trained on: the logged negative episodes of the 512 labelled training courses (replayed with B0)
lgep = lg["episodes"]
tr_setups = np.asarray(lgep["setup"])
sel_tr = np.flatnonzero(tr_setups < 512)
ro_tr = DT.rollout_chunked(ref, st, tr_setups[sel_tr], torch.from_numpy(np.asarray(lgep["noise"])[sel_tr]))
print("replay of logged negatives still negative:", float(ro_tr.negative.mean()), "n", len(sel_tr))
fan_train = fan_pack(ro_tr, tr_setups[sel_tr], "train", 3)
# (b) held-out courses: base failures on the CRN eval noises
fan_held = fan_pack(base, idx, "held", 3)
pickle.dump(dict(courses=courses, courses_geo=courses_geo, res=res, fan_train=fan_train, fan_held=fan_held, replay_neg=float(ro_tr.negative.mean()),
                 order=[a for a, _ in ARMS if a in pols]), open(os.path.join(OUT, "policy_data.pkl"), "wb"))
print("saved")
