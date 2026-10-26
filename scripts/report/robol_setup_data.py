"""Data for robol_setup.png: 3 held-out courses with 16 base rollouts each + one negative episode's label windows."""
import pickle
from robol_common import *

ref, lg, st, sel, idx, noise = setup()
ro = DT.rollout_chunked(ref, st, idx, noise)
oc = outcome(ro)
print("base held clean/neg/timeout", (oc == 0).mean(), (oc == 1).mean(), (oc == 2).mean())
courses = pick_courses(oc, sel, 3)
print("courses (held idx, setup row)", courses)
ev = E.events(ro.pos, ro.contact)
panels = []
for j, c in courses:
    ii = np.arange(j * M, j * M + 16)
    panels.append(dict(course=pack_course(st, c), sid=int(st.sid[c]), setup=int(c),
                       pos=[ro.pos[i][~np.isnan(ro.pos[i, :, 0])] for i in ii], oc=oc[ii]))
# an example negative: a single contact event at chunk >= 4, taken from the chosen courses, else any held course
best = None
for pool in (np.concatenate([np.arange(j * M, j * M + 16) for j, _ in courses]), np.arange(len(idx))):
    for i in pool:
        e = ev[i]
        if ro.negative[i] and len(e) == 1 and e[0][0] == "contact":
            hi = min(e[0][1] // 8, 17)
            if 4 <= hi <= 12:
                best = i; break
    if best is not None:
        break
i = best
hi = min(ev[i][0][1] // 8, 17)
n_ran = int(ro.ran[i].sum())
P = ro.pos[i][~np.isnan(ro.pos[i, :, 0])]
ex = dict(course=pack_course(st, int(idx[i])), pos=P, hi=hi, n_ran=n_ran, onset=int(ev[i][0][1]), events=ev[i],
          course_in_panels=any(int(idx[i]) == c for _, c in courses))
pickle.dump(dict(panels=panels, example=ex, base_rates=dict(clean=float((oc == 0).mean()), neg=float((oc == 1).mean()), timeout=float((oc == 2).mean())),
                 courses=courses), open(os.path.join(OUT, "setup_data.pkl"), "wb"))
print("example ep", i, "hi", hi, "n_ran", n_ran, "events", ev[i])
