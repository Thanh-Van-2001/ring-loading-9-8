"""AlphaEvolve's published sequence for Problem 6.61, taken exactly (each float as its exact binary rational).
Prints its exact alpha and an exact upper bound on the gap of its ring lift R(u,v).
L* is exact (Lemma 2: split loads + opposite-edge cut); the upper bound on L comes from a MILP routing that is then
re-evaluated in exact arithmetic, so the printed gap bound is rigorous whatever the solver's tolerance.
    python -I ae_raw_lift.py"""
import json
import os
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certificate import cut_bound, load_of, ring, seq_alpha  # noqa: E402
from ring_build import solve  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
raw = json.load(open(os.path.join(HERE, "instances", "alphaevolve_raw.json")))
u = [F(a) for a, b in raw]; v = [F(b) for a, b in raw]
assert all(x >= 0 for x in u + v)
viol = max(a + b for a, b in zip(u, v))
print(f"m={len(u)}  max(u_i+v_i) = {float(viol):.17f}  min entry = {float(min(u + v)):.6f}")
if viol > 1:                                   # published floats may exceed 1 by rounding; scale down exactly
    s = 1 / viol; u = [x * s for x in u]; v = [x * s for x in v]
    print(f"  scaled by 1/{float(viol):.17f} so that u_i+v_i <= 1 holds exactly")
a1, _ = seq_alpha(u, v, set(range(1, len(u))))
a2, _ = seq_alpha(u, v, set(range(0, len(u) + 1)))
print(f"alpha(k=1..m-1) = {float(a1):.12f}   alpha(k=0..m) = {float(a2):.12f}   47/42 = {47 / 42:.12f}")

n, cross, fill, B, Bmax = ring(u, v)
dem = cross + fill
D = max(d for _, _, d in dem)
cb, _ = cut_bound(n, dem)
assert cb == Bmax
_, Lmilp, mi = solve(n, dem)
y = np.round(mi.x[:-1]).astype(int)            # 1 = clockwise
routed = [(s, t, d, bool(y[k])) for k, (s, t, d) in enumerate(dem)]
L = max(load_of(n, routed))
print(f"ring: {n} nodes, {len(dem)} demands, D = {float(D):.12f}, L* = {float(Bmax):.12f} (exact)")
print(f"exact max load of the MILP routing = {float(L):.12f}  =>  (L - L*)/D <= {float((L - Bmax) / D):.12f}"
      f"   (23/21 = {23 / 21:.12f}, 11/10 = 1.1)")
