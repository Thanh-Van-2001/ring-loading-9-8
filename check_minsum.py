"""Third check, written separately by a reviewing agent: rebuilds the ring from the paper's text, verifies Lemma 4
for all z, and decides "max load < L* + gap" WITHOUT Lemma 6: for every crossing routing z and every R it tests the
necessary condition  sum_e min{s in Sigma_e : s <= R, 2s >= l_e + R - T} <= R,  then runs an exact DP on survivors.
    python -I check_minsum.py instances/instance_m15_den16_a.json 9 8"""
import json, sys
from fractions import Fraction as F
from math import lcm
import numpy as np

path, gapnum, gapden = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(path))
def rat(x):
    for D in (16, 48, 84, 168, 840):
        f = F(x).limit_denominator(D)
        if abs(float(f) - x) < 1e-12: return f
    raise SystemExit("rat")
u = [rat(x) for x in d["u"]]; v = [rat(x) for x in d["v"]]
m = len(u); n = 2 * m
# paper indexing: demand i (1..m) between nodes i-1 and m+i-1; cw path edges i-1..m+i-2
cwset = [set(range(i - 1, m + i - 1)) for i in range(1, m + 1)]
B = [sum(u[i] if e in cwset[i] else v[i] for i in range(m)) for e in range(n)]
Bbar = max(B); phi = [Bbar - b for b in B]
pieces = []
for e in range(n):
    r = phi[e]
    while r >= 1: pieces.append((e, F(1))); r -= 1
    if r > 0: pieces.append((e, r))
sizes = [u[i] + v[i] for i in range(m)] + [p for _, p in pieces]
print("Bbar", Bbar, "max phi", max(phi), "sum phi", sum(phi), "#pieces", len(pieces), "#demands", len(sizes), "D", max(sizes))
print("size-1 crossing:", [i + 1 for i in range(m) if u[i] + v[i] == 1])
print("pieces per edge:", {e: [str(p) for ee, p in pieces if ee == e] for e in range(n) if phi[e] > 0})
den = lcm(*[s.denominator for s in sizes + u + v], Bbar.denominator, gapden)
T = Bbar + F(gapnum, gapden)
Tm = int(T * den) - 1   # max load strictly below T  <=> <= Tm (loads are multiples of 1/den)
assert all((s * den).denominator == 1 for s in sizes)
print("den", den, "T", T, "Tm/den", F(Tm, den))
# loads with fillers short, all z
Z = np.arange(2 ** m)[:, None]; Y = (Z >> np.arange(m)) & 1   # Y=1 cw
cw = np.array([[1 if e in cwset[i] else 0 for e in range(n)] for i in range(m)])
dd = np.array([int((u[i] + v[i]) * den) for i in range(m)])
fil = np.zeros(n, np.int64)
for e, p in pieces: fil[e] += int(p * den)
ell = Y @ (cw * dd[:, None]) + (1 - Y) @ ((1 - cw) * dd[:, None]) + fil
# Lemma 4 check: z_i = v_i if cw else -u_i ; edge k-1 -> Bbar + Delta_k ; edge m+k-1 -> Bbar - Delta_k
U = np.array([int(x * den) for x in u]); V = np.array([int(x * den) for x in v])
z = np.where(Y == 1, V, -U); P = np.cumsum(z, axis=1); S = P[:, -1:]
Delta = 2 * P - S   # Delta_k for k=1..m
bb = int(Bbar * den)
assert np.array_equal(ell[:, :m], bb + Delta) and np.array_equal(ell[:, m:], bb - Delta), "Lemma 4 fails"
print("Lemma 4 verified for all", 2 ** m, "z; min_z max ell =", F(int(ell.max(1).min()), den))
# subset sums per edge
subs = []
for e in range(n):
    s = {0}
    for ee, p in pieces:
        if ee == e: s |= {x + int(p * den) for x in s}
    subs.append(sorted(s))
Rmax = int(fil.sum())
# vectorised necessary condition: for some R, every edge has a subset sum s<=R with 2s >= ell+R-Tm, and sum of minimal such <= R
INF = 10 ** 9
alive = np.zeros(2 ** m, bool)
for R in range(Rmax + 1):
    need = ell + R - Tm
    lo = np.full(ell.shape, INF, np.int64)
    for e in range(n):
        for s in reversed(subs[e]):
            if s <= R:
                lo[:, e] = np.where(2 * s >= need[:, e], s, lo[:, e])
    alive |= (lo.sum(1) <= R)
cand = np.flatnonzero(alive)
print("z passing the (prefilter-free) necessary condition:", len(cand))
# exact DP on candidates, also compute exact min max load per candidate
def feasible(l, Tq):
    for R in range(Rmax + 1):
        reach = {0}
        for e in range(n):
            al = [s for s in subs[e] if s <= R and 2 * s >= l[e] + R - Tq]
            reach = {x + s for x in reach for s in al if x + s <= R}
            if not reach: break
        if R in reach: return R
    return None
bad = 0
for zb in cand:
    l = [int(x) for x in ell[zb]]
    if feasible(l, Tm) is not None: bad += 1; print("FEASIBLE z", zb)
    # exact optimum for this z
    lo_, hi_ = 0, max(l)
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if feasible(l, mid) is not None: hi_ = mid
        else: lo_ = mid + 1
    print(f"  z={zb:0{m}b} (MSB=demand {m}) min max load over filler reroutings = {F(lo_, den)}")
print("routings with max load < T:", bad)
