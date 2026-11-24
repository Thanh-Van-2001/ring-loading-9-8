# Exact proof that the ring instance built by ring_build.py has unsplittable optimum L >= L* + 9/8 D.
# For each crossing choice z (2^m), loads with fillers short are l_e. Rerouting fillers (total R, r_e from edge e)
# gives load l_e + R - 2 r_e. Even with fractional fillers (a relaxation), max <= T is achievable iff
#   max l <= T,  or  [l1 > T >= l2 and l1 + l2 <= 2T]  (necessary; capacity is left to the exact DP below).
# If no z passes, L >= T exactly.  Integer arithmetic in units of 1/den (loads are multiples of 1/den).
import sys, itertools, numpy as np
from fractions import Fraction as F
from math import lcm
sys.path.insert(0, '.'); from ring_build import load_uv, build, cw_edges
u, v = load_uv(sys.argv[1]); m = len(u); n, dem, Bmax = build(u, v)
den = lcm(*[x[2].denominator for x in dem], Bmax.denominator, 8)   # all loads are integers in units of 1/den
cross, fill = dem[:m], dem[m:]
Fcap = np.zeros(n, np.int64); base = np.zeros(n, np.int64)
for s, t, d in fill: Fcap[s] += int(d * den); base[s] += int(d * den)       # filler on edge s, routed short
cwm = np.zeros((m, n), bool)
for i, (s, t, d) in enumerate(cross): cwm[i, cw_edges(s, t, n)] = True
dd = np.array([int(d * den) for s, t, d in cross])
T = int(Bmax * den + F(9, 8) * den)            # target L* + 9/8 (loads strictly below T must be impossible)
Tm = T - 1                                      # test achievability of max load <= T - 1/den
bad = 0; minmax = None
for bits in range(2 ** m):
    y = np.array([(bits >> i) & 1 for i in range(m)], bool)   # 1 = clockwise
    l = base + ((cwm & y[:, None]) | (~cwm & ~y[:, None])).astype(np.int64).T @ dd
    o = np.argsort(l)[::-1]; l1, l2 = l[o[0]], l[o[1]]
    ok = l1 <= Tm or (l1 > Tm >= l2 and l1 + l2 <= 2 * Tm)
    bad += ok; minmax = l1 if minmax is None else min(minmax, l1)
print(f'n={n} demands={len(dem)} L*={Bmax}  min over z of max load (fillers short) = {F(int(minmax), den)}')
print(f'choices z where load < L* + 9/8 is achievable even with fractional filler rerouting: {bad} of {2**m}')
print('=> L >= L* + 9/8 D  PROVEN' if bad == 0 else '=> necessary condition alone is not enough; exact DP on these choices below')

# ---- Exact discrete check for the z that pass the fractional relaxation ----
# For fixed z and total rerouted R: each edge needs r_e >= (l_e + R - Tm)/2, r_e a subset sum of its filler chunks,
# and sum r_e = R.  Subset-sum DP over edges in integers; feasible for some R  <=>  load <= Tm achievable.
import numba as nb
chunks = [[] for _ in range(n)]
for s, t, d in fill: chunks[s].append(int(d * den))
maxsub = 0; subs = []
for e in range(n):
    S = {0}
    for c in chunks[e]: S |= {x + c for x in S}
    subs.append(sorted(S)); maxsub = max(maxsub, len(S))
SUB = np.full((n, maxsub), -1, np.int64)
for e in range(n): SUB[e, :len(subs[e])] = subs[e]
Rmax = int(sum(Fcap))

@nb.njit(cache=True)
def feasible(l, Tm, SUB, Rmax):
    n = l.shape[0]
    for R in range(0, Rmax + 1):
        reach = np.zeros(R + 1, np.bool_); reach[0] = True
        for e in range(n):
            need = l[e] + R - Tm            # need 2 r_e >= need
            nxt = np.zeros(R + 1, np.bool_)
            for j in range(SUB.shape[1]):
                r = SUB[e, j]
                if r < 0 or r > R: continue
                if 2 * r < need: continue
                for x in range(R + 1 - r):
                    if reach[x]: nxt[x + r] = True
            reach = nxt
        if reach[R]: return R
    return -1

cnt_feasible = 0
for bits in range(2 ** m):
    y = np.array([(bits >> i) & 1 for i in range(m)], bool)
    l = base + ((cwm & y[:, None]) | (~cwm & ~y[:, None])).astype(np.int64).T @ dd
    o = np.argsort(l)[::-1]; l1, l2 = l[o[0]], l[o[1]]
    if not (l1 <= Tm or (l1 > Tm >= l2 and l1 + l2 <= 2 * Tm)): continue
    R = feasible(l.astype(np.int64), Tm, SUB, Rmax)
    if R >= 0: cnt_feasible += 1; print('feasible z', bits, 'R', F(R, den))
print(f'exact discrete check: {cnt_feasible} choices z can get below L* + 9/8')
print('=> L >= L* + 9/8 D  PROVEN EXACTLY' if cnt_feasible == 0 else '=> NOT proven')
