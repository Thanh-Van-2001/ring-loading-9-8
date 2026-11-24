# Build a ring-loading instance from a (u,v) sequence and compute split optimum L* (LP + exact cut bound)
# and unsplittable optimum L (MILP). Gap = (L - L*)/D.
import sys, json, numpy as np
from fractions import Fraction as F
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def load_uv(path):
    d = json.load(open(path)); u0, v0 = d['u'], d['v']
    for D in (16, 48, 84, 168, 840, 10**4):
        u = [F(x).limit_denominator(D) for x in u0]; v = [F(x).limit_denominator(D) for x in v0]
        if max(abs(float(a) - b) for a, b in zip(u + v, u0 + v0)) < 1e-9: return u, v
    raise SystemExit('rationalise failed')

def cw_edges(s, t, n):   # clockwise s -> t uses edges s, s+1, ..., t-1 (mod n)
    e = []; j = s
    while j != t: e.append(j); j = (j + 1) % n
    return e

def build(u, v, filler='adjacent'):
    m = len(u); n = 2 * m; dem = []
    for i in range(m): dem.append((i, m + i, u[i] + v[i]))
    B = [F(0)] * n
    for i in range(m):
        cw = set(cw_edges(i, m + i, n))
        for e in range(n): B[e] += u[i] if e in cw else v[i]
    Bmax = max(B)
    for e in range(n):
        r = Bmax - B[e]
        while r > 0:
            c = min(r, F(1)); dem.append((e, (e + 1) % n, c)); r -= c
    return n, dem, Bmax

def incidence(n, dem):
    P = np.zeros((n, len(dem)))   # 1 if edge on cw path
    for k, (s, t, d) in enumerate(dem):
        for e in cw_edges(s, t, n): P[e, k] = 1
    return P

def solve(n, dem):
    K = len(dem); d = np.array([float(x[2]) for x in dem]); P = incidence(n, dem)
    # load_e = sum_k P[e,k]*d_k*y_k + (1-P[e,k])*d_k*(1-y_k) <= L   (y = cw fraction)
    A = (P * d) - ((1 - P) * d); const = ((1 - P) * d).sum(1)
    Aub = np.hstack([A, -np.ones((n, 1))]); bub = -const
    c = np.r_[np.zeros(K), 1.0]; bnd = [(0, 1)] * K + [(0, None)]
    lp = linprog(c, A_ub=Aub, b_ub=bub, bounds=bnd, method='highs')
    mi = milp(c, constraints=LinearConstraint(Aub, -np.inf, bub), integrality=np.r_[np.ones(K), 0],
              bounds=Bounds(np.r_[np.zeros(K), 0], np.r_[np.ones(K), np.inf]), options={'time_limit': 600, 'mip_rel_gap': 0})
    return lp.fun, mi.fun, mi

def cut_bound(n, dem):
    best = F(0)
    for a in range(n):
        for b in range(a + 1, n):
            side = set(range(a + 1, b + 1))
            sep = sum((d for s, t, d in dem if (s in side) != (t in side)), F(0))
            best = max(best, sep / 2)
    return best

if __name__ == '__main__':
    u, v = load_uv(sys.argv[1]); n, dem, Bmax = build(u, v)
    D = max(x[2] for x in dem); Lstar, L, mi = solve(n, dem); cb = cut_bound(n, dem)
    print(f'm={len(u)} ring nodes={n} demands={len(dem)} D={D} split-routing max load={Bmax} ({float(Bmax):.4f})')
    print(f'L* (LP) = {Lstar:.6f}   exact cut lower bound on L* = {cb} ({float(cb):.6f})')
    print(f'L (MILP, unsplittable) = {L:.6f}  status={mi.status}  gap (L-L*)/D = {(L - Lstar)/float(D):.6f}')
