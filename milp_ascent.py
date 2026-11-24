# Exact local test: can alpha rise above its current value? For each tight z, alpha_z = max over pieces (k, s);
# alpha_z rises iff at least ONE tied piece rises -> disjunctive LP = MILP over tied-piece choices.
import json, sys, numpy as np
sys.path.insert(0, '.')
from scipy.optimize import milp, LinearConstraint, Bounds
from ringlib import alpha_exact, near_active
from ascent import pieces

def local_milp(u, v, r=0.01, delta=1e-9, kconv='m-1'):
    m = len(u); x = np.r_[u, v]; a = alpha_exact(u, v, 1e9)
    zs, cnt = near_active(u, v, a + delta, 100000)
    rows = []; groups = []
    for b in zs:
        A = pieces(int(b), m); val = A @ x; g = []
        for k in range(m - 1):
            for s in (1.0, -1.0):
                if s * val[k] >= a - 1e-9: g.append(len(rows)); rows.append(s * A[k])
        groups.append(g)
    P = len(rows); R = np.array(rows); M = 4 * m * r + 1
    # vars: dx (2m), t, y (P)
    nv = 2 * m + 1 + P; cons_A = []; lo = []; hi = []
    for p in range(P):           # R_p dx - t + M y_p >= -... :  R_p dx >= t - M(1 - y_p)
        row = np.zeros(nv); row[:2*m] = R[p]; row[2*m] = -1; row[2*m+1+p] = -M; cons_A.append(row); lo.append(-M); hi.append(np.inf)
    for g in groups:
        row = np.zeros(nv); row[[2*m+1+p for p in g]] = 1; cons_A.append(row); lo.append(1); hi.append(np.inf)
    for i in range(m):           # (u+du) + (v+dv) <= 1
        row = np.zeros(nv); row[i] = 1; row[m+i] = 1; cons_A.append(row); lo.append(-np.inf); hi.append(1 - u[i] - v[i])
    lb = np.r_[np.maximum(-r, -x), -1, np.zeros(P)]; ub = np.r_[np.full(2*m, r), 1, np.ones(P)]
    c = np.zeros(nv); c[2*m] = -1
    res = milp(c, constraints=LinearConstraint(np.array(cons_A), lo, hi), integrality=np.r_[np.zeros(2*m+1), np.ones(P)],
               bounds=Bounds(lb, ub), options={'time_limit': 300})
    ties = [len(g) for g in groups]
    return a, cnt, ties, res

if __name__ == '__main__':
    for f in sys.argv[1:]:
        d = json.load(open(f)); u = np.array(d['u']); v = np.array(d['v'])
        a, cnt, ties, res = local_milp(u, v)
        t = -res.fun if res.x is not None else None
        print(f'{f}: alpha={a:.6f} tight={cnt} tied pieces per z: min={min(ties)} max={max(ties)} total={sum(ties)}  best local rise t={t}  status={res.status}')
        if t and t > 1e-9:
            dx = res.x[:2*len(u)]
            for sc in (1, .5, .25, .1, .05):
                xn = np.clip(np.r_[u, v] + sc * dx, 0, None); m = len(u)
                print('   step', sc, 'alpha =', alpha_exact(xn[:m].copy(), xn[m:].copy(), 1e9))
