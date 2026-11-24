# Exact max_x alpha(x) for fixed m: MILP over a growing set of z (constraint generation).
import sys, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
sys.path.insert(0, '.'); from ringlib import alpha_exact, near_active
from ascent import pieces

def solve(m, Z, tl):
    P = 2 * (m - 1); nz = len(Z); nv = 2 * m + 1 + nz * P; M = m + 2.0
    A = lil_matrix((m + nz + nz * P, nv)); lo = []; hi = []; r = 0
    for i in range(m): A[r, i] = 1; A[r, m + i] = 1; lo.append(-np.inf); hi.append(1); r += 1
    for j in range(nz):
        base = 2 * m + 1 + j * P
        for q in range(P): A[r, base + q] = 1
        lo.append(1); hi.append(np.inf); r += 1
    for j, b in enumerate(Z):
        Pk = pieces(int(b), m); base = 2 * m + 1 + j * P
        for k in range(m - 1):
            for t, s in enumerate((1.0, -1.0)):
                row = s * Pk[k]                           # s*L(x) - alpha - M*y >= -M
                for c in np.nonzero(row)[0]: A[r, c] = row[c]
                A[r, 2 * m] = -1; A[r, base + 2 * k + t] = -M; lo.append(-M); hi.append(np.inf); r += 1
    c = np.zeros(nv); c[2 * m] = -1
    integ = np.r_[np.zeros(2 * m + 1), np.ones(nz * P)]
    lb = np.zeros(nv); ub = np.r_[np.ones(2 * m), 1.3, np.ones(nz * P)]
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={'time_limit': tl, 'disp': False})
    return res

def run(m, tl=600, maxit=200):
    rng = np.random.default_rng(m); Z = list(rng.integers(0, 2 ** m, 8)); t0 = time.time(); lb = 0
    for it in range(maxit):
        res = solve(m, Z, tl)
        if res.x is None: print('no sol', res.message); return
        x = res.x; u, v = x[:m].copy(), x[m:2 * m].copy(); ubd = x[2 * m]
        if res.status != 0: ubd = -res.mip_dual_bound if hasattr(res, 'mip_dual_bound') and res.mip_dual_bound is not None else ubd
        a = alpha_exact(u, v, 1e9); lb = max(lb, a)
        new, cnt = near_active(u, v, ubd - 1e-7, 30)
        print(f'm={m} it={it} |Z|={len(Z)} ub={ubd:.6f} alpha(x)={a:.6f} best_lb={lb:.6f} t={time.time()-t0:.0f}s st={res.status}', flush=True)
        if a >= ubd - 1e-7 or len(new) == 0: print(f'm={m} OPT={ubd:.9f}'); return ubd, u, v
        Z += [b for b in new if b not in Z][:10]
    print('maxit')

if __name__ == '__main__':
    for m in [int(x) for x in sys.argv[1].split(',')]: run(m, tl=float(sys.argv[2]))
