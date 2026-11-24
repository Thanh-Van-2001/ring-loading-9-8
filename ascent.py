import sys, json, time, numpy as np
from scipy.optimize import linprog
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from ringlib import alpha_exact, near_active

def pieces(b, m, fullk=False):
    # rows: coefficient vectors over x=[u,v] for L_k, k=1..m-1 ; L_k = sum_{i<=k} z_i - sum_{i>k} z_i
    bits = np.array([(b >> i) & 1 for i in range(m)])
    K = m if fullk else m - 1
    A = np.zeros((K, 2 * m))
    for k in range(1, K + 1):
        c = np.where(np.arange(m) < k, 1.0, -1.0)
        A[k-1, :m] = np.where(bits == 1, -c, 0.0)      # u coeffs
        A[k-1, m:] = np.where(bits == 0, c, 0.0)       # v coeffs
    return A

def ascend(u, v, iters=300, delta=0.02, r=0.05, verbose=False, fullk=False):
    m = len(u); x = np.concatenate([u, v]); a = alpha_exact(u, v, 1e9, fullk)
    for it in range(iters):
        zs, cnt = near_active(x[:m].copy(), x[m:].copy(), a + delta, 20000, fullk)
        rows = []
        for b in zs:
            A = pieces(int(b), m, fullk); val = A @ x
            j = np.argmax(np.abs(val)); s = np.sign(val[j]) or 1.0
            rows.append(-s * A[j])                    # -(s*A_j)(x+dx) + t <= 0
        R = np.array(rows); nz = R.shape[0]
        # vars: dx (2m), t ; maximize t
        A_ub = np.hstack([R, np.ones((nz, 1))]); b_ub = -(R @ x)
        # u_i+v_i <= 1
        C = np.zeros((m, 2*m + 1)); C[np.arange(m), np.arange(m)] = 1; C[np.arange(m), m + np.arange(m)] = 1
        A_ub = np.vstack([A_ub, C]); b_ub = np.concatenate([b_ub, 1 - (x[:m] + x[m:])])
        bounds = [(max(-r, -xi), r) for xi in x] + [(None, None)]
        res = linprog(np.r_[np.zeros(2*m), -1.0], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0: r *= 0.5; continue
        step = res.x[:2*m]; ok = False
        for sc in (1.0, 0.5, 0.25, 0.1):
            xn = np.clip(x + sc * step, 0, None); s_ = xn[:m] + xn[m:]; f = np.maximum(s_, 1.0)
            xn[:m] /= f; xn[m:] /= f
            an = alpha_exact(xn[:m].copy(), xn[m:].copy(), 1e9, fullk)
            if an > a + 1e-12: x, a, ok = xn, an, True; break
        if ok: r = min(r * 1.5, 0.2)
        else:
            r *= 0.5; delta = min(delta * 1.5, 0.2)
        if verbose and it % 10 == 0: print(it, round(a, 9), nz, cnt, round(r, 5), flush=True)
        if r < 1e-7: break
    return x[:m], x[m:], a

if __name__ == '__main__':
    sol = json.load(open(sys.argv[2])); u = np.array([abs(p) for p, q in sol]); v = np.array([abs(q) for p, q in sol])
    t = time.time(); u2, v2, a2 = ascend(u, v, iters=int(sys.argv[3]), verbose=True)
    print('start', alpha_exact(u, v, 1e9), 'end', a2, 'secs', round(time.time() - t, 1))
    json.dump([[float(p), float(q)] for p, q in zip(u2, v2)], open(sys.argv[4], 'w'))
