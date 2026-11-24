# Iterated local search chain: perturb / insert / delete / reverse, then LP active-set ascent. Writes best per m.
import sys, json, time, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ascent import ascend
from ringlib import alpha_exact

def norm(u, v):
    u, v = np.clip(u, 0.0, 1), np.clip(v, 0.0, 1); f = np.maximum(u + v, 1); return u / f, v / f

def perturb(rng, u, v, mlo, mhi):
    m = len(u); k = rng.integers(6); u, v = u.copy(), v.copy()
    if k == 0 or k == 1:
        s = [0.005, 0.02, 0.06][rng.integers(3)]; u += rng.normal(0, s, m); v += rng.normal(0, s, m)
    elif k == 2 and m < mhi:
        i = rng.integers(m + 1); w = rng.random(); s = rng.uniform(0.5, 1)
        u, v = np.insert(u, i, w * s), np.insert(v, i, (1 - w) * s)
    elif k == 3 and m > mlo:
        i = rng.integers(m); u, v = np.delete(u, i), np.delete(v, i)
    elif k == 4 and m < mhi:      # split element i into two
        i = rng.integers(m); a = rng.uniform(0.3, 0.7)
        u = np.insert(u, i, u[i]); v = np.insert(v, i, v[i]); u[i] *= a; v[i] *= a; u[i+1] *= (1-a); v[i+1] *= (1-a)
        u[i] += rng.uniform(0, 0.2); v[i+1] += rng.uniform(0, 0.2)
    else:
        i, j = sorted(rng.choice(m, 2, replace=False)); u[i:j+1] = u[i:j+1][::-1].copy(); v[i:j+1] = v[i:j+1][::-1].copy()
    return norm(u, v)

if __name__ == '__main__':
    seed, secs, mlo, mhi, outdir = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rng = np.random.default_rng(seed); os.makedirs(outdir, exist_ok=True)
    sol = json.load(open(os.path.join(HERE, 'ae_ring_sol.json')))
    u, v = np.array([abs(p) for p, q in sol]), np.array([abs(q) for p, q in sol])
    elite = len(sys.argv) > 6 and sys.argv[6].startswith('elite'); FK = len(sys.argv) > 6 and sys.argv[6].endswith('fullk'); stall = 0
    if len(sys.argv) > 7:   # init from file (dict with u, v)
        dd = json.load(open(sys.argv[7])); u, v = np.array(dd['u'], float), np.array(dd['v'], float)
    elif seed % 2 == 1 and not elite:   # odd chains start random
        m = rng.integers(mlo, mhi + 1); s = rng.uniform(0.6, 1, m); w = rng.random(m); u, v = w * s, (1 - w) * s
    u, v, a = ascend(u, v, iters=60, fullk=FK); best = {}; gbest = (a, u.copy(), v.copy()); t0 = time.time(); it = 0; T = 0.01
    while time.time() - t0 < secs:
        it += 1
        un, vn = perturb(rng, u, v, mlo, mhi)
        un, vn, an = ascend(un, vn, iters=40, fullk=FK)
        if an > a - T * rng.exponential():   # annealed acceptance
            u, v, a = un, vn, an
        if elite:
            if an > gbest[0] + 1e-12: gbest = (an, un.copy(), vn.copy()); stall = 0
            else: stall += 1
            if stall >= 25: u, v, a = gbest[1].copy(), gbest[2].copy(), gbest[0]; stall = 0
        m = len(u)
        if a > best.get(m, 0) + 1e-12:
            best[m] = a
            json.dump({'alpha': a, 'm': m, 'u': u.tolist(), 'v': v.tolist(), 'it': it, 't': time.time() - t0},
                      open(f'{outdir}/best_s{seed}_m{m}.json', 'w'))
            print(f'{time.time()-t0:7.0f}s it={it} m={m} alpha={a:.10f}', flush=True)
        T = max(0.001, T * 0.999)
