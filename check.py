import sys, json, numpy as np, time
sys.path.insert(0, sys.argv[1]); from ringlib import alpha_exact, alpha_brute
sol = json.load(open(sys.argv[2])); u = np.array([abs(a) for a, b in sol]); v = np.array([abs(b) for a, b in sol])
print('m', len(u), 'max u+v', (u+v).max())
t = time.time(); a = alpha_exact(u, v, 1e9); print('bnb', a, time.time()-t)
t = time.time(); print('brute', alpha_brute(list(u), list(v)), time.time()-t)
rng = np.random.default_rng(0)
for _ in range(200):
    m = rng.integers(2, 12); x = rng.random((m, 2)); x /= np.maximum(1, x.sum(1))[:, None]
    assert abs(alpha_exact(x[:,0].copy(), x[:,1].copy(), 1e9) - alpha_brute(list(x[:,0]), list(x[:,1]))) < 1e-12
print('random cross-check OK')
