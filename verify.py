# Independent exact verification: rationalize, check constraints exactly, brute-force alpha with integers.
import json, sys, itertools, numpy as np
from fractions import Fraction as F
from math import lcm
d = json.load(open(sys.argv[1])); u0, v0 = d['u'], d['v']; m = len(u0)
for D in (8, 16, 24, 32, 48, 64, 96, 128, 168, 240, 336, 420, 480, 840, 1680, 10**4, 10**6):
    u = [F(x).limit_denominator(D) for x in u0]; v = [F(x).limit_denominator(D) for x in v0]
    err = max(abs(float(a) - b) for a, b in zip(u + v, u0 + v0))
    if err < 1e-9 and all(a >= 0 and b >= 0 and a + b <= 1 for a, b in zip(u, v)): break
else:
    raise SystemExit('no faithful rationalisation')
L = lcm(*[x.denominator for x in u + v]); U = [int(x * L) for x in u]; V = [int(x * L) for x in v]
print('m', m, 'D', D, 'err', err, 'common denominator', L, 'max(u+v)', max(a + b for a, b in zip(u, v)))
best = None; best_strict = None; arg = None
for z in itertools.product(*[(V[i], -U[i]) for i in range(m)]):
    T = sum(z); S = list(itertools.accumulate(z))
    a = max(abs(2 * S[k - 1] - T) for k in range(1, m))          # k = 1..m-1 (AlphaEvolve scorer)
    b = max(a, abs(T))                                            # also k = 0 and k = m
    if best is None or a < best: best, arg = a, z
    if best_strict is None or b < best_strict: best_strict = b
print('alpha  (k=1..m-1) =', F(best, L), '=', float(F(best, L)))
print('alpha' + "'" + ' (k=0..m)   =', F(best_strict, L), '=', float(F(best_strict, L)))
print('beats 47/42:', F(best, L) > F(47, 42))
print('u =', [str(x) for x in u]); print('v =', [str(x) for x in v])
