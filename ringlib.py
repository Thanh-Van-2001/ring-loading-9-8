import numpy as np, numba as nb, itertools

@nb.njit(cache=True)
def alpha_exact(u, v, ub, fullk=False):
    # min over z in prod{v_i,-u_i} of 2*max_{k<m}|S_k - T/2|.  B&B DFS, prune by range of prefix sums.
    m = u.shape[0]
    best = ub
    # suffix max possible |remaining| not needed; use range bound
    zi = np.zeros(m, np.int64); S = np.zeros(m + 1); mn = np.zeros(m + 1); mx = np.zeros(m + 1)
    mn[0] = 1e18; mx[0] = -1e18
    d = 0; zi[0] = -1
    while d >= 0:
        zi[d] += 1
        if zi[d] > 1:
            d -= 1; continue
        z = v[d] if zi[d] == 0 else -u[d]
        S[d + 1] = S[d] + z
        if d + 1 < m:            # prefix k=d+1 is constrained (k<=m-1)
            mn[d + 1] = min(mn[d], S[d + 1]); mx[d + 1] = max(mx[d], S[d + 1])
            if mx[d + 1] - mn[d + 1] >= best: continue
            d += 1; zi[d] = -1
        else:
            h = S[m] / 2.0
            a = 2.0 * max(mx[d] - h, h - mn[d])
            if fullk: a = max(a, abs(S[m]))
            if a < best: best = a
    return best

def alpha_brute(u, v):
    m = len(u); best = 1e18
    for z in itertools.product(*[(v[i], -u[i]) for i in range(m)]):
        best = min(best, max(abs(sum(z[:k]) - sum(z[k:])) for k in range(1, m)))
    return best

@nb.njit(cache=True)
def near_active(u, v, thr, maxcnt, fullk=False):
    # all z (bitmask, bit i=1 -> -u_i) with alpha_z <= thr
    m = u.shape[0]; out = np.zeros(maxcnt, np.int64); cnt = 0
    zi = np.zeros(m, np.int64); S = np.zeros(m + 1); mn = np.zeros(m + 1); mx = np.zeros(m + 1)
    mn[0] = 1e18; mx[0] = -1e18; d = 0; zi[0] = -1
    while d >= 0:
        zi[d] += 1
        if zi[d] > 1:
            d -= 1; continue
        z = v[d] if zi[d] == 0 else -u[d]
        S[d + 1] = S[d] + z
        if d + 1 < m:
            mn[d + 1] = min(mn[d], S[d + 1]); mx[d + 1] = max(mx[d], S[d + 1])
            if mx[d + 1] - mn[d + 1] > thr: continue
            d += 1; zi[d] = -1
        else:
            h = S[m] / 2.0
            a = 2.0 * max(mx[d] - h, h - mn[d])
            if fullk: a = max(a, abs(S[m]))
            if a <= thr:
                if cnt < maxcnt:
                    b = 0
                    for i in range(m):
                        if zi[i] == 1: b |= (1 << i)
                    out[cnt] = b
                cnt += 1
    return out[:min(cnt, maxcnt)], cnt
