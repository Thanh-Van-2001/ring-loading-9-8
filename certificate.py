"""Independent exact certificate for every number in paper/main.tex.

Integers and Fractions only (numpy int64 for the 2^m enumeration; no floats in any decision,
no numba, no LP/MILP). Written separately from verify.py / ring_exact.py as a second implementation.

    python -I certificate.py            -> prints all numbers quoted in the note
"""
import json
import os
import sys
import time
from fractions import Fraction as F
from math import lcm

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(HERE, "instances")


def load(name, dens=(16, 48, 84, 168, 840)):
    d = json.load(open(os.path.join(INST, name)))
    for D in dens:
        u = [F(x).limit_denominator(D) for x in d["u"]]
        v = [F(x).limit_denominator(D) for x in d["v"]]
        if max(abs(float(a) - b) for a, b in zip(u + v, d["u"] + d["v"])) < 1e-12:
            assert all(a >= 0 for a in u + v) and all(a + b <= 1 for a, b in zip(u, v))
            return u, v
    raise SystemExit(f"{name}: no faithful small-denominator form")


# ---------------------------------------------------------------- sequence constant
def seq_alpha(u, v, ks):
    """min over z in prod{v_k, -u_k} of max_{k in ks} |sum_{i<=k} z_i - sum_{i>k} z_i|; exact. Returns (value, argmin bits)."""
    m = len(u); Q = lcm(*[x.denominator for x in u + v])
    U = [int(x * Q) for x in u]; V = [int(x * Q) for x in v]
    best, arg = None, None
    for bits in range(2 ** m):
        z = [V[i] if (bits >> i) & 1 else -U[i] for i in range(m)]
        S = sum(z); P = 0; worst = abs(S) if 0 in ks else 0
        for k in range(1, m + 1):
            P += z[k - 1]
            if k in ks:
                worst = max(worst, abs(2 * P - S))
            if best is not None and worst >= best:
                break
        if best is None or worst < best:
            best, arg = worst, bits
    return F(best, Q), arg


# ---------------------------------------------------------------- ring lift
def ring(u, v):
    """30-node ring from (u, v): crossing demands (i, i+m) of size u_i+v_i and adjacent fillers on each edge.
    Edge e joins nodes e and e+1 (mod n); the clockwise path i -> i+m uses edges i..i+m-1."""
    m = len(u); n = 2 * m
    B = [sum((u[i] if i <= e < i + m else v[i]) for i in range(m)) for e in range(n)]   # split loads
    Bmax = max(B)
    cross = [(i, i + m, u[i] + v[i]) for i in range(m)]
    fill = []
    for e in range(n):
        r = Bmax - B[e]
        while r > 0:
            c = min(r, F(1)); fill.append((e, (e + 1) % n, c)); r -= c
    return n, cross, fill, B, Bmax


def load_of(n, routed):
    """routed: list of (s, t, d, clockwise?) -> exact load per edge."""
    L = [F(0)] * n
    for s, t, d, cw in routed:
        e = s if cw else t
        stop = t if cw else s
        while e != stop:
            L[e] += d; e = (e + 1) % n
    return L


def cut_bound(n, dem):
    best, arg = F(0), None
    for a in range(n):
        for b in range(a + 1, n):
            sep = sum((d for s, t, d in dem if ((a < s <= b) != (a < t <= b))), F(0))
            if sep / 2 > best:
                best, arg = sep / 2, (a, b, sep)
    return best, arg


def unsplit_search(n, cross, fill, target, strict):
    """Decide whether some unsplittable routing has max load < target (strict=True) or <= target (strict=False).
    Exact: enumerate all 2^m crossing choices; fillers enter only through r_e (rerouted amount from edge e,
    a subset sum of its chunks) and R = sum r_e:  load_e = l_e + R - 2 r_e.  Returns (#prefilter passes, witness)."""
    m = len(cross)
    den = lcm(*[d.denominator for _, _, d in cross + fill], target.denominator)   # loads are integers in units 1/den
    Tm = int(target * den) - (1 if strict else 0)
    base = np.zeros(n, np.int64)
    chunks = [[] for _ in range(n)]
    for s, t, d in fill:
        base[s] += int(d * den); chunks[s].append(int(d * den))
    subs = []
    for e in range(n):
        S = {0: ()}
        for j, c in enumerate(chunks[e]):
            for x, w in list(S.items()):
                S.setdefault(x + c, w + (j,))
        subs.append(S)
    Rmax = sum(sum(c) for c in chunks)
    cwm = np.zeros((m, n), np.int64)
    for i, (s, t, d) in enumerate(cross):
        cwm[i, s:t] = 1
    dd = np.array([int(d * den) for _, _, d in cross], np.int64)
    bits = np.arange(2 ** m)[:, None]
    Y = ((bits >> np.arange(m)) & 1).astype(np.int64)                    # 1 = clockwise
    Lall = Y @ (cwm * dd[:, None]) + (1 - Y) @ ((1 - cwm) * dd[:, None]) + base
    srt = np.sort(Lall, axis=1); l1, l2 = srt[:, -1], srt[:, -2]
    ok = (l1 <= Tm) | ((l1 > Tm) & (l2 <= Tm) & (l1 + l2 <= 2 * Tm))   # necessary condition (paper, Lemma 6)
    passes = np.flatnonzero(ok)
    for zb in passes:
        l = [int(x) for x in Lall[zb]]
        for R in range(Rmax + 1):
            reach = [1]; allowed = []
            for e in range(n):
                al = [r for r in subs[e] if r <= R and 2 * r >= l[e] + R - Tm]
                allowed.append(al)
                nxt = 0
                for r in al:
                    nxt |= reach[-1] << r
                reach.append(nxt & ((1 << (R + 1)) - 1))
                if not reach[-1]:
                    break
            if len(reach) == n + 1 and (reach[-1] >> R) & 1:
                rs, x = [0] * n, R                                          # backtrack one witness
                for e in range(n - 1, -1, -1):
                    for r in allowed[e]:
                        if x - r >= 0 and (reach[e] >> (x - r)) & 1:
                            rs[e] = r; x -= r; break
                routed = [(s, t, d, bool((zb >> i) & 1)) for i, (s, t, d) in enumerate(cross)]
                for e in range(n):
                    moved = set(subs[e][rs[e]])
                    for j, (s, t, d) in enumerate([f for f in fill if f[0] == e]):
                        routed.append((s, t, d, j not in moved))           # moved filler goes the long way
                return len(passes), (int(zb), [F(r, den) for r in rs], routed)
    return len(passes), None


def report_ring(tag, u, v, gap):
    n, cross, fill, B, Bmax = ring(u, v)
    dem = cross + fill
    D = max(d for _, _, d in dem)
    phi = [Bmax - b for b in B]
    split = [(s, t, a, True) for (s, t, _), a in zip(cross, u)] + [(s, t, b, False) for (s, t, _), b in zip(cross, v)] \
        + [(s, t, d, True) for s, t, d in fill]
    Ls = load_of(n, split)
    cb, arg = cut_bound(n, dem)
    two = sorted({s for s, _, _ in fill if sum(1 for f in fill if f[0] == s) > 1})
    print(f"[{tag}] ring: n={n} nodes, {len(dem)} demands ({len(cross)} crossing + {len(fill)} fillers), D={D}")
    print(f"[{tag}]   crossing demands of size D: {[i + 1 for i, (_, _, d) in enumerate(cross) if d == D]};"
          f" filler total sum(phi) = {sum(phi)}, max phi_e = {max(phi)}, edges with no filler {[e for e in range(n) if phi[e] == 0]}")
    print(f"[{tag}]   edges carrying two filler pieces (same node pair): {len(two)} -> {two}")
    print(f"[{tag}]   split routing max load = {max(Ls)} (all edges equal: {len(set(Ls)) == 1});"
          f" cut bound = {cb} at edges {arg[:2]}, separated demand {arg[2]}  => L* = {max(Ls) if cb == max(Ls) else '?'}")
    t0 = time.time()
    np_, wit = unsplit_search(n, cross, fill, Bmax + gap, strict=True)
    print(f"[{tag}]   max load < L* + {gap}: {np_} of {2 ** len(cross)} crossing choices pass the necessary condition;"
          f" exact DP finds {'a routing' if wit else 'none'}  ({time.time() - t0:.1f} s)")
    if wit:
        print(f"[{tag}]   -> NOT lossless: a routing has exact max load {max(load_of(n, wit[2]))} = L* + {max(load_of(n, wit[2])) - Bmax}")
        return None
    np2, wit2 = unsplit_search(n, cross, fill, Bmax + gap, strict=False)
    L = load_of(n, wit2[2])
    cwd = [i + 1 for i in range(len(cross)) if (wit2[0] >> i) & 1]
    print(f"[{tag}]   witness: crossing demands routed clockwise {cwd}, the rest counterclockwise;"
          f" rerouted filler total {sum(wit2[1])}, exact max load {max(L)} = L* + {max(L) - Bmax}")
    return n, cross, fill, Bmax, wit2


def subdivide(u, v, n, cross, fill, Bmax, wit2, out):
    """Paper Lemma 8: on every edge carrying two filler pieces, insert a node x_e and move the remainder piece to
    (e, x_e).  Checks that all node pairs are distinct, that the split routing sigma (fillers short) has max load Bmax,
    and that the witness routing (its fillers are all short) keeps its max load.  Lemma 8(a) gives the matching lower
    bounds.  Writes the instance to `out`."""
    two = {e for e in range(n) if sum(1 for f in fill if f[0] == e) > 1}
    pos, N = {}, 0
    for e in range(n):
        pos[e] = N; N += 1
        if e in two:
            pos[("x", e)] = N; N += 1
    cr = [(pos[s], pos[t], d) for s, t, d in cross]
    fl, seen = [], set()
    for s, t, d in fill:
        if s in seen:                                                # second piece on this edge -> (e, x_e)
            fl.append((pos[s], pos[("x", s)], d))
        else:
            fl.append((pos[s], pos[t] if t != 0 else 0, d)); seen.add(s)
    pairs = {frozenset((s, t)) for s, t, _ in cr + fl}
    assert len(pairs) == len(cr) + len(fl), "duplicate node pair"
    sigma = [(s, t, a, True) for (s, t, _), a in zip(cr, u)] + [(s, t, b, False) for (s, t, _), b in zip(cr, v)] \
        + [(s, t, d, True) for s, t, d in fl]
    assert all(r[3] for r in wit2[2][len(cross):]), "witness reroutes a filler"
    wit = [(s, t, d, wit2[2][i][3]) for i, (s, t, d) in enumerate(cr)] + [(s, t, d, True) for s, t, d in fl]
    Ls, Lw = max(load_of(N, sigma)), max(load_of(N, wit))
    print(f"[sub] subdivided {len(two)} edges -> {N} nodes, {len(pairs)} demands on {len(pairs)} distinct node pairs,"
          f" D = {max(d for *_, d in cr + fl)}; split routing max load {Ls}; witness max load {Lw}")
    json.dump({"n": N, "demands": [[s, t, str(d)] for s, t, d in cr + fl], "L_star": str(Ls), "L": str(Lw),
               "note": "nodes 0..n-1 on a cycle, edge j joins j and j+1 (mod n); see paper Lemma 8 / Corollary 9"},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    for name in ("instance_m15_den16_a.json", "instance_m15_den16_b.json", "instance_m16_den48.json",
                 "alphaevolve_47_42_polished.json"):
        u, v = load(name)
        m = len(u)
        a1, z1 = seq_alpha(u, v, set(range(1, m)))
        a2, _ = seq_alpha(u, v, set(range(0, m + 1)))
        print(f"{name}: m={m}, largest denominator {max(x.denominator for x in u + v)}, zeros = {sum(x == 0 for x in u + v)};"
              f" alpha(K={{1..m-1}}) = {a1}, alpha(K={{0..m}}) = {a2}; a minimising z takes v_k for k in"
              f" {[i + 1 for i in range(m) if (z1 >> i) & 1]}")
        if name == "instance_m15_den16_a.json":                      # Table 1 of the paper
            Q = 16; z = [v[i] if (z1 >> i) & 1 else -u[i] for i in range(m)]; S = sum(z)
            print("   Table 1 Delta_k (k=1..m):", [str(2 * sum(z[:k]) - S) for k in range(1, m + 1)])
    u, v = load("instance_m15_den16_a.json")
    res = report_ring("9/8", u, v, F(9, 8))
    subdivide(u, v, *res, os.path.join(INST, "ring_instance_distinct_pairs_n43.json"))
    for name in ("instance_m15_den16_b.json", "instance_m16_den48.json"):
        report_ring(name[9:-5], *load(name), F(9, 8))
    u, v = load("alphaevolve_47_42_polished.json")
    report_ring("AE-polished", u, v, F(23, 21))
    print(f"total {time.time() - t0:.0f} s")
