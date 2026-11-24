"""Figure 1 of the note: the 30-node ring instance.  python -I paper/make_figure.py -> paper/fig_ring.pdf"""
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from certificate import load, ring  # noqa: E402

CROSS, FILL, INK, MUTED = "#1f5fa8", "#c05a00", "#1a1a1a", "#6b6b6b"
u, v = load("instance_m15_den16_a.json")
n, cross, fill, B, Bmax = ring(u, v)
m = n // 2
phi = [Bmax - b for b in B]                                   # filler amount on each edge

fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.set_aspect("equal"); ax.axis("off")
ang = lambda j: math.pi / 2 - 2 * math.pi * j / n           # node 0 at the top, clockwise
P = [(math.cos(ang(j)), math.sin(ang(j))) for j in range(n)]
for e in range(n):                                            # ring edges; width = filler amount on the edge
    a0, a1 = ang(e), ang(e + 1)
    ts = [a0 + (a1 - a0) * k / 16 for k in range(17)]
    ax.plot([math.cos(t) for t in ts], [math.sin(t) for t in ts], color=FILL, lw=0.8 + 2.4 * float(phi[e]),
            solid_capstyle="butt")
for i, (s, t, d) in enumerate(cross):                         # crossing demands; width = size
    ax.plot([P[s][0], P[t][0]], [P[s][1], P[t][1]], color=CROSS, lw=0.5 + 1.6 * float(d), alpha=0.9)
ax.scatter([p[0] for p in P], [p[1] for p in P], s=9, color=INK, zorder=5)
for j in (0, m // 2, m, 3 * m // 2 + 1):
    ax.text(1.13 * P[j][0], 1.13 * P[j][1], str(j), ha="center", va="center", fontsize=8, color=MUTED)
for e in (0, m):                                              # the cut used in Lemma 2
    a = ang(e + 0.5)
    ax.plot([0.9 * math.cos(a), 1.1 * math.cos(a)], [0.9 * math.sin(a), 1.1 * math.sin(a)], color=INK, lw=1.0, ls="--")
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(HERE, "fig_ring.pdf"))
print("ok", [str(x) for x in phi[:3]], "max filler on an edge", max(phi))
