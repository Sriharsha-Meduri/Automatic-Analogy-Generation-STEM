# -*- coding: utf-8 -*-
"""Forest plot of the pairwise strategy mean-rank differences with 95% CIs and the TOST
equivalence band, matching the paper's existing figure style (see make_figures.py)."""
import json, os, itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
STR = ["free", "everyday", "sports", "cooking"]
MARGIN = 0.30

rows = [json.loads(l) for l in open(ROOT + "/results/rankings.jsonl", encoding="utf-8") if json.loads(l).get("ranks")]
cs = defaultdict(lambda: defaultdict(list))
for r in rows:
    for st, rk in r["ranks"].items():
        cs[r["concept_id"]][st].append(rk)
concepts = [c for c in cs if all(st in cs[c] for st in STR)]
M = {st: np.array([np.mean(cs[c][st]) for c in concepts]) for st in STR}
n = len(concepts)

pairs = list(itertools.combinations(STR, 2))
labels, mds, los, his = [], [], [], []
for a, b in pairs:
    d = M[a] - M[b]; md = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    labels.append(f"{a} − {b}"); mds.append(md); los.append(md - 1.96 * se); his.append(md + 1.96 * se)

fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(pairs))[::-1]
ax.axvspan(-MARGIN, MARGIN, color="#55A868", alpha=0.12, label=f"equivalence band ($\\pm${MARGIN} rank)")
ax.axvline(0, color="gray", lw=1, ls="--")
for i, (yy, md, lo, hi) in enumerate(zip(y, mds, los, his)):
    ax.plot([lo, hi], [yy, yy], color=PAL[0], lw=2)
    ax.plot(md, yy, "o", color=PAL[0], ms=7)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Difference in mean rank (95% CI); negative = left strategy ranks better")
ax.set_title("RQ1: all strategy pairs are equivalent within a small margin")
ax.set_xlim(-0.4, 0.4); ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT + "/figures/rq1_equivalence.png"); plt.close(fig)
print("wrote figures/rq1_equivalence.png (n=%d concepts)" % n)
