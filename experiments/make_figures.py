# -*- coding: utf-8 -*-
"""Publication figures for the analogy study. Reads the result files and writes
PNGs to figures/. Each figure is guarded so it is skipped if its data is absent.
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
FIGDIR = os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from analyze import krippendorff_alpha  # reuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
STRAT = ["free", "everyday", "sports", "cooking"]
DIMS = ["conceptual_accuracy", "clarity", "memorability", "non_misleading", "overall_usefulness"]
DIMLAB = ["Conceptual\naccuracy", "Clarity", "Memorability", "Non-\nmisleading", "Overall\nusefulness"]
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

def _jsonl(p):
    p = os.path.join(ROOT, p)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def _load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def _ci95(vals):
    a = np.array(vals, dtype=float)
    if len(a) < 2:
        return 0.0
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))

def fig_strategy_usefulness(J):
    by = defaultdict(list)
    for r in J:
        s = (r["scores"] or {}).get("overall_usefulness")
        if s is not None:
            by[r["strategy"]].append(s)
    if not by:
        return
    means = [np.mean(by[s]) for s in STRAT]
    errs = [_ci95(by[s]) for s in STRAT]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(STRAT, means, yerr=errs, capsize=5, color=PAL[:4], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean overall usefulness (1-5)")
    ax.set_xlabel("Analogy generation strategy")
    ax.set_title("Judged usefulness by prompting strategy")
    ax.set_ylim(1, 5)
    for i, m in enumerate(means):
        ax.text(i, m + errs[i] + 0.05, f"{m:.2f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq1_strategy_usefulness.png")); plt.close(fig)
    print("  wrote rq1_strategy_usefulness.png")

def fig_dimension_heatmap(J):
    M = np.full((len(STRAT), len(DIMS)), np.nan)
    for si, st in enumerate(STRAT):
        for di, d in enumerate(DIMS):
            vals = [(r["scores"] or {}).get(d) for r in J if r["strategy"] == st]
            vals = [v for v in vals if v is not None]
            if vals:
                M[si, di] = np.mean(vals)
    if np.isnan(M).all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(DIMLAB, fontsize=9)
    ax.set_yticks(range(len(STRAT))); ax.set_yticklabels(STRAT)
    for i in range(len(STRAT)):
        for j in range(len(DIMS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] > 3.2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean score (1-5)")
    ax.set_title("Rubric dimension scores by strategy")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq1_dimension_heatmap.png")); plt.close(fig)
    print("  wrote rq1_dimension_heatmap.png")

def fig_interjudge_alpha(J):
    judges = sorted(set(r["judge"] for r in J))
    items = sorted(set((r.get("generator"), r["concept_id"], r["strategy"]) for r in J))
    idx = {it: i for i, it in enumerate(items)}
    alphas = []
    for d in DIMS:
        mat = np.full((len(judges), len(items)), np.nan)
        for r in J:
            s = (r["scores"] or {}).get(d)
            if s is not None:
                mat[judges.index(r["judge"]), idx[(r.get("generator"), r["concept_id"], r["strategy"])]] = s
        alphas.append(krippendorff_alpha(mat, "interval"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(DIMS)), alphas, color=PAL[4], edgecolor="black", linewidth=0.6)
    ax.axhline(0.667, ls="--", color="gray", lw=1); ax.text(len(DIMS)-0.5, 0.68, "0.67", color="gray", fontsize=8)
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(DIMLAB, fontsize=9)
    ax.set_ylabel("Krippendorff's alpha"); ax.set_ylim(min(0, min(alphas)-0.05), 1)
    ax.set_title("Inter-judge agreement by rubric dimension")
    for i, a in enumerate(alphas):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq1_interjudge_alpha.png")); plt.close(fig)
    print("  wrote rq1_interjudge_alpha.png")

def fig_generator_comparison(J):
    by = defaultdict(list)
    for r in J:
        s = (r["scores"] or {}).get("overall_usefulness")
        if s is not None:
            by[r.get("generator")].append(s)
    if len(by) < 2:
        return
    gens = sorted(by)
    means = [np.mean(by[g]) for g in gens]; errs = [_ci95(by[g]) for g in gens]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(gens, means, yerr=errs, capsize=5, color=PAL[:len(gens)], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean overall usefulness (1-5)"); ax.set_ylim(0, 3)
    ax.set_title("Analogy usefulness by generator model")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for i, m in enumerate(means):
        ax.text(i, m + errs[i] + 0.05, f"{m:.2f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq1_generator_comparison.png")); plt.close(fig)
    print("  wrote rq1_generator_comparison.png")

def fig_subject_strategy(J):
    subj = sorted(set(r["subject"] for r in J))
    if not subj:
        return
    data = {st: [] for st in STRAT}
    for st in STRAT:
        for sb in subj:
            vals = [(r["scores"] or {}).get("overall_usefulness") for r in J
                    if r["strategy"] == st and r["subject"] == sb]
            vals = [v for v in vals if v is not None]
            data[st].append(np.mean(vals) if vals else 0)
    x = np.arange(len(subj)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, st in enumerate(STRAT):
        ax.bar(x + (i - 1.5) * w, data[st], w, label=st, color=PAL[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(subj); ax.set_ylabel("Mean overall usefulness (0-3)")
    ax.set_ylim(1, 5); ax.legend(title="Strategy", ncol=4, fontsize=9)
    ax.set_title("Strategy usefulness by subject")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq1_subject_strategy.png")); plt.close(fig)
    print("  wrote rq1_subject_strategy.png")

def fig_capability_gradient(C):
    # accuracy per student model, grouped condition (baseline / standard / analogy)
    def grp(cond):
        return "analogy" if cond.startswith("analogy") else cond
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in C:
        if r["pick"] is None:
            continue
        g = grp(r["condition"]); agg[r["model"]][g][0] += r["is_correct"]; agg[r["model"]][g][1] += 1
    if not agg:
        return
    # order students by a rough capability proxy embedded in the name
    order = ["qwen2.5:0.5b", "llama3.2:1b", "gemma2:2b", "llama3.2:3b", "qwen2.5:3b"]
    models = [m for m in order if m in agg] + [m for m in agg if m not in order]
    conds = ["baseline", "standard", "analogy"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ci, cond in enumerate(conds):
        ys = [100 * agg[m][cond][0] / agg[m][cond][1] if agg[m][cond][1] else np.nan for m in models]
        ax.plot(range(len(models)), ys, marker="o", label=cond, color=PAL[ci], linewidth=2)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MCQ accuracy (%)"); ax.set_title("Comprehension by student capability and teaching condition")
    ax.legend(title="Condition"); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq3_capability_gradient.png")); plt.close(fig)
    print("  wrote rq3_capability_gradient.png")

def fig_feature_importance():
    try:
        from features import build_table, _design_matrix
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        print(f"  (skip feature importance: {e})"); return
    rows, y = build_table()
    if not rows:
        return
    X, names = _design_matrix(rows)
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(np.array(X), np.array(y))
    imp = sorted(zip(names, rf.feature_importances_), key=lambda t: t[1])[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([n for n, _ in imp], [v for _, v in imp], color=PAL[0], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Random-forest importance"); ax.set_title("RQ2: features predicting analogy usefulness")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "rq2_feature_importance.png")); plt.close(fig)
    print("  wrote rq2_feature_importance.png")

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    J = _jsonl("results/judgments.jsonl")
    C = _jsonl("results/comprehension.jsonl")
    print(f"figures: {len(J)} judgments, {len(C)} comprehension trials")
    if J:
        fig_strategy_usefulness(J); fig_dimension_heatmap(J); fig_interjudge_alpha(J)
        fig_generator_comparison(J); fig_subject_strategy(J)
    if C:
        fig_capability_gradient(C)
    fig_feature_importance()
    print("done.")

if __name__ == "__main__":
    main()
