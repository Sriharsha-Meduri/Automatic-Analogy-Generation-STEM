# -*- coding: utf-8 -*-
"""Analyse comprehension + judge results: condition accuracies, analogy-vs-standard
gains with paired significance tests, per-strategy rankings, inter-judge
Krippendorff alpha, and judge-vs-comprehension correlation."""
import json, os, math
from collections import defaultdict
import numpy as np
from scipy import stats
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"

def load_jsonl(p):
    p = os.path.join(ROOT, p)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def krippendorff_alpha(mat, level="interval"):
    """mat: coders x units, np.nan for missing."""
    data = np.asarray(mat, dtype=float)
    units = [data[:, j][~np.isnan(data[:, j])] for j in range(data.shape[1])]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    dfun = (lambda a, b: (a - b) ** 2) if level in ("interval", "ordinal") else (lambda a, b: float(a != b))
    Do_num, n_total = 0.0, 0
    for u in units:
        m = len(u)
        Do_num += sum(dfun(a, b) for a in u for b in u) / (m - 1)
        n_total += m
    Do = Do_num / n_total
    allv = np.concatenate(units); N = len(allv)
    De = sum(dfun(a, b) for a in allv for b in allv) / (N * (N - 1))
    return 1 - Do / De if De > 0 else float("nan")

def analyse_comprehension():
    rows = load_jsonl("results/comprehension.jsonl")
    if not rows:
        print("no comprehension data yet"); return None
    # accuracy per (concept, condition) averaged over models/questions/shuffles
    by_cc = defaultdict(list)
    for r in rows:
        if r["pick"] is not None:
            by_cc[(r["concept_id"], r["condition"])].append(r["is_correct"])
    concepts = sorted(set(k[0] for k in by_cc))
    conditions = sorted(set(k[1] for k in by_cc))
    print(f"\n=== COMPREHENSION ({len(rows)} trials, {len(concepts)} concepts, models={sorted(set(r['model'] for r in rows))}) ===")
    cond_acc = {}
    for cond in conditions:
        vals = [np.mean(by_cc[(c, cond)]) for c in concepts if (c, cond) in by_cc]
        cond_acc[cond] = (np.mean(vals), np.std(vals), len(vals))
        print(f"  {cond:22s} acc={np.mean(vals)*100:5.1f}%  (n_concepts={len(vals)})")
    # analogy gain over standard, paired per concept
    print("\n  -- analogy strategies vs standard (paired over concepts) --")
    strat_results = {}
    for cond in conditions:
        if not cond.startswith("analogy:"):
            continue
        pairs = [(np.mean(by_cc[(c, cond)]), np.mean(by_cc[(c, "standard")]))
                 for c in concepts if (c, cond) in by_cc and (c, "standard") in by_cc]
        gains = [a - s for a, s in pairs]
        if len(gains) >= 5:
            try:
                w, p = stats.wilcoxon([a for a, s in pairs], [s for a, s in pairs])
            except Exception:
                w, p = float("nan"), float("nan")
        else:
            w, p = float("nan"), float("nan")
        strat_results[cond] = {"mean_gain": float(np.mean(gains)), "n": len(gains), "wilcoxon_p": float(p)}
        print(f"  {cond:22s} mean gain over standard = {np.mean(gains)*100:+5.1f} pts  (p={p:.3f}, n={len(gains)})")
    return {"cond_acc": {k: v[0] for k, v in cond_acc.items()}, "strategies": strat_results, "concepts": concepts}

def _holm(pvals):
    """Holm-Bonferroni corrected p-values, preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda k: (pvals[k] if pvals[k] == pvals[k] else 9))
    out = [float("nan")] * m
    running = 0.0
    for rank, k in enumerate(order):
        if pvals[k] != pvals[k]:
            continue
        running = max(running, min(1.0, pvals[k] * (m - rank)))
        out[k] = running
    return out

def analyse_judges(comp=None):
    import itertools
    rows = load_jsonl("results/judgments.jsonl")
    if not rows:
        print("\nno judge data yet"); return None
    judges = sorted(set(r["judge"] for r in rows))
    gens = sorted(set(r.get("generator") for r in rows))
    dims = ["conceptual_accuracy", "clarity", "memorability", "non_misleading", "overall_usefulness"]
    strategies = ["free", "everyday", "sports", "cooking"]
    print(f"\n=== JUDGE PANEL ({len(rows)} judgments, judges={judges}, generators={gens}) ===")

    # inter-judge Krippendorff alpha per dimension (units = generator|concept|strategy)
    items = sorted(set((r.get("generator"), r["concept_id"], r["strategy"]) for r in rows))
    idx = {it: i for i, it in enumerate(items)}
    print("  inter-judge Krippendorff alpha (interval, self-judgment excluded):")
    for d in dims:
        mat = np.full((len(judges), len(items)), np.nan)
        for r in rows:
            s = (r["scores"] or {}).get(d)
            if s is not None:
                mat[judges.index(r["judge"]), idx[(r.get("generator"), r["concept_id"], r["strategy"])]] = s
        print(f"    {d:22s} alpha = {krippendorff_alpha(mat, 'interval'):.3f}")

    # per-dimension mean by strategy
    print("\n  mean rubric scores by strategy (1-5, pooled over judges/generators/concepts):")
    print("    " + " " * 10 + "".join(f"{d[:10]:>12s}" for d in dims))
    by_sd = {st: {d: [] for d in dims} for st in strategies}
    for r in rows:
        if r["strategy"] not in by_sd:
            continue
        for d in dims:
            s = (r["scores"] or {}).get(d)
            if s is not None:
                by_sd[r["strategy"]][d].append(s)
    for st in strategies:
        vals = "".join(f"{np.mean(by_sd[st][d]):>12.2f}" if by_sd[st][d] else f"{'-':>12s}" for d in dims)
        print(f"    {st:10s}{vals}")

    # Friedman omnibus across strategies on overall_usefulness (aggregated per concept)
    cs = defaultdict(list)
    for r in rows:
        s = (r["scores"] or {}).get("overall_usefulness")
        if s is not None:
            cs[(r["concept_id"], r["strategy"])].append(s)
    concepts = sorted(set(k[0] for k in cs))
    complete = [c for c in concepts if all((c, st) in cs for st in strategies)]
    if len(complete) >= 5:
        cols = [[float(np.mean(cs[(c, st)])) for c in complete] for st in strategies]
        try:
            chi, p = stats.friedmanchisquare(*cols)
            print(f"\n  Friedman across strategies (overall_usefulness, n={len(complete)} concepts): "
                  f"chi2={chi:.2f}, p={p:.4f}")
        except Exception as e:
            print(f"\n  Friedman failed: {e}")
        pairs = list(itertools.combinations(range(len(strategies)), 2))
        praw = []
        for i, j in pairs:
            try:
                _, pp = stats.wilcoxon(cols[i], cols[j])
            except Exception:
                pp = float("nan")
            praw.append(pp)
        for (i, j), hp in zip(pairs, _holm(praw)):
            print(f"    {strategies[i]:>9s} vs {strategies[j]:<9s} p_holm={hp:.3f}")

    # overall ranking + per-subject best strategy
    by_strat = defaultdict(list)
    subj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = (r["scores"] or {}).get("overall_usefulness")
        if s is not None:
            by_strat[r["strategy"]].append(s)
            subj[r["subject"]][r["strategy"]].append(s)
    print("\n  overall usefulness ranking:")
    for st in sorted(by_strat, key=lambda x: -np.mean(by_strat[x])):
        print(f"    {st:10s} {np.mean(by_strat[st]):.2f}  (n={len(by_strat[st])})")
    print("  best strategy per subject:")
    for sb in sorted(subj):
        means = {st: np.mean(v) for st, v in subj[sb].items() if v}
        best = max(means, key=means.get)
        alls = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(means.items(), key=lambda t: -t[1]))
        print(f"    {sb:12s} best={best:9s}  [{alls}]")

def analyse_rankings():
    rows = [r for r in load_jsonl("results/rankings.jsonl") if r.get("ranks")]
    if not rows:
        print("\nno ranking data yet"); return None
    strategies = ["free", "everyday", "sports", "cooking"]
    judges = sorted(set(r["judge"] for r in rows))
    gens = sorted(set(r.get("generator") for r in rows))
    print(f"\n=== COMPARATIVE RANKINGS ({len(rows)} valid rankings, judges={judges}, generators={gens}) ===")
    # mean rank per strategy (1=best, 4=worst)
    by = defaultdict(list)
    for r in rows:
        for st, rk in r["ranks"].items():
            by[st].append(rk)
    print("  mean rank per strategy (1=best):")
    for st in sorted(strategies, key=lambda s: np.mean(by[s]) if by[s] else 9):
        if by[st]:
            print(f"    {st:10s} {np.mean(by[st]):.2f}  (top-1 rate {np.mean([x==1 for x in by[st]])*100:4.0f}%, n={len(by[st])})")
    # Friedman across strategies (per concept, mean rank over judges+generators)
    cs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for st, rk in r["ranks"].items():
            cs[r["concept_id"]][st].append(rk)
    concepts = [c for c in cs if all(st in cs[c] for st in strategies)]
    if len(concepts) >= 5:
        cols = [[np.mean(cs[c][st]) for c in concepts] for st in strategies]
        try:
            chi, p = stats.friedmanchisquare(*cols)
            print(f"  Friedman across strategies (n={len(concepts)} concepts): chi2={chi:.2f}, p={p:.4f}")
        except Exception as e:
            print(f"  Friedman failed: {e}")
    # inter-judge agreement: mean pairwise Spearman on the 4-strategy rank vectors, per shared item
    from itertools import combinations
    groups = defaultdict(dict)
    for r in rows:
        groups[(r.get("generator"), r["concept_id"])][r["judge"]] = [r["ranks"].get(st) for st in strategies]
    corrs = []
    for item, jr in groups.items():
        for j1, j2 in combinations(sorted(jr), 2):
            a, b = jr[j1], jr[j2]
            if None not in a and None not in b:
                rho = stats.spearmanr(a, b).correlation
                if rho == rho:
                    corrs.append(rho)
    if corrs:
        print(f"  inter-judge agreement: mean pairwise Spearman = {np.mean(corrs):.3f} (over {len(corrs)} judge-pairs x items)")
    # per-subject best strategy
    subj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for st, rk in r["ranks"].items():
            subj[r["subject"]][st].append(rk)
    print("  best strategy per subject (lowest mean rank):")
    for sb in sorted(subj):
        means = {st: np.mean(v) for st, v in subj[sb].items() if v}
        best = min(means, key=means.get)
        alls = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(means.items(), key=lambda t: t[1]))
        print(f"    {sb:12s} best={best:9s}  [{alls}]")
    return {"strategies": strategies, "mean_rank": {s: float(np.mean(by[s])) for s in strategies if by[s]}}

def analyse_probe():
    rows = [r for r in load_jsonl("results/rank_probe.jsonl") if r.get("ranks")]
    if not rows:
        return None
    print(f"\n=== JUDGE-VALIDATION PROBE ({len(rows)} rankings) ===")
    tiers = ["good", "decent", "vague", "wrong"]
    by = defaultdict(list)
    for r in rows:
        for t, rk in r["ranks"].items():
            by[t].append(rk)
    print("  mean rank per planted quality tier (1=best; expect good<decent<vague<wrong):")
    for t in tiers:
        if by[t]:
            print(f"    {t:8s} {np.mean(by[t]):.2f}")
    # per-judge discrimination: does the judge rank 'wrong' worse than 'good'?
    print("  per-judge discrimination (fraction with rank(wrong) > rank(good)):")
    for j in sorted(set(r["judge"] for r in rows)):
        jr = [r for r in rows if r["judge"] == j]
        disc = [r["ranks"].get("wrong", 0) > r["ranks"].get("good", 9) for r in jr]
        print(f"    {j:14s} {np.mean(disc)*100:4.0f}%  (n={len(jr)})")

if __name__ == "__main__":
    comp = analyse_comprehension()
    analyse_judges(comp)
    analyse_rankings()
    analyse_probe()
    print("\ndone.")
