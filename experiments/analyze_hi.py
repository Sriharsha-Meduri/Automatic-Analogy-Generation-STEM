# -*- coding: utf-8 -*-
"""Cross-lingual analysis (RQ4): does the English picture replicate in Hindi, and
how far do small open models degrade in a mid-resource language? Reports, for each
language, parse-success, mean rank per strategy, ranking Friedman + inter-judge
Spearman, probe discrimination, absolute-score leniency, and the length/score
(verbosity) correlation, then prints an English-vs-Hindi comparison."""
import json, os, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy import stats
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
STRATS = ["free", "everyday", "sports", "cooking"]
DIMS = ["conceptual_accuracy", "clarity", "memorability", "non_misleading", "overall_usefulness"]

def jl(p):
    p = os.path.join(ROOT, p)
    out = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out

def jf(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def rank_stats(rankfile):
    rows = [r for r in jl(rankfile) if r.get("ranks")]
    total = len(jl(rankfile))
    if not rows:
        return None
    by = defaultdict(list)
    for r in rows:
        for st, rk in r["ranks"].items():
            by[st].append(rk)
    cs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for st, rk in r["ranks"].items():
            cs[r["concept_id"]][st].append(rk)
    concepts = [c for c in cs if all(st in cs[c] for st in STRATS)]
    friedman = None
    if len(concepts) >= 5:
        cols = [[np.mean(cs[c][st]) for c in concepts] for st in STRATS]
        try:
            chi, p = stats.friedmanchisquare(*cols); friedman = (chi, p)
        except Exception:
            pass
    groups = defaultdict(dict)
    for r in rows:
        groups[(r.get("generator"), r["concept_id"])][r["judge"]] = [r["ranks"].get(st) for st in STRATS]
    corrs = []
    for _, jr in groups.items():
        for j1, j2 in combinations(sorted(jr), 2):
            a, b = jr[j1], jr[j2]
            if None not in a and None not in b:
                rho = stats.spearmanr(a, b).correlation
                if rho == rho:
                    corrs.append(rho)
    return {"parse": (len(rows), total), "mean_rank": {s: float(np.mean(by[s])) for s in STRATS if by[s]},
            "friedman": friedman, "spearman": float(np.mean(corrs)) if corrs else float("nan")}

def probe_stats(probefile):
    rows = [r for r in jl(probefile) if r.get("ranks")]
    if not rows:
        return None
    tiers = ["good", "decent", "vague", "wrong"]
    by = defaultdict(list)
    for r in rows:
        for t, rk in r["ranks"].items():
            by[t].append(rk)
    disc = np.mean([r["ranks"].get("wrong", 0) > r["ranks"].get("good", 9) for r in rows])
    return {"tiers": {t: float(np.mean(by[t])) for t in tiers if by[t]}, "wrong_below_good": float(disc), "n": len(rows)}

def score_stats(judgefile, genfile):
    rows = jl(judgefile)
    if not rows:
        return None
    parsed = sum(1 for r in rows if r.get("scores") and all(r["scores"].get(d) is not None for d in DIMS))
    by_strat = defaultdict(list)
    for r in rows:
        s = (r["scores"] or {}).get("overall_usefulness")
        if s is not None:
            by_strat[r["strategy"]].append(s)
    # verbosity: length vs overall_usefulness
    gens = jf(genfile) or []
    lens = {}
    for c in gens:
        for st, a in c.get("analogies", {}).items():
            lens[(c.get("generator"), c["id"], st)] = len((a or "").split())
    xs, ys = [], []
    for r in rows:
        ou = (r["scores"] or {}).get("overall_usefulness")
        k = (r.get("generator"), r["concept_id"], r["strategy"])
        if ou is not None and k in lens:
            xs.append(lens[k]); ys.append(ou)
    rho = stats.spearmanr(xs, ys).correlation if len(xs) > 5 else float("nan")
    meanlen = {s: float(np.mean([lens[k] for k in lens if k[2] == s])) for s in STRATS if any(k[2] == s for k in lens)}
    return {"parse": (parsed, len(rows)), "mean_overall": {s: float(np.mean(by_strat[s])) for s in STRATS if by_strat[s]},
            "len_score_spearman": float(rho), "mean_len": meanlen}

def report(lang, rankf, probef, judgef, genf):
    print(f"\n########## {lang} ##########")
    r = rank_stats(rankf)
    if r:
        pr = r["parse"]; print(f"ranking: parsed {pr[0]}/{pr[1]} ({100*pr[0]/pr[1]:.0f}%)")
        print("  mean rank per strategy:", {k: round(v, 2) for k, v in sorted(r["mean_rank"].items(), key=lambda t: t[1])})
        if r["friedman"]:
            print(f"  Friedman chi2={r['friedman'][0]:.2f} p={r['friedman'][1]:.3f}")
        print(f"  inter-judge Spearman={r['spearman']:.3f}")
    p = probe_stats(probef)
    if p:
        print(f"probe (n={p['n']}): tiers={ {k: round(v,2) for k,v in p['tiers'].items()} } wrong>good={100*p['wrong_below_good']:.0f}%")
    s = score_stats(judgef, genf)
    if s:
        ps = s["parse"]; print(f"absolute: parsed {ps[0]}/{ps[1]} ({100*ps[0]/ps[1]:.0f}%)")
        print("  mean overall by strategy:", {k: round(v, 2) for k, v in sorted(s["mean_overall"].items(), key=lambda t: -t[1])})
        print(f"  length-score Spearman (verbosity bias)={s['len_score_spearman']:.3f}")
        print("  mean analogy length (words):", {k: round(v) for k, v in s["mean_len"].items()})
    return {"rank": r, "probe": p, "score": s}

if __name__ == "__main__":
    en = report("ENGLISH", "results/rankings.jsonl", "results/rank_probe.jsonl",
                "results/judgments.jsonl", "data/gen_all.json")
    hi = report("HINDI", "results/rankings_hi.jsonl", "results/rank_probe_hi.jsonl",
                "results/judgments_hi.jsonl", "data/gen_hi_all.json")
    print("\n########## ENGLISH vs HINDI (replication) ##########")
    if en["score"] and hi["score"]:
        print(f"  absolute parse rate:  en {100*en['score']['parse'][0]/en['score']['parse'][1]:.0f}%   hi {100*hi['score']['parse'][0]/hi['score']['parse'][1]:.0f}%")
        print(f"  verbosity (len-score rho): en {en['score']['len_score_spearman']:+.2f}   hi {hi['score']['len_score_spearman']:+.2f}")
    if en["rank"] and hi["rank"]:
        print(f"  inter-judge Spearman: en {en['rank']['spearman']:.2f}   hi {hi['rank']['spearman']:.2f}")
    if en["probe"] and hi["probe"]:
        print(f"  probe wrong>good:     en {100*en['probe']['wrong_below_good']:.0f}%   hi {100*hi['probe']['wrong_below_good']:.0f}%")
    print("\ndone.")
