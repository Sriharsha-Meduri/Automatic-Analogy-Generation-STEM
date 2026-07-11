# -*- coding: utf-8 -*-
"""RQ2: can we predict an analogy's judged quality from cheap, interpretable
features (strategy, subject, length, readability, concreteness proxies, simile
markers)? This module extracts features from an analogy string and, when run as
a script, joins them to the judge scores and fits an interpretable predictor,
reporting cross-validated performance and feature importances.

Dependency-free feature extraction; predictor uses scikit-learn if available,
else falls back to a numpy least-squares fit."""
import json, os, re, math
from collections import defaultdict

ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"

VOWELS = "aeiouy"
CONNECTIVES = ("like", "as", "just as", "similar", "imagine", "think of",
               "because", "so that", "whereas", "while")

def _syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)

def flesch_reading_ease(text):
    words = re.findall(r"[A-Za-z]+", text)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words or not sents:
        return None
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syl / len(words))

def extract_features(analogy, strategy=None, subject=None, grade=None):
    """Return a flat dict of numeric/categorical features for one analogy."""
    text = analogy or ""
    words = re.findall(r"[A-Za-z']+", text)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    low = text.lower()
    n_words = len(words)
    feats = {
        "n_words": n_words,
        "n_sentences": len(sents),
        "words_per_sentence": (n_words / len(sents)) if sents else 0.0,
        "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "flesch": flesch_reading_ease(text) or 0.0,
        "has_number": int(bool(re.search(r"\d", text))),
        "n_commas": text.count(","),
        "pct_long_words": (sum(1 for w in words if len(w) >= 7) / n_words) if n_words else 0.0,
        "n_connectives": sum(low.count(c) for c in CONNECTIVES),
        "has_simile_marker": int(bool(re.search(r"\b(like|as if|just as|similar to)\b", low))),
        "second_person": int(bool(re.search(r"\b(you|your|imagine)\b", low))),
    }
    if strategy is not None:
        feats["strategy"] = strategy
    if subject is not None:
        feats["subject"] = subject
    if grade is not None:
        feats["grade"] = grade
    return feats

# ------------------------- script: fit the predictor -------------------------

def _load(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def _load_jsonl(path):
    p = os.path.join(ROOT, path)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def build_table(gen_glob="data/gen_*.json", judg="results/judgments.jsonl"):
    import glob
    judgments = _load_jsonl(judg)
    if not judgments:
        return [], []
    # mean overall_usefulness (target) per (generator? we keep it simple: per concept_id+strategy)
    scores = defaultdict(list)
    for j in judgments:
        s = (j.get("scores") or {}).get("overall_usefulness")
        if s is not None:
            scores[(j["concept_id"], j["strategy"])].append(s)
    rows, targets = [], []
    for gp in glob.glob(os.path.join(ROOT, gen_glob)):
        for c in json.load(open(gp, encoding="utf-8")):
            for strat, analogy in c.get("analogies", {}).items():
                key = (c["id"], strat)
                if key not in scores:
                    continue
                f = extract_features(analogy, strat, c["subject"], c["grade"])
                rows.append(f); targets.append(sum(scores[key]) / len(scores[key]))
    return rows, targets

def _design_matrix(rows):
    cat = ["strategy", "subject"]
    num = [k for k in rows[0] if k not in cat]
    levels = {c: sorted({r[c] for r in rows if c in r}) for c in cat}
    names = list(num)
    for c in cat:
        names += [f"{c}={v}" for v in levels[c]]
    X = []
    for r in rows:
        row = [float(r.get(k, 0.0)) for k in num]
        for c in cat:
            row += [1.0 if r.get(c) == v else 0.0 for v in levels[c]]
        X.append(row)
    return X, names

def main():
    rows, y = build_table()
    if not rows:
        print("no analogies+judgments to fit yet (run generation + judges first)"); return
    print(f"RQ2 predictor: {len(rows)} analogies, target=mean overall_usefulness")
    X, names = _design_matrix(rows)
    import numpy as np
    X = np.array(X); y = np.array(y)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.linear_model import Ridge
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        rf = RandomForestRegressor(n_estimators=300, random_state=0)
        r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
        mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
        rf.fit(X, y)
        imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])
        print(f"  RandomForest 5-fold R2={r2:.3f}  MAE={mae:.3f}")
        print("  top feature importances:")
        for nm, v in imp[:12]:
            print(f"    {nm:22s} {v:.3f}")
        rg = Ridge(alpha=1.0).fit((X - X.mean(0)) / (X.std(0) + 1e-9), y)
        coef = sorted(zip(names, rg.coef_), key=lambda t: -abs(t[1]))
        print("  standardized Ridge coefficients (sign = direction):")
        for nm, v in coef[:12]:
            print(f"    {nm:22s} {v:+.3f}")
    except ImportError:
        # numpy-only OLS fallback
        Xb = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        yhat = Xb @ beta
        ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
        print(f"  OLS in-sample R2={1 - ss_res/ss_tot:.3f} (install scikit-learn for CV + importances)")
        for nm, b in sorted(zip(names, beta[1:]), key=lambda t: -abs(t[1]))[:12]:
            print(f"    {nm:22s} {b:+.3f}")

if __name__ == "__main__":
    main()
