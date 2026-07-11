# -*- coding: utf-8 -*-
"""RQ1 driver: generate analogies with several local models, merge, judge with the
full panel (self-judgment auto-skipped), then run analysis + the RQ2 predictor.
Every stage is resumable, so this can be re-run safely after interruptions.

Run in the background once all models are present:
  python experiments/run_rq1.py
"""
import json, os, sys, subprocess, time

ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
PY = sys.executable
CONCEPTS = "data/concepts_all.json"

# (generator model, output file). qwen file reuses the earlier partial run.
GENERATORS = [
    ("qwen2.5:3b",  "data/gen_qwen2.5-3b.json"),
    ("llama3.2:3b", "data/gen_llama3.2-3b.json"),
    ("gemma2:2b",   "data/gen_gemma2-2b.json"),
]
PANEL = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3.5"]

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t = time.time()
    r = subprocess.run([PY] + cmd, cwd=ROOT)
    print(f"<<< rc={r.returncode} in {time.time()-t:.0f}s", flush=True)
    if r.returncode != 0:
        print(f"!! stage failed: {cmd}", flush=True)
    return r.returncode

def main():
    # 1) generation
    for model, out in GENERATORS:
        run(["experiments/run_generate.py", "--concepts", CONCEPTS,
             "--gen-model", model, "--out-concepts", out,
             "--log", "results/generation_log.jsonl"])
    # 2) merge generator outputs into one judgeable file (tagged with generator)
    allc = []
    for model, out in GENERATORS:
        p = os.path.join(ROOT, out)
        if not os.path.exists(p):
            print(f"!! missing {out}, skipping in merge", flush=True); continue
        for c in json.load(open(p, encoding="utf-8")):
            c["generator"] = model
            allc.append(c)
    json.dump(allc, open(os.path.join(ROOT, "data/gen_all.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"merged {len(allc)} analogy-sets -> data/gen_all.json", flush=True)
    # 3) PRIMARY: comparative ranking (fast, robust to leniency)
    run(["experiments/run_rank.py", "--concepts", "data/gen_all.json",
         "--out", "results/rankings.jsonl", "--models"] + PANEL)
    # 4) judge-validation probe (planted known-quality analogies)
    run(["experiments/run_rank.py", "--concepts", "data/probe_discrimination.json",
         "--out", "results/rank_probe.jsonl", "--models"] + PANEL)
    # 5) SECONDARY: absolute 1-5 rubric scoring (dimension detail + RQ2 features)
    run(["experiments/run_judges.py", "--concepts", "data/gen_all.json",
         "--out", "results/judgments.jsonl", "--models"] + PANEL)
    # 6) analysis + RQ2 predictor + figures
    run(["experiments/analyze.py"])
    run(["experiments/features.py"])
    run(["experiments/make_figures.py"])
    print("\n=== RQ1 PIPELINE COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
