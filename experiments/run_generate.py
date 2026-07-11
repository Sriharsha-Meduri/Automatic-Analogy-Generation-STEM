# -*- coding: utf-8 -*-
"""Analogy generation harness (the independent variable of the study).

For every concept we ask a generator model to produce one analogy under each
GENERATION STRATEGY (a constraint on the analogy's source domain). The strategy
is the manipulated variable; everything else (concept, standard explanation) is
held constant. Outputs are logged with the exact prompt for full reproducibility
and assembled into an augmented concepts file that the judge and comprehension
harnesses consume.

Usage:
  python run_generate.py --concepts data/concepts_pilot16.json \
      --gen-model qwen2.5:3b --out-concepts data/gen_qwen2.5-3b.json
"""
import json, os, re, time, argparse, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"

# The manipulated variable: each strategy constrains the analogy's source domain.
# 'free' is the unconstrained reference (model chooses any domain).
STRATEGIES = {
    "free":     "Use the single best analogy you can think of, from any source domain.",
    "everyday": "Use an analogy drawn from ordinary everyday life (the home, daily routines, common objects).",
    "sports":   "Use an analogy drawn from sports, games, or physical play.",
    "cooking":  "Use an analogy drawn from cooking, baking, or the kitchen.",
}

SYSTEM = ("You are an expert STEM teacher. You explain a concept with ONE vivid analogy that a "
          "school student (Class 8 to 12) can understand. The analogy must be scientifically "
          "faithful: the important relationships in the concept must correspond to matching "
          "relationships in the analogy. Do not teach a misconception for the sake of simplicity.")

def build_prompt(c, instruction):
    return (f"Concept ({c['subject']}, Class {c['grade']}): {c['concept']}\n"
            f"Standard explanation: {c['standard_explanation']}\n\n"
            f"Task: {instruction} Write ONLY the analogy itself, 2 to 4 sentences, no preamble, "
            f"no restating the concept name, no bullet points. Make sure it maps correctly onto the concept.")

def ollama_chat(model, system, user, seed, temperature, timeout=240):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": 220, "seed": seed}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {e}"
            time.sleep(2)

def clean(text):
    if not text or text.startswith("__ERROR__"):
        return text
    t = text.strip()
    # strip common preambles a model might emit despite instructions
    t = re.sub(r"^(sure[,!.]?|here('?s| is)[^:]*:|analogy:)\s*", "", t, flags=re.I).strip()
    t = t.strip('"').strip()
    return t

def seed_for(concept_id, strat):
    # deterministic but strategy/concept specific, for reproducible variety
    return (abs(hash((concept_id, strat))) % 100000) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="data/concepts_pilot16.json")
    ap.add_argument("--gen-model", required=True)
    ap.add_argument("--out-concepts", required=True)
    ap.add_argument("--log", default="results/generation_log.jsonl")
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    base = json.load(open(os.path.join(ROOT, args.concepts), encoding="utf-8"))
    outpath = os.path.join(ROOT, args.out_concepts)
    logpath = os.path.join(ROOT, args.log)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    # resume: reuse any analogies already generated in a prior run
    cache = {}
    if os.path.exists(outpath):
        for c in json.load(open(outpath, encoding="utf-8")):
            cache[c["id"]] = c.get("analogies", {})

    out = []
    logf = open(logpath, "a", encoding="utf-8")
    t0 = time.time(); n = 0
    for c in base:
        analogies = dict(cache.get(c["id"], {}))
        for strat, instruction in STRATEGIES.items():
            if analogies.get(strat):
                continue
            prompt = build_prompt(c, instruction)
            resp = ollama_chat(args.gen_model, SYSTEM, prompt,
                               seed=seed_for(c["id"], strat), temperature=args.temp)
            analogy = clean(resp)
            analogies[strat] = analogy
            logf.write(json.dumps({"generator": args.gen_model, "concept_id": c["id"],
                                   "strategy": strat, "system": SYSTEM, "prompt": prompt,
                                   "analogy": analogy}) + "\n"); logf.flush()
            n += 1
            print(f"  [{n}] {args.gen_model} {c['id']} {strat}: {str(analogy)[:80]}", flush=True)
        newc = dict(c); newc["analogies"] = analogies; newc["generator"] = args.gen_model
        out.append(newc)
        json.dump(out, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logf.close()
    print(f"DONE: {n} analogies generated in {time.time()-t0:.0f}s -> {outpath}", flush=True)

if __name__ == "__main__":
    main()
