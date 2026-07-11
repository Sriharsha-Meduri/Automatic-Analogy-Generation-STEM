# -*- coding: utf-8 -*-
"""Comprehension-outcome experiment: do analogies help a 'student' LLM answer
concept questions better than a standard explanation alone?

For each (concept, question, option-shuffle, condition, student model) we ask the
model to pick a multiple-choice answer and score it objectively. Conditions:
  baseline            : question only (no teaching text)
  standard            : standard explanation + question
  analogy:<strategy>  : standard explanation + that analogy + question
Resumable: appends one JSON line per trial; skips trials already recorded.
"""
import json, os, sys, random, re, time, argparse, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
LETTERS = ["A", "B", "C", "D"]

def ollama_chat(model, system, user, seed=0, timeout=120):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.0, "num_predict": 16, "seed": seed}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {e}"
            time.sleep(2)

def parse_letter(resp):
    if not resp or resp.startswith("__ERROR__"):
        return None
    up = resp.strip().upper()
    if up and up[0] in LETTERS:
        return up[0]
    m = re.search(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", " " + up + " ")
    return m.group(1) if m else None

def shuffled_options(options, correct, seed):
    items = [(k, options[k]) for k in LETTERS if k in options]
    rng = random.Random(seed)
    rng.shuffle(items)
    new_opts, new_correct = {}, None
    for i, (old_k, text) in enumerate(items):
        nk = LETTERS[i]
        new_opts[nk] = text
        if old_k == correct:
            new_correct = nk
    return new_opts, new_correct

def build_conditions(c):
    conds = [("baseline", None, ""), ("standard", None, c["standard_explanation"])]
    for strat, text in c["analogies"].items():
        conds.append((f"analogy:{strat}", strat, c["standard_explanation"] + " " + text))
    return conds

SYSTEM = ("You are a student taking a multiple-choice quiz. If study material is provided, use it. "
          "Answer with ONLY the single letter (A, B, C, or D) of the best option, nothing else.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="data/pilot_concepts.json")
    ap.add_argument("--out", default="results/comprehension.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--shuffles", type=int, default=3)
    args = ap.parse_args()

    concepts = json.load(open(os.path.join(ROOT, args.concepts), encoding="utf-8"))
    outpath = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    done = set()
    if os.path.exists(outpath):
        for line in open(outpath, encoding="utf-8"):
            try:
                r = json.loads(line); done.add((r["model"], r["concept_id"], r["q_index"], r["shuffle"], r["condition"]))
            except Exception:
                pass
    print(f"{len(concepts)} concepts | models={args.models} | shuffles={args.shuffles} | already done={len(done)}", flush=True)

    t0 = time.time(); n = 0
    with open(outpath, "a", encoding="utf-8") as fh:
        for model in args.models:
            for c in concepts:
                if not c.get("questions"):
                    continue  # analogy-only concepts (no MCQs) are not part of RQ3
                conds = build_conditions(c)
                for qi, q in enumerate(c["questions"]):
                    for s in range(args.shuffles):
                        opts, correct = shuffled_options(q["options"], q["answer"], seed=1000 * qi + s)
                        optblock = "\n".join(f"{k}) {opts[k]}" for k in LETTERS if k in opts)
                        for cond, strat, ctx in conds:
                            key = (model, c["id"], qi, s, cond)
                            if key in done:
                                continue
                            user = ((f"Study material: {ctx}\n\n" if ctx else "") +
                                    f"Question: {q['q']}\n{optblock}\n\nAnswer with only the letter.")
                            resp = ollama_chat(model, SYSTEM, user, seed=42)
                            pick = parse_letter(resp)
                            rec = {"model": model, "concept_id": c["id"], "subject": c["subject"],
                                   "grade": c["grade"], "q_index": qi, "shuffle": s, "condition": cond,
                                   "strategy": strat, "correct": correct, "pick": pick,
                                   "is_correct": int(pick == correct), "raw": resp[:80]}
                            fh.write(json.dumps(rec) + "\n"); fh.flush()
                            n += 1
                            if n % 25 == 0:
                                el = time.time() - t0
                                print(f"  {n} trials | {el:.0f}s | {n/el:.2f}/s | last {model} {c['id']} q{qi} {cond} -> {pick}/{correct}", flush=True)
    print(f"DONE: {n} new trials in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
