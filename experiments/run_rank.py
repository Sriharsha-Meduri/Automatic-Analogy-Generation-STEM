# -*- coding: utf-8 -*-
"""Comparative-ranking judge panel (robust primary RQ1 measure).

Absolute 1-5 scoring by small local models is leniency-prone and the judges
anchor their scales differently, hurting inter-rater agreement. Ranking sidesteps
this: for each concept (holding the generator fixed) a judge sees all four
strategy analogies, with letter labels shuffled to control position bias, and
ranks them 1 (best) to 4 (worst). We later report mean rank per strategy, a
Friedman test, and Kendall's W as the inter-judge agreement statistic.
Resumable: one JSON line per (judge, generator, concept)."""
import json, os, re, time, random, argparse, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
LETTERS = ["A", "B", "C", "D"]
STRATEGIES = ["free", "everyday", "sports", "cooking"]

SYSTEM = ("You are an experienced STEM teacher comparing teaching analogies. You will see several "
          "analogies for the SAME concept, each labelled with a letter. Rank them from best to worst "
          "as teaching aids for a school student (Class 8 to 12), judging conceptual accuracy, "
          "clarity, and whether they mislead. Output ONLY lines of the form 'LETTER: rank', one per "
          "analogy, where 1 is best. Use each rank exactly once. No commentary.")

def ollama_chat(model, system, user, timeout=180):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.0, "num_predict": 40, "seed": 42}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if a == 2:
                return f"__ERROR__ {e}"
            time.sleep(2)

def parse_ranking(resp, letters):
    if not resp or resp.startswith("__ERROR__"):
        return None
    ranks = {}
    for L in letters:
        m = re.search(rf"(?:^|[^A-Za-z]){L}[^A-Za-z0-9]{{0,4}}?([1-4])", resp)
        if m:
            ranks[L] = int(m.group(1))
    # require every letter ranked and all ranks distinct (a valid permutation)
    if len(ranks) != len(letters) or len(set(ranks.values())) != len(letters):
        return None
    return ranks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="data/gen_all.json")
    ap.add_argument("--out", default="results/rankings.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    concepts = json.load(open(os.path.join(ROOT, args.concepts), encoding="utf-8"))
    outpath = os.path.join(ROOT, args.out); os.makedirs(os.path.dirname(outpath), exist_ok=True)
    done = set()
    if os.path.exists(outpath):
        for line in open(outpath, encoding="utf-8"):
            try:
                r = json.loads(line); done.add((r["judge"], r["generator"], r["concept_id"]))
            except Exception:
                pass
    print(f"{len(concepts)} concept-sets | judges={args.models} | done={len(done)}", flush=True)
    t0 = time.time(); n = 0
    with open(outpath, "a", encoding="utf-8") as fh:
        for judge in args.models:
            for c in concepts:
                gen = c.get("generator", "unknown")
                if judge == gen:
                    continue  # no self-judging
                if (judge, gen, c["id"]) in done:
                    continue
                strategies = [s for s in STRATEGIES if s in c.get("analogies", {})] \
                    or list(c.get("analogies", {}).keys())
                strategies = strategies[:len(LETTERS)]
                if len(strategies) < 2:
                    continue
                # shuffle strategy -> letter assignment (seeded by judge+concept for reproducibility)
                rng = random.Random(hash((judge, gen, c["id"])) & 0xFFFFFFFF)
                order = strategies[:]; rng.shuffle(order)
                letters = LETTERS[:len(order)]
                lab2strat = {L: st for L, st in zip(letters, order)}
                block = "\n\n".join(f"{L}) {c['analogies'][lab2strat[L]]}" for L in letters)
                user = (f"Concept: {c['concept']}\nStandard explanation: {c['standard_explanation']}\n\n"
                        f"Analogies:\n{block}\n\nRank all {len(letters)} from best (1) to worst "
                        f"({len(letters)}). Output '<letter>: <rank>' lines only, each rank used once.")
                resp = ollama_chat(judge, SYSTEM, user)
                ranks = parse_ranking(resp, letters)
                strat_ranks = ({lab2strat[L]: rk for L, rk in ranks.items()} if ranks else None)
                rec = {"judge": judge, "generator": gen, "concept_id": c["id"], "subject": c["subject"],
                       "grade": c["grade"], "ranks": strat_ranks, "raw": resp[:120]}
                fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                if n % 20 == 0:
                    print(f"  {n} rankings | {time.time()-t0:.0f}s | last {judge}<-{gen} {c['id']} -> {strat_ranks}", flush=True)
    print(f"DONE: {n} new rankings in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
