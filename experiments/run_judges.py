# -*- coding: utf-8 -*-
"""Multi-model judge panel: each local model scores every analogy on the SAQR
rubric (1-5 per dimension). We later report inter-judge agreement (Krippendorff
alpha) as the automated analogue of human inter-rater agreement, plus mean
quality per prompting strategy. Resumable (one JSON line per analogy x judge)."""
import json, os, re, time, argparse, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
DIMS = ["conceptual_accuracy", "clarity", "memorability", "non_misleading", "overall_usefulness"]

def ollama_chat(model, system, user, timeout=180):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.0, "num_predict": 64, "seed": 42}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if a == 2:
                return f"__ERROR__ {e}"
            time.sleep(2)

def parse_scores(resp):
    if not resp or resp.startswith("__ERROR__"):
        return None
    out = {}
    for d in DIMS:
        m = re.search(d + r"\D{0,6}?([1-5])", resp, re.I)
        out[d] = int(m.group(1)) if m else None
    return out if any(v is not None for v in out.values()) else None

def build_rubric_text(rubric):
    lines = []
    for dim in rubric["dimensions"]:
        lines.append(f"- {dim['key']} ({dim['name']}): {dim['question']} "
                     f"1={dim['anchors']['1']} 3={dim['anchors']['3']} 5={dim['anchors']['5']}")
    return "\n".join(lines)

SYSTEM = ("You are an experienced STEM teacher evaluating a teaching analogy on a fixed rubric. "
          "Score each dimension from 1 to 5 as a whole number, using the full range: 1 is poor, "
          "3 is adequate, 5 is excellent. Do not give every dimension the same high score; "
          "discriminate. Output ONLY lines of the form 'dimension_key: score', one per dimension, "
          "no commentary.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="data/pilot_concepts.json")
    ap.add_argument("--out", default="results/judgments.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    concepts = json.load(open(os.path.join(ROOT, args.concepts), encoding="utf-8"))
    rubric = json.load(open(os.path.join(ROOT, "data/rubric.json"), encoding="utf-8"))
    rubric_text = build_rubric_text(rubric)
    outpath = os.path.join(ROOT, args.out); os.makedirs(os.path.dirname(outpath), exist_ok=True)
    done = set()
    if os.path.exists(outpath):
        for line in open(outpath, encoding="utf-8"):
            try:
                r = json.loads(line); done.add((r["judge"], r.get("generator", "?"), r["concept_id"], r["strategy"]))
            except Exception:
                pass
    print(f"{len(concepts)} concepts | judges={args.models} | done={len(done)}", flush=True)
    t0 = time.time(); n = 0
    with open(outpath, "a", encoding="utf-8") as fh:
        for judge in args.models:
            for c in concepts:
                gen = c.get("generator", "unknown")
                if judge == gen:
                    continue  # never let a model judge its own analogies (self-preference bias)
                for strat, analogy in c["analogies"].items():
                    if (judge, gen, c["id"], strat) in done:
                        continue
                    user = (f"Concept: {c['concept']}\nStandard explanation: {c['standard_explanation']}\n\n"
                            f"Analogy to score: {analogy}\n\nRubric (score each 1 to 5):\n{rubric_text}\n\n"
                            f"Give your scores now, one 'key: score' per line.")
                    resp = ollama_chat(judge, SYSTEM, user)
                    scores = parse_scores(resp)
                    rec = {"judge": judge, "generator": gen, "concept_id": c["id"], "subject": c["subject"],
                           "grade": c["grade"], "strategy": strat, "scores": scores, "raw": resp[:240]}
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                    if n % 20 == 0:
                        print(f"  {n} judgments | {time.time()-t0:.0f}s | last {judge}<-{gen} {c['id']} {strat} -> {scores}", flush=True)
    print(f"DONE: {n} new judgments in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
