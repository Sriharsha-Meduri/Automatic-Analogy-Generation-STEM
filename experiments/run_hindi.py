# -*- coding: utf-8 -*-
"""Hindi arm: a self-contained cross-lingual replication of the generation and
evaluation pipeline, with all prompts in Hindi. Kept separate from the English
harnesses so it can run independently. Tests whether the English findings (weak
strategy effect, verbosity bias, screen-not-rank judges) replicate in Hindi, and
how far small open models cope with a mid-resource language.

Stages (all resumable): generate -> merge -> rank -> probe-rank -> score.
  python run_hindi.py
"""
import json, os, re, time, random, argparse, urllib.request, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console/file is cp1252 by default
except Exception:
    pass

OLLAMA = "http://127.0.0.1:11434/api/chat"
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
LETTERS = ["A", "B", "C", "D"]
STRATS = ["free", "everyday", "sports", "cooking"]
DIMS = ["conceptual_accuracy", "clarity", "memorability", "non_misleading", "overall_usefulness"]
GENERATORS = [("qwen2.5:3b", "data/gen_hi_qwen2.5-3b.json"),
              ("llama3.2:3b", "data/gen_hi_llama3.2-3b.json"),
              ("gemma2:2b", "data/gen_hi_gemma2-2b.json")]
PANEL = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3.5"]

GEN_SYSTEM = ("आप एक अनुभवी STEM शिक्षक हैं। आप किसी अवधारणा को एक जीवंत उपमा (analogy) से समझाते हैं "
              "जिसे कक्षा 8 से 12 का विद्यार्थी समझ सके। उपमा वैज्ञानिक रूप से सही होनी चाहिए: अवधारणा के "
              "महत्वपूर्ण संबंध उपमा के संबंधों से मेल खाने चाहिए। सरलता के लिए कोई ग़लत धारणा न सिखाएँ।")
STRAT_INSTR = {
    "free": "किसी भी क्षेत्र से सबसे अच्छी उपमा का प्रयोग करें।",
    "everyday": "रोज़मर्रा के जीवन (घर, दैनिक दिनचर्या, आम वस्तुओं) से उपमा लें।",
    "sports": "खेल-कूद या खेल से उपमा लें।",
    "cooking": "खाना पकाने या रसोई से उपमा लें।",
}
RANK_SYSTEM = ("आप एक अनुभवी STEM शिक्षक हैं जो शिक्षण-उपमाओं की तुलना कर रहे हैं। आपको एक ही अवधारणा के लिए "
               "कई उपमाएँ दिखाई जाएँगी, हर एक पर एक अक्षर का लेबल होगा। इन्हें कक्षा 8 से 12 के विद्यार्थी के लिए "
               "सर्वश्रेष्ठ से सबसे कमज़ोर क्रम में रैंक करें। केवल 'अक्षर: रैंक' पंक्तियाँ दें (अंग्रेज़ी अंकों में), जहाँ 1 सबसे अच्छा "
               "है। हर रैंक एक ही बार प्रयोग करें। कोई टिप्पणी नहीं।")
JUDGE_SYSTEM = ("आप एक अनुभवी STEM शिक्षक हैं जो एक निश्चित rubric पर किसी शिक्षण-उपमा का मूल्यांकन कर रहे हैं। "
                "हर आयाम को 1 से 5 के बीच पूर्णांक (अंग्रेज़ी अंकों में) अंक दें, पूरी range का प्रयोग करें: 1 खराब, 3 ठीक-ठाक, "
                "5 उत्कृष्ट। हर आयाम को एक जैसा ऊँचा अंक न दें; अंतर करें। केवल 'dimension_key: score' पंक्तियाँ दें "
                "(dimension_key अंग्रेज़ी में), कोई टिप्पणी नहीं।")
DIM_GLOSS = {
    "conceptual_accuracy": "संकल्पनात्मक सटीकता (क्या उपमा अवधारणा के मुख्य संबंधों पर सही बैठती है?)",
    "clarity": "स्पष्टता (क्या विद्यार्थी के लिए समझना आसान है और समझ बढ़ाती है?)",
    "memorability": "स्मरणीयता (क्या यह जीवंत, ठोस और याद रखने योग्य है?)",
    "non_misleading": "भ्रामक-सरलीकरण का अभाव (क्या यह कोई ग़लत धारणा नहीं फैलाती?)",
    "overall_usefulness": "समग्र उपयोगिता (पढ़ाने के लिए यह कुल मिलाकर कितनी उपयोगी है?)",
}
_DEVA = {ord(d): str(i) for i, d in enumerate("०१२३४५६७८९")}

def norm(s):
    return s.translate(_DEVA) if s else s

def chat(model, system, user, num_predict, temperature=0.0, seed=42, timeout=240):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": num_predict, "seed": seed}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if a == 2:
                return f"__ERROR__ {e}"
            time.sleep(2)

def load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def clean(t):
    if not t or t.startswith("__ERROR__"):
        return t
    t = re.sub(r"^(यह उपमा|उपमा[:：]?|analogy:)\s*", "", t.strip(), flags=re.I)
    return t.strip().strip('"').strip()

# ---------- generation ----------
def generate():
    base = load("data/concepts_hi.json")
    logf = open(os.path.join(ROOT, "results/generation_log_hi.jsonl"), "a", encoding="utf-8")
    for model, out in GENERATORS:
        cache = {c["id"]: c.get("analogies", {}) for c in (load(out) or [])}
        res = []
        for c in base:
            an = dict(cache.get(c["id"], {}))
            for s in STRATS:
                if an.get(s):
                    continue
                prompt = (f"अवधारणा ({c['subject']}, कक्षा {c['grade']}): {c['concept']}\n"
                          f"मानक व्याख्या: {c['standard_explanation']}\n\nकार्य: {STRAT_INSTR[s]} केवल उपमा लिखें, "
                          f"2 से 4 वाक्य, कोई भूमिका नहीं। ध्यान दें कि उपमा अवधारणा पर सही बैठे।")
                a = clean(chat(model, GEN_SYSTEM, prompt, 260, temperature=0.7, seed=(abs(hash((c['id'], s))) % 90000) + 1))
                an[s] = a
                logf.write(json.dumps({"generator": model, "concept_id": c["id"], "strategy": s,
                                       "prompt": prompt, "analogy": a}, ensure_ascii=False) + "\n"); logf.flush()
                print(f"  gen {model} {c['id']} {s}: {str(a)[:60]}", flush=True)
            nc = dict(c); nc["analogies"] = an; nc["generator"] = model
            res.append(nc)
            json.dump(res, open(os.path.join(ROOT, out), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logf.close()
    # merge
    allc = []
    for model, out in GENERATORS:
        for c in (load(out) or []):
            c["generator"] = model; allc.append(c)
    json.dump(allc, open(os.path.join(ROOT, "data/gen_hi_all.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"merged {len(allc)} Hindi analogy-sets", flush=True)

# ---------- ranking ----------
def parse_ranking(resp, letters):
    if not resp or resp.startswith("__ERROR__"):
        return None
    resp = norm(resp)
    ranks = {}
    for L in letters:
        m = re.search(rf"(?:^|[^A-Za-z]){L}[^A-Za-z0-9]{{0,4}}?([1-4])", resp)
        if m:
            ranks[L] = int(m.group(1))
    if len(ranks) != len(letters) or len(set(ranks.values())) != len(letters):
        return None
    return ranks

def rank(concepts_path, outpath):
    concepts = load(concepts_path)
    op = os.path.join(ROOT, outpath)
    done = set()
    if os.path.exists(op):
        for l in open(op, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["judge"], r["generator"], r["concept_id"]))
            except Exception:
                pass
    fh = open(op, "a", encoding="utf-8")
    for judge in PANEL:
        for c in concepts:
            gen = c.get("generator", "unknown")
            if judge == gen or (judge, gen, c["id"]) in done:
                continue
            strategies = [s for s in STRATS if s in c.get("analogies", {})] or list(c.get("analogies", {}).keys())
            strategies = strategies[:len(LETTERS)]
            if len(strategies) < 2:
                continue
            rng = random.Random(abs(hash((judge, gen, c["id"]))) & 0xFFFFFFFF)
            order = strategies[:]; rng.shuffle(order)
            letters = LETTERS[:len(order)]
            lab = {L: st for L, st in zip(letters, order)}
            block = "\n\n".join(f"{L}) {c['analogies'][lab[L]]}" for L in letters)
            user = (f"अवधारणा: {c['concept']}\nमानक व्याख्या: {c['standard_explanation']}\n\nउपमाएँ:\n{block}\n\n"
                    f"सभी {len(letters)} को सर्वश्रेष्ठ (1) से सबसे कमज़ोर तक रैंक करें। केवल '<अक्षर>: <रैंक>' पंक्तियाँ दें।")
            resp = chat(judge, RANK_SYSTEM, user, 40)
            ranks = parse_ranking(resp, letters)
            sr = ({lab[L]: rk for L, rk in ranks.items()} if ranks else None)
            fh.write(json.dumps({"judge": judge, "generator": gen, "concept_id": c["id"], "subject": c["subject"],
                                 "grade": c["grade"], "ranks": sr, "raw": resp[:120]}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  rank {judge}<-{gen} {c['id']} -> {sr}", flush=True)
    fh.close()

# ---------- absolute scoring ----------
def parse_scores(resp):
    if not resp or resp.startswith("__ERROR__"):
        return None
    resp = norm(resp)
    out = {}
    for d in DIMS:
        m = re.search(d + r"\D{0,6}?([1-5])", resp, re.I)
        out[d] = int(m.group(1)) if m else None
    return out if any(v is not None for v in out.values()) else None

def score(outpath="results/judgments_hi.jsonl"):
    concepts = load("data/gen_hi_all.json")
    rubric = load("data/rubric.json")
    rubric_text = "\n".join(f"- {d['key']}: {DIM_GLOSS[d['key']]}" for d in rubric["dimensions"])
    op = os.path.join(ROOT, outpath)
    done = set()
    if os.path.exists(op):
        for l in open(op, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["judge"], r.get("generator"), r["concept_id"], r["strategy"]))
            except Exception:
                pass
    fh = open(op, "a", encoding="utf-8")
    for judge in PANEL:
        for c in concepts:
            gen = c.get("generator", "unknown")
            if judge == gen:
                continue
            for strat, analogy in c["analogies"].items():
                if (judge, gen, c["id"], strat) in done:
                    continue
                user = (f"अवधारणा: {c['concept']}\nमानक व्याख्या: {c['standard_explanation']}\n\n"
                        f"मूल्यांकन के लिए उपमा: {analogy}\n\nRubric (हर आयाम को 1 से 5 अंक दें):\n{rubric_text}\n\n"
                        f"अब अपने अंक दें, हर पंक्ति 'key: score' के रूप में।")
                resp = chat(judge, JUDGE_SYSTEM, user, 64)
                sc = parse_scores(resp)
                fh.write(json.dumps({"judge": judge, "generator": gen, "concept_id": c["id"], "subject": c["subject"],
                                     "grade": c["grade"], "strategy": strat, "scores": sc, "raw": resp[:200]},
                                    ensure_ascii=False) + "\n"); fh.flush()
                print(f"  score {judge}<-{gen} {c['id']} {strat} -> {sc}", flush=True)
    fh.close()

if __name__ == "__main__":
    t0 = time.time()
    print("=== HINDI: generation ==="); generate()
    print("=== HINDI: ranking ==="); rank("data/gen_hi_all.json", "results/rankings_hi.jsonl")
    print("=== HINDI: probe ==="); rank("data/probe_hi.json", "results/rank_probe_hi.jsonl")
    print("=== HINDI: absolute scoring ==="); score()
    print(f"=== HINDI PIPELINE COMPLETE in {time.time()-t0:.0f}s ===")
