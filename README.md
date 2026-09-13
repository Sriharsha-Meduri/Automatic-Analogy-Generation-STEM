# Automatic Analogy Generation for STEM Education

**Prompting Strategies and the Limits of Automated Quality Evaluation**

Sriharsha Meduri, Mohit Pratap Singh Rathore, and Gunveer Kalsi (Oviqo, India); Pratap Chandra Mandal (Indian Institute of Management Shillong, India)

This repository contains the code, dataset, and raw model outputs for a study of how to prompt
language models for good STEM teaching analogies, and how to evaluate analogy quality
automatically using only small open models on a single consumer GPU (an NVIDIA GTX 1650, 4 GB),
with no paid API and no human annotation in the loop. The manuscript is being posted to arXiv as a preprint; the repository holds the code, data,
results, and figures that reproduce it.

## What the study does

For 64 school-curriculum concepts across Physics, Chemistry, Mathematics, and Biology
(grades 8 to 12), we generate analogies under four source-domain prompting strategies
(unconstrained `free`, `everyday`, `sports`, `cooking`) with three open generator models, and
evaluate every analogy with a four-model judge panel using a purpose-built five-dimension
rubric (SAQR). We then repeat the whole pipeline in Hindi on a paired 20-concept subset. Four
research questions:

- **RQ1** Do the prompting strategies differ in analogy quality?
- **RQ2** What do the automated judges reward (an audit of judge bias)?
- **RQ3** Do analogies improve a "student" model's comprehension, and how does that depend on
  the student's capability?
- **RQ4** Do the findings replicate in Hindi, and how far do small models degrade outside English?

## Headline findings

- **No strategy effect (with a twist).** Constraining the source domain does not beat letting
  the model choose freely: the small panel finds all four strategies statistically equivalent
  (TOST, all pairs within +/-0.3 rank). A larger, cleaner judge does resolve a modest but
  significant advantage for the unconstrained strategy (p<0.001), so the equivalence is partly
  the small panel's own limitation, and the practical guidance is the same: let the model
  choose the domain.
- **The findings hold at twice the model scale, and the biases shrink.** A 7B judge screens the
  probe just as well (100%), the verbosity bias essentially vanishes (length-score correlation
  +0.15 -> -0.02), and its two channels then agree; the no-effect result also holds when a 7B
  model writes the analogies. Judge scale is the clearest lever for more reliable automated
  evaluation.
- **Judges screen well but rank poorly.** A planted-quality probe (a deliberately wrong
  analogy in each set) is caught by every judge 100% of the time, yet inter-judge agreement on
  the real comparison is near zero. Automated panels are trustworthy for catching bad
  analogies, not for finely ranking good ones.
- **"Quality" prediction is really bias prediction.** A random forest predicts judged score
  with R^2 = 0.72, but from length, lexical complexity, and subject, not from the strategy: a
  verbosity bias, not a quality signal.
- **Comprehension gains come from explanation, not analogy.** A plain explanation lifts the
  weakest student model's accuracy by about 16 points; adding an analogy on top yields no
  reliable gain for any student.
- **The tooling is English-first.** In Hindi the no-effect finding appears to hold, though the
  automated evaluation degrades so sharply that we read this as tentative rather than a firm
  replication: rubric output parses only 35% of the time (versus 97% in English) and the judges
  catch the wrong analogy only 58% of the time (versus 100%).

Full numbers, statistics, and caveats are documented in the analysis scripts and the raw
outputs under `results/`; the manuscript is maintained separately.

## Repository layout

```
data/             concepts (64 English, 20 Hindi), SAQR rubric, per-generator analogies, probes
experiments/      generation, ranking, scoring, comprehension, prediction, analysis, figures
results/          raw model outputs (JSONL): rankings, judgments, comprehension trials, logs
figures/          publication figures (PNG)
```

Key scripts in `experiments/`:

| Script | Purpose |
| --- | --- |
| `run_generate.py` | Generate analogies per strategy with a local model (the independent variable) |
| `run_rank.py` | Comparative-ranking judge panel (primary quality signal) |
| `run_judges.py` | Absolute SAQR rubric scoring (secondary) |
| `run_comprehension.py` | Student-capability comprehension experiment (RQ3) |
| `features.py` | Feature-based judge-score predictor (RQ2 audit) |
| `run_rq1.py` | End-to-end English driver (generate, merge, rank, probe, score, analyse, plot) |
| `run_hindi.py` | Self-contained Hindi replication (RQ4) |
| `analyze.py` / `analyze_hi.py` | Analysis and English-vs-Hindi comparison |
| `make_figures.py` / `make_equiv.py` | Publication figures (incl. the RQ1 equivalence forest plot) |

## Reproducing

1. Install [Ollama](https://ollama.com) and pull the models (each as a separate `ollama pull`):
   `qwen2.5:3b llama3.2:3b gemma2:2b phi3.5 llama3.2:1b qwen2.5:0.5b`.
2. English pipeline: `python experiments/run_rq1.py`.
3. Comprehension (RQ3): `python experiments/run_comprehension.py --concepts data/gen_qwen2.5-3b.json --out results/comprehension.jsonl --models qwen2.5:0.5b llama3.2:1b gemma2:2b qwen2.5:3b --shuffles 2`.
4. Hindi replication (RQ4): `python experiments/run_hindi.py`.
5. Analyse and plot: `python experiments/analyze.py`, `python experiments/analyze_hi.py`, `python experiments/make_figures.py`.

Every stage writes append-only JSONL and is resumable; evaluation calls use temperature 0 with
fixed seeds, and option/label orders are deterministically seeded, so the pipeline is
reproducible.

## A note on honesty

The judges and the "students" are language models used as automated proxies for human raters
and learners, not replacements for them. This fully-automated regime is a deliberate scope, and
every quality and comprehension number should be read as a proxy measurement; where the
automation is unreliable (small-model leniency, the Hindi breakdown), we say so and quantify it.
No ratings, annotations, or results in this repository are fabricated: all model outputs were
actually produced by the models named, and the raw outputs are included for inspection.
