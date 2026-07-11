# -*- coding: utf-8 -*-
"""Generate the data-driven appendix (rubric, concept table, prompts) as LaTeX,
so these stay in sync with the actual data and code."""
import json, os, sys
ROOT = r"C:/Users/Admin/Desktop/AnalogyPaper"
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import run_generate as G
import run_rank as RK
import run_judges as JU

def esc(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum ")]:
        s = s.replace(a, b)
    return s

rubric = json.load(open(ROOT + "/data/rubric.json", encoding="utf-8"))
concepts = json.load(open(ROOT + "/data/concepts_all.json", encoding="utf-8"))
L = []

# --- Rubric ---
L.append(r"\section{The SAQR rubric in full}\label{app:rubric}")
L.append(esc(rubric["scale"]) + "\n")
for d in rubric["dimensions"]:
    L.append(r"\paragraph{%s (\texttt{%s}).} %s" % (esc(d["name"]), esc(d["key"]), esc(d["question"])))
    L.append(r"\emph{1:} %s \quad \emph{3:} %s \quad \emph{5:} %s" %
             (esc(d["anchors"]["1"]), esc(d["anchors"]["3"]), esc(d["anchors"]["5"])) + "\n")

# --- Concept table ---
L.append(r"\section{Concept dataset}\label{app:concepts}")
L.append("The %d concepts, balanced across four subjects and grades 8--12. Concepts marked $\\star$ "
         "carry the four-option multiple-choice items used in the comprehension study." % len(concepts))
L.append(r"\begin{center}\small")
L.append(r"\begin{tabular}{llcl}")
L.append(r"\toprule")
L.append(r"ID & Subject & Grade & Concept \\ \midrule")
for c in sorted(concepts, key=lambda x: (x["subject"], x["grade"], x["id"])):
    star = r"$\star$" if "questions" in c else ""
    L.append("%s & %s & %d & %s %s \\\\" % (esc(c["id"]), esc(c["subject"]), c["grade"], esc(c["concept"]), star))
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append(r"\end{center}")

# --- Prompts ---
L.append(r"\section{Prompts}\label{app:prompts}")
L.append(r"\paragraph{Analogy generation (system).} \small\ttfamily " + esc(G.SYSTEM) + r"\normalfont\normalsize")
L.append(r"\paragraph{Generation strategy instructions.}\begin{itemize}\setlength\itemsep{0pt}")
for k, v in G.STRATEGIES.items():
    L.append(r"\item \textbf{%s}: %s" % (esc(k), esc(v)))
L.append(r"\end{itemize}")
L.append(r"\paragraph{Comparative ranking (system).} \small\ttfamily " + esc(RK.SYSTEM) + r"\normalfont\normalsize")
L.append(r"\paragraph{Absolute rubric scoring (system).} \small\ttfamily " + esc(JU.SYSTEM) + r"\normalfont\normalsize")

open(ROOT + "/paper/appendix_generated.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote paper/appendix_generated.tex (%d concepts, %d rubric dims)" % (len(concepts), len(rubric["dimensions"])))
