"""Update the language-steering table for the model list of the journal version.

The published table (data/results_of_rq2_published.tex) is the source for every LLM that
was already in it: those cells are copied verbatim rather than recomputed, so nothing
already reported changes. Four things happen here:

  * the DeepSeek-V3 and DeepSeek-R1 columns are removed;
  * the gpt-4-0613 column is replaced by a gpt-4.1 column computed from the new run;
  * two rows are added, the anonymous few-shot conditions of code/aaai_anon.py, computed
    for all six LLMs;
  * the AVG column is recomputed as the mean of the six remaining LLMs.

Everything computed here follows the scoring of show_results/rq2_results.py, so a computed
cell is comparable with the published ones beside it. Three of its conventions are easy to
miss and all three matter:

  * the per-country SKIP list, which drops that country's five worked examples;
  * the probability vector is read in the order the LLM emitted it, and the *human*
    vector is built in that same order rather than in the questionnaire's;
  * a question answered twice in a results.csv counts twice.

Reproducing it exactly is possible -- 272 of the 300 published cells come back to the
digit -- but only by also reproducing its integer-key bug, which read_llm() declines to;
see the comment there. With the keys normalised the figure is 236 of 300. The gap is
entirely cells whose published value is wrong, so the deviation is the price of scoring
the new rows correctly rather than a disagreement about method. Two consequences are
stated in the paper rather than hidden here:

  * Germany's published block sits about 0.002 below what today's answers score, for
    every model and every row. No skip list explains it; a results.csv most likely grew
    after the table was made.
  * a published cell and a computed cell in the same block can differ by up to ~0.02 for
    reasons that have nothing to do with the prompt. The named/anonymous few-shot
    comparison \\S5.1 draws is therefore quoted from `--check`, which scores both rows
    the same way, not from the difference of two table cells.

  python plot/steering_table.py --check      # print the computed cells before writing
  python plot/steering_table.py
"""
import argparse
import ast
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import C, ROOT, stars, write_table                 # noqa: E402

csv.field_size_limit(10 ** 7)
AAAI = C.AAAI_RESULTS
PUBLISHED = f"{ROOT}/data/results_of_rq2_published.tex"
# the conference appendix carries the eight countries the main table left out, in the
# same nine-column layout, so both files are parsed the same way
APPENDIX = f"{ROOT}/data/results_of_rq2_published_appendix.tex"

# Three tables. The main text carries three countries -- one Latin-script and two not,
# with China the block the conference version also showed -- and the remaining four
# one-country-per-language blocks go to the appendix. The three Spanish-speaking ones
# stay in a table of their own, since they share a language and have to be read against
# each other rather than against a language that names them.
TABLES = [("results_of_rq2", "tab:steerability_results",
           ["Germany", "Japan", "China"]),
          ("results_of_rq2_more", "tab:steerability_results_more",
           ["Korea", "Russia", "Viet Nam", "Brazil"]),
          ("results_of_rq2_es", "tab:steerability_results_es",
           ["Argentina", "Chile", "Uruguay"])]
LANG_OF = {"China": "Zh.", "Germany": "De.", "Japan": "Ja.", "Korea": "Ko.",
           "Russia": "Ru.", "Viet Nam": "Vi.", "Brazil": "Pt.", "Argentina": "Es.",
           "Chile": "Es.", "Uruguay": "Es."}
HEAD_RE = r"multicolumn\{\d+\}\{c\}\{ ?\\textit\{([A-Za-z ]+?)(?: \([A-Za-z]+\))?\}\}"

OLD_COLS = ["Aya23", "Llama3", "Qwen2.5", "GPT-3.5", "GPT-4", "GPT-5", "DS-V3", "DS-R1",
            "AVG."]
NEW_COLS = ["Aya23", "Llama3", "Qwen2.5", "GPT-3.5", "GPT-4.1", "GPT-5", "AVG."]
REPLACES = "GPT-4"          # the published column the new one takes the place of
NEW_MODEL = "gpt-4.1"
# the directory each column's answers are filed under in the conference repository
MODEL_DIR = {"Aya23": "CohereLabs/aya-23-35B",
             "Llama3": "meta-llama/Meta-Llama-3-70B-Instruct",
             "Qwen2.5": "Qwen/Qwen2.5-72B-Instruct",
             "GPT-3.5": "gpt-3.5-turbo", "GPT-4.1": NEW_MODEL, "GPT-5": "gpt-5"}

CODE = {"China": 156, "Germany": 276, "Japan": 392, "Korea": 410, "Russia": 643,
        "Viet Nam": 704, "Brazil": 76, "Argentina": 32, "Chile": 152, "Uruguay": 858}
# the directory name under results/RQ2 differs from the label printed in the table. The
# space is not simply dropped: the conference repository spells it "Vietnam", so deriving
# the directory from "Viet Nam" alone yields a path that does not exist and silently
# turned every Viet Nam cell of the computed column into "--".
DIR_OF = dict({c: c.replace(" ", "") for c in CODE}, **{"Viet Nam": "Vietnam"})
# the five example questions shown in that country's few-shot block, dropped from scoring
SKIP = {32: ["Q40", "Q60", "Q80", "Q150", "Q160"], 152: ["Q40", "Q60", "Q80", "Q150", "Q160"],
        858: ["Q40", "Q60", "Q80", "Q150", "Q160"], 156: ["Q60", "Q70", "Q110", "Q150", "Q160"],
        392: ["Q60", "Q70", "Q90", "Q110", "Q130"], 410: ["Q40", "Q80", "Q150", "Q160", "Q170"],
        276: ["Q40", "Q80", "Q150", "Q160", "Q170"], 643: ["Q40", "Q60", "Q80", "Q150", "Q160"],
        704: ["Q40", "Q60", "Q80", "Q150", "Q160"], 76: ["Q40", "Q60", "Q80", "Q150", "Q160"]}
# the rows of a country block, in order, with their directory under results/RQ2. The
# first six are the published ones and are copied; the last two are computed here.
# The rows in the order they are printed, as (label, directory under results/RQ2, index
# of that row in the published file). The third field is None for the two rows computed
# here, and it is what keeps the printed order independent of the published one: the
# published file has the persona-carrying few-shot pair at 4 and 5, and the table prints
# it after the pair that carries no persona, so position cannot be used to find the data.
#
# `Few-shot` shows the country's distributions and nothing else; `Few. + Per.` also names
# the country through a persona, which is what meister-etal-2025-benchmarking calls
# few-shot steering and what the published table reports.
ROWS = [("No Steering", None, 0),
        ("Language Steering", "language", 1),
        ("Persona Steering", "persona", 2),
        ("Persona + Language", "language_persona", 3),
        ("Few-shot Steering", "few-shot-anon", None),
        ("Few-shot Steering + Language", "language_few-shot-anon", None),
        ("Few. + Per.", "few-shot", 4),
        ("Few. + Per. + Language", "language_few-shot", 5)]
N_PUBLISHED = 6
PUB_ROWS = sorted([r for r in ROWS if r[2] is not None], key=lambda r: r[2])
ANON_ROWS = [r for r in ROWS if r[2] is None]
# each row is bolded against the other row of its pair, and the pairs are adjacent
PARTNER = {ROWS[i][0]: ROWS[i + 1 if i % 2 == 0 else i - 1][0] for i in range(len(ROWS))}
# Every row head is generated rather than copied from the published file, which pads them
# to inconsistent widths and, for the two few-shot rows, no longer carries the name we
# print. `%s` takes the abbreviation of that country's language.
HEADS = [r"    No Steering (En.) ",
         r"    \ \ +\textit{Language Steering} (%s) ",
         r"    Persona Steering (En.) ",
         r"    \ \ +\textit{Language Steering} (%s) ",
         r"    Few-shot Steering (En.) ",
         r"    \ \ +\textit{Language Steering} (%s) ",
         r"    Few. + Per. (En.) ",
         r"    \ \ +\textit{Language Steering} (%s) "]


# ------------------------------------------------------------------ the new column
_WVS = None


def wvs():
    global _WVS
    if _WVS is None:
        _WVS = pd.read_csv(C.WVS_CSV, low_memory=False)
    return _WVS


def read_llm(path):
    """-> (question ids in file order, {qid: probs}, {qid: the key order emitted}).

    The id list is returned separately from the dict because rq2_results.py accumulates
    it as a list: a question answered twice in a results.csv is scored twice, and
    gpt-3.5-turbo has files where 73 of 214 rows are such repeats.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["type"] == "Opinion-Dependent"]
    qids, probs, order = [], {}, {}
    for _, r in df.iterrows():
        keys = ast.literal_eval(r["choice_keys"])
        qids.append(r["id"])
        try:
            d = ast.literal_eval(r["response"])
        except Exception:
            try:
                d = ast.literal_eval(re.sub(r"(\d+):\s*([\d.]+%)", r"'\1': '\2'",
                                            r["response"]))
            except Exception:
                continue
        # The keys are normalised to strings, which rq2_results.py does not do. Without
        # it, a response that parses to integer keys keeps them, the padding below adds
        # '1' beside 1, and the vector comes out at twice its length with the model's
        # mass in the first half and zeros in the second -- a score of about 0.72 where
        # the truth is 0.91. It is not a convention worth reproducing: 30.4% of the
        # gpt-4.1 answers are integer-keyed, against 0.9% or less for every model of the
        # published table, whose cells therefore barely move under the normalisation.
        clean = {str(k): float(str(v).strip("%")) / 100 for k, v in d.items()}
        for k in keys:
            clean.setdefault(str(k), 0.0)
        # rq2_results.py takes d_clean.values(): the order the LLM emitted, with any
        # omitted option appended. Kept so a computed cell matches the ones beside it.
        probs[r["id"]] = list(clean.values())
        order[r["id"]] = list(clean.keys())
    return qids, probs, order


DROPPED = []               # (path, qid) of answers no distance can be computed from


def per_question(llm, code, skip, path=""):
    """-> {qid: alignment score}, and the id list with repeats, for the mean."""
    qids, probs, order = llm
    qids = list(qids)
    for q in skip:
        while q in qids:
            qids.remove(q)
    sub = wvs()[wvs().B_COUNTRY == int(code)]
    out = {}
    for q, keys in order.items():
        # the human vector is built in the LLM's key order, not the questionnaire's;
        # they differ whenever a model emits its options out of order
        counts = [int((sub[q] == int(k)).sum()) for k in keys]
        if sum(counts) == 0:
            continue
        p1 = probs[q]
        p2 = [c / sum(counts) for c in counts]
        if len(p1) != len(p2):
            continue
        if any(x < 0 for x in p1):
            # gpt-5 wrote one Brazilian answer as '10':'-0.30%'. A negative mass is not a
            # distribution and the distance is undefined, so the question is dropped the
            # way an unparseable answer is -- and counted, so it is not dropped silently.
            DROPPED.append((path, q))
            continue
        n = len(p1)
        sup = np.arange(1, n + 1)
        out[q] = 1 - wasserstein_distance(sup, sup, p1, p2) / (n - 1)
    return out, [q for q in qids if q in out]


def row_scores(country, model_dir, rows=ROWS):
    """-> {row label: ({qid: score}, [qid with repeats])} for one country and model."""
    out = {}
    for label, method, *_ in rows:
        path = (f"{AAAI}/RQ1/{model_dir}/results.csv" if method is None else
                f"{AAAI}/RQ2/{method}/{DIR_OF[country]}/{model_dir}/results.csv")
        llm = read_llm(path)
        out[label] = per_question(llm, CODE[country], list(SKIP[CODE[country]]), path) \
            if llm else None
    return out


def _cell(scores, label, partner):
    """One finished cell: the mean, bolded against its partner row, plus the stars.

    The partner is passed rather than derived from the row index, because the two new
    rows are scored in a list that also holds No Steering and their neighbour there is
    not the row they are read against."""
    d = scores.get(label)
    if not d:
        return "--"
    per, qids = d
    mean = float(np.mean([per[q] for q in qids]))
    p = float("nan")
    base = scores.get("No Steering")
    if label != "No Steering" and base:
        qs = sorted(set(per) & set(base[0]))
        if len(qs) > 1:
            p = float(ttest_rel([per[q] for q in qs], [base[0][q] for q in qs]).pvalue)
    other = scores.get(partner)
    num = f"{mean:.4f}"
    if other and mean > float(np.mean([other[0][q] for q in other[1]])):
        num = r"\textbf{%s}" % num
    return num + stars(p)                   # the table bolds the number, not the stars


def new_column(countries, model=NEW_MODEL):
    """-> {(country, published row index): cell} for one model, over the published rows."""
    cells = {}
    for country in countries:
        scores = row_scores(country, model, PUB_ROWS)
        for label, _, pub in PUB_ROWS:
            cells[(country, pub)] = _cell(scores, label, PARTNER[label])
    return cells


def anon_rows(countries):
    """-> {(country, label, column): cell string} for the two new rows, all models.

    Each new row is bolded against the other new row, and starred against No Steering of
    the same LLM, which is the convention of every other pair in the table.
    """
    keep = [ROWS[0]] + ANON_ROWS
    cells = {}
    for country in countries:
        for col, mdir in MODEL_DIR.items():
            scores = row_scores(country, mdir, keep)
            for label, _, _ in ANON_ROWS:
                cells[(country, label, col)] = _cell(scores, label, PARTNER[label])
    return cells


def compare_fewshot(countries):
    """Named against anonymous few-shot, both scored here, so the difference is the
    prompt and nothing else.

    The table cannot answer this: its named rows are published values and its anonymous
    rows are computed, and the two conventions disagree on the answers a model wrote with
    unquoted keys. This prints the like-for-like number the paper quotes.
    """
    pairs = [("Few. + Per.", "Few-shot Steering", "En."),
             ("Few. + Per. + Language", "Few-shot Steering + Language", "lang")]
    keep = [ROWS[0]] + ANON_ROWS + [r for r in PUB_ROWS if r[2] >= 4]
    deltas = {k: [] for k in ("En.", "lang")}
    for country in countries:
        for col, mdir in MODEL_DIR.items():
            s = row_scores(country, mdir, keep)
            for named, anon, arm in pairs:
                a, b = s.get(named), s.get(anon)
                if not a or not b:
                    print(f"  {country:10s} {col:8s} {arm:4s} "
                          f"{'named' if not a else 'anonymous'} missing")
                    continue
                m_a = float(np.mean([a[0][q] for q in a[1]]))
                m_b = float(np.mean([b[0][q] for q in b[1]]))
                qs = sorted(set(a[0]) & set(b[0]))
                p = (float(ttest_rel([b[0][q] for q in qs], [a[0][q] for q in qs]).pvalue)
                     if len(qs) > 1 else float("nan"))
                deltas[arm].append(m_b - m_a)
                print(f"  {country:10s} {col:8s} {arm:4s} named={m_a:.4f} "
                      f"anon={m_b:.4f} delta={m_b - m_a:+.4f}{stars(p)}")
    for arm, ds in deltas.items():
        if ds:
            print(f"\n{arm:4s} mean delta over {len(ds)} (country, LLM) cells: "
                  f"{np.mean(ds):+.4f}   anonymous higher in {sum(d > 0 for d in ds)}")
    # the claim is that the name is worth less once the questionnaire is in the target
    # language, which is a statement about the two arms of the same (country, LLM) cell
    en, la = deltas["En."], deltas["lang"]
    if len(en) == len(la) and en:
        from scipy.stats import wilcoxon
        d = np.array(la) - np.array(en)
        print(f"\nlang delta minus En. delta: {np.mean(d):+.4f} over {len(d)} cells, "
              f"positive in {int((d > 0).sum())}, p={wilcoxon(d).pvalue:.4g} (Wilcoxon)")
    report_dropped()


def size_rows():
    """The two computed rows of the appendix size table, as LaTeX.

    That table is written by hand in the manuscript and covers one Qwen2.5 ladder on two
    countries, so only the rows are emitted here; the scoring is the same code the other
    tables use, which is the reason this lives here rather than in a script of its own.
    """
    sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "72B"]
    dirs = {s: f"Qwen/Qwen2.5-{s}-Instruct" for s in sizes}
    keep = [ROWS[0]] + ANON_ROWS
    # Every cell of this table is a mean over the questions the model answered in a
    # parseable form, so a cell whose model failed on some of them is computed the same
    # way as the rest -- just over fewer questions. That is why nothing is dropped here.
    # It is not free, though: Qwen2.5-1.5B replies to many of these prompts with a single
    # option ("1. Mentioned") instead of a distribution, and 116 of its 564 stayed that
    # way through 25 sampled attempts. Restricting the models that answered everything to
    # the subset 1.5B managed moves their scores by up to +0.0135 and -0.0270, with the
    # sign varying by model, so the coverage of the thin cells is reported in the caption
    # rather than left for the reader to assume.
    MIN_COVERAGE = 0.90                     # only decides what gets flagged, not printed
    for country in ["China", "Germany"]:
        print(f"\n%% {country}")
        cells, counts = {}, {}
        for s in sizes:
            scores = row_scores(country, dirs[s], keep)
            for label, _, _ in ANON_ROWS:
                cells[(label, s)] = _cell(scores, label, PARTNER[label])
                counts[(label, s)] = len(scores[label][0]) if scores.get(label) else 0
        for label, _, _ in ANON_ROWS:
            full = max(counts[(label, s)] for s in sizes)
            for s in sizes:
                if counts[(label, s)] < MIN_COVERAGE * full:
                    print(f"%% thin cell: {label} at {s} is {counts[(label, s)]}/{full} "
                          f"questions ({counts[(label, s)] / full:.0%})")
        for j, (label, _, _) in enumerate(ANON_ROWS):
            row = [cells[(label, s)] for s in sizes]
            avg = np.nanmean([value_of(c) for c in row])
            other = np.nanmean([value_of(cells[(ANON_ROWS[1 - j][0], s)]) for s in sizes])
            a = r"\textbf{%.4f}" % avg if avg > other else f"{avg:.4f}"
            head = HEADS[4 + j] % LANG_OF[country] if "%s" in HEADS[4 + j] else HEADS[4 + j]
            print(f"{head}& " + " & ".join(row) + f" & {a}" + r" \\")


def report_dropped():
    if DROPPED:
        print(f"\ndropped {len(DROPPED)} answer(s) carrying a negative probability:")
        for path, q in sorted(set(DROPPED)):
            print(f"  {q:6s} {path.replace(AAAI + '/', '')}")


# ------------------------------------------------ splice it into the published table
def value_of(cell):
    m = re.search(r"0\.\d+", cell)
    return float(m.group(0)) if m else float("nan")


def blocks(path):
    """-> {country: [the six data rows as they stand in that file]}."""
    out, cur = {}, None
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.search(HEAD_RE, line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            continue
        if cur is None or "&" not in line:
            continue
        cells = [c.strip() for c in line.split("&", 1)[1].strip().rstrip("\\").split("&")]
        if len(cells) == len(OLD_COLS):
            out[cur].append(line)
    return {k: v for k, v in out.items() if len(v) == N_PUBLISHED}


def render(countries, src, new, anon, label):
    """One finished table for the given countries."""
    keep = [OLD_COLS.index(REPLACES if c == "GPT-4.1" else c)
            for c in NEW_COLS if c != "AVG."]
    at = NEW_COLS.index("GPT-4.1")
    n_body = len(NEW_COLS) + 1
    # [t], not [tp]. [tp] was needed when one table held seven country blocks, taller
    # than the top area of a page. With three or four blocks the table is about half a
    # page, and jnlp_3.8.cls sets \floatpagefraction to .5, so a `p` lets LaTeX give it a
    # float page of its own -- which it tries before the top of the next page -- and
    # leave the bottom half of that page empty. [t] puts the text under it instead.
    lines = [r"\begin{table*}[t]", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{4pt}",
             r"    \begin{tabular}{l" + "l" * len(NEW_COLS) + "}", r"    \toprule",
             r"    \textbf{Method} & "
             + " & ".join(f"\\textbf{{{c}}}" for c in NEW_COLS) + r" \\"]
    for country in countries:
        lines += [r"    \midrule",
                  # the band is a pale blue (set by hand in the manuscript), light enough
                  # that the italic country name and the rules either side still read first
                  r"    \rowcolor[HTML]{EDF1FF} \multicolumn{%d}{c}{ \textit{%s}} \\"
                  % (n_body, country), r"    \midrule"]
        # first pass: the cells and the AVG of every row, so a row can be bolded against
        # its partner; second pass: emit, with a rule between the pairs
        parsed = []
        for ri, (row_label, _, pub) in enumerate(ROWS):
            if pub is None:
                # a computed row carries no published cells, so nothing is spliced
                kept = [anon.get((country, row_label, c), "--")
                        for c in NEW_COLS if c != "AVG."]
            else:
                cells = [c.strip() for c in
                         src[country][pub].split("&", 1)[1].strip().rstrip("\\").split("&")]
                kept = [cells[i] for i in keep]
                kept[at] = new.get((country, pub), "--")
            head = HEADS[ri] % LANG_OF[country] if "%s" in HEADS[ri] else HEADS[ri]
            parsed.append((head, kept, np.nanmean([value_of(c) for c in kept])))
        for ri, (head, kept, avg) in enumerate(parsed):
            other = parsed[ri - 1 if ri % 2 else ri + 1][2]
            cell = r"\textbf{%.4f}" % avg if avg > other else f"{avg:.4f}"
            lines.append(f"{head}& " + " & ".join(kept) + f" & {cell}" + r" \\")
            if ri % 2 and ri < len(parsed) - 1:
                # a \midrule between the pairs, as the manuscript has it by hand
                lines.append(r"    \midrule")
    who = ", ".join(countries[:-1]) + " and " + countries[-1]
    if label == "tab:steerability_results":
        # the one table in the main text carries the conventions; the other two are the
        # same table for other countries and only say which countries, then point here
        caption = (r"Alignment scores for " + who
                   + r" under different steering methods. ``()'' gives the prompt "
                     r"language: ``En.'' for English, otherwise that block's country's "
                     r"language. ``Few-shot Steering'' shows the country's answer "
                     r"distributions alone; ``Few. + Per.'' also names the country "
                     # a caption is a moving argument and this class expands \cite inside
                     # one into something carrying a parameter token, stopping the build
                     r"through a persona \protect\cite{meister-etal-2025-benchmarking}. "
                     r"AVG. is the mean over the six LLMs. $t$-test against No Steering: "
                     r"* $p<0.05$, ** $p<0.01$, *** $p<0.001$.")
    else:
        caption = (r"Alignment scores for " + who + r" under different steering methods. "
                   r"Columns, conditions and significance markers are as in "
                   r"Table~\ref{tab:steerability_results}.")
    lines += [r"    \bottomrule", r"    \end{tabular}",
              r"    \caption{" + caption + "}",
              r"    \label{%s}" % label, r"\end{table*}"]
    return "\n".join(l for l in lines if l is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--size-rows", action="store_true",
                    help="print the two computed rows of the appendix size table")
    args = ap.parse_args()

    if args.size_rows:
        size_rows()
        report_dropped()
        return

    src = blocks(PUBLISHED)
    if os.path.exists(APPENDIX):
        src.update(blocks(APPENDIX))
    # The journal version keeps only the first table; the other two need the country
    # blocks of the conference appendix, which this repository does not carry.
    tables = [t for t in TABLES if all(c in src for c in t[2])]
    skipped = [t[0] for t in TABLES if t not in tables]
    if skipped:
        print("skipped (no published block):", ", ".join(skipped))
    wanted = [c for _, _, cs in tables for c in cs]
    print("source blocks:", ", ".join(sorted(src)))

    if args.check:
        compare_fewshot(wanted)
        return

    new = new_column(wanted)
    anon = anon_rows(wanted)

    for name, label, countries in tables:
        write_table(name, render(countries, src, new, anon, label))
    print(f"replaced {REPLACES} by {NEW_MODEL}, dropped DS-V3 and DS-R1, "
          f"added {len(ROWS) - N_PUBLISHED} rows, "
          f"recomputed AVG over {len(NEW_COLS) - 1} LLMs")


if __name__ == "__main__":
    main()
