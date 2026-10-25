"""Redraw the temporal figure (figures/waves_alignment_scores.pdf) for the current LLMs.

The conference version drew it from show_results/rq3_show_result_all_models.ipynb, whose
panels were Llama3, Qwen2.5, GPT-4, GPT-5 and DeepSeek-R1. The journal version drops the
DeepSeek family and replaces gpt-4-0613 by gpt-4.1, which leaves four panels.

A panel exists for an LLM only when Eq. (3) selects at least five countries, the rule the
threshold was chosen for. With tau = 0.02 that holds for Llama3, Qwen2.5, GPT-4.1 and
GPT-5; GPT-3.5-Turbo selects none and Aya-23-35B selects four, which is why neither
appeared in the published figure either.

Every LLM's country set is computed with Eq. (3) from the current data. The sets printed
in the conference version for Llama3, Qwen2.5 and GPT-5 used to be kept verbatim so those
panels would not move, but they are not what Eq. (3) gives: Llama3's lacked the United
States, Qwen2.5's Hong Kong, and GPT-5's had Serbia where Eq. (3) selects Chile, Mongolia
and Singapore. The figure then did not follow the rule the text states, so it no longer
reproduces the conference panels.

Scoring follows the notebook: the 75 questions common to waves 5-7, minus the five used
as few-shot examples, with the per-wave question ids taken from the id triple in
WV567.jsonl.

  python plot/temporal.py
"""
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import default_alignment as D                                       # noqa: E402
import matplotlib.colors  # noqa: E402
from paper_style import (C, MODEL_COLOR, MODEL_LABEL, plt, save,
                         uniform_font)  # noqa: E402

AAAI = C.AAAI_REPO            # for dataset/questions/WV567.jsonl
TAU = 0.02
MIN_COUNTRIES = 5
# the five few-shot examples of the unsteered prompt, dropped from scoring
EXAMPLE_IDS = {"Q60", "Q70", "Q90", "Q110", "Q130"}
CANDIDATES = ["aya-23-35b", "llama3-70b", "qwen2.5-72b", "gpt-3.5-turbo", "gpt-4.1",
              "gpt-5"]
MARKER = {"aya-23-35b": "*", "llama3-70b": "o", "qwen2.5-72b": "v",
          "gpt-3.5-turbo": "P", "gpt-4.1": "s", "gpt-5": "^"}
# paper_style.MODEL_COLOR: each model keeps the colour it has in Figure 4, so a reader
# who has learnt the key there reads this figure without the panel titles.


def wave_question_ids():
    """-> ({qid: keys} for wave 7, 6, 5), from the id triple in WV567.jsonl."""
    w7, w6, w5 = {}, {}, {}
    for line in open(f"{AAAI}/dataset/questions/WV567.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if item["type"] != "Opinion-Dependent" or item["id"][0] in EXAMPLE_IDS:
            continue
        w7[item["id"][0]] = item["choice_keys"]
        w6[item["id"][1]] = item["choice_keys"]
        w5[item["id"][2]] = item["choice_keys"]
    return w7, w6, w5


def llm_answers(model, qids):
    df = pd.read_csv(f"{C.AAAI_RESULTS}/RQ1/{D.RESULT_PATH[model]}/results.csv",
                     low_memory=False)
    df = df[df["type"] == "Opinion-Dependent"]
    out = {}
    for q in qids:
        rows = df[df["id"] == q]["response"].values
        if not len(rows):
            continue
        try:
            d = ast.literal_eval(rows[0])
        except Exception:
            try:
                d = ast.literal_eval(re.sub(r"(\d+):\s*([\d.]+%)", r"'\1': '\2'", rows[0]))
            except Exception:
                continue
        out[q] = {str(k): float(str(v).strip("%")) / 100 for k, v in d.items()}
    return out


def alignment(d1, d2):
    """The notebook zips the two dicts positionally; both are keyed by question id and
    built in the same order, so the pairing is by question."""
    scores = []
    for p1, p2 in zip(d1.values(), d2.values()):
        if len(p1) != len(p2):
            continue
        n = len(p1)
        sup = np.array([int(k) for k in sorted(p1, key=int)])
        a = [p1[k] for k in sorted(p1, key=int)]
        b = [p2[k] for k in sorted(p2, key=int)]
        scores.append(1 - wasserstein_distance(sup, sup, a, b) / (n - 1))
    return float(np.mean(scores)) if scores else float("nan")


def country_sets(all_models=False):
    """-> {model: [country codes]} for every LLM that Eq. (3) gives at least five.

    With all_models, the threshold is dropped and every LLM gets a panel drawn over all
    the countries it has results for. The appendix shows that version, so a reader can
    see the two LLMs the main text has to leave out and why."""
    per, avg = D.load_humans()
    avg_sc = {c: D.alignment(avg, h)[0] for c, h in per.items()}
    out = {}
    for m in CANDIDATES:
        d = D.load_llm(m)
        if d is None:
            print(f"  {m:14s} no results")
            continue
        sc = {c: D.alignment(d, h)[0] for c, h in per.items()}
        sel = [str(c) for c in sorted(sc) if abs(sc[c] - avg_sc[c]) < TAU]
        if all_models:
            out[m] = sel or [str(c) for c in sorted(sc)]
            print(f"  {m:14s} {len(out[m]):2d} countries (threshold dropped)")
            continue
        mark = "" if len(sel) >= MIN_COUNTRIES else "  <- fewer than five, no panel"
        print(f"  {m:14s} {len(sel):2d} countries (tau={TAU}){mark}")
        if len(sel) >= MIN_COUNTRIES:
            out[m] = sel
    return out


def wave_scores(model, codes):
    w7, w6, w5 = wave_question_ids()
    llm = llm_answers(model, w7.keys())
    out = []
    for path, qmap in [(f"{C.WVS_DIR}/WV5.csv", w5),
                       (f"{C.WVS_DIR}/WV6.csv", w6),
                       (f"{C.WVS_DIR}/WV7.csv", w7)]:
        wave = os.path.basename(path)[:3]
        sep, col = ((";", "V2") if wave in ("WV5", "WV6") else (",", "B_COUNTRY"))
        df = pd.read_csv(path, sep=sep, low_memory=False)
        per_country = []
        for code in codes:
            sub = df[df[col] == int(code)]
            if sub.empty:
                continue
            human = {}
            for q, keys in qmap.items():
                if q not in sub.columns:
                    continue
                counts = {k: int((sub[q] == int(k)).sum()) for k in keys}
                if sum(counts.values()) == 0:
                    continue
                human[q] = {k: v / sum(counts.values()) for k, v in counts.items()}
            s = alignment(llm, human)
            if not np.isnan(s):
                per_country.append(s)
        out.append(per_country)
        print(f"      {wave}: {len(per_country)} countries, "
              f"mean {np.mean(per_country):.4f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-models", action="store_true",
                    help="one panel per LLM with no country threshold, for the appendix")
    args = ap.parse_args()
    print("country sets, Eq. (3):")
    sets = country_sets(args.all_models)
    models = [m for m in CANDIDATES if m in sets]
    ncol = 2
    nrow = int(np.ceil(len(models) / ncol))
    # the notebook drew this at 6 x 1.8*nrow; four panels do not need that much page,
    # so the canvas is four fifths of it and the figure is included at 0.8\columnwidth,
    # which keeps it at its natural size instead of letting LaTeX rescale the type
    with uniform_font():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.8, 1.44 * nrow), squeeze=False)
        axes = axes.flatten()
        for i, model in enumerate(models):
            print(f"  {model}:")
            waves = wave_scores(model, sets[model])
            x = ["Wave 5", "Wave 6", "Wave 7"]
            y = [float(np.mean(w)) for w in waves]
            err = [float(np.std(w)) for w in waves]
            color = MODEL_COLOR[model]
            ax = axes[i]
            ax.plot(x, y, color=color, marker=MARKER[model], label=MODEL_LABEL[model],
                    markerfacecolor="none", markersize=10)
            ax.fill_between(x, [a - b for a, b in zip(y, err)],
                            [a + b for a, b in zip(y, err)], color=color, alpha=0.10)
            ax.set_title(MODEL_LABEL[model])
            ax.grid(axis="both", color="grey", linewidth=0.5, linestyle="-", alpha=0.5)
        for ax in axes[len(models):]:
            ax.axis("off")
        plt.tight_layout()
        save(fig, "waves_alignment_scores_all_models" if args.all_models
             else "waves_alignment_scores")


if __name__ == "__main__":
    main()
