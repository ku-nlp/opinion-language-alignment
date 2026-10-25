"""Redraw the figures of the Default Opinion Alignment section for the current LLMs.

The conference version drew these from show_results/*.ipynb in the AAAI repo with a
model list that included the DeepSeek family and gpt-4-0613. This script reproduces the
same four figures from the same result files, but takes the model list from
config.MODELS_PAPER so that every figure in the journal version shows the same six LLMs.

The scoring follows the notebooks exactly: only Opinion-Dependent questions, and the
five questions used as few-shot examples in that setting (Q60, Q70, Q90, Q110, Q130) are
dropped, so the numbers here reproduce the published ones.

  figures/compare_llms.pdf         alignment score of each LLM with each country
  figures/compare_countries.pdf    the first and last 6 countries for the anchor LLM
  figures/compare_llms_humans.pdf  LLM versus the average human opinion distribution
  figures/human_performance.pdf    alignment among countries, with the anchor LLM

  python plot/default_alignment.py
  python plot/default_alignment.py --anchor gpt-5
"""
import argparse
import ast
import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import wasserstein_distance

# The three panels of the global-dimension figure sit side by side at 0.32 of \textwidth
# each, separated by \hfill. Drawing them four times that size and letting LaTeX shrink
# them is what made the type illegible, so each is drawn at the width it is printed at and
# all three share one height, which also makes the row line up. Keep these three numbers
# and the three subfigure widths in epaper_3.8.tex equal, or the type size stops matching
# across the row. Panel (a) used to take 0.37 against 0.308 for the other two; the swarm
# had air to spare while (b) lost half its box to country names and (c)'s labels collided.


PANEL_H = 1.55
PANEL_W = {"a": 0.32 * 420 / 72, "b": 0.32 * 420 / 72, "c": 0.32 * 420 / 72}
# a panel is barely two inches wide, so the type of the shared style would overrun it
PANEL_RC = {"font.size": 5, "axes.labelsize": 5, "xtick.labelsize": 4.5,
            "ytick.labelsize": 4.5, "legend.fontsize": 4.5,
            "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
            "axes.labelpad": 1.5,
            # matplotlib's 0.8pt spines are sized for its 10pt type; next to 4.5pt tick
            # labels they read as a frame drawn in marker, so the lines scale down with it
            "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "xtick.major.size": 2.0, "ytick.major.size": 2.0}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (C, BLUE_ORANGE, HEAT, MODEL_LABEL, MODEL_SHORT, ROOT,
                         ink_on, plt, save, shown)                           # noqa: E402

csv.field_size_limit(10 ** 7)

# where each LLM's unsteered answers live inside the AAAI results tree
# The model-size appendix (Figure 10) is the same swarm for one family at six sizes.
# 32B has results too but is left out: the section's text names the five smaller models
# it evaluated, and 72B is the one already in the main figures.
SIZE_PATH = {s: f"Qwen/Qwen2.5-{s}-Instruct"
             for s in ["0.5B", "1.5B", "3B", "7B", "14B", "72B"]}

RESULT_PATH = {
    "aya-23-35b": "CohereLabs/aya-23-35B", "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct", "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4.1": "gpt-4.1", "gpt-5": "gpt-5",
}
# the notebooks drop these five: they are the few-shot examples of the unsteered prompt
EXAMPLE_IDS = {"Q60", "Q70", "Q90", "Q110", "Q130"}


def parse_response(s):
    """The notebooks' parser, kept as is so the numbers reproduce."""
    try:
        d = ast.literal_eval(s)
    except Exception:
        try:
            d = ast.literal_eval(re.sub(r"(\d+):\s*([\d.]+%)", r"'\1': '\2'", s))
        except Exception:
            return None
    try:
        return {str(k): float(str(v).strip("%")) / 100 for k, v in d.items()}
    except Exception:
        return None


def alignment(d1, d2):
    """-> (mean, std) over the questions both sides have, in the paper's metric."""
    scores = per_question(d1, d2)
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def per_question(d1, d2):
    """-> the alignment of each question the two sides share, which is what the mean and
    the standard deviation in the figures are taken over and what the t-test needs."""
    scores = []
    for q, p1 in d1.items():
        p2 = d2.get(q)
        if p2 is None or len(p1) != len(p2):
            continue
        n = len(p1)
        sup = np.arange(1, n + 1)
        a = [v for _, v in sorted(p1.items(), key=lambda kv: int(kv[0]))]
        b = [v for _, v in sorted(p2.items(), key=lambda kv: int(kv[0]))]
        scores.append(1 - wasserstein_distance(sup, sup, a, b) / (n - 1))
    return scores


def load_llm(model):
    path = f"{C.AAAI_RESULTS}/RQ1/{RESULT_PATH[model]}/results.csv"
    if not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["type"] != "Opinion-Dependent" or r["id"] in EXAMPLE_IDS:
            continue
        v = parse_response(r["response"])
        if v:
            out[r["id"]] = v
    return out


def load_humans():
    """-> ({country_code: {qid: dist}}, {qid: dist} for the average human)."""
    gt = json.load(open(f"{ROOT}/data/ground_truth.json"))
    keys = {d["id"]: [str(k) for k in d["choice_keys"]]
            for d in (json.loads(l) for l in
                      open(f"{ROOT}/questions/WV7_English.jsonl", encoding="utf-8"))}
    per_country = {}
    for code in C.COUNTRY_ALL:
        g = gt.get(str(code))
        if not g:
            continue
        per_country[code] = {q: dict(zip(keys[q], v)) for q, v in g.items()
                             if q not in EXAMPLE_IDS and q in keys}
    avg, counts = {}, {}
    for d in per_country.values():
        for q, dist in d.items():
            if q not in avg:
                avg[q] = {k: 0.0 for k in dist}
                counts[q] = 0
            for k, v in dist.items():
                avg[q][k] += v
            counts[q] += 1
    avg = {q: {k: v / counts[q] for k, v in dist.items()} for q, dist in avg.items()}
    return per_country, avg


def fig_compare_llms(scores, labels=None, name="compare_llms", figsize=None,
                     rc=None, dot=1.6, rotation=30):
    """Swarm of the 66 per-country scores for each model, with the mean as a bar.

    Figure 2(a) draws it for the six LLMs inside a third of the text width; the model
    size appendix draws the same figure for the Qwen2.5 sizes at twice that, so the
    box, the type and the dot all take an argument rather than the panel's constants."""
    labels = labels or MODEL_SHORT
    with plt.rc_context(PANEL_RC if rc is None else rc):
      import seaborn as sns
      models = list(scores)
      # the full names do not fit six abreast at this width
      rows = [(labels[m], v) for m in models for v in scores[m].values()]
      import pandas as pd
      df = pd.DataFrame(rows, columns=["llm", "score"])
      fig = plt.figure(figsize=figsize or (PANEL_W["a"], PANEL_H))
      ax = sns.swarmplot(x="llm", y="score", data=df, hue="score", legend=False,
                         alpha=0.9, palette=BLUE_ORANGE, zorder=8, linewidth=0.0, size=dot)
      for i, m in enumerate(models):
          ax.hlines(np.mean(list(scores[m].values())), i - 0.4, i + 0.4,
                    color="k", lw=1.2, zorder=10)
      plt.xlabel("")
      plt.ylabel("Alignment Score")
      plt.xticks(rotation=rotation, ha="right" if rotation else "center")
      plt.grid(axis="y", color="grey", linewidth=0.5)
      plt.tight_layout()
      save(fig, name)


def stars(p):
    """The conventional three-tier significance mark; blank when the test does not reach
    0.05, so an unmarked country is one the LLM treats no differently from the average."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fig_compare_countries(scores, stds, raws, anchor, top_n=6):
  with plt.rc_context(PANEL_RC):
      from matplotlib.colors import Normalize
      from scipy.stats import ttest_1samp
      order = sorted(scores[anchor].items(), key=lambda x: -x[1])
      pick = order[:top_n] + order[-top_n:]
      names = [shown(C.COUNTRY_ALL[c]) for c, _ in pick]
      means = np.array([v for _, v in pick])
      cis = np.array([stds[anchor][c] for c, _ in pick])
      # the dashed line is the LLM's mean over every country it has a score for, not over
      # the twelve drawn here: it is the same quantity as the black bar in panel (a), and
      # it is the null value each country's questions are tested against below.
      grand = float(np.mean(list(scores[anchor].values())))
      cmap, norm = BLUE_ORANGE, Normalize(vmin=means.min(), vmax=means.max())
      colors = cmap(norm(means))
      fig, ax = plt.subplots(figsize=(PANEL_W["b"], PANEL_H))
      y = np.arange(len(names))
      for i in range(len(names)):
          ax.hlines(y[i], means[i] - cis[i], means[i] + cis[i], colors=colors[i],
                    linewidth=1.1, zorder=3)
          ax.plot(means[i], y[i], "o", markersize=2.6, color=colors[i],
                  markeredgecolor=colors[i], zorder=2)
          # one-sample t-test of the country's per-question alignments against that mean
          mark = stars(ttest_1samp(raws[anchor][pick[i][0]], grand).pvalue)
          if mark:
              ax.annotate(mark, (means[i], y[i]), textcoords="offset points",
                          xytext=(0, 2.2), ha="center", va="bottom", fontsize=4,
                          color="black", annotation_clip=False, zorder=4)
      ax.set_yticks(y)
      ax.set_yticklabels(names)
      # headroom for the first row's mark, which otherwise sits on the axis line
      ax.set_ylim(len(names) - 0.5, -0.95)
      ax.axvline(grand, linestyle="--", linewidth=0.8, color="black", zorder=1)
      ax.yaxis.grid(True, color="grey", linewidth=0.4)
      ax.locator_params(axis="x", nbins=4)      # three ticks collide at this width
      plt.tight_layout()
      _save(fig, "compare_countries")


def fig_compare_llms_humans(scores, avg_scores, anchor, label_n=10):
  with plt.rc_context(PANEL_RC):
      """Each country as a point: the average human on x, the anchor LLM on y, with y=x."""
      codes = [c for c in scores[anchor] if c in avg_scores]
      x = np.array([avg_scores[c] for c in codes])
      y = np.array([scores[anchor][c] for c in codes])
      fig, ax = plt.subplots(figsize=(PANEL_W["c"], PANEL_H))
      # coloured by the residual, which is what the panel is about. A diverging map
      # centred on y = x reads better in principle but not here: almost every country
      # sits below the line, so it spends half its range on a handful of points and
      # leaves the dense cluster in its invisible near-white middle.
      ax.scatter(x, y, s=9, c=y - x, cmap=BLUE_ORANGE, zorder=3, alpha=0.8, linewidth=0)
      lo, hi = min(x.min(), y.min()) - .01, max(x.max(), y.max()) + .01
      ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1, label="$y = x$")
      # label the extremes of the residual plus the countries the text discusses
      named = {"the United States", "Chile", "Slovakia", "Mongolia", "mainland China",
               "Myanmar", "Egypt", "Libya"}
      res = y - x
      show = set(np.argsort(res)[:label_n // 2]) | set(np.argsort(-res)[:label_n // 2])
      show |= {i for i, c in enumerate(codes) if C.COUNTRY_ALL[c] in named}
      for i in show:
          ax.annotate(shown(C.COUNTRY_ALL[codes[i]]), (x[i], y[i]), fontsize=3.6,
                      xytext=(2, 1.5), textcoords="offset points")
      ax.set_xlabel("Avg. humans' opinions")
      ax.set_ylabel(MODEL_LABEL[anchor])
      ax.legend(frameon=True, fontsize=5, loc="upper left")
      plt.tight_layout()
      _save(fig, "compare_llms_humans")


def fig_human_performance(per_country, llm, anchor, scores, top_n=6, name=None):
    """Country-to-country alignment, with the anchor LLM as the last row and column.

    top_n=None keeps every country instead of the first and last few, which is the
    appendix version of the same figure."""
    order = sorted(scores[anchor].items(), key=lambda x: -x[1])
    pick = [c for c, _ in (order if top_n is None else order[:top_n] + order[-top_n:])]
    names = [shown(C.COUNTRY_ALL[c]) for c in pick] + [MODEL_LABEL[anchor]]
    dists = [per_country[c] for c in pick] + [llm]
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = alignment(dists[i], dists[j])[0]
    big = len(names) > 20
    fig, ax = plt.subplots(figsize=(11, 9.6) if big else (5.6, 4.4))
    im = ax.imshow(M, cmap=HEAT)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=90 if big else 45, ha="center" if big else "right",
                       fontsize=4.5 if big else 7)
    ax.set_yticklabels(names, fontsize=4.5 if big else 7)
    if not big:                       # 67 x 67 cells leave no room for the numbers
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color=ink_on(HEAT, im.norm, M[i, j]))
    fig.colorbar(im, ax=ax, shrink=.8)
    plt.tight_layout()
    save(fig, name or "human_performance")


# set by --per-model so save() can name a file after the LLM it was drawn for
SUFFIX = [""]


def _save(fig, name):
    return save(fig, name + SUFFIX[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="gpt-5",
                    help="the LLM the country-level figures are drawn for")
    ap.add_argument("--sizes", action="store_true",
                    help="write only figures/compare_llms_size.pdf, the same swarm for "
                         "the Qwen2.5 size sweep instead of the six LLMs")
    ap.add_argument("--per-model", action="store_true",
                    help="write one country-level pair per LLM, suffixed with its name, "
                         "for the appendix; the main-text figures are left alone")
    args = ap.parse_args()

    per_country, avg_human = load_humans()
    if args.sizes:
        RESULT_PATH.clear()
        RESULT_PATH.update(SIZE_PATH)
    llms = {m: d for m in RESULT_PATH if (d := load_llm(m))}
    missing = [m for m in RESULT_PATH if m not in llms]
    if missing:
        print(f"! no results under {C.AAAI_RESULTS}/RQ1 for: {', '.join(missing)}")
    if not args.sizes and args.anchor not in llms:
        sys.exit(f"anchor {args.anchor} has no results")

    scores, stds, raws = {}, {}, {}
    for m, d in llms.items():
        scores[m], stds[m], raws[m] = {}, {}, {}
        for code, h in per_country.items():
            qs = per_question(d, h)
            if qs:
                scores[m][code], stds[m][code] = float(np.mean(qs)), float(np.std(qs))
                raws[m][code] = qs
        print(f"{m:14s} mean over {len(scores[m])} countries: "
              f"{np.mean(list(scores[m].values())):.4f}")
    avg_scores = {c: alignment(avg_human, h)[0] for c, h in per_country.items()}

    if args.sizes:
        # drawn at 0.65\linewidth of a figure*, so it gets the shared type size rather
        # than the 5pt of a Figure 2 panel, and a dot that scales with the wider box
        fig_compare_llms(scores, labels={s: s for s in SIZE_PATH},
                         name="compare_llms_size", figsize=(0.65 * 420 / 72, 2.35),
                         rc={}, dot=2.6, rotation=0)
        return

    if args.per_model:
        # the appendix shows the same two country-level figures for every LLM; the file
        # name carries the model so the main-text pair is not overwritten
        for m in llms:
            SUFFIX[0] = "_" + m
            fig_compare_countries(scores, stds, raws, m)
            fig_compare_llms_humans(scores, avg_scores, m)
        SUFFIX[0] = ""
        # the appendix also carries the country-to-country heatmap over all 66
        fig_human_performance(per_country, llms[args.anchor], args.anchor, scores,
                              top_n=None, name="human_performance_all")
        return

    fig_compare_llms(scores)
    fig_compare_countries(scores, stds, raws, args.anchor)
    fig_compare_llms_humans(scores, avg_scores, args.anchor)
    fig_human_performance(per_country, llms[args.anchor], args.anchor, scores)

    order = sorted(scores[args.anchor].items(), key=lambda x: -x[1])
    print(f"\n{args.anchor}: top 6  "
          + ", ".join(f"{C.COUNTRY_ALL[c]} {v:.4f}" for c, v in order[:6]))
    print(f"{' ' * len(args.anchor)}  bottom 6  "
          + ", ".join(f"{C.COUNTRY_ALL[c]} {v:.4f}" for c, v in order[-6:]))


if __name__ == "__main__":
    main()
