r"""The opening example figure.

  figures/fig_demo.pdf   one WVS question, four countries' answers, and GPT-5's

The conference version drew this by hand: a schematic with invented bars, flags and
"good/bad alignment" brackets. That artwork is under the conference publisher's
copyright, so it is not reused here. This version is drawn from the data instead, which
also makes the point better: the reader sees the actual answers rather than a cartoon of
them, and the alignment score of Eq. (2) is printed for each country.

Q169 is chosen because it separates the four countries the paper discusses further on,
and because all four answered it. Nothing is normalised or smoothed.

  python plot/demo.py
"""
import ast
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from paper_style import MODEL_COLOR, ROOT, TEXTWIDTH, plt, save  # noqa: E402
import config as C  # noqa: E402

QID = "Q169"
ANCHOR = "gpt-5"
# code, display name, colour; ordered as they are drawn, best-aligned first. The four
# hues are the four of paper_style.MODEL_COLOR that Figure 3 draws, so the figures of
# this paper work from one set of colours rather than one set per figure. They are drawn
# the way Figure 3 draws its standard-deviation band rather than the way it draws its
# line: a pale wash of the hue, with the hue itself only on the outline. At full strength
# four saturated fills this size fight each other and the GPT-5 outline across them.
COUNTRIES = [("840", "the United States", MODEL_COLOR["gpt-4.1"]),
             ("152", "Chile", MODEL_COLOR["gpt-5"]),
             ("104", "Myanmar", MODEL_COLOR["qwen2.5-72b"]),
             ("818", "Egypt", MODEL_COLOR["llama3-70b"])]
# the four options are numbered 1-4 in the prompt itself, so the ticks carry the number
# and the caption spells them out; the words do not fit four abreast at this width
SHORT = ["1", "2", "3", "4"]

# the model outline is neutral rather than a fifth hue: it is the same distribution
# in all four panels, and amber was the one colour the vermillion country could be
# confused with where the two met inside a bar
MODEL, INK = "#333333", "#333333"
# how much of the hue the bar's fill keeps. Figure 3's band is the hue at alpha 0.10,
# which is too faint to carry a bar on its own; this is the same idea one stop stronger,
# and it is blended rather than made transparent so the GPT-5 outline over it is clean.
WASH = 0.22


def tint(c, a=WASH):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def alignment(p, q):
    n = len(p)
    x = np.arange(n)
    return 1 - wasserstein_distance(x, x, p, q) / (n - 1)


def main():
    questions = {json.loads(l)["id"]: json.loads(l)
                 for l in open(f"{ROOT}/questions/WV7_English.jsonl")}
    q = questions[QID]
    truth = json.load(open(f"{ROOT}/data/ground_truth.json"))

    csv.field_size_limit(10 ** 7)
    llm = None
    for r in csv.DictReader(open(C.result_file(ANCHOR))):
        if r["cond"] == "base" and r["qid"] == QID:
            d = ast.literal_eval(r["parsed"])
            v = np.array([float(str(d[k]).rstrip("%"))
                          for k in ast.literal_eval(r["choice_keys"])])
            llm = v / v.sum()
            break
    assert llm is not None and len(llm) == len(q["choices"])

    fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(TEXTWIDTH, 1.95),
                             sharey=True, layout="constrained")
    x = np.arange(len(llm))
    for ax, (code, name, col) in zip(axes, COUNTRIES):
        h = np.array(truth[code][QID], dtype=float)
        h = h / h.sum()
        ax.bar(x, h, width=.72, facecolor=tint(col), edgecolor=col, lw=.9,
               zorder=2)
        # the model is the same in every panel, so it is the outline the country's bars
        # are read against rather than a second set of bars competing with them
        bars = ax.bar(x, llm, width=.72, facecolor="none", edgecolor=MODEL, lw=1.3,
                      zorder=3, label="GPT-5")
        ax.set_title(f"{name}\n$\\mathcal{{A}} = {alignment(llm, h):.4f}$", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="0.9", lw=.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of answers")
    axes[0].set_yticks([0, .5, 1])

    # every panel has its own country colour, so the key for the filled bars is the four
    # of them side by side rather than one swatch that would name only the first panel
    swatch = tuple(Patch(facecolor=tint(c), edgecolor=c, lw=.9)
                   for _, _, c in COUNTRIES)
    fig.legend([swatch, bars[0]], ["people of the country", "GPT-5"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.6, columnspacing=1.8)
    stmt = q["question"].split(":")[-1].strip().strip("?").strip().strip("'")
    fig.suptitle(f"\u201c{stmt}\u201d", fontsize=8, color=INK)
    save(fig, "fig_demo")


if __name__ == "__main__":
    main()
