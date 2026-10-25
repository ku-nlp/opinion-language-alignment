"""Figure for the channel-geometry subsection, in L1 distance.

  figures/fig_l2minmaxdist.pdf   L2 distance between the three channel displacements,
                                 min-max scaled per LLM over the plotted layers

  python plot/l1dist.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import MODEL_LABEL, ROOT, TEXTWIDTH, plt, save, uniform_font  # noqa

ORDER = ["aya-23-35b", "llama3-70b", "qwen2.5-72b"]
# drawn back to front, so the curve the text argues about ends up on top.
# language+persona and language+few-shot land almost on top of each other, which is the
# result rather than a plotting problem. The amber one underneath is therefore drawn wide
# and semi-transparent and the other is a thin dashed line in a clearly different hue:
# a red dash that stays inside an amber band for the whole depth of the model is harder
# to dismiss as two curves of one colour being hard to tell apart, and wherever the two
# do come apart the red leaves the band and is seen leaving it.
# key, label, colour, linestyle, linewidth, alpha
CURVES = [("persona-fewshot", "persona vs. few-shot", "#2DACE1", "-", 1.6, 1.0),
          ("lang-persona", "language vs. persona", "#f0b429", "-", 2.6, 0.55),
          ("lang-fewshot", "language vs. few-shot", "#D62728", "--", 1.1, 1.0)]


def fig_dist(d, metric):
    d = d[d.metric == metric]

    with uniform_font():
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15), sharey=False,
                                 layout="constrained")
        for ax, m in zip(axes, ORDER):
            g = d[d.model == m]
            for z, (key, lab, col, ls, lw, al) in enumerate(CURVES):
                # layer 0 is the embedding: nothing has been computed there and the three
                # channels lie on top of each other, which only compresses the axis
                s = g[(g.pair == key) & (g.layer > 0)].sort_values("layer")
                # +-1 SD over the 12 languages the curve is the mean of. The column is
                # already in l1dist.csv (population SD, n = 12), so nothing is recomputed.
                ax.fill_between(s["layer"], s["d1"] - s["sd"], s["d1"] + s["sd"],
                                color=col, alpha=.15, lw=0, zorder=2)
                ax.plot(s["layer"], s["d1"], ls, lw=lw, alpha=al, color=col,
                        label=lab, zorder=3 + z)
            ax.set_title(MODEL_LABEL[m], pad=2)
            ax.set_xlabel("layer")
        # what the distance is between is the caption's job, and "displacements" was
        # not even the word the text uses for it ("representation change vectors")
        axes[0].set_ylabel("$L_2$ distance")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=3, frameon=False,
                   handlelength=1.6, columnspacing=1.6, borderpad=0.0)
        save(fig, "fig_l2dist")


def main():
    d = pd.read_csv(f"{ROOT}/results/l1dist.csv")
    # Min-max, per model over all three curves at once: (X - Xmin) / (Xmax - Xmin) taken
    # over the layers that are plotted. Sharing one min and one max across the three
    # curves is what keeps their relative heights, which a per-curve min-max would lose.
    mm = d[(d.metric == "l2_raw") & (d.layer > 0)].copy()
    g = mm.groupby("model")["d1"]
    mm["d1"] = (mm.d1 - g.transform("min")) / (g.transform("max") - g.transform("min"))
    mm["metric"] = "l2_minmax"

    d = pd.concat([d, mm], ignore_index=True)
    fig_dist(d, "l2_raw")


if __name__ == "__main__":
    main()
