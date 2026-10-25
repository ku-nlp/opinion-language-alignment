r"""Figure for the target-group subsection.

  figures/speaker_gain.pdf   per-group gain of writing the questionnaire in a language,
                             for the three languages that have a candidate home country

What the reader has to be able to do here is follow one LLM across the rows of a panel,
because \S5.2 compares groups inside a model and says in so many words that the mean over
the models is not what the argument rests on. So the markers carry the colour and the bars,
which are that mean, are drawn in grey as context. Colour is not spent on the home country
either: the star in the tick label already marks it, and colouring it would have said the
opposite of what the Russian panel shows, where Ukraine sits above Russia.

  python plot/speaker_gain.py
"""
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (MODEL_COLOR, MODELS, MODEL_SHORT, ROOT, TEXTWIDTH, plt,
                         save, shown)  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402

# star = the language's home country. Official status will not do as the marker: Chinese
# is also official in Singapore, Turkish in Cyprus, and Russian in Kyrgyzstan and
# Kazakhstan, so it does not separate the starred row from the rest.
PANELS = [("zh", "Simplified Chinese", "mainland China"),
          ("tr", "Turkish", "Turkey"),
          ("ru", "Russian", "Russia")]

BAR, RULE = "#d9d9d9", "#333333"
# paper_style.MODEL_COLOR, the one colour per LLM every model-vs-model figure shares.
# This is the figure that needs six *categories* told apart rather than a scale, which is
# why the set is categorical and not the blue-to-yellow ramp of Figure 2: a six-step ramp
# puts four of the models in mid-tones that cannot be followed from row to row, which is
# exactly what the reader has to do here. Shapes are paired against the hues so that the
# two blues and the two oranges are never told apart by hue alone.
MARKER = {"gpt-3.5-turbo": "o", "gpt-4.1": "s", "gpt-5": "v",
          "aya-23-35b": "D", "llama3-70b": "^", "qwen2.5-72b": "P"}
STYLE = {m: (MODEL_COLOR[m], MARKER[m]) for m in MARKER}
# "X" was tried here and lost: at this size a white edge eats a glyph made of thin arms,
# so the marker read as an outline while the other five read as solid.
MS = 4.6


def main():
    d = pd.read_csv(f"{ROOT}/results/speaker_rank.csv")
    g = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in d.itertuples():
        g[r.lang][r.country][r.model] = float(r.gain)

    fig, axes = plt.subplots(
        3, 1, figsize=(TEXTWIDTH, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 4], "hspace": 0.34,
                     "left": 0.20, "right": 0.78, "top": 0.96, "bottom": 0.18})
    for ax, (lg, title, home) in zip(axes, PANELS):
        groups = g[lg]
        mean = {c: float(np.mean(list(v.values()))) for c, v in groups.items()}
        order = sorted(mean, key=lambda c: mean[c])   # best on top
        tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        for i, c in enumerate(order):
            ax.barh(i, mean[c], height=.5, zorder=2, color=BAR)
            for m in MODELS:
                col, mk = STYLE[m]
                ax.plot(groups[c][m], i, mk, ms=MS, zorder=4,
                        color=col, mec="white", mew=.6)
            ax.text(1.12, i, f"{mean[c]:+.4f}", transform=tr, va="center",
                    ha="right")
        ax.axvline(0, color=RULE, lw=.9, zorder=3)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([("$\\star$ " if c == home else "") + shown(c) for c in order])
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylim(-.6, len(order) - .4)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.88", lw=.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    axes[0].set_xlim(-0.052, 0.052)
    top = mtrans.blended_transform_factory(axes[0].transAxes, axes[0].transData)
    axes[0].text(1.12, 1.62, "mean", transform=top, ha="right", va="bottom")
    axes[-1].set_xlabel("gain over an English questionnaire, $G(lang, g)$")

    marks = [plt.Line2D([], [], ls="", marker=STYLE[m][1], color=STYLE[m][0],
                        mec="white", mew=.6, ms=MS) for m in MODELS]
    fig.legend(marks, [MODEL_SHORT[m] for m in MODELS],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6,
               frameon=False, handlelength=.9, columnspacing=1.2, handletextpad=.4)
    # exactly TEXTWIDTH wide, so \includegraphics[width=\textwidth] does not rescale
    # the type -- see the note on save() in paper_style
    save(fig, "speaker_gain", tight=False)


if __name__ == "__main__":
    main()
