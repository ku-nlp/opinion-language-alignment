"""Shared loading and styling for the figures and tables of the paper.

Kept separate from results.ipynb: the notebook is exploratory and its figures are sized
for a screen, these are sized for a two-column page and write straight into
figures/ and tables/ so a number can never be stale relative to the text.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the JNLP class sets \textwidth to 420pt; figures are drawn at that width so
# \includegraphics[width=\textwidth] does not rescale the type
TEXTWIDTH = 420 / 72.0
sys.path.insert(0, f"{ROOT}/code")
import config as C                                                    # noqa: E402

FIG_DIR = f"{ROOT}/figures"
TAB_DIR = f"{ROOT}/tables"

# GPT models first, then open-weight ones by family; the order is the one used in every
# table and every figure so panels can be compared across pages
MODELS = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5",
          "aya-23-35b", "llama3-70b", "qwen2.5-72b"]
MODEL_LABEL = {"gpt-3.5-turbo": "GPT-3.5-Turbo", "gpt-4.1": "GPT-4.1", "gpt-5": "GPT-5",
               "aya-23-35b": "Aya-23-35B", "llama3-70b": "Llama-3-70B",
               "qwen2.5-72b": "Qwen2.5-72B"}
MECH_MODELS = ["llama3-70b", "aya-23-35b", "qwen2.5-72b"]
# table headers only; the wide tables do not fit a two-column page at the full names
MODEL_SHORT = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4.1": "GPT-4.1", "gpt-5": "GPT-5",
               "aya-23-35b": "Aya-35B", "llama3-70b": "Llama-70B",
               "qwen2.5-72b": "Qwen-72B"}

# Ramps for the figures that colour by value: both run from the paper's blue, #2DACE1,
# to a coral orange, #E8765A, with low values blue and high values orange.
#
# BLUE_ORANGE is for Figure 2, where colour is redundant -- the beeswarm and the ranking
# plot colour each point by the very score its position already shows, and the scatter
# by a residual the y = x line already shows. Its middle is a light grey rather than
# white, because these are points on a white page: a white middle would make the dense
# cluster of mid-range countries disappear, which is also why panel (c) is not drawn
# with a diverging map centred on y = x.
#
# HEAT is for the country-to-country heat map, where colour is the only thing that shows
# the block structure. There every cell is filled, so the middle can go to near-white and
# the two blocks read as blue against orange. Earlier ramps were dropped for looking dark
# and dull on the page: cividis (navy to khaki through a desaturated slate), a blue to
# yellow ramp (#12607D ... #E9CF4F) with a dark teal low end and a khaki middle, and the
# matching blue to yellow for Figure 2 (#2DACE1 ... #E5CF52).
BLUE_ORANGE = LinearSegmentedColormap.from_list(
    "blue_orange", ["#2DACE1", "#7CC6E6", "#C9C6C6", "#F0A584", "#E8765A"])
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#2DACE1", "#A6DCF1", "#F7F7F7", "#F6C1A6", "#E8765A"])


def ink_on(cmap, norm, v):
    """Black or white, whichever the cell colour at `v` can actually be read against.
    A fixed threshold on the data cannot do this: which half of the range is dark
    depends on the ramp, and both of ours are light at both ends but not in between."""
    r, g, b = cmap(norm(v))[:3]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return "k" if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.3 else "w"


# One colour per LLM, shared by every figure that puts models side by side, so a model
# keeps its colour from page to page. Okabe-Ito, the colourblind-safe set, with its sky
# blue moved to the #2DACE1 the rest of the figures use; its orange and vermillion are
# left for the models because no figure uses them for anything else.
# How a country is named on a figure. config.COUNTRY_ALL keeps "mainland China" because
# the persona prompts and the result files use it; the paper calls code 156 "China", as
# the WVS itself does, so only the printed label changes.
DISPLAY_NAME = {"mainland China": "China"}


def shown(name):
    return DISPLAY_NAME.get(name, name)


MODEL_COLOR = {"gpt-3.5-turbo": "#0072B2",   # blue
               "gpt-4.1":       "#2DACE1",   # sky blue
               "gpt-5":         "#009E73",   # green
               "aya-23-35b":    "#E69F00",   # orange
               "llama3-70b":    "#D55E00",   # vermillion
               "qwen2.5-72b":   "#CC79A7"}   # reddish purple

LANGNAME = {"el": "Greek", "tr": "Turkish", "uk": "Ukrainian", "ru": "Russian",
            "kk": "Kazakh", "ky": "Kyrgyz", "zh": "Chinese", "en": "English",
            "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese", "de": "German",
            "pt": "Portuguese", "fr": "French"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# The size every piece of text in a figure is drawn at, where the figure asks for one.
# It is the `font.size` above, so a figure that opts in only loses the title/label/tick
# spread, it does not change scale relative to the figures that do not.
FIG_FONT = 8


def uniform_font(size=FIG_FONT):
    """Draw titles, axis labels, tick labels and legends at one size.

    The rcParams above give those four things three different sizes (8.5/8/7), which is
    matplotlib's usual hierarchy and is meant to be read at a figure that LaTeX shrinks.
    This class is single-column -- \\columnwidth is the full 420pt -- so these figures go
    in at their natural size and the spread stays visible on the page instead.

    Wrap creation, plotting and saving: a tick label takes its size when the Axes is
    made, a title when set_title runs, so leaving any of them outside the block leaves
    that text at the old size.
    """
    keys = ["font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize"]
    return plt.rc_context({k: size for k in keys})


def have(model, name):
    return os.path.exists(f"{ROOT}/results/{model}/{name}")


def read(name, models=None):
    """Concatenate one per-model results file across models, tagged with `model`."""
    ms = [m for m in (models or MODELS) if have(m, name)]
    return pd.concat([pd.read_csv(f"{ROOT}/results/{m}/{name}").assign(model=m)
                      for m in ms], ignore_index=True)


def yerr(df, value="gain"):
    """Asymmetric bootstrap whiskers, clipped at zero against float noise."""
    return np.vstack([np.maximum(df[value] - df["lo"], 0),
                      np.maximum(df["hi"] - df[value], 0)])


def sig(df):
    """95% interval excludes zero."""
    return (df["lo"] > 0) | (df["hi"] < 0)


def stars(p):
    """The paper's significance convention: * .05, ** .01, *** .001 (t-test)."""
    if p is None or np.isnan(p):
        return ""
    return "$^{***}$" if p < .001 else "$^{**}$" if p < .01 else "$^{*}$" if p < .05 else ""


def save(fig, name, pad=None, tight=True):
    """`pad` overrides savefig.pad_inches, in inches.

    Worth raising where a rotated axis label is longer than the axes it sits beside:
    savefig's tight bbox measures the text layout box, which for rotated text runs a
    couple of points short of the glyph ink, and the default 0.02in pad is not enough to
    cover the difference -- the last character comes out sliced at the page edge.

    `tight=False` writes the canvas at exactly `figsize` instead of cropping to the ink.
    Use it for a set of panels that are included side by side or stacked at one
    \\linewidth: the tight bbox depends on how far the labels happen to reach, so the
    panels come out different widths, LaTeX rescales each by a different factor, and the
    same 8pt lands on the page at a different size in every panel.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f"{FIG_DIR}/{name}.pdf"
    kw = {} if pad is None else {"pad_inches": pad}
    # savefig(bbox_inches=None) does not mean "no crop", it means "read
    # rcParams['savefig.bbox']", which is "tight" above -- so the rcParam is what has to
    # be cleared.
    with plt.rc_context({} if tight else {"savefig.bbox": None}):
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  figure  figures/{name}.pdf")


def write_table(name, body):
    os.makedirs(TAB_DIR, exist_ok=True)
    path = f"{TAB_DIR}/{name}.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body if body.endswith("\n") else body + "\n")
    print(f"  table   tables/{name}.tex")


def num(x, digits=4, plus=True, bold=False, sup=None):
    """One number as a single math group, so \\mathbf and superscripts stay inside it."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{digits}f}" if plus else f"{x:.{digits}f}"
    if bold:
        s = r"\mathbf{" + s + "}"
    if sup:
        s += "^{" + str(sup) + "}"
    return f"${s}$"
