"""Tables for the speaker-group subsection.

  tables/speaker_target.tex        the group each LLM moves towards, one row per language

The appendix that held the two tables below was cut to save page charges, and the main
text took over a one-line-per-language summary of the detailed table instead. Their
functions stay, uncalled, in case the appendix comes back:

  tables/speaker_stats.tex         the groups compared, one row per language
  tables/speaker_gain_detail.tex   every group's gain, largest first, one row per language

The quantity is a gain over the English arm, so a population that is simply easy to align
with does not win by being easy; both arms score the same people and that cancels. See
code/speaker_rank.py.

  python plot/speaker.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import C, MODELS, MODEL_SHORT, ROOT, write_table  # noqa

LANGNAME = {"ar": "Arabic", "el": "Greek", "en": "English", "es": "Spanish",
            "ru": "Russian", "tr": "Turkish", "zh": "Chinese (simpl.)",
            "zt": "Chinese (trad.)"}
# how the two Chinese rows are named in the manuscript's table
ROWNAME = dict(LANGNAME, zh="Simplified Chinese", zt="Traditional Chinese")
# The country where the language is the national one, where WV7 surveys it. Read only to
# say whether the population the language moves towards is that country; nothing is
# scored against it. Chinese has one homeland but two questionnaires, and Spanish has
# none, since Spain is not in WV7.
HOME = {"ar": None, "el": 300, "en": 840, "es": None,
        "ru": 643, "tr": 792, "zh": 156, "zt": None}
# English is absent on purpose: its `language` arm is the English baseline
# itself, so its gain is zero by construction. See code/speaker_rank.py.
ORDER = ["zh", "zt", "ru", "tr", "el", "es", "ar"]


def load(model=None):
    """-> one row per (language, population); averaged over the LLMs, or for one of them."""
    d = pd.read_csv(f"{ROOT}/results/speaker_rank.csv")
    if model:
        d = d[d.model == model]
    g = (d.groupby(["lang", "code", "country"])
          .agg(gain=("gain", "mean"), pos=("gain", lambda s: int((s > 0).sum())),
               sig=("lo", lambda s: 0), n=("gain", "size"))
          .reset_index())
    # a cell counts as significant when its bootstrap interval clears zero
    d["is_sig"] = (d.lo > 0) | (d.hi < 0)
    s = d.groupby(["lang", "code"])["is_sig"].sum().rename("sig")
    return g.drop(columns="sig").merge(s, on=["lang", "code"])




def top_agreement(lang):
    """How many of the six LLMs put the same population first as the average does."""
    d = pd.read_csv(f"{ROOT}/results/speaker_rank.csv")
    d = d[d.lang == lang]
    avg_top = (d.groupby("code")["gain"].mean().idxmax())
    per = d.loc[d.groupby("model")["gain"].idxmax()]
    return int((per.code == avg_top).sum()), int(per.model.nunique())


# the detailed table has six columns of these, so the longest names set its width
DETAIL_SHORT = {"New Zealand": "N.Zeal.", "Kyrgyzstan": "Kyrgyz.",
                "Kazakhstan": "Kazakh.", "Guatemala": "Guatem.",
                "Nicaragua": "Nicarag.", "Puerto Rico": "P.Rico",
                "Argentina": "Argent.", "Venezuela": "Venez.", "Colombia": "Colomb.",
                "Singapore": "Singap.", "Australia": "Austral.", "Hong Kong": "HK",
                "Great Britain": "Britain", "Northern Ireland": "N.Irel.",
                "mainland China": "China", "the United States": "USA"}
SHORT = {"mainland China": "China", "the United States": "USA",
         "Great Britain": "Britain", "Northern Ireland": "N.~Ireland",
         "Puerto Rico": "P.~Rico"}


def tab_gain_detail():
    """Table 3 with every group in the cell, not only the one that comes first.

    Same rows and columns, so the two can be read against each other; what changes is
    that a cell lists all of that language's groups with the gain that LLM gives them,
    largest first. The main table says which group wins, this one says by how much and
    what the rest look like -- including that for Spanish, Arabic and Greek the winner
    of a column can still have a negative gain.
    """
    d = pd.read_csv(f"{ROOT}/results/speaker_rank.csv")
    langs = [l for l in ORDER if l in set(d.lang)]
    rows = [r"\begin{table*}[p]", r"    \centering", r"    \tiny",
            r"    \setlength{\tabcolsep}{1pt}",
            # a cell is a stack of groups, so the columns need ruling apart or the eye
            # reads across the gaps between one language's lines into the next column
            r"    \begin{tabular}{@{}" + "|".join("l" * (len(MODELS) + 1)) + "@{}}",
            r"    \toprule",
            r"    \textbf{Language} & "
            + " & ".join(f"\\textbf{{{MODEL_SHORT[m]}}}" for m in MODELS) + r" \\",
            r"    \midrule"]
    for i, lang in enumerate(langs):
        cells = []
        for m in MODELS:
            s = d[(d.lang == lang) & (d.model == m)].sort_values("gain", ascending=False)
            items = [f"{DETAIL_SHORT.get(r.country, SHORT.get(r.country, r.country.replace('the ', '')))}"
                     f"~${r.gain:+.4f}$" for _, r in s.iterrows()]
            cells.append(r"\makecell[l]{" + r" \\ ".join(items) + "}")
        if i:
            rows.append(r"    \midrule")
        rows.append(f"    {ROWNAME[lang]} & " + " & ".join(cells) + r" \\")
    rows += [r"    \bottomrule", r"    \end{tabular}",
             r"    \caption{Every group that answered in a language, with the gain of "
             r"Eq.~\eqref{eq:speaker_gain} that each LLM gives it, largest first, so the "
             r"first line of a cell is the target group of that language for that LLM. ",
             r"    Country or region names are abbreviated to fit the six columns. ",
             r"    }",
             r"    \label{tab:speaker_gain_detail}", r"\end{table*}"]
    write_table("speaker_gain_detail", "\n".join(rows))


# Table 2's column heads, as the manuscript has them (the names of Table 1), not the
# MODEL_SHORT of the figures
TARGET_HEAD = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4.1": "GPT-4.1", "gpt-5": "GPT-5",
               "aya-23-35b": "Aya23", "llama3-70b": "Llama3", "qwen2.5-72b": "Qwen2.5"}


def tab_target():
    """Which group each LLM moves towards, for all seven languages, in the main text.

    The first line of every cell of tab_gain_detail(), plus whether that group's lead over
    the runner-up is more than noise: * marks the paired t-test of code/speaker_rank.py
    (p_gap of the rank-2 row) at p < 0.05. The home country is named once, in parentheses
    after the language, rather than starred in the cells. Reproduces the manuscript's
    Table 2 (tables/speaker_target.tex) byte for byte, caption included.
    """
    d = pd.read_csv(f"{ROOT}/results/speaker_rank.csv")
    langs = [l for l in ORDER if l in set(d.lang)]
    rows = [r"\begin{table}[t]", r"    \centering", r"    \scriptsize",
            r"    \setlength{\tabcolsep}{4pt}",
            r"    \begin{tabular}{@{}l|" + "l" * len(MODELS) + "@{}}",
            r"    \toprule",
            r"    \textbf{Language} & "
            + " & ".join(f"\\textbf{{{TARGET_HEAD[m]}}}" for m in MODELS) + r" \\",
            r"    \midrule"]
    for lang in langs:
        x = d[d.lang == lang]
        cells = []
        for m in MODELS:
            s = x[x.model == m].sort_values("rank")
            top, second = s.iloc[0], s.iloc[1]
            name = SHORT.get(top.country, top.country)
            if second.p_gap < 0.05:
                name += r"$^{*}$"
            cells.append(r"\makecell[l]{" + name + r" \\ " + f"${top.gain:+.4f}$" + "}")
        home = HOME.get(lang)
        label = ROWNAME[lang]
        if home is not None:
            hname = C.COUNTRY_ALL[home]
            label += f" ({SHORT.get(hname, hname)})"
        rows.append(f"    {label} & " + " & ".join(cells) + r" \\")
    rows += [r"    \bottomrule", r"    \end{tabular}",
             r"    \caption{Target groups of language steering, defined as the country- or "
             r"region-level groups with the largest alignment gains. ",
             r"    In each cell, the first line names the target group and the second "
             r"reports its gain. ",
             r"    The language's home country is given in parentheses when it is "
             r"represented in the WVS; ",
             r"    Traditional Chinese and Arabic are not assigned a home country because "
             r"both are shared across multiple countries and regions; Spanish has no "
             r"home-country label because Spain is not surveyed in wave 7.",
             r"    * indicates that the target gain is significantly larger than the gain "
             r"of the second-ranked group ($p<0.05$, paired $t$-test over questions).}",
             r"    \label{tab:speaker_target}", r"\end{table}"]
    write_table("speaker_target", "\n".join(rows))


def tab_stats():
    """Every (country, interview language) group the subsection works with.

    Counted from LNGE_ISO rather than assumed: the number of countries that answered in
    a language is not something the survey states anywhere, and it is what decides which
    languages the question can be asked of at all.
    """
    import pandas as pd
    df = pd.read_csv(C.WVS_CSV, low_memory=False, usecols=["B_COUNTRY", "LNGE_ISO"])
    big = df.groupby(["LNGE_ISO", "B_COUNTRY"]).size().reset_index(name="n")
    big = big[big.n >= C.MIN_CELL_N]
    keep = {l for l, g in big.groupby("LNGE_ISO") if len(g) > 1}

    def block(lang, codes):
        """The group names, largest first. The respondent counts still order them, they
        are just not printed."""
        s = big[(big.LNGE_ISO == lang) & (big.B_COUNTRY.isin(codes))] \
            .sort_values("n", ascending=False)
        full = [C.COUNTRY_ALL.get(int(c), str(c)) for c in s.B_COUNTRY]
        return ", ".join(SHORT.get(n, n).replace("the ", "") for n in full)

    rows = [r"\begin{table*}[t]", r"    \centering", r"    \scriptsize",
            r"    \setlength{\tabcolsep}{4pt}",
            r"    \begin{tabular}{@{}l p{11.6cm}@{}}", r"    \toprule",
            r"    \textbf{Language} & \textbf{Groups} \\", r"    \midrule"]
    # ORDER omits English: ten countries answered in it, so it meets the selection rule,
    # but its prompt is the English baseline itself and its gain is zero by
    # construction. The Method says so; listing it here would only invite the question.
    for lang in ORDER:
        src = "zh" if lang in ("zh", "zt") else lang
        if src not in keep:
            continue
        codes = ([c for c, s in C.ZH_SCRIPT.items() if s == lang]
                 if lang in ("zh", "zt") else
                 list(big[big.LNGE_ISO == lang].B_COUNTRY))
        rows.append(f"    {ROWNAME[lang]} & {block(src, codes)} \\\\")
        rows.append(r"    \addlinespace[2pt]")
    rows += [r"    \bottomrule", r"    \end{tabular}",
             r"    \caption{Country- or region-level respondent groups that answered "
             r"the WVS questionnaire in each language.}",
             r"    \label{tab:speaker_stats}", r"\end{table*}"]
    write_table("speaker_stats", "\n".join(rows))


def console(g):
    for lang in [l for l in ORDER if l in set(g.lang)]:
        s = g[g.lang == lang].sort_values("gain", ascending=False)
        home = HOME.get(lang)
        rank = ([c for c in s.code].index(home) + 1) if home in set(s.code) else None
        print(f"{LANGNAME[lang]:18s} n={len(s):2d}  best {s.iloc[0].country:18s} "
              f"{s.iloc[0].gain:+.4f}   homeland rank: {rank}")


def main():
    g = load()
    tab_target()
    console(g)


if __name__ == "__main__":
    main()
