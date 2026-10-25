"""Table 4: prompt sensitivity of the LLMs.

For every LLM the default conference-format run (results/aaai/RQ1) is scored against the
66 WV7 countries, and so is each sensitivity run (results/aaai/Discussion/sensitivity):
`order` shuffles the question options, `number` limits the few-shot examples from five
to three, and `prob` changes the probability distribution of the few-shot examples. The
table reports Pearson's r between the 66 default scores and the 66 scores of each
setting, with its p-value.

  python plot/sensitivity.py
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from default_alignment import (EXAMPLE_IDS, RESULT_PATH, alignment,  # noqa: E402
                               load_humans, parse_response)
from paper_style import MODELS, MODEL_SHORT  # noqa: E402
import config as C  # noqa: E402

SETTINGS = ["order", "number", "prob"]


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["type"] != "Opinion-Dependent" or r["id"] in EXAMPLE_IDS:
            continue
        v = parse_response(r["response"])
        if v:
            out[r["id"]] = v
    return out


def country_scores(llm, humans):
    return {code: alignment(llm, dist)[0] for code, dist in humans.items()}


def main():
    humans, _ = load_humans()
    print(f"{'Models':10s} " + " ".join(f"{s:>16s}" for s in SETTINGS))
    for model in MODELS:
        base = f"{C.AAAI_RESULTS}/RQ1/{RESULT_PATH[model]}/results.csv"
        if not os.path.exists(base):
            print(f"{MODEL_SHORT[model]:10s} (no default run)")
            continue
        s0 = country_scores(load(base), humans)
        cells = []
        for setting in SETTINGS:
            path = f"{C.AAAI_RESULTS}/Discussion/sensitivity/{setting}/{RESULT_PATH[model]}/results.csv"
            if not os.path.exists(path):
                cells.append(f"{'-':>16s}")
                continue
            s1 = country_scores(load(path), humans)
            codes = sorted(set(s0) & set(s1))
            r, p = pearsonr([s0[c] for c in codes], [s1[c] for c in codes])
            cells.append(f"{r:8.4f} (p={p:.0e})")
        print(f"{MODEL_SHORT[model]:10s} " + " ".join(cells))


if __name__ == "__main__":
    main()
