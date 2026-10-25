"""Build the five-example blocks that every prompt carries.

Format follows the AAAI repo's inputs/few_shot/*.txt byte for byte in structure: a
header naming the two JSON fields, a format template, then five worked examples of
"question with numbered options -> dictionary of percentages".

Two things differ from the original, both to make groups comparable:

  * one fixed set of example questions for every group and every language (the original
    varies it per country, which changes what leaks and what is asked)
  * those five questions are excluded from scoring, so no answer is ever both shown in a
    prompt and graded -- see config.FEWSHOT_IDS

Distribution variants:
  random     fixed pseudo-random percentages, identical across languages. Used by the
             conditions that must not leak anything; only demonstrates the format.
  <code>     the whole country's real distribution
  <code>|<l> the real distribution of that country's <l>-language respondents

  python code/fewshot.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT_DIR = f"{C.ROOT}/few_shot"
SEED = 0
# a second, independent draw of the same shape. It is the control for the few-shot
# channel: `random2` differs from `random` only in the numbers, not in whose answers
# they are, so it measures how far the residual stream moves when the example
# percentages change for no reason at all.
SEED2 = 1


def pct(vec):
    """-> "{'1': '31.01%', ...}" with keys 1..N, as in the original files."""
    return "{" + ", ".join(f"'{i}': '{100 * p:.2f}%'" for i, p in enumerate(vec, 1)) + "}"


def random_dists(items, seed=SEED):
    """One fixed pseudo-random distribution per question, shared by every language."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in items:
        v = rng.dirichlet(np.ones(len(d["choice_keys"])))
        out[d["id"]] = v / v.sum()
    return out


def block(qlabel, alabel, item, dist):
    """One worked example: question, numbered options, answer dictionary."""
    opts = "\n".join(f"'{i}'. {c}" for i, c in enumerate(item["choices"], 1))
    return f"{qlabel}: {item['question']}\n{opts}\n{alabel}: {pct(dist)}"


def render(assets, items, dists):
    a = assets
    head = (f"{a['header1']}\n---\n\n{a['header2']}\n\n"
            f"{a['qlabel']}: ${{question}}\n{a['alabel']}: ${{answer}}\n")
    body = ""
    for i, it in enumerate(items):
        # the original separates the first example with one newline and the rest with two
        body += "\n------ \n" + ("" if i == 0 else "\n")
        body += block(a["qlabel"], a["alabel"], it, dists[it["id"]]) + "\n"
    return head + body + "\n----\n\n"


def load_items(lang):
    path = f"{C.QUESTION_DIR}/WV7_{C.LANG_FILE[lang]}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}


def main():
    assets = json.load(open(f"{C.ROOT}/code/lang_assets.json", encoding="utf-8"))
    gt = json.load(open(f"{C.DATA_DIR}/ground_truth.json"))
    ex_ids = sorted(C.FEWSHOT_IDS)
    # every language that has a questionnaire and a homeland in WV7, not just the ones
    # the behavioural analysis steers with: the mechanism experiment needs a `random`
    # block for each of them too
    langs = sorted(set(C.STEERING_LANGS) | set(C.LANG_HOME) | {"en"}
                   | {l for c in C.DID_CELLS for l in c["langs"]}
                   # Spanish reaches this list only here: it is nobody's LANG_HOME,
                   # because Spain is not in WV7, but thirteen WV7 countries answered
                   # in it, so the country-blind `language` prompt needs its block
                   | set(C.SPEAKER_PROMPT_LANGS))

    items = {l: load_items(l) for l in langs}
    missing = {l: [q for q in ex_ids if q not in items[l]] for l in langs}
    for l, m in missing.items():
        if m:
            sys.exit(f"{l} lacks example questions {m}; pick a different FEWSHOT_IDS")

    ex_items = [items["en"][q] for q in ex_ids]
    variants = {"random": random_dists(ex_items, SEED),
                "random2": random_dists(ex_items, SEED2)}
    os.makedirs(OUT_DIR, exist_ok=True)

    # which (language, group) pairs any condition can ask for
    want = [(l, v) for l in langs for v in variants]
    # the mechanism experiment shows every WV7 country's real distribution in English
    want += [("en", str(code)) for code in C.COUNTRY_ALL]
    for cell in C.DID_CELLS:
        code = cell["code"]
        for l in cell["langs"] + ["en"]:
            want.append((l, str(code)))                    # few_shot_country
            for sub in cell["langs"]:
                if l in ("en", sub):
                    want.append((l, f"{code}|{sub}"))       # few_shot_subgroup

    written = 0
    for lang, group in sorted(set(want)):
        if group in variants:
            dists = variants[group]
        else:
            g = gt.get(group)
            if g is None:
                print(f"  ! no ground truth for {group}, skipped")
                continue
            if any(q not in g for q in ex_ids):
                print(f"  ! {group} lacks {[q for q in ex_ids if q not in g]}, skipped")
                continue
            dists = {q: g[q] for q in ex_ids}
        text = render(assets[lang], [items[lang][q] for q in ex_ids], dists)
        open(f"{OUT_DIR}/lang-{lang}_dist-{group}.txt", "w", encoding="utf-8").write(text)
        written += 1

    print(f"wrote {written} files to {OUT_DIR}")
    print(f"examples: {ex_ids}  (excluded from scoring)")
    for f in sorted(os.listdir(OUT_DIR))[:6]:
        print(f"  {f}")
    print("  ...")


if __name__ == "__main__":
    main()
