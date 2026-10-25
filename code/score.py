"""Alignment scores, and the within-country difference-in-differences.

Metric is the AAAI one: A = mean_q [ 1 - WD(D_model(q), D_human(q)) / (|N|-1) ], the
Wasserstein distance taken over the ordinal option support 1..|N|.

Every model distribution is scored against every subgroup of the country it belongs to,
because the question is not "how well does steering X do" but "does steering aimed at
subgroup 1 land closer to subgroup 1 than to subgroup 2". For a country with languages
L1, L2 and a steering family F that is applied per language:

    DiD(F) = [A(F@L1, L1) - A(F@L1, L2)] - [A(F@L2, L1) - A(F@L2, L2)]

Both bracketed terms are within one steering condition, and both subgroups sit inside
one country, so "how alignable is this country" cancels out of every term -- which is
what the raw scores could never separate.

  python code/score.py --model qwen2.5-72b
"""
import argparse
import hashlib
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_1samp, wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from parse import match_llm_str, to_vector

N_BOOT = 2000

# steering families that exist in both an English and a target-language form; the DiD is
# defined on the target-language ones, where the questionnaire language is the treatment
FAMILIES = {
    # English prompt fed the subgroup's own answers. Intended as an upper bound, but it
    # is not one: five examples move the model about half as far as switching the
    # questionnaire language does (TV 0.065 vs 0.142), so treat it as a weak control.
    "few_shot_subgroup": "few_shot_country",
    "language": "base",
    "language_persona": "persona",
    "language_few_shot_country": "few_shot_country",
    "language_few_shot_subgroup": "few_shot_subgroup",
}


def alignment_per_question(model_dist, human_dist):
    out = {}
    for q, p1 in model_dist.items():
        p2 = human_dist.get(q)
        if p2 is None or len(p1) != len(p2):
            continue
        n = len(p1)
        support = np.arange(1, n + 1)
        out[q] = 1 - wasserstein_distance(support, support, p1, p2) / (n - 1)
    return out


def pval(values):
    """Two-sided one-sample t-test against zero, over questions.

    The paper reports significance with a t-test, so every paired difference we add
    here uses the same convention rather than reading it off the bootstrap interval.
    """
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.any(a != a[0]):
        return float("nan")
    return float(ttest_1samp(a, 0.0).pvalue)


def boot(values, n_boot=N_BOOT):
    """Item-level bootstrap CI; greedy decoding leaves questions as the only noise.

    The RNG is seeded from the values themselves rather than drawn from one shared
    stream, so a CI depends only on the numbers it summarises. With a shared stream,
    adding one country to the scoring loop shifts every CI computed after it by ~1e-3 of
    Monte-Carlo noise, which is enough to flip a borderline significance verdict in a
    table that has nothing to do with the country that was added."""
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    seed = int.from_bytes(hashlib.blake2b(a.tobytes(), digest_size=8).digest(), "big")
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def load(model):
    """-> {(cond, unit): {qid: vector}}, re-parsing raw text so a parser fix reaches old runs."""
    dists, dropped = {}, 0
    with open(C.result_file(model), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r["parsed"] or match_llm_str(r.get("raw", ""))
            v = to_vector(p, json.loads(r["choice_keys"])) if p else None
            if v is None:
                dropped += 1
                continue
            dists.setdefault((r["cond"], r["unit"]), {})[r["qid"]] = v
    return dists, dropped


def unit_for(cond, cell, lang):
    """Which `unit` holds the distribution for this condition, country and language."""
    code = cell["code"]
    return {
        "base": "-",
        "persona": str(code),
        "few_shot_country": str(code),
        "few_shot_subgroup": f"{code}|{lang}",
        "language": lang,
        "language_persona": f"{lang}|{code}",
        "language_few_shot_country": f"{lang}|{code}",
        "language_few_shot_subgroup": f"{lang}|{code}|{lang}",
    }[cond]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    gt = json.load(open(f"{C.DATA_DIR}/ground_truth.json"))
    dists, dropped = load(args.model)
    print(f"unusable responses dropped: {dropped}")

    rows, per_question = [], {}
    for cell in C.DID_CELLS:
        code, langs = cell["code"], cell["langs"]
        for cond in C.CONDITIONS_FULL:
            # the language the steering is aimed at ('-' where the condition has none)
            aims = langs if cond.startswith("language") or "subgroup" in cond else ["-"]
            for aim in aims:
                md = dists.get((cond, unit_for(cond, cell, aim if aim != "-" else langs[0])))
                if not md:
                    continue
                for target in langs:                      # score against BOTH subgroups
                    human = gt.get(f"{code}|{target}")
                    if human is None:
                        continue
                    per_q = alignment_per_question(md, human)
                    if not per_q:
                        continue
                    vals = list(per_q.values())
                    lo, hi = boot(vals)
                    per_question[(cell["country"], cond, aim, target)] = per_q
                    rows.append(dict(model=args.model, country=cell["country"],
                                     country_code=code, cond=cond, aim=aim,
                                     target=target, n_q=len(vals),
                                     A=float(np.mean(vals)), lo=lo, hi=hi))

    out = f"{C.RESULT_DIR}/{args.model}/scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per-question scores, kept so the DiD gets a paired bootstrap rather than a point
    # --- how much does each steering language lean towards its language's homeland? ---
    # Cyprus: Greek-steered vs Turkish-steered, scored against Greece and Turkey.
    # Countries with only one reference (Russia) get a plain difference instead.
    # --- switching the questionnaire language while the content stays fixed ---------
    # persona and few-shot both describe the same country in every condition; the only
    # thing that changes is what language the questionnaire is written in. Paired per
    # question against the homeland of that language, so the comparison is
    #   A(steered in Greek -> Greece) - A(steered in English -> Greece)
    SWITCH = {
        "persona": ("persona", "language_persona"),
        "few_shot_country": ("few_shot_country", "language_few_shot_country"),
        "few_shot_subgroup": ("few_shot_subgroup", "language_few_shot_subgroup"),
        "no_country_info": ("base", "language"),
    }
    switch_rows = []
    for cell in C.CELLS:
        code = cell["code"]
        for lang in cell["langs"]:
            # two targets for the same pair of model outputs:
            #   subgroup -- the people of this country who answered in this language
            #   homeland -- the country where this language is the national language,
            #               where WV7 surveyed one
            targets = [("subgroup", f"{code}|{lang}", C.COUNTRY_DISPLAY[code])]
            fcode = cell.get("foreign", {}).get(lang)
            if fcode:
                targets.append(("homeland", str(fcode), C.COUNTRY_DISPLAY[fcode]))
            for kind, gkey, refname in targets:
                human = gt.get(gkey)
                if human is None:
                    continue
                for fam, (en_cond, tgt_cond) in SWITCH.items():
                    a = dists.get((en_cond, unit_for(en_cond, cell, lang)))
                    b = dists.get((tgt_cond, unit_for(tgt_cond, cell, lang)))
                    if not a or not b:
                        continue
                    pa = alignment_per_question(a, human)
                    pb = alignment_per_question(b, human)
                    qs = sorted(set(pa) & set(pb))
                    if not qs:
                        continue
                    v = np.array([pb[q] - pa[q] for q in qs])
                    lo_, hi_ = boot(v)
                    switch_rows.append(dict(
                        model=args.model, country=cell["country"], country_code=code,
                        lang=lang, family=fam, target=kind, ref=refname, n_q=len(qs),
                        A_en=float(np.mean([pa[q] for q in qs])),
                        A_lang=float(np.mean([pb[q] for q in qs])),
                        gain=float(v.mean()), lo=lo_, hi=hi_, p=pval(v)))
    if switch_rows:
        with open(f"{C.RESULT_DIR}/{args.model}/switch.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(switch_rows[0]))
            w.writeheader()
            w.writerows(switch_rows)

    # --- country-wide examples vs the subgroup's own examples -----------------------
    # Identical prompts except for whose five answer distributions the few-shot block
    # shows. Scored against the subgroup the subgroup-examples were drawn from, so the
    # question is simply whether the finer-grained examples find their own speakers.
    FEWSHOT_PAIRS = {
        "English questionnaire": ("few_shot_country", "few_shot_subgroup"),
        "target-language questionnaire": ("language_few_shot_country",
                                          "language_few_shot_subgroup"),
    }
    fewshot_rows = []
    for cell in C.CELLS:
        code = cell["code"]
        for lang in cell["langs"]:
            human = gt.get(f"{code}|{lang}")
            if human is None:
                continue
            for label, (c_cond, s_cond) in FEWSHOT_PAIRS.items():
                a = dists.get((c_cond, unit_for(c_cond, cell, lang)))
                b = dists.get((s_cond, unit_for(s_cond, cell, lang)))
                if not a or not b:
                    continue
                pa, pb = alignment_per_question(a, human), alignment_per_question(b, human)
                qs = sorted(set(pa) & set(pb))
                if not qs:
                    continue
                v = np.array([pb[q] - pa[q] for q in qs])
                lo_, hi_ = boot(v)
                fewshot_rows.append(dict(
                    model=args.model, country=cell["country"], country_code=code,
                    lang=lang, questionnaire=label, n_q=len(qs),
                    A_country_examples=float(np.mean([pa[q] for q in qs])),
                    A_subgroup_examples=float(np.mean([pb[q] for q in qs])),
                    gain=float(v.mean()), lo=lo_, hi=hi_, p=pval(v)))
    if fewshot_rows:
        with open(f"{C.RESULT_DIR}/{args.model}/fewshot.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fewshot_rows[0]))
            w.writeheader()
            w.writerows(fewshot_rows)

    lean_rows, foreign_rows = [], []
    for cell in C.CELLS:
        foreign = cell.get("foreign", {})
        if not foreign:
            continue
        for cond in ("language", "language_persona",
                     "language_few_shot_country", "language_few_shot_subgroup"):
            per = {}
            for lang in cell["langs"]:
                md = dists.get((cond, unit_for(cond, cell, lang)))
                if not md:
                    continue
                for flang, fcode in foreign.items():
                    human = gt.get(str(fcode))
                    if human:
                        per[(lang, fcode)] = alignment_per_question(md, human)
                        vals = list(per[(lang, fcode)].values())
                        lo_, hi_ = boot(vals)
                        foreign_rows.append(dict(
                            model=args.model, country=cell["country"], cond=cond,
                            steer_lang=lang, ref_code=fcode,
                            ref=C.COUNTRY_DISPLAY[fcode], n_q=len(vals),
                            A=float(np.mean(vals)), lo=lo_, hi=hi_))
            if len(foreign) == 2:
                (fa, ca), (fb, cb) = list(foreign.items())
                need = [(fa, ca), (fa, cb), (fb, ca), (fb, cb)]
                if any(k not in per for k in need):
                    continue
                qs = sorted(set.intersection(*[set(per[k]) for k in need]))
                v = np.array([(per[need[0]][q] - per[need[1]][q])
                              - (per[need[2]][q] - per[need[3]][q]) for q in qs])
                lo, hi = boot(v)
                lean_rows.append(dict(country=cell["country"], cond=cond, kind="did",
                                      lang_a=fa, ref_a=C.COUNTRY_DISPLAY[ca],
                                      lang_b=fb, ref_b=C.COUNTRY_DISPLAY[cb],
                                      value=float(v.mean()), lo=lo, hi=hi, n_q=len(qs)))
            else:
                (fl, fc), = foreign.items()
                other = [l for l in cell["langs"] if l != fl][0]
                if (fl, fc) not in per or (other, fc) not in per:
                    continue
                qs = sorted(set(per[(fl, fc)]) & set(per[(other, fc)]))
                v = np.array([per[(fl, fc)][q] - per[(other, fc)][q] for q in qs])
                lo, hi = boot(v)
                lean_rows.append(dict(country=cell["country"], cond=cond, kind="diff",
                                      lang_a=fl, ref_a=C.COUNTRY_DISPLAY[fc],
                                      lang_b=other, ref_b=C.COUNTRY_DISPLAY[fc],
                                      value=float(v.mean()), lo=lo, hi=hi, n_q=len(qs)))
    if foreign_rows:
        with open(f"{C.RESULT_DIR}/{args.model}/foreign_scores.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(foreign_rows[0]))
            w.writeheader()
            w.writerows(foreign_rows)

    if lean_rows:
        with open(f"{C.RESULT_DIR}/{args.model}/lean.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lean_rows[0]))
            w.writeheader()
            w.writerows(lean_rows)

    idx = {(r["country"], r["cond"], r["aim"], r["target"]): r["A"] for r in rows}
    print(f"\n{'country':12s} {'family':28s} {'DiD':>8s} {'95% CI':>18s} {'sig':>4s}  "
          f"{'A(L1,L1)':>9s}{'A(L1,L2)':>9s}{'A(L2,L1)':>9s}{'A(L2,L2)':>9s}")
    did_rows = []
    for cell in C.DID_CELLS:
        l1, l2 = cell["langs"]
        for fam in FAMILIES:
            key = lambda a, t: (cell["country"], fam, a, t)
            pq = [per_question.get(key(l1, l1)), per_question.get(key(l1, l2)),
                  per_question.get(key(l2, l1)), per_question.get(key(l2, l2))]
            if any(v is None for v in pq):
                continue
            # the DiD is a per-question quantity; bootstrap it paired over questions so
            # the shared "how alignable is this country" component cancels inside every
            # resample instead of being averaged away first
            qs = sorted(set.intersection(*[set(d) for d in pq]))
            per_q_did = np.array([(pq[0][q] - pq[1][q]) - (pq[2][q] - pq[3][q]) for q in qs])
            did = float(per_q_did.mean())
            lo, hi = boot(per_q_did)
            sig = "*" if lo > 0 or hi < 0 else ""
            q = [float(np.mean([d[x] for x in qs])) for d in pq]
            did_rows.append(dict(country=cell["country"], family=fam, did=did,
                                 lo=lo, hi=hi, p=pval(per_q_did),
                                 n_q=len(qs), l1=l1, l2=l2))
            print(f"{cell['country']:12s} {fam:28s} {did:+8.4f} "
                  f"[{lo:+7.4f},{hi:+7.4f}] {sig:>4s}  "
                  + "".join(f"{v:9.4f}" for v in q))

    with open(f"{C.RESULT_DIR}/{args.model}/did.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(did_rows[0]))
        w.writeheader()
        w.writerows(did_rows)
    print(f"\n{'country':12s} {'lang':5s} {'target':9s} {'family':18s} "
          f"{'gain':>8s} {'95% CI':>18s} {'sig':>4s}")
    for r in switch_rows:
        if r["target"] != "subgroup":
            continue
        sig = "*" if r["lo"] > 0 or r["hi"] < 0 else ""
        print(f"{r['country']:12s} {r['lang']:5s} {r['target']:9s} {r['family']:18s} "
              f"{r['gain']:+8.4f} [{r['lo']:+7.4f},{r['hi']:+7.4f}] {sig:>4s}")

    print(f"\n{'country':12s} {'cond':28s} {'lean':>8s} {'95% CI':>18s} {'sig':>4s}  towards")
    for r in lean_rows:
        sig = "*" if r["lo"] > 0 or r["hi"] < 0 else ""
        towards = (f"{r['lang_a']}->{r['ref_a']} vs {r['lang_b']}->{r['ref_b']}"
                   if r["kind"] == "did" else
                   f"{r['lang_a']} vs {r['lang_b']}, both -> {r['ref_a']}")
        print(f"{r['country']:12s} {r['cond']:28s} {r['value']:+8.4f} "
              f"[{r['lo']:+7.4f},{r['hi']:+7.4f}] {sig:>4s}  {towards}")

    print(f"\nDiD > 0 means steering in a subgroup's own language lands closer to that "
          f"subgroup.\nwrote {out}, did.csv, lean.csv, switch.csv, fewshot.csv")


if __name__ == "__main__":
    main()
