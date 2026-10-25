"""How much does writing the questionnaire in a language move an LLM towards each of the
populations that answered in it?

\\S6.1 holds the country fixed and varies the questionnaire language. This asks the
opposite: the prompt is written in one language, carries no persona and no examples, and
is compared against every WV7 population that answered the survey in that language.

The quantity reported is a gain, not a level:

    G(lang, g) = A(prompt in lang, g) - A(prompt in English, g)

Both arms are country-blind, so the only thing that changes between them is the language
of the questionnaire. The raw level cannot be read across populations: some are simply
easier to align with than others, whatever the prompt, and that is a property of the
population rather than of the language. Subtracting the English arm removes it, exactly
as Eq. (5) of \\S6.1 does for the subgroups of one country.

The English arm is the `base` condition of jobs.py, which already covers every question
any of these languages asks, so nothing extra has to be run for it.

Only seven of the survey's 52 interview languages have more than one such population at
config.MIN_CELL_N respondents (config.SPEAKER_LANGS), and one of those, Arabic, has no
questionnaire we can prompt in. The rest have exactly one candidate, so the ranking is
empty for them.

The condition is `language` from jobs.py, which is country-blind: one run per language,
shared by every cell that speaks it. Spanish is not the language of any study cell, so
code/run_langonly.py produces it separately.

  python code/speaker_rank.py            -> results/speaker_rank.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from score import alignment_per_question, boot, load, pval

MODELS = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5",
          "aya-23-35b", "llama3-70b", "qwen2.5-72b"]


def speaker_groups(gt):
    """-> {prompt language: [country codes]} for the languages with more than one.

    The key is the language we write the prompt in, which is the interview-language code
    except for Chinese. WVS files all five Chinese-speaking populations under `zh`, but
    two of them were surveyed on a simplified questionnaire and three on a traditional
    one; putting the simplified prompt to the traditional-script populations would
    confound script with opinion. config.ZH_SCRIPT splits them, and each half is scored
    against the questionnaire it was actually surveyed in.
    """
    out = {}
    for key in gt:
        if "|" not in key:
            continue
        code, lang = key.split("|")
        code = int(code)
        if lang == "zh":
            lang = C.ZH_SCRIPT.get(code, "zh")
        out.setdefault(lang, []).append(code)
    return {l: sorted(set(v)) for l, v in out.items() if len(set(v)) > 1}


def gt_key(code, lang):
    """The ground-truth key for a population, whose language code is the survey's."""
    return f"{code}|{'zh' if lang == 'zt' else lang}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    gt = json.load(open(f"{C.DATA_DIR}/ground_truth.json"))
    groups = speaker_groups(gt)

    rows = []
    for model in ([args.model] if args.model else MODELS):
        if not os.path.exists(C.result_file(model)):
            print(f"  ! no raw.csv/parsed.csv for {model}, skipped")
            continue
        dists, _ = load(model)
        base_md = dists.get(("base", "-"))
        if not base_md:
            print(f"  ! {model}: no English `base` run, skipped")
            continue
        for lang, codes in sorted(groups.items()):
            if lang == "en":
                # the English arm of the `language` condition is the English baseline,
                # byte for byte, so every gain here is zero by construction rather than
                # by measurement. jobs.py deduplicates the two into one request.
                continue
            md = dists.get(("language", lang))
            if not md:
                print(f"  ! {model}: no `language` run for {lang}, skipped")
                continue
            per, base_per = {}, {}
            for code in codes:
                human = gt[gt_key(code, lang)]
                a, b = alignment_per_question(md, human), \
                    alignment_per_question(base_md, human)
                if a and b:
                    per[code], base_per[code] = a, b
            if len(per) < 2:
                continue
            qs = sorted(set.intersection(*[set(per[c]) & set(base_per[c]) for c in per]))
            # the gain from writing the questionnaire in this language, per population
            gain = {c: np.array([per[c][q] - base_per[c][q] for q in qs]) for c in per}
            means = {c: float(gain[c].mean()) for c in per}
            level = {c: float(np.mean([per[c][q] for q in qs])) for c in per}
            order = sorted(means, key=lambda c: -means[c])
            best = order[0]
            for rank, code in enumerate(order, 1):
                lo, hi = boot(gain[code])
                # also paired against the winner, so the ordering carries a p-value
                d = np.array([gain[best][q_i] - gain[code][q_i]
                              for q_i in range(len(qs))])
                rows.append(dict(model=model, lang=lang, rank=rank, code=code,
                                 country=C.COUNTRY_ALL.get(code, str(code)),
                                 gain=means[code], lo=lo, hi=hi,
                                 p=pval(gain[code]), n_q=len(qs),
                                 A_lang=level[code],
                                 A_en=level[code] - means[code],
                                 gap_to_best=float(d.mean()),
                                 p_gap=pval(d) if code != best else float("nan")))
        print(f"  {model}: done", flush=True)

    out = f"{C.RESULT_DIR}/speaker_rank.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        print(f"\n=== prompt language: {lang} "
              f"({len({r['country'] for r in sub})} populations) ===")
        wins = {}
        for m in MODELS:
            top = [r for r in sub if r["model"] == m and r["rank"] == 1]
            if top:
                wins[m] = top[0]["country"]
        for m, c in wins.items():
            print(f"  {m:15s} nearest: {c}")
        tally = {}
        for c in wins.values():
            tally[c] = tally.get(c, 0) + 1
        print("  tally: " + ", ".join(f"{c} {n}" for c, n in
                                      sorted(tally.items(), key=lambda t: -t[1])))
        agg, lvl, sig = {}, {}, {}
        for r in sub:
            agg.setdefault(r["country"], []).append(r["gain"])
            lvl.setdefault(r["country"], []).append(r["A_lang"])
            sig.setdefault(r["country"], []).append(r["lo"] > 0 or r["hi"] < 0)
        print(f"  {'population':22s} {'gain':>8s} {'>0':>5s} {'sig':>4s} "
              f"{'A(lang)':>8s}")
        for c, v in sorted(agg.items(), key=lambda t: -np.mean(t[1])):
            print(f"    {c:22s} {np.mean(v):+8.4f} "
                  f"{sum(x > 0 for x in v):>3d}/{len(v):<2d} {sum(sig[c]):>3d}  "
                  f"{np.mean(lvl[c]):8.4f}")


if __name__ == "__main__":
    main()
