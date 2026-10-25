"""Human answer distributions, per country and per interview-language subgroup.

Two levels, because the whole design turns on the difference between them:

  country   every respondent of the country, whatever questionnaire they answered
  subgroup  respondents of that country who answered in one particular language

Counting follows the AAAI convention exactly: for each question, count how many
respondents picked each option listed in the ENGLISH questionnaire's `choice_keys`, then
normalise. WVS releases WV7 harmonised to the master codebook, so the English codes are
the ones the data speaks, regardless of what a national questionnaire printed.
Out-of-range responses (the negative don't-know / refused codes) drop out.

Also emitted are the foreign reference countries a subgroup might lean towards, so the
"which country does this subgroup resemble" comparison needs no second pass over the CSV.

  python code/ground_truth.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def english_questions():
    path = f"{C.QUESTION_DIR}/WV7_English.jsonl"
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [d for d in items if d["type"] == "Opinion-Dependent"]


def distribution(sub, qids, choices):
    out = {}
    for q in qids:
        counts = [int((sub[q] == opt).sum()) for opt in choices[q]]
        total = sum(counts)
        if total:
            out[q] = [c / total for c in counts]
    return out


def main():
    items = english_questions()
    qids = [d["id"] for d in items]
    choices = {d["id"]: [int(k) for k in d["choice_keys"]] for d in items}
    print(f"opinion questions: {len(qids)}")

    df = pd.read_csv(C.WVS_CSV, usecols=["B_COUNTRY", "LNGE_ISO"] + qids, low_memory=False)
    print(f"respondents: {len(df)}")

    groups, gt = [], {}

    def add(key, sub, **meta):
        if key in gt:
            return                       # a study country is also one of the 66
        gt[key] = distribution(sub, qids, choices)
        groups.append(dict(key=key, n=int(len(sub)), n_questions=len(gt[key]), **meta))

    # every WV7 country at country level: the mechanism experiment shows the model each
    # country's real answer distribution as few-shot examples, so all 66 are needed
    for code in sorted(C.COUNTRY_ALL):
        add(str(code), df[df.B_COUNTRY == code], level="country", country_code=code,
            country=C.COUNTRY_ALL[code], lang=None)

    seen_foreign = set()
    for cell in C.CELLS:
        code = cell["code"]
        country = df[df.B_COUNTRY == code]
        add(str(code), country, level="country", country_code=code,
            country=cell["country"], lang=None)   # already added above; kept for clarity
        for lang in cell["langs"]:
            sub = country[country.LNGE_ISO == lang]
            if len(sub) < C.MIN_CELL_N:
                print(f"  ! {cell['country']}/{lang}: only {len(sub)} respondents, skipped")
                continue
            add(f"{code}|{lang}", sub, level="subgroup", country_code=code,
                country=cell["country"], lang=lang)
        # the country each subgroup might resemble, scored later against the subgroups
        for lang, fcode in cell.get("foreign", {}).items():
            if fcode in seen_foreign:
                continue
            seen_foreign.add(fcode)
            add(str(fcode), df[df.B_COUNTRY == fcode], level="foreign", country_code=fcode,
                country=C.COUNTRY_DISPLAY.get(fcode, str(fcode)), lang=None)

    # Every (country, interview language) cell of the languages that more than one
    # country answered in. The study cells above are a handful of countries chosen for
    # the within-country contrast; these are the other side of the same field, needed to
    # ask which of the populations that answered in a language an unsteered prompt in
    # that language lands nearest. Spanish alone contributes thirteen countries.
    big = (df.groupby(["LNGE_ISO", "B_COUNTRY"]).size().reset_index(name="n")
             .query(f"n >= {C.MIN_CELL_N}"))
    multi = [l for l, g in big.groupby("LNGE_ISO") if len(g) > 1]
    for lang in sorted(multi):
        for code in sorted(big[big.LNGE_ISO == lang].B_COUNTRY):
            sub = df[(df.B_COUNTRY == code) & (df.LNGE_ISO == lang)]
            add(f"{code}|{lang}", sub, level="speaker_group", country_code=int(code),
                country=C.COUNTRY_ALL.get(int(code), str(code)), lang=lang)

    # the two-language countries that the selection rule removes: nothing is scored
    # against them, they only fill the lower block of the paper's first table
    for cell in C.DISPLAY_ONLY_CELLS:
        code = cell["code"]
        country = df[df.B_COUNTRY == code]
        for lang in cell["langs"]:
            sub = country[country.LNGE_ISO == lang]
            add(f"{code}|{lang}", sub, level="subgroup_excluded", country_code=code,
                country=cell["country"], lang=lang)

    os.makedirs(C.DATA_DIR, exist_ok=True)
    json.dump(groups, open(f"{C.DATA_DIR}/groups.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(gt, open(f"{C.DATA_DIR}/ground_truth.json", "w"))

    print(f"\n{'key':12s} {'level':10s} {'country':14s} {'lang':5s} {'n':>6s} {'questions':>10s}")
    for g in [g for g in groups if g["level"] != "country" or g["country_code"] in
              {c["code"] for c in C.CELLS} |
              {f for c in C.CELLS for f in c.get("foreign", {}).values()}]:
        print(f"{g['key']:12s} {g['level']:10s} {g['country']:14s} "
              f"{(g['lang'] or '-'):5s} {g['n']:6d} {g['n_questions']:10d}")
    print(f"\nwrote {C.DATA_DIR}/groups.json and ground_truth.json")


if __name__ == "__main__":
    main()
