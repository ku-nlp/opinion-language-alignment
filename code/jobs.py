"""The full list of prompts to run, shared by the API and the open-weight runners.

Eight conditions. Each row of the design answers a different question about where the
group information comes from:

                        English questionnaire        target-language questionnaire
  nothing               base                         language
  country named         persona                      language_persona
  country's answers     few_shot_country             language_few_shot_country
  subgroup's answers    few_shot_subgroup            language_few_shot_subgroup

`base` and `language` carry no country information at all, so one run of each serves
every group; the rest are per group. The scoring unit is a (country, interview-language)
subgroup, and the test is a difference-in-differences within one country, so the two
language columns must exist for both of a country's languages.

Few-shot examples are excluded from the evaluation set (config.FEWSHOT_IDS), so nothing
shown inside a prompt is ever scored.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import prompts as P

_PERSONA = json.load(open(f"{C.ROOT}/code/persona_slots.json", encoding="utf-8"))
_SPEC = json.load(open(C.LANG_INSTR, encoding="utf-8"))
_SPEC["en"] = dict(_SPEC["en_baseline"])


def persona_phrase(lang, country_code):
    """The target-language noun phrase naming the group, e.g. 'ομάδα με ...'."""
    p = _PERSONA[lang]
    name = p["countries"].get(C.COUNTRY_DISPLAY[country_code], C.COUNTRY_DISPLAY[country_code])
    return p["group"].format(name)


def language_persona_prompt(lang, country_code, item):
    """The `language` prompt with exactly one phrase inserted, nothing else changed."""
    if lang == "en":
        # English persona and English "language persona" are the same prompt
        return P.persona_prompt(f"people from {C.COUNTRY_DISPLAY[country_code]}", item)
    spec = _SPEC[lang]
    p = _PERSONA[lang]
    who = persona_phrase(lang, country_code)
    prefix = p["instr0"].format(who) + p["instr1"].format(who)
    return P._assemble(spec, prefix, item["question"], item["answer"])


def few_shot_prompt(lang, group, item, persona=False, country_code=None):
    """A prompt whose examples carry a real distribution instead of random numbers."""
    spec = _SPEC[lang]
    text = open(f"{C.FEWSHOT_DIR}/lang-{lang}_dist-{group}.txt", encoding="utf-8").read()
    if persona and lang == "en":
        # persona_slots.json holds the target-language wordings only; English has its own
        # spec block, the same one persona_prompt uses, so the two English persona
        # conditions differ in their examples and in nothing else
        e = P._SPEC["en_persona"]
        who = f"people from {C.COUNTRY_DISPLAY[country_code]}"
        prefix = e["instr"][0].format(who) + e["instr"][1].format(who)
        spec = e
    elif persona:
        p = _PERSONA[lang]
        who = persona_phrase(lang, country_code)
        prefix = p["instr0"].format(who) + p["instr1"].format(who)
    else:
        prefix = "".join(spec["instr"])
    return P._assemble(spec, prefix, item["question"], item["answer"], fewshot_text=text)


def eval_ids(cell):
    """Questions scorable for a country: present in both its languages and in English."""
    have = [{d["id"] for d in P.load_items(l)} for l in cell["langs"] + ["en"]]
    return [d["id"] for d in P.load_items("en")
            if all(d["id"] in h for h in have) and d["id"] not in C.FEWSHOT_IDS]


def build_jobs():
    """-> list of dicts: cond, unit, qid, choice_keys, prompt.

    `unit` identifies what the prompt is conditioned on, so the runner can resume and the
    scorer can find it again: '-' for the country-blind conditions, else the group key
    ('196', '196|el') optionally prefixed by the questionnaire language.
    """
    items = {l: {d["id"]: d for d in P.load_items(l)}
             for l in {l for c in C.DID_CELLS for l in c["langs"]} | {"en"}}
    jobs, seen = [], set()

    def add(cond, unit, lang, qid, prompt):
        if (cond, unit, qid) in seen:
            return
        seen.add((cond, unit, qid))
        jobs.append(dict(cond=cond, unit=unit, qid=qid,
                         choice_keys=json.dumps(items[lang][qid]["choice_keys"]),
                         prompt=prompt))

    for cell in C.DID_CELLS:
        code = cell["code"]
        for qid in eval_ids(cell):
            it_en = items["en"][qid]

            # --- English questionnaire ------------------------------------
            add("base", "-", "en", qid, P.base_prompt(it_en))
            add("persona", str(code), "en", qid,
                P.persona_prompt(f"people from {C.COUNTRY_DISPLAY[code]}", it_en))
            add("few_shot_country", str(code), "en", qid,
                few_shot_prompt("en", str(code), it_en))
            for lang in cell["langs"]:
                add("few_shot_subgroup", f"{code}|{lang}", "en", qid,
                    few_shot_prompt("en", f"{code}|{lang}", it_en))

            # --- target-language questionnaire ----------------------------
            for lang in cell["langs"]:
                it = items[lang][qid]
                add("language", lang, lang, qid, P.language_prompt(lang, it))
                add("language_persona", f"{lang}|{code}", lang, qid,
                    language_persona_prompt(lang, code, it))
                add("language_few_shot_country", f"{lang}|{code}", lang, qid,
                    few_shot_prompt(lang, str(code), it))
                add("language_few_shot_subgroup", f"{lang}|{code}|{lang}", lang, qid,
                    few_shot_prompt(lang, f"{code}|{lang}", it))
    return jobs


def summarise(jobs):
    import collections
    by = collections.Counter(j["cond"] for j in jobs)
    print(f"{len(jobs)} prompts")
    for cond in C.CONDITIONS_FULL:
        units = len({j["unit"] for j in jobs if j["cond"] == cond})
        print(f"  {cond:28s} {by[cond]:5d} prompts over {units:2d} unit(s)")


if __name__ == "__main__":
    js = build_jobs()
    summarise(js)
    print("\nsample prompt tails:")
    for cond in C.CONDITIONS_FULL:
        j = next(x for x in js if x["cond"] == cond)
        print(f"\n--- {cond} / unit={j['unit']} / {j['qid']} ---")
        print(j["prompt"][:150].replace("\n", "\\n"))
        print("   ...")
        print(j["prompt"][-160:].replace("\n", "\\n"))
