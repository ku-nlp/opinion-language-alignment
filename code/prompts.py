"""Prompt construction, ported from the AAAI repo's utils/common.py.

The original hard-codes one branch per language inside get_prompt_of_language(); the
same text lives here in code/lang_instructions.json, keyed by ISO code, so a new
language is a data change rather than a code change. Every builder here is verified
byte-identical to the original for the nine languages the original supports --
see selftest() at the bottom.

Conditions, following the paper's naming:

  base            English questionnaire, no country information          (RQ1)
  persona         English questionnaire + "a group of {demonym}"
  few_shot        English questionnaire + that group's real distributions
  language        target-language questionnaire, no country information
  language_persona    target language + persona
  language_few_shot   target language + that group's real distributions

`few_shot` differs from the original in two ways, both deliberate:
  * one fixed set of example questions for every group, so groups stay comparable
    (the original uses a different set per country, 3-5 of which are also scored)
  * the examples are dropped from the evaluation set, so no answer is ever both
    shown in the prompt and scored
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

_SPEC = json.load(open(C.LANG_INSTR, encoding="utf-8"))
# For an English-speaking subgroup the "language" condition IS the English no-steering
# prompt; alias it so callers can treat en like any other language. A copy, not the same
# object, so selftest's swap does not hit it twice.
_SPEC["en"] = dict(_SPEC["en_baseline"])
_FEWSHOT_CACHE = {}


def load_items(lang):
    """Opinion-Dependent questions of one questionnaire, in questionnaire order."""
    path = f"{C.QUESTION_DIR}/WV7_{C.LANG_FILE[lang]}.jsonl"
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [d for d in items if d["type"] == "Opinion-Dependent"]


def eval_items(lang):
    """Questions actually scored: everything except the shared few-shot examples."""
    return [d for d in load_items(lang) if d["id"] not in C.FEWSHOT_IDS]


def _fewshot(name):
    if name not in _FEWSHOT_CACHE:
        _FEWSHOT_CACHE[name] = open(f"{C.FEWSHOT_DIR}/{name}", encoding="utf-8").read()
    return _FEWSHOT_CACHE[name]


def _assemble(spec, prefix, question, answer, fewshot_text=None):
    """prefix + examples + turn marker + question + options + answer marker.

    tail is [turn..., question_label, separator, answer_label]; the original writes
    those as four separate `prompt +=` calls, and splitting from the right keeps
    languages whose turn marker is more than one string working.
    """
    turn, (qlab, sep, alab) = spec["tail"][:-3], spec["tail"][-3:]
    body = fewshot_text if fewshot_text is not None else _fewshot(spec["fewshot"])
    return prefix + body + "".join(turn) + qlab + question + sep + answer + alab


def base_prompt(item_en):
    s = _SPEC["en_baseline"]
    return _assemble(s, "".join(s["instr"]), item_en["question"], item_en["answer"])


def persona_prompt(group, item_en):
    """`group` is the noun phrase inserted after "a group of", e.g. "people from Cyprus"."""
    s = _SPEC["en_persona"]
    prefix = s["instr"][0].format(group) + s["instr"][1].format(group)
    return _assemble(s, prefix, item_en["question"], item_en["answer"])


def language_prompt(lang, item):
    s = _SPEC[lang]
    return _assemble(s, "".join(s["instr"]), item["question"], item["answer"])


def selftest():
    """Rebuild the original's prompts from its own code and compare byte for byte."""
    sys.path.insert(0, C.AAAI_REPO)
    cwd = os.getcwd()
    os.chdir(C.AAAI_REPO)
    # compare against the original using ITS few-shot blocks; ours hold different
    # example questions on purpose, so only the instruction assembly is under test
    orig_dir, C.FEWSHOT_DIR = C.FEWSHOT_DIR, f"{C.AAAI_REPO}/inputs/few_shot"
    for v in _SPEC.values():
        v["fewshot"], v["fewshot_aaai"] = v["fewshot_aaai"], v["fewshot"]
    _FEWSHOT_CACHE.clear()
    try:
        from utils.common import (get_prompt_of_language, get_prompt_of_rq1,
                                  get_prompt_of_rq2, country_human_dict)
        import argparse
        ok = fail = 0
        pairs = [("China", "zh"), ("Japan", "ja"), ("Germany", "de"), ("Korea", "ko"),
                 ("Russia", "ru"), ("Vietnam", "vi"), ("Brazil", "pt"),
                 ("Argentina", "es")]
        for country, lang in pairs:
            for it in load_items(lang)[:4]:
                a, b = get_prompt_of_language(country, it), language_prompt(lang, it)
                ok, fail = (ok + 1, fail) if a == b else (ok, fail + 1)
                if a != b:
                    print(f"  MISMATCH language/{lang} {it['id']}")
        for it in load_items("en")[:4]:
            a, b = get_prompt_of_rq1(it), base_prompt(it)
            ok, fail = (ok + 1, fail) if a == b else (ok, fail + 1)
            if a != b:
                print(f"  MISMATCH base {it['id']}")
            ns = argparse.Namespace(steering_method="persona", country="China")
            a = get_prompt_of_rq2(ns, it)
            b = persona_prompt(country_human_dict["China"], it)
            ok, fail = (ok + 1, fail) if a == b else (ok, fail + 1)
            if a != b:
                print(f"  MISMATCH persona {it['id']}")
        print(f"selftest vs the AAAI implementation: {ok} identical, {fail} different")
    finally:
        os.chdir(cwd)
        C.FEWSHOT_DIR = orig_dir
        for v in _SPEC.values():
            v["fewshot"], v["fewshot_aaai"] = v["fewshot_aaai"], v["fewshot"]
        _FEWSHOT_CACHE.clear()


if __name__ == "__main__":
    selftest()
