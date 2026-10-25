"""The two anonymous few-shot conditions of the steering table.

The conference few-shot prompt names the country four times ("a group of Chinese", "eine
Gruppe von Deutschen"), so it carries two cues at once: the country's answer distributions
*and* its name. \\S5.2 measures a few-shot channel that carries the distributions only,
which is a different prompt from the one \\S5.1 reports. These two conditions close that
gap by running, in the setting of \\S5.1, a few-shot prompt with the country name removed
and nothing else changed:

    few-shot-anon            English questionnaire, examples of country c, c unnamed
    language_few-shot-anon   c's questionnaire,     examples of country c, c unnamed

The prompts are produced by taking the conference prompt and substituting the demonym, so
what is left is the conference prompt minus the country name -- the difference between the
two rows of the table is one noun phrase, not a rewrite. The example blocks themselves
(inputs/few_shot/lang-*_dist-*.txt) already carry distributions only and are untouched.

Every substitution below is asserted to fire the expected number of times, so a change in
the conference wording is a crash rather than a silently unsteered prompt.

  python code/aaai_anon.py            # print every prompt head, and the diff against the
                                      # named version, for inspection
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                                                   # noqa: E402

sys.path.insert(0, C.AAAI_REPO)
_CWD = os.getcwd()
os.chdir(C.AAAI_REPO)                       # the repo's helpers use relative paths
import utils.common as _common                                       # noqa: E402
from utils.common import (country_human_dict, get_prompt_of_language_few_shot,  # noqa
                          get_prompt_of_rq2)
os.chdir(_CWD)

if not hasattr(_common, "args"):            # see run_aaai_batch.py
    _common.args = type("_Stub", (), {"culture": ""})()

METHODS = ["few-shot-anon", "language_few-shot-anon"]

# (old, new, expected number of hits) per country, applied in order to the *target
# language* prompt. The English prompt needs no table: its demonym comes from
# country_human_dict and always appears as "group of <demonym>".
_ANON = {
    # 一组中国人 / 该组中国人 / 一组中国人 / 一组中国受访者
    "China": [("中国", "", 4)],
    # 日本人グループ x2, 日本が..., 日本人のグループ. The second occurrence is already
    # determined ("その日本人グループ"), so it takes "その" rather than a second "ある".
    "Japan": [("その日本人グループ", "そのグループ", 1),
              ("日本人グループ", "あるグループ", 1),
              ("まず、日本が", "まず、あるグループが", 1),
              ("日本人のグループ", "あるグループ", 1)],
    # einer/der Gruppe von Deutschen x3, Antworten von Deutschen x1
    "Germany": [("Deutschen", "Menschen", 4)],
    # 한국인들의 / 해당 그룹의 한국인들이 / 한국인들이 / 한 그룹의 한국인들이
    "Korea": [("한국인들", "사람들", 4)],
    # группы россиян x3, группы российских респондентов x1
    "Russia": [("российских респондентов", "респондентов", 1),
               ("россиян", "людей", 3)],
    # nhóm người Việt Nam x3, nhóm người trả lời là người Việt Nam x1
    "Vietnam": [("nhóm người trả lời là người Việt Nam", "nhóm người trả lời", 1),
                ("người Việt Nam", "người", 3)],
    # grupo de brasileiros x3, grupo de respondentes brasileiros x1
    "Brazil": [("grupo de respondentes brasileiros", "grupo de respondentes", 1),
               ("grupo de brasileiros", "grupo de pessoas", 3)],
}
# the three Spanish blocks differ only in the demonym
for _c, _d in [("Argentina", "argentinos"), ("Chile", "chilenos"),
               ("Uruguay", "uruguayos")]:
    _ANON[_c] = [(f"grupo de encuestados {_d}", "grupo de encuestados", 1),
                 (f"grupo de {_d}", "grupo de personas", 3)]

# The instruction, the example block and the question are one string, and a demonym can
# occur inside a question. Substitution is therefore confined to the instruction, which
# ends exactly where the example block begins -- and the example block is a file, so the
# split point is found rather than guessed. Guessing it does not work: several plausible
# markers ("На основе", "Dados los campos") also occur inside the instruction itself.
_FEWSHOT_LANG = {"China": "Zh", "Japan": "Ja", "Germany": "De", "Korea": "Ko",
                 "Russia": "Ru", "Vietnam": "Vi", "Brazil": "Pt",
                 "Argentina": "Es", "Chile": "Es", "Uruguay": "Es"}


def _apply(text, rules, where):
    """Substitute inside the instruction only, asserting every rule fires as expected."""
    for old, new, n in rules:
        got = text.count(old)
        assert got == n, f"{where}: {old!r} appears {got} times, expected {n}"
        text = text.replace(old, new)
    return text


def _split(prompt, country, english=False):
    """-> (instruction, example block + question), split at the example file verbatim."""
    lang = "En" if english else _FEWSHOT_LANG[country]
    with open(f"{C.AAAI_REPO}/inputs/few_shot/lang-{lang}_dist-{country}.txt",
              encoding="utf-8") as f:
        block = f.read()
    i = prompt.index(block)
    return prompt[:i], prompt[i:]


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def anon_prompt(method, country, data):
    """The conference few-shot prompt for `country`, with the country name removed."""
    if method == "few-shot-anon":
        named = get_prompt_of_rq2(_Args(steering_method="few-shot", country=country), data)
        human = country_human_dict[country]
        instr, rest = _split(named, country, english=True)
        rules = [(f"group of {human}", "group of people", 4)]
        return _apply(instr, rules, f"{country} (En)") + rest
    if method == "language_few-shot-anon":
        named = get_prompt_of_language_few_shot(country, data)
        instr, rest = _split(named, country)
        return _apply(instr, _ANON[country], f"{country} (target language)") + rest
    raise ValueError(method)


def _demo():
    """Print the instruction of both conditions for every country, named and anonymous."""
    os.chdir(C.AAAI_REPO)
    from utils.common import country_language_dict
    from utils.wvs import WVS
    for country in ["China", "Germany", "Japan", "Korea", "Russia", "Vietnam", "Brazil",
                    "Argentina", "Chile", "Uruguay"]:
        for method in METHODS:
            lang = country_language_dict[country] if "language" in method else "English"
            d = WVS("WV7", lang).data_list[0]
            named = (get_prompt_of_language_few_shot(country, d)
                     if "language" in method else
                     get_prompt_of_rq2(_Args(steering_method="few-shot",
                                             country=country), d))
            anon = anon_prompt(method, country, d)
            en = "language" not in method
            i_n, i_a = _split(named, country, en)[0], _split(anon, country, en)[0]
            print(f"\n=== {country} / {method} " + "=" * 30)
            print("named: " + i_n.strip())
            print("anon : " + i_a.strip())
            assert i_a != i_n, "the country name was not removed"
            assert anon[len(i_a):] == named[len(i_n):], "the example block changed"


if __name__ == "__main__":
    _demo()
