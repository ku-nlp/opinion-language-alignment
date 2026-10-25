"""Steering-prompt boilerplate for languages the AAAI repo never covered.

`language` steering means the whole prompt is in the target language: the task
instructions, the few-shot block's headers, and the Question/Answer labels. The repo
hard-codes those for nine languages; Greek, Turkish, Ukrainian, Kazakh and Kyrgyz need
them written. They are translated by a model, with the eight existing non-English
versions supplied as style anchors, and then checked structurally -- a wrong script or a
label that leaked English would silently make "language steering" a mixed-language
prompt, which is exactly the condition the design has to avoid.

  python code/lang_assets.py generate --langs Greek,Turkish,Ukrainian,Kazakh,Kyrgyz
  python code/lang_assets.py persona
  python code/lang_assets.py check
"""
import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ASSETS = f"{C.ROOT}/code/lang_assets.json"
INSTR = C.LANG_INSTR
NAME2CODE = {v: k for k, v in C.LANG_FILE.items()}
ANCHORS = ["ru", "de", "vi", "ko"]      # varied scripts and word orders
SCRIPT = {"el": "GREEK", "uk": "CYRILLIC", "kk": "CYRILLIC", "ky": "CYRILLIC",
          "tr": "LATIN", "ru": "CYRILLIC", "zh": "CJK", "ko": "HANGUL",
          # `zt` shares the CJK block with `zh`; the script check cannot tell simplified
          # from traditional, and what it is here to catch is English leaking through
          "ar": "ARABIC", "zt": "CJK"}

FIELDS = """\
instr0   the long task instruction: a distribution over answer choices will be shown, \
five examples follow, then express the distribution for the asked question, do not copy \
the example distributions, answer only as a dictionary mapping answer-choice number to \
probability, one sentence, nothing else
instr1   the short follow-up: first some examples of the task to show the answer \
formatting, then the model performs the task itself
turn     the "your turn" marker that precedes the real question (keep any leading \
newline and trailing space as in the anchors)
qlabel   the word labelling a question, e.g. "Question"
alabel   the word labelling an answer, e.g. "Answer"
header1  the few-shot block's first line: given the fields `question`, produce the \
fields `answer` -- keep the two backquoted English identifiers `question` and `answer` \
untranslated
header2  the few-shot block's second line: follow the following format"""

SYSTEM = """You translate the boilerplate of a survey-simulation prompt into {lang}.

The prompt asks a language model to report the distribution of answers a group of people \
would give to a World Values Survey question. Every part of it must be in {lang}: a \
respondent-facing instrument mixed with English would defeat the experiment.

You are given the English original and the same text as it was already written for four \
other languages. Match those: same register (polite, addressed to the reader), same \
length, same level of explicitness. They are the house style.

Fields to produce:
{fields}

Keep the backquoted identifiers `question` and `answer` in header1 untranslated -- they \
name JSON fields, not words. Preserve leading newlines and trailing spaces exactly where \
the anchors have them. Reply only through the structured output."""

SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in
                   ["instr0", "instr1", "turn", "qlabel", "alabel", "header1", "header2"]},
    "required": ["instr0", "instr1", "turn", "qlabel", "alabel", "header1", "header2"],
    "additionalProperties": False,
}


def script_share(text, name):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(name in unicodedata.name(c, "") for c in letters) / len(letters)


def anchors_payload(spec, assets):
    out = {}
    for code in ANCHORS:
        s, a = spec[code], assets[code]
        out[C.LANG_FILE[code]] = {
            "instr0": s["instr"][0], "instr1": s["instr"][1],
            "turn": s["tail"][0], "qlabel": a["qlabel"], "alabel": a["alabel"],
            "header1": a["header1"], "header2": a["header2"],
        }
    en = spec["en_baseline"]
    out["English"] = {"instr0": en["instr"][0], "instr1": en["instr"][1],
                      "turn": en["tail"][0], "qlabel": assets["en"]["qlabel"],
                      "alabel": assets["en"]["alabel"],
                      "header1": assets["en"]["header1"],
                      "header2": assets["en"]["header2"]}
    return out


def generate(langs, model, effort):
    from openai import OpenAI
    cli = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
    spec = json.load(open(INSTR, encoding="utf-8"))
    assets = json.load(open(ASSETS, encoding="utf-8"))
    anchors = anchors_payload(spec, assets)

    for name in langs:
        code = NAME2CODE[name]
        kw = dict(model=model,
                  messages=[{"role": "system",
                             "content": SYSTEM.format(lang=name, fields=FIELDS)},
                            {"role": "user",
                             "content": "English original and existing translations:\n"
                                        + json.dumps(anchors, ensure_ascii=False, indent=1)
                                        + f"\n\nNow produce the same fields in {name}."}],
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": "boilerplate", "strict": True,
                                                   "schema": SCHEMA}})
        if effort:
            kw["reasoning_effort"] = effort
        r = cli.chat.completions.create(**kw)
        d = json.loads(r.choices[0].message.content)

        spec[code] = {
            "instr": [d["instr0"], d["instr1"]],
            "fewshot": f"lang-{code}_dist-random.txt",
            # the original writes turn / "\nQ: " / "\n" / "\nA:" as four appends
            "tail": [d["turn"], f"\n{d['qlabel']}: ", "\n", f"\n{d['alabel']}:"],
        }
        assets[code] = {"header1": d["header1"], "header2": d["header2"],
                        "qlabel": d["qlabel"], "alabel": d["alabel"]}
        print(f"{name} ({code}): {r.usage.prompt_tokens} in, {r.usage.completion_tokens} out")

    json.dump(spec, open(INSTR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(assets, open(ASSETS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {INSTR}\nwrote {ASSETS}")


def check():
    spec = json.load(open(INSTR, encoding="utf-8"))
    assets = json.load(open(ASSETS, encoding="utf-8"))
    en = spec["en_baseline"]
    print(f"{'lang':6s} {'script':>7s} {'instr0 len':>11s} {'vs en':>6s} "
          f"{'qlabel':>12s} {'alabel':>12s}  issues")
    for code in sorted(set(spec) - {"en_baseline", "en_persona"}):
        s, a = spec[code], assets.get(code, {})
        joined = s["instr"][0] + s["instr"][1] + "".join(s["tail"])
        sc = SCRIPT.get(code)
        share = script_share(joined, sc) if sc else float("nan")
        bad = []
        if sc and share < 0.8:
            bad.append(f"only {share:.0%} {sc}")
        if len(s["tail"]) != 4:
            bad.append(f"tail has {len(s['tail'])} parts, expected 4")
        if "`question`" not in a.get("header1", "") or "`answer`" not in a.get("header1", ""):
            bad.append("header1 lost the `question`/`answer` identifiers")
        if code != "en" and a.get("qlabel") == "Question":
            bad.append("qlabel still English")
        ratio = len(s["instr"][0]) / len(en["instr"][0])
        if not 0.5 < ratio < 2.0:
            bad.append(f"instr0 length {ratio:.2f}x English")
        print(f"{code:6s} {share:7.2f} {len(s['instr'][0]):11d} {ratio:6.2f} "
              f"{a.get('qlabel', '-'):>12s} {a.get('alabel', '-'):>12s}  "
              f"{'; '.join(bad) if bad else 'ok'}")


PERSONA_SYSTEM = """You add a persona phrase to an existing {lang} prompt, changing \
nothing else.

The prompt instructs a model to report how people would answer a survey question. Two \
sentences of it must be turned from "express the distribution of answers" into "express \
the distribution of answers FROM A GROUP OF <people from some country>", exactly as the \
English version does:

  "please express the distribution of answers for the question asked"
    -> "please express the distribution of answers from a group of {{}} for the question asked"
  "then you will perform the task yourself."
    -> "then you will perform the task yourself for {{}}."

Return each sentence rewritten in {lang} with the literal placeholder {{}} where the \
group phrase goes, plus `group`, the {lang} noun phrase for "people from {{}}" (again \
with a {{}} placeholder for the country name), and `countries`, the {lang} name of each \
listed country.

Hard requirement: apart from the inserted phrase, the rewritten text must be CHARACTER \
FOR CHARACTER the original. Do not fix typos, do not re-punctuate, do not reorder. \
Grammar may force the phrase to a different position than in English -- that is fine and \
expected -- but nothing else may move. The two versions are subtracted from each other \
in the analysis, so any other edit shows up as a persona effect that is not one."""

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "instr0": {"type": "string"},
        "instr1": {"type": "string"},
        "group": {"type": "string"},
        "countries": {"type": "object",
                      "properties": {}, "additionalProperties": {"type": "string"}},
    },
    "required": ["instr0", "instr1", "group", "countries"],
    "additionalProperties": False,
}


def insertions_only(original, modified):
    """-> list of inserted fragments, or None if anything was deleted or changed."""
    import difflib
    ins = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, modified,
                                                       autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            return None
        ins.append(modified[j1:j2])
    return ins


def persona(model, effort, langs=None):
    """Persona slots, merged into the existing file.

    `langs` restricts the run to the named languages. Regenerating a language that has
    already been run would silently reword its persona phrase, so every prompt built
    afterwards would differ from the ones already in results/*/raw.csv; the default is
    therefore to fill in only what is missing rather than to rebuild everything."""
    from openai import OpenAI
    cli = OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())
    spec = json.load(open(INSTR, encoding="utf-8"))
    countries = sorted({c["country"] for c in C.DID_CELLS} |
                       {C.COUNTRY_DISPLAY[f] for c in C.DID_CELLS
                        for f in c.get("foreign", {}).values()})
    path = f"{C.ROOT}/code/persona_slots.json"
    out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    want = set(spec) - {"en_baseline", "en_persona"}
    if langs:
        want &= {NAME2CODE.get(l, l) for l in langs}
    for code in sorted(want):
        name = C.LANG_FILE[code]
        s = spec[code]
        kw = dict(model=model,
                  messages=[{"role": "system", "content": PERSONA_SYSTEM.format(lang=name)},
                            {"role": "user", "content": json.dumps(
                                {"instr0": s["instr"][0], "instr1": s["instr"][1],
                                 "countries": countries}, ensure_ascii=False)}],
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": "persona", "strict": False,
                                                   "schema": PERSONA_SCHEMA}})
        if effort:
            kw["reasoning_effort"] = effort
        d = json.loads(cli.chat.completions.create(**kw).choices[0].message.content)
        i0 = insertions_only(s["instr"][0], d["instr0"])
        i1 = insertions_only(s["instr"][1], d["instr1"])
        status = "ok" if i0 is not None and i1 is not None else "REWROTE TEXT"
        if i0 is not None and (not i0 or "{}" not in "".join(i0)):
            status = "no {} placeholder in instr0"
        print(f"{code:4s} {name:12s} {status:22s} inserted: "
              f"{[x[:40] for x in (i0 or [])] + [x[:40] for x in (i1 or [])]}")
        out[code] = dict(instr0=d["instr0"], instr1=d["instr1"], group=d["group"],
                         countries=d["countries"], clean=(i0 is not None and i1 is not None))
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")
    print("Native-speaker review is still needed before publication; the structural check "
          "only proves nothing outside the inserted phrase moved.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--langs", required=True, help="comma-separated language names")
    g.add_argument("--model", default="gpt-5.5")
    g.add_argument("--effort", default=None)
    g.set_defaults(fn=lambda a: generate(a.langs.split(","), a.model, a.effort))
    p = sub.add_parser("persona")
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--effort", default=None)
    p.add_argument("--langs", default=None,
                   help="comma-separated language names; default rebuilds ALL, which "
                        "rewords languages that have already been run")
    p.set_defaults(fn=lambda a: persona(a.model, a.effort,
                                        a.langs.split(",") if a.langs else None))
    c = sub.add_parser("check")
    c.set_defaults(fn=lambda a: check())
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
