"""Build a target-language WV7 questionnaire from an official WVS country PDF.

The English reference (questions/WV7_English.jsonl) is the anchor for everything: it
fixes which items exist, their response codes, and -- importantly -- the *rewriting
convention*. A WVS PDF prints most items as rows of a grid sharing one stem:

    I'd like to ask you how much you trust people from various groups. Could you tell
    me for each whether you trust people from this group completely, somewhat, ...
                          Trust completely  Trust somewhat  Do not trust very much ...
    Q58 Your family              1                2                   3            ...

while the reference renders Q58 as the standalone "How much do you trust your family?".
So this is not transcription: the model has to find the stem and the row in the target
PDF and compose them the same way, using the PDF's official wording rather than
translating the English.

Because the English master PDF is itself one of the inputs, the pipeline has a free
accuracy harness: run it on English and diff against the reference we already have.

  python code/questionnaire.py selftest --max-batches 2
  python code/questionnaire.py auto --pdf questionaire/..._Ukrainian.pdf --lang Ukrainian
  python code/questionnaire.py emit --lang Greek --batch 1        # manual/ChatGPT-UI path
  python code/questionnaire.py build --lang Greek --input raw/greek.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

BATCH = 20
REF = "English"
MODEL = "gpt-5.5"
KEY_FILE = os.path.expanduser("~/.openai_key")
PDF_DIR = f"{C.ROOT}/questionaire"
TEXT_CACHE = f"{PDF_DIR}/.text"
ENGLISH_PDF = "F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf"

# Expected script per language, used to catch "the model answered in English".
# Latin-script targets can't be checked this way.
SCRIPT = {
    "Greek": "GREEK", "Russian": "CYRILLIC", "Ukrainian": "CYRILLIC",
    "Kazakh": "CYRILLIC", "Kyrgyz": "CYRILLIC", "Korean": "HANGUL",
    "Chinese": "CJK",
}

SYSTEM = """You reconstruct World Values Survey Wave 7 questionnaire items in a target \
language from the official country questionnaire PDF.

You are given the full text of the official {lang} WVS-7 questionnaire, and a list of \
reference items in English. For each reference item, locate the SAME item in the {lang} \
questionnaire and report its {lang} wording.

This is NOT a translation task and NOT a verbatim transcription task.

WVS questionnaires print most items as rows of a grid that share one stem, e.g. a stem \
"I'd like to ask you how much you trust people from various groups ..." followed by rows \
"Q58 Your family", "Q59 Your neighborhood". The English reference has already merged stem \
and row into a standalone question ("How much do you trust your family?"). Do the same in \
{lang}: build a standalone question out of the {lang} stem and the {lang} row label, using \
the PDF's own wording for both. Never translate from the English reference -- it is there \
to tell you WHICH item is meant and HOW it was phrased as a standalone question, not to be \
translated.

Mirror the reference's SENTENCE STRUCTURE as closely as {lang} allows -- same question \
type, same clause order, same level of politeness, same treatment of the shared stem -- \
while the substantive words come from the PDF. The nine questionnaires already built this \
way have to stay comparable across languages, so a structurally different rendering of the \
same item is a defect even when it is a fine sentence.

Where existing renderings of the same item in OTHER languages are shown, they are the \
convention applied by a native speaker of that language: study how they turned the PDF's \
stem and row into one sentence and follow the same treatment in {lang}. They are examples \
of form, not of content.

Rules:
0. Give the respondent-facing wording only. Drop interviewer instructions such as \
"(code one answer)", "(read out)", "(do not read)", "(code if volunteered only)" and \
question numbers, exactly as the reference does.
1. "choices" must have exactly as many entries as the reference, listed in the ORDER THEY \
ARE PRINTED IN THE PDF, and "choice_keys" must be the response codes the PDF prints for \
them, in the same order. Reproduce the target questionnaire as the respondent saw it: its \
own option order and its own numbering, even where they differ from the English reference. \
The reference is only there to tell you which item is meant.
2. Option labels should be the PDF's official {lang} labels, copied as printed. If a grid \
abbreviates a label ("ВВ" for "важко відповісти"), give the full respondent-facing phrase \
the PDF uses elsewhere for that response, not the abbreviation.
3. If an item is genuinely absent from this questionnaire, set "found": false and leave \
"question" and "choices" empty. Never invent, back-translate or guess wording. A missing \
item is fine; a fabricated one is not.
4. "pdf_item" is the item number as printed in the PDF (e.g. "Q58"), for auditing.
5. Reply only through the structured output; no commentary."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "found": {"type": "boolean"},
                    "pdf_item": {"type": "string"},
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                    "choice_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "found", "pdf_item", "question", "choices", "choice_keys"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

PROMPT = """The attached PDF is the OFFICIAL {lang} version of the World Values Survey \
Wave 7 questionnaire.

For each reference item below, find the SAME item in the PDF and give its {lang} wording, \
composed into a standalone question the way the English reference is. Do not translate the \
English; use the PDF's own {lang} wording.

Rules:
1. Output one JSON object per line (JSONL). No markdown fences, no commentary, nothing else.
2. Copy "id" unchanged from the reference item.
3. "choices" must be listed in the order they are PRINTED IN THE PDF, and "choice_keys" \
must be the response codes the PDF prints for them, in the same order.
4. If an item is genuinely absent from this PDF, output {{"id": "...", "missing": true}} \
for it. Never invent or back-translate wording.
5. Include "pdf_item": "<the item number as printed in the PDF>" so the mapping can be \
audited afterwards.
6. Do NOT output an "answer" field; it is regenerated automatically.

Reference items ({n} of them, batch {b} of {nb}):
{items}"""


# ---------------------------------------------------------------- reference & io

def reference():
    path = f"{C.ROOT}/questions/WV7_{REF}.jsonl"
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [d for d in items if d["type"] == "Opinion-Dependent"]


def ref_line(d, shots=None):
    """One reference item, optionally with existing renderings in other languages.

    The eight questionnaires already in the repo are the only record of how the
    stem-plus-row rewriting convention is applied outside English, so showing a couple of
    them per item teaches the convention far better than describing it.
    """
    rec = {"reference_english": {k: d[k] for k in ("id", "question", "choices")},
           "response_codes": d["choice_keys"]}
    ex = {}
    for lang, items in (shots or {}).items():
        o = items.get(d["id"])
        if o and len(o["choices"]) == len(d["choices"]):
            ex[lang] = {"question": o["question"], "choices": o["choices"]}
    if ex:
        rec["existing_renderings"] = ex
    if not shots:
        rec = {k: d[k] for k in ("id", "question", "choices", "choice_keys")}
    return json.dumps(rec, ensure_ascii=False)


def load_shots(names):
    out = {}
    for name in names:
        path = f"{C.ROOT}/questions/WV7_{name}.jsonl"
        if not os.path.exists(path):
            sys.exit(f"no such questionnaire for few-shot: {path}")
        out[name] = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
    return out


# quotes and non-breaking spaces only: dashes carry punctuation the existing
# questionnaires preserve, so folding them would rewrite their text
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', " ": " "})


def clean(s):
    """Typographic normalisation only -- the PDFs use curly quotes, the reference doesn't."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).translate(QUOTES)).strip()


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip().strip(",")
            if not line or line in "[]" or line.startswith("```"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! line {ln} is not JSON, skipped: {e}", file=sys.stderr)
    return out


RTL = re.compile(r"[֐-׿؀-ۿ܀-ݏ]")


def pdf_text(path):
    """Extract once and cache; WVS PDFs are text-based, no OCR needed.

    pdfplumber emits glyphs in visual order, which for a right-to-left script means every
    Arabic or Hebrew word comes out reversed: the Tunisia PDF yields `لاوملأا` where the
    page reads `الأموال`. PyMuPDF returns logical order. It is used only when the page
    actually contains RTL text, so the extraction of the questionnaires already built
    from this function is untouched.
    """
    os.makedirs(TEXT_CACHE, exist_ok=True)
    cache = f"{TEXT_CACHE}/{os.path.basename(path)}.txt"
    if os.path.exists(cache):
        return open(cache, encoding="utf-8").read()
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, 1):
            pages.append(f"\n=== PAGE {i} ===\n" + (p.extract_text() or ""))
    text = "".join(pages)
    if RTL.search(text):
        import fitz
        doc = fitz.open(path)
        text = "".join(f"\n=== PAGE {i} ===\n" + p.get_text()
                       for i, p in enumerate(doc, 1))
        print("  right-to-left script detected: extracted with PyMuPDF instead")
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------- validation

def script_share(text, name):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(name in unicodedata.name(c, "") for c in letters) / len(letters)


# Items an official questionnaire prints, but not as the master's question. The option
# count still matches, so nothing downstream would catch them; they are listed here so a
# rebuild reproduces the same file and the reason survives with it. This follows what the
# French build did to Q111 by accident of the source: where the printed item is not the
# master's item, it is dropped rather than patched.
SOURCE_MISMATCH = {
    "ChineseTraditional": {
        # Q122-Q129 are printed as "is X related to Hong Kong or not", answered on
        # related / hard to say / unrelated, where the master asks agree-disagree about
        # what immigration does. Three options either way, a different scale entirely.
        **{f"Q{n}": "asks whether the item is related to Hong Kong, not agree/disagree"
           for n in range(122, 130)},
        # The Hong Kong PDF prints 墮落 (moral degeneration) where the master asks about
        # abortion; 墮胎 does not occur anywhere in the document.
        "Q184": "prints 墮落 (degeneration) where the master asks about abortion",
    },
}


def validate(got, lang):
    """-> (usable items, problems, ids the model called missing, ids never returned)"""
    ref = {d["id"]: d for d in reference()}
    out, problems, missing, seen = [], [], [], set()
    expect = SCRIPT.get(lang)
    dropped = SOURCE_MISMATCH.get(lang, {})

    for d in got:
        qid = d.get("id")
        if qid not in ref:
            problems.append(f"{qid}: not an opinion-dependent question id in the reference")
            continue
        if qid in dropped:
            problems.append(f"{qid}: dropped, {dropped[qid]}")
            continue
        if qid in seen:
            problems.append(f"{qid}: duplicated in the input")
            continue
        seen.add(qid)
        if d.get("missing") or d.get("found") is False:
            missing.append(qid)
            continue

        r = ref[qid]
        q, ch = clean(d.get("question") or ""), d.get("choices")
        if not q or not isinstance(ch, list):
            problems.append(f"{qid}: missing question or choices")
            continue
        if len(ch) != len(r["choice_keys"]):
            problems.append(f"{qid}: {len(ch)} choices but the reference has "
                            f"{len(r['choice_keys'])} response codes")
            continue
        keys = [str(k).strip() for k in (d.get("choice_keys") or r["choice_keys"])]
        if len(keys) != len(ch):
            problems.append(f"{qid}: {len(keys)} choice_keys for {len(ch)} choices")
            continue
        if len(set(keys)) != len(keys):
            problems.append(f"{qid}: duplicated response codes {keys}")
            continue
        if any(not isinstance(c, str) or not clean(c) for c in ch):
            problems.append(f"{qid}: an empty choice")
            continue
        if lang != REF and q == clean(r["question"]):
            problems.append(f"{qid}: question is byte-identical to the English -- "
                            f"probably not transcribed")
            continue
        if expect and script_share(q, expect) < 0.5:
            problems.append(f"{qid}: question is not mostly {expect} script -- "
                            f"probably English or a placeholder")
            continue

        out.append({
            "id": qid,
            "question": q,
            "choices": [clean(c) for c in ch],
            # the codes the target PDF prints, in the target PDF's own option order
            "choice_keys": keys,
            # numbered by the response code, NOT 1..N -- Q119 and Q122-Q129 code
            # "Hard to say" as 0, and the prompt shows the model these very keys, so
            # renumbering here would make the model emit keys nothing looks up.
            "answer": "\n".join(f"{k}. {clean(c)}" for k, c in zip(keys, ch)),
            "type": "Opinion-Dependent",
        })

    out.sort(key=lambda d: list(ref).index(d["id"]))
    return out, problems, missing, [q for q in ref if q not in seen]


def report(out, problems, missing, absent, n_ref):
    print(f"usable          {len(out):3d}/{n_ref}")
    print(f"marked missing  {len(missing):3d}   {' '.join(missing)}")
    print(f"never returned  {len(absent):3d}   {' '.join(absent)}")
    if problems:
        print(f"\nrejected ({len(problems)}):")
        for p in problems:
            print(f"  {p}")


def write_out(out, lang, force):
    path = f"{C.ROOT}/questions/WV7_{lang}.jsonl"
    if os.path.exists(path) and not force:
        sys.exit(f"\n{path} exists; pass --force to overwrite")
    with open(path, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nwrote {path}")
    return path


# ---------------------------------------------------------------- API agent

def client():
    from openai import OpenAI
    if not os.path.exists(KEY_FILE):
        sys.exit(f"no API key at {KEY_FILE}")
    return OpenAI(api_key=open(KEY_FILE).read().strip())


def ask(cli, model, lang, text, items, effort, shots):
    """One API call for one batch of reference items."""
    msgs = [
        {"role": "system", "content": SYSTEM.format(lang=lang)},
        # PDF first so the long, constant part of the prompt can be cached across batches
        {"role": "user", "content": f"OFFICIAL {lang.upper()} WVS-7 QUESTIONNAIRE:\n\n{text}"},
        {"role": "user", "content": "Reference items (English):\n" +
                                    "\n".join(ref_line(d, shots) for d in items)},
    ]
    kw = dict(model=model, messages=msgs,
              response_format={"type": "json_schema",
                               "json_schema": {"name": "wvs_batch", "strict": True,
                                               "schema": SCHEMA}})
    if effort:
        kw["reasoning_effort"] = effort
    r = cli.chat.completions.create(**kw)
    return json.loads(r.choices[0].message.content)["items"], r.usage


VERIFY_SYSTEM = """You audit a machine-built translation of a World Values Survey \
questionnaire. For each item you get the English reference and the {lang} version that was \
produced. Report any item where the {lang} version would give a respondent a different \
task than the English one.

Flag, in particular:
- the i-th {lang} option does not mean the same as the i-th English option (options \
permuted, a scale reversed, a "don't know"/"hard to say" option in the wrong slot). This \
is the failure that matters most.
- an option rendered as an abbreviation or code rather than the phrase a respondent hears
- interviewer instructions or question numbers left in the text
- the question asking about something other than what the English asks about

Do not flag wording that is merely a different but faithful rendering, a different register, \
or a longer/shorter phrasing. Only flag differences that change what is being asked or which \
answer maps to which response code. Set ok=false only when you would want a human to look."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "issue": {"type": "string"},
                },
                "required": ["id", "ok", "issue"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def verify(cli, model, lang, out, effort, batch_size, workers):
    """Second model pass over the finished items; catches order/meaning errors."""
    ref = {d["id"]: d for d in reference()}
    pairs = [{"id": d["id"],
              "english": {"question": ref[d["id"]]["question"],
                          "choices": ref[d["id"]]["choices"]},
              lang.lower(): {"question": d["question"], "choices": d["choices"]}}
             for d in out]
    chunks = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    def one(batch):
        kw = dict(model=model,
                  messages=[{"role": "system", "content": VERIFY_SYSTEM.format(lang=lang)},
                            {"role": "user", "content": "\n".join(
                                json.dumps(p, ensure_ascii=False) for p in batch)}],
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": "audit", "strict": True,
                                                   "schema": VERIFY_SCHEMA}})
        if effort:
            kw["reasoning_effort"] = effort
        try:
            r = cli.chat.completions.create(**kw)
            return json.loads(r.choices[0].message.content)["items"]
        except Exception as e:
            print(f"  ! verify batch failed: {e}", file=sys.stderr)
            return []

    flagged = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, chunks):
            flagged.extend(r for r in res if not r.get("ok"))
    return sorted(flagged, key=lambda r: list(ref).index(r["id"]) if r["id"] in ref else 0)


def run_agent(pdf, lang, model, batch_size, workers, max_batches, effort, retry, shot_names):
    cli = client()
    text = pdf_text(pdf)
    shots = load_shots([n for n in shot_names if n != lang])
    ref = reference()
    batches = [ref[i:i + batch_size] for i in range(0, len(ref), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
    print(f"{os.path.basename(pdf)}: {len(text)} chars | {len(batches)} batches "
          f"x {batch_size} items | model={model} | few-shot={list(shots) or 'none'}")

    got, tok_in, tok_out, tok_cached = [], 0, 0, 0

    def one(i, b):
        for attempt in range(3):
            try:
                return ask(cli, model, lang, text, b, effort, shots)
            except Exception as e:
                if attempt == 2:
                    print(f"  ! batch {i + 1} failed: {e}", file=sys.stderr)
                    return [], None
        return [], None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i, b): i for i, b in enumerate(batches)}
        for fu in cf.as_completed(futs):
            items, usage = fu.result()
            got.extend(items)
            if usage:
                tok_in += usage.prompt_tokens
                tok_out += usage.completion_tokens
                det = getattr(usage, "prompt_tokens_details", None)
                tok_cached += getattr(det, "cached_tokens", 0) or 0
            print(f"  batch {futs[fu] + 1}/{len(batches)} -> {len(items)} items", flush=True)

    if retry:
        _, problems, missing, absent = validate(got, lang)
        scope = {d["id"] for b in batches for d in b}
        # only chase items the first pass was actually asked for -- otherwise
        # --max-batches silently turns into a full run
        bad = ({p.split(":")[0] for p in problems} | set(absent)) & scope
        if bad:
            print(f"\nsecond pass on {len(bad)} items the first pass did not deliver")
            byid = {d["id"]: d for d in ref}
            todo = [byid[q] for q in ref_order(ref, bad)]
            small = [todo[i:i + 5] for i in range(0, len(todo), 5)]
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                for items, usage in ex.map(lambda b: one(-1, b), small):
                    got = [g for g in got if g.get("id") not in {i["id"] for i in items}]
                    got.extend(items)
                    if usage:
                        tok_in += usage.prompt_tokens
                        tok_out += usage.completion_tokens
                        det = getattr(usage, "prompt_tokens_details", None)
                        tok_cached += getattr(det, "cached_tokens", 0) or 0

    # cached input is billed at a large discount, so it is the number that decides
    # what a run actually costs -- the PDF is a constant prefix across batches
    print(f"\ntokens: {tok_in} in ({tok_cached} of them cached), {tok_out} out")
    return got, batches


def ref_order(ref, ids):
    return [d["id"] for d in ref if d["id"] in ids]


# ---------------------------------------------------------------- commands

def cmd_emit(args):
    ref = reference()
    nb = (len(ref) + BATCH - 1) // BATCH
    if not 1 <= args.batch <= nb:
        sys.exit(f"batch must be in 1..{nb}")
    chunk = ref[(args.batch - 1) * BATCH: args.batch * BATCH]
    print(PROMPT.format(lang=args.lang, n=len(chunk), b=args.batch, nb=nb,
                        items="\n".join(ref_line(d) for d in chunk)))


def cmd_build(args):
    ref = reference()
    got = load_jsonl(args.input)
    print(f"reference: {len(ref)} opinion questions | input has {len(got)} objects\n")
    out, problems, missing, absent = validate(got, args.lang)
    report(out, problems, missing, absent, len(ref))
    if args.check_only:
        print("\n--check-only: nothing written")
        return
    if not out:
        sys.exit("\nnothing usable, not writing")
    write_out(out, args.lang, args.force)


def cmd_auto(args):
    got, batches = run_agent(args.pdf, args.lang, args.model, args.batch_size,
                             args.workers, args.max_batches, args.effort,
                             not args.no_retry, [x for x in args.fewshot.split(",") if x])
    n_ref = sum(len(b) for b in batches)
    save_raw(got, args.lang)
    out, problems, missing, absent = validate(got, args.lang)
    absent = [q for q in absent if q in {d["id"] for b in batches for d in b}]
    report(out, problems, missing, absent, n_ref)

    if out and not args.no_verify:
        print(f"\naudit pass over {len(out)} items ...")
        flagged = verify(client(), args.model, args.lang, out, args.effort,
                         args.batch_size, args.workers)
        print(f"flagged {len(flagged)}/{len(out)}:")
        for r in flagged:
            print(f"  {r['id']}: {r['issue']}")

    if args.check_only:
        print("\n--check-only: nothing written")
        return
    write_out(out, args.lang, args.force)


def save_raw(got, tag):
    os.makedirs(f"{C.ROOT}/raw", exist_ok=True)
    path = f"{C.ROOT}/raw/{tag.lower()}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in got:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"raw model output -> {path}\n")
    return path


def cmd_selftest(args):
    """Run the agent on the English master PDF and diff against the known reference."""
    pdf = f"{PDF_DIR}/{ENGLISH_PDF}"
    got, batches = run_agent(pdf, REF, args.model, args.batch_size, args.workers,
                             args.max_batches, args.effort, not args.no_retry,
                             [x for x in args.fewshot.split(",") if x])
    save_raw(got, "selftest_english")
    scope = {d["id"] for b in batches for d in b}
    out, problems, missing, absent = validate(got, REF)
    absent = [q for q in absent if q in scope]
    report(out, problems, missing, absent, len(scope))

    ref = {d["id"]: d for d in reference()}
    # same typographic normalisation as the writer, plus quote-style folding: the
    # reference uses curly quotes where the PDF uses straight ones and vice versa, and
    # that difference is not an error worth counting
    norm = lambda s: clean(s).lower().rstrip("?.:").replace("'", '"')
    exact_q = exact_c = both = 0
    diffs = []
    for d in out:
        r = ref[d["id"]]
        q_ok = norm(d["question"]) == norm(r["question"])
        c_ok = [norm(c) for c in d["choices"]] == [norm(c) for c in r["choices"]]
        exact_q += q_ok
        exact_c += c_ok
        both += q_ok and c_ok
        if not (q_ok and c_ok):
            diffs.append((d["id"], q_ok, c_ok, d, r))

    n = len(out)
    print(f"\n=== agreement with questions/WV7_English.jsonl (n={n} recovered) ===")
    if n:
        print(f"  question matches   {exact_q:3d}/{n}  ({100*exact_q/n:.1f}%)")
        print(f"  choices match      {exact_c:3d}/{n}  ({100*exact_c/n:.1f}%)")
        print(f"  both               {both:3d}/{n}  ({100*both/n:.1f}%)")
    print(f"\nfirst {min(len(diffs), args.show)} disagreements:")
    for qid, q_ok, c_ok, d, r in diffs[:args.show]:
        print(f"\n  {qid}  question={'ok' if q_ok else 'DIFF'} choices={'ok' if c_ok else 'DIFF'}")
        if not q_ok:
            print(f"    ref   {r['question']}")
            print(f"    model {d['question']}")
        if not c_ok:
            print(f"    ref   {r['choices']}")
            print(f"    model {d['choices']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def api_args(p):
        p.add_argument("--model", default=MODEL)
        p.add_argument("--batch-size", type=int, default=BATCH)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--max-batches", type=int, default=0, help="0 = all; use to cap cost")
        p.add_argument("--effort", default=None, help="reasoning effort, if the model takes it")
        p.add_argument("--no-retry", action="store_true", help="skip the second pass")
        p.add_argument("--fewshot", default="Russian,Spanish",
                       help="comma-separated existing questionnaires to show as examples "
                            "of the rewriting convention; '' disables")

    e = sub.add_parser("emit", help="print a prompt for one batch (manual ChatGPT-UI path)")
    e.add_argument("--lang", required=True)
    e.add_argument("--batch", type=int, default=1)
    e.set_defaults(fn=cmd_emit)

    b = sub.add_parser("build", help="validate a JSONL file and write the questionnaire")
    b.add_argument("--lang", required=True)
    b.add_argument("--input", required=True)
    b.add_argument("--check-only", action="store_true")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("auto", help="end-to-end: PDF -> questionnaire, via the API")
    a.add_argument("--pdf", required=True)
    a.add_argument("--lang", required=True)
    a.add_argument("--check-only", action="store_true")
    a.add_argument("--force", action="store_true")
    a.add_argument("--no-verify", action="store_true", help="skip the audit pass")
    api_args(a)
    a.set_defaults(fn=cmd_auto)

    s = sub.add_parser("selftest", help="run on the English master PDF and score the result")
    s.add_argument("--show", type=int, default=10, help="how many disagreements to print")
    api_args(s)
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
