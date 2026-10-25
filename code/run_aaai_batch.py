"""Re-run the conference experiments for one API model through the OpenAI Batch API.

The journal version uses one model list everywhere, which means gpt-4-0613 is replaced by
gpt-4.1 and the DeepSeek family is dropped. The DeepSeek results are simply not used, but
gpt-4.1 has never been run, so every experiment of the conference version has to be
produced for it: the unsteered run, the five steering methods on ten countries, and the
three sensitivity settings. The temporal analysis needs no extra run, because it scores
the unsteered answers against the human data of three waves.

Prompts are built by importing the conference repository's own functions rather than
re-implementing them, so a gpt-4.1 row is produced by exactly the code that produced the
other rows. Results are written into that repository in its own layout, next to the
existing models:

    results/RQ1/gpt-4.1/results.csv
    results/RQ2/<method>/<country>/gpt-4.1/results.csv
    results/Discussion/sensitivity/<setting>/gpt-4.1/results.csv

  python code/run_aaai_batch.py count                     # what would be sent, and why
  python code/run_aaai_batch.py submit  --model gpt-4.1
  python code/run_aaai_batch.py status  --model gpt-4.1
  python code/run_aaai_batch.py collect --model gpt-4.1
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.path.insert(0, C.AAAI_REPO)
os.chdir(C.AAAI_REPO)                       # the repo's helpers use relative paths
import utils.common as _common                                   # noqa: E402
from utils.common import (get_prompt_of_rq1, get_prompt_of_rq2,  # noqa: E402
                          match_api_str, read_file_to_string)
from utils.wvs import WVS                                        # noqa: E402

# The Japanese branch of get_prompt_of_language_persona ends in `.format(args.culture)`,
# where `args` is not in scope. The string it formats has no placeholder, so the call is
# a no-op as long as the name resolves; we give the module a stub rather than editing it.
if not hasattr(_common, "args"):
    _common.args = type("_Stub", (), {"culture": ""})()

SYSTEM = "You are a helpful assistant."
FIELDS = ["id", "question", "choices", "choice_keys", "answer", "type", "response"]
STATE_DIR = f"{C.ROOT}/results/aaai_batch"

METHODS = ["persona", "few-shot", "language", "language_persona", "language_few-shot"]
COUNTRIES = ["China", "Japan", "Germany", "Korea", "Russia", "Vietnam", "Brazil",
             "Argentina", "Chile", "Uruguay"]
# the three prompt-sensitivity settings of the Discussion section. `order` shuffles the
# options of every question, the other two replace the five worked examples.
SENSITIVITY = {"order": dict(language="shuffle", few_shot=None),
               "number": dict(language="English",
                              few_shot="inputs/sensitivity/number.txt"),
               "prob": dict(language="English",
                            few_shot="inputs/sensitivity/prob.txt")}


class _Args:                                 # get_prompt_of_rq2 reads attributes off args
    def __init__(self, **kw):
        self.__dict__.update(kw)


def rq1_prompt_with(data, few_shot_path):
    """get_prompt_of_rq1 with a different example block, for the sensitivity settings."""
    p = get_prompt_of_rq1(data)
    default = read_file_to_string("inputs/few_shot/lang-En_dist-random.txt")
    return p.replace(default, read_file_to_string(few_shot_path))


def build():
    """-> [(custom_id, prompt, data_row)] for every request the conference version needs."""
    jobs = []
    for d in WVS("WV7", "English").data_list:
        jobs.append((f"RQ1||{d['id']}", get_prompt_of_rq1(d), d))

    from utils.common import country_language_dict
    for method in METHODS:
        for country in COUNTRIES:
            lang = country_language_dict[country] if "language" in method else "English"
            args = _Args(steering_method=method, country=country)
            for d in WVS("WV7", lang).data_list:
                jobs.append((f"RQ2|{method}|{country}|{d['id']}",
                             get_prompt_of_rq2(args, d), d))

    for name, cfg in SENSITIVITY.items():
        for d in WVS("WV7", cfg["language"]).data_list:
            p = (get_prompt_of_rq1(d) if cfg["few_shot"] is None
                 else rq1_prompt_with(d, cfg["few_shot"]))
            jobs.append((f"SENS|{name}||{d['id']}", p, d))
    return jobs


def split_id(custom_id):
    """-> (kind, a, b, qid). A custom_id carries as many labels as its kind needs
    (`RQ1||Q1`, `RQ2|persona|China|Q1`, `SENS|order||Q1`), and the question id is always
    the last field."""
    parts = custom_id.split("|")
    kind, qid = parts[0], parts[-1]
    labels = parts[1:-1] + ["", ""]
    return kind, labels[0], labels[1], qid


def out_path(custom_id, model):
    kind, a, b, _ = split_id(custom_id)
    if kind == "RQ1":
        return f"{C.AAAI_RESULTS}/RQ1/{model}/results.csv"
    if kind == "RQ2":
        return f"{C.AAAI_RESULTS}/RQ2/{a}/{b}/{model}/results.csv"
    return f"{C.AAAI_RESULTS}/Discussion/sensitivity/{a}/{model}/results.csv"


def client():
    from openai import OpenAI
    return OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())


def state_path(model):
    os.makedirs(STATE_DIR, exist_ok=True)
    return f"{STATE_DIR}/{model}.json"


def cmd_count(args):
    jobs = build()
    chars = sum(len(p) for _, p, _ in jobs)
    tok = chars / 3.5
    by = {}
    for cid, _, _ in jobs:
        by[cid.split("|")[0]] = by.get(cid.split("|")[0], 0) + 1
    print(f"{len(jobs)} requests: " + ", ".join(f"{k} {v}" for k, v in by.items()))
    print(f"~{tok / 1e6:.2f}M input tokens, ~{len(jobs) * 25 / 1e6:.3f}M output tokens")
    print(f"gpt-4.1 batch price: ${tok / 1e6 * 1.0 + len(jobs) * 25 / 1e6 * 4.0:.2f}")


def cmd_submit(args):
    jobs = build()
    # skip anything already collected, so a partially failed batch is resumed. The key is
    # (output file, question id): a custom_id has a variable number of label slots, but
    # the file it lands in plus the question identifies a row uniquely.
    done, seen_files = set(), set()
    for cid, _, _ in jobs:
        p = out_path(cid, args.model)
        if p in seen_files or not os.path.exists(p):
            continue
        seen_files.add(p)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            done.add((p, r["id"]))
    todo = [j for j in jobs if (out_path(j[0], args.model), split_id(j[0])[3]) not in done]
    print(f"{len(jobs)} requests, {len(done)} already present, {len(todo)} to send")
    if not todo:
        return

    os.makedirs(STATE_DIR, exist_ok=True)
    req_path = f"{STATE_DIR}/{args.model}.requests.jsonl"
    with open(req_path, "w", encoding="utf-8") as f:
        for cid, prompt, _ in todo:
            f.write(json.dumps({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": args.model, "temperature": args.temperature,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}]}},
                ensure_ascii=False) + "\n")
    cli = client()
    up = cli.files.create(file=open(req_path, "rb"), purpose="batch")
    b = cli.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                           completion_window="24h")
    json.dump(dict(batch_id=b.id, model=args.model, n=len(todo)),
              open(state_path(args.model), "w"), indent=1)
    print(f"batch {b.id}: {len(todo)} requests, status {b.status}")


def cmd_status(args):
    b = client().batches.retrieve(json.load(open(state_path(args.model)))["batch_id"])
    print(f"{b.id}  {b.status}  {b.request_counts}")


def cmd_collect(args):
    cli = client()
    b = cli.batches.retrieve(json.load(open(state_path(args.model)))["batch_id"])
    if b.status != "completed":
        sys.exit(f"batch is {b.status}, not completed")
    text = cli.files.content(b.output_file_id).text
    meta = {cid: d for cid, _, d in build()}

    have = {}
    rows, bad, dup = {}, 0, 0
    for line in text.splitlines():
        r = json.loads(line)
        cid = r["custom_id"]
        body = (r.get("response") or {}).get("body") or {}
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            bad += 1
            continue
        # the repository strips a leading "Answer: " before matching
        try:
            content = content.split("Answer: ")[1]
        except IndexError:
            pass
        try:
            resp = match_api_str(content)
        except Exception:
            resp = None
        if resp is None:
            bad += 1
            continue
        d = meta[cid]
        path = out_path(cid, args.model)
        if path not in have:
            have[path] = ({r["id"] for r in csv.DictReader(open(path, encoding="utf-8"))}
                          if os.path.exists(path) else set())
        if d["id"] in have[path]:
            dup += 1
            continue
        have[path].add(d["id"])
        rows.setdefault(path, []).append(
            [d["id"], d["question"], d["choices"], d["choice_keys"], d["answer"],
             d["type"], resp])

    for path, items in rows.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerows(items)
    print(f"wrote {sum(len(v) for v in rows.values())} rows to {len(rows)} files, "
          f"{bad} unparseable, {dup} already present")
    if bad:
        print("re-run submit to fill the gaps")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("count", cmd_count), ("submit", cmd_submit),
                     ("status", cmd_status), ("collect", cmd_collect)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name != "count":
            s.add_argument("--model", default="gpt-4.1")
        if name == "submit":
            s.add_argument("--temperature", type=float, default=0)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
