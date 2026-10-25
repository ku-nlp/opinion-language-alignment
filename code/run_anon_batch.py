"""Run the two anonymous few-shot conditions for one API model, through the Batch API.

Deliberately separate from run_aaai_batch.py: that script rebuilds every conference
experiment and would, for a model whose sensitivity runs are missing, quietly send them
too. This one sends the two new conditions and nothing else.

Output goes into the conference repository in its own layout, so the scorer finds the new
rows next to the old ones:

    results/RQ2/few-shot-anon/<country>/<model>/results.csv
    results/RQ2/language_few-shot-anon/<country>/<model>/results.csv

  python code/run_anon_batch.py count
  python code/run_anon_batch.py submit  --model gpt-4.1
  python code/run_anon_batch.py status  --model gpt-4.1
  python code/run_anon_batch.py collect --model gpt-4.1
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                                                   # noqa: E402
from aaai_anon import METHODS, anon_prompt                           # noqa: E402

sys.path.insert(0, C.AAAI_REPO)
from utils.common import country_language_dict, match_api_str        # noqa: E402
from utils.wvs import WVS                                            # noqa: E402

SYSTEM = "You are a helpful assistant."
FIELDS = ["id", "question", "choices", "choice_keys", "answer", "type", "response"]
STATE_DIR = f"{C.ROOT}/results/aaai_batch"
COUNTRIES = ["China", "Japan", "Germany", "Korea", "Russia", "Vietnam", "Brazil",
             "Argentina", "Chile", "Uruguay"]

# gpt-5 rejects an explicit temperature and bills its chain of thought as output; the task
# is to emit one dictionary of percentages, so there is nothing to reason through.
BODY = {"gpt-5": dict(reasoning_effort="minimal", max_completion_tokens=2048)}
DEFAULT_BODY = dict(temperature=0, max_tokens=256)


def build(countries=None):
    """-> [(custom_id, prompt, data_row)] for the two anonymous conditions.

    `countries` narrows the run; the size table of the appendix covers China and Germany
    only, so the five small models have no reason to answer the other eight."""
    cwd = os.getcwd()
    os.chdir(C.AAAI_REPO)                   # WVS() and the example files are relative
    try:
        jobs = []
        for method in METHODS:
            for country in (countries or COUNTRIES):
                lang = country_language_dict[country] if "language" in method else "English"
                for d in WVS("WV7", lang).data_list:
                    jobs.append((f"RQ2|{method}|{country}|{d['id']}",
                                 anon_prompt(method, country, d), d))
        return jobs
    finally:
        os.chdir(cwd)


def out_path(custom_id, model):
    _, method, country, _ = custom_id.split("|")
    return f"{C.AAAI_RESULTS}/RQ2/{method}/{country}/{model}/results.csv"


def client():
    from openai import OpenAI
    return OpenAI(api_key=open(os.path.expanduser("~/.openai_key")).read().strip())


def state_path(model):
    os.makedirs(STATE_DIR, exist_ok=True)
    return f"{STATE_DIR}/{model}.anon.json"


def cmd_count(args):
    jobs = build()
    chars = sum(len(p) for _, p, _ in jobs)
    by = {}
    for cid, _, _ in jobs:
        by[cid.split("|")[1]] = by.get(cid.split("|")[1], 0) + 1
    print(f"{len(jobs)} requests: " + ", ".join(f"{k} {v}" for k, v in by.items()))
    print(f"~{chars / 3.5 / 1e6:.2f}M input tokens per model")


def cmd_submit(args):
    jobs = build()
    done, seen = set(), set()
    for cid, _, _ in jobs:
        p = out_path(cid, args.model)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            done.add((p, r["id"]))
    todo = [j for j in jobs if (out_path(j[0], args.model), j[0].split("|")[3]) not in done]
    print(f"{len(jobs)} requests, {len(done)} already present, {len(todo)} to send")
    if not todo:
        return

    body = BODY.get(args.model, DEFAULT_BODY)
    req_path = f"{STATE_DIR}/{args.model}.anon.requests.jsonl"
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(req_path, "w", encoding="utf-8") as f:
        for cid, prompt, _ in todo:
            f.write(json.dumps({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": args.model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}],
                         **body}}, ensure_ascii=False) + "\n")
    print(f"wrote {req_path} ({os.path.getsize(req_path) / 1e6:.1f} MB), body={body}")
    if args.dry_run:
        print("--dry-run: not submitting")
        return

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
    if b.errors and getattr(b.errors, "data", None):
        for e in b.errors.data[:5]:
            print(f"  error: {e.message}")


def cmd_collect(args):
    cli = client()
    b = cli.batches.retrieve(json.load(open(state_path(args.model)))["batch_id"])
    # a cancelled batch keeps the responses it had already finished, which is the whole
    # point of cancelling one that has stalled; `fill` then covers what is left
    if b.status not in ("completed", "cancelled"):
        sys.exit(f"batch is {b.status}, not completed")
    if not b.output_file_id:
        sys.exit(f"batch is {b.status} and produced no output file")
    meta = {cid: d for cid, _, d in build()}

    have, rows, bad, dup = {}, {}, 0, 0
    for line in cli.files.content(b.output_file_id).text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        body = (r.get("response") or {}).get("body") or {}
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            bad += 1
            continue
        content = content.split("Answer: ")[-1]     # the repo strips this before matching
        try:
            resp = match_api_str(content)
        except Exception:
            resp = None
        if resp is None:
            bad += 1
            continue
        d = meta[r["custom_id"]]
        path = out_path(r["custom_id"], args.model)
        if path not in have:
            have[path] = ({x["id"] for x in csv.DictReader(open(path, encoding="utf-8"))}
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


def cmd_cancel(args):
    """Stop a batch that has stalled. The completed responses stay available: a cancelled
    batch keeps its output file, so `collect` still picks up everything that finished."""
    b = client().batches.cancel(json.load(open(state_path(args.model)))["batch_id"])
    print(f"{b.id}  {b.status}  {b.request_counts}")


def cmd_fill(args):
    """Run whatever is still missing through the synchronous API.

    A batch occasionally leaves a handful of requests unfinished for the whole 24h
    window. There are too few of them to be worth another batch, and they are the
    difference between a complete column and a column of dashes.
    """
    jobs = build()
    done, seen = set(), set()
    for cid, _, _ in jobs:
        p = out_path(cid, args.model)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            done.add((p, r["id"]))
    todo = [j for j in jobs if (out_path(j[0], args.model), j[0].split("|")[3]) not in done]
    print(f"{len(todo)} missing")
    if not todo:
        return
    cli = client()
    body = BODY.get(args.model, DEFAULT_BODY)
    rows, bad = {}, 0
    for i, (cid, prompt, d) in enumerate(todo, 1):
        r = cli.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}], **body)
        out = (r.choices[0].message.content or "").split("Answer: ")[-1]
        try:
            resp = match_api_str(out)
        except Exception:
            resp = None
        if resp is None:
            bad += 1
            continue
        rows.setdefault(out_path(cid, args.model), []).append(
            [d["id"], d["question"], d["choices"], d["choice_keys"], d["answer"],
             d["type"], resp])
        print(f"  {i}/{len(todo)} {cid}", flush=True)
    for path, items in rows.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerows(items)
    print(f"wrote {sum(len(v) for v in rows.values())} rows, {bad} unparseable")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("count", cmd_count), ("submit", cmd_submit),
                     ("status", cmd_status), ("collect", cmd_collect),
                     ("cancel", cmd_cancel), ("fill", cmd_fill)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name != "count":
            s.add_argument("--model", required=True)
        if name == "submit":
            s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
