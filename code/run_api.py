"""Run the closed-weight models through the OpenAI Batch API.

Batch costs half of the synchronous API and is allowed up to 24h, which is free for us:
the open-weight models are running on the GPUs meanwhile. One batch holds the whole
7966-prompt job list comfortably (the limit is 50k requests / 200MB).

  python code/run_api.py submit  --model gpt-4.1          # -> batch id, returns at once
  python code/run_api.py status  --model gpt-4.1
  python code/run_api.py collect --model gpt-4.1          # -> results/<model>/raw.csv

State lives in results/<model>/batch.json, so submit/collect can be days apart. `submit`
skips prompts already present in raw.csv, so a partially failed batch is resumed rather
than repeated.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from jobs import build_jobs
from parse import match_llm_str

SYSTEM = "You are a helpful assistant."
KEY_FILE = os.path.expanduser("~/.openai_key")
FIELDS = ["cond", "unit", "qid", "choice_keys", "parsed", "raw"]


def client():
    from openai import OpenAI
    return OpenAI(api_key=open(KEY_FILE).read().strip())


def paths(model):
    d = f"{C.RESULT_DIR}/{model}"
    os.makedirs(d, exist_ok=True)
    return d, f"{d}/raw.csv", f"{d}/batch.json", f"{d}/requests.jsonl"


def done_keys(raw_path):
    if not os.path.exists(raw_path):
        return set()
    with open(raw_path, newline="", encoding="utf-8") as f:
        return {(r["cond"], r["unit"], r["qid"]) for r in csv.DictReader(f)}


def custom_id(j):
    return f"{j['cond']}|{j['unit']}|{j['qid']}"


def dedupe(jobs):
    """-> {custom_id of a representative job: (prompt, [jobs sharing that prompt])}

    Singapore's English arm makes `language` identical to `base`, `language_persona`
    identical to `persona`, and so on: 508 of 7966 prompts are literal duplicates.
    Greedy decoding would return the same text for each, so send one and fan it out.
    """
    by = {}
    for j in jobs:
        by.setdefault(j["prompt"], []).append(j)
    return {custom_id(v[0]): (k, v) for k, v in by.items()}


def submit(args):
    _, raw_path, state_path, req_path = paths(args.model)
    if getattr(args, "redo_unparsed", False) and os.path.exists(raw_path):
        # unparseable rows sit in raw.csv with an empty `parsed`, so `done_keys` would
        # skip them. Drop them and let this submission ask for those prompts again.
        with open(raw_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if r["parsed"] or match_llm_str(r["raw"])]
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(keep)
        print(f"dropped {len(rows) - len(keep)} unparseable rows for a retry")
    jobs = build_jobs()
    done = done_keys(raw_path)
    todo = [j for j in jobs if (j["cond"], j["unit"], j["qid"]) not in done]
    print(f"{len(jobs)} prompts, {len(done)} already collected, {len(todo)} to submit")
    if not todo:
        print("nothing to do")
        return

    groups = dedupe(todo)
    print(f"{len(groups)} unique prompts after dedupe "
          f"({len(todo) - len(groups)} duplicates fanned out at collect time)")
    with open(req_path, "w", encoding="utf-8") as f:
        for cid, (prompt, _) in groups.items():
            f.write(json.dumps({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": args.model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}],
                         "max_completion_tokens": args.max_tokens,
                         **({"temperature": args.temperature}
                            if args.temperature is not None else {}),
                         # reasoning models bill the chain of thought as output; the task
                         # asks for one dictionary of percentages, so there is nothing to
                         # think through and the tokens would be pure cost
                         **({"reasoning_effort": args.effort} if args.effort else {})},
            }, ensure_ascii=False) + "\n")
    size = os.path.getsize(req_path) / 1e6
    print(f"wrote {req_path} ({size:.1f} MB)")
    if args.dry_run:
        print("--dry-run: not submitting")
        return

    cli = client()
    up = cli.files.create(file=open(req_path, "rb"), purpose="batch")
    batch = cli.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                               completion_window="24h",
                               metadata={"model": args.model, "project": "jnlp_wvs"})
    json.dump({"batch_id": batch.id, "model": args.model, "n": len(todo),
               "input_file_id": up.id}, open(state_path, "w"), indent=1)
    print(f"submitted batch {batch.id} ({len(todo)} requests)\nstate -> {state_path}")


def _state(model):
    _, _, state_path, _ = paths(model)
    if not os.path.exists(state_path):
        sys.exit(f"no batch on record for {model}; run submit first")
    return json.load(open(state_path))


def status(args):
    b = client().batches.retrieve(_state(args.model)["batch_id"])
    c = b.request_counts
    print(f"{b.id}  status={b.status}  "
          f"completed={c.completed}/{c.total} failed={c.failed}")
    if b.errors and getattr(b.errors, "data", None):
        for e in b.errors.data[:5]:
            print(f"  error: {e.message}")


def collect(args):
    st = _state(args.model)
    cli = client()
    b = cli.batches.retrieve(st["batch_id"])
    if b.status != "completed":
        print(f"batch is {b.status}, not completed"
              + (f" ({b.request_counts.completed}/{b.request_counts.total})"
                 if b.request_counts else ""))
        if not args.wait:
            return
        while b.status in ("validating", "in_progress", "finalizing"):
            time.sleep(args.poll)
            b = cli.batches.retrieve(st["batch_id"])
            print(f"  {b.status} {b.request_counts.completed}/{b.request_counts.total}",
                  flush=True)
        if b.status != "completed":
            sys.exit(f"batch ended as {b.status}")

    _, raw_path, _, _ = paths(args.model)
    groups = dedupe(build_jobs())
    # A submission only contains the prompts that were outstanding at the time, so the
    # custom_id it used is the first *undone* member of a prompt group, which need not be
    # the representative dedupe picks here from the full job list. Index every member, or
    # a response that was paid for is dropped by the `g is None` branch below.
    by_id = {custom_id(j): g for g in groups.values() for j in g[1]}
    done = done_keys(raw_path)
    text = cli.files.content(b.output_file_id).text
    rows, unparsed, fanned = [], 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = by_id.get(r["custom_id"])
        if g is None:
            continue
        body = (r.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        out = choices[0]["message"]["content"].strip() if choices else ""
        p = match_llm_str(out)
        unparsed += p is None
        # skip members already in raw.csv, so re-running collect cannot duplicate rows
        fresh = [j for j in g[1] if (j["cond"], j["unit"], j["qid"]) not in done]
        for j in fresh:
            rows.append([j["cond"], j["unit"], j["qid"], j["choice_keys"], p or "", out])
        fanned += max(len(fresh) - 1, 0)
    print(f"fanned {fanned} duplicate prompts back out")

    new = not os.path.exists(raw_path)
    with open(raw_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerows(rows)
    print(f"collected {len(rows)} responses, {unparsed} unparseable -> {raw_path}")
    if b.error_file_id:
        err = cli.files.content(b.error_file_id).text
        path = f"{C.RESULT_DIR}/{args.model}/batch_errors.jsonl"
        open(path, "w", encoding="utf-8").write(err)
        print(f"batch reported errors -> {path}; rerun submit to retry those prompts")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("submit", submit), ("status", status), ("collect", collect)]:
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        if name == "submit":
            p.add_argument("--max-tokens", type=int, default=256)
            p.add_argument("--effort", default=None,
                           help="reasoning_effort for reasoning models, e.g. minimal")
            p.add_argument("--temperature", type=float, default=None,
                           help="omitted by default, which is OpenAI's 1.0")
            p.add_argument("--redo-unparsed", action="store_true",
                           help="resubmit only the prompts that produced no distribution")
            p.add_argument("--dry-run", action="store_true")
        if name == "collect":
            p.add_argument("--wait", action="store_true", help="poll until the batch ends")
            p.add_argument("--poll", type=int, default=120)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
