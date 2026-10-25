"""Run the country-blind `language` condition for languages outside the study cells.

\\S6.1 asks which subgroup of one country a questionnaire language points at. A different
question is which of the *countries* that answered in a language an unsteered prompt in
that language lands nearest, and it needs the same condition -- the questionnaire in that
language, no persona, no examples -- for every language more than one country answered in.

jobs.py already produces that condition, but only for the languages of config.DID_CELLS,
because it walks those cells. Spanish is answered by thirteen WV7 countries and by none
of them, so it never gets built. This script builds the same prompt for any language and
appends it to the same results/<model>/raw.csv with cond="language" and unit=<lang>.
Nothing downstream is disturbed: score.py looks conditions up by the languages of
DID_CELLS, so a row for a language that is not in one is inert.

  python code/run_langonly.py count  --langs es
  python code/run_langonly.py submit --model gpt-4.1 --langs es
  python code/run_langonly.py collect --model gpt-4.1 --langs es
  python code/run_langonly.py vllm   --model qwen2.5-72b --langs es --tp 4
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import prompts as P
from parse import match_llm_str
from run_api import FIELDS, client, dedupe, done_keys, paths

SYSTEM = "You are a helpful assistant."


def build(langs):
    """-> the same job dicts jobs.py emits for the `language` condition."""
    en = {d["id"]: d for d in P.load_items("en")}
    jobs = []
    for lang in langs:
        items = {d["id"]: d for d in P.load_items(lang)}
        for qid in sorted(set(items) & set(en)):
            if qid in C.FEWSHOT_IDS:
                continue
            jobs.append(dict(cond="language", unit=lang, qid=qid,
                             choice_keys=json.dumps(items[qid]["choice_keys"]),
                             prompt=P.language_prompt(lang, items[qid])))
    return jobs


def todo_for(model, langs):
    _, raw_path, _, _ = paths(model)
    done = done_keys(raw_path)
    jobs = build(langs)
    return jobs, [j for j in jobs if (j["cond"], j["unit"], j["qid"]) not in done]


def cmd_count(args):
    jobs = build(args.langs)
    by = {}
    for j in jobs:
        by[j["unit"]] = by.get(j["unit"], 0) + 1
    print(f"{len(jobs)} prompts: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))


def cmd_submit(args):
    jobs, todo = todo_for(args.model, args.langs)
    print(f"{len(jobs)} prompts, {len(jobs) - len(todo)} already present, "
          f"{len(todo)} to send")
    if not todo:
        return
    d, _, _, _ = paths(args.model)
    req = f"{d}/langonly_requests.jsonl"
    groups = dedupe(todo)
    body = ({"reasoning_effort": "minimal", "max_completion_tokens": 2048}
            if args.model == "gpt-5" else {"temperature": 0, "max_tokens": 256})
    with open(req, "w", encoding="utf-8") as f:
        for cid, (prompt, _) in groups.items():
            f.write(json.dumps({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": args.model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}],
                         **body}}, ensure_ascii=False) + "\n")
    cli = client()
    up = cli.files.create(file=open(req, "rb"), purpose="batch")
    b = cli.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                           completion_window="24h")
    json.dump(dict(batch_id=b.id, model=args.model, langs=args.langs),
              open(state_path(args.model, args.langs), "w"), indent=1)
    print(f"batch {b.id}: {len(groups)} requests, status {b.status}")


def state_path(model, langs):
    """One file per (model, language set).

    Keying it by model alone loses a batch: submitting Spanish and then Arabic for the
    same model overwrote the first batch id while that batch was still running, and the
    only way back was the id printed at submit time.
    """
    d, _, _, _ = paths(model)
    return f"{d}/langonly_batch_{'-'.join(sorted(langs))}.json"


def _state(model, langs):
    p = state_path(model, langs)
    if os.path.exists(p):
        return json.load(open(p))
    old = f"{paths(model)[0]}/langonly_batch.json"      # written before the rename
    if os.path.exists(old):
        return json.load(open(old))
    sys.exit(f"no batch on record for {model} / {langs}")


def cmd_status(args):
    b = client().batches.retrieve(_state(args.model, args.langs)["batch_id"])
    print(f"{b.id}  {b.status}  {b.request_counts}")


def cmd_collect(args):
    cli = client()
    bid = args.batch_id or _state(args.model, args.langs)["batch_id"]
    b = cli.batches.retrieve(bid)
    if b.status not in ("completed", "cancelled") or not b.output_file_id:
        sys.exit(f"batch is {b.status}")
    _, raw_path, _, _ = paths(args.model)
    _, todo = todo_for(args.model, args.langs)
    members = {cid: g[1] for cid, g in dedupe(todo).items()}
    rows, unparsed = [], 0
    for line in cli.files.content(b.output_file_id).text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = members.get(r["custom_id"])
        if g is None:
            continue
        body = (r.get("response") or {}).get("body") or {}
        ch = body.get("choices") or []
        out = ch[0]["message"]["content"].strip() if ch else ""
        p = match_llm_str(out)
        unparsed += p is None
        for j in g:
            rows.append([j["cond"], j["unit"], j["qid"], j["choice_keys"], p or "", out])
    _append(raw_path, rows)
    print(f"collected {len(rows)} responses, {unparsed} unparseable -> {raw_path}")


def _append(raw_path, rows):
    new = not os.path.exists(raw_path)
    with open(raw_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerows(rows)


def cmd_vllm(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_anon_llm import chat_wrap

    repo = C.MODELS[args.model]
    _, raw_path, _, _ = paths(args.model)
    jobs, todo = todo_for(args.model, args.langs)
    print(f"{len(jobs)} prompts, {len(jobs) - len(todo)} already present, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return
    groups = dedupe(todo)
    keys = list(groups)
    tok = AutoTokenizer.from_pretrained(repo)
    llm = LLM(model=repo, tensor_parallel_size=args.tp, dtype="float16",
              gpu_memory_utilization=0.90, max_model_len=4096, trust_remote_code=True)
    texts = [chat_wrap(tok, groups[k][0]) for k in keys]
    pending, parsed, raw = list(range(len(keys))), [None] * len(keys), [None] * len(keys)
    for attempt in range(args.max_retries + 1):
        if not pending:
            break
        sp = (SamplingParams(temperature=0.0, max_tokens=args.max_tokens) if attempt == 0
              else SamplingParams(temperature=0.7, top_p=0.9, seed=attempt,
                                  max_tokens=args.max_tokens))
        outs = llm.generate([texts[i] for i in pending], sp)
        still = []
        for i, o in zip(pending, outs):
            raw[i] = o.outputs[0].text.strip()
            p = match_llm_str(raw[i])
            if p is None:
                still.append(i)
            else:
                parsed[i] = p
        print(f"  pass {attempt}: parsed {len(pending) - len(still)}, failed {len(still)}",
              flush=True)
        pending = still
    rows = []
    for i, k in enumerate(keys):
        for j in groups[k][1]:
            rows.append([j["cond"], j["unit"], j["qid"], j["choice_keys"],
                         parsed[i] or "", raw[i]])
    _append(raw_path, rows)
    print(f"wrote {len(rows)} rows -> {raw_path}; unparseable: {len(pending)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("count", cmd_count), ("submit", cmd_submit),
                     ("status", cmd_status), ("collect", cmd_collect),
                     ("vllm", cmd_vllm)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--langs", nargs="+", required=True)
        if name != "count":
            s.add_argument("--model", required=True)
        if name == "collect":
            s.add_argument("--batch-id", default=None,
                           help="collect a specific batch, for one whose "
                                "state file was overwritten")
        if name == "vllm":
            s.add_argument("--tp", type=int, default=1)
            s.add_argument("--max-tokens", type=int, default=256)
            s.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
