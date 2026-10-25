"""Run the open-weight models with vLLM.

Same job list and same output schema as run_api.py, so the two paths are
interchangeable downstream. Decoding matches the AAAI setup: greedy, 256 new tokens,
fp16. Unparseable outputs are retried with sampling and the retry counts are printed, so
any deviation from pure greedy is visible rather than silent.

Duplicate prompts (Singapore's English arm makes `language` identical to `base`, etc.)
are generated once and fanned out, exactly as in the batch runner.

  python code/run_llm.py --model qwen2.5-72b --tp 2
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from jobs import build_jobs
from parse import match_llm_str
from run_api import FIELDS, dedupe, done_keys, paths

SYSTEM = "You are a helpful assistant."


def chat_wrap(tokenizer, text):
    """Prefer system+user, as the AAAI code does; fall back if the template forbids it."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: run only N prompts")
    ap.add_argument("--redo-unparsed", action="store_true",
                    help="drop rows that produced no distribution and run them again")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    repo = C.MODELS[args.model]
    _, raw_path, _, _ = paths(args.model)

    if args.redo_unparsed and os.path.exists(raw_path):
        # most unparsed rows are 256-token truncations: the model restates the question
        # in the target language and runs out of budget. Re-running them with a larger
        # limit is the fix; editing the stored text would not be.
        with open(raw_path, newline="", encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f) if r["parsed"] or match_llm_str(r["raw"])]
        with open(raw_path, newline="", encoding="utf-8") as f:
            n_before = sum(1 for _ in csv.DictReader(f))   # rows, not lines: raw text
                                                           # contains embedded newlines
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(keep)
        print(f"dropped {n_before - len(keep)} unparsed rows for a retry")

    jobs = build_jobs()
    done = done_keys(raw_path)
    todo = [j for j in jobs if (j["cond"], j["unit"], j["qid"]) not in done]
    groups = dedupe(todo)
    keys = list(groups)
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(jobs)} prompts, {len(done)} already done, {len(todo)} to run, "
          f"{len(keys)} unique after dedupe")
    if not keys:
        print("nothing to do")
        return

    tok = AutoTokenizer.from_pretrained(repo)
    llm = LLM(model=repo, tensor_parallel_size=args.tp, dtype="float16",
              gpu_memory_utilization=args.gpu_frac, max_model_len=4096,
              trust_remote_code=True)

    texts = [chat_wrap(tok, groups[k][0]) for k in keys]
    pending = list(range(len(keys)))
    parsed = [None] * len(keys)
    raw = [None] * len(keys)

    for attempt in range(args.max_retries + 1):
        if not pending:
            break
        sp = (SamplingParams(temperature=0.0, max_tokens=args.max_tokens) if attempt == 0
              else SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens,
                                  seed=attempt))
        print(f"\n-- pass {attempt} ({'greedy' if attempt == 0 else 'sampled'}): "
              f"{len(pending)} prompts")
        outs = llm.generate([texts[i] for i in pending], sp)
        still = []
        for i, o in zip(pending, outs):
            raw[i] = o.outputs[0].text.strip()
            p = match_llm_str(raw[i])
            if p is None:
                still.append(i)
            else:
                parsed[i] = p
        print(f"   parsed {len(pending) - len(still)}, failed {len(still)}")
        pending = still

    new = not os.path.exists(raw_path)
    fanned = 0
    with open(raw_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        for i, k in enumerate(keys):
            for j in groups[k][1]:
                w.writerow([j["cond"], j["unit"], j["qid"], j["choice_keys"],
                            parsed[i] or "", raw[i]])
            fanned += len(groups[k][1]) - 1

    print(f"\nwrote {raw_path} (+{fanned} fanned-out duplicates); "
          f"unparseable after {args.max_retries} retries: {len(pending)}")


if __name__ == "__main__":
    main()
