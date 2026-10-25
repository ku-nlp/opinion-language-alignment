"""Run the two anonymous few-shot conditions on an open-weight model with vLLM.

The conference runner (global_opinion_alignment/code/RQ2/prob_by_llm.py) generates one
prompt at a time with transformers and retries in a `while` loop until the output parses,
which for 2864 prompts is hours of mostly idle GPU. This does the same job with vLLM:
same greedy decoding, same output schema, same retry-with-sampling escape hatch as
run_llm.py, so the new rows are produced the way the journal version's other open-weight
rows are.

Results are written into the conference repository, next to the existing models:

    results/RQ2/<method>/<country>/<repo>/results.csv

where <repo> is the HuggingFace name the conference results are already filed under
("CohereLabs/aya-23-35B"), not our short key.

  python code/run_anon_llm.py --model qwen2.5-72b --tp 2
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                                                   # noqa: E402
from run_anon_batch import FIELDS, build, out_path                   # noqa: E402

sys.path.insert(0, C.AAAI_REPO)
from utils.common import match_llm_str as _match_llm_str             # noqa: E402

SYSTEM = "You are a helpful assistant."


def match_llm_str(text):
    """match_llm_str, but a malformed answer is a failure rather than a crash.

    Its pattern accepts an unbalanced quote (`'?\\d+'?`), so an answer like
    `{1': '2.36%', '2': '97.64%'}` matches and then raises SyntaxError out of
    ast.literal_eval. Qwen2.5-1.5B writes that often enough to have killed a whole run,
    taking the vLLM engine down with it. The conference function is left untouched --
    the published numbers were produced by it -- and only the call is guarded here.
    """
    try:
        return _match_llm_str(text)
    except Exception:
        return None


def chat_wrap(tokenizer, text):
    """Prefer system+user, as the conference code does; fall back if the template forbids."""
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
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: run only N prompts")
    ap.add_argument("--countries", default=None,
                    help="comma-separated subset, e.g. China,Germany")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    repo = C.MODELS[args.model]
    jobs = build(args.countries.split(",") if args.countries else None)
    done, seen = set(), set()
    for cid, _, _ in jobs:
        p = out_path(cid, repo)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            done.add((p, r["id"]))
    todo = [j for j in jobs if (out_path(j[0], repo), j[0].split("|")[3]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(jobs)} prompts, {len(done)} already present, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(repo)
    llm = LLM(model=repo, tensor_parallel_size=args.tp, dtype="float16",
              gpu_memory_utilization=args.gpu_frac, max_model_len=args.max_model_len,
              trust_remote_code=True)

    texts = [chat_wrap(tok, p) for _, p, _ in todo]
    pending = list(range(len(todo)))
    parsed = [None] * len(todo)

    for attempt in range(args.max_retries + 1):
        if not pending:
            break
        sp = (SamplingParams(temperature=0.0, max_tokens=args.max_tokens) if attempt == 0
              else SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens,
                                  seed=attempt))
        print(f"\n-- pass {attempt} ({'greedy' if attempt == 0 else 'sampled'}): "
              f"{len(pending)} prompts", flush=True)
        outs = llm.generate([texts[i] for i in pending], sp)
        still = []
        for i, o in zip(pending, outs):
            p = match_llm_str(o.outputs[0].text.strip())
            if p is None:
                still.append(i)
            else:
                parsed[i] = p
        print(f"   parsed {len(pending) - len(still)}, failed {len(still)}", flush=True)
        pending = still

    rows = {}
    for i, (cid, _, d) in enumerate(todo):
        if parsed[i] is None:
            continue
        rows.setdefault(out_path(cid, repo), []).append(
            [d["id"], d["question"], d["choices"], d["choice_keys"], d["answer"],
             d["type"], parsed[i]])
    for path, items in rows.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerows(items)
    print(f"\nwrote {sum(len(v) for v in rows.values())} rows to {len(rows)} files; "
          f"unparseable after {args.max_retries} retries: {len(pending)}")


if __name__ == "__main__":
    main()
