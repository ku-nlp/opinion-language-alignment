"""Extract and store the last-token residual stream for every prompt condition.

The behavioural result to explain: switching the questionnaire to a language moves the
model towards the population of that language's *homeland* more than towards this
country's speakers of it. Everything downstream asks how the language direction
v_lang(L) = h[lang_L] - h[base] relates to the persona directions
v_persona(C) = h[persona_C] - h[base].

Conditions per question:
    base            English questionnaire, random numbers in the five examples
    persona_<C>     base + "a group of people from C"
    fewshot_<C>     base with the example percentages replaced by C's real WV7
                    distribution -- byte-identical to base except for those numbers
    placebo         base + the same insertion naming no population ("the third floor"),
                    the floor for the persona channel: inserting *any* phrase displaces
                    the residual stream
    random2         base with a second independent draw of random numbers, the floor for
                    the few-shot channel: changing the example percentages displaces the
                    stream even when they describe nobody
    lang_<L>        the L questionnaire, no persona

Countries are the WV7 set minus any that lack one of the five example questions, so the
persona and few-shot channels rank an identical candidate list and their ranks are
comparable.

Hidden states are written out in full rather than reduced to a Gram matrix. A Gram
matrix is a sufficient statistic for the cosines, but it sums over questions, which
destroys the per-question resolution needed to bootstrap a confidence interval -- and it
would force a full re-run for any later change of contrast. At fp16 the array is 7-15 GB
per model, so HIDDEN_DIR should point at a large disk.

Layout, under HIDDEN_DIR:
    <HIDDEN_DIR>/<model>/hidden.npy   float16 [n_questions, n_conditions, n_layers, d]
    <HIDDEN_DIR>/<model>/mask.npy     bool    [n_questions, n_conditions]
    <HIDDEN_DIR>/<model>/meta.json    qids, condition names, layer count, width

  python code/activations.py --model qwen2.5-72b
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import prompts as P

SYSTEM = "You are a helpful assistant."
PLACEBO = "people from the third floor"


class LastTokenTap:
    """Capture the last-position residual stream of every decoder block.

    Verified against output_hidden_states: identical for every layer except the last,
    where HF additionally applies the final norm before appending. Taking the raw block
    output keeps all layers on one footing. Moved to CPU inside the hook because
    device_map="auto" spreads layers over several GPUs.
    """

    def __init__(self, layers):
        self.buf = {}
        self.handles = [blk.register_forward_hook(self._make(i))
                        for i, blk in enumerate(layers)]

    def _make(self, idx):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            self.buf[idx] = h[:, -1, :].detach().to(torch.float16).cpu()
        return fn

    def collect(self, n):
        return torch.stack([self.buf[i] for i in range(n)], dim=1)   # [B, L, d]

    def close(self):
        for h in self.handles:
            h.remove()


def candidate_countries():
    """WV7 countries whose few-shot block can be built (all five examples present)."""
    import json as _json
    gt = _json.load(open(f"{C.DATA_DIR}/ground_truth.json"))
    return [c for c in sorted(C.COUNTRY_ALL)
            if all(q in gt.get(str(c), {}) for q in C.FEWSHOT_IDS)]


def conditions():
    """-> (ordered [(name, build(qid) -> prompt or None)], questionnaire items)."""
    import jobs as J
    items = {l: {d["id"]: d for d in P.load_items(l)}
             for l in set(C.LANG_HOME) | {"en"}}
    conds = [("base", lambda q: P.base_prompt(items["en"][q]))]
    for code in candidate_countries():
        conds.append((f"persona_{code}",
                      lambda q, n=C.COUNTRY_ALL[code]:
                      P.persona_prompt(f"people from {n}", items["en"][q])))
    for code in candidate_countries():
        conds.append((f"fewshot_{code}",
                      lambda q, c=code: J.few_shot_prompt("en", str(c), items["en"][q])))
    conds.append(("placebo", lambda q: P.persona_prompt(PLACEBO, items["en"][q])))
    conds.append(("random2",
                  lambda q: J.few_shot_prompt("en", "random2", items["en"][q])))
    for lang in sorted(C.LANG_HOME):
        if lang == "en":
            continue                       # the English questionnaire is `base`
        conds.append((f"lang_{lang}",
                      lambda q, l=lang: (P.language_prompt(l, items[l][q])
                                         if q in items[l] else None)))
    return conds, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-questions", type=int, default=0, help="0 = all")
    ap.add_argument("--out-dir", default=C.HIDDEN_DIR)
    ap.add_argument("--resume", action="store_true",
                    help="keep questions already written and continue from there")
    ap.add_argument("--q-start", type=int, default=0)
    ap.add_argument("--q-end", type=int, default=0,
                    help="process questions [q-start, q-end); 0 = to the end. Questions "
                         "are independent, so two machines can take disjoint ranges into "
                         "separate --out-dir trees and be merged afterwards")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    repo = C.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"                      # so position -1 is the real last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # AutoModel, not AutoModelForCausalLM: only hidden states are needed, and running a
    # 256k-vocab LM head over the whole sequence is wasted work and enough extra memory
    # traffic to crash the launch on some cards.
    model = AutoModel.from_pretrained(repo, dtype=torch.float16, device_map="auto")
    model.eval()
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    n_layers, width = len(layers), model.config.hidden_size
    tap = LastTokenTap(layers)

    conds, items = conditions()
    qids = list(items["en"])
    if args.max_questions:
        qids = qids[: args.max_questions]
    names = [n for n, _ in conds]
    Q, K = len(qids), len(conds)
    print(f"{repo}: {n_layers} layers x {width} dims | {K} conditions x {Q} questions "
          f"-> {Q * K * n_layers * width * 2 / 1e9:.1f} GB")
    if args.q_start or args.q_end:
        print(f"  shard: questions [{args.q_start}, {args.q_end or Q})")

    out = f"{args.out_dir}/{args.model}"
    os.makedirs(out, exist_ok=True)
    # resume: the mask is flushed as we go, so a crash costs the current question only
    done_path, mask_path = f"{out}/hidden.npy", f"{out}/mask.npy"
    resume = (args.resume and os.path.exists(done_path) and os.path.exists(mask_path)
              and np.load(mask_path).shape == (Q, K))
    arr = np.lib.format.open_memmap(done_path, mode="r+" if resume else "w+",
                                    dtype=np.float16, shape=(Q, K, n_layers, width))
    mask = np.load(mask_path) if resume else np.zeros((Q, K), dtype=bool)
    if resume:
        print(f"resuming: {int(mask.any(axis=1).sum())}/{Q} questions already stored")
    json.dump(dict(model=args.model, repo=repo, qids=qids, conditions=names,
                   n_layers=n_layers, width=width),
              open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)

    @torch.no_grad()
    def embed(texts):
        outs = []
        for i in range(0, len(texts), args.batch_size):
            batch = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True)
                for t in texts[i: i + args.batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            tap.buf.clear()
            model(**enc)
            outs.append(tap.collect(n_layers))
        return torch.cat(outs, 0)

    lo, hi = args.q_start, (args.q_end or Q)
    for qi, qid in enumerate(qids):
        if not (lo <= qi < hi) or mask[qi].any():
            continue
        built = [(k, fn(qid)) for k, (_, fn) in enumerate(conds)]
        have = [(k, t) for k, t in built if t is not None]
        H = embed([t for _, t in have]).numpy()               # [n_have, n_layers, width]
        for j, (k, _) in enumerate(have):
            arr[qi, k] = H[j]
            mask[qi, k] = True
        if (qi + 1) % 5 == 0 or qi == hi - 1:
            arr.flush()
            np.save(mask_path, mask)
            print(f"  {qi + 1}/{Q} questions", flush=True)

    tap.close()
    arr.flush()
    del arr
    np.save(mask_path, mask)
    print(f"\nwrote {out}/hidden.npy  ({os.path.getsize(f'{out}/hidden.npy') / 1e9:.1f} GB)")
    print(f"      {out}/mask.npy, meta.json")


if __name__ == "__main__":
    main()
