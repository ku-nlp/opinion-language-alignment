"""L1 distance between the directions the three steering channels write.

No PCA and no cosine: each channel is compared as the vector it actually writes.

    v_lang(l) = h[lang_l]    - h[base]
    v_per(c)  = h[persona_c] - h[base]
    v_few(c)  = h[fewshot_c] - h[base]

Every vector is first scaled to unit L2 norm. Without that the comparison would be a
ranking of the channels by how hard they push rather than by where: at the last layer of
Llama3 the three displacements have norms of about 67, 13 and 3, so a raw L1 distance
between them is 95% a statement about the few-shot channel being small. After the
scaling, what is left is direction, and the L1 distance between two directions is

    d(a, b) = || a/||a||_2 - b/||b||_2 ||_1 .

Each language is paired with the homeland of that language. The reference is the same
distance computed against the other 65 countries of the same channel, that is, the
distance to a country the language has nothing to do with; it says what an unrelated pair
looks like at that layer, which a raw L1 distance in 8192 dimensions does not.

Every layer is computed, and the per-layer channel means are cached so that a later
contrast does not have to read the 68 GB of dumps again.

  python code/l1dist.py               -> results/l1dist.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

MODELS = ["aya-23-35b", "llama3-70b", "qwen2.5-72b"]
CHANNELS = [("lang", "lang_"), ("persona", "persona_"), ("fewshot", "fewshot_")]
CACHE = f"{C.RESULT_DIR}/channel_means"


def unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def base_norm(model):
    """Mean L2 norm of the unsteered hidden state at each layer, cached.

    A per-layer divisor has to come from outside the quantities being plotted, or the
    figure normalises itself; the size of the residual stream is that outside scale.
    Only the base condition is read, which is one condition out of 147."""
    path = f"{CACHE}/{model}_basenorm.npy"
    if os.path.exists(path):
        return np.load(path)
    d = f"{C.HIDDEN_DIR}/{model}"
    meta = json.load(open(f"{d}/meta.json"))
    conds = meta["conditions"]
    n_q, n_l, w = len(meta["qids"]), meta["n_layers"], meta["width"]
    h = np.memmap(f"{d}/hidden.npy", dtype=np.float16, mode="r",
                  shape=(n_q, len(conds), n_l, w))
    b = np.asarray(h[:, conds.index("base"), :, :], dtype=np.float32)
    v = np.linalg.norm(b, axis=2).mean(0)          # [n_layers]
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, v)
    print(f"  {model}: base norms cached", flush=True)
    return v


def channel_means(model):
    """-> {channel: (names, [n_cond, n_layers, w] float32)}, cached on disk."""
    path = f"{CACHE}/{model}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {n: (list(z[f"{n}_names"]), z[n].astype(np.float32))
                for n, _ in CHANNELS}

    d = f"{C.HIDDEN_DIR}/{model}"
    meta = json.load(open(f"{d}/meta.json"))
    conds = meta["conditions"]
    idx = {c: i for i, c in enumerate(conds)}
    n_q, n_l, w = len(meta["qids"]), meta["n_layers"], meta["width"]
    h = np.memmap(f"{d}/hidden.npy", dtype=np.float16, mode="r",
                  shape=(n_q, len(conds), n_l, w))
    mask = np.load(f"{d}/mask.npy")
    base = np.asarray(h[:, idx["base"], :, :], dtype=np.float32)

    out, store = {}, {}
    for name, pref in CHANNELS:
        names = [c for c in conds if c.startswith(pref)]
        ok = np.all(mask[:, [idx[c] for c in names]], axis=1) & mask[:, idx["base"]]
        sel = np.where(ok)[0]
        acc = np.empty((len(names), n_l, w), dtype=np.float32)
        for i, c in enumerate(names):
            acc[i] = (np.asarray(h[sel, idx[c], :, :], dtype=np.float32)
                      - base[sel]).mean(0)
        out[name] = (names, acc)
        store[name] = acc.astype(np.float16)
        store[f"{name}_names"] = np.array(names)
        print(f"  {model}: {name} read ({len(names)} directions, {len(sel)} questions)",
              flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **store)
    print(f"  cached -> {path}", flush=True)
    return out


def run(model):
    ch = channel_means(model)
    lang_names, L = ch["lang"]
    per_names, P = ch["persona"]
    few_names, F = ch["fewshot"]
    per_code = [int(c.split("_")[1]) for c in per_names]
    few_code = [int(c.split("_")[1]) for c in few_names]
    n_l = L.shape[1]

    pairs = [(c[5:], C.LANG_HOME[c[5:]]) for c in lang_names
             if c[5:] in C.LANG_HOME and C.LANG_HOME[c[5:]] in per_code]

    # For unit vectors ||a-b||_2^2 = 2 - 2 cos(a,b), so the L2 curve is the cosine curve
    # in other units and orders the pairs identically. L1 is not a monotone function of
    # the cosine, so it is the one that can disagree; both are emitted to make that
    # visible rather than asserted.
    METRICS = {"l1": lambda D: np.abs(D).sum(-1),
               "l2": lambda D: np.sqrt((D ** 2).sum(-1))}

    # "raw" leaves the displacements as they are, so the distance carries how far each
    # cue pushes as well as where it pushes; "unit" scales every vector to length one
    # first and therefore compares direction only.
    VARIANTS = ["raw", "unit"]

    # One constant for the whole model: the mean length of a displacement over every
    # layer and every condition. Dividing by it puts the three LLMs on a common scale,
    # whose residual streams differ by an order of magnitude, while leaving the shape
    # across depth exactly as it is. A per-layer divisor would instead flatten that shape,
    # which is why it is reported separately as l2_scaled.
    gscale = float(np.mean([np.linalg.norm(X, axis=2).mean() for X in (L, P, F)]))
    bnorm = base_norm(model)
    print(f"  {model}: global scale {gscale:.1f}, base norm "
          f"{bnorm[0]:.0f} -> {bnorm[-1]:.0f}", flush=True)

    rows, per_lang = [], []
    for layer in range(n_l):
        vecs = {"raw": (L[:, layer, :], P[:, layer, :], F[:, layer, :]),
                "unit": (unit(L[:, layer, :]), unit(P[:, layer, :]),
                         unit(F[:, layer, :]))}
        # One scalar per layer: the mean length of a displacement there. Residual-stream
        # norms grow by an order of magnitude with depth, which in the raw plot buries
        # every early layer against the last few. Dividing all curves of a layer by the
        # same number removes that growth and changes nothing within a layer.
        scale = float(np.mean([np.linalg.norm(X[:, layer, :], axis=1).mean()
                               for X in (L, P, F)]))
        rows.append(dict(model=model, layer=layer, n_layers=n_l, metric="scale",
                         pair="mean displacement norm", d1=scale, sd=0.0, n=3))
        for vname in VARIANTS:
            Lu, Pu, Fu = vecs[vname]
            for mname, f in METRICS.items():
                key = mname if vname == "unit" else f"{mname}_raw"
                acc = {k: [] for k in ("lang-persona", "lang-fewshot", "persona-fewshot",
                                       "lang-persona (other)", "lang-fewshot (other)")}
                for l, home in pairs:
                    i = lang_names.index(f"lang_{l}")
                    jp, jf = per_code.index(home), few_code.index(home)
                    acc["lang-persona"].append(f(Lu[i] - Pu[jp]))
                    acc["lang-fewshot"].append(f(Lu[i] - Fu[jf]))
                    acc["persona-fewshot"].append(f(Pu[jp] - Fu[jf]))
                    # the same distance to every country that is not the homeland
                    oth = np.ones(len(per_code), bool)
                    oth[jp] = False
                    acc["lang-persona (other)"].append(f(Lu[i] - Pu[oth]).mean())
                    oth = np.ones(len(few_code), bool)
                    oth[jf] = False
                    acc["lang-fewshot (other)"].append(f(Lu[i] - Fu[oth]).mean())
                # the same distances before the average over languages, so a figure can
                # show whether the ordering of the three curves holds language by
                # language or only in the mean
                if key == "l2_raw":
                    for j, (l, _) in enumerate(pairs):
                        for k in ("lang-persona", "lang-fewshot", "persona-fewshot"):
                            per_lang.append(dict(model=model, layer=layer, n_layers=n_l,
                                                 lang=l, pair=k, d1=float(acc[k][j])))
                for k, v in acc.items():
                    rows.append(dict(model=model, layer=layer, n_layers=n_l, metric=key,
                                     pair=k, d1=float(np.mean(v)), sd=float(np.std(v)),
                                     n=len(v)))
                    if key == "l2_raw":
                        rows.append(dict(model=model, layer=layer, n_layers=n_l,
                                         metric="l2_scaled", pair=k,
                                         d1=float(np.mean(v)) / scale,
                                         sd=float(np.std(v)) / scale, n=len(v)))
                        rows.append(dict(model=model, layer=layer, n_layers=n_l,
                                         metric="l2_global", pair=k,
                                         d1=float(np.mean(v)) / gscale,
                                         sd=float(np.std(v)) / gscale, n=len(v)))
                        rows.append(dict(model=model, layer=layer, n_layers=n_l,
                                         metric="l2_relbase", pair=k,
                                         d1=float(np.mean(v)) / bnorm[layer],
                                         sd=float(np.std(v)) / bnorm[layer], n=len(v)))
    print(f"  {model}: {n_l} layers done", flush=True)
    return rows, per_lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    rows, per_lang_all = [], []
    for m in ([args.model] if args.model else MODELS):
        if not (os.path.exists(f"{C.HIDDEN_DIR}/{m}/hidden.npy")
                or os.path.exists(f"{CACHE}/{m}.npz")):
            print(f"  ! no dump for {m}, skipped")
            continue
        r, pl = run(m)
        rows += r
        per_lang_all += pl
    d = pd.DataFrame(rows)
    out = f"{C.RESULT_DIR}/l1dist.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(d)} rows)")
    pl = pd.DataFrame(per_lang_all)
    out2 = f"{C.RESULT_DIR}/l1dist_by_lang.csv"
    pl.to_csv(out2, index=False)
    print(f"wrote {out2}  ({len(pl)} rows, {pl.lang.nunique()} languages)")

    for mname in ("l2_relbase", "l2_scaled", "l2_raw", "l1", "l2"):
        print(f"\nlast layer, mean {mname.upper()} distance over the 12 pairs:")
        for m in d.model.unique():
            g = d[(d.model == m) & (d.metric == mname)
                  & (d.layer == d[d.model == m].layer.max())]
            print(f"  {m:13s} " + "  ".join(
                f"{k}={g[g.pair == k].d1.iloc[0]:7.3f}" for k in
                ("lang-persona", "lang-fewshot", "persona-fewshot",
                 "lang-persona (other)", "lang-fewshot (other)")))


if __name__ == "__main__":
    main()
