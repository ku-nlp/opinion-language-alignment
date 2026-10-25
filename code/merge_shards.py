"""Merge per-question shards of a hidden-state dump.

Questions are independent, so two machines can each take a disjoint range of them into
its own --out-dir. Each shard file has the full [Q, K, L, d] shape with only its own rows
written; merging copies the rows each shard's mask marks as done.

  python code/merge_shards.py --model qwen2.5-72b --shard <dir> [--shard <dir> ...]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--shard", action="append", required=True,
                help="directory holding hidden.npy/mask.npy for one shard")
ap.add_argument("--out", default=None, help="target dir (default: the first shard)")
a = ap.parse_args()

target = a.out or a.shard[0]
def load_mask(d, shape_q):
    """A run started before incremental mask saving leaves no mask.npy; a question that
    was never written is all zeros, so the mask can be recovered from the data."""
    p = f"{d}/mask.npy"
    if os.path.exists(p):
        return np.load(p)
    print(f"  {d}: no mask.npy, reconstructing from non-zero rows")
    a = np.load(f"{d}/hidden.npy", mmap_mode="r")
    m = np.zeros((a.shape[0], a.shape[1]), dtype=bool)
    for q in range(a.shape[0]):
        m[q] = np.any(a[q, :, 0, :] != 0, axis=-1)
    return m


tgt_mask = load_mask(target, None)
arr = np.load(f"{target}/hidden.npy", mmap_mode="r+")
meta = json.load(open(f"{target}/meta.json"))
added = 0
for d in a.shard:
    if os.path.realpath(d) == os.path.realpath(target):
        continue
    m = load_mask(d, None)
    src = np.load(f"{d}/hidden.npy", mmap_mode="r")
    assert src.shape == arr.shape, f"{d}: shape {src.shape} != {arr.shape}"
    assert json.load(open(f"{d}/meta.json"))["conditions"] == meta["conditions"], \
        f"{d}: different condition list"
    for q in np.where(m.any(axis=1) & ~tgt_mask.any(axis=1))[0]:
        arr[q] = src[q]
        tgt_mask[q] = m[q]
        added += 1
    arr.flush()
np.save(f"{target}/mask.npy", tgt_mask)
n = int(tgt_mask.any(axis=1).sum())
print(f"merged {added} questions into {target}: {n}/{tgt_mask.shape[0]} present")
