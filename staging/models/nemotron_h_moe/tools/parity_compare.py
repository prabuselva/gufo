#!/usr/bin/env python3
"""Compare per-step logits dumps between the Gufo CPU oracle and the pinned
llama.cpp probe (raw little-endian f32, one vocabulary block per step).

Usage: parity_compare.py ORACLE.F32 LLAMA.F32 [--vocab N]
Prints per-step argmax match, max abs diff, max relative diff, cosine
similarity and top-10 overlap. The relative gate applies over the union of
each side's top --top-k (default 100) logits, i.e. the sampling region; a
deep CPU oracle accumulates quantized dot products in a different order than
llama.cpp's integer vec_dot kernels, so tail logits (rank > top-k, small
magnitudes) drift beyond 5% only from summation noise. Exit 0 iff every
step's argmax matches, the top-k relative diff is at most --max-rel
(default 0.05) and cosine similarity is at least --min-cos (default 0.99).
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("llama")
    ap.add_argument("--vocab", type=int, default=131072)
    ap.add_argument("--max-rel", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()

    a = np.fromfile(args.oracle, dtype="<f4")
    b = np.fromfile(args.llama, dtype="<f4")
    if a.shape != b.shape or a.size % args.vocab != 0:
        print(f"shape mismatch: {a.shape} vs {b.shape} (vocab {args.vocab})")
        return 1
    a = a.reshape(-1, args.vocab)
    b = b.reshape(-1, args.vocab)

    ok = True
    for step in range(a.shape[0]):
        va, vb = a[step], b[step]
        amax_a = int(np.argmax(va))
        amax_b = int(np.argmax(vb))
        diff = np.abs(va - vb)
        denom = np.maximum(np.abs(va), 1.0)
        max_abs = float(diff.max())
        max_rel_global = float((diff / denom).max())
        top_union = np.union1d(
            np.argsort(va)[-args.top_k :], np.argsort(vb)[-args.top_k :]
        )
        max_rel = float((diff[top_union] / denom[top_union]).max())
        cos = float(
            np.dot(va, vb)
            / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-30)
        )
        top_a = set(np.argsort(va)[-10:].tolist())
        top_b = set(np.argsort(vb)[-10:].tolist())
        argmax_ok = amax_a == amax_b
        rel_ok = max_rel <= args.max_rel
        cos_ok = cos >= args.min_cos
        ok = ok and argmax_ok and rel_ok and cos_ok
        print(
            f"step {step}: argmax {amax_a} vs {amax_b} "
            f"{'MATCH' if argmax_ok else 'MISMATCH'} | "
            f"max_abs {max_abs:.6g} | max_rel(top{args.top_k}) {max_rel:.4g} "
            f"{'ok' if rel_ok else 'FAIL'} | max_rel(all) "
            f"{max_rel_global:.4g} | cos {cos:.6f} "
            f"{'ok' if cos_ok else 'FAIL'} | "
            f"top10 overlap {len(top_a & top_b)}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
