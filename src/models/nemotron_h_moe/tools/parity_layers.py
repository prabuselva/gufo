#!/usr/bin/env python3
"""Bisect per-layer residual divergence between oracle and llama dumps.

Reads the raw f32 residual traces written by reference_test.cpp
(GUFO_NEMOTRON_LAYERDUMP: num_layers rows of hidden per step) and
nemotron_logits_probe.cpp (GUFO_LLAMA_LAYER_DUMP: same layout).
Reports cosine similarity per (step, layer) and the first diverging layer.
"""

import sys

import numpy as np

COS_OK = 0.99


def load(path, hidden):
    data = np.fromfile(path, dtype=np.float32)
    if data.size % hidden:
        raise ValueError(f"{path}: size {data.size} not divisible by {hidden}")
    rows = data.reshape(-1, hidden)
    if rows.shape[0] % 88:
        raise ValueError(f"{path}: {rows.shape[0]} rows not a multiple of 88")
    return rows.reshape(-1, 88, hidden)


def cos(a, b):
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 and db == 0.0:
        return 1.0
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    oracle = load(sys.argv[1], hidden)
    llama = load(sys.argv[2], hidden)
    if oracle.shape != llama.shape:
        print(f"shape mismatch: {oracle.shape} vs {llama.shape}")
        return 1
    ok = True
    for step in range(oracle.shape[0]):
        first_bad = None
        for layer in range(oracle.shape[1]):
            c = cos(oracle[step, layer].astype(np.float64),
                    llama[step, layer].astype(np.float64))
            if c < COS_OK:
                if first_bad is None:
                    first_bad = layer
                print(
                    f"step {step} layer {layer}: cos {c:.6f} "
                    f"|a| {np.abs(oracle[step, layer]).mean():.4g} "
                    f"|b| {np.abs(llama[step, layer]).mean():.4g}"
                )
        if first_bad is None:
            print(f"step {step}: all {oracle.shape[1]} layer inputs match")
        else:
            ok = False
            print(f"step {step}: first divergence at layer {first_bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
