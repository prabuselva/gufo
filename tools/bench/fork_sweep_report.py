#!/usr/bin/env python3
"""Parse fork_sweep cell outputs into per-model result dicts.

Usage: fork_sweep_report.py <key> [<key> ...]
Reads ${GUFO_FORK_SWEEP_DIR:-/tmp/opencode/sweep}/<key>/*.out and prints one
JSON object per model: loads (list of seconds), cells {test: t/s} (last
occurrence wins per file key), spec {test: t/s}.
"""
import json
import os
import re
import sys

PP_ORDER = ["pp512", "pp2048", "pp4096", "pp8192", "pp16384", "pp32768",
            "pp65536", "pp100000"]
LOAD_RE = re.compile(r"\[Model Load\]:\s*([\d.]+)\s*s")
ROW_RE = re.compile(
    r"^\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)")


def parse_file(path: str):
    load = None
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOAD_RE.search(line)
            if m:
                load = float(m.group(1))
            r = ROW_RE.match(line)
            if r:
                rows[r.group(1)] = float(r.group(2))
    return load, rows


def main(keys):
    out = {}
    base = os.environ.get("GUFO_FORK_SWEEP_DIR", "/tmp/opencode/sweep")
    for key in keys:
        d = os.path.join(base, key)
        loads, cells, spec = [], {}, {}
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not name.endswith(".out"):
                continue
            load, rows = parse_file(os.path.join(d, name))
            if load is not None and not name.startswith("spec-"):
                loads.append(load)
            if name.startswith("spec-"):
                spec.update({f"{k}_mtp": v for k, v in rows.items()})
            else:
                cells.update(rows)
        out[key] = {"loads": loads, "cells": cells, "spec": spec}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
