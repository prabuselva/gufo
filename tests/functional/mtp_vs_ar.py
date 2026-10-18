#!/usr/bin/env python3
"""Record whether speculative MTP output is width-invariant against AR.

Greedy decode must emit identical token sequences with and without the MTP
draft path. Each depth runs `gufo bench -v` twice (AR and MTP) and extracts
the per-run `output_sha256=` traces from the tg lines. Diverging hashes are
recorded in the artifact; `--gate` additionally fails the exit code so the
fix (fork width-invariant wmma_gemv verification) can flip the suite green.

The benchmark needs exclusive GPU access; launch this through
tools/bench/gpu_exclusive.sh so the resident server is stopped first.

Example:

  tools/bench/gpu_exclusive.sh -- python3 tests/functional/mtp_vs_ar.py \
      --binary build/release/gufo \
      --model /path/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf \
      --mtp-model /path/mtp-gemma-4-26B-A4B-it-Q8_0.gguf \
      --output /tmp/mtp-vs-ar
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

SHA_RE = re.compile(r"output_sha256=([0-9a-f]{64})")
DECODED_RE = re.compile(r"drafted=(\d+) accepted=(\d+)")
TEXT_RE = re.compile(r" text=(.*)$")


def run_bench(binary, model, depth, n_gen, mtp_model=None, draft_tokens=None):
    command = [binary, "bench", "--model", model, "--n-depth", str(depth),
               "--n-gen", str(n_gen), "--repetitions", "1", "-v"]
    if mtp_model is None:
        command += ["--speculative", "off"]
    else:
        command += ["--speculative", "mtp", "--mtp-model", mtp_model]
        if draft_tokens:
            command += ["--draft-tokens", str(draft_tokens),
                        "--min-draft-tokens", "1"]
    started = subprocess.run(command, capture_output=True, text=True)
    combined = started.stdout + started.stderr
    sha256s = SHA_RE.findall(combined)
    drafted = DECODED_RE.findall(combined)
    texts = TEXT_RE.findall(combined)
    if started.returncode != 0 or not sha256s:
        raise RuntimeError(
            f"bench failed rc={started.returncode}: {' '.join(command)}\n"
            f"{combined[-2000:]}")
    record = {"command": command, "rc": started.returncode,
              "output_sha256": sha256s, "deterministic": len(set(sha256s)) == 1}
    if drafted:
        record["drafted"], record["accepted"] = (
            int(drafted[0][0]), int(drafted[0][1]))
    if texts:
        record["text_prefix"] = texts[0][:160]
    # Only decidable when the decoded snippet is present; an empty
    # snippet is unknown, not degenerate.
    record["degenerate"] = (len(set(texts[0].split())) < 8) if texts else None
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--depths", default="0,8192,16384")
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--n-gen", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero on any hash mismatch")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    width_invariant = True
    for depth in [int(value) for value in args.depths.split(",") if value]:
        ar = run_bench(args.binary, args.model, depth, args.n_gen)
        mtp = run_bench(args.binary, args.model, depth, args.n_gen,
                        args.mtp_model, args.draft_tokens)
        match = ar["output_sha256"] == mtp["output_sha256"]
        width_invariant = width_invariant and match
        rows.append({"depth": depth, "autoregressive": ar, "mtp": mtp,
                     "hash_match": match})
        print(f"depth={depth} hash_match={match} "
              f"ar={ar['output_sha256'][0][:16]} mtp={mtp['output_sha256'][0][:16]}")

    result = {"model": args.model, "mtp_model": args.mtp_model,
              "draft_tokens": args.draft_tokens, "n_gen": args.n_gen,
              "width_invariant": width_invariant, "rows": rows}
    report = args.output / "mtp_vs_ar.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"width_invariant={width_invariant} artifact={report}")
    if args.gate and not width_invariant:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
