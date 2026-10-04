# AGENTS.md

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

A resident `gufo serve` holds most VRAM and contends for the GPU, so any GPU
benchmark needs exclusive access. Run it through
`tools/bench/gpu_exclusive.sh [options] -- <command>`: it stops the server,
waits for VRAM to release, runs one command, then respawns the server via the
launch script and polls `/v1/models` until ready before returning (override with
`--launch-script`, `--port`, `--expect-model`, `--ready-timeout`, `--settle`).
`--no-respawn` stops + runs only, leaving the respawn to a foreground harness;
`tools/bench/gpu_respawn_server.sh` is the canonical foreground respawn. Never
time a benchmark while the server is resident; contention and lazy allocation
make such numbers meaningless.

### Running a benchmark from the harness

`gpu_exclusive.sh` blocks for the whole run (stop server -> benchmark -> respawn
-> poll), which can exceed a tool's wall-clock limit. Do NOT run it directly in
one blocking tool call, and do NOT background the respawn: an automated harness
kills a call's whole process tree on timeout, which would take an in-flight
respawn down with it and leave the server down. Split it so the benchmark runs in
the background but the respawn runs in the foreground:

1. Launch the benchmark DETACHED in the background with
   `tools/bench/bench_bg.sh <name> <inner_timeout_sec> <cmd...>`. It runs
   `gpu_exclusive.sh --no-respawn` (stop + run only) under `timeout`, writes
   `${GUFO_BENCH_STATE_DIR:-/tmp/opencode}/gufo_bench_<name>.{log,pid,status}`,
   and returns immediately. Launch it with `setsid ... & disown` so its PID is a
   process group the watchdog can kill as a unit.
2. Run `tools/bench/bench_wait.sh <name> <total_sec> <ready_grace>` in the
   FOREGROUND with a tool timeout LARGER than `total_sec` plus the respawn time.
   This harness process stays alive, polls the background run, kills the process
   group if the budget is exceeded, then respawns the resident server in the
   foreground (`gpu_respawn_server.sh`) and confirms `/v1/models` before it
   returns. The watchdog is what brings the server back; never continue the
   session until it reports done.

```sh
tools/bench/bench_bg.sh pp16k 600 \
  build/release/gufo bench --model "$MODEL" --n-prompt 16384 --n-gen 1 & disown
tools/bench/bench_wait.sh pp16k 660 90   # foreground; tool timeout > 660s + respawn
cat /tmp/opencode/gufo_bench_pp16k.log   # read the real numbers afterwards
```

The watchdog may print `finished rc=0` immediately when the benchmark completed
between tool calls; always read the `.log` for the actual results.

### Iterate at short context first

Long-context prefill is slow (pp102400 ~90 s/prefill) and the long-context gap
widens with depth, so a lever that helps at 16K may not at 100K. But benchmarking
small levers at 32-100K wastes minutes per A/B. Iterate every candidate at
**<=16K context first** (pp2048/pp8192/pp16384, ~1-20 s/prefill); only escalate a
lever that shows a real win at short context to 32K/65K/100K confirmation.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
