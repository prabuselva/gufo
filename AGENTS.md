# AGENTS.md

## Platform and build

Production packages target Linux x86-64 AMD Strix Halo (`gfx1151`); Windows
is a supported native build target for development. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

Do not invoke `nix` on this machine; it is unavailable. Use the plain
`cmake`/`ctest` presets and host tools (e.g. `clang-format` on `PATH`) instead.

```sh
# Production build without Nix, with the documented dependencies installed
cmake --preset release                # discovery build
# Add -DGUFO_EXTENSIONS_FROM_LOCK=ON for the shipped extension selection.
cmake --build --preset release --parallel 4
```

Measure performance with `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
cmake --preset gpu-test
cmake --build --preset gpu-test --target <test-target>
ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

Model kernel tests (`EXCLUDE_FROM_ALL`) are absent from the default
`cmake --build --preset gpu-test`, and `ctest` never compiles: always pass the
explicit `--target` after editing kernels or a gate verdict comes from a stale
executable.

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

Text API functional/regression tests live in [tests/functional/](tests/functional/README.md).
Select suites explicitly. Model-specific changes need the affected model/modes;
use all four profiles only for shared behavior changes. Run correctness on the
candidate; use main for matched affected timing controls and suspected regressions,
not a duplicate full correctness run. Reuse baseline evidence only when revision,
toolchain, harness, model, settings and cache history match. Include `long-context`
and `cache` for continuation changes; verify loaded mode and actual drafts.
Correctness and expected prefill/cache work are strict; timings are
mandatory per request/phase at 5% and 10 ms, never averaged across requests.
Run once, investigate flags, then alternate main/PR only for affected histories;
use unchanged-main controls when needed and retain every result. Fix confirmed
regressions before publishing. Noisy evidence stays visibly inconclusive and
unqualified; do not widen margins or stop at reporting failures. Devlab keeps
inconclusive timing advisory on both ordinary and release PRs. Use the code
diff to select affected metrics; unrelated timing variance does not justify
another full matrix.
For reported coding-agent loops, also replay the affected workflow with
`tests/functional/pi_agent.py`; retain its sessions and inspect actual tool results.
Retain numerical quality tests and the standard speed benchmark. Keep these
tests in `tests/functional/`, outside hosted model CI; avoid full sweeps.

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
respawn down with it and leave the server down.

Instead run the benchmark in the background and the respawn in the foreground,
but ALWAYS launch BOTH in the SAME tool call (one shell session), never as two
separate calls. The watchdog must attach to the live run the instant it starts:
if `bench_bg.sh` and `bench_wait.sh` are separate tool calls, the gap between
them lets the benchmark finish and the server get respawned out-of-band, so the
watchdog only prints `already serving; skipping respawn` and never exercises the
wait+respawn path, and the detached run is no longer tied to this session.

1. `tools/bench/bench_bg.sh <name> <inner_timeout_sec> <cmd...>` runs
   `gpu_exclusive.sh --no-respawn` (stop + run only) under `timeout`, writes
   `${GUFO_BENCH_STATE_DIR:-/tmp/opencode}/gufo_bench_<name>.{log,pid,status}`,
   and returns immediately. Launch it with `setsid ... & disown` so its PID is a
   process group the watchdog can kill as a unit. The detached run also carries
   a fallback respawner: if it finds the watchdog gone (`gufo_bench_<name>.watchdog.pid`
   dead), no `gpu_exclusive.sh` running, and the port unserved after a grace
   period, it respawns the server itself, serialized by a respawn flock.
2. IMMEDIATELY, in the same command, run
   `tools/bench/bench_wait.sh <name> <total_sec> <ready_grace>` in the FOREGROUND
   with a tool timeout LARGER than `total_sec` plus the respawn time. It stays
   alive, polls the background run, kills the process group if the budget is
   exceeded, then respawns the resident server in the foreground
   (`gpu_respawn_server.sh`) and confirms `/v1/models` before returning. The
   watchdog is what brings the server back; never continue the session until it
   reports done. Tool-call timeouts are MILLISECONDS: `<total_sec>` is seconds,
   but the call's timeout argument is not - a mistyped small timeout kills the
   watchdog instantly and leaves the server down until the fallback fires.

```sh
# ONE tool call: launch detached, then wait + respawn in the same shell.
setsid tools/bench/bench_bg.sh pp16k 600 \
  build/release/gufo bench --model "$MODEL" --n-prompt 16384 --n-gen 1 & disown
sleep 2
tools/bench/bench_wait.sh pp16k 660 90   # foreground; tool timeout > 660s + respawn
cat /tmp/opencode/gufo_bench_pp16k.log   # read the real numbers afterwards
```

The watchdog prints `finished rc=0` as soon as the benchmark's status file
appears; always read the `.log` for the actual results.

### Iterate at short context first

Long-context prefill is slow (pp102400 ~90 s/prefill) and the long-context gap
widens with depth, so a lever that helps at 16K may not at 100K. But benchmarking
small levers at 32-100K wastes minutes per A/B. Iterate every candidate at
**<=16K context first** (pp2048/pp8192/pp16384, ~1-20 s/prefill); only escalate a
lever that shows a real win at short context to 32K/65K/100K confirmation.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use the shared Qwen control-token constants for framing; see
  [src/families/qwen/AGENTS.md](src/families/qwen/AGENTS.md). Inspect their definitions
  when reviewing or changing behavior, and keep reference fixtures independent.
- Put Windows-specific code in `src/core/platform/`: one leaf per OS facility
  (files, pages, sockets), whose sources may branch on `_WIN32` or split into
  `<leaf>_posix.cpp`/`<leaf>_windows.cpp` selected in CMake. Outside it and
  vendored code, call the leaf, with no `windows.h` or `_WIN32`/`_MSC_VER`
  checks. POSIX code moves into a leaf, verbatim, once it gains a Windows
  counterpart. Never emulate POSIX with macros or copy a file per OS; when the
  logic itself differs, add `<file>_windows.cpp` beside it.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message. Title pull requests
  according to [the release policy](docs/RELEASING.md); squash merging retains
  that title on `main`. Mark breaking changes with `!` in the title, and leave
  version and changelog updates to the release workflow.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- The source extension API is versioned independently from Gufo. Public headers
  under `include/gufo/ext/`, their transitive `SURFACE` closure (including
  forwarded core/cache headers), the metadata schema and public CMake contracts
  require an `ext-api` PR label and one new, uniquely named Markdown record under
  `include/gufo/ext/changes/<major>.<minor>/` when declarations/contracts change.
  Records are permanent and immutable. CMake derives the API revision from the
  active series in `include/gufo/ext/api-series.json` plus its revision-record
  count. The `baseline.md` record documents the initial contracts without advancing
  it. Do not manually bump versions for additive changes.
  Run `python3 tools/ci/check-extension-api.py --write` after header/include
  changes; the PR diff gate also checks forwarded declarations. Before 1.0, a minor bump
  may break extensions; revision bumps are additive. Fix in-tree users together.
  The guarantee is source compatibility after rebuilding, never binary ABI.
  Breaking changes explicitly advance the compatibility series in api-series.json
  and add a record in the new series, retaining all historical records. Extension
  API and the future experimental flake `extensions` argument are excluded from Gufo's
  release compatibility surface; CLI/HTTP/public existing flake breaks still
  require `!`. See [the release policy](docs/RELEASING.md).
- Follow the user's remote workflow and preserve unrelated work.
