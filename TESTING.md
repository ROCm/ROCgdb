# Testing ROCgdb

ROCgdb is AMD's fork of the [GNU Debugger (GDB)](https://www.gnu.org/software/gdb)
that adds heterogeneous debugging support for AMD GPUs. It inherits GDB's testing
infrastructure and adds a GPU-specific test suite and CI on top of it.

This document summarizes ROCgdb's testing **strategy** — what kinds of tests we
have, what they look like, when they run, and how to add new ones. It is a
practical starting point, not a complete manual.

> **Upstream is the source of truth.** GDB's testing conventions are defined
> upstream, and where this document diverges from upstream, upstream wins. Read
> these first and treat them as authoritative:
> - `gdb/testsuite/README` — the reference for the GDB testsuite.
> - <https://sourceware.org/gdb/wiki/TestingGDB> — the GDB testing wiki.
>
> For building and installing ROCgdb (toolchain, dependencies, `configure`
> options), see `README-ROCM.md` and `gdb/README`; build/runtime prerequisites
> are intentionally **not** duplicated here.

## Test strategies at a glance

| Strategy | What it validates | Framework | Location |
|----------|-------------------|-----------|----------|
| **Unit tests (selftests)** | Small, self-contained C++ helpers, compiled into GDB and run in-process | GDB selftest framework | `gdb/unittests/` (+ selftests embedded in subsystem sources) |
| **Functional tests** | End-to-end debugger behavior driven through the real GDB interface, including all GPU debugging | DejaGnu (Tcl/Expect) | `gdb/testsuite/gdb.*/` |
| **Performance tests** | Timing/throughput of expensive operations | GDB perftest harness | `gdb/testsuite/gdb.perf/` |

GPU testing is not a separate strategy: the GPU tests (`gdb/testsuite/gdb.rocm/`)
are part of the DejaGnu functional suite, and there is a GPU performance test as
well. GPU-specific notes are called out inline below.

## 1. Unit tests (selftests)

GDB's "unit tests" are **selftests**: small checks compiled into the debugger and
run in-process, rather than a separate GoogleTest/pytest binary. They cover
self-contained logic that needs no running inferior (utilities, containers,
parsers, encoders). Most live in `gdb/unittests/`, with more embedded in the
subsystem sources they exercise.

Selftests are built into development builds by default and run via the
`maintenance selftest` command:

```shell
gdb --batch -ex "maintenance selftest"
```

They also run as part of the functional suite (`gdb.gdb/unittest.exp`), so a
failing selftest is a normal test regression. New standalone, inferior-free
logic should come with a selftest.

## 2. Functional tests (DejaGnu)

The bulk of ROCgdb's validation — including all GPU debugging — is end-to-end
functional testing through the real debugger interface, using the DejaGnu
framework under `gdb/testsuite/`. Tests are grouped by area in `gdb.*/`
directories (e.g. `gdb.base/`, `gdb.cp/`, `gdb.python/`, `gdb.mi/`, and the
GPU-specific `gdb.rocm/`). A test is an `.exp` script, usually paired with a
source program of the same basename that DejaGnu compiles on the fly.

### Running

Use the `check-gdb` target from the top of the build tree, selecting tests with
`TESTS`:

```shell
# Whole suite
make check-gdb

# A subset or a single test
make check-gdb TESTS="gdb.rocm/*.exp"
make check-gdb TESTS="gdb.rocm/alu-exceptions.exp"
```

Other testsuite targets (the whole-suite `check`, the `check-read1` /
`check-readmore` I/O-stress variants, parallel and per-board variants, etc.) are
most directly invoked from the testsuite build directory, e.g.:

```shell
make -C build/gdb/testsuite check-read1 TESTS="gdb.base/break.exp"
```

Invoking `runtest` directly is possible but not recommended for routine use — it
needs the right environment (e.g. `-data-directory`) that the make targets set up
for you. Prefer `make check`/`check-gdb`.

Results land in the build tree as `gdb.sum` (summary) and `gdb.log` (detail),
with per-test artifacts under `outputs/` for replaying failures. Results use the
standard DejaGnu codes (`PASS`, `FAIL`, `UNTESTED`, `UNSUPPORTED`, `UNRESOLVED`,
`XFAIL`, `KFAIL`); see `gdb/testsuite/README` for their meanings.

### GPU tests (`gdb.rocm/`)

The GPU suite exercises ROCgdb-specific behavior: device-code breakpoints and
stepping, wave/lane inspection, watchpoints, GPU exceptions, core dumps, HIP and
OpenMP-offload features, and multi-process GPU debugging. Its helpers live in
`gdb/testsuite/lib/rocm.exp`, and tests opt in through a capability gate that
requires an amd-dbgapi-enabled build and at least one AMD GPU present. Note that
this gate does not guarantee the device is actually *supported* — see below.

Practical notes for running the GPU tests:

- **Hardware matters.** These tests need a working ROCm stack and supported AMD
  GPU(s). On unsupported or mixed hardware, expect real failures rather than
  clean skips: to get meaningful results, every visible device should be
  supported, or the run should be isolated to a supported device (e.g. via the
  usual ROCm device-visibility controls).
- **GPU access is serialized.** GPUs are a shared, limited resource, so GPU tests
  wrap their device interaction in `with_rocm_gpu_lock` (in `lib/rocm.exp`). This
  takes a lock so that parallel test runs don't fight over the same device; any
  new GPU test that touches hardware should do the same.

### CI test runner

CI drives the suite through `.github/scripts/test_rocgdb.py`, a wrapper around
`make check` that runs the GPU and CPU test sets across the supported compilers,
retries flaky failures, and applies a known-failures ("ignore") list. It is the
recommended entry point for reproducing CI locally; run it with `--help` for
current options rather than relying on a fixed list here.

## 3. Performance tests

Performance tests live in `gdb/testsuite/gdb.perf/` and use GDB's perftest
harness. They measure the cost of expensive operations (symbol lookup, large
solib counts, etc.) and include a GPU perf test. They are **not** part of
`make check`; they are run on demand and record their measurements to a log for
comparison. ROCgdb does not currently enforce automated performance regression
thresholds in CI — perf tests are used for before/after comparison on the same
hardware. See `gdb/testsuite/gdb.perf/README`.

## 4. When tests run

CI is implemented with GitHub Actions under `.github/workflows/` (the AMD
"TheRock" build/test pipeline). On pull requests and pushes to the staging
branches, ROCgdb is built and the testsuite is run on both CPU and GPU runners
via the CI runner script above. Documentation-only changes and PRs labeled to
skip CI do not trigger full runs. The workflow files are the source of truth for
exact triggers and cadence.

## 5. How to write tests

Pick the layer that fits:

- **No running inferior needed** (a pure helper, container, parser) → add a
  **selftest** next to the code it covers.
- **Observable debugger behavior** (breakpoints, stepping, printing, GPU/wave
  handling) → add a **DejaGnu `.exp`** test in the matching `gdb.*/` directory
  (`gdb.rocm/` for GPU behavior).
- **Cost of an operation** → add a **perf** test under `gdb.perf/`.

For functional tests, follow the upstream conventions in `gdb/testsuite/README`:
an `.exp` driver plus a same-named source program, one focused feature per file,
and precise output matching so real regressions are caught. Gate
environment-dependent tests so they are skipped (not failed) where the feature
genuinely cannot run, and keep tests parallel-safe. A GPU test additionally loads
`rocm.exp`, requires the GPU capability gate, compiles its program as HIP device
code, and serializes hardware access with `with_rocm_gpu_lock`.

A GPU functional test looks roughly like this (illustrative skeleton — copy an
existing `gdb.rocm/` test as your real template):

```tcl
load_lib rocm.exp                 ;# GPU test helpers
require allow_hip_tests           ;# skip unless the build + GPU can run it
standard_testfile .cpp            ;# pairs <name>.exp with <name>.cpp

if {[build_executable "failed to prepare $testfile" $testfile $srcfile \
        {debug hip}]} {
    return                        ;# compile failure -> UNTESTED
}

clean_restart
gdb_load $binfile

with_rocm_gpu_lock {              ;# serialize shared-GPU access
    if {![runto_main]} {
        return
    }
    # ... gdb_test / gdb_test_multiple assertions for the feature ...
}
```
