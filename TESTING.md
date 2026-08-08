# Testing ROCgdb

**ROCgdb is AMD's fork of the [GNU Debugger (GDB)](https://www.gnu.org/software/gdb)
that adds heterogeneous debugging support for AMD GPUs.** It inherits GDB's
testing infrastructure and adds a GPU-specific test suite (`gdb.rocm/`) and CI on
top of it. The goal is to keep ROCgdb correct for both CPU and GPU debugging as
it tracks upstream GDB and successive ROCm releases.

**Testing spans a broad surface.** ROCgdb must work across many AMD GPU
architectures, host operating systems, and toolchains, on top of everything
upstream GDB already supports. Testing every combination on every change is not
practical, so tests are layered by cost: fast, high-signal suites run on every
change, while configurations CI cannot cover per PR - for example GPU
architectures without dedicated per-PR runner capacity - run nightly or on
demand.

**Testing should be accessible to contributors.** Wherever possible the same
`make check` targets run on a developer's machine and in CI, so behavior can be
validated locally first, with CI providing consistent, representative
environments.

> **Upstream is the source of truth.** GDB's testing conventions are defined
> upstream, and where this document diverges, upstream wins. Read these first:
> - `gdb/testsuite/README` - the reference for the GDB testsuite.
> - <https://sourceware.org/gdb/wiki/TestingGDB> - the GDB testing wiki.
>
> This page summarizes and extends those conventions for ROCgdb; it is a
> practical starting point, not a complete manual.

## Testing principles

### Make tests accessible during development

Every fix or feature should come with a test that verifies the specific behavior
being changed, and that a contributor can run quickly while developing. Selftests
need no GPU and run in-process; the functional testsuite runs against a locally
built debugger, so most behavior can be validated locally before CI.

### Use layered validation

ROCgdb is validated in three layers, cheapest first. Prefer the cheapest layer
that can catch a given problem:

- **Unit tests (selftests)** - in-process C++ checks compiled into GDB
  (`gdb/unittests/`); run with `maintenance selftest` and also exercised by the
  functional testsuite.
- **Functional tests** - end-to-end DejaGnu tests under `gdb/testsuite/gdb.*/`,
  including the GPU suite `gdb.rocm/`. This is where most ROCgdb testing happens.
- **Performance tests** - the perftest harness under `gdb/testsuite/gdb.perf/`
  (currently exercised only rarely).

### Scale coverage to available resources

High-signal suites run on every change; less common configurations run nightly or
on demand. Some GPUs cannot debug multiple processes concurrently, so GPU tests
serialize device access with a lock (`with_rocm_gpu_lock`); `-j$(nproc)` therefore
speeds up compilation more than GPU test execution.

### Use static analysis for mechanical checks

Formatting and other mechanical checks run as pre-commit hooks in CI
(`.github/workflows/pre-commit.yml`), keeping human review focused on substance.

### Add reliable tests to required CI

CI runs the testsuite through `.github/scripts/test_rocgdb.py` and compares
results against a known-failures list. Only add tests that are deterministic, so
required CI stays trustworthy and failures are actionable.

## Testing changes to ROCgdb

### Core, target-independent changes

Cover self-contained logic (utilities, containers, parsers) with a **selftest**
next to the code, and observable debugger behavior with a **DejaGnu** test in the
matching `gdb.*/` directory, chosen by area: core debugger features (`gdb.base`),
architecture (`gdb.arch`), language support such as C++ (`gdb.cp`) and Python
(`gdb.python`), the machine interface (`gdb.mi`), and so on. See
[Run the same tests locally and in CI](#run-the-same-tests-locally-and-in-ci) for
how to run a chosen subset.

### AMDGPU and ROCm-specific changes

Add a test under `gdb.rocm/`. A GPU test loads `rocm.exp`, gates on a capability
check (`require allow_hip_tests`), compiles its program as HIP device code, and
serializes device access with `with_rocm_gpu_lock`. For the mechanics of writing
`.exp` tests, follow `gdb/testsuite/README` and copy an existing `gdb.rocm/` test
as a template.

### Performance-sensitive changes

Add or run a **perf** test under `gdb.perf/`. Performance tests are not part of
`make check`; run them on demand and compare measurements before and after on the
same hardware. ROCgdb does not currently enforce automated regression thresholds
in CI. See `gdb/testsuite/gdb.perf/README`.

## Testing ROCgdb against ROCm

### Build with amd-dbgapi

GPU debugging requires ROCgdb built `--with-amd-dbgapi` and a working ROCm stack
(the amd-dbgapi library and a HIP compiler). Without these the GPU capability
gate reports the `gdb.rocm/` tests as unsupported and they do not run.

### Keep test environments reproducible

The testsuite reads the ROCm environment it needs (e.g. `ROCM_PATH`) and manages
any GPU-test-specific variables where required, so a correct ROCm install and an
amd-dbgapi-enabled build are generally all that is needed. Avoid depending on ad
hoc environment tweaks in individual tests.

### Run the same tests locally and in CI

The entry point is the GDB testsuite in both cases. CI wraps it with
`.github/scripts/test_rocgdb.py`, which runs the CPU and GPU test sets across the
supported toolchains, retries flaky failures, and applies the known-failures
list. It is also the easiest way to reproduce a CI run locally; run it with
`--help` for the current options.

## Validating ROCgdb on hardware and in CI

### Test on supported GPU hardware

GPU results are only meaningful on supported devices. On unsupported or mixed
hardware, expect real failures rather than clean skips: ensure every visible
device is supported, or restrict the run to a supported device with
`ROCR_VISIBLE_DEVICES`.

### When tests run

CI is implemented with GitHub Actions under `.github/workflows/`. On pull
requests and pushes to the staging branches, ROCgdb is built and tested on both
CPU and GPU runners. At present each PR runs the GPU suite (`gdb.rocm`, excluding
corefile tests) on GPU runners and `gdb.dwarf2` on CPU runners, with GPU corefile
tests being added on a dedicated runner. Documentation-only changes and PRs
labeled to skip CI do not trigger full runs. The workflow files are the source of
truth for exact triggers and cadence.

### Read results and triage failures

Results are written in the build tree as `gdb.sum` (summary) and `gdb.log`
(detail), with per-test artifacts under `outputs/` for replaying a failure. They
use the standard DejaGnu codes (`PASS`, `FAIL`, `UNTESTED`, `UNSUPPORTED`,
`UNRESOLVED`, `XFAIL`, `KFAIL`); see `gdb/testsuite/README` for their meanings.
