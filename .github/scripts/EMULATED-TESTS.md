# Running the ROCgdb tests without hardware

`--emulate GPU` runs the tests on the rocjitsu userspace emulator instead of
real hardware, so the GPU tests can run on a machine with no AMD GPU. Which
tests run is still up to `--tests`, `--gpu-tests` and the rest, exactly as on
hardware.

GPU is a mirage profile (`mi300x`, `mi350x`, `mi450x`) or the gfx target it
emulates (`gfx942`, `gfx950`, `gfx1250`), and the ROCm tree under test has to
have been built for it.

| Profile | gfx target |
|---|---|
| `mi300x` | `gfx942` |
| `mi350x` | `gfx950` |
| `mi450x` | `gfx1250` |

## Usage

Point the run at the ROCm tree under test and name the GPU to emulate:

```bash
OUTPUT_ARTIFACTS_DIR=~/therock/build/dist/rocm \
    python test_rocgdb.py --emulate mi350x --gpu-tests

OUTPUT_ARTIFACTS_DIR=~/therock/build/dist/rocm \
    python test_rocgdb.py --emulate mi350x --gpu-tests --parallel
```

## Requirements

- Linux. No AMD GPU, no `/dev/kfd`, no root.
- A ROCm tree carrying the emulation tools, which a normal TheRock build and
  the nightly tarballs both have. A minimal tree built to work on the debug
  tools alone needs `-D THEROCK_ENABLE_EMULATION=ON`.
- `dejagnu` and a C compiler, as on hardware.

## Notes

- `--default-timeout` is raised automatically, since emulated tests are slower.
- `--parallel` is worth using: each job gets its own emulated GPU instead of
  queueing for one. Since memory grows with the job count, `--jobs` is capped at
  what the machine can hold, and a `--jobs` that will not fit is flagged.
- `--one-by-one` gives each test a session, and so a clean GPU, of its own.
- `--sanity-check` is ignored (not meaningful for an emulated device).

## Troubleshooting

- **`mirage NOT found on PATH`**: the run is pointed at a tree that is not a
  ROCm build carrying the emulation tools, or at one built without them, which
  wants `-D THEROCK_ENABLE_EMULATION=ON`.
- **Tests fail or time out**: first raise `--default-timeout`, since emulated
  tests are slower. If they still fail, it may be a genuine test, emulation, or
  emulated-GPU issue rather than the harness.
