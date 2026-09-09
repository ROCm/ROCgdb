"""rocjitsu emulator support: run the GPU test suites without an AMD GPU.

rocjitsu emulates an AMDGPU in userspace by interposing the KFD (/dev/kfd)
ioctls, and `mirage run` is the CLI that owns an emulated GPU session for the
lifetime of the command it launches. So "running on an emulated GPU" means
nothing more than launching the GPU-touching command inside such a session --
which is what Emulator.wrap() produces.

This module is the support code for that. Its caller is test_rocgdb.py
--emulate PROFILE, which wraps the `make check` invocations so the tests run on
the emulated GPU while the harness itself -- configure, result parsing, ignore
lists, reports -- stays outside the session.

Both halves come from the ROCm tree under test, and the run's own environment
finds them: mirage from bin on PATH, and the emulator library it loads from lib
on LD_LIBRARY_PATH. Nothing here asks the user to configure anything.

Errors are raised as EmulatorError; nothing here exits the process, so callers
report failures in their own style.

See EMULATED-TESTS.md for how to use the flag this module implements.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The emulated GPU profiles mirage ships, and the GPU each one emulates. The
# profile decides what the test programs are compiled for, so it has to match
# what the ROCm tree under test was built for.
PROFILE_ARCH = {"mi300x": "gfx942", "mi350x": "gfx950", "mi450x": "gfx1250"}
ARCH_PROFILE = {arch: profile for profile, arch in PROFILE_ARCH.items()}

# Emulated tests run far slower than on hardware, so dejagnu's 10 second
# default per-testcase timeout is what makes the slow ones fail rather than any
# real problem.
DEFAULT_TIMEOUT = 600

# What a parallel emulated run costs, as measured on a gdb.rocm run: a fixed
# cost, plus roughly a GB for each job, since a job holds an emulated GPU
# rather than waiting for one. Wall clock stops improving well before memory
# does, so the ceiling costs almost nothing in time while keeping the memory
# bill bounded.
JOB_MEMORY_GB = 1.0
BASE_MEMORY_GB = 8.0
JOB_CEILING = 16


class EmulatorError(Exception):
    """A problem setting up or driving the emulated GPU."""


def profile_choices() -> List[str]:
    """The accepted --emulate values, for help text and error messages."""
    return list(PROFILE_ARCH) + list(ARCH_PROFILE)


def normalize_profile(spec: str) -> Tuple[str, str]:
    """Resolve a profile spec to (mirage profile, gfx target).

    Accepts the mirage profile ("mi350x"), the same without the trailing x
    ("mi350"), or the GPU it emulates ("gfx950"), because CI thinks in AMDGPU
    families while mirage thinks in profiles.
    """
    key = spec.strip().lower()
    for candidate in (key, f"{key}x"):
        if candidate in PROFILE_ARCH:
            return candidate, PROFILE_ARCH[candidate]
    if key in ARCH_PROFILE:
        return ARCH_PROFILE[key], key
    raise EmulatorError(
        f"unknown emulated GPU {spec!r}; expected one of "
        f"{', '.join(profile_choices())}"
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _available_memory_gb() -> Optional[float]:
    """Memory this machine can hand out right now, or None if unknowable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


class JobAdvice(NamedTuple):
    """How many parallel jobs to run, and what to tell the user about it."""

    jobs: int
    message: str
    warn: bool


def parallel_jobs(
    requested: Optional[int], cpu_count: int, available_gb: Optional[float] = None
) -> JobAdvice:
    """Decide how many jobs a parallel emulated run should use.

    On hardware the job count is a CPU question, because the jobs queue for the
    one GPU either way. Under emulation each job holds an emulated GPU of its
    own, so the job count is a memory question too, and the harness default (as
    many jobs as CPUs) can ask for far more memory than a machine has -- which
    surfaces as tests killed by the OOM killer, indistinguishable from real
    failures.

    So a default is capped: by what memory allows, and by the point where more
    jobs stop buying wall clock. A number the caller asked for is honoured,
    since they may know something we do not (a big machine, a small suite), but
    it is flagged when memory says it will not fit.
    """
    if available_gb is None:
        available_gb = _available_memory_gb()

    fits = None
    if available_gb is not None:
        fits = max(1, int((available_gb - BASE_MEMORY_GB) / JOB_MEMORY_GB))

    if requested is not None:
        if fits is not None and requested > fits:
            return JobAdvice(
                requested,
                f"--jobs {requested} means {requested} emulated GPUs, roughly "
                f"{BASE_MEMORY_GB + requested * JOB_MEMORY_GB:.0f}GB, and only "
                f"{available_gb:.0f}GB is available: the OOM killer will take "
                f"tests down, which reads as failures. "
                f"{_plural(fits, 'job')} fit here.",
                True,
            )
        return JobAdvice(requested, "", False)

    jobs = min(cpu_count, JOB_CEILING)
    reason = (
        f"the {JOB_CEILING} job ceiling" if jobs == JOB_CEILING else "this host's CPUs"
    )
    if fits is not None and fits < jobs:
        jobs, reason = fits, f"{available_gb:.0f}GB of available memory"
    return JobAdvice(
        jobs,
        f"--emulate --parallel: using {_plural(jobs, 'job')}, from {reason}. "
        f"Each job holds an emulated GPU of its own (about "
        f"{JOB_MEMORY_GB:.0f}GB), and more than {JOB_CEILING} stops buying "
        f"wall clock; pass --jobs to override.",
        False,
    )


def find_mirage(path: Optional[str] = None) -> Path:
    """Return the mirage binary that will own the emulated session.

    Found on `path`, the run's own PATH, which starts with the bin of the ROCm
    tree under test, so a tree built with the emulation tools provides its own
    mirage and the caller configures nothing.
    """
    on_path = shutil.which("mirage", path=path)
    if on_path:
        return Path(on_path).resolve()

    raise EmulatorError(
        "no mirage found: --emulate needs it to own the emulated GPU session. "
        "Either the run is pointed at a tree that is not a ROCm build with the "
        "emulation tools, or that tree was built without them "
        "(-D THEROCK_ENABLE_EMULATION=ON)."
    )


def check_profile_arch(rocm_root: Path, profile: str, arch: str) -> None:
    """Refuse a profile the ROCm tree was not built for.

    The profile decides which GPU the test programs are compiled for, so a tree
    built for another family fails deep inside the tests instead of here.
    TheRock records what it built for in the dist info; a tree without that
    file (a plain ROCm install) is left alone.
    """
    info = rocm_root / "share" / "therock" / "dist_info.json"
    try:
        targets = json.loads(info.read_text()).get("dist_amdgpu_targets", "")
    except (OSError, ValueError):
        return
    # A local build writes the list as it was given to cmake, a comma or space
    # separated string; the nightly tarballs write cmake's own list separator.
    built_for = targets.replace(",", " ").replace(";", " ").split()
    if not built_for or arch in built_for:
        return
    alternatives = sorted({ARCH_PROFILE[a] for a in built_for if a in ARCH_PROFILE})
    hint = (
        f"emulate {' or '.join(alternatives)} instead"
        if alternatives
        else "point the run at a matching tree"
    )
    raise EmulatorError(
        f"{profile} emulates {arch}, but {rocm_root} was built for "
        f"{' '.join(built_for)}: {hint}, or rebuild the tree for {arch}."
    )


class Emulator:
    """An emulated GPU: a mirage binary and the profile it brings up."""

    def __init__(self, profile: str, arch: str, mirage: Path):
        self.profile = profile
        self.arch = arch
        self.mirage = mirage
        # Written on demand by runtest_wrapper(), then reused: one script
        # serves every job of every iteration.
        self._runtest: Optional[Path] = None

    @classmethod
    def resolve(
        cls, spec: str, rocm_root: Path, path: Optional[str] = None
    ) -> "Emulator":
        """Build an Emulator for `spec`, validating everything it needs.

        `path` is the run's PATH, where mirage is found; mirage then loads the
        emulator library itself, from the tree's lib on LD_LIBRARY_PATH.
        """
        profile, arch = normalize_profile(spec)
        check_profile_arch(Path(rocm_root), profile, arch)
        return cls(profile, arch, find_mirage(path))

    def wrap(self, cmd: Sequence, *, env: Optional[Dict[str, str]] = None) -> List[str]:
        """Return `cmd` rewritten to run inside an emulated GPU session.

        The session lives for exactly as long as the command, and the whole
        process subtree it spawns (make -> runtest -> amdclang++, rocgdb, the
        inferior) shares that one emulated GPU. mirage passes its own
        environment on to the command, so `env` is only for the few variables
        that must differ inside the session.
        """
        wrapped = [str(self.mirage), "run", "--profile", self.profile]
        for key, value in (env or {}).items():
            wrapped += ["--env", f"{key}={value}"]
        wrapped.append("--")
        return wrapped + [str(c) for c in cmd]

    def runtest_wrapper(self) -> Path:
        """Return a `runtest` replacement that sessions each dejagnu job.

        Wrapping the whole `make check` (see wrap()) puts every parallel job on
        the one emulated GPU that session owns, and the testsuite then
        serializes them anyway: `with_rocm_gpu_lock` exists because hardware has
        a single GPU, and it holds the lock for the GPU-touching part of nearly
        every gdb.rocm test.

        Under emulation a GPU costs a process rather than a socket, so a job can
        have one to itself. The testsuite's parallel machinery runs one
        `$(RUNTEST)` per test file, so pointing RUNTEST at this script is what
        turns "one session, jobs queueing on a lock" into "a session per job".
        The lock itself is made per-job rather than removed, so the tests keep
        using the same code path -- lock_dir() honours GDB_LOCK_DIR -- and a
        job still cannot collide with itself.

        The script and the lock directories live in a temporary directory that
        is removed when the process exits.
        """
        if self._runtest is not None:
            return self._runtest

        scratch = Path(tempfile.mkdtemp(prefix="rocjitsu-runtest-"))
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        script = scratch / "runtest-in-session"
        session = shlex.join(self.wrap(["runtest"]))
        # Everything interpolated into the script is quoted for the shell that
        # will run it: $TMPDIR decides where the scratch directory lands, and it
        # is not ours to assume anything about.
        template = shlex.quote(f"{scratch}/gdb-lock.XXXXXX")
        # Created executable, since dejagnu execs it as $(RUNTEST).
        with os.fdopen(
            os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755),
            "w",
            encoding="utf-8",
        ) as out:
            out.write(
                "#!/bin/sh\n"
                "# Generated by rocjitsu_emulator.py: run one dejagnu job on an\n"
                "# emulated GPU of its own, with a GPU lock of its own.\n"
                f"lock_dir=$(mktemp -d {template}) || exit 1\n"
                "trap 'rm -rf \"$lock_dir\"' HUP INT TERM\n"
                f'{session} "GDB_LOCK_DIR=$lock_dir" "$@"\n'
                "status=$?\n"
                'rm -rf "$lock_dir"\n'
                "exit $status\n"
            )
        self._runtest = script
        return script

    def cleanup(self) -> None:
        """Reclaim what a killed `mirage run` left behind (best effort).

        A run tears its own session down on exit, but SIGKILL (which is how the
        harness enforces its wall-clock timeouts) leaves no code of mirage's to
        run at all. `mirage cleanup` is the documented remedy; it leaves live
        sessions alone, so calling it mid-run is safe.

        Not every mirage has the subcommand: it landed in August 2026, and a
        dist tarball carries whatever mirage its ROCm tree was pinned to. Say
        so rather than failing, since the run itself is unaffected -- what is
        left behind is a stranded emulated GPU, costing memory until the host
        is rebooted or a newer mirage cleans up after it.
        """
        try:
            done = subprocess.run(
                [str(self.mirage), "cleanup"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"mirage cleanup failed: {e}")
            return
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip().splitlines()
            logger.warning(
                f"could not reclaim the emulated GPU: {self.mirage} cleanup "
                f"exited {done.returncode}"
                + (f" ({detail[0]})" if detail else "")
                + ". A killed test may have left an emulated GPU running."
            )

    def summary(self) -> List[Tuple[str, str]]:
        """Label/value pairs describing this emulated GPU, for config output."""
        return [
            ("Emulated GPU", f"{self.profile} ({self.arch})"),
            ("mirage", str(self.mirage)),
        ]

    def __str__(self) -> str:
        return f"{self.profile}/{self.arch} via {self.mirage}"
