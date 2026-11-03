#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import atexit
import ctypes
import functools
import glob
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
import types
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Dict, List, NoReturn, Optional, Set, Tuple, Union

# True if this Python runs on Windows, natively or built against the Cygwin or
# MSYS2 runtime, which reports sys.platform "cygwin".
IS_WINDOWS = sys.platform in ("win32", "cygwin")
# True if this Python uses Windows' own APIs, processes and drive-letter paths.
# A Cygwin or MSYS2 Python on Windows uses POSIX ones instead.
IS_NATIVE_WINDOWS = sys.platform == "win32"
# True if this Python is built against the Cygwin or MSYS2 runtime, which runs
# on Windows but takes POSIX paths.
IS_POSIX_WINDOWS = IS_WINDOWS and not IS_NATIVE_WINDOWS
# True if this Python runs on Linux.
IS_LINUX = sys.platform.startswith("linux")

# "resource" doesn't exist in a native Windows Python.  Its only user,
# setup_core_file_info(), runs on Linux only.
if not IS_NATIVE_WINDOWS:
    import resource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Status symbols used throughout.
STATUS_PASS = "[✓]"
STATUS_FAIL = "[X]"
STATUS_WARN = "[!]"
STATUS_FLAKY = "[~]"

# Test result categories.
TEST_CATEGORIES = [
    "PASS",
    "FAIL",
    "ERROR",
    "UNRESOLVED",
    "TIMEOUT",
    "UNTESTED",
    "UNSUPPORTED",
    "XFAIL",
    "KFAIL",
    "FLAKY",  # Tests that failed initially but passed on retry.
]

# Default wall-clock timeout (seconds) for each `make check` invocation in
# --one-by-one mode when --one-by-one-test-timeout is not provided.
ONE_BY_ONE_DEFAULT_WALL_CLOCK_TIMEOUT = 3600

# Default wall-clock timeout (seconds) for each --sanity-check probe when
# --sanity-check-timeout is not provided. A trivial known-good HIP kernel
# should return near-instantly; this is a starting value to be tuned
# experimentally on real hardware.
SANITY_CHECK_DEFAULT_TIMEOUT = 10

# Sanitizers we pre-configure on every run. If rocgdb wasn't built with the
# corresponding -fsanitize=..., the matching *_OPTIONS env var is silently
# ignored at runtime, so listing extras here is harmless.
SANITIZERS = ("asan", "tsan", "ubsan", "msan", "lsan", "hwasan")

# Supported toolchains, keyed by the identifier the toolchain option accepts.
# Each one maps to the compiler label (a load bearing key used across results,
# ignore lists, and timing output) and the C, C++, and Fortran executables we
# pass to the testsuite. Insertion order defines the default run order.
TOOLCHAINS = {
    "gnu": {"label": "GCC", "cc": "gcc", "cxx": "g++", "fc": "gfortran"},
    "llvm": {"label": "LLVM", "cc": "clang", "cxx": "clang++", "fc": "flang"},
}

TIMING_LOG_FILENAME = "rocgdb_timing.log"

# Patterns for parsing DejaGnu .sum lines.
RESULT_LINE_RE = re.compile(
    r"^(PASS|FAIL|XFAIL|UNTESTED|UNSUPPORTED|KFAIL|UNRESOLVED): (.+)$"
)
TIMEOUT_RE = re.compile(r"\(timeout\)")
RUNNING_EXP_RE = re.compile(r"Running\s+(\S+\.exp)")

# Category display configuration: prefix symbol.
CATEGORY_DISPLAY = {
    "PASS": STATUS_PASS,
    "FAIL": STATUS_FAIL,
    "ERROR": STATUS_FAIL,
    "UNRESOLVED": STATUS_FAIL,
    "TIMEOUT": STATUS_WARN,
    "UNTESTED": STATUS_WARN,
    "UNSUPPORTED": STATUS_WARN,
    "XFAIL": STATUS_WARN,
    "KFAIL": STATUS_WARN,
    "FLAKY": STATUS_FLAKY,
}


def _log_error_and_exit(message: str, exit_code: int = 1) -> NoReturn:
    """
    Log an error message and exit the program.

    Args:
        message: Error message to log.
        exit_code: Exit code (default 1).
    """
    logger.error(f"{STATUS_FAIL} Error: {message}")
    sys.exit(exit_code)


# Killing the test process tree on Linux.
#
# "make check" runs in a session of its own, so its process group holds the
# processes it starts, except those that start sessions of their own, as
# expect's spawn does for gdb, which then takes the pty as its controlling
# terminal.  So killing "make check"'s process group would not kill every
# descendant.  Instead, while "make check" runs, the script is a child
# subreaper, which makes every process the make command started the script's
# descendant even after that process's parent dies.  On a timeout, or once
# the command ends, the script kills them all (see _LinuxLauncher).


# The prctl option, from linux/prctl.h.
_PR_SET_CHILD_SUBREAPER = 36


def _linux_set_child_subreaper(enable: bool) -> None:
    """
    Linux only: make this process a child subreaper, or stop it being one.
    While it is one, a process whose parent dies is reparented to this
    process instead of to init, so that every process this process
    started, directly or not, stays its descendant until it exits.

    Args:
        enable: Whether to become a subreaper.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, int(enable), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_CHILD_SUBREAPER): {os.strerror(errno)}")


def _linux_read_process_table() -> Dict[int, Tuple[int, str]]:
    """
    Linux only: read every process's parent PID and state letter from /proc.

    Returns:
        A map from PID to (parent PID, state), state "Z" for a zombie.
    """
    table = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            stat = _linux_read_proc_stat(int(entry))
            if stat is not None:
                table[int(entry)] = stat
    return table


def _linux_read_proc_stat(pid: int) -> Optional[Tuple[int, str]]:
    """
    Linux only: read a process's parent PID and state letter from /proc.

    Args:
        pid: The process.

    Returns:
        (parent PID, state), or None if there is no such process.
    """
    # Read as bytes: the kernel cuts the command name at 15 bytes, which can
    # split a multi-byte character, and the name isn't needed anyway.
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return None
    # The command name, in parentheses, may itself hold spaces and
    # parentheses, so count the fields that follow it from the last ")".
    fields = stat[stat.rindex(b")") + 2 :].split()
    return int(fields[1]), fields[0].decode()


def _linux_kill_descendants(
    spare: Optional[int] = None, timeout: float = 10.0
) -> List[str]:
    """
    Linux only: kill every descendant of this process, and reap the ones
    that were reparented to it, until none is left alive or the timeout
    passes.  Logs those still alive then.  This process must be a child
    subreaper (see _linux_set_child_subreaper), or descendants whose parent
    died are out of its reach.  It can't tell one command's processes from
    another's, so it must not run while this process has any other child
    to keep, such as a second Popen.

    Args:
        spare: A child of this process to kill but not reap, because a Popen
            object will.
        timeout: Seconds to keep trying, for processes that take long to die,
            such as one stuck in a driver.

    Returns:
        The processes sent SIGKILL, each as "PID (command line)".
    """
    me = os.getpid()
    deadline = time.monotonic() + timeout
    # Keyed by PID, as a process that takes a while to die is signaled
    # again on the next pass.
    killed = {}
    while True:
        table = _linux_read_process_table()
        children = defaultdict(list)
        for pid, (ppid, _) in table.items():
            children[ppid].append(pid)
        tree = set()
        pending = [me]
        while pending:
            for child in children[pending.pop()]:
                if child not in tree:
                    tree.add(child)
                    pending.append(child)
        alive = sorted(pid for pid in tree if table[pid][1] != "Z")
        tree.add(me)
        for pid in alive:
            name = _linux_kill_tree_process(pid, tree)
            if name is not None:
                killed.setdefault(pid, f"{pid} ({name})")
        _linux_reap_children(table, spare)
        if not alive:
            return list(killed.values())
        if time.monotonic() > deadline:
            logger.warning(f"Processes still alive after SIGKILL: {alive}")
            return list(killed.values())
        time.sleep(0.05)


def _linux_reap_children(
    table: Dict[int, Tuple[int, str]], spare: Optional[int] = None
) -> None:
    """
    Linux only: reap the zombie children of this process.

    Args:
        table: The process table, as _linux_read_process_table returns it.
        spare: A child of this process not to reap, because a Popen object
            will.
    """
    me = os.getpid()
    for pid, (ppid, state) in table.items():
        # A zombie child of this process holds its PID until reaped, so
        # reaping by PID here can't hit another process.
        if ppid == me and state == "Z" and pid != spare:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def _linux_kill_tree_process(pid: int, tree: Set[int]) -> Optional[str]:
    """
    Linux only: send SIGKILL to a process found in this process's tree,
    unless its PID now belongs to a process outside the tree.  Where pidfds
    are unavailable, a PID reused in between can still redirect the signal.

    Args:
        pid: The process.
        tree: This process and its descendants, as last read.

    Returns:
        The start of the process's command line if the signal was sent,
        else None.
    """
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    except (OSError, AttributeError):
        # pidfd_open needs Linux 5.3, and some container seccomp profiles
        # refuse it.  os.pidfd_open itself only exists from Python 3.9 on,
        # and only in a Python built against Linux 5.3 or later headers.
        # Fall back to a plain kill after the same check below.
        pidfd = None
    try:
        # The pidfd refers to the process that had the PID when it was
        # opened, and the signal goes to that process or to none, never to a
        # later holder of the PID.  Checking the parent after opening it tells
        # whether that process is in the tree: a new process with the PID is
        # only accepted if its parent is in the tree, so a descendant too.
        stat = _linux_read_proc_stat(pid)
        if stat is None or stat[0] not in tree:
            return None
        try:
            with open(f"/proc/{pid}/cmdline", errors="replace") as f:
                name = f.read().replace("\0", " ").strip()[:100]
        except OSError:
            name = "?"
        if pidfd is None:
            os.kill(pid, signal.SIGKILL)
        else:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        return name
    except ProcessLookupError:
        return None
    except PermissionError:
        # Such as a descendant running a setuid program.  It stays alive, and
        # _linux_kill_descendants reports it once its timeout passes.
        return None
    finally:
        if pidfd is not None:
            os.close(pidfd)


# Windows support.
#
# On Windows, the testsuite's shell, make and DejaGnu come from MSYS2 (see
# _windows_resolve_msys2_root), and the script runs under either of two kinds
# of Python: a native one, such as python.org's or MSYS2's UCRT64 one, or one
# built against the MSYS2 runtime, such as MSYS2's /usr/bin/python3 (see
# IS_WINDOWS, IS_NATIVE_WINDOWS, IS_POSIX_WINDOWS and _is_msys2_python).  An
# MSYS2 Python uses POSIX paths, so the paths the script is given go through
# host_path.  A Cygwin Python is refused (see main).
#
# The functions and classes that only work on Windows carry a _windows_ or
# _Windows prefix, and those that only work under an MSYS2 Python an _msys2_
# one.
#
# "make check" runs in a job object (_WindowsJob), so that a timeout can
# kill the whole test process tree, and under a native Python also in a
# hidden console of its own, so that a test's Ctrl-C can't reach the script
# (see _WindowsLauncher).  The test programs find the HIP runtime under test
# through the DLL directory that _windows_set_dll_directory sets.  A packaged
# Python, whose children don't inherit it, is refused (see main).
#
# The Win32 structures below mirror winnt.h field for field: ctypes lays
# them out from _fields_, and SetInformationJobObject rejects a structure of
# the wrong size.


class _WindowsIoCounters(ctypes.Structure):
    """Win32 IO_COUNTERS, a member of JOBOBJECT_EXTENDED_LIMIT_INFORMATION."""

    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _WindowsJobObjectBasicLimitInformation(ctypes.Structure):
    """
    Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION, a member of
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION.  Its LimitFlags selects the limits
    a job enforces.
    """

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _WindowsJobObjectExtendedLimitInformation(ctypes.Structure):
    """
    Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION, which sets a job's limits
    through SetInformationJobObject.  JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is
    only accepted in this structure, not in the basic one alone.
    """

    _fields_ = [
        ("BasicLimitInformation", _WindowsJobObjectBasicLimitInformation),
        ("IoInfo", _WindowsIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# Win32 values, from winbase.h and winnt.h.
#
# A CreateProcess flag.  subprocess exports some of CreateProcess's flags,
# but not this one.
_CREATE_SUSPENDED = 0x00000004
# The information class of SetInformationJobObject that takes a
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION.
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
# A job limit flag: closing the last handle to the job kills its processes.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
# The OpenProcess access rights that AssignProcessToJobObject requires.
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


# Cached, since each WinDLL or CDLL call returns a new library object, and
# the prototypes set below only apply to the object they are set on.  A
# function instead of a module-level object, as kernel32 only exists on
# Windows.
@functools.lru_cache(maxsize=None)
def _windows_kernel32():
    """
    Windows only: kernel32, with the prototypes of the functions this script
    calls, from a native, Cygwin or MSYS2 Python.
    """
    if IS_NATIVE_WINDOWS:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    else:
        # A Cygwin or MSYS2 Python has no WinDLL, but x86-64 Windows has a
        # single calling convention, so CDLL calls into kernel32 just as well.
        kernel32 = ctypes.CDLL("kernel32.dll")
    handle = ctypes.c_void_p
    for name, argtypes, restype in (
        ("CreateJobObjectW", [ctypes.c_void_p, ctypes.c_wchar_p], handle),
        (
            "SetInformationJobObject",
            [handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
            ctypes.c_int,
        ),
        ("AssignProcessToJobObject", [handle, handle], ctypes.c_int),
        ("TerminateJobObject", [handle, ctypes.c_uint32], ctypes.c_int),
        ("OpenProcess", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], handle),
        ("CloseHandle", [handle], ctypes.c_int),
        ("SetDllDirectoryW", [ctypes.c_wchar_p], ctypes.c_int),
        (
            "GetCurrentPackageFullName",
            [ctypes.POINTER(ctypes.c_uint32), ctypes.c_wchar_p],
            ctypes.c_long,
        ),
        ("GetLastError", [], ctypes.c_uint32),
        ("GetModuleHandleW", [ctypes.c_wchar_p], handle),
    ):
        function = getattr(kernel32, name)
        function.argtypes = argtypes
        function.restype = restype
    return kernel32


def _windows_last_error() -> int:
    """
    Windows only: the Win32 error code of the last kernel32 call that failed.
    Under a Cygwin or MSYS2 Python it is best effort, and may come from a
    Win32 call the runtime made since.
    """
    if IS_NATIVE_WINDOWS:
        # Saved by ctypes right after the call, as kernel32 was loaded with
        # use_last_error.
        return ctypes.get_last_error()
    # ctypes only supports use_last_error where os.name is "nt", so this
    # reads the thread's error code later, in a call of its own.
    return _windows_kernel32().GetLastError()


def _windows_check_result(result: Optional[int], what: str) -> None:
    """
    Windows only: raise OSError if a kernel32 call reported failure.

    Args:
        result: The call's result, false on failure.
        what: The function called, for the message.
    """
    if not result:
        raise OSError(f"{what} failed with error {_windows_last_error()}")


def _windows_path(path: Union[str, Path]) -> str:
    """
    Windows only: the Windows form of a path, as Win32 functions take it.
    Under an MSYS2 Python, the path is in its POSIX form and is converted.

    Args:
        path: The path to convert.
    """
    if IS_NATIVE_WINDOWS:
        return str(path)
    return _msys2_cygpath("-w", path)


def _windows_process_handle(process: subprocess.Popen) -> int:
    """
    Native Windows Python only: the process handle of a process Popen
    started.

    Args:
        process: The process.
    """
    # Popen has no public way to get the handle CreateProcess returned, so
    # this reads CPython's private attribute for it, on Windows a Handle,
    # which converts to the handle's value.
    return int(process._handle)


class _WindowsJob:
    """
    Windows only: a job object holding a process and every process it
    starts, so that the whole tree can be killed at once.  Closing the job
    kills whatever is still in it.
    """

    def __init__(self) -> None:
        """Create an empty job, set to kill its processes when closed."""
        kernel32 = _windows_kernel32()
        self._handle = kernel32.CreateJobObjectW(None, None)
        _windows_check_result(self._handle, "CreateJobObjectW")
        info = _WindowsJobObjectExtendedLimitInformation()
        # Don't add JOB_OBJECT_LIMIT_BREAKAWAY_OK: the MSYS2 runtime, like
        # Cygwin's, then starts its children with CREATE_BREAKAWAY_FROM_JOB,
        # and they would escape the job.
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            _windows_check_result(
                kernel32.SetInformationJobObject(
                    self._handle,
                    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                ),
                "SetInformationJobObject",
            )
        except OSError:
            self.close()
            raise

    def add(self, process: subprocess.Popen) -> None:
        """
        Put a process in the job.  It must not have started any process of
        its own yet, since a process started before this wouldn't be in the
        job: a native Python creates it suspended and resumes it afterwards,
        and an MSYS2 Python starts it held at a gate (see _WindowsLauncher).

        Args:
            process: The process to add.
        """
        # A process started by a process in a job is in the job too, whatever
        # its session, which is what a process group can't do: expect puts
        # what it spawns in a session of its own.
        kernel32 = _windows_kernel32()
        if IS_NATIVE_WINDOWS:
            _windows_check_result(
                kernel32.AssignProcessToJobObject(
                    self._handle, _windows_process_handle(process)
                ),
                "AssignProcessToJobObject",
            )
            return
        # An MSYS2 Python has the process's Windows PID in /proc.  Popen only
        # returns once the child has exec'd the gate shell, and the MSYS2
        # runtime points winpid at the Windows process an exec creates before
        # then, so this is the shell's process, not the forked one's.
        winpid = int(Path(f"/proc/{process.pid}/winpid").read_text())
        handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, winpid
        )
        _windows_check_result(handle, "OpenProcess")
        try:
            _windows_check_result(
                kernel32.AssignProcessToJobObject(self._handle, handle),
                "AssignProcessToJobObject",
            )
        finally:
            kernel32.CloseHandle(handle)

    def kill(self) -> None:
        """Kill every process in the job, or warn if that fails."""
        if not _windows_kernel32().TerminateJobObject(self._handle, 1):
            logger.warning(
                f"{STATUS_WARN} TerminateJobObject failed with error "
                f"{_windows_last_error()}: the test's processes may be left "
                "running until the job is closed."
            )

    def close(self) -> None:
        """Close the job, killing whatever is still in it."""
        if self._handle:
            _windows_kernel32().CloseHandle(self._handle)
            self._handle = None


# Cached, as _windows_kernel32 is.
@functools.lru_cache(maxsize=None)
def _windows_ntdll():
    """Native Windows Python only: ntdll, with NtResumeProcess's prototype."""
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return ntdll


def _windows_resume_process(process: subprocess.Popen) -> None:
    """
    Native Windows Python only: resume a process created with
    CREATE_SUSPENDED.

    Args:
        process: The process to resume.
    """
    # Popen closes the handle to the process's initial thread, so resume the
    # whole process instead.  NtResumeProcess isn't in the Win32 API, but
    # has been exported by ntdll since Windows XP.
    status = _windows_ntdll().NtResumeProcess(_windows_process_handle(process))
    if status != 0:
        raise OSError(f"NtResumeProcess failed with status {status:#x}")


def _msys2_cygpath(option: str, path: Union[str, Path]) -> str:
    """
    MSYS2 Python only: convert a path with MSYS2's cygpath.

    Args:
        option: cygpath's conversion option, "-u" for the POSIX form or "-w"
            for the Windows one.
        path: The path to convert.

    Returns:
        The converted path.
    """
    # Run by its full path in this Python's own MSYS2, since PATH has no
    # /usr/bin when the script is started from cmd.exe instead of an MSYS2
    # shell.  Not from the directory of sys.executable, which in a venv is
    # the venv's own bin.
    result = subprocess.run(
        ["/usr/bin/cygpath", option, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _windows_relay_output(stream) -> None:
    """
    Used on Windows only, though nothing in it depends on Windows: copy a
    child's binary output stream to this process's stdout, line by line,
    until the stream closes.  If this process's stdout fails, keep reading
    the stream and drop what is read.

    Args:
        stream: The child's stdout, opened in binary mode.
    """
    for line in iter(stream.readline, b""):
        try:
            # Text written through sys.stdout, by print() say, would
            # otherwise come out after this line.
            sys.stdout.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        except (OSError, ValueError, AttributeError):
            # Such as a BrokenPipeError when this process's stdout is a
            # closed pipe, a ValueError when sys.stdout is closed, or an
            # AttributeError when it is None (pythonw) or has no buffer.
            # Stopping reading would leave the child blocked on a full pipe
            # until the timeout.
            pass
    stream.close()


class _Launcher:
    """
    Starts a command for _run_command, and kills and cleans up after it.
    This base class starts the command as it is, and kills only the command
    itself.  _LinuxLauncher and _WindowsLauncher can also kill every process
    the command starts.
    """

    def __init__(
        self,
        cmd: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        capture_output: bool,
    ) -> None:
        """
        Args:
            cmd: Command and arguments to execute.
            cwd: Working directory for command execution.
            env: Environment variables for command.
            capture_output: Whether to capture stdout/stderr.
        """
        # The command, as results and errors report it.
        self.cmd = cmd
        # What Popen starts, and its keyword arguments.  A subclass may adjust
        # both.
        self._argv = cmd
        pipe = subprocess.PIPE if capture_output else None
        self._popen_kwargs = {
            "cwd": str(cwd) if cwd else None,
            "env": env,
            "stdout": pipe,
            "stderr": pipe,
            "text": True,
        }

    def popen(self) -> subprocess.Popen:
        """Start the command."""
        return subprocess.Popen(self._argv, **self._popen_kwargs)

    def started(self, process: subprocess.Popen) -> None:
        """Finish starting the command, once popen has returned."""

    def wait(
        self, process: subprocess.Popen, timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Wait for the command to end, and return its stdout and stderr, each
        None unless captured.

        Raises:
            TimeoutExpired if `timeout` seconds pass first.
        """
        return process.communicate(timeout=timeout)

    def kill(self, process: subprocess.Popen) -> None:
        """Kill the command, and whatever else this launcher can kill."""
        process.kill()

    def close(self, process: Optional[subprocess.Popen]) -> None:
        """
        Clean up after the command, whether it ended normally or was killed.
        `process` is None if popen failed.
        """


class _LinuxLauncher(_Launcher):
    """
    Linux only: a _Launcher that kills the command's whole process tree, as
    a child subreaper (see _linux_kill_descendants), including what the
    command left running when it ended.  It reaps what was reparented to
    this process as the command runs.
    """

    def __init__(
        self,
        cmd: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        capture_output: bool,
    ) -> None:
        """
        See _Launcher.  This process is a child subreaper from here until
        close.
        """
        super().__init__(cmd, cwd, env, capture_output)
        # A session of its own keeps the terminal's signals, such as a
        # Ctrl-C typed there, away from the test processes, which this
        # script kills itself.
        self._popen_kwargs["start_new_session"] = True
        # The command's session of its own is not enough: expect's spawn
        # starts gdb in another session, out of the process group's reach.
        # As a subreaper, this process can find and kill the whole tree
        # instead.  Here instead of in popen, so that a failure isn't
        # reported as one to start the command.
        _linux_set_child_subreaper(True)

    def _kill_tree(self, process: Optional[subprocess.Popen], what: str) -> None:
        """
        Kill the command's tree, and log what was killed.

        Args:
            process: The command's process, or None if popen failed.
            what: How the log describes the processes, after "Killed N
                processes".
        """
        killed = _linux_kill_descendants(spare=process.pid if process else None)
        if killed:
            noun = "process" if len(killed) == 1 else "processes"
            logger.info(f"Killed {len(killed)} {noun} {what}: " + ", ".join(killed))

    def wait(
        self, process: subprocess.Popen, timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Wait in slices, and reap in between: a process reparented to this
        # one stays a zombie until reaped, holding its PID, which counts
        # against the process limits.  A long run leaves many, such as a
        # "sleep" for each time DejaGnu closes a spawned program (see
        # _run_command).  communicate can be called again after a timeout
        # without losing output.
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            step = 1.0
            if deadline is not None:
                step = max(0.0, min(step, deadline - time.monotonic()))
            try:
                return process.communicate(timeout=step)
            except subprocess.TimeoutExpired as e:
                if deadline is not None and time.monotonic() >= deadline:
                    e.timeout = timeout
                    raise
            _linux_reap_children(_linux_read_process_table(), spare=process.pid)

    def kill(self, process: subprocess.Popen) -> None:
        self._kill_tree(process, f"of the {self.cmd[0]} process tree")

    def close(self, process: Optional[subprocess.Popen]) -> None:
        # Kill what the command left behind, and reap what was reparented
        # to this process.
        try:
            self._kill_tree(process, f"left behind by {self.cmd[0]}")
        finally:
            _linux_set_child_subreaper(False)


class _WindowsLauncher(_Launcher):
    """
    Windows only: a _Launcher that finds the command's program as the child
    would, and can run the command in a job object (see _WindowsJob), so
    that killing the job kills the command's whole process tree, and under a
    native Python, in a hidden console of its own, with its output relayed
    (see own_console in _run_command).
    """

    def __init__(
        self,
        cmd: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        capture_output: bool,
        kill_process_tree: bool,
        own_console: bool,
    ) -> None:
        """
        See _Launcher, and also:

        Args:
            kill_process_tree: Whether to run the command in a job object.
            own_console: Whether to run the command in a console of its own,
                under a native Python.
        """
        if IS_NATIVE_WINDOWS:
            # CreateProcess looks for a plain program name in this process's
            # PATH, not in the PATH of the environment the child is given,
            # which is where setup_environment put the MSYS2 tools.  Resolve
            # it here.
            search_path = (env if env is not None else os.environ).get("PATH")
            resolved = shutil.which(cmd[0], path=search_path)
            if resolved:
                cmd = [resolved, *cmd[1:]]
        super().__init__(cmd, cwd, env, capture_output)
        # Whether the command's output comes back through a pipe, for a
        # thread to copy to this process's stdout (see
        # _windows_relay_output), and that thread once started.
        self._relay = False
        self._relay_thread: Optional[threading.Thread] = None
        # The job holding the command and every process it starts.
        self._job: Optional[_WindowsJob] = None
        # The gate's pipe: the gate shell reads from _gate_read, and this
        # process writes to _gate_write once the shell is in the job.
        self._gate_read: Optional[int] = None
        self._gate_write: Optional[int] = None
        if own_console and IS_NATIVE_WINDOWS:
            # When expect sends ^C (send_gdb "\003") to the native
            # rocgdb.exe, MSYS2's pty raises a console Ctrl-C event, which
            # reaches every process on rocgdb's console.  Sharing this
            # process's console, the testsuite would take this script down
            # with KeyboardInterrupt.
            self._popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            self._popen_kwargs["startupinfo"] = startupinfo
            # This process's console handles are no use in another console,
            # so the child gets no stdin, and output that is not captured
            # comes back through a pipe.
            self._popen_kwargs["stdin"] = subprocess.DEVNULL
            if not capture_output:
                self._popen_kwargs["stdout"] = subprocess.PIPE
                self._popen_kwargs["stderr"] = subprocess.STDOUT
                # Relayed output is copied as bytes, with no decoding to fail.
                self._popen_kwargs["text"] = False
                self._relay = True
        if kill_process_tree:
            self._job = _WindowsJob()
            # The command must not start anything before it is in the job, or
            # that would escape it.
            if IS_NATIVE_WINDOWS:
                self._popen_kwargs["creationflags"] = (
                    self._popen_kwargs.get("creationflags", 0) | _CREATE_SUSPENDED
                )
            else:
                # In an MSYS2 Python, subprocess starts processes through
                # the runtime's fork and exec, which have no suspended start.
                # Calling CreateProcess directly would leave converting the
                # environment and arguments, and waiting, to this script.
                # Instead the command starts held at a gate: a shell that
                # waits for a line on a pipe, written once the shell is in the
                # job, and then execs the command.  The MSYS2 runtime creates
                # an exec'd program as a new Windows process started by the
                # shell's, so it is in the job too.  If the pipe closes with
                # no line, the shell exits instead.
                self._gate_read, self._gate_write = os.pipe()
                gate = (
                    f"read -r _ <&{self._gate_read} || exit 125; "
                    f'exec {self._gate_read}<&-; exec "$@"'
                )
                # /usr/bin/sh for the same reason as /usr/bin/cygpath in
                # _msys2_cygpath.
                self._argv = ["/usr/bin/sh", "-c", gate, "sh", *cmd]
                self._popen_kwargs["pass_fds"] = (self._gate_read,)

    def _close_gate(self) -> None:
        """Close this process's end of the gate's pipe, if still open."""
        if self._gate_write is not None:
            os.close(self._gate_write)
            self._gate_write = None

    def popen(self) -> subprocess.Popen:
        try:
            return super().popen()
        finally:
            # The gate shell has its own copy of the read end by now, or
            # never will.
            if self._gate_read is not None:
                os.close(self._gate_read)
                self._gate_read = None

    def started(self, process: subprocess.Popen) -> None:
        if self._job is not None:
            # The command only runs once it is in the job.  If adding it
            # fails, it never runs: a native Python's stays suspended until
            # _run_command kills it, and an MSYS2 Python's gate shell sees the
            # pipe close with no line, and exits.
            try:
                self._job.add(process)
                if IS_NATIVE_WINDOWS:
                    _windows_resume_process(process)
                else:
                    os.write(self._gate_write, b"\n")
            finally:
                self._close_gate()
        if self._relay:
            self._relay_thread = threading.Thread(
                target=_windows_relay_output, args=(process.stdout,), daemon=True
            )
            self._relay_thread.start()

    def wait(
        self, process: subprocess.Popen, timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[str]]:
        if self._relay_thread is None:
            return super().wait(process, timeout)
        # The relay thread owns the pipe, so wait instead of communicating.
        # close joins the thread.
        process.wait(timeout=timeout)
        return None, None

    def kill(self, process: subprocess.Popen) -> None:
        if self._job is None:
            super().kill(process)
            return
        self._job.kill()
        # The job normally held the command, so this is a fallback for a
        # Ctrl-C that arrived before the command was added to the job, or
        # for adding it failing, which isn't expected: none of the failures
        # Microsoft documents applies to an empty job with no limits other
        # than JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, like this one.  Either way
        # the command is still held, suspended or at the gate, so it has
        # started nothing.  Under an MSYS2 Python, killing a process the job
        # already killed fails with EPERM, which Popen doesn't ignore.
        try:
            process.kill()
        except OSError:
            pass

    def close(self, process: Optional[subprocess.Popen]) -> None:
        self._close_gate()
        if self._job is not None:
            # Closing the job kills whatever the command left running.
            self._job.close()
        if self._relay_thread is not None:
            # Let the thread copy what the command wrote last.  With the job
            # closed, nothing can hold the pipe open any more, so this ends
            # promptly.  The timeout is for a command run without a job.
            self._relay_thread.join(timeout=10)


def _new_launcher(
    cmd: List[str],
    cwd: Optional[Path],
    env: Optional[Dict[str, str]],
    capture_output: bool,
    kill_process_tree: bool,
    own_console: bool,
) -> _Launcher:
    """The _Launcher for this platform and these _run_command arguments."""
    if IS_WINDOWS:
        return _WindowsLauncher(
            cmd, cwd, env, capture_output, kill_process_tree, own_console
        )
    if IS_LINUX and kill_process_tree:
        return _LinuxLauncher(cmd, cwd, env, capture_output)
    return _Launcher(cmd, cwd, env, capture_output)


def _run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    check: bool = True,
    error_msg: Optional[str] = None,
    timeout: Optional[int] = None,
    kill_process_tree: bool = False,
    own_console: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command with consistent error handling.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory for command execution.
        env: Environment variables for command.
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        error_msg: Custom error message on failure.
        timeout: Optional wall-clock timeout in seconds; raises
            subprocess.TimeoutExpired if exceeded.
        kill_process_tree: On TimeoutExpired, kill the command along with
            its descendants (e.g. runtest/expect/gdb spawned by `make`) so they
            are not orphaned, and kill whatever is left when the command
            ends.  On Windows they run in a job object (see _WindowsJob),
            and on Linux the script tracks them as its descendants (see
            _linux_kill_descendants).  Elsewhere only the command itself is
            killed.
        own_console: In a native Windows Python, run the command in a hidden
            console of its own, with no stdin, so that a console Ctrl-C event
            raised inside it doesn't reach this process.  Output that is not
            captured is relayed to this process's stdout.  Under an MSYS2
            Python or on Linux it has no effect.  An MSYS2 Python survives
            such an event, whether it was started from an MSYS2 shell or
            from cmd.exe.

    Returns:
        CompletedProcess result.

    Exits:
        Code 1 if check=True, command exits with non-zero status, and error_msg is provided.

    Raises:
        CalledProcessError if check=True, command exits with non-zero status, and error_msg is None.
        TimeoutExpired if timeout is set and exceeded.
    """
    launcher = _new_launcher(
        cmd, cwd, env, capture_output, kill_process_tree, own_console
    )
    cmd = launcher.cmd
    process = None
    try:
        try:
            process = launcher.popen()
        except OSError as e:
            if error_msg:
                _log_error_and_exit(f"{error_msg}: {e}")
            else:
                _log_error_and_exit(f"Failed to execute {cmd[0]!r}: {e}")
        try:
            launcher.started(process)
            out, err = launcher.wait(process, timeout)
        except BaseException:
            # Cover TimeoutExpired and any other escaping exception
            # (KeyboardInterrupt, OSError mid-read, etc.) so the make /
            # runtest / expect / gdb process tree is never orphaned.
            launcher.kill(process)
            try:
                out, err = launcher.wait(process, None)
            except Exception:
                out, err = None, None
            raise
    finally:
        # A launcher that kills the whole process tree kills it both in the
        # except above and here, on purpose.  There the tree must die before
        # the command is waited for, since on a timeout the command wouldn't
        # end on its own.  Here it catches what a command that ended normally
        # left running.  A normal run of the testsuite leaves a "sleep 5"
        # behind: DejaGnu's close_wait_program cancels its kill watchdog by
        # killing the watchdog's shell, but not the sleep that shell is
        # running.
        launcher.close(process)
    result = subprocess.CompletedProcess(cmd, process.returncode, out, err)
    if check and process.returncode != 0:
        exc = subprocess.CalledProcessError(process.returncode, cmd, out, err)
        if error_msg:
            _log_error_and_exit(f"{error_msg}: {exc}")
        raise exc
    return result


def _read_file_lines(file_path: Path, error_context: str = "") -> List[str]:
    """
    Read file lines with error handling.

    Args:
        file_path: Path to file to read.
        error_context: Context for error message.

    Returns:
        List of stripped lines from file.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f]
    except OSError as e:
        _log_error_and_exit(
            f"Failed to read {file_path}: {e}"
            f"{'. ' + error_context if error_context else ''}"
        )


def _extract_test_file_from_name(test_name: str) -> str:
    """
    Extract test file path from full test name.

    Args:
        test_name: Full test name (e.g., "gdb.rocm/foo.exp: test description").

    Returns:
        Test file path without description (e.g., "gdb.rocm/foo.exp").
    """
    return test_name.split(": ", 1)[0]


def load_ignore_list_from_json(json_path: Path) -> Dict[str, List[str]]:
    """
    Load test ignore list from JSON file.

    Args:
        json_path: Path to JSON file containing ignore list.

    Returns:
        Dictionary mapping compiler labels to lists of ignored test paths.
        Returns empty dict if file doesn't exist.

    Exits:
        Code 1 if JSON file exists but is malformed.
    """
    if not json_path.exists():
        logger.warning(
            f"Ignore list file not found: {json_path}. Using empty ignore list."
        )
        return {}

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        _log_error_and_exit(f"Failed to read {json_path}: {e}")
    except json.JSONDecodeError as e:
        _log_error_and_exit(f"Failed to parse JSON from {json_path}: {e}")

    if not isinstance(data, dict):
        _log_error_and_exit(
            f"Invalid ignore list format in {json_path}: expected a JSON object"
        )

    # Validate structure: each key should map to a list of strings.
    for key, value in data.items():
        if not isinstance(value, list):
            _log_error_and_exit(
                f"Invalid ignore list format in {json_path}: "
                f"key '{key}' should map to a list"
            )
        if not all(isinstance(item, str) for item in value):
            _log_error_and_exit(
                f"Invalid ignore list format in {json_path}: "
                f"all items in list for key '{key}' should be strings"
            )

    return data


def select_toolchains(requested: Optional[List[str]]) -> List[str]:
    """
    Resolve the list of toolchain identifiers to run.

    When no toolchain is requested, all of them run in definition order.
    Repeated names are dropped while keeping the first occurrence so the same
    toolchain never runs twice.

    Args:
        requested: Toolchain identifiers from --toolchain, or None when the
            flag was omitted.

    Returns:
        Toolchain identifiers to run, in order, with duplicates removed.
    """
    names = requested if requested is not None else list(TOOLCHAINS.keys())
    return list(dict.fromkeys(names))


def _non_negative_int(value: str) -> int:
    """
    Validate argument is a non-negative integer.

    Args:
        value: String value from command line.

    Returns:
        Non-negative integer value.

    Raises:
        argparse.ArgumentTypeError: If value is negative.
    """
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a non-negative integer")
    return ivalue


def _timeout_value(value: str) -> int:
    """
    Validate timeout is a positive integer (seconds).

    Args:
        value: String value from command line.

    Returns:
        Integer timeout value.

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"timeout must be a positive integer (got {ivalue})"
        )
    return ivalue


def _parallel_fraction(value: str) -> float:
    """
    Validate argument is a float in the range (0.0, 1.0].

    Args:
        value: String value from command line.

    Returns:
        Float fraction value.

    Raises:
        argparse.ArgumentTypeError: If value is not in (0.0, 1.0].
    """
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid number")
    if not (0.0 < fvalue <= 1.0):
        raise argparse.ArgumentTypeError(
            f"--fraction must be in the range (0.0, 1.0], got {fvalue}"
        )
    return fvalue


def _positive_nonzero_int(value: str) -> int:
    """
    Validate argument is a positive non-zero integer.

    Args:
        value: String value from command line.

    Returns:
        Positive non-zero integer value.

    Raises:
        argparse.ArgumentTypeError: If value is not positive and non-zero.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive non-zero integer")
    return ivalue


def _load_emulator_support() -> types.ModuleType:
    """
    Import the rocjitsu emulator support module, needed only for --emulate.

    Imported lazily so this file keeps working as a single script: hardware
    runs never reach this, and an emulated run reports exactly what is missing
    instead of failing at import time. Returns the module, and exits 1 if it is
    not beside this script.
    """
    try:
        import rocjitsu_emulator
    except ImportError as e:
        _log_error_and_exit(
            "--emulate needs rocjitsu_emulator.py next to this script in "
            f"{Path(__file__).resolve().parent}, but it could not be "
            f"imported: {e}"
        )
    return rocjitsu_emulator


def host_path(value: str) -> Path:
    """
    Turn a path given on the command line or in the environment into one this
    Python can use.  In an MSYS2 Python, a Windows path such as
    C:\\rocm becomes its POSIX form, /c/rocm.

    Args:
        value: The path as given.

    Returns:
        The path.

    Raises:
        argparse.ArgumentTypeError: If the path is empty, or cygpath can't
            convert it.
    """
    # Path("") is the current directory, which a forgotten or mistyped value
    # would otherwise quietly stand for.  As an option's type=, argparse
    # reports an ArgumentTypeError as a usage error after the option's name,
    # not as a traceback.
    if not value:
        raise argparse.ArgumentTypeError("the path is empty")
    # Such a Python takes no drive letters, and the colon after a drive
    # letter would split a PATH entry built from the path.  cygpath -u
    # leaves a path that is already POSIX as it is.
    if IS_POSIX_WINDOWS:
        try:
            return Path(_msys2_cygpath("-u", value))
        except subprocess.CalledProcessError as e:
            # cygpath's own message already says what failed.
            raise argparse.ArgumentTypeError(
                e.stderr.strip() or f"cygpath can't convert {value!r}"
            ) from None
    return Path(value)


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the ROCgdb test suite.

    Returns:
        Parsed arguments with validated paths, timeout, and retry settings.

    Exits:
        Code 1 if arguments are invalid or paths don't exist.
    """
    parser = argparse.ArgumentParser(
        description="Run ROCgdb test suite with different compilers.",
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=100
        ),
        epilog="""Examples:
  python %(prog)s --testsuite-dir /path/to/testsuite --rocgdb-bin /path/to/rocgdb
  python %(prog)s --rocgdb-bin /path/to/rocgdb
  python %(prog)s --tests gdb.base/break.exp gdb.base/call-ar-st.exp
  python %(prog)s --gpu-tests
  python %(prog)s --cpu-tests
  python %(prog)s --parallel
  python %(prog)s --parallel -j4
  python %(prog)s --parallel --fraction 0.5
  python %(prog)s --group-results
  python %(prog)s --default-timeout 300
  python %(prog)s --retry-timeout 600
  python %(prog)s --max-failed-retries 2
  python %(prog)s --optimization="-O0"
  python %(prog)s --runtestflags="--target_board=hip -debug"
  python %(prog)s --no-xfail
  python %(prog)s --quiet
  python %(prog)s --skip-failed-test-log
  python %(prog)s --output-ignore-list-file custom_ignore_list.json
  python %(prog)s --one-by-one --tests gdb.rocm/foo.exp
  python %(prog)s --one-by-one --sanity-check --gpu-tests
  python %(prog)s --timing --gpu-tests
  python %(prog)s --gpu-corefile-tests
  python %(prog)s --toolchain llvm
  python %(prog)s --emulate mi350x --gpu-tests
  python %(prog)s --emulate gfx942 --one-by-one --tests gdb.rocm/simple.exp

        """,
    )

    parser.add_argument(
        "--skip-failed-test-log",
        action="store_false",
        dest="dump_failed_test_log",
        help="Skip dumping gdb.log to the console for failed tests. Default is to dump the log.",
    )
    parser.add_argument(
        "--group-results",
        action="store_true",
        help="Group test results in summary output. Default is off.",
    )
    parser.add_argument(
        "--max-failed-retries",
        type=_non_negative_int,
        default=3,
        help="Maximum number of times to retry failed tests. Default is 3.",
    )
    parser.add_argument(
        "--no-xfail",
        action="store_true",
        help="Do not use ignore lists for failed tests. Default is to use the ignore lists.",
    )
    parser.add_argument(
        "--retry-ignored-tests",
        action="store_true",
        help="Retry tests that are on the ignore list when they fail. "
        "Default is off: ignored tests that fail on the first run are not retried.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="",
        help="Optimization level to pass to compiler (e.g., -O0, -Os, -Og).",
    )
    parser.add_argument(
        "--toolchain",
        nargs="+",
        default=None,
        metavar="NAME",
        choices=list(TOOLCHAINS.keys()),
        help="Toolchain(s) to run tests against. Choices: "
        f"{', '.join(TOOLCHAINS.keys())}. Accepts one or more values; the flag "
        "requires at least one. If omitted, all toolchains run and the results "
        "are compared. Passing a single toolchain (e.g. --toolchain llvm) runs "
        "only that toolchain and skips the cross-compiler comparison.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel. Default is off."
    )
    parallel_jobs_group = parser.add_mutually_exclusive_group()
    parallel_jobs_group.add_argument(
        "-j",
        "--jobs",
        type=_positive_nonzero_int,
        default=None,
        metavar="N",
        help="Number of parallel jobs for make (e.g., -j4). Only valid with --parallel. "
        "Mutually exclusive with --fraction. "
        "If --parallel is used without -j or --fraction, defaults to os.cpu_count().",
    )
    parallel_jobs_group.add_argument(
        "-f",
        "--fraction",
        type=_parallel_fraction,
        default=None,
        metavar="F",
        help="Fraction of os.cpu_count() to use as the number of parallel jobs "
        "(e.g., 0.5 for half the CPUs). Must be in the range (0.0, 1.0]. "
        "Only valid with --parallel. Mutually exclusive with -j/--jobs. "
        "The resulting job count is max(1, int(cpu_count * F)).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not output configure/testsuite commands. Default is off.",
    )
    parser.add_argument(
        "--rocgdb-bin",
        type=host_path,
        help="Path to the ROCgdb executable. Validated for existence when "
        "supplied. Default: <rocm_dir>/bin/rocgdb (rocgdb.exe on Windows), "
        "where rocm_dir is auto-discovered (see --try-rocm-path).",
    )
    parser.add_argument(
        "--msys2-root",
        type=host_path,
        help="Windows only: the MSYS2 installation (the directory holding "
        "usr/bin) whose sh, make and DejaGnu run the testsuite. "
        "Default: the MSYS2 this script runs from, when started from an "
        "MSYS2 shell, else C:/msys64. Under MSYS2's own Python, only that "
        "Python's installation is accepted, and is the default.",
    )
    parser.add_argument(
        "--runtestflags",
        type=str,
        default="",
        help="Additional flags for RUNTESTFLAGS (e.g., '--target_board=hip -debug').",
    )

    # Create mutually exclusive group for test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--tests",
        nargs="+",
        default=None,
        help="List of tests to run (individual .exp files or directories). "
        "For directories, the /*.exp suffix is automatically added. "
        "Mutually exclusive with --gpu-tests, --cpu-tests, and --gpu-corefile-tests. "
        "Default (if no test option specified) is gdb.rocm/*.exp and gdb.dwarf2/*.exp",
    )
    test_group.add_argument(
        "--gpu-tests",
        action="store_true",
        help="Run GPU-specific tests (gdb.rocm directory only). "
        "Mutually exclusive with --tests, --cpu-tests, and --gpu-corefile-tests.",
    )
    test_group.add_argument(
        "--cpu-tests",
        action="store_true",
        help="Run all CPU tests (all testsuite directories except gdb.rocm). "
        "Automatically discovers all gdb.* directories with .exp files. "
        "Mutually exclusive with --tests, --gpu-tests, and --gpu-corefile-tests.",
    )
    test_group.add_argument(
        "--gpu-corefile-tests",
        action="store_true",
        help="Run GPU system-generated corefile tests (gdb.rocm/runtime-core*.exp). "
        "New tests are picked up automatically when they match that pattern. "
        "Mutually exclusive with --tests, --gpu-tests, and --cpu-tests.",
    )

    parser.add_argument(
        "--testsuite-dir",
        type=host_path,
        help="Path to the GDB testsuite directory. Validated for existence "
        "when supplied. Default: <rocm_dir>/tests/rocgdb/gdb/testsuite, "
        "where rocm_dir is auto-discovered (see --try-rocm-path).",
    )
    parser.add_argument(
        "--default-timeout",
        type=_timeout_value,
        default=None,
        help="Default timeout in seconds for all test runs from the start. "
        "If not specified, no timeout is set initially.",
    )
    parser.add_argument(
        "--retry-timeout",
        type=_timeout_value,
        default=100,
        help="Timeout in seconds for timeout-based retry runs only. "
        "Only applies when retrying tests that timed out. Default is 100.",
    )
    parser.add_argument(
        "--emulate",
        type=str,
        default=None,
        metavar="GPU",
        help="Run the tests on an emulated GPU (rocjitsu) instead of real "
        "hardware; no AMD GPU needed. GPU is a mirage profile (mi300x, "
        "mi350x, mi450x) or the gfx target it emulates (gfx942, gfx950, "
        "gfx1250), which the ROCm tree must have been built for. Raises "
        "--default-timeout, since emulated tests are slower. Not supported "
        "on Windows.",
    )
    parser.add_argument(
        "--try-rocm-path",
        action="store_true",
        help="Try to use ROCM_PATH environment variable to determine the ROCm tree root. "
        "If ROCM_PATH is set and valid, it takes precedence over OUTPUT_ARTIFACTS_DIR and script location detection. "
        "If ROCM_PATH is not set, falls back to OUTPUT_ARTIFACTS_DIR or script location detection. "
        "If ROCM_PATH is invalid, issues an error.",
    )
    parser.add_argument(
        "--check-type",
        type=str,
        choices=["check", "check-read1", "check-readmore"],
        default="check",
        help="Which GDB testsuite target to invoke. 'check' is the normal run. "
        "'check-read1' runs under an LD_PRELOAD shim that forces read(2) to "
        "return 1 byte at a time (stress-tests GDB's incremental I/O parsing). "
        "'check-readmore' forces large reads (the opposite extreme). These "
        "are upstream GDB testsuite modes and can surface latent testcase "
        "regex bugs that 'check' alone misses. Default: 'check'.",
    )
    parser.add_argument(
        "--ignore-list-file",
        type=host_path,
        required=False,
        metavar="PATH",
        help="Path to JSON file containing test ignore lists. "
        "If not specified, uses rocgdb_ignore_list.json in the same directory as this script. "
        "This option requires a value.",
    )
    parser.add_argument(
        "--output-ignore-list-file",
        type=host_path,
        required=False,
        metavar="PATH",
        help="Path to output JSON file for generating a new ignore list from test failures. "
        "When specified, the testsuite runs normally and all failures are written to this file "
        "at the end of the run. Common failures go into 'Generic' category, compiler-specific "
        "failures go into compiler-specific categories (e.g., 'GCC', 'LLVM'). "
        "This option requires a value.",
    )
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="Run each test in its own `make check` invocation and save its gdb.log "
        "under --one-by-one-log-dir. Each invocation is subject to a wall-clock limit; "
        "see --one-by-one-test-timeout. Mutually exclusive with --parallel.",
    )
    parser.add_argument(
        "--one-by-one-log-dir",
        type=host_path,
        default=None,
        metavar="PATH",
        help="Root directory for per-test gdb.log captures in --one-by-one mode. "
        "Default: <testsuite_dir>/one_by_one_logs.",
    )
    parser.add_argument(
        "--one-by-one-test-timeout",
        type=_positive_nonzero_int,
        default=None,
        metavar="SECS",
        help="Wall-clock timeout (seconds) per individual `make check` invocation "
        "in --one-by-one mode. Defaults to 3600.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Before each one-by-one test, run a known-good HIP program "
        "(built once per run from gdb.rocm/simple.cpp) and bail out if the GPU stops "
        "responding. If the probe times out or exits non-zero, the system is "
        "declared unreliable: a dmesg.log is captured under the one-by-one log "
        "dir and the run aborts. Requires --one-by-one.",
    )
    parser.add_argument(
        "--sanity-check-timeout",
        type=_positive_nonzero_int,
        default=None,
        metavar="SECS",
        help="Wall-clock timeout (seconds) for each --sanity-check probe. "
        f"Defaults to {SANITY_CHECK_DEFAULT_TIMEOUT}. Requires --sanity-check.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Record each test's wall-clock duration and write rocgdb_timing.log "
        "grouped by compiler and directory, sorted from longest to shortest "
        "running test. Implies --one-by-one. The recorded duration is the "
        "maximum time of a successful run; tests that never passed show N/A.",
    )
    parser.add_argument(
        "--timing-log-file",
        type=host_path,
        default=None,
        metavar="PATH",
        help="Output path for the timing log. Default: "
        "<testsuite_dir>/rocgdb_timing.log. Requires --timing.",
    )

    args = parser.parse_args()

    # --timing is a specialization of one-by-one mode; enable it implicitly so
    # a single flag turns on the timing run. Set before the one-by-one /
    # parallel validation below so --timing --parallel is rejected too.
    if args.timing:
        args.one_by_one = True
    if args.timing_log_file is not None and not args.timing:
        _log_error_and_exit("--timing-log-file requires --timing.")

    # Validate that -j/--jobs and --fraction are only used with --parallel.
    if args.jobs is not None and not args.parallel:
        _log_error_and_exit(
            "-j/--jobs can only be specified when --parallel is provided."
        )
    if args.fraction is not None and not args.parallel:
        _log_error_and_exit(
            "-f/--fraction can only be specified when --parallel is provided."
        )

    # Validate --one-by-one constraints.
    if args.one_by_one and args.parallel:
        _log_error_and_exit("--one-by-one cannot be combined with --parallel.")
    if (
        args.one_by_one_log_dir is not None or args.one_by_one_test_timeout is not None
    ) and not args.one_by_one:
        _log_error_and_exit(
            "--one-by-one-log-dir / --one-by-one-test-timeout require --one-by-one."
        )

    # rocjitsu has no Windows KMD, so it can't emulate a GPU there.  Ignoring
    # the option instead would run the tests on the real GPU while the command
    # line says they are emulated.
    if args.emulate is not None and IS_WINDOWS:
        _log_error_and_exit("--emulate is not supported on Windows.")

    # An emulated GPU can't wedge the way this probe catches: it is built fresh
    # per test, and dmesg says nothing about a userspace device. Ignore rather
    # than reject, so one command line serves hardware and the emulator.
    if args.emulate is not None and args.sanity_check:
        logger.warning(
            f"{STATUS_WARN} --emulate: ignoring --sanity-check (and "
            "--sanity-check-timeout). The probe detects a wedged GPU; an "
            "emulated one is built fresh for every test."
        )
        args.sanity_check = False
        # Clearing the timeout too, or the "--sanity-check-timeout requires
        # --sanity-check" check below would reject a command line we just said
        # we were ignoring.
        if args.sanity_check_timeout is not None:
            logger.debug(
                "--emulate: also clearing --sanity-check-timeout "
                f"({args.sanity_check_timeout}s), since --sanity-check is ignored."
            )
        args.sanity_check_timeout = None

    # Validate --sanity-check constraints. The probe is only meaningful in
    # one-by-one mode, where it runs between per-test `make check` invocations.
    # Check the --parallel conflict first so the message points at the real
    # incompatibility instead of suggesting the user add --one-by-one (which
    # would then trip the mutual-exclusion error above).
    if args.sanity_check and args.parallel:
        _log_error_and_exit(
            "--sanity-check cannot be combined with --parallel "
            "(it runs in --one-by-one mode)."
        )
    if args.sanity_check and not args.one_by_one:
        _log_error_and_exit("--sanity-check requires --one-by-one.")
    if args.sanity_check_timeout is not None and not args.sanity_check:
        _log_error_and_exit("--sanity-check-timeout requires --sanity-check.")
    # Resolve the effective probe timeout now that the "set without
    # --sanity-check" case has been rejected, so downstream code sees a value.
    if args.sanity_check_timeout is None:
        args.sanity_check_timeout = SANITY_CHECK_DEFAULT_TIMEOUT

    # Validate paths independently if provided. Either flag may be supplied
    # alone; whichever is omitted is auto-discovered in _resolve_rocm_paths.
    if args.testsuite_dir is not None:
        args.testsuite_dir = validate_path(args.testsuite_dir, is_dir=True)
    if args.rocgdb_bin is not None:
        args.rocgdb_bin = validate_path(args.rocgdb_bin, is_file=True)

    # Reject an unknown emulated GPU here rather than after the environment is
    # set up, and raise the timeout: emulated tests run far slower than on
    # hardware, where dejagnu's default per-testcase timeout is enough. Left at
    # that default the slow tests fail for want of time rather than for a real
    # reason.
    if args.emulate is not None:
        emulator_support = _load_emulator_support()
        try:
            emulator_support.normalize_profile(args.emulate)
        except emulator_support.EmulatorError as e:
            _log_error_and_exit(str(e))
        if args.default_timeout is None:
            args.default_timeout = emulator_support.DEFAULT_TIMEOUT
            logger.info(
                "--emulate: using a default timeout of "
                f"{args.default_timeout}s (override with --default-timeout)"
            )
        # Under emulation the job count is a memory question, not just a CPU
        # one, so the emulator decides what an unspecified --jobs should be
        # rather than leaving it at "as many as there are CPUs". --fraction is
        # resolved to a concrete count here and fed through as a request, so the
        # same memory check (and OOM warning) covers it too, so downstream
        # code then only has to deal with args.jobs.
        if args.parallel:
            cpu_count = os.cpu_count() or 1
            requested = args.jobs
            if args.fraction is not None:
                requested = max(1, int(cpu_count * args.fraction))
            advice = emulator_support.parallel_jobs(requested, cpu_count)
            args.jobs = advice.jobs
            args.fraction = None
            if advice.warn:
                logger.warning(f"{STATUS_WARN} {advice.message}")
            elif advice.message:
                logger.info(advice.message)

    # Adjust retry-timeout if default-timeout is larger.
    if args.default_timeout is not None and args.default_timeout > args.retry_timeout:
        logger.warning(
            f"--retry-timeout is being overridden from {args.retry_timeout}s to "
            f"{args.default_timeout}s because --default-timeout is larger."
        )
        args.retry_timeout = args.default_timeout

    return args


class TestResults:
    """Class to store and manage test results across multiple compiler runs."""

    def __init__(self) -> None:
        """Initialize test results storage with empty data structures."""
        # Flag to control whether to group results in output.
        self.group_results: bool = False

        # Our main data structure for storing results. The mapping goes like this:
        #
        # - 1 compiler label maps to N labels (PASS, FAIL etc).
        # - 1 label maps to N test files (gdb.rocm/simple.exp for example).
        # - 1 test file maps to N test descriptions (the complete test
        #   line output by dejagnu).
        self.test_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Track currently failed tests by compiler_label.
        self.failed_tests: Dict[str, Set[str]] = defaultdict(set)

        # Track flaky tests (failed initially but passed on retry) by compiler_label.
        self.flaky_tests: Dict[str, Set[str]] = defaultdict(set)

        # Track harness-level errors (ERROR lines with no preceding
        # "Running ....exp") per compiler_label. Replaced on each
        # update_results call so it reflects the most recent run's state.
        self.harness_errors: Dict[str, List[str]] = defaultdict(list)

        # Per-test timings for --timing mode, keyed by compiler_label then
        # test file. Value is the maximum duration (seconds) of a *successful*
        # run, or None if the test ran but never passed. Populated by
        # _execute_one_by_one via record_timing.
        self.timings: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)

    def record_timing(
        self, compiler_label: str, test_file: str, duration: float, passed: bool
    ) -> None:
        """
        Record a per-test wall-clock duration for --timing mode.

        For a successful run, keep the maximum duration seen across all runs
        (both compilers' retries), since the goal is spotting slow tests and
        worst-case is the useful signal. A failed run never overwrites a
        recorded success; a test that only ever fails is stored as None so the
        report can show N/A.

        Args:
            compiler_label: Compiler identifier (e.g., "GCC", "LLVM").
            test_file: Test file path (e.g., "gdb.rocm/simple.exp").
            duration: Wall-clock duration of this run in seconds.
            passed: Whether this run completed successfully.

        Returns:
            None
        """
        per_compiler = self.timings[compiler_label]
        if passed:
            prev = per_compiler.get(test_file)
            if prev is None or duration > prev:
                per_compiler[test_file] = duration
        elif test_file not in per_compiler:
            # Record the failure only if no successful timing exists yet.
            per_compiler[test_file] = None

    def cleanup_old_entries(self, compiler_label: str, tests: List[str]) -> None:
        """
        Remove stale test results before retry runs to prevent duplicate entries.

        Args:
            compiler_label: Compiler identifier (e.g., "GCC", "LLVM").
            tests: List of test file names to remove.

        Returns:
            None
        """
        # Case: first run for this compiler — nothing to clean. Use
        # `in` rather than indexing to avoid the defaultdict factory
        # materializing a phantom empty entry for the compiler.
        if compiler_label not in self.test_data:
            return

        logger.info(f"Retry run. Cleaning dictionary entries for {' '.join(tests)}.")

        # Clear the failed tests set for this compiler label.
        self.failed_tests[compiler_label].clear()

        # Clear tests entries from the main test results database for this
        # compiler label.
        for test in tests:
            for category in TEST_CATEGORIES:
                category_results = self.test_data[compiler_label].get(category)
                if category_results and test in category_results:
                    del category_results[test]

    def update_results(
        self, compiler_label: str, tests: List[str], results_file: Path
    ) -> None:
        """
        Parse and update test results from a DejaGnu results file.

        A single pass over the file extracts both result lines (PASS/FAIL/...)
        and ERROR lines, attributing each ERROR to the most recently seen
        `Running ....exp` line. ERRORs without a preceding Running line are
        recorded as harness-level errors.

        Args:
            compiler_label: Compiler name (e.g., "GCC", "LLVM").
            tests: List of test file names to clear before updating.
            results_file: Path to results file with lines like "STATUS: test_description".

        Returns:
            None
        """
        self.cleanup_old_entries(compiler_label, tests)

        new_errors: Dict[str, List[str]] = defaultdict(list)
        orphan_errors: List[str] = []
        current_test_file: Optional[str] = None

        for line in _read_file_lines(results_file, "during results update"):
            if not line:
                continue

            running_match = RUNNING_EXP_RE.search(line)
            if running_match:
                segments = os.path.normpath(running_match.group(1)).split(os.path.sep)
                current_test_file = "/".join(segments[-2:])

            if line.startswith("ERROR:"):
                msg = line[len("ERROR:") :].strip()
                if current_test_file is None:
                    orphan_errors.append(msg)
                else:
                    new_errors[current_test_file].append(f"{current_test_file}: {msg}")
                continue

            match = RESULT_LINE_RE.match(line)
            if match:
                status, test_name = match.groups()
                test_file = _extract_test_file_from_name(test_name)
                is_timeout = TIMEOUT_RE.search(test_name)

                # Add the entry to our test results database.
                self.test_data[compiler_label][status][test_file].append(test_name)

                # For timeouts, we add a duplicate since we want to
                # explicitly list tests that ran into timeouts.
                if is_timeout:
                    self.test_data[compiler_label]["TIMEOUT"][test_file].append(
                        test_name
                    )

                # Track the current list of failed tests. Timeouts on
                # statuses other than FAIL/UNRESOLVED (e.g. a KFAIL that
                # happened to time out) are harmless and intentionally
                # excluded.
                if status in ["FAIL", "UNRESOLVED"]:
                    self.failed_tests[compiler_label].add(test_file)

        # Merge ERROR entries into the existing dict so that ERRORs for tests
        # not being rerun are preserved (cleanup_old_entries has already
        # cleared the slots for the tests in the current run).
        for test_file, messages in new_errors.items():
            self.test_data[compiler_label]["ERROR"][test_file] = messages
            self.failed_tests[compiler_label].add(test_file)

        # Replace harness errors with this run's orphans.
        self.harness_errors[compiler_label] = orphan_errors

    def get_failed_tests(self, compiler_label: str) -> List[str]:
        """
        Get currently failed tests for a compiler.

        Args:
            compiler_label: Compiler identifier.

        Returns:
            List of failed test file paths.
        """
        # Use .get to avoid materializing a phantom empty entry for
        # compilers that ran without any failures.
        return list(self.failed_tests.get(compiler_label, set()))

    def mark_flaky_tests(self, compiler_label: str, test_files: Set[str]) -> None:
        """
        Mark tests as flaky (failed initially but passed on retry).

        Args:
            compiler_label: Compiler identifier (e.g., "GCC", "LLVM").
            test_files: Set of test file paths that were flaky.

        Returns:
            None
        """
        if not test_files:
            return

        # Add to flaky tests set.
        self.flaky_tests[compiler_label].update(test_files)

        # Add to test_data under FLAKY category with marker.
        for test_file in test_files:
            flaky_description = f"{test_file}: initially failed, passed on retry"
            self.test_data[compiler_label]["FLAKY"][test_file].append(flaky_description)

        logger.info(
            f"{STATUS_FLAKY} {len(test_files)} test(s) became stable after retry for {compiler_label}"
        )

    def print_all_summaries(self) -> None:
        """
        Print test summaries for all compilers.

        Returns:
            None
        """
        if not self.test_data:
            logger.info(f"\n{STATUS_FAIL} No test results to display.")
            return

        print_section("COMBINED GDB TESTSUITE ANALYSIS")

        for compiler_label in sorted(self.test_data.keys()):
            self._print_summary(compiler_label, self.test_data[compiler_label])

        self.print_comparison()

    def _print_summary(
        self, compiler_label: str, details: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """
        Print test summary for a single compiler.

        Args:
            compiler_label: Compiler identifier.
            details: Mapping from status category to test files and descriptions.

        Returns:
            None
        """
        print_section(f"{compiler_label}", border_char="-", inline=True)
        for cat, prefix_indicator in CATEGORY_DISPLAY.items():
            count = sum(len(tests) for tests in details.get(cat, {}).values())

            if count > 0:
                header = f"  {prefix_indicator} {cat}: {count}"
                logger.info(header)

            # Print details for non-PASS tests.
            if count > 0 and cat != "PASS":
                if self.group_results:
                    self._print_grouped_tests(details[cat])
                else:
                    for test_file in details[cat]:
                        for test_name in details[cat][test_file]:
                            logger.info(f"          {test_name}")

        harness_errors = self.harness_errors.get(compiler_label, [])
        if harness_errors:
            logger.info(f"  {STATUS_FAIL} HARNESS ERRORS: {len(harness_errors)}")
            for msg in harness_errors:
                logger.info(f"          {msg}")

    def _print_grouped_tests(self, test_list: Dict[str, List[str]]) -> None:
        """
        Print tests grouped by directory, file, and description.

        Args:
            test_list: Mapping of test file paths to descriptions.

        Returns:
            None
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for file, descriptions in test_list.items():
            test_dir, test_file = os.path.split(file)
            prefix_len = len(file) + 2
            for test_description in descriptions:
                grouped[test_dir][test_file].append(test_description[prefix_len:])

        for directory in sorted(grouped.keys()):
            logger.info(f"     * {directory}")
            for filename in sorted(grouped[directory].keys()):
                descriptions = grouped[directory][filename]
                if len(descriptions) == 1 and descriptions[0]:
                    logger.info(f"          * {filename}: {descriptions[0]}")
                elif len(descriptions) == 1:
                    logger.info(f"          * {filename}")
                else:
                    logger.info(f"          * {filename}")
                    for desc in descriptions:
                        if desc:
                            logger.info(f"              * {desc}")

    def print_comparison(self) -> None:
        """
        Print failing tests exclusive to each compiler.

        Returns:
            None
        """
        if len(self.failed_tests) <= 1:
            return

        print_section("EXCLUSIVE FAILING TESTS COMPARISON")

        for compiler_label in sorted(self.failed_tests.keys()):
            other_failed = set.union(
                *(
                    self.failed_tests[other]
                    for other in self.failed_tests
                    if other != compiler_label
                )
            )
            exclusive_tests = self.failed_tests[compiler_label] - other_failed

            print_section(f"{compiler_label}", border_char="-", inline=True)
            if exclusive_tests:
                for test in sorted(exclusive_tests):
                    logger.info(f"  {STATUS_FAIL} {test}")
            else:
                logger.info("  No exclusive failing tests.")

    def check_all_pass_or_xfailed(
        self, xfailed_tests: Dict[str, List[str]]
    ) -> Tuple[bool, Dict[str, Dict]]:
        """
        Check if all tests passed or are in the expected failure list.

        Args:
            xfailed_tests: Compiler labels mapped to expected failing test paths.
                Must include "Generic" key for cross-compiler failures.

        Returns:
            Tuple of (overall_pass, details) where:
                - overall_pass: True if no unexpected failures.
                - details: Per-compiler dict with passed status, total_failed count,
                  unexpected_failures, unused_xfails, expected_generic,
                  expected_compiler_specific lists, and flaky_tests count.
        """
        overall_pass = True
        details = {}
        generic_xfailed = set(xfailed_tests.get("Generic", []))

        # Include all compilers that ran tests, so all-passing compilers also
        # show up in the final status report.
        all_compilers = set(self.test_data.keys())

        for compiler_label in all_compilers:
            failed_test_files = set(self.failed_tests.get(compiler_label, set()))
            compiler_xfailed = set(xfailed_tests.get(compiler_label, []))
            all_xfailed = generic_xfailed | compiler_xfailed

            unexpected_failures = failed_test_files - all_xfailed
            unused_xfails = all_xfailed - failed_test_files
            expected_generic = failed_test_files & generic_xfailed
            expected_compiler_specific = failed_test_files & compiler_xfailed

            harness_errors = self.harness_errors.get(compiler_label, [])
            has_harness_errors = bool(harness_errors)
            compiler_pass = len(unexpected_failures) == 0 and not has_harness_errors
            overall_pass = overall_pass and compiler_pass

            flaky_count = len(self.flaky_tests.get(compiler_label, set()))

            details[compiler_label] = {
                "passed": compiler_pass,
                "total_failed": len(failed_test_files),
                "unexpected_failures": sorted(unexpected_failures),
                "unused_xfails": sorted(unused_xfails),
                "expected_generic": sorted(expected_generic),
                "expected_compiler_specific": sorted(expected_compiler_specific),
                "flaky_tests": sorted(self.flaky_tests.get(compiler_label, set())),
                "flaky_count": flaky_count,
                "harness_errors": list(harness_errors),
            }

        return overall_pass, details

    def generate_ignore_list(self) -> Dict[str, List[str]]:
        """
        Generate ignore list from current test failures.

        Analyzes failures across all compilers and categorizes them as:
        - Generic: Tests that failed for all compilers
        - Compiler-specific: Tests that failed only for specific compilers

        Returns:
            Dictionary with keys "Generic" and compiler labels (e.g., "GCC", "LLVM"),
            each mapping to a sorted list of failed test file paths.
        """
        # Collect all failed test files per compiler that ran tests.
        all_compiler_failures = {}

        for compiler_label in sorted(self.test_data.keys()):
            failed_test_files = set(self.failed_tests.get(compiler_label, set()))
            all_compiler_failures[compiler_label] = failed_test_files

        # Determine which failures are common to all compilers.
        if len(all_compiler_failures) > 1:
            generic_failures = set.intersection(*all_compiler_failures.values())
        else:
            # 0 compilers (no runs) or 1 compiler (all failures are compiler-specific).
            generic_failures = set()

        # Build the ignore list structure.
        ignore_list = {"Generic": sorted(generic_failures)}

        # Add compiler-specific failures.
        for compiler_label in sorted(all_compiler_failures.keys()):
            compiler_specific = all_compiler_failures[compiler_label] - generic_failures
            ignore_list[compiler_label] = sorted(compiler_specific)

        return ignore_list

    def print_final_status(
        self, xfailed_tests: Dict[str, List[str]], no_xfail: bool
    ) -> bool:
        """
        Print final PASS/FAIL status for all compilers.

        Args:
            xfailed_tests: Compiler labels mapped to expected failures.
            no_xfail: If True, ignore the expected failure lists.

        Returns:
            True if all tests passed or are expected failures, False otherwise.
        """
        overall_pass, details = self.check_all_pass_or_xfailed(
            {} if no_xfail else xfailed_tests
        )

        print_section("FINAL TEST STATUS")

        for compiler_label in sorted(details.keys()):
            detail = details[compiler_label]
            status_symbol = STATUS_PASS if detail["passed"] else STATUS_FAIL
            status_text = "PASS" if detail["passed"] else "FAIL"

            logger.info(f"{status_symbol} {compiler_label}: {status_text}")

            if detail["total_failed"]:
                logger.info(
                    f"{STATUS_FAIL} Total Failed Tests: {detail['total_failed']}"
                )

            if detail["flaky_count"] > 0:
                logger.info(
                    f"{STATUS_FLAKY} Flaky Tests (passed after retry) ({detail['flaky_count']}):"
                )
                for test in detail["flaky_tests"]:
                    logger.info(f"       {test}")

            if detail["expected_generic"]:
                logger.info(
                    f"{STATUS_WARN} Ignored Failures (Generic) ({len(detail['expected_generic'])}):"
                )
                for test in detail["expected_generic"]:
                    logger.info(f"       {test}")

            if detail["expected_compiler_specific"]:
                logger.info(
                    f"{STATUS_WARN} Ignored Failures ({compiler_label}) ({len(detail['expected_compiler_specific'])}):"
                )
                for test in detail["expected_compiler_specific"]:
                    logger.info(f"       {test}")

            if detail["unexpected_failures"]:
                logger.info(
                    f"{STATUS_FAIL} Unexpected Failures ({len(detail['unexpected_failures'])}):"
                )
                for test in detail["unexpected_failures"]:
                    logger.info(f"       {test}")

            if detail["harness_errors"]:
                logger.info(
                    f"{STATUS_FAIL} Harness Errors ({len(detail['harness_errors'])}):"
                )
                for msg in detail["harness_errors"]:
                    logger.info(f"       {msg}")

            if detail["unused_xfails"]:
                logger.info(
                    f"{STATUS_WARN} Unused Ignored Failures ({len(detail['unused_xfails'])}):"
                )
                for test in detail["unused_xfails"]:
                    logger.info(f"       {test}")
            logger.info("")

        overall_status = "PASS" if overall_pass else "FAIL"
        overall_symbol = STATUS_PASS if overall_pass else STATUS_FAIL
        print_section(f"{overall_symbol} OVERALL STATUS: {overall_status}")
        logger.info("")

        return overall_pass


def print_section(
    title: str,
    border_char: str = "=",
    width: int = 80,
    center: bool = True,
    inline: bool = False,
    color: Optional[str] = None,
) -> None:
    """
    Print a formatted section header to console.

    Args:
        title: Text to display in the header.
        border_char: Character for border line (default "=").
        width: Total header width (default 80).
        center: Center title in multi-line mode (default True).
        inline: Single-line format if True, multi-line if False (default False).
        color: ANSI color code (e.g., "\\033[92m" for green).

    Returns:
        None
    """
    reset = "\033[0m"
    apply_color = (
        (lambda text: f"{color}{text}{reset}") if color else (lambda text: text)
    )

    # Always add a newline to the beginning of the section.
    logger.info("")

    if inline:
        # Prepare inline title.
        title_str = f" {title} "
        remaining = width - len(title_str)
        if remaining < 0:
            remaining = 0
        left = border_char * (remaining // 2)
        right = border_char * (remaining - len(left))
        logger.info(apply_color(f"{left}{title_str}{right}"))
    else:
        # Multi-line section style.
        border = border_char * width
        title_line = f"{title:^{width}}" if center else title
        logger.info(apply_color(border))
        logger.info(apply_color(title_line))
        logger.info(apply_color(border))


def _log_aligned_fields(
    fields: List[Union[str, Tuple[str, str]]],
    indent: str = "  ",
) -> None:
    """
    Log (label, value) pairs with colons aligned to the longest label with an associated value.

    Plain string entries are logged as-is (pass-through lines), skipping the
    colon and padding — useful for pass-through lines mixed in with key/value rows.

    Args:
        fields: List of (label, value) tuples or plain strings for pass-through lines.
        indent: Leading whitespace prefix for every row.

    Returns:
        None
    """
    if not fields:
        return
    labelled = [entry[0] for entry in fields if isinstance(entry, tuple)]
    width = max(len(label) for label in labelled) if labelled else 0
    for entry in fields:
        if isinstance(entry, str):
            logger.info(f"{indent}{entry}")
        else:
            label, value = entry
            logger.info(f"{indent}{label:<{width}} : {value}")


def validate_required_files(required_files: Dict[str, Path]) -> None:
    """
    Validate presence of required files and print status.

    Args:
        required_files: File descriptions mapped to filesystem paths.

    Returns:
        None

    Exits:
        Code 1 if any required file is missing.
    """
    print_section("Required files")
    all_valid = True

    fields: List[Tuple[str, str]] = []
    for name, path in required_files.items():
        present = path.is_file()
        status = STATUS_PASS if present else STATUS_FAIL
        fields.append((f"{status} {name}", str(path)))
        if not present:
            all_valid = False
    _log_aligned_fields(fields)

    if not all_valid:
        _log_error_and_exit("One or more required files are missing.")


def _prepend_to_path_var(env_vars: Dict[str, str], name: str, parts: List[str]) -> None:
    """
    Prepend `parts` to a path-style env var in `env_vars`.

    The caller is responsible for logging the change (so multiple env-var
    updates can be printed as a single aligned block).

    Args:
        env_vars: Environment variable dictionary to mutate.
        name: Name of the path-style variable (e.g., "PATH", "LD_LIBRARY_PATH").
        parts: New entries to prepend, in order.

    Returns:
        None
    """
    existing = env_vars.get(name, "")
    env_vars[name] = os.pathsep.join(parts + ([existing] if existing else []))


def setup_sanitizer_environment(env_vars: Dict[str, str], testsuite_dir: Path) -> None:
    """
    Pre-configure *SAN_OPTIONS for every sanitizer we know about.

    This runs unconditionally. When rocgdb is not sanitizer-instrumented,
    the matching env vars are ignored by the (absent) sanitizer runtime,
    so the only side effect of a normal run is the creation of an empty
    sanitizer log directory.

    For each sanitizer in SANITIZERS we set:
        halt_on_error=0          - sanitizer findings do not abort the run
        detect_leaks=0           - asan only. ASan ships LSan integrated;
                                   silence leak reports by default. Override
                                   with ASAN_OPTIONS=detect_leaks=1 for a
                                   dedicated leak run.
        log_path=<dir>/log       - reports go to <dir>/log.<pid>; the
                                   sanitizer runtime appends .<pid> per
                                   process, so one log file is produced
                                   per process. All sanitizers share a
                                   single directory because log_path lives
                                   in the sanitizer common flags - in a
                                   combined build (e.g. ASan + UBSan) the
                                   last-initialized runtime would overwrite
                                   any per-sanitizer log_path. Reports
                                   self-identify in their content.
        suppressions=<file>      - only when the .supp template exists;
                                   empty suppression files are still
                                   attached intentionally so users can
                                   populate the templates without code
                                   changes. suppressions is per-runtime
                                   and does not collide.

    If the shared log directory cannot be created (e.g. read-only
    testsuite mount), the log_path option is omitted and the sanitizer
    falls back to stderr - still non-fatal.
    """
    print_section("Sanitizer environment")

    suppressions_dir = Path(__file__).resolve().parent / "sanitizers"
    log_dir = testsuite_dir / "sanitizers_output"
    log_dir_ready = True
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_dir_ready = False
        logger.warning(
            f"{STATUS_WARN} Could not create sanitizer log dir "
            f"{log_dir}: {exc}. Sanitizer reports will go to stderr."
        )

    fields: List[Tuple[str, str]] = []
    for sanitizer in SANITIZERS:
        options = ["halt_on_error=0"]
        if sanitizer == "asan":
            options.append("detect_leaks=0")
        if log_dir_ready:
            options.append(f"log_path={log_dir}/log")

        suppression_file = suppressions_dir / f"{sanitizer}.supp"
        if suppression_file.is_file():
            options.append(f"suppressions={suppression_file}")

        env_var_name = f"{sanitizer.upper()}_OPTIONS"
        # Preserve any pre-existing options from the caller's environment.
        # Sanitizer option parsing is last-wins, so appending the existing
        # value after our defaults lets the user override them.
        existing = env_vars.get(env_var_name, "")
        if existing:
            options.append(existing)
        env_vars[env_var_name] = ":".join(options)
        fields.append((env_var_name, env_vars[env_var_name]))

    _log_aligned_fields(fields)


def setup_environment(
    artifacts_dir: Path, try_rocm_path: bool, msys2_root: Optional[Path] = None
) -> Dict[str, str]:
    """
    Configure environment variables for test suite execution.

    Args:
        artifacts_dir: Path to artifacts directory with binaries and libraries.
        try_rocm_path: Whether --try-rocm-path was passed.  When False,
            ROCM_PATH is removed from the environment so a system-wide
            setting cannot interfere with the LLVM toolchain paths we
            configure explicitly.
        msys2_root: On Windows, the MSYS2 installation whose sh, make and
            DejaGnu run the testsuite.  Its usr/bin is put on PATH.  Unused
            elsewhere.

    Returns:
        Updated environment variables dictionary.
    """
    print_section("Setting up environment variables")

    # Copy current environment.
    env_vars = os.environ.copy()

    # Drop ROCM_PATH when the caller has not opted in to using it.  A
    # system-wide ROCM_PATH can redirect LLVM tool lookups away from the
    # artifacts tree we set up below, producing hard-to-diagnose failures.
    if not try_rocm_path and "ROCM_PATH" in env_vars:
        del env_vars["ROCM_PATH"]
        logger.info("Dropped ROCM_PATH from environment (--try-rocm-path not set)")

    # Add ROCgdb and LLVM binaries to PATH and libraries to LD_LIBRARY_PATH.
    #
    # Please note LLVM has switched location for its executables/libraries.
    # It used to be llvm/bin (and llvm/lib) but now is lib/llvm/bin (and
    # lib/llvm/lib). A llvm -> lib/llvm symlink is kept for backwards
    # compatibility, but the new path should be used moving forward.
    path_parts = [
        f"{artifacts_dir}/bin",
        f"{artifacts_dir}/llvm/bin",
        f"{artifacts_dir}/lib/llvm/bin",
    ]
    ld_library_parts = [
        f"{artifacts_dir}/lib",
        f"{artifacts_dir}/llvm/lib",
        f"{artifacts_dir}/lib/llvm/lib",
    ]

    # Determine target triple.  It only extends LD_LIBRARY_PATH, which Windows
    # doesn't use (see below).
    target_triple = None
    amdclang_path = artifacts_dir / "lib" / "llvm" / "bin" / "amdclang++"
    if not IS_WINDOWS and amdclang_path.exists():
        try:
            target_triple = subprocess.run(
                [str(amdclang_path), "--print-target-triple"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"'{amdclang_path} --print-target-triple' failed; "
                "this suggests a broken toolchain."
            ) from exc

    if target_triple:
        ld_library_parts += [f"{artifacts_dir}/lib/llvm/lib/{target_triple}"]

    if IS_WINDOWS:
        # The Windows loader finds DLLs via PATH, and TheRock installs them next
        # to the executables in bin/, so there is no library path to set.  The
        # testsuite's configure, make and DejaGnu come from MSYS2.  Its usr/bin
        # goes ahead of the system directories, so that the shell scripts get
        # MSYS2's `find` and `sort` instead of Windows'.
        ld_library_parts = []
        path_parts.append(str(msys2_root / "usr" / "bin"))
        # Without MSYSTEM, MSYS2's uname would describe the MSYS environment
        # itself, so the testsuite's configure would pick a *-msys triplet, and
        # the gdb.rocm tests would end up all UNSUPPORTED.
        env_vars["MSYSTEM"] = "UCRT64"
        # An MSYS2 login shell points CONFIG_SITE at its environment's
        # config.site, which the testsuite's configure would then read.
        if env_vars.pop("CONFIG_SITE", None) is not None:
            logger.info("Dropped CONFIG_SITE from environment")

    _prepend_to_path_var(env_vars, "PATH", path_parts)
    fields = [("PATH (prepended)", os.pathsep.join(path_parts))]
    if ld_library_parts:
        _prepend_to_path_var(env_vars, "LD_LIBRARY_PATH", ld_library_parts)
        fields.append(
            ("LD_LIBRARY_PATH (prepended)", os.pathsep.join(ld_library_parts))
        )

    # Check if we are running within a github actions context, where a
    # non-system version of Python is being used. If so, we have
    # pythonLocation set to the base location of the Python interpreter.
    if "pythonLocation" in env_vars:
        # Set PYTHONHOME so rocgdb can initialize Python properly.
        env_vars["PYTHONHOME"] = env_vars["pythonLocation"]
        pythonhome_value = env_vars["PYTHONHOME"]
    else:
        pythonhome_value = "not set (no pythonLocation env var)"
    fields.append(("PYTHONHOME", pythonhome_value))

    _log_aligned_fields(fields)

    return env_vars


def _windows_is_packaged() -> bool:
    """
    Native Windows Python only: whether this process runs as a packaged app,
    as the Microsoft Store's Python does.
    """
    # With no buffer, a packaged process gets ERROR_INSUFFICIENT_BUFFER, as its
    # package name doesn't fit, and any other gets APPMODEL_ERROR_NO_PACKAGE.
    error_insufficient_buffer = 122
    length = ctypes.c_uint32(0)
    result = _windows_kernel32().GetCurrentPackageFullName(
        ctypes.byref(length), None
    )
    return result == error_insufficient_buffer


def _windows_set_dll_directory(path: Path) -> None:
    """
    Windows only: search `path` for DLLs ahead of the system directories, in
    this process and in every process it starts.  This process must not be a
    packaged one, such as the Microsoft Store's Python, as the processes a
    packaged one starts don't inherit the setting.

    Args:
        path: Directory holding the DLLs to prefer.
    """
    # The loader searches C:\Windows\System32 before PATH, so a HIP runtime that
    # Adrenalin installed in System32 would be loaded by the test programs
    # instead of the one under test.  The SetDllDirectoryW directory is
    # searched before System32, and child processes inherit it, down through
    # make, sh, expect and rocgdb to the test programs.  SetDllDirectoryW's
    # documentation limits the inheritance to processes that are not packaged
    # or protected ones.
    win_path = _windows_path(path)
    _windows_check_result(
        _windows_kernel32().SetDllDirectoryW(win_path),
        f"SetDllDirectoryW({win_path!r})",
    )


def _windows_find_file(name: str, search_path: Optional[str]) -> Optional[str]:
    """
    Native Windows Python only: find a file named exactly `name` in the
    directories of a PATH value, whether or not its extension is in PATHEXT.

    Args:
        name: The file name.
        search_path: The PATH value to search.

    Returns:
        The file's path, or None if no directory has it.
    """
    # Windows has no executable bit, and a native Python's which() only takes
    # a name with an extension from PATHEXT, as cmd.exe does.  Before Python
    # 3.12 it never tries the name as given, and from 3.12.8 and 3.13.1 on
    # (gh-127001), it no longer does when looking for an executable.  That
    # misses MSYS2's runtest, a shell script that make runs through sh.
    for directory in (search_path or "").split(os.pathsep):
        if directory and os.path.isfile(os.path.join(directory, name)):
            return os.path.join(directory, name)
    return None


def check_executables(executables: List[str], env_vars: Dict[str, str]) -> None:
    """
    Verify required executables are in the prepared PATH.

    Args:
        executables: List of executable names to check.
        env_vars: Prepared environment supplying the PATH used for the lookup.

    Returns:
        None

    Exits:
        Code 1 if any executable is missing.
    """
    print_section("Required executables")
    missing = []
    search_path = env_vars.get("PATH")

    for exe in executables:
        path = shutil.which(exe, path=search_path)
        # An MSYS2 Python's which() has the runtime's own idea of executable,
        # which MSYS2's runtest passes, but a native Python's does not.
        if path is None and IS_NATIVE_WINDOWS:
            path = _windows_find_file(exe, search_path)
        if path:
            logger.info(f"{STATUS_PASS} {exe:15} found at: {path}")
        else:
            missing.append(exe)
            logger.info(f"{STATUS_FAIL} {exe:15} NOT found on PATH")

    if missing:
        _log_error_and_exit(f"Missing {len(missing)} executables required for testing.")


def setup_core_file_info() -> None:
    """
    Set system core file size limit to unlimited and display core file pattern.

    The outcome is communicated to the user via logger.info / logger.warning
    so the result is visible in the run banner.
    """
    print_section("Core file information")

    # Display current core file pattern.
    core_pattern_file = Path("/proc/sys/kernel/core_pattern")
    try:
        if core_pattern_file.exists():
            core_pattern = core_pattern_file.read_text().strip()
            logger.info(f"System core file pattern: {core_pattern}")
        else:
            logger.info("System core file pattern: N/A (file not found)")
    except OSError as e:
        logger.warning(f"System core file pattern: N/A (unable to read: {e})")

    try:
        resource.setrlimit(
            resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)

        if soft == resource.RLIM_INFINITY:
            logger.info(f"{STATUS_PASS} Core file size limit set to unlimited")
        else:
            logger.warning(f"Core file size limit is {soft}, not unlimited")
            logger.warning("Core file tests may not execute properly")

    except (OSError, ValueError) as e:
        logger.error(f"{STATUS_FAIL} Unable to set core file size limit: {e}")
        logger.warning("Core file tests will not be executed")


def cleanup_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Clean build artifacts and recreate site.exp configuration.

    Args:
        test_suite_dir: Path to test suite directory.
        env_vars: Environment variables for cleanup commands.
        quiet: Suppress command output if True (default False).

    Returns:
        None

    Exits:
        Code 1 if cleanup fails.
    """
    logger.info("Cleaning test suite directory...")
    cmd = ["make", "clean"]
    logger.info(f"Executing cleanup: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Failed to clean test directory",
    )

    logger.info("Removing old site.exp and site.bak...")
    for filename in ["site.exp", "site.bak"]:
        (test_suite_dir / filename).unlink(missing_ok=True)

    logger.info("Creating site.exp...")
    cmd = ["make", "site.exp"]
    logger.info(f"Executing: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Failed to create site.exp",
    )


def configure_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Run test suite configuration script.

    Args:
        test_suite_dir: Path to test suite directory with configure script.
        env_vars: Environment variables for configuration.
        quiet: Suppress command output if True (default False).

    Returns:
        None

    Exits:
        Code 1 if configuration fails.
    """
    configure_script = test_suite_dir / "configure"
    cmd = ["sh", str(configure_script)]
    logger.info(f"Executing: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Test suite configuration failed",
    )
    cleanup_test_suite(test_suite_dir, env_vars, quiet)


def set_test_timeout(test_suite_dir: Path, timeout_value: int) -> None:
    """
    Set gdb_test_timeout in site.exp configuration file.

    Args:
        test_suite_dir: Path to test suite directory with site.exp file.
        timeout_value: Timeout in seconds for test runs.

    Returns:
        None

    Exits:
        Code 1 if site.exp is missing or cannot be written.
    """
    site_exp_file = test_suite_dir / "site.exp"
    if not site_exp_file.is_file():
        _log_error_and_exit(f"site.exp not found at {site_exp_file}")
    try:
        with open(site_exp_file, "a", encoding="utf-8") as f:
            f.write(f"\nset gdb_test_timeout {timeout_value}\n")
        logger.info(
            f"{STATUS_PASS} Successfully set gdb_test_timeout to {timeout_value} in {site_exp_file}"
        )
    except OSError as e:
        _log_error_and_exit(f"Failed to write timeout to {site_exp_file}: {e}")


def discover_test_directories(
    testsuite_dir: Path, exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Discover all test directories in the testsuite.

    Args:
        testsuite_dir: Path to testsuite directory.
        exclude: List of directory names to exclude (default None).

    Returns:
        Sorted list of test directory names (e.g., ["gdb.base", "gdb.threads"]).
    """
    if exclude is None:
        exclude = []

    test_dirs = []
    for item in testsuite_dir.iterdir():
        if item.is_dir() and item.name.startswith("gdb.") and item.name not in exclude:
            if any(item.glob("*.exp")):
                test_dirs.append(item.name)

    return sorted(test_dirs)


def _glob_posix(pattern: str, root: Path) -> List[str]:
    """
    Find the paths under `root` that match `pattern`.

    Args:
        pattern: The glob pattern, relative to `root`.
        root: The directory to search.

    Returns:
        The matching paths, sorted, relative to `root` and with forward
        slashes, as runtest requires in TESTS.
    """
    # glob separates with backslashes on Windows.
    return sorted(Path(m).as_posix() for m in glob.glob(pattern, root_dir=str(root)))


def expand_test_paths(test_list: List[str], testsuite_dir: Path) -> List[str]:
    """
    Expand test paths to a list of .exp files, validating existence.

    Args:
        test_list: Test paths (individual .exp files or directories).
        testsuite_dir: Base directory to resolve relative paths.

    Returns:
        List of .exp test file paths.

    Exits:
        Code 1 if any test file is missing.
    """
    expanded_tests: List[str] = []

    for test_path in test_list:
        # On Windows, gdb.rocm\simple.exp names a test as well, but runtest
        # requires forward slashes in TESTS.  PureWindowsPath, since under an
        # MSYS2 Python a Path is a POSIX one, which takes the backslash as
        # part of a name, and so would glob.
        if IS_WINDOWS:
            test_path = PureWindowsPath(test_path).as_posix()
        if not test_path.endswith(".exp"):
            pattern = f"{test_path}/*.exp"
            matches = _glob_posix(pattern, testsuite_dir)
            if matches:
                expanded_tests.extend(matches)
                logger.info(
                    f"Expanded directory '{test_path}' to {len(matches)} test files"
                )
            else:
                logger.warning(f"No test files found matching pattern: {pattern}")
        else:
            expanded_tests.append(test_path)
            logger.info(f"Added test file: {test_path}")

    # Drop duplicates while preserving first-occurrence order so that
    # overlapping --tests arguments do not silently run the same test twice.
    deduped_tests = list(dict.fromkeys(expanded_tests))
    if len(deduped_tests) != len(expanded_tests):
        dropped = len(expanded_tests) - len(deduped_tests)
        logger.warning(
            f"Dropped {dropped} duplicate test path(s) from the expanded test list"
        )
    expanded_tests = deduped_tests

    # Verify that all expanded files exist.
    for file_path in expanded_tests:
        abs_path = testsuite_dir / file_path
        if not abs_path.is_file():
            _log_error_and_exit(f"Missing or invalid test file: {file_path}")

    return expanded_tests


def print_env_variables() -> None:
    """
    Print all environment variables for debugging.

    Returns:
        None
    """
    print_section("Environment Variables")
    _log_aligned_fields([(key, value) for key, value in os.environ.items()])


def validate_path(path: Path, is_dir: bool = False, is_file: bool = False) -> Path:
    """
    Validate path exists and matches expected type.

    Args:
        path: Path to validate.
        is_dir: Require directory if True (default False).
        is_file: Require file if True (default False).

    Returns:
        The resolved (absolute, symlink-followed) path.

    Exits:
        Code 1 if path doesn't exist or is wrong type.
    """
    resolved_path = path.resolve()

    if is_dir and not resolved_path.is_dir():
        _log_error_and_exit(f"Directory does not exist: {resolved_path}")
    if is_file and not resolved_path.is_file():
        _log_error_and_exit(f"File does not exist: {resolved_path}")

    return resolved_path


def validate_rocgdb(rocgdb_bin: Path, env_vars: Dict[str, str]) -> None:
    """
    Validate ROCgdb executable can run successfully.

    Args:
        rocgdb_bin: Path to ROCgdb executable.
        env_vars: Prepared environment (PATH/LD_LIBRARY_PATH/sanitizer options)
            to use for the validation subprocesses.

    Returns:
        None

    Exits:
        Code 1 if ROCgdb fails to execute or its python validation fails.
    """
    print_section("ROCgdb launcher data")

    try:
        # First invoke the rocgdb launcher in debug mode.
        print_section("ROCgdb launcher start", border_char="-", inline=True)
        env = env_vars.copy()
        env["ROCGDB_WRAPPER_DEBUG"] = "1"
        result = _run_command(
            [str(rocgdb_bin), "--version"], env=env, capture_output=True
        )

        for line in result.stderr.splitlines():
            logger.info(line)
        logger.info("ROCgdb launcher ran successfully.")

        # Now validate that we can launch rocgdb at all.
        print_section("ROCgdb executable start", border_char="-", inline=True)
        result = _run_command(
            [str(rocgdb_bin), "--version"], env=env_vars, capture_output=True
        )

        for line in result.stdout.splitlines():
            logger.info(line)
        logger.info("ROCgdb executable ran successfully.")
    except subprocess.CalledProcessError as e:
        _log_error_and_exit(f"ROCgdb did not run successfully: {e.stderr}")

    # Validate internal Python environment.
    print_section("ROCgdb internal python data", border_char="-", inline=True)

    py_cmd = (
        "import sys, os, sysconfig, gdb; "
        "v = sys.version_info; "
        "version = f'{v.major}.{v.minor}.{v.micro}'; "
        "supports = bool(sysconfig.get_config_var('Py_ENABLE_SHARED')); "
        "lib_dir = sysconfig.get_config_var('LIBDIR'); "
        "ld_lib = sysconfig.get_config_var('LDLIBRARY'); "
        "lib_path = os.path.join(lib_dir, ld_lib) if (supports and lib_dir and ld_lib) else 'N/A'; "
        "print(f'Executable: {sys.executable}'); "
        "print(f'Version: {version}'); "
        "print(f'Supports libpython: {supports}'); "
        "print(f'libpython Path: {lib_path}'); "
        'print(f\'libpython Version: {sysconfig.get_config_var("VERSION") or "N/A"}\'); '
        "print(f'GDB python modules path: {gdb.PYTHONDIR}'); "
    )

    result = _run_command(
        [str(rocgdb_bin), "-batch", "-ex", f"python {py_cmd}"],
        env=env_vars,
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        # Parse "label: value" pairs printed by the embedded script and
        # re-emit them with colons aligned, matching the surrounding
        # banner blocks. Lines without a colon are passed through verbatim.
        parsed: List[Union[str, Tuple[str, str]]] = []
        for line in result.stdout.splitlines():
            label, sep, value = line.partition(":")
            if sep:
                parsed.append((label.strip(), value.strip()))
            else:
                parsed.append(line)
        _log_aligned_fields(parsed)
        logger.info("ROCgdb internal python validation successful.")
    elif "Python scripting is not supported in this copy of GDB" in result.stderr:
        logger.warning(
            "Python scripting is not supported in this copy of GDB. Testing will proceed without Python support."
        )
    else:
        _log_error_and_exit(f"ROCgdb python validation failed: {result.stderr}")


def run_tests(
    test_suite_dir: Path,
    rocgdb_bin: Path,
    env_vars: Dict[str, str],
    tests: List[str],
    cc: str,
    cxx: str,
    fc: str,
    compiler_label: str,
    test_results: "TestResults",
    args: argparse.Namespace,
    xfailed_tests: Optional[Dict[str, List[str]]] = None,
    sanity_check_exe: Optional[Path] = None,
    emulator=None,
) -> None:
    """
    Run ROCgdb test suite with retry logic for failed tests.

    Args:
        test_suite_dir: Path to test suite directory.
        rocgdb_bin: Path to ROCgdb binary.
        env_vars: Environment variables for test execution.
        tests: Test file names to run.
        cc: C compiler command.
        cxx: C++ compiler command.
        fc: Fortran compiler command.
        compiler_label: Compiler identifier (e.g., "GCC").
        test_results: TestResults object for storing results.
        args: Parsed command-line arguments.
        xfailed_tests: Compiler labels mapped to expected failing test paths.
            Used to skip retrying ignored tests when --retry-ignored-tests is off.
        sanity_check_exe: When supplied (--sanity-check mode), the pre-built
            known-good HIP executable run before each test in one-by-one mode.
            None disables the probe.
        emulator: When supplied (--emulate mode), the emulated GPU each
            `make check` is run inside. None runs on real hardware.

    Returns:
        None
    """
    max_iterations = 1 + args.max_failed_retries
    current_tests = tests

    for iteration in range(1, max_iterations + 1):
        print_section(f"ROCgdb tests - {compiler_label} - Iteration {iteration}")
        logger.info(f"Number of tests to run: {len(current_tests)}")

        # Track which tests we're running in this iteration (for flaky detection).
        tests_in_iteration = set(current_tests)

        configure_test_suite(test_suite_dir, env_vars, args.quiet)

        # default_timeout applies on every iteration. On retry iterations
        # following a TIMEOUT, use retry_timeout instead (parse_arguments
        # guarantees it is >= default_timeout).
        timeout_to_set = args.default_timeout
        if iteration > 1 and test_results.test_data.get(compiler_label, {}).get(
            "TIMEOUT"
        ):
            timeout_to_set = args.retry_timeout

        if timeout_to_set is not None:
            set_test_timeout(test_suite_dir, timeout_to_set)

        start_time = time.perf_counter()

        runtestflags_str = _build_runtestflags(
            rocgdb_bin, cc, cxx, fc, args.optimization, args.runtestflags
        )

        if args.one_by_one:
            if args.one_by_one_test_timeout is not None:
                wall_clock = args.one_by_one_test_timeout
            else:
                # gdb_test_timeout is per-testcase, not per-.exp. Default
                # generously and let the user override via --one-by-one-test-timeout.
                wall_clock = ONE_BY_ONE_DEFAULT_WALL_CLOCK_TIMEOUT
                logger.info(
                    f"Using default one-by-one wall-clock timeout: {wall_clock}s "
                    f"(override with --one-by-one-test-timeout)"
                )
            log_dir = args.one_by_one_log_dir or (test_suite_dir / "one_by_one_logs")

            _execute_one_by_one(
                test_suite_dir,
                current_tests,
                runtestflags_str,
                args.check_type,
                args.quiet,
                env_vars,
                log_dir,
                compiler_label,
                iteration,
                wall_clock,
                test_results if args.timing else None,
                sanity_check_exe,
                args.sanity_check_timeout,
                emulator,
            )
        else:
            cmd = [
                "make",
                args.check_type,
                f"RUNTESTFLAGS={runtestflags_str}",
                f"TESTS={' '.join(current_tests)}",
            ]
            if args.parallel:
                if args.fraction is not None:
                    jobs = max(1, int(os.cpu_count() * args.fraction))
                else:
                    jobs = args.jobs or os.cpu_count()
                cmd += ["FORCE_PARALLEL=1", f"-j{jobs}"]

            if emulator is not None:
                if args.parallel:
                    # A session per job rather than one for all of them, so the
                    # jobs do not queue on a single emulated GPU. Quoted because
                    # make expands $(RUNTEST) into a shell recipe.
                    wrapper = shlex.quote(str(emulator.runtest_wrapper()))
                    cmd.append(f"RUNTEST={wrapper}")
                else:
                    # One session for the whole `make check`, as on hardware.
                    cmd = emulator.wrap(cmd)

            logger.info(
                f"Executing tests with {compiler_label} - Iteration {iteration}: {shlex.join(cmd)}"
            )
            _run_command(
                cmd,
                cwd=test_suite_dir,
                env=env_vars,
                capture_output=args.quiet,
                check=False,
                kill_process_tree=True,
                own_console=True,
            )

        duration = time.perf_counter() - start_time

        print_section(
            f"Tests with {compiler_label} - Iteration {iteration} completed in {duration:.4f} seconds."
        )

        # Parse test results from gdb.sum.
        results_file = test_suite_dir / "gdb.sum"
        test_results.update_results(compiler_label, current_tests, results_file)

        failed_tests = test_results.get_failed_tests(compiler_label)

        # Detect flaky tests: tests we ran this iteration that now have PASS
        # entries (and are no longer in failed_tests). Restrict to tests we
        # actually ran so tests missing from gdb.sum aren't misreported as flaky.
        if iteration > 1:
            passed_this_iteration = {
                test_file
                for test_file in test_results.test_data.get(compiler_label, {}).get(
                    "PASS", {}
                )
                if test_file in tests_in_iteration
            }
            failed_test_files = set(failed_tests)
            newly_passed = passed_this_iteration - failed_test_files
            if newly_passed:
                test_results.mark_flaky_tests(compiler_label, newly_passed)

        if not failed_tests:
            logger.info(
                f"{STATUS_PASS} No failing tests for {compiler_label}. Stopping iterations."
            )
            break
        elif iteration < max_iterations:
            # Collect the failed tests for the next iteration.  By default,
            # tests on the ignore list are not retried: they are known failures
            # and re-running them (potentially many times, for two compilers)
            # adds significant wall-clock time for no diagnostic value.
            # --retry-ignored-tests or --no-xfail restore the old behaviour.
            retry_candidates = set(failed_tests)
            if not args.retry_ignored_tests and not args.no_xfail and xfailed_tests:
                all_xfailed = set(xfailed_tests.get("Generic", [])) | set(
                    xfailed_tests.get(compiler_label, [])
                )
                skipped = retry_candidates & all_xfailed
                retry_candidates -= all_xfailed
                if skipped:
                    logger.info(
                        f"{STATUS_WARN} Skipping retry of {len(skipped)} ignored test(s) "
                        f"for {compiler_label} (use --retry-ignored-tests to retry them):"
                    )
                    for t in sorted(skipped):
                        logger.info(f"       {t}")
            if not retry_candidates:
                logger.info(
                    f"{STATUS_PASS} All remaining failures for {compiler_label} are on the "
                    f"ignore list. Stopping iterations."
                )
                break
            current_tests = sorted(retry_candidates)
            logger.info(
                f"{STATUS_FAIL} {len(current_tests)} failing test(s) found for {compiler_label}. "
                f"Proceeding to iteration {iteration + 1} with only failed tests."
            )
        else:
            logger.info(
                f"{STATUS_FAIL} {len(failed_tests)} failing test(s) remain for {compiler_label} "
                f"after {max_iterations} iterations."
            )

            # Print the contents of gdb.log in a visually uncluttered way.
            # Skipped in --one-by-one mode since per-test logs are already on disk.
            if args.dump_failed_test_log and not args.one_by_one:
                gdb_log_file = test_suite_dir / "gdb.log"
                if gdb_log_file.is_file():
                    print_section("Contents of gdb.log")
                    # Read directly rather than via _read_file_lines: gdb.log
                    # uses leading whitespace for nesting, which .strip() would
                    # flatten and make the failure dump much harder to read.
                    try:
                        contents = gdb_log_file.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError as e:
                        logger.warning(f"Failed to read {gdb_log_file}: {e}")
                    else:
                        for line in contents.splitlines():
                            logger.info(line)
                else:
                    logger.warning(
                        f"gdb.log not found at {gdb_log_file}; skipping dump."
                    )


@functools.lru_cache(maxsize=None)
def _is_msys2_python() -> bool:
    """
    Whether this Python is built against the MSYS2 runtime, as opposed to a
    native Windows one, a Cygwin one, or one on another OS.
    """
    if not IS_POSIX_WINDOWS:
        return False
    # Ask which runtime this process has loaded.  uname, and so
    # platform.system(), can't tell: MSYS2's follows MSYSTEM, and reports
    # MSYS_NT only in the MSYS environment, MINGW64_NT and the like in the
    # others.  GetModuleHandleW never loads the DLL it is asked about.
    return bool(_windows_kernel32().GetModuleHandleW("msys-2.0.dll"))


def _windows_resolve_msys2_root(msys2_root: Optional[Path]) -> Path:
    """
    Windows only: pick the MSYS2 installation that runs the testsuite, and
    log which one and why.  In an MSYS2 Python that is the Python's own
    installation, "/".  Otherwise it is --msys2-root if given, else the MSYS2
    this script was started from, else C:/msys64.

    Args:
        msys2_root: The --msys2-root value, or None if it was not given.

    Returns:
        The MSYS2 root directory.

    Exits:
        Code 1 if an MSYS2 Python is given another installation.
    """
    if _is_msys2_python():
        # An MSYS2 program hands any MSYS2 program it starts, even one from
        # another installation, the environment's paths in POSIX form.  A
        # program from another installation reads them through its own mount
        # table, where they can name other directories (/tmp, say, as
        # C:\WINDOWS), and configure then can't run the compiler.
        own_root = Path("/")
        if msys2_root is not None:
            # Compare the directories, not the names: the root can also be
            # spelled in its POSIX form, such as /c/msys64, which is "/" only
            # as a directory.  A directory that doesn't exist isn't it either.
            try:
                same = msys2_root.samefile(own_root)
            except OSError:
                same = False
            if not same:
                _log_error_and_exit(
                    f"--msys2-root {msys2_root} is not this MSYS2 Python's own "
                    f"installation ({_msys2_cygpath('-w', own_root)}).  Drop "
                    "--msys2-root, or run the script with that installation's "
                    "Python or with a native Windows Python."
                )
        logger.info(f"Using this MSYS2 Python's own MSYS2: {own_root}")
        return own_root
    if msys2_root is not None:
        logger.info(f"Using MSYS2 from --msys2-root: {msys2_root}")
        return msys2_root
    # An MSYS2 login shell sets MSYSTEM_PREFIX to its environment's prefix,
    # such as /ucrt64, or /usr for MSYS, always a directory directly under
    # the MSYS2 root.  A native program started from the shell sees it in
    # Windows form, C:/msys64/ucrt64.
    prefix = os.environ.get("MSYSTEM_PREFIX")
    if prefix:
        root = host_path(prefix).parent
        logger.info(f"Using the MSYS2 this script runs from: {root}")
        return root
    root = host_path("C:/msys64")
    logger.info(f"Using the default MSYS2: {root}")
    return root


def _resolve_rocm_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """
    Resolve the ROCm tree root, ROCgdb binary, and testsuite directory.

    The ROCm tree root is always auto-discovered from:
    ROCM_PATH (if --try-rocm-path) > OUTPUT_ARTIFACTS_DIR > script location.

    The binary and testsuite paths are then derived from that root unless
    the user overrode either with --rocgdb-bin / --testsuite-dir, in which
    case the supplied path is honored verbatim. The two override flags are
    independent — supplying one does not require the other.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (rocm_dir, rocgdb_bin, rocgdb_testsuite_dir).

    Exits:
        Code 1 if a configured environment-variable path does not exist.
    """
    rocm_dir = None

    # An environment variable set to the empty string counts as unset, as an
    # empty path would otherwise mean the current directory.
    if args.try_rocm_path:
        rocm_path_env = os.getenv("ROCM_PATH")
        if rocm_path_env:
            tmp_dir = host_path(rocm_path_env).resolve()
            if not tmp_dir.is_dir():
                _log_error_and_exit(
                    f"ROCM_PATH is set to '{tmp_dir}' but the path does not exist"
                )
            rocm_dir = tmp_dir
            logger.info(f"Using ROCM_PATH: {rocm_dir}")
        else:
            logger.info(
                "--try-rocm-path flag set but ROCM_PATH environment variable is not "
                "set or is empty. "
                "Falling back to OUTPUT_ARTIFACTS_DIR or script location."
            )

    if rocm_dir is None:
        output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
        if output_artifacts_dir:
            # OUTPUT_ARTIFACTS_DIR is set, verify it exists.
            rocm_dir = host_path(output_artifacts_dir).resolve()
            if not rocm_dir.is_dir():
                _log_error_and_exit(
                    f"OUTPUT_ARTIFACTS_DIR is set to '{rocm_dir}' but the path does not exist"
                )
            logger.info(f"Using OUTPUT_ARTIFACTS_DIR: {rocm_dir}")
        else:
            # OUTPUT_ARTIFACTS_DIR is not set. Fall back to the script's own
            # location: when installed, this file lives at
            # <rocm_tree>/tests/rocgdb/test_rocgdb.py, so three .parent hops
            # land on <rocm_tree>. This branch only produces a usable path
            # when run from the installed location, not from the source repo.
            script_path = Path(__file__).resolve()
            rocm_dir = script_path.parent.parent.parent
            logger.info(f"Using script-based path: {rocm_dir}")

    # Windows has no launcher wrapper script choosing between Python variants,
    # so the executable itself is the default there.
    default_rocgdb = "rocgdb.exe" if IS_WINDOWS else "rocgdb"
    rocgdb_bin = (
        args.rocgdb_bin
        if args.rocgdb_bin is not None
        else rocm_dir / "bin" / default_rocgdb
    )
    rocgdb_testsuite_dir = (
        args.testsuite_dir
        if args.testsuite_dir is not None
        else rocm_dir / "tests" / "rocgdb" / "gdb" / "testsuite"
    )
    return rocm_dir, rocgdb_bin, rocgdb_testsuite_dir


def main() -> None:
    """
    Main entry point for ROCgdb test suite runner.

    Orchestrates argument parsing, environment setup, test execution,
    and result reporting for GCC and LLVM compilers.

    Returns:
        None

    Exits:
        Code 0 if all tests pass or are expected failures, Code 1 otherwise.
    """
    # A Cygwin Python works in Cygwin paths (/cygdrive/c/...), which mean
    # nothing to the testsuite's MSYS2 programs.  For a non-Cygwin program it
    # starts, MSYS2's included, Cygwin converts a few environment variables,
    # such as PATH, but not the arguments, so any path passed as an argument,
    # such as the path to the testsuite's configure script, would reach the
    # program unusable.
    if IS_POSIX_WINDOWS and not _is_msys2_python():
        _log_error_and_exit(
            "A Cygwin Python can't run the testsuite's MSYS2 programs.  "
            "Use a native Windows Python or MSYS2's."
        )
    # The processes a packaged Python starts don't inherit the DLL directory
    # that _windows_set_dll_directory sets, so the test programs would load
    # the HIP runtime that Adrenalin installs in System32 instead of the one
    # under test, and every result would be for the wrong runtime, with
    # nothing to show it.
    if IS_NATIVE_WINDOWS and _windows_is_packaged():
        _log_error_and_exit(
            "A packaged Python, such as the Microsoft Store's, can't make the "
            "test programs use the HIP runtime under test.  Use one from "
            "python.org, or MSYS2's."
        )
    args = parse_arguments()

    # Determine ignore list file path.
    if args.ignore_list_file is None:
        # Default to rocgdb_ignore_list.json in the same directory as this script.
        script_dir = Path(__file__).parent
        args.ignore_list_file = script_dir / "rocgdb_ignore_list.json"
    else:
        # User explicitly supplied a path — it must exist.
        args.ignore_list_file = validate_path(args.ignore_list_file, is_file=True)

    # Determine ignore list file status for configuration display.
    if args.ignore_list_file.exists():
        ignore_list_file_status = f"Reading from {args.ignore_list_file}"
    else:
        ignore_list_file_status = "disabled"

    # Determine output ignore list file status for configuration display.
    if args.output_ignore_list_file is not None:
        output_ignore_list_file_status = f"Outputting to {args.output_ignore_list_file}"
    else:
        output_ignore_list_file_status = "disabled"

    start_time = time.perf_counter()

    rocm_dir, rocgdb_bin, rocgdb_testsuite_dir = _resolve_rocm_paths(args)

    rocgdb_configure_script = rocgdb_testsuite_dir / "configure"

    # Handle test selection based on command-line options.
    # This must happen before print_configuration() since it displays args.tests.
    if args.gpu_tests:
        args.tests = ["gdb.rocm"]
        logger.info("Using --gpu-tests: running gdb.rocm tests only")
    elif args.gpu_corefile_tests:
        corefile_tests = _glob_posix("gdb.rocm/runtime-core*.exp", rocgdb_testsuite_dir)
        if not corefile_tests:
            _log_error_and_exit(
                "No GPU corefile tests found matching gdb.rocm/runtime-core*.exp"
            )
        args.tests = corefile_tests
        logger.info(
            f"Using --gpu-corefile-tests: found {len(corefile_tests)} test(s): "
            f"{' '.join(corefile_tests)}"
        )
    elif args.cpu_tests:
        args.tests = discover_test_directories(
            rocgdb_testsuite_dir, exclude=["gdb.rocm"]
        )
        if not args.tests:
            _log_error_and_exit("No CPU test directories found in testsuite")
        logger.info(
            f"Using --cpu-tests: discovered {len(args.tests)} CPU test directories"
        )
    elif args.tests is None:
        # Default case: no test option was specified.
        args.tests = ["gdb.rocm", "gdb.dwarf2"]
        logger.info("Using default tests: gdb.rocm and gdb.dwarf2")

    # Print env variables.
    print_env_variables()

    # Print Python information.
    print_python_info()

    # Build the compiler configurations from the selected toolchains.
    selected_toolchains = select_toolchains(args.toolchain)

    # Show configuration summary.
    print_configuration(
        rocgdb_bin,
        rocgdb_testsuite_dir,
        rocgdb_configure_script,
        rocm_dir,
        args,
        ignore_list_file_status,
        output_ignore_list_file_status,
        selected_toolchains,
    )

    # Load ignore list from JSON file.
    xfailed_tests = load_ignore_list_from_json(args.ignore_list_file)

    # Validate critical files exist.
    required_files = {
        # Windows has no launcher script (see _resolve_rocm_paths).
        "ROCgdb executable" if IS_WINDOWS else "ROCgdb launcher": rocgdb_bin,
        "ROCgdb configure script": rocgdb_configure_script,
    }
    if IS_WINDOWS:
        args.msys2_root = _windows_resolve_msys2_root(args.msys2_root)
        required_files["MSYS2 shell"] = args.msys2_root / "usr" / "bin" / "sh.exe"
    validate_required_files(required_files)

    if IS_LINUX:
        # Set core dump file limit and display info.
        setup_core_file_info()

    # Prepare environment for tests.
    env_vars = setup_environment(rocm_dir, args.try_rocm_path, args.msys2_root)
    if IS_WINDOWS:
        _windows_set_dll_directory(rocm_dir / "bin")
    setup_sanitizer_environment(env_vars, rocgdb_testsuite_dir)

    # Validate that we can run rocgdb.
    validate_rocgdb(rocgdb_bin, env_vars)
    compilers = [
        (
            TOOLCHAINS[name]["cc"],
            TOOLCHAINS[name]["cxx"],
            TOOLCHAINS[name]["fc"],
            TOOLCHAINS[name]["label"],
        )
        for name in selected_toolchains
    ]

    print_section("Expanding test paths")
    tests = expand_test_paths(args.tests, rocgdb_testsuite_dir)

    if not tests:
        _log_error_and_exit("No test files found")

    # Verify executables presence. We only require the compilers of the selected
    # toolchains, in first seen order with duplicates dropped, so a run of one
    # toolchain does not fail because another toolchain's compilers are absent.
    # amdclang++ is needed to compile GPU kernels for gdb.rocm tests; require it
    # whenever any gdb.rocm test is in scope.
    toolchain_executables = list(
        dict.fromkeys(exe for cc, cxx, fc, _ in compilers for exe in (cc, cxx, fc))
    )
    has_rocm_tests = any(t.startswith("gdb.rocm/") or t == "gdb.rocm" for t in tests)
    required_executables = ["make", *toolchain_executables, "runtest"]
    if has_rocm_tests:
        required_executables.insert(1, "amdclang++")
    # mirage is deliberately not listed here: Emulator.resolve() below looks it
    # up via find_mirage() and raises a more specific error if it is absent, so
    # checking it here too would just search the same PATH twice.
    check_executables(required_executables, env_vars)

    # Resolve the emulated GPU, if one was asked for.
    emulator = None
    if args.emulate is not None:
        emulator_support = _load_emulator_support()
        try:
            emulator = emulator_support.Emulator.resolve(
                args.emulate, rocm_dir, path=env_vars.get("PATH")
            )
        except emulator_support.EmulatorError as e:
            _log_error_and_exit(str(e))
        # A `make check` killed part-way (Ctrl-C, or a job's session SIGKILLed)
        # gives mirage no chance to tear its session down, stranding an emulated
        # GPU. The one-by-one path reclaims between tests; this sweeps whatever
        # is left at exit, so a --parallel or single-session run cannot leak one.
        # mirage cleanup leaves live sessions alone, so the final sweep is safe.
        atexit.register(emulator.cleanup)
        print_section("Emulated GPU (no hardware in use)")
        _log_aligned_fields(emulator.summary())

    # The sanity check probes the GPU with a HIP program, so it is only
    # meaningful when gdb.rocm tests will actually run. Reject the combination
    # early rather than compiling a HIP program for a CPU-only run.
    if args.sanity_check and not has_rocm_tests:
        _log_error_and_exit(
            "--sanity-check requires gdb.rocm tests to be in scope, but none "
            "were selected."
        )

    # Build the sanity-check executable once per invocation. It is rebuilt
    # unconditionally so a binary left over from a previous run (possibly for a
    # different GPU) can never cause a spurious "system unreliable" abort.
    sanity_check_exe = None
    if args.sanity_check:
        log_dir = args.one_by_one_log_dir or (rocgdb_testsuite_dir / "one_by_one_logs")
        sanity_check_exe = _build_sanity_check_executable(
            rocgdb_testsuite_dir, log_dir, env_vars
        )

    # Initialize test result tracking.
    test_results = TestResults()
    test_results.group_results = args.group_results

    # Run tests for all configured compilers.
    for cc, cxx, fc, compiler_label in compilers:
        run_tests(
            rocgdb_testsuite_dir,
            rocgdb_bin,
            env_vars,
            tests,
            cc,
            cxx,
            fc,
            compiler_label,
            test_results,
            args,
            xfailed_tests,
            sanity_check_exe,
            emulator,
        )

    # Final summaries.
    test_results.print_all_summaries()
    overall_pass = test_results.print_final_status(xfailed_tests, args.no_xfail)

    # Write the per-test timing log if requested.
    if args.timing:
        timing_path = args.timing_log_file or (
            rocgdb_testsuite_dir / TIMING_LOG_FILENAME
        )
        write_timing_log(test_results.timings, timing_path)
        logger.info(f"{STATUS_PASS} Timing log written to {timing_path}")

    # Generate and write output ignore list if requested.
    if args.output_ignore_list_file is not None:
        if len(selected_toolchains) < 2:
            logger.warning(
                f"{STATUS_WARN} --output-ignore-list-file with a single toolchain "
                "produces no Generic entries. Failures that would be Generic in a "
                "two-toolchain run are filed under the single compiler label only, "
                "so the generated list will not suppress those tests when run with "
                "all toolchains."
            )
        ignore_list = test_results.generate_ignore_list()
        try:
            with open(args.output_ignore_list_file, "w", encoding="utf-8") as f:
                json.dump(ignore_list, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Add trailing newline.
            logger.info(
                f"{STATUS_PASS} Generated ignore list written to {args.output_ignore_list_file}"
            )
        except OSError as e:
            print_section("Generated ignore list (write failed)")
            for line in json.dumps(
                ignore_list, indent=2, ensure_ascii=False
            ).splitlines():
                logger.info(line)
            _log_error_and_exit(
                f"Failed to write ignore list to {args.output_ignore_list_file}: {e}"
            )

    logger.info("")
    duration = time.perf_counter() - start_time
    logger.info(f"Total test run duration: {duration:.4f} seconds.")

    sys.exit(0 if overall_pass else 1)


def _build_runtestflags(
    rocgdb_bin: Path,
    cc: str,
    cxx: str,
    fc: str,
    optimization: str,
    runtestflags: str,
) -> str:
    """
    Build RUNTESTFLAGS string for test execution.

    Args:
        rocgdb_bin: Path to ROCgdb binary.
        cc: C compiler command.
        cxx: C++ compiler command.
        fc: Fortran compiler command.
        optimization: Optimization flags.
        runtestflags: Additional flags.

    Returns:
        Complete RUNTESTFLAGS string.
    """
    # Use forward slashes because on Windows the path ends up in Tcl, where a
    # backslash is an escape character.
    parts = [
        f"GDB={shlex.quote(rocgdb_bin.as_posix())}",
        f"CC_FOR_TARGET={shlex.quote(cc)}",
        f"CXX_FOR_TARGET={shlex.quote(cxx)}",
        f"F77_FOR_TARGET={shlex.quote(fc)}",
        f"F90_FOR_TARGET={shlex.quote(fc)}",
        "HIP_COMPILER_FOR_TARGET=amdclang++",
    ]
    if optimization:
        parts.append(f"CFLAGS_FOR_TARGET={shlex.quote(optimization)}")
    if runtestflags:
        parts.append(runtestflags)

    return " ".join(parts)


def _decode_stream(data: Union[str, bytes, None]) -> str:
    """
    Normalize a captured subprocess stream to text.

    TimeoutExpired.stdout/stderr may be bytes, str, or None depending on how
    far communicate() progressed; coerce all three to a string for logging.

    Args:
        data: A captured stream value (str, bytes, or None).

    Returns:
        The decoded text, or "" when there was nothing captured.
    """
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _build_sanity_check_executable(
    test_suite_dir: Path, log_dir: Path, env_vars: Dict[str, str]
) -> Path:
    """
    Build the sanity-check HIP executable from gdb.rocm/simple.cpp.

    simple.cpp is a self-contained HIP program with its own main() that launches
    a trivial kernel and asserts the result, so it compiles directly with
    amdclang++ and needs none of the testsuite's wrapper/driver machinery. The
    executable is written to <log_dir>/sanity_check (sanity_check.exe on
    Windows) and rebuilt unconditionally on every invocation:
    --offload-arch=native bakes in the build-time GPU, so a binary left over
    from a previous run (possibly on different hardware) must never be
    reused, or it could cause a spurious "system unreliable" abort.

    Args:
        test_suite_dir: Path to the testsuite directory (source of simple.cpp).
        log_dir: One-by-one log dir root; the executable is written here so it
            sits next to any dmesg.log the probe writes.
        env_vars: Prepared environment for the compile. Supplies the same
            PATH/LD_LIBRARY_PATH as the testsuite run so amdclang++ and its HIP
            toolchain resolve identically.

    Returns:
        Path to the freshly built sanity-check executable.

    Exits:
        Code 1 if the source is missing or the compile fails (a setup error,
        distinct from the unreliable-system condition the probe detects).
    """
    src = test_suite_dir / "gdb.rocm" / "simple.cpp"
    if not src.is_file():
        _log_error_and_exit(f"Sanity-check source not found: {src}")

    # On Windows the linker and CreateProcess both expect the .exe.
    out = log_dir / ("sanity_check.exe" if IS_WINDOWS else "sanity_check")

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log_error_and_exit(f"Failed to create sanity-check log dir {log_dir}: {e}")

    # Mirror the essential flags the testsuite uses to build gdb.rocm programs
    # (see gdb/testsuite/boards/hip.exp and the command echoed by simple.exp):
    #
    # . -x hip puts amdclang++ in HIP mode so the device headers and toolchain
    #   are on the include path. Without it the source is compiled as host C++
    #   and <hip/hip_runtime.h> is not found (and --offload-arch is reported as
    #   an unused argument).
    # . --offload-arch=native auto-detects the installed GPU.
    # . -mllvm=-amdgpu-spill-cfi-saved-regs matches the known-good recipe.
    # . -Wno-unused-command-line-argument / -Wno-unknown-warning-option keep the
    #   HIP link step quiet, as the board file does.
    cmd = [
        "amdclang++",
        "-x",
        "hip",
        "--offload-arch=native",
        "-mllvm=-amdgpu-spill-cfi-saved-regs",
        "-Wno-unused-command-line-argument",
        "-Wno-unknown-warning-option",
        str(src),
        "-o",
        str(out),
    ]
    logger.info(f"Building sanity-check executable: {shlex.join(cmd)}")
    result = _run_command(env=env_vars, cmd=cmd, capture_output=True, check=False)
    if result.returncode != 0:
        _log_error_and_exit(
            f"Failed to build sanity-check executable from {src}: {result.stderr}"
        )
    logger.info(f"{STATUS_PASS} Sanity-check executable built at {out}")
    return out


def _dump_dmesg(log_dir: Path, env_vars: Dict[str, str]) -> None:
    """
    Capture kernel ring buffer output to <log_dir>/dmesg.log.  On Windows,
    only warn that there is nothing to capture.

    On many systems reading the kernel log is restricted
    (kernel.dmesg_restrict), so a plain `dmesg` fails for non-root users. We
    first try `dmesg`, then fall back to non-interactive `sudo -n dmesg` (which
    never blocks on a password prompt). If neither yields output, we warn and
    write nothing rather than saving a misleading file containing only the
    permission error.

    Best-effort throughout: a missing binary, a non-zero exit, or a write
    failure is warned about but never raised, so it cannot mask the
    unreliable-system exit that calls it.

    Args:
        log_dir: Directory to write dmesg.log into.
        env_vars: Prepared environment for the dmesg invocation.

    Returns:
        None
    """
    if IS_WINDOWS:
        logger.warning(f"{STATUS_WARN} No kernel log to capture on Windows.")
        return

    dmesg_path = log_dir / "dmesg.log"

    def _try(cmd: List[str]) -> Optional[str]:
        try:
            result = _run_command(cmd, env=env_vars, capture_output=True, check=False)
        except SystemExit:
            # _run_command calls _log_error_and_exit on OSError (e.g. the
            # binary is missing) even with no error_msg; treat that as "this
            # command is unavailable" so diagnostics never abort the abort.
            return None
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return None

    output = _try(["dmesg"]) or _try(["sudo", "-n", "dmesg"])
    if not output:
        logger.warning(
            f"{STATUS_WARN} Could not capture dmesg (missing binary or "
            f"insufficient privileges); skipping {dmesg_path}."
        )
        return

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        dmesg_path.write_text(output, encoding="utf-8")
        logger.info(f"{STATUS_WARN} dmesg captured to {dmesg_path}")
    except OSError as e:
        logger.warning(f"{STATUS_WARN} Failed to write {dmesg_path}: {e}")


def _run_sanity_check(
    exe: Path, timeout: int, env_vars: Dict[str, str], log_dir: Path
) -> None:
    """
    Run the sanity-check executable as a GPU health probe.

    A trivial known-good HIP kernel should return well within `timeout`. If it
    times out or exits non-zero, the system is declared unreliable: dmesg is
    captured and the run aborts immediately (sys.exit(1)) rather than letting
    every remaining test fail for reasons unrelated to the code under test.

    Args:
        exe: Path to the sanity-check executable.
        timeout: Wall-clock timeout (seconds) for the probe.
        env_vars: Prepared environment for the probe.
        log_dir: One-by-one log dir root; dmesg.log is written here on failure.

    Returns:
        None on success.

    Exits:
        Code 1 if the probe times out or exits non-zero.
    """
    reason: Optional[str] = None
    output = ""
    start = time.perf_counter()
    try:
        result = _run_command(
            [str(exe)],
            env=env_vars,
            capture_output=True,
            check=False,
            timeout=timeout,
            kill_process_tree=True,
        )
    except subprocess.TimeoutExpired as exc:
        reason = f"sanity check timed out after {timeout}s"
        # TimeoutExpired carries whatever the child emitted before the kill;
        # it may be bytes or str depending on how far communicate() got.
        output = _decode_stream(exc.stdout) + _decode_stream(exc.stderr)
    else:
        if result.returncode != 0:
            reason = f"sanity check exited with code {result.returncode}"
            output = (result.stdout or "") + (result.stderr or "")
    duration = time.perf_counter() - start

    if reason is not None:
        logger.error(
            f"{STATUS_FAIL} System has become unreliable: {reason}. "
            f"Aborting the test run."
        )
        output = output.strip()
        if output:
            logger.error(f"{STATUS_FAIL} Sanity check output:")
            for line in output.splitlines():
                logger.error(f"       {line}")
        _dump_dmesg(log_dir, env_vars)
        sys.exit(1)

    print_section(f"{STATUS_PASS} Sanity check passed in {duration:.2f}s")


def _execute_one_by_one(
    test_suite_dir: Path,
    tests: List[str],
    runtestflags_str: str,
    check_type: str,
    quiet: bool,
    env_vars: Dict[str, str],
    log_dir: Path,
    compiler_label: str,
    iteration: int,
    wall_clock_timeout: int,
    test_results: Optional["TestResults"],
    sanity_check_exe: Optional[Path],
    sanity_check_timeout: int,
    emulator=None,
) -> None:
    """
    Run each test in its own `make check` invocation, capturing per-test
    gdb.log under <log_dir>/<compiler>/{pass,fail}/<group>/ and writing a
    sibling .status file with timing and iteration metadata. Per-test gdb.sum
    content is aggregated into the canonical gdb.sum so the existing
    result-parsing path consumes it unchanged.

    A test is bucketed into "fail/" if its per-test gdb.sum contains any line
    starting with FAIL:, UNRESOLVED:, or ERROR:, or if the wall-clock timeout
    fires; otherwise "pass/".

    Args:
        test_suite_dir: Path to test suite directory.
        tests: Test files to run individually (e.g. ["gdb.rocm/foo.exp"]).
        runtestflags_str: RUNTESTFLAGS string built by _build_runtestflags.
        check_type: GDB testsuite target to invoke (check / check-read1 / check-readmore).
        quiet: Capture subprocess output if True.
        env_vars: Environment variables for test execution.
        log_dir: Root directory for per-test gdb.log captures.
        compiler_label: Compiler identifier (e.g., "GCC").
        iteration: Current retry iteration (1-based) — controls log filename suffix.
        wall_clock_timeout: Wall-clock timeout (seconds) for each `make check`.
        test_results: When supplied (--timing mode), each test's duration is
            recorded via record_timing. None disables timing collection.
        sanity_check_exe: When supplied (--sanity-check mode), the pre-built
            known-good HIP executable run before each test as a GPU health
            probe. None disables the probe.
        sanity_check_timeout: Wall-clock timeout (seconds) for each probe.

    Returns:
        None

    Exits:
        Code 1 if the aggregated gdb.sum cannot be written, or if a sanity-check
        probe declares the system unreliable.
    """
    aggregated_lines: List[str] = []
    retry_number = iteration - 1
    base_suffix = "" if retry_number == 0 else f".retry-{retry_number}"

    for test in tests:
        # Probe the GPU before each test. Bails out (sys.exit(1)) if the system
        # has become unreliable, capturing dmesg.log under log_dir.
        if sanity_check_exe is not None:
            _run_sanity_check(sanity_check_exe, sanity_check_timeout, env_vars, log_dir)

        cmd = [
            "make",
            check_type,
            f"RUNTESTFLAGS={runtestflags_str}",
            f"TESTS={test}",
        ]
        # A session per test, so no test can be affected by GPU state an
        # earlier one left behind.
        if emulator is not None:
            cmd = emulator.wrap(cmd)
        logger.info(f"[one-by-one] {compiler_label} - {test}: {shlex.join(cmd)}")

        # Purge stale gdb.sum / gdb.log so a killed or failing `make` cannot
        # cause this iteration to read the previous test's leftovers.
        for stale in (test_suite_dir / "gdb.sum", test_suite_dir / "gdb.log"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(
                    f"{STATUS_WARN} {test}: failed to clear stale {stale.name}: {e}"
                )

        wall_clock_timed_out = False
        start = time.perf_counter()
        try:
            _run_command(
                cmd,
                cwd=test_suite_dir,
                env=env_vars,
                capture_output=quiet,
                check=False,
                timeout=wall_clock_timeout,
                kill_process_tree=True,
                own_console=True,
            )
        except subprocess.TimeoutExpired:
            wall_clock_timed_out = True
            logger.warning(
                f"{STATUS_WARN} {test}: wall-clock timeout after {wall_clock_timeout}s"
            )
            # SIGKILL left mirage no chance to release its session, so reclaim
            # it before the next test starts its own.
            if emulator is not None:
                emulator.cleanup()
        duration = time.perf_counter() - start

        # Read per-test gdb.sum for both bucketing and aggregation.
        gdb_sum = test_suite_dir / "gdb.sum"
        per_test_lines: List[str] = []
        if gdb_sum.is_file():
            per_test_lines = _read_file_lines(
                gdb_sum, f"during one-by-one aggregation of {test}"
            )
        elif not wall_clock_timed_out:
            # `make` either exited before runtest could produce gdb.sum or
            # runtest itself failed without writing one. Without this
            # synthesized line the test would contribute nothing to the
            # aggregated gdb.sum and silently count as a pass.
            logger.warning(
                f"{STATUS_WARN} {test}: gdb.sum not produced; "
                f"recording as UNRESOLVED."
            )
            per_test_lines.append(
                f"UNRESOLVED: {test}: one-by-one make produced no gdb.sum"
            )

        # On wall-clock timeout, synthesize a FAIL line so update_results
        # tracks the timeout. The "(timeout)" suffix triggers the existing
        # TIMEOUT classification via TIMEOUT_RE.
        if wall_clock_timed_out:
            per_test_lines.append(
                f"FAIL: {test}: one-by-one wall-clock timeout (timeout)"
            )

        aggregated_lines.extend(per_test_lines)

        # Determine pass/fail using the same definition as TestResults, and
        # pick the most informative status label by scanning for the highest
        # severity prefix present (TIMEOUT > ERROR > UNRESOLVED > FAIL > PASS).
        has_error = has_unresolved = has_fail = False
        for line in per_test_lines:
            if line.startswith("ERROR:"):
                has_error = True
            elif line.startswith("UNRESOLVED:"):
                has_unresolved = True
            elif line.startswith("FAIL:"):
                has_fail = True
        test_failed = wall_clock_timed_out or has_fail or has_unresolved or has_error

        if test_results is not None:
            test_results.record_timing(
                compiler_label, test, duration, passed=not test_failed
            )

        status_dir = "fail" if test_failed else "pass"
        if wall_clock_timed_out:
            status_text = "TIMEOUT"
        elif has_error:
            status_text = "ERROR"
        elif has_fail:
            status_text = "FAIL"
        elif has_unresolved:
            status_text = "UNRESOLVED"
        else:
            status_text = "PASS"

        test_path = Path(test)
        target_dir = log_dir / compiler_label / status_dir / test_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        log_target = target_dir / f"{test_path.stem}{base_suffix}.log"
        status_target = target_dir / f"{test_path.stem}{base_suffix}.status"

        gdb_log = test_suite_dir / "gdb.log"
        if gdb_log.is_file():
            try:
                shutil.copy2(gdb_log, log_target)
            except OSError as e:
                logger.warning(
                    f"{STATUS_WARN} {test}: failed to copy gdb.log to {log_target}: {e}"
                )
        else:
            logger.warning(
                f"{STATUS_WARN} {test}: gdb.log not produced; skipping log capture."
            )

        status_content = (
            f"test: {test}\n"
            f"compiler: {compiler_label}\n"
            f"iteration: {iteration}\n"
            f"retry_number: {retry_number}\n"
            f"status: {status_text}\n"
            f"duration_seconds: {duration:.3f}\n"
            f"wall_clock_timeout: {bool(wall_clock_timed_out)}\n"
        )
        try:
            status_target.write_text(status_content, encoding="utf-8")
        except OSError as e:
            logger.warning(
                f"{STATUS_WARN} {test}: failed to write status file {status_target}: {e}"
            )

    aggregated_sum = test_suite_dir / "gdb.sum"
    try:
        aggregated_sum.write_text("\n".join(aggregated_lines) + "\n", encoding="utf-8")
    except OSError as e:
        _log_error_and_exit(f"Failed to write aggregated gdb.sum: {e}")


def _timing_entries_total(entries: List[Tuple[str, Optional[float]]]) -> float:
    """
    Sum the measured durations in a list of (test_file, duration) entries.

    Entries whose duration is None (the test ran but never passed) contribute
    nothing, so a directory of only such entries totals zero and sorts last.

    Args:
        entries: List of (test_file, duration) pairs for one directory.

    Returns:
        Total measured duration in seconds.
    """
    return sum(duration for _, duration in entries if duration is not None)


def write_timing_log(
    timings: Dict[str, Dict[str, Optional[float]]], out_path: Path
) -> None:
    """
    Write the per-test timing log for --timing mode.

    Entries are grouped by compiler, then by test directory (e.g. gdb.rocm),
    and within each directory sorted by duration descending (longest-running
    test first). Tests that ran but never passed are shown as N/A and sorted
    after all timed tests.

    Args:
        timings: Nested mapping compiler_label -> test_file -> duration (or
            None if the test never passed).
        out_path: Destination path for the timing log.

    Exits:
        Code 1 if the file cannot be written.
    """
    lines: List[str] = [
        "# ROCgdb per-test timing (one-by-one mode)",
        "# Duration is the maximum wall-clock time of a successful run.",
        "# N/A means the test ran but never passed.",
    ]

    print_width = 80
    for compiler_label in timings.keys():
        per_compiler = timings[compiler_label]
        lines.append("")
        lines.append("=" * print_width)
        lines.append(f"Compiler: {compiler_label}")
        lines.append("=" * print_width)

        # Group this compiler's tests by directory.
        grouped: Dict[str, List[Tuple[str, Optional[float]]]] = defaultdict(list)
        for test_file, duration in per_compiler.items():
            directory = os.path.dirname(test_file) or "."
            grouped[directory].append((test_file, duration))

        # Order directories by total measured time descending. A directory with
        # only N/A entries totals zero and sorts last.
        dir_totals = {d: _timing_entries_total(grouped[d]) for d in grouped}
        for directory in sorted(grouped, key=dir_totals.__getitem__, reverse=True):
            total = dir_totals[directory]
            lines.append(f"\n{directory}  (total {total:.2f}s)")
            # Timed tests first (longest to shortest), then N/A tests by name.
            # Two-pass stable sort: name ascending first, then duration
            # descending, so N/A ties break alphabetically.
            entries = sorted(grouped[directory], key=lambda item: item[0])
            entries = sorted(
                entries,
                key=lambda item: (item[1] is not None, item[1] or 0.0),
                reverse=True,
            )
            for test_file, duration in entries:
                value = f"{duration:10.2f}s" if duration is not None else f"{'N/A':>11}"
                lines.append(f"  {value}  {test_file}")

    try:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        _log_error_and_exit(f"Failed to write timing log to {out_path}: {e}")


def print_python_info() -> None:
    """
    Print Python interpreter information including version and libpython details.

    Returns:
        None
    """
    print_section("Python information")

    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    supports_libpython = bool(sysconfig.get_config_var("Py_ENABLE_SHARED"))

    lib_path = "N/A"
    lib_version = "N/A"

    if supports_libpython:
        lib_dir = sysconfig.get_config_var("LIBDIR")
        ld_library = sysconfig.get_config_var("LDLIBRARY")
        if lib_dir and ld_library:
            lib_path = os.path.join(lib_dir, ld_library)
        lib_version = str(sysconfig.get_config_var("VERSION") or "N/A")

    lib_path_exists = lib_path != "N/A" and os.path.exists(lib_path)

    if not supports_libpython:
        supports_value = "No"
    elif lib_path_exists:
        supports_value = "Yes"
    else:
        supports_value = "Supported but libpython is missing"

    lib_path_value = (
        f"{lib_path} (missing)"
        if lib_path != "N/A" and not lib_path_exists
        else lib_path
    )

    _log_aligned_fields(
        [
            ("Executable", sys.executable),
            ("Version", version),
            ("Supports libpython", supports_value),
            ("libpython Path", lib_path_value),
            ("libpython Version", lib_version),
        ]
    )


def print_configuration(
    rocgdb_bin: Path,
    testsuite_dir: Path,
    configure_script: Path,
    rocm_tree_root: Path,
    args: argparse.Namespace,
    ignore_list_file_status: str,
    output_ignore_list_file_status: str,
    selected_toolchains: List[str],
) -> None:
    """
    Display the ROCgdb test configuration in a formatted table.

    Args:
        rocgdb_bin: Path to the ROCgdb binary.
        testsuite_dir: Path to the test suite directory.
        configure_script: Path to the configure script for the test suite.
        rocm_tree_root: Path to the root of the ROCm tree.
        args: Parsed command-line arguments containing additional configuration values.
        ignore_list_file_status: Status message for ignore list file.
        output_ignore_list_file_status: Status message for output ignore list file.
        selected_toolchains: Resolved toolchain identifiers to display, in run order.

    Returns:
        None

    Notes:
        - Prints an ASCII table summarizing binary location, directories, test selection,
          and execution settings.
        - Fields come directly from parsed CLI arguments.
    """

    if not args.parallel:
        parallel_info = "Disabled"
    elif args.fraction is not None:
        jobs = max(1, int(os.cpu_count() * args.fraction))
        parallel_info = f"Enabled ({jobs} jobs, from {args.fraction} * cpu_count)"
    elif args.jobs is not None:
        parallel_info = f"Enabled ({args.jobs} jobs)"
    else:
        parallel_info = f"Enabled ({os.cpu_count()} jobs, from cpu_count)"

    default_timeout_display = (
        f"{args.default_timeout} seconds"
        if args.default_timeout is not None
        else "Dejagnu's default"
    )

    toolchains_display = ", ".join(
        f"{name} ({TOOLCHAINS[name]['label']})" for name in selected_toolchains
    )

    one_by_one_timeout_display = (
        f"{args.one_by_one_test_timeout} seconds"
        if args.one_by_one_test_timeout is not None
        else f"{ONE_BY_ONE_DEFAULT_WALL_CLOCK_TIMEOUT} seconds (auto)"
    )

    fields = [
        ("OS", platform.system()),
        ("ROCm Directory", rocm_tree_root),
        ("ROCgdb Binary", rocgdb_bin),
        ("Testsuite Directory", testsuite_dir),
        ("Configure Script", configure_script),
        ("Tests", " ".join(args.tests)),
        ("Toolchains", toolchains_display),
        ("Check Type", args.check_type),
        ("Parallel Execution", parallel_info),
        ("One-by-one Mode", "Enabled" if args.one_by_one else "Disabled"),
    ]
    if args.one_by_one:
        fields.append(
            (
                "One-by-one Log Dir",
                args.one_by_one_log_dir or (testsuite_dir / "one_by_one_logs"),
            )
        )
        fields.append(("One-by-one Test Timeout", one_by_one_timeout_display))
        if args.sanity_check:
            fields.append(("Sanity Check", "Enabled"))
            fields.append(
                ("Sanity Check Timeout", f"{args.sanity_check_timeout} seconds")
            )
    if args.timing:
        fields.append(
            (
                "Timing Log",
                args.timing_log_file or (testsuite_dir / TIMING_LOG_FILENAME),
            )
        )
    fields.extend(
        [
            ("Use FAIL ignore list", "Not using" if args.no_xfail else "Using"),
            (
                "Retry ignored tests",
                "Enabled" if args.retry_ignored_tests else "Disabled",
            ),
            ("Group Results", "Enabled" if args.group_results else "Disabled"),
            ("Default Timeout", default_timeout_display),
            ("Retry Timeout", f"{args.retry_timeout} seconds"),
            ("Max Failed Retries", args.max_failed_retries),
            ("Optimization", args.optimization or "None"),
            ("Additional Runtest Flags", args.runtestflags or "None"),
            ("Ignore List", ignore_list_file_status),
            ("Output Ignore List", output_ignore_list_file_status),
        ]
    )
    print_section("ROCgdb Test Suite Configuration")
    _log_aligned_fields([(label, str(value)) for label, value in fields])


if __name__ == "__main__":
    main()
