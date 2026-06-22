#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import fnmatch
import glob
import json
import logging
import os
import platform
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Set, Tuple, Union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Valid tier names recognised by --tier. Mirrors VALID_TEST_CATEGORIES in
# TheRock's test_runner.py and the tiers in .github/test-runner/
# test_categories.yaml (from which gen_ctestfile.py builds CTestTestfile.cmake).
VALID_TIERS = ("quick", "standard", "comprehensive", "full")

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

# Sanitizers we pre-configure on every run. If rocgdb wasn't built with the
# corresponding -fsanitize=..., the matching *_OPTIONS env var is silently
# ignored at runtime, so listing extras here is harmless.
SANITIZERS = ("asan", "tsan", "ubsan", "msan", "lsan", "hwasan")

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


def _run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    check: bool = True,
    error_msg: Optional[str] = None,
    timeout: Optional[int] = None,
    kill_process_group: bool = False,
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
        kill_process_group: Run the command in a new process group/session and,
            on TimeoutExpired, SIGKILL the entire group so descendants
            (e.g. runtest/expect/gdb spawned by `make`) are not orphaned.

    Returns:
        CompletedProcess result.

    Exits:
        Code 1 if check=True, command exits with non-zero status, and error_msg is provided.

    Raises:
        CalledProcessError if check=True, command exits with non-zero status, and error_msg is None.
        TimeoutExpired if timeout is set and exceeded.
    """
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=kill_process_group,
        )
    except OSError as e:
        if error_msg:
            _log_error_and_exit(f"{error_msg}: {e}")
        else:
            _log_error_and_exit(f"Failed to execute {cmd[0]!r}: {e}")
    try:
        out, err = process.communicate(timeout=timeout)
    except BaseException:
        # Cover TimeoutExpired and any other escaping exception
        # (KeyboardInterrupt, OSError mid-read, etc.) so the make /
        # runtest / expect / gdb process tree is never orphaned. With a new
        # session we can SIGKILL the whole group; otherwise just the child.
        if kill_process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        try:
            out, err = process.communicate()
        except Exception:
            out, err = None, None
        raise
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
        "--optimization",
        type=str,
        default="",
        help="Optimization level to pass to compiler (e.g., -O0, -Os, -Og).",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel. Default is off."
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=_positive_nonzero_int,
        default=None,
        metavar="N",
        help="Number of parallel jobs for make (e.g., -j4). Only valid with --parallel. "
        "If --parallel is used without -j, defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not output configure/testsuite commands. Default is off.",
    )
    parser.add_argument(
        "--rocgdb-bin",
        type=Path,
        help="Path to the ROCgdb executable. Validated for existence when "
        "supplied. Default: <rocm_dir>/bin/rocgdb, where rocm_dir is "
        "auto-discovered (see --try-rocm-path).",
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
        "Mutually exclusive with --gpu-tests and --cpu-tests. "
        "Default (if no test option specified) is gdb.rocm/*.exp and gdb.dwarf2/*.exp",
    )
    test_group.add_argument(
        "--gpu-tests",
        action="store_true",
        help="Run GPU-specific tests (gdb.rocm directory only). "
        "Mutually exclusive with --tests and --cpu-tests.",
    )
    test_group.add_argument(
        "--cpu-tests",
        action="store_true",
        help="Run all CPU tests (all testsuite directories except gdb.rocm). "
        "Automatically discovers all gdb.* directories with .exp files. "
        "Mutually exclusive with --tests and --gpu-tests.",
    )
    test_group.add_argument(
        "--tier",
        type=str,
        default=os.environ.get("TEST_TYPE"),
        choices=list(VALID_TIERS) + [None],
        help="Test tier to run (quick/standard/comprehensive/full). When set, "
        "loads test_patterns from <testsuite-dir>/test_categories.yaml and "
        "uses them in place of the other selection flags. Falls back to the "
        "TEST_TYPE env var when unset. This is the primary entry point used by "
        "TheRock's test_runner.py + ctest (see the installed CTestTestfile.cmake). "
        "Mutually exclusive with --tests, --gpu-tests and --cpu-tests.",
    )

    parser.add_argument(
        "--testsuite-dir",
        type=Path,
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
        type=Path,
        required=False,
        metavar="PATH",
        help="Path to JSON file containing test ignore lists. "
        "If not specified, uses rocgdb_ignore_list.json in the same directory as this script. "
        "This option requires a value.",
    )
    parser.add_argument(
        "--output-ignore-list-file",
        type=Path,
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
        type=Path,
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

    args = parser.parse_args()

    # Validate that -j/--jobs is only used with --parallel.
    if args.jobs is not None and not args.parallel:
        _log_error_and_exit(
            "-j/--jobs can only be specified when --parallel is provided."
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

    # Validate paths independently if provided. Either flag may be supplied
    # alone; whichever is omitted is auto-discovered in _resolve_rocm_paths.
    if args.testsuite_dir is not None:
        args.testsuite_dir = validate_path(args.testsuite_dir, is_dir=True)
    if args.rocgdb_bin is not None:
        args.rocgdb_bin = validate_path(args.rocgdb_bin, is_file=True)

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


def setup_environment(artifacts_dir: Path) -> Dict[str, str]:
    """
    Configure environment variables for test suite execution.

    Args:
        artifacts_dir: Path to artifacts directory with binaries and libraries.

    Returns:
        Updated environment variables dictionary.
    """
    print_section("Setting up environment variables")

    # Copy current environment.
    env_vars = os.environ.copy()

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
    _prepend_to_path_var(env_vars, "PATH", path_parts)
    _prepend_to_path_var(env_vars, "LD_LIBRARY_PATH", ld_library_parts)

    # Check if we are running within a github actions context, where a
    # non-system version of Python is being used. If so, we have
    # pythonLocation set to the base location of the Python interpreter.
    if "pythonLocation" in env_vars:
        # Set PYTHONHOME so rocgdb can initialize Python properly.
        env_vars["PYTHONHOME"] = env_vars["pythonLocation"]
        pythonhome_value = env_vars["PYTHONHOME"]
    else:
        pythonhome_value = "not set (no pythonLocation env var)"

    _log_aligned_fields(
        [
            ("PATH (prepended)", os.pathsep.join(path_parts)),
            ("LD_LIBRARY_PATH (prepended)", os.pathsep.join(ld_library_parts)),
            ("PYTHONHOME", pythonhome_value),
        ]
    )

    return env_vars


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


def _load_test_categories_yaml(testsuite_dir: Path) -> dict:
    """
    Load test_categories.yaml from an installed testsuite tree.

    ROCgdb's source copy lives in .github/test-runner/; TheRock installs it
    next to the testsuite at <testsuite_dir>/test_categories.yaml.

    Args:
        testsuite_dir: Path to the GDB testsuite directory.

    Returns:
        Parsed YAML config as a dict.

    Exits:
        Code 1 if the YAML file is missing or PyYAML is unavailable.
    """
    yaml_path = testsuite_dir / "test_categories.yaml"
    if not yaml_path.is_file():
        _log_error_and_exit(
            f"--tier was provided but {yaml_path} does not exist. "
            "The installed ROCgdb testsuite is missing test_categories.yaml; "
            "use one of --tests/--gpu-tests/--cpu-tests instead, or rebuild "
            "against a ROCgdb that ships it."
        )

    try:
        import yaml
    except ImportError:
        _log_error_and_exit(
            "PyYAML is required to use --tier. Install it via `pip install pyyaml`."
        )

    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_tier_tests(
    tier: str, testsuite_dir: Path, fallback_tests: List[str]
) -> List[str]:
    """
    Expand a tier name to the underlying list of .exp file paths.

    Reads test_patterns + exclude from test_categories.yaml for the requested
    tier, globs the patterns relative to testsuite_dir, applies fnmatch-based
    exclude filtering, and returns a sorted, de-duplicated path list ready to
    be used as the --tests value.

    Args:
        tier: One of VALID_TIERS.
        testsuite_dir: Path to the GDB testsuite directory.
        fallback_tests: Tests to use if the tier expands to nothing (logged as
            a warning, not an error).

    Returns:
        List of .exp paths relative to testsuite_dir.

    Exits:
        Code 1 if tier is missing from the YAML.
    """
    cfg = _load_test_categories_yaml(testsuite_dir)
    categories = cfg.get("test_categories") or {}
    tier_cfg = categories.get(tier)
    if not tier_cfg:
        _log_error_and_exit(
            f"--tier '{tier}' not found in test_categories.yaml "
            f"(known tiers: {sorted(categories.keys())})."
        )

    patterns = tier_cfg.get("test_patterns") or []
    excludes = tier_cfg.get("exclude") or []

    matched: Set[str] = set()
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        if any(ch in pattern for ch in ("*", "?", "[")):
            for hit in testsuite_dir.glob(pattern):
                if hit.is_file() and hit.suffix == ".exp":
                    matched.add(str(hit.relative_to(testsuite_dir)))
        else:
            candidate = testsuite_dir / pattern
            if candidate.is_file():
                matched.add(pattern)
            else:
                logger.warning(
                    f"Tier '{tier}' references {pattern}, but it does not exist "
                    f"under {testsuite_dir}. Skipping."
                )

    filtered: List[str] = []
    for rel_path in sorted(matched):
        if any(fnmatch.fnmatch(rel_path, pat) for pat in excludes):
            continue
        filtered.append(rel_path)

    if not filtered:
        logger.warning(
            f"Tier '{tier}' expanded to an empty test list (patterns: "
            f"{patterns}, excludes: {excludes}). Falling back to: {fallback_tests}."
        )
        return list(fallback_tests)

    logger.info(
        f"Tier '{tier}' expanded to {len(filtered)} test file(s) "
        f"(from {len(patterns)} pattern(s), after {len(excludes)} exclude(s))."
    )
    return filtered


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
        if not test_path.endswith(".exp"):
            pattern = f"{test_path}/*.exp"
            matches = sorted(glob.glob(pattern, root_dir=str(testsuite_dir)))
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
            )
        else:
            cmd = [
                "make",
                args.check_type,
                f"RUNTESTFLAGS={runtestflags_str}",
                f"TESTS={' '.join(current_tests)}",
            ]
            if args.parallel:
                jobs = args.jobs or os.cpu_count()
                cmd += ["FORCE_PARALLEL=1", f"-j{jobs}"]

            logger.info(
                f"Executing tests with {compiler_label} - Iteration {iteration}: {shlex.join(cmd)}"
            )
            _run_command(
                cmd,
                cwd=test_suite_dir,
                env=env_vars,
                capture_output=args.quiet,
                check=False,
                kill_process_group=True,
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
            # Only rerun failed tests next time.
            current_tests = sorted(set(failed_tests))
            logger.info(
                f"{STATUS_FAIL} {len(failed_tests)} failing test(s) found for {compiler_label}. "
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

    if args.try_rocm_path:
        rocm_path_env = os.getenv("ROCM_PATH")
        if rocm_path_env is not None:
            tmp_dir = Path(rocm_path_env).resolve()
            if not tmp_dir.is_dir():
                _log_error_and_exit(
                    f"ROCM_PATH is set to '{tmp_dir}' but the path does not exist"
                )
            rocm_dir = tmp_dir
            logger.info(f"Using ROCM_PATH: {rocm_dir}")
        else:
            logger.info(
                "--try-rocm-path flag set but ROCM_PATH environment variable is not set. "
                "Falling back to OUTPUT_ARTIFACTS_DIR or script location."
            )

    if rocm_dir is None:
        output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
        if output_artifacts_dir is not None:
            # OUTPUT_ARTIFACTS_DIR is set, verify it exists.
            rocm_dir = Path(output_artifacts_dir).resolve()
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

    rocgdb_bin = (
        args.rocgdb_bin if args.rocgdb_bin is not None else rocm_dir / "bin" / "rocgdb"
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
    if args.tier:
        # Primary entry point for TheRock's test_runner.py + ctest: expand the
        # tier from test_categories.yaml into a concrete .exp list.
        logger.info(f"Resolving --tier '{args.tier}' against test_categories.yaml")
        args.tests = _resolve_tier_tests(
            args.tier,
            rocgdb_testsuite_dir,
            fallback_tests=["gdb.rocm", "gdb.dwarf2"],
        )
    elif args.gpu_tests:
        args.tests = ["gdb.rocm"]
        logger.info("Using --gpu-tests: running gdb.rocm tests only")
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

    # Show configuration summary.
    print_configuration(
        rocgdb_bin,
        rocgdb_testsuite_dir,
        rocgdb_configure_script,
        rocm_dir,
        args,
        ignore_list_file_status,
        output_ignore_list_file_status,
    )

    # Load ignore list from JSON file.
    xfailed_tests = load_ignore_list_from_json(args.ignore_list_file)

    # Validate critical files exist.
    validate_required_files(
        {
            "ROCgdb launcher": rocgdb_bin,
            "ROCgdb configure script": rocgdb_configure_script,
        }
    )

    if platform.system() == "Linux":
        # Set core dump file limit and display info.
        setup_core_file_info()

    # Prepare environment for tests.
    env_vars = setup_environment(rocm_dir)
    setup_sanitizer_environment(env_vars, rocgdb_testsuite_dir)

    # Validate that we can run rocgdb.
    validate_rocgdb(rocgdb_bin, env_vars)

    # Verify executables presence.
    check_executables(
        [
            "make",
            "amdclang++",
            "gcc",
            "g++",
            "gfortran",
            "clang",
            "clang++",
            "flang",
            "runtest",
        ],
        env_vars,
    )

    print_section("Expanding test paths")
    tests = expand_test_paths(args.tests, rocgdb_testsuite_dir)

    if not tests:
        _log_error_and_exit("No test files found")

    # Compiler configurations.
    compilers = [
        ("gcc", "g++", "gfortran", "GCC"),
        ("clang", "clang++", "flang", "LLVM"),
    ]

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
        )

    # Final summaries.
    test_results.print_all_summaries()
    overall_pass = test_results.print_final_status(xfailed_tests, args.no_xfail)

    # Generate and write output ignore list if requested.
    if args.output_ignore_list_file is not None:
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
    parts = [
        f"GDB={shlex.quote(str(rocgdb_bin))}",
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

    Returns:
        None

    Exits:
        Code 1 if the aggregated gdb.sum cannot be written.
    """
    aggregated_lines: List[str] = []
    retry_number = iteration - 1
    base_suffix = "" if retry_number == 0 else f".retry-{retry_number}"

    for test in tests:
        cmd = [
            "make",
            check_type,
            f"RUNTESTFLAGS={runtestflags_str}",
            f"TESTS={test}",
        ]
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
                kill_process_group=True,
            )
        except subprocess.TimeoutExpired:
            wall_clock_timed_out = True
            logger.warning(
                f"{STATUS_WARN} {test}: wall-clock timeout after {wall_clock_timeout}s"
            )
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

    Returns:
        None

    Notes:
        - Prints an ASCII table summarizing binary location, directories, test selection,
          and execution settings.
        - Fields come directly from parsed CLI arguments.
    """

    if not args.parallel:
        parallel_info = "Disabled"
    elif args.jobs is None:
        parallel_info = f"Enabled ({os.cpu_count()} jobs, from cpu_count)"
    else:
        parallel_info = f"Enabled ({args.jobs} jobs)"

    default_timeout_display = (
        f"{args.default_timeout} seconds"
        if args.default_timeout is not None
        else "Dejagnu's default"
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
    fields.extend(
        [
            ("Use FAIL ignore list", "Not using" if args.no_xfail else "Using"),
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
