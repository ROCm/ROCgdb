#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for automerge.py."""

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Import the module under test without triggering _parse_args() or WORK_DIR
# resolution (both run at import time via module-level code).
# ---------------------------------------------------------------------------

import importlib
import importlib.util
import os


def _load_automerge():
    """Load automerge as a module, neutralising its module-level side-effects."""
    spec = importlib.util.spec_from_file_location(
        "automerge",
        Path(__file__).parent / "automerge.py",
    )
    # Patch sys.argv so argparse doesn't see pytest/unittest args.
    with patch("sys.argv", ["automerge.py"]):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


am = _load_automerge()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed(returncode=0, stdout="", stderr=""):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# run() / run_net()
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    @patch("subprocess.run")
    def test_run_raises_on_nonzero_with_check(self, mock_subproc):
        mock_subproc.return_value = _completed(returncode=1, stderr="oops")
        with self.assertRaises(subprocess.CalledProcessError):
            am.run(["false"])

    @patch("subprocess.run")
    def test_run_returns_result_when_check_false(self, mock_subproc):
        mock_subproc.return_value = _completed(returncode=1, stderr="oops")
        result = am.run(["false"], check=False)
        self.assertEqual(result.returncode, 1)

    @patch("subprocess.run")
    def test_run_redacts_matching_args(self, mock_subproc):
        mock_subproc.return_value = _completed()
        with patch("builtins.print") as mock_print:
            am.run(["git", "push", "https://token@host/repo"], redact=["token"])
        printed = " ".join(str(a) for a in mock_print.call_args_list)
        self.assertNotIn("token", printed)
        self.assertIn("<redacted>", printed)


class TestRunNet(unittest.TestCase):
    @patch("time.sleep")
    def test_retries_on_network_error_then_succeeds(self, mock_sleep):
        network_err = subprocess.CalledProcessError(
            128, ["git"], "", "could not resolve host"
        )
        ok = _completed()

        call_count = 0

        def fake_run(cmd, cwd=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise network_err
            return ok

        with patch.object(am, "run", side_effect=fake_run):
            result = am.run_net(["git", "fetch"])
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    def test_raises_connectivity_error_after_all_retries(self, mock_sleep):
        network_err = subprocess.CalledProcessError(
            128, ["git"], "", "could not resolve host"
        )

        with patch.object(am, "run", side_effect=network_err):
            with self.assertRaises(am.ConnectivityError):
                am.run_net(["git", "fetch"])

        # sleep called once per inter-attempt gap
        self.assertEqual(mock_sleep.call_count, len(am.RETRY_DELAYS))

    def test_reraises_non_network_error_immediately(self):
        non_net_err = subprocess.CalledProcessError(1, ["git"], "", "permission denied")

        with patch.object(am, "run", side_effect=non_net_err):
            with self.assertRaises(subprocess.CalledProcessError):
                am.run_net(["git", "fetch"])


# ---------------------------------------------------------------------------
# find_open_conflict_pr / find_open_ff_pr
# ---------------------------------------------------------------------------


class TestFindOpenPr(unittest.TestCase):
    def test_returns_url_when_present(self):
        with patch.object(
            am,
            "run",
            return_value=_completed(stdout="https://github.com/ROCm/ROCgdb/pull/42\n"),
        ):
            url = am.find_open_conflict_pr()
        self.assertEqual(url, "https://github.com/ROCm/ROCgdb/pull/42")

    def test_returns_none_when_no_pr(self):
        with patch.object(am, "run", return_value=_completed(stdout="")):
            url = am.find_open_conflict_pr()
        self.assertIsNone(url)

    def test_raises_on_gh_cli_failure(self):
        with patch.object(
            am,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["gh"], "", "auth error"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                am.find_open_conflict_pr()

    def test_ff_returns_none_when_no_pr(self):
        with patch.object(am, "run", return_value=_completed(stdout="")):
            url = am.find_open_ff_pr()
        self.assertIsNone(url)


# ---------------------------------------------------------------------------
# probe_clean_prefix
# ---------------------------------------------------------------------------


class TestProbeCleanPrefix(unittest.TestCase):
    def _run_side_effect(self, outcomes):
        """
        outcomes: list of returncode ints, one per git-merge call.
        All other run() calls (checkout, branch -D, merge --abort) succeed.
        """
        merge_returncodes = iter(outcomes)

        def fake_run(cmd, cwd=None, check=True, **kwargs):
            if cmd[:2] == ["git", "merge"] and "--abort" not in cmd:
                rc = next(merge_returncodes)
                result = _completed(returncode=rc)
                if check and rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd, "", "conflict")
                return result
            return _completed()

        return fake_run

    def test_all_clean(self):
        commits = ["aaa", "bbb", "ccc"]
        with patch.object(am, "run", side_effect=self._run_side_effect([0, 0, 0])):
            last_clean, first_conflict = am.probe_clean_prefix(Path("/repo"), commits)
        self.assertEqual(last_clean, "ccc")
        self.assertIsNone(first_conflict)

    def test_first_conflicts(self):
        commits = ["aaa", "bbb"]
        with patch.object(am, "run", side_effect=self._run_side_effect([1])):
            last_clean, first_conflict = am.probe_clean_prefix(Path("/repo"), commits)
        self.assertIsNone(last_clean)
        self.assertEqual(first_conflict, "aaa")

    def test_partial_clean_prefix(self):
        commits = ["aaa", "bbb", "ccc"]
        with patch.object(am, "run", side_effect=self._run_side_effect([0, 1])):
            last_clean, first_conflict = am.probe_clean_prefix(Path("/repo"), commits)
        self.assertEqual(last_clean, "aaa")
        self.assertEqual(first_conflict, "bbb")

    def test_cleanup_runs_even_when_abort_raises(self):
        """Branch cleanup (checkout --detach + branch -D) must run even if merge --abort raises."""
        cleanup_calls = []

        def fake_subproc(cmd, **kwargs):
            if cmd[:2] == ["git", "merge"] and "--abort" not in cmd:
                return _completed(returncode=1, stderr="conflict")
            if "--abort" in cmd:
                # Simulate merge --abort failing (e.g. no merge in progress).
                return _completed(returncode=1, stderr="cannot abort")
            if "--detach" in cmd or "-D" in cmd:
                cleanup_calls.append(list(cmd))
            return _completed()

        with patch("subprocess.run", side_effect=fake_subproc):
            # merge --abort returns non-zero; run() with check=True will raise.
            # probe_clean_prefix calls merge --abort with check=True, so it raises.
            with self.assertRaises(subprocess.CalledProcessError):
                am.probe_clean_prefix(Path("/repo"), ["aaa"])

        self.assertEqual(len(cleanup_calls), 2)


# ---------------------------------------------------------------------------
# Early-exit gate (main integration slice — no actual git/gh calls)
# ---------------------------------------------------------------------------


class TestEarlyExitGate(unittest.TestCase):
    def _patch_main_deps(self, conflict_pr=None, ff_pr=None):
        """Patch just enough of main() to test the gate logic."""
        patches = [
            patch("pathlib.Path.mkdir"),
            patch.object(am, "run_net", return_value=_completed()),
            patch.object(am, "run", return_value=_completed(stdout="")),
            patch.object(am, "find_open_conflict_pr", return_value=conflict_pr),
            patch.object(am, "find_open_ff_pr", return_value=ff_pr),
        ]
        return patches

    def test_skips_when_conflict_pr_open(self):
        patches = self._patch_main_deps(
            conflict_pr="https://github.com/ROCm/ROCgdb/pull/1"
        )
        # Also patch repo/.git exists check so clone is skipped.
        with patch("pathlib.Path.exists", return_value=True), patches[0], patches[
            1
        ], patches[2], patches[3], patches[4]:
            with patch.object(am, "probe_clean_prefix") as mock_probe:
                am.main()
                mock_probe.assert_not_called()

    def test_skips_when_ff_pr_open(self):
        patches = self._patch_main_deps(ff_pr="https://github.com/ROCm/ROCgdb/pull/2")
        with patch("pathlib.Path.exists", return_value=True), patches[0], patches[
            1
        ], patches[2], patches[3], patches[4]:
            with patch.object(am, "probe_clean_prefix") as mock_probe:
                am.main()
                mock_probe.assert_not_called()


# ---------------------------------------------------------------------------
# Dry-run end-to-end flow
# ---------------------------------------------------------------------------


class TestDryRun(unittest.TestCase):
    """
    Exercises the full main() flow with --dry-run set, verifying that
    probe_clean_prefix runs but no push or PR-creation calls are made.
    """

    def _run_dry(self, probe_result):
        """
        Run main() with DRY_RUN=True, returning the set of run_net call args.
        probe_result: (last_clean, first_conflict) tuple returned by probe.
        """
        run_net_calls = []

        def fake_run(cmd, cwd=None, check=True, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            # Return a plausible stdout for commands that produce output.
            if "remote" in cmd and cmd[-1] == "remote":
                return _completed(stdout="origin\nsourceware\n")
            if "merge-base" in cmd:
                return _completed(stdout="base0000000000000000000000000000000000000000")
            if "log" in cmd and "--format=%H" in cmd:
                return _completed(stdout="aaa\nbbb\nccc")
            if "log" in cmd and "--format=%H %s" in cmd:
                return _completed(stdout="aaa subject (Author <a@b.com>)")
            if "diff" in cmd:
                return _completed(stdout="file.c\n")
            return _completed()

        def fake_run_net(cmd, cwd=None):
            run_net_calls.append(list(cmd))
            return _completed()

        with patch("pathlib.Path.mkdir"), patch(
            "pathlib.Path.exists", return_value=True
        ), patch.object(am, "run", side_effect=fake_run), patch.object(
            am, "run_net", side_effect=fake_run_net
        ), patch.object(
            am, "find_open_conflict_pr", return_value=None
        ), patch.object(
            am, "find_open_ff_pr", return_value=None
        ), patch.object(
            am, "probe_clean_prefix", return_value=probe_result
        ), patch.object(
            am, "open_ff_pr"
        ) as mock_ff_pr, patch.object(
            am, "open_conflict_pr"
        ) as mock_conflict_pr, patch.object(
            am, "push_master_mirror"
        ) as mock_mirror, patch.object(
            am, "DRY_RUN", True
        ):
            am.main()

        return run_net_calls, mock_ff_pr, mock_conflict_pr, mock_mirror

    def test_dry_run_all_clean_skips_push_and_pr(self):
        run_net_calls, mock_ff_pr, mock_conflict_pr, mock_mirror = self._run_dry(
            probe_result=("ccc", None)
        )
        # Only the two fetches should have gone out, no push.
        push_calls = [c for c in run_net_calls if "push" in c]
        self.assertEqual(push_calls, [])
        mock_ff_pr.assert_not_called()
        mock_mirror.assert_not_called()

    def test_dry_run_conflict_skips_push_and_pr(self):
        run_net_calls, mock_ff_pr, mock_conflict_pr, mock_mirror = self._run_dry(
            probe_result=(None, "aaa")
        )
        push_calls = [c for c in run_net_calls if "push" in c]
        self.assertEqual(push_calls, [])
        mock_conflict_pr.assert_not_called()
        mock_mirror.assert_not_called()

    def test_dry_run_partial_prefix_skips_push_and_pr(self):
        run_net_calls, mock_ff_pr, mock_conflict_pr, mock_mirror = self._run_dry(
            probe_result=("aaa", "bbb")
        )
        push_calls = [c for c in run_net_calls if "push" in c]
        self.assertEqual(push_calls, [])
        mock_ff_pr.assert_not_called()
        mock_mirror.assert_not_called()


if __name__ == "__main__":
    unittest.main()
