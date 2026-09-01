#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pure logic in rocjitsu_emulator.py.

These cover profile-alias resolution, the parallel-job/memory arithmetic, and
the build-target check. None of them need a GPU or the mirage binary, so they
run anywhere `python3` does:

    python3 -m unittest test_rocjitsu_emulator
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocjitsu_emulator as rj  # noqa: E402


class NormalizeProfileTests(unittest.TestCase):
    def test_known_profiles_map_to_arch(self):
        self.assertEqual(rj.normalize_profile("mi300x"), ("mi300x", "gfx942"))
        self.assertEqual(rj.normalize_profile("mi350x"), ("mi350x", "gfx950"))
        self.assertEqual(rj.normalize_profile("mi450x"), ("mi450x", "gfx1250"))

    def test_trailing_x_is_optional(self):
        # CI often names families without the trailing x.
        self.assertEqual(rj.normalize_profile("mi350"), ("mi350x", "gfx950"))
        self.assertEqual(rj.normalize_profile("mi300"), ("mi300x", "gfx942"))

    def test_gfx_target_resolves_to_profile(self):
        self.assertEqual(rj.normalize_profile("gfx942"), ("mi300x", "gfx942"))
        self.assertEqual(rj.normalize_profile("gfx1250"), ("mi450x", "gfx1250"))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(rj.normalize_profile("  MI300X "), ("mi300x", "gfx942"))
        self.assertEqual(rj.normalize_profile("GFX950"), ("mi350x", "gfx950"))

    def test_unknown_spec_raises(self):
        for bad in ("", "foobar", "gfx999", "mi350z"):
            with self.assertRaises(rj.EmulatorError):
                rj.normalize_profile(bad)


class ParallelJobsTests(unittest.TestCase):
    def test_requested_fits_is_honoured_quietly(self):
        advice = rj.parallel_jobs(2, 8, available_gb=64)
        self.assertEqual(advice.jobs, 2)
        self.assertFalse(advice.warn)
        self.assertEqual(advice.message, "")

    def test_requested_over_capacity_warns_but_is_honoured(self):
        advice = rj.parallel_jobs(16, 32, available_gb=12)
        self.assertEqual(advice.jobs, 16)
        self.assertTrue(advice.warn)
        self.assertIn("fit here", advice.message)

    def test_requested_below_one_job_warns(self):
        # 7.5GB is under BASE(8) + JOB(1); even --jobs 1 should warn.
        advice = rj.parallel_jobs(1, 8, available_gb=7.5)
        self.assertEqual(advice.jobs, 1)
        self.assertTrue(advice.warn)

    def test_auto_uses_cpus_when_memory_is_ample(self):
        advice = rj.parallel_jobs(None, 8, available_gb=64)
        self.assertEqual(advice.jobs, 8)
        self.assertFalse(advice.warn)

    def test_auto_is_capped_by_ceiling(self):
        advice = rj.parallel_jobs(None, 64, available_gb=1000)
        self.assertEqual(advice.jobs, rj.JOB_CEILING)
        self.assertFalse(advice.warn)
        self.assertIn("ceiling", advice.message)

    def test_auto_is_capped_by_memory(self):
        # (12 - 8) / 1 == 4 jobs fit.
        advice = rj.parallel_jobs(None, 32, available_gb=12)
        self.assertEqual(advice.jobs, 4)
        self.assertFalse(advice.warn)
        self.assertIn("available memory", advice.message)

    def test_auto_below_one_job_still_runs_one_and_warns(self):
        advice = rj.parallel_jobs(None, 8, available_gb=7.5)
        self.assertEqual(advice.jobs, 1)
        self.assertTrue(advice.warn)

    def test_unknown_memory_falls_back_to_cpus_without_warning(self):
        with mock.patch.object(rj, "_available_memory_gb", return_value=None):
            advice = rj.parallel_jobs(None, 8)
        self.assertEqual(advice.jobs, 8)
        self.assertFalse(advice.warn)


class CheckProfileArchTests(unittest.TestCase):
    def _make_tree(self, targets):
        """Return a rocm_root Path whose dist_info.json holds `targets`.

        `targets` may be any JSON value, or the sentinel `...` to skip writing
        the whole top-level dict, or `None` to skip the file entirely.
        """
        root = Path(self._tmp.name)
        if targets is _SKIP_FILE:
            return root
        info = root / "share" / "therock" / "dist_info.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        if targets is _RAW_BAD_JSON:
            info.write_text("{ not valid json")
        elif targets is _TOP_LEVEL_LIST:
            info.write_text(json.dumps(["gfx942"]))
        else:
            info.write_text(json.dumps({"dist_amdgpu_targets": targets}))
        return root

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_missing_file_is_left_alone(self):
        root = self._make_tree(_SKIP_FILE)
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_matching_target_passes(self):
        root = self._make_tree("gfx942 gfx950")
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_comma_separated_targets(self):
        root = self._make_tree("gfx942,gfx950")
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_semicolon_separated_targets(self):
        # The nightly tarballs write cmake's own ';' list separator.
        root = self._make_tree("gfx942;gfx950")
        self.assertIsNone(rj.check_profile_arch(root, "mi350x", "gfx950"))

    def test_mismatch_raises_with_hint(self):
        root = self._make_tree("gfx950")
        with self.assertRaises(rj.EmulatorError) as ctx:
            rj.check_profile_arch(root, "mi300x", "gfx942")
        self.assertIn("gfx942", str(ctx.exception))

    def test_empty_targets_is_left_alone(self):
        root = self._make_tree("")
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_non_string_targets_is_left_alone(self):
        root = self._make_tree(["gfx942"])
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_top_level_not_a_dict_is_left_alone(self):
        root = self._make_tree(_TOP_LEVEL_LIST)
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))

    def test_malformed_json_is_left_alone(self):
        root = self._make_tree(_RAW_BAD_JSON)
        self.assertIsNone(rj.check_profile_arch(root, "mi300x", "gfx942"))


# Sentinels for _make_tree, distinct from any real JSON value it might write.
_SKIP_FILE = object()
_RAW_BAD_JSON = object()
_TOP_LEVEL_LIST = object()


if __name__ == "__main__":
    unittest.main()
