#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for update_therock_deps.py."""

import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Load module under test without triggering __main__, then register it in
# sys.modules so @patch("update_therock_deps.*") resolves correctly.
# ---------------------------------------------------------------------------


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "update_therock_deps",
        Path(__file__).parent / "update_therock_deps.py",
    )
    mod = importlib.util.module_from_spec(spec)
    with patch("sys.argv", ["update_therock_deps.py"]):
        spec.loader.exec_module(mod)
    sys.modules["update_therock_deps"] = mod
    return mod


m = _load_module()

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SHA = "a" * 40
SHA2 = "b" * 40
DIGEST = "sha256:" + "c" * 64
DIGEST2 = "sha256:" + "d" * 64
IMAGE = f"ghcr.io/rocm/therock_build_manylinux_x86_64@{DIGEST}"
IMAGE2 = f"ghcr.io/rocm/therock_build_manylinux_x86_64@{DIGEST2}"

BASE_PINS = {
    "therock_commit_ref": SHA,
    "therock_commit_created": "2026-07-30T10:07:58Z",
    "build_image": IMAGE,
    "build_image_created": "2026-07-27T09:37:30Z",
}


def _pins(**overrides):
    return {**BASE_PINS, **overrides}


def _make_response(body: bytes | dict, headers: dict | None = None):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers = MagicMock()
    resp.headers.get = lambda key, default="": (headers or {}).get(key, default)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="err", hdrs={}, fp=None
    )


# ---------------------------------------------------------------------------
# urlopen_with_retry
# ---------------------------------------------------------------------------


class TestUrlOpenWithRetry(unittest.TestCase):
    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_success_first_attempt(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.return_value = resp
        result = m.urlopen_with_retry("http://example.com")
        self.assertEqual(result, resp)
        mock_sleep.assert_not_called()

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_5xx_then_succeeds(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.side_effect = [_http_error(503), _http_error(503), resp]
        result = m.urlopen_with_retry("http://example.com")
        self.assertEqual(result, resp)
        self.assertEqual(mock_open.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_raises_immediately_on_4xx(self, mock_open, mock_sleep):
        mock_open.side_effect = _http_error(404)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            m.urlopen_with_retry("http://example.com")
        self.assertEqual(ctx.exception.code, 404)
        mock_open.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_raises_immediately_on_401(self, mock_open, mock_sleep):
        mock_open.side_effect = _http_error(401)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            m.urlopen_with_retry("http://example.com")
        self.assertEqual(ctx.exception.code, 401)
        mock_open.assert_called_once()

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_raises_after_max_retries_on_5xx(self, mock_open, mock_sleep):
        mock_open.side_effect = _http_error(500)
        with self.assertRaises(urllib.error.HTTPError):
            m.urlopen_with_retry("http://example.com")
        self.assertEqual(mock_open.call_count, m.MAX_RETRIES)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_url_error(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.side_effect = [urllib.error.URLError("transient"), resp]
        result = m.urlopen_with_retry("http://example.com")
        self.assertEqual(result, resp)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_raises_after_max_retries_on_url_error(self, mock_open, mock_sleep):
        mock_open.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(urllib.error.URLError):
            m.urlopen_with_retry("http://example.com")
        self.assertEqual(mock_open.call_count, m.MAX_RETRIES)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_sleep_uses_exponential_backoff(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.side_effect = [
            urllib.error.URLError("err"),
            urllib.error.URLError("err"),
            resp,
        ]
        m.urlopen_with_retry("http://example.com")
        self.assertEqual(mock_sleep.call_args_list[0][0][0], 2)
        self.assertEqual(mock_sleep.call_args_list[1][0][0], 4)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_timeout_error(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.side_effect = [TimeoutError("timed out"), resp]
        result = m.urlopen_with_retry("http://example.com")
        self.assertEqual(result, resp)
        self.assertEqual(mock_open.call_count, 2)

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_raises_immediately_on_499(self, mock_open, mock_sleep):
        mock_open.side_effect = _http_error(499)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            m.urlopen_with_retry("http://example.com")
        self.assertEqual(ctx.exception.code, 499)
        mock_open.assert_called_once()

    @patch("update_therock_deps.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_exactly_500(self, mock_open, mock_sleep):
        resp = _make_response(b"ok")
        mock_open.side_effect = [_http_error(500), resp]
        result = m.urlopen_with_retry("http://example.com")
        self.assertEqual(result, resp)
        self.assertEqual(mock_open.call_count, 2)


# ---------------------------------------------------------------------------
# build_commit_message
# ---------------------------------------------------------------------------


class TestBuildCommitMessage(unittest.TestCase):
    def test_both_changed(self):
        pins = {"therock_commit_ref": SHA, "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA2, DIGEST2)
        self.assertIn(f"commit: {SHA[:12]} -> {SHA2[:12]}", msg)
        self.assertIn("build image:", msg)

    def test_only_commit_changed(self):
        pins = {"therock_commit_ref": SHA, "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA2, DIGEST)
        self.assertIn("commit:", msg)
        self.assertNotIn("build image:", msg)

    def test_only_digest_changed(self):
        pins = {"therock_commit_ref": SHA, "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA, DIGEST2)
        self.assertNotIn("commit:", msg)
        self.assertIn("build image:", msg)

    def test_nothing_changed(self):
        pins = {"therock_commit_ref": SHA, "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA, DIGEST)
        self.assertNotIn("commit:", msg)
        self.assertNotIn("build image:", msg)

    def test_empty_old_commit_shows_none(self):
        pins = {"therock_commit_ref": "", "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA2, DIGEST)
        self.assertIn("(none)", msg)
        self.assertIn(f"-> {SHA2[:12]}", msg)

    def test_missing_old_commit_key_shows_none(self):
        pins = {"build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA2, DIGEST)
        self.assertIn("(none)", msg)

    def test_malformed_old_digest_not_shown(self):
        pins = {"therock_commit_ref": SHA, "build_image": "ghcr.io/foo/bar"}
        msg = m.build_commit_message(pins, SHA, DIGEST2)
        self.assertNotIn("build image:", msg)

    def test_old_image_without_at_not_shown(self):
        pins = {"therock_commit_ref": SHA, "build_image": "ghcr.io/foo/bar:latest"}
        msg = m.build_commit_message(pins, SHA, DIGEST2)
        self.assertNotIn("build image:", msg)

    def test_title_always_present(self):
        msg = m.build_commit_message({}, SHA, DIGEST)
        self.assertIn("Update TheRock dependencies", msg)

    def test_digest_prefix_shown(self):
        pins = {"therock_commit_ref": SHA, "build_image": IMAGE}
        msg = m.build_commit_message(pins, SHA, DIGEST2)
        self.assertIn(DIGEST[:19], msg)
        self.assertIn(DIGEST2[:19], msg)

    def test_missing_build_image_key_not_shown(self):
        pins = {"therock_commit_ref": SHA}
        msg = m.build_commit_message(pins, SHA, DIGEST2)
        self.assertNotIn("build image:", msg)


# ---------------------------------------------------------------------------
# read_deps / write_deps
# ---------------------------------------------------------------------------


class TestReadWriteDeps(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mktemp(suffix=".json"))
        self._orig = m.CONFIG_FILE
        m.CONFIG_FILE = self._tmp

    def tearDown(self):
        m.CONFIG_FILE = self._orig
        self._tmp.unlink(missing_ok=True)

    def test_round_trip(self):
        m.write_deps(BASE_PINS)
        result = m.read_deps()
        self.assertEqual(result, BASE_PINS)

    def test_write_produces_valid_json(self):
        m.write_deps(BASE_PINS)
        parsed = json.loads(self._tmp.read_text())
        self.assertEqual(parsed, BASE_PINS)

    def test_write_adds_trailing_newline(self):
        m.write_deps(BASE_PINS)
        self.assertTrue(self._tmp.read_text().endswith("\n"))

    def test_write_uses_indent_2(self):
        m.write_deps({"k": "v"})
        content = self._tmp.read_text()
        self.assertIn("  ", content)


# ---------------------------------------------------------------------------
# ensure_label
# ---------------------------------------------------------------------------


class TestEnsureLabel(unittest.TestCase):
    @patch("update_therock_deps.run")
    def test_label_found(self, mock_run):
        mock_run.return_value = MagicMock(stdout=json.dumps([{"name": "therock-deps"}]))
        m.ensure_label("therock-deps")  # should not raise

    @patch("update_therock_deps.run")
    def test_label_not_found_raises(self, mock_run):
        mock_run.return_value = MagicMock(stdout=json.dumps([{"name": "other"}]))
        with self.assertRaises(ValueError) as ctx:
            m.ensure_label("therock-deps")
        self.assertIn("therock-deps", str(ctx.exception))

    @patch("update_therock_deps.run")
    def test_empty_list_raises(self, mock_run):
        mock_run.return_value = MagicMock(stdout="[]")
        with self.assertRaises(ValueError):
            m.ensure_label("therock-deps")

    @patch("update_therock_deps.run")
    def test_partial_match_in_list_still_raises(self, mock_run):
        # "therock-deps-old" is present but not "therock-deps" — must not match as substring
        mock_run.return_value = MagicMock(
            stdout=json.dumps([{"name": "therock-deps-old"}])
        )
        with self.assertRaises(ValueError):
            m.ensure_label("therock-deps")


# ---------------------------------------------------------------------------
# find_open_update_pr
# ---------------------------------------------------------------------------


class TestFindOpenUpdatePr(unittest.TestCase):
    @patch("update_therock_deps.run")
    def test_returns_url_when_present(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="https://github.com/ROCm/ROCgdb/pull/1\n"
        )
        result = m.find_open_update_pr()
        self.assertEqual(result, "https://github.com/ROCm/ROCgdb/pull/1")

    @patch("update_therock_deps.run")
    def test_returns_none_when_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        self.assertIsNone(m.find_open_update_pr())

    @patch("update_therock_deps.run")
    def test_returns_none_on_whitespace(self, mock_run):
        mock_run.return_value = MagicMock(stdout="   \n")
        self.assertIsNone(m.find_open_update_pr())


# ---------------------------------------------------------------------------
# get_therock_commit
# ---------------------------------------------------------------------------


class TestGetTheRockCommit(unittest.TestCase):
    def _branch_resp(self, sha=SHA):
        return _make_response({"commit": {"sha": sha}})

    def _commit_resp(self, date="2026-08-10T12:34:56Z"):
        return _make_response({"commit": {"committer": {"date": date}}})

    @patch("update_therock_deps.urlopen_with_retry")
    def test_returns_sha_and_timestamp(self, mock_open):
        mock_open.side_effect = [self._branch_resp(), self._commit_resp()]
        sha, ts = m.get_therock_commit(fetch_timestamp=True)
        self.assertEqual(sha, SHA)
        self.assertEqual(ts, "2026-08-10T12:34:56Z")

    @patch("update_therock_deps.urlopen_with_retry")
    def test_no_timestamp_skips_commits_api(self, mock_open):
        mock_open.side_effect = [self._branch_resp()]
        sha, ts = m.get_therock_commit(fetch_timestamp=False)
        self.assertEqual(sha, SHA)
        self.assertIsNone(ts)
        mock_open.assert_called_once()

    @patch("update_therock_deps.urlopen_with_retry")
    def test_invalid_sha_raises(self, mock_open):
        mock_open.side_effect = [_make_response({"commit": {"sha": "notasha"}})]
        with self.assertRaises(ValueError) as ctx:
            m.get_therock_commit()
        self.assertIn("Invalid TheRock SHA", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_empty_sha_raises(self, mock_open):
        mock_open.side_effect = [_make_response({})]
        with self.assertRaises(ValueError):
            m.get_therock_commit()

    @patch("update_therock_deps.urlopen_with_retry")
    def test_null_commit_in_branch_response_raises(self, mock_open):
        mock_open.side_effect = [_make_response({"commit": None})]
        with self.assertRaises(ValueError):
            m.get_therock_commit()

    @patch("update_therock_deps.urlopen_with_retry")
    def test_missing_commit_object_raises(self, mock_open):
        mock_open.side_effect = [
            self._branch_resp(),
            _make_response({"message": "Not Found"}),
        ]
        with self.assertRaises(ValueError) as ctx:
            m.get_therock_commit()
        self.assertIn("no commit object", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_null_committer_raises(self, mock_open):
        mock_open.side_effect = [
            self._branch_resp(),
            _make_response({"commit": {"committer": None}}),
        ]
        with self.assertRaises(ValueError) as ctx:
            m.get_therock_commit()
        self.assertIn("no committer", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_missing_committer_date_raises(self, mock_open):
        mock_open.side_effect = [
            self._branch_resp(),
            _make_response({"commit": {"committer": {"name": "Bot"}}}),
        ]
        with self.assertRaises(ValueError) as ctx:
            m.get_therock_commit()
        self.assertIn("no committer date", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_uses_gh_token_when_set(self, mock_open):
        mock_open.side_effect = [self._branch_resp(), self._commit_resp()]
        with patch.dict("os.environ", {"GH_TOKEN": "mytoken"}):
            m.get_therock_commit()
        req = mock_open.call_args_list[0][0][0]
        self.assertIn("mytoken", req.get_header("Authorization"))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_no_gh_token_no_auth_header(self, mock_open):
        mock_open.side_effect = [self._branch_resp(), self._commit_resp()]
        env = {k: v for k, v in __import__("os").environ.items() if k != "GH_TOKEN"}
        with patch.dict("os.environ", env, clear=True):
            m.get_therock_commit()
        req = mock_open.call_args_list[0][0][0]
        self.assertIsNone(req.get_header("Authorization"))


# ---------------------------------------------------------------------------
# get_build_digest
# ---------------------------------------------------------------------------


class TestGetBuildDigest(unittest.TestCase):
    CONFIG_DIGEST = "sha256:" + "e" * 64

    def _token_resp(self):
        return _make_response({"token": "ghcrtoken"})

    def _manifest_resp(self, image_digest=DIGEST, config_digest=None):
        if config_digest is None:
            config_digest = self.CONFIG_DIGEST
        return _make_response(
            {"config": {"digest": config_digest}},
            headers={"Docker-Content-Digest": image_digest},
        )

    def _config_blob_resp(self, created="2026-07-27T09:37:30.626203739Z"):
        return _make_response({"created": created})

    @patch("update_therock_deps.urlopen_with_retry")
    def test_returns_digest_and_timestamp(self, mock_open):
        mock_open.side_effect = [
            self._token_resp(),
            self._manifest_resp(),
            self._config_blob_resp(),
        ]
        digest, created = m.get_build_digest(fetch_timestamp=True)
        self.assertEqual(digest, DIGEST)
        self.assertEqual(created, "2026-07-27T09:37:30.626203739Z")

    @patch("update_therock_deps.urlopen_with_retry")
    def test_no_timestamp_skips_config_blob(self, mock_open):
        mock_open.side_effect = [self._token_resp(), self._manifest_resp()]
        digest, created = m.get_build_digest(fetch_timestamp=False)
        self.assertEqual(digest, DIGEST)
        self.assertIsNone(created)
        self.assertEqual(mock_open.call_count, 2)

    @patch("update_therock_deps.urlopen_with_retry")
    def test_missing_token_raises(self, mock_open):
        mock_open.side_effect = [_make_response({"error": "unauthorized"})]
        with self.assertRaises(ValueError) as ctx:
            m.get_build_digest()
        self.assertIn("no token", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_null_token_raises(self, mock_open):
        mock_open.side_effect = [_make_response({"token": None})]
        with self.assertRaises(ValueError) as ctx:
            m.get_build_digest()
        self.assertIn("no token", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_invalid_digest_raises(self, mock_open):
        resp = _make_response(
            {"config": {"digest": self.CONFIG_DIGEST}},
            headers={"Docker-Content-Digest": "notadigest"},
        )
        mock_open.side_effect = [self._token_resp(), resp]
        with self.assertRaises(ValueError) as ctx:
            m.get_build_digest()
        self.assertIn("Invalid build image digest", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_missing_config_key_raises(self, mock_open):
        resp = _make_response({}, headers={"Docker-Content-Digest": DIGEST})
        mock_open.side_effect = [self._token_resp(), resp]
        with self.assertRaises(ValueError) as ctx:
            m.get_build_digest()
        self.assertIn("config blob digest", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_missing_created_field_raises(self, mock_open):
        mock_open.side_effect = [
            self._token_resp(),
            self._manifest_resp(),
            _make_response({}),
        ]
        with self.assertRaises(ValueError) as ctx:
            m.get_build_digest()
        self.assertIn("'created'", str(ctx.exception))

    @patch("update_therock_deps.urlopen_with_retry")
    def test_nanosecond_precision_preserved(self, mock_open):
        ts = "2026-07-27T09:37:30.626203739Z"
        mock_open.side_effect = [
            self._token_resp(),
            self._manifest_resp(),
            self._config_blob_resp(created=ts),
        ]
        _, created = m.get_build_digest()
        self.assertEqual(created, ts)


# ---------------------------------------------------------------------------
# open_update_pr
# ---------------------------------------------------------------------------


PR_ARGS = (
    "my-branch",
    SHA,
    "2026-08-10T12:34:56Z",
    False,
    DIGEST,
    "2026-08-10T00:00:00Z",
)


class TestOpenUpdatePr(unittest.TestCase):
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.ensure_label")
    def test_returns_pr_url(self, mock_label, mock_run):
        mock_run.return_value = MagicMock(
            stdout="https://github.com/ROCm/ROCgdb/pull/42\n"
        )
        url = m.open_update_pr(*PR_ARGS)
        self.assertEqual(url, "https://github.com/ROCm/ROCgdb/pull/42")

    @patch("update_therock_deps.run")
    @patch("update_therock_deps.ensure_label")
    def test_raises_on_empty_url(self, mock_label, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        with self.assertRaises(ValueError) as ctx:
            m.open_update_pr(*PR_ARGS)
        self.assertIn("no PR URL", str(ctx.exception))

    @patch("update_therock_deps.run")
    @patch("update_therock_deps.ensure_label")
    def test_calls_ensure_label_before_create(self, mock_label, mock_run):
        mock_run.return_value = MagicMock(
            stdout="https://github.com/ROCm/ROCgdb/pull/1"
        )
        m.open_update_pr(*PR_ARGS)
        mock_label.assert_called_once_with(m.UPDATE_LABEL)

    @patch("update_therock_deps.run")
    @patch("update_therock_deps.ensure_label")
    def test_body_contains_commit_ref_and_date(self, mock_label, mock_run):
        mock_run.return_value = MagicMock(
            stdout="https://github.com/ROCm/ROCgdb/pull/1"
        )
        m.open_update_pr(
            "branch", SHA, "2026-08-10T12:34:56Z", False, DIGEST, "2026-08-10T00:00:00Z"
        )
        body = mock_run.call_args[0][0][mock_run.call_args[0][0].index("--body") + 1]
        self.assertIn(SHA[:12], body)
        self.assertIn("2026-08-10", body)
        self.assertNotIn("digest", body.lower())

    @patch("update_therock_deps.run")
    @patch("update_therock_deps.ensure_label")
    def test_body_contains_digest_when_digest_changed(self, mock_label, mock_run):
        mock_run.return_value = MagicMock(
            stdout="https://github.com/ROCm/ROCgdb/pull/1"
        )
        m.open_update_pr(
            "branch", SHA, "2026-08-10T12:34:56Z", True, DIGEST2, "2026-08-11T09:00:00Z"
        )
        body = mock_run.call_args[0][0][mock_run.call_args[0][0].index("--body") + 1]
        self.assertIn(DIGEST2[:19], body)
        self.assertIn("2026-08-11", body)


# ---------------------------------------------------------------------------
# run_update
# ---------------------------------------------------------------------------


class TestRunUpdate(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mktemp(suffix=".json"))
        self._orig_cf = m.CONFIG_FILE
        m.CONFIG_FILE = self._tmp

    def tearDown(self):
        m.CONFIG_FILE = self._orig_cf
        self._tmp.unlink(missing_ok=True)

    def _write_pins(self, pins=None):
        self._tmp.write_text(json.dumps(pins or BASE_PINS, indent=2) + "\n")

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    def test_raises_if_config_missing(self, *_):
        self._tmp.unlink(missing_ok=True)
        with self.assertRaises(FileNotFoundError):
            m.run_update(dry_run=False)

    @patch("update_therock_deps.read_deps")
    @patch("update_therock_deps.find_open_update_pr")
    def test_skips_if_open_pr_exists(self, mock_find, mock_read):
        self._write_pins()
        mock_find.return_value = "https://github.com/ROCm/ROCgdb/pull/99"
        result = m.run_update(dry_run=False)
        self.assertEqual(result, "skipped-existing-pr")
        mock_read.assert_not_called()

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_not_needed_when_nothing_changed(self, mock_find, mock_commit, mock_digest):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA, "2026-07-30T10:07:58Z")
        mock_digest.return_value = (DIGEST, "2026-07-27T09:37:30Z")
        result = m.run_update(dry_run=False)
        self.assertEqual(result, "not-needed")
        mock_commit.assert_called_once_with(fetch_timestamp=True)
        mock_digest.assert_called_once_with(fetch_timestamp=True)

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    def test_dry_run_returns_dry_run_when_changed(self, mock_commit, mock_digest):
        self._write_pins()
        mock_commit.return_value = (SHA2, None)
        mock_digest.return_value = (DIGEST2, None)
        result = m.run_update(dry_run=True)
        self.assertEqual(result, "dry-run")

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    def test_dry_run_not_needed_when_unchanged(self, mock_commit, mock_digest):
        self._write_pins()
        mock_commit.return_value = (SHA, None)
        mock_digest.return_value = (DIGEST, None)
        result = m.run_update(dry_run=True)
        self.assertEqual(result, "not-needed")

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    def test_raises_on_missing_therock_commit_created(self, mock_commit, mock_digest):
        pins = {k: v for k, v in BASE_PINS.items() if k != "therock_commit_created"}
        self._write_pins(pins)
        mock_commit.return_value = (SHA2, None)
        mock_digest.return_value = (DIGEST2, None)
        with self.assertRaises(ValueError) as ctx:
            m.run_update(dry_run=True)
        self.assertIn("therock_commit_created", str(ctx.exception))

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    def test_raises_on_missing_build_image_created(self, mock_commit, mock_digest):
        pins = {k: v for k, v in BASE_PINS.items() if k != "build_image_created"}
        self._write_pins(pins)
        mock_commit.return_value = (SHA2, None)
        mock_digest.return_value = (DIGEST2, None)
        with self.assertRaises(ValueError) as ctx:
            m.run_update(dry_run=True)
        self.assertIn("build_image_created", str(ctx.exception))

    @patch("update_therock_deps.open_update_pr")
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_full_update_both_changed_writes_correct_pins(
        self, mock_find, mock_commit, mock_digest, mock_run, mock_pr
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_pr.return_value = "https://github.com/ROCm/ROCgdb/pull/100"

        result = m.run_update(dry_run=False)
        self.assertEqual(result, "created")

        written = json.loads(self._tmp.read_text())
        self.assertEqual(written["therock_commit_ref"], SHA2)
        self.assertEqual(written["therock_commit_created"], "2026-08-10T12:34:56Z")
        self.assertIn(DIGEST2, written["build_image"])
        self.assertEqual(written["build_image_created"], "2026-08-11T00:00:00Z")

    @patch("update_therock_deps.open_update_pr")
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_only_commit_changed_preserves_build_image_created(
        self, mock_find, mock_commit, mock_digest, mock_run, mock_pr
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST, "2026-08-11T00:00:00Z")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_pr.return_value = "https://github.com/ROCm/ROCgdb/pull/100"

        m.run_update(dry_run=False)
        written = json.loads(self._tmp.read_text())
        self.assertEqual(
            written["build_image_created"], BASE_PINS["build_image_created"]
        )
        self.assertEqual(written["therock_commit_created"], "2026-08-10T12:34:56Z")

    @patch("update_therock_deps.open_update_pr")
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_only_digest_changed_preserves_therock_commit_created(
        self, mock_find, mock_commit, mock_digest, mock_run, mock_pr
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_pr.return_value = "https://github.com/ROCm/ROCgdb/pull/100"

        m.run_update(dry_run=False)
        written = json.loads(self._tmp.read_text())
        self.assertEqual(
            written["therock_commit_created"], BASE_PINS["therock_commit_created"]
        )
        self.assertEqual(written["build_image_created"], "2026-08-11T00:00:00Z")

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_invalid_commit_created_format_raises(
        self, mock_find, mock_commit, mock_digest
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "not-a-date")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        with self.assertRaises(ValueError) as ctx:
            m.run_update(dry_run=False)
        self.assertIn("commit_created format", str(ctx.exception))

    @patch("update_therock_deps.open_update_pr")
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_branch_name_contains_date_and_sha(
        self, mock_find, mock_commit, mock_digest, mock_run, mock_pr
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_pr.return_value = "https://github.com/ROCm/ROCgdb/pull/100"

        m.run_update(dry_run=False)
        checkout_cmd = mock_run.call_args_list[0][0][0]
        branch = checkout_cmd[-1]
        self.assertIn("2026-08-10", branch)
        self.assertIn(SHA2[:8], branch)

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_dry_run_does_not_check_for_existing_pr(
        self, mock_find, mock_commit, mock_digest
    ):
        self._write_pins()
        mock_commit.return_value = (SHA2, None)
        mock_digest.return_value = (DIGEST2, None)
        m.run_update(dry_run=True)
        mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    def test_raises_on_nonzero_exit_by_default(self):
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
            with self.assertRaises(subprocess.CalledProcessError):
                m.run(["false"])

    def test_no_raise_when_check_false(self):
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stdout="", stderr="")
            result = m.run(["false"], check=False)
            self.assertEqual(result.returncode, 1)

    def test_returns_completed_process(self):
        with patch("subprocess.run") as mock_sub:
            cp = MagicMock(returncode=0, stdout="hello", stderr="")
            mock_sub.return_value = cp
            result = m.run(["echo", "hello"])
            self.assertEqual(result, cp)


import subprocess  # noqa: E402 — needed for CalledProcessError reference above


# ---------------------------------------------------------------------------
# run_update (additional)
# ---------------------------------------------------------------------------


class TestRunUpdateAdditional(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mktemp(suffix=".json"))
        self._orig_cf = m.CONFIG_FILE
        m.CONFIG_FILE = self._tmp

    def tearDown(self):
        m.CONFIG_FILE = self._orig_cf
        self._tmp.unlink(missing_ok=True)

    def _write_pins(self, pins=None):
        self._tmp.write_text(json.dumps(pins or BASE_PINS, indent=2) + "\n")

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_fetch_timestamp_true_in_live_mode(
        self, mock_find, mock_commit, mock_digest
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        with patch("update_therock_deps.run") as mock_run, patch(
            "update_therock_deps.open_update_pr", return_value="http://pr"
        ):
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            m.run_update(dry_run=False)
        mock_commit.assert_called_once_with(fetch_timestamp=True)
        mock_digest.assert_called_once_with(fetch_timestamp=True)

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_commit_created_none_guard_raises(
        self, mock_find, mock_commit, mock_digest
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, None)  # None despite live mode
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        with self.assertRaises(ValueError) as ctx:
            m.run_update(dry_run=False)
        self.assertIn("commit_created is None", str(ctx.exception))

    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_build_created_none_guard_raises(self, mock_find, mock_commit, mock_digest):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, None)  # None despite live mode
        with self.assertRaises(ValueError) as ctx:
            m.run_update(dry_run=False)
        self.assertIn("build_created is None", str(ctx.exception))

    @patch("update_therock_deps.open_update_pr")
    @patch("update_therock_deps.run")
    @patch("update_therock_deps.get_build_digest")
    @patch("update_therock_deps.get_therock_commit")
    @patch("update_therock_deps.find_open_update_pr")
    def test_build_image_field_has_full_uri(
        self, mock_find, mock_commit, mock_digest, mock_run, mock_pr
    ):
        self._write_pins()
        mock_find.return_value = None
        mock_commit.return_value = (SHA2, "2026-08-10T12:34:56Z")
        mock_digest.return_value = (DIGEST2, "2026-08-11T00:00:00Z")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_pr.return_value = "https://github.com/ROCm/ROCgdb/pull/100"

        m.run_update(dry_run=False)
        written = json.loads(self._tmp.read_text())
        self.assertTrue(
            written["build_image"].startswith(
                "ghcr.io/rocm/therock_build_manylinux_x86_64@"
            )
        )
        self.assertIn(DIGEST2, written["build_image"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):
    @patch("update_therock_deps.run_update")
    def test_returns_0_on_success(self, mock_update):
        mock_update.return_value = "created"
        with patch("sys.argv", ["update_therock_deps.py"]):
            rc = m.main()
        self.assertEqual(rc, 0)

    @patch("update_therock_deps.run_update")
    def test_returns_1_on_exception(self, mock_update):
        mock_update.side_effect = ValueError("boom")
        with patch("sys.argv", ["update_therock_deps.py"]):
            rc = m.main()
        self.assertEqual(rc, 1)

    @patch("update_therock_deps.run_update")
    def test_passes_dry_run_flag(self, mock_update):
        mock_update.return_value = "dry-run"
        with patch("sys.argv", ["update_therock_deps.py", "--dry-run"]):
            m.main()
        mock_update.assert_called_once_with(True)

    @patch("update_therock_deps.run_update")
    def test_no_dry_run_by_default(self, mock_update):
        mock_update.return_value = "not-needed"
        with patch("sys.argv", ["update_therock_deps.py"]):
            m.main()
        mock_update.assert_called_once_with(False)

    @patch("update_therock_deps.run_update")
    def test_returns_1_on_file_not_found(self, mock_update):
        mock_update.side_effect = FileNotFoundError("missing")
        with patch("sys.argv", ["update_therock_deps.py"]):
            rc = m.main()
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
