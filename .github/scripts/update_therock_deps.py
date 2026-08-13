#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Update TheRock references in ROCgdb's CI workflow.

Fetches the latest TheRock commit and the latest therock_build_manylinux_x86_64
container digest, updates .github/workflows/therock-ci-linux.yml, then pushes a
branch and opens a ROCgdb PR labelled 'therock-deps'. If an update PR already
carries that label, the run skips without touching git.

The test container is not updated here: it is pinned transitively through
THEROCK_COMMIT_REF (TheRock's fetch_test_configurations.py supplies the test
image), so bumping the commit ref updates it automatically.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROCGDB_REPO = "ROCm/ROCgdb"
BASE_BRANCH = "amd-staging"
CI_LINUX = Path(".github/workflows/therock-ci-linux.yml")

THEROCK_REPO_URL = "https://github.com/ROCm/TheRock.git"
BUILD_IMAGE = "ghcr.io/rocm/therock_build_manylinux_x86_64"

UPDATE_LABEL = "therock-deps"
BRANCH_PREFIX = "users/github/update-therock-refs"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

MAX_RETRIES = 3  # number of retries for network requests

# Never pass secrets as CLI args — use env vars (e.g. GH_TOKEN) instead.


# ---------------------------------------------------------------------------
# Network helper
# ---------------------------------------------------------------------------


def urlopen_with_retry(req, timeout: int = 30):
    """Open a URL with retries on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"[RETRY] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, printing the command that is executed."""
    print(f"[RUN] {' '.join(shlex.quote(str(a)) for a in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0 and result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


# ---------------------------------------------------------------------------
# GitHub helpers (gh CLI)
# ---------------------------------------------------------------------------


def ensure_label(label: str) -> None:
    """Verify that the label exists in the repo; raise if it does not."""
    result = run(
        [
            "gh",
            "label",
            "list",
            "--repo",
            ROCGDB_REPO,
            "--search",
            label,
            "--json",
            "name",
        ],
    )
    existing = [entry["name"] for entry in json.loads(result.stdout or "[]")]
    if label not in existing:
        raise RuntimeError(
            f"Label '{label}' not found in {ROCGDB_REPO}. "
            "Please create it in the repo settings before running."
        )


def find_open_update_pr() -> str | None:
    """Return the URL of an open, labelled update PR if one exists, else None."""
    result = run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            ROCGDB_REPO,
            "--base",
            BASE_BRANCH,
            "--state",
            "open",
            "--label",
            UPDATE_LABEL,
            "--json",
            "url",
            "--jq",
            ".[0].url // empty",
        ],
    )
    url = (result.stdout or "").strip()
    return url if url else None


# ---------------------------------------------------------------------------
# Fetch latest references
# ---------------------------------------------------------------------------


def get_therock_commit() -> tuple[str, str]:
    """Return (commit_sha, today) for TheRock's main branch."""
    result = run(["git", "ls-remote", THEROCK_REPO_URL, "refs/heads/main"])
    parts = result.stdout.split()
    commit = parts[0] if parts else ""
    if not SHA_RE.match(commit):
        raise ValueError(f"Invalid TheRock SHA received: {commit!r}")
    return commit, date.today().isoformat()


def get_build_digest() -> tuple[str, str]:
    """Return (digest, today) for the latest build container image.

    Uses the OCI Distribution API directly — no external tools required.
    The digest comes from the Docker-Content-Digest response header.
    """
    # GHCR requires an anonymous bearer token even for public images.
    image_path = BUILD_IMAGE.removeprefix("ghcr.io/")
    token_url = (
        f"https://ghcr.io/token?scope=repository:{image_path}:pull&service=ghcr.io"
    )
    with urlopen_with_retry(token_url) as resp:
        token = json.loads(resp.read())["token"]

    req = urllib.request.Request(
        f"https://ghcr.io/v2/{image_path}/manifests/latest",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.oci.image.index.v1+json"
            ),
        },
    )
    with urlopen_with_retry(req) as resp:
        digest = resp.headers.get("Docker-Content-Digest", "")

    if not DIGEST_RE.match(digest):
        raise ValueError(f"Invalid build image digest received: {digest!r}")
    return digest, date.today().isoformat()


# ---------------------------------------------------------------------------
# File update
# ---------------------------------------------------------------------------


def read_current(content: str) -> dict[str, str]:
    """Extract the current commit ref and build digest from the CI workflow content."""
    values: dict[str, str] = {}
    commit = re.search(r"THEROCK_COMMIT_REF:\s+([0-9a-f]{40})", content)
    if commit:
        values["commit"] = commit.group(1)
    digest = re.search(re.escape(BUILD_IMAGE) + r"@(sha256:[0-9a-f]{64})", content)
    if digest:
        values["build_digest"] = digest.group(1)
    return values


def render_ci_linux(
    content: str, commit: str, commit_date: str, build_digest: str, build_date: str
) -> str:
    """Return updated content for the CI workflow.

    Pure: does not write to disk. Raises if either expected pattern is not found.
    """
    updated, n = re.subn(
        r"THEROCK_COMMIT_REF: .* # .*",
        f"THEROCK_COMMIT_REF: {commit} # {commit_date}",
        content,
    )
    if n != 1:
        raise ValueError(
            f"Expected exactly 1 match for THEROCK_COMMIT_REF pattern, got {n}. "
            "Check that the line has a trailing '# date' comment."
        )
    updated, n = re.subn(
        re.escape(BUILD_IMAGE) + r"@sha256:[0-9a-f]+ # .*",
        f"{BUILD_IMAGE}@{build_digest} # {build_date}",
        updated,
    )
    if n != 1:
        raise ValueError(
            f"Expected exactly 1 match for build image pattern, got {n}. "
            "Check that the line has a trailing '# date' comment."
        )
    return updated


# ---------------------------------------------------------------------------
# Branch / commit / PR
# ---------------------------------------------------------------------------


def build_commit_message(old: dict[str, str], commit: str, digest: str) -> str:
    lines = ["Update TheRock dependencies and container images", ""]
    old_commit = old.get("commit")
    if old_commit and old_commit != commit:
        lines.append(f"commit: {old_commit[:12]} -> {commit[:12]}")
    old_digest = old.get("build_digest")
    if old_digest and old_digest != digest:
        lines.append(f"build image: {old_digest[:19]}... -> {digest[:19]}...")
    return "\n".join(lines)


def open_update_pr(branch: str, commit: str) -> str:
    title = f"Update TheRock dependencies ({date.today().isoformat()})"
    body = (
        "## Automated TheRock dependency update\n\n"
        f"Bumps `THEROCK_COMMIT_REF` to `{commit[:12]}` and refreshes the "
        "`therock_build_manylinux_x86_64` build container digest in "
        "`.github/workflows/therock-ci-linux.yml`.\n\n"
        "Opened automatically by the TheRock dependency update workflow. "
        "Please kick CI manually (close and reopen, or push an empty commit), "
        "then merge."
    )
    ensure_label(UPDATE_LABEL)
    result = run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            ROCGDB_REPO,
            "--base",
            BASE_BRANCH,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
            "--label",
            UPDATE_LABEL,
        ],
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summary(result: str, pr: str | None = None) -> None:
    print("\n=== Summary ===")
    print(f"Result: {result}")
    if pr:
        print(f"PR: {pr}")


def run_update(dry_run: bool) -> str:
    if not CI_LINUX.exists():
        raise FileNotFoundError(f"{CI_LINUX} not found; run from the repo root.")

    if not dry_run:
        existing = find_open_update_pr()
        if existing:
            print(f"Open update PR already exists: {existing}")
            summary("skipped-existing-pr", existing)
            return "skipped-existing-pr"

    content = CI_LINUX.read_text()
    old = read_current(content)
    commit, commit_date = get_therock_commit()
    build_digest, build_date = get_build_digest()

    updated = render_ci_linux(content, commit, commit_date, build_digest, build_date)
    if updated == content:
        print("\nReferences already current; nothing to update.")
        summary("not-needed")
        return "not-needed"

    print("\nUpdating references:")
    if old.get("commit") != commit:
        print(f"  commit:      {old.get('commit', '?')[:12]} -> {commit[:12]}")
    if old.get("build_digest") != build_digest:
        print(
            f"  build image: {old.get('build_digest', '?')[:19]}... "
            f"-> {build_digest[:19]}..."
        )

    if dry_run:
        summary("dry-run")
        return "dry-run"

    branch = f"{BRANCH_PREFIX}-{commit_date}-{commit[:8]}"
    message = build_commit_message(old, commit, build_digest)

    run(["git", "checkout", "-B", branch])
    CI_LINUX.write_text(updated)
    run(["git", "add", str(CI_LINUX)])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", branch])

    pr = open_update_pr(branch, commit)
    summary("created", pr)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update TheRock references in ROCgdb's CI workflow."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and update the file, print the diff, but skip branch, "
        "commit, push and PR creation.",
    )
    args = parser.parse_args()

    try:
        run_update(args.dry_run)
        return 0
    except Exception as exc:  # noqa: BLE001 - surface any failure as a red run
        print(f"Error: {exc}", file=sys.stderr)
        summary("error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
