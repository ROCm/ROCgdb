#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Update TheRock references in ROCgdb's CI dependency pin file.

Fetches the latest TheRock commit and the latest therock_build_manylinux_x86_64
container digest, updates .github/configs.json, then pushes a branch and
opens a ROCgdb PR labelled 'therock-deps'. If an update PR already carries that
label, the run skips without touching git.

The test container is not updated here: it is pinned transitively through
therock_commit_ref (TheRock's fetch_test_configurations.py supplies the test
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
CONFIG_FILE = Path(".github/configs.json")

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
# Dependency file read / write
# ---------------------------------------------------------------------------


def read_deps() -> dict[str, str]:
    """Read and return the current dependency pins from CONFIG_FILE."""
    return json.loads(CONFIG_FILE.read_text())


def write_deps(
    pins: dict[str, str],
    commit: str,
    commit_date: str,
    build_digest: str,
    build_date: str,
) -> None:
    """Write updated dependency pins to CONFIG_FILE."""
    updated = dict(pins)
    updated["therock_commit_ref"] = commit
    updated["therock_commit_date"] = commit_date
    updated["build_image"] = f"{BUILD_IMAGE}@{build_digest}"
    updated["build_image_date"] = build_date
    CONFIG_FILE.write_text(json.dumps(updated, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Branch / commit / PR
# ---------------------------------------------------------------------------


def build_commit_message(pins: dict[str, str], commit: str, digest: str) -> str:
    lines = ["Update TheRock dependencies and container images", ""]
    old_commit = pins.get("therock_commit_ref")
    if old_commit and old_commit != commit:
        lines.append(f"commit: {old_commit[:12]} -> {commit[:12]}")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    if old_digest and old_digest != digest:
        lines.append(f"build image: {old_digest[:19]}... -> {digest[:19]}...")
    return "\n".join(lines)


def open_update_pr(branch: str, commit: str) -> str:
    title = f"Update TheRock dependencies ({date.today().isoformat()})"
    body = (
        "## Automated TheRock dependency update\n\n"
        f"Bumps `therock_commit_ref` to `{commit[:12]}` and refreshes the "
        "`therock_build_manylinux_x86_64` build container digest in "
        "`.github/configs.json`.\n\n"
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
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"{CONFIG_FILE} not found; run from the repo root.")

    if not dry_run:
        existing = find_open_update_pr()
        if existing:
            print(f"Open update PR already exists: {existing}")
            summary("skipped-existing-pr", existing)
            return "skipped-existing-pr"

    pins = read_deps()
    commit, commit_date = get_therock_commit()
    build_digest, build_date = get_build_digest()

    old_commit = pins.get("therock_commit_ref", "")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    commit_changed = old_commit != commit
    digest_changed = old_digest != build_digest

    print("\n=== TheRock dependency update ===\n")
    if commit_changed:
        print(f"TheRock ref update:      {old_commit[:12]} -> {commit[:12]}")
    else:
        print(f"TheRock ref:             unchanged ({commit[:12]})")
    if digest_changed:
        print(
            f"Build container digest:  {old_digest[:19]}... "
            f"-> {build_digest[:19]}..."
        )
    else:
        print(f"Build container digest:  unchanged ({build_digest[:19]}...)")

    if not commit_changed and not digest_changed:
        print("\nNo updates: TheRock ref and build container digest are both current.")
        summary("not-needed")
        return "not-needed"

    if dry_run:
        changes = []
        if commit_changed:
            changes.append("TheRock ref")
        if digest_changed:
            changes.append("build container digest")
        print(f"\nDry run: would update {' and '.join(changes)}.")
        summary("dry-run")
        return "dry-run"

    branch = f"{BRANCH_PREFIX}-{commit_date}-{commit[:8]}"
    message = build_commit_message(pins, commit, build_digest)

    run(["git", "checkout", "-B", branch])
    write_deps(pins, commit, commit_date, build_digest, build_date)
    run(["git", "add", str(CONFIG_FILE)])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", branch])

    pr = open_update_pr(branch, commit)
    summary("created", pr)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update TheRock references in ROCgdb's CI dependency pin file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and compute changes, print what would be updated, but skip "
        "branch, commit, push and PR creation.",
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
