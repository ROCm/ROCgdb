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
import os
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


def urlopen_with_retry(req: str | urllib.request.Request, timeout: int = 30):
    """Open a URL with retries on transient errors (5xx, timeout); raises immediately on 4xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            if attempt == MAX_RETRIES:
                raise
            print(f"[RETRY] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"[RETRY] attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(2**attempt)
    raise RuntimeError("urlopen_with_retry: unreachable")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, printing the command that is executed."""
    print(f"[RUN] {' '.join(shlex.quote(str(a)) for a in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
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
        raise ValueError(
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


def get_therock_commit(fetch_timestamp: bool = True) -> tuple[str, str | None]:
    """Return (commit_sha, commit_created) for TheRock's main branch.

    Uses the GitHub branches API so the call goes through urlopen_with_retry.
    If fetch_timestamp is False, only the SHA is fetched and None is returned
    in place of the timestamp (useful for dry-run mode).
    """
    headers = {"Accept": "application/vnd.github+json"}
    if gh_token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {gh_token}"

    # Use the REST API for the branch HEAD so the call goes through urlopen_with_retry
    # and benefits from the same retry logic as the other network operations.
    branch_url = "https://api.github.com/repos/ROCm/TheRock/branches/main"
    req = urllib.request.Request(branch_url, headers=headers)
    with urlopen_with_retry(req) as resp:
        branch_data = json.loads(resp.read())
    commit = (branch_data.get("commit") or {}).get("sha", "")
    if not SHA_RE.match(commit):
        raise ValueError(f"Invalid TheRock SHA received: {commit!r}")

    if not fetch_timestamp:
        return commit, None

    api_url = f"https://api.github.com/repos/ROCm/TheRock/commits/{commit}"
    req = urllib.request.Request(api_url, headers=headers)
    with urlopen_with_retry(req) as resp:
        data = json.loads(resp.read())
    commit_obj = data.get("commit")
    if not commit_obj:
        raise ValueError(
            f"GitHub API returned no commit object for {commit!r}: {data.get('message', '')!r}"
        )
    committer = commit_obj.get("committer")
    if not committer:
        raise ValueError(f"GitHub API returned no committer for {commit!r}")
    commit_created = committer.get("date")
    if not commit_created:
        raise ValueError(f"GitHub API returned no committer date for {commit!r}")
    # e.g. "2026-08-10T12:34:56Z"
    return commit, commit_created


def get_build_digest(fetch_timestamp: bool = True) -> tuple[str, str | None]:
    """Return (digest, build_created) for the latest build container image.

    Uses the OCI Distribution API directly — no external tools required.
    The digest comes from the Docker-Content-Digest response header.
    The image creation timestamp comes from the OCI config blob's 'created' field.

    If fetch_timestamp is False, the config blob is not fetched and None is returned
    in place of the timestamp (useful for dry-run mode where only the digest is needed).
    """
    # GHCR requires an anonymous bearer token even for public images.
    image_path = BUILD_IMAGE.removeprefix("ghcr.io/")
    token_url = (
        f"https://ghcr.io/token?scope=repository:{image_path}:pull&service=ghcr.io"
    )
    with urlopen_with_retry(token_url) as resp:
        token_data = json.loads(resp.read())
    token = token_data.get("token")
    if not token:
        raise ValueError(
            f"GHCR token endpoint returned no token: {token_data.get('error', '')!r}"
        )

    auth_header = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{image_path}/manifests/latest",
        headers={
            **auth_header,
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.oci.image.manifest.v1+json"
            ),
        },
    )
    with urlopen_with_retry(req) as resp:
        digest = resp.headers.get("Docker-Content-Digest", "")
        manifest = json.loads(resp.read())

    if not DIGEST_RE.match(digest):
        raise ValueError(f"Invalid build image digest received: {digest!r}")

    if not fetch_timestamp:
        return digest, None

    config_digest = (manifest.get("config") or {}).get("digest")
    if not config_digest:
        raise ValueError("Manifest has no config blob digest")
    config_req = urllib.request.Request(
        f"https://ghcr.io/v2/{image_path}/blobs/{config_digest}",
        headers=auth_header,
    )
    with urlopen_with_retry(config_req) as config_resp:
        config = json.loads(config_resp.read())

    created = config.get("created")
    if not created:
        raise ValueError("Config blob has no 'created' field")

    return digest, created


# ---------------------------------------------------------------------------
# Dependency file read / write
# ---------------------------------------------------------------------------


def read_deps() -> dict[str, str]:
    """Read and return the current dependency pins from CONFIG_FILE."""
    return json.loads(CONFIG_FILE.read_text())


def write_deps(pins: dict[str, str]) -> None:
    """Write updated dependency pins to CONFIG_FILE."""
    CONFIG_FILE.write_text(json.dumps(pins, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Branch / commit / PR
# ---------------------------------------------------------------------------


def build_commit_message(pins: dict[str, str], commit: str, digest: str) -> str:
    lines = ["Update TheRock dependencies and container images", ""]
    old_commit = pins.get("therock_commit_ref", "")
    if old_commit != commit:
        old_commit_display = old_commit[:12] if old_commit else "(none)"
        lines.append(f"commit: {old_commit_display} -> {commit[:12]}")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    if old_digest and old_digest != digest and DIGEST_RE.match(old_digest):
        lines.append(f"build image: {old_digest[:19]}... -> {digest[:19]}...")
    return "\n".join(lines)


def open_update_pr(
    branch: str,
    commit: str,
    commit_created: str,
    digest_changed: bool,
    build_digest: str,
    build_created: str,
) -> str:
    title = f"Automated TheRock dependency update ({date.today().isoformat()})"
    commit_date = commit_created[:10]
    changes = [f"* Bump `therock_commit_ref` to `{commit[:12]}` ({commit_date})"]
    if digest_changed:
        build_date = build_created[:10]
        changes.append(
            f"* Bump `therock_build_manylinux_x86_64` digest to"
            f" `{build_digest[:19]}...` ({build_date})"
        )
    body = (
        "\n".join(changes) + "\n\n"
        "Opened automatically by the TheRock dependency update workflow. "
        "Please validate CI and merge."
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
    pr = result.stdout.strip()
    if not pr:
        raise ValueError("gh pr create succeeded but returned no PR URL")
    return pr


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

    # Validate required keys before any git side-effects.
    if "therock_commit_created" not in pins:
        raise ValueError("therock_commit_created missing from configs.json")
    if "build_image_created" not in pins:
        raise ValueError("build_image_created missing from configs.json")

    commit, commit_created = get_therock_commit(fetch_timestamp=not dry_run)
    build_digest, build_created = get_build_digest(fetch_timestamp=not dry_run)

    old_commit = pins.get("therock_commit_ref", "")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    old_digest_valid = bool(DIGEST_RE.match(old_digest))
    commit_changed = old_commit != commit
    digest_changed = old_digest != build_digest

    print("\n=== TheRock dependency update ===\n")
    if commit_changed:
        old_commit_display = old_commit[:12] if old_commit else "(none)"
        print(f"TheRock ref update:      {old_commit_display} -> {commit[:12]}")
    else:
        print(f"TheRock ref:             unchanged ({commit[:12]})")
    if digest_changed:
        old_digest_display = (
            f"{old_digest[:19]}..." if old_digest_valid else "(unknown)"
        )
        print(
            f"Build container digest:  {old_digest_display} -> {build_digest[:19]}..."
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

    if commit_created is None:
        raise ValueError("commit_created is None despite fetch_timestamp=True")
    if build_created is None:
        raise ValueError("build_created is None despite fetch_timestamp=True")
    date_prefix = commit_created[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_prefix):
        raise ValueError(f"Unexpected commit_created format: {commit_created!r}")
    branch = f"{BRANCH_PREFIX}-{date_prefix}-{commit[:8]}"
    message = build_commit_message(pins, commit, build_digest)

    run(["git", "checkout", "-B", branch])
    updated = dict(pins)
    updated["therock_commit_ref"] = commit
    updated["therock_commit_created"] = (
        commit_created if commit_changed else pins["therock_commit_created"]
    )
    updated["build_image"] = f"{BUILD_IMAGE}@{build_digest}"
    updated["build_image_created"] = (
        build_created if digest_changed else pins["build_image_created"]
    )
    write_deps(updated)
    run(["git", "add", str(CONFIG_FILE)])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", branch])

    pr = open_update_pr(
        branch,
        commit,
        commit_created,
        digest_changed,
        build_digest,
        build_created,
    )
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
