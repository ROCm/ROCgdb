#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Update TheRock references in ROCgdb's CI dependency pin file.

Fetches the latest TheRock commit and the latest therock_build_manylinux_x86_64
container digest, updates .github/configs.json, then pushes a branch and
opens a ROCgdb PR labelled 'therock-deps'. If an update PR already carries that
label, the run skips without touching git.

For multi-arch CI, also scans recent TheRock CI runs on main to find the newest
successful run whose artifacts are still available for the required GPU families
and stages. Two independent baseline run IDs are maintained:

  therock_baseline_run_id       release build (multi_arch_ci.yml)
  therock_baseline_run_id_asan  ASAN build   (multi_arch_ci_asan.yml)

Each is paired with the TheRock commit at which that run executed. The ASAN
nightly runs infrequently, so its ref/run-id may lag the release ref by days.
When no usable ASAN baseline is found the existing ASAN fields are preserved.

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
from datetime import date, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROCGDB_REPO = "ROCm/ROCgdb"
THEROCK_REPO = "ROCm/TheRock"
BASE_BRANCH = "amd-staging"
CONFIG_FILE = Path(".github/configs.json")

THEROCK_REPO_URL = "https://github.com/ROCm/TheRock.git"
BUILD_IMAGE = "ghcr.io/rocm/therock_build_manylinux_x86_64"

UPDATE_LABEL = "therock-deps"
BRANCH_PREFIX = "users/github/update-therock-refs"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

MAX_RETRIES = 3

# Stages rocgdb does not affect; artifacts for these are copied from a baseline.
PREBUILT_STAGES = (
    "runtime-tests",
    "wsl-rocdxg",
    "math-libs",
    "comm-libs",
    "storage-libs",
    "dctools-core",
    "profiler-apps",
    "cv-libs",
    "media-libs",
)

# GPU families rocgdb multi-arch CI targets.
REQUIRED_FAMILIES = ("gfx94X", "gfx950")

# Maximum age (days) for a baseline artifact bundle to be considered usable.
MAX_ARTIFACT_AGE_DAYS = 80

# Number of recent runs to inspect when searching for a usable baseline.
BASELINE_SCAN_LIMIT = 20

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
# GitHub REST helpers
# ---------------------------------------------------------------------------


def _gh_token() -> str:
    result = run(["gh", "auth", "token"], check=False)
    return result.stdout.strip()


def gh_api(path: str, *, token: str | None = None) -> dict | list:
    """Fetch a GitHub API endpoint and return parsed JSON."""
    url = f"https://api.github.com/{path.lstrip('/')}"
    tok = token or _gh_token()
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen_with_retry(req) as resp:
        return json.loads(resp.read())


def ensure_label(label: str) -> None:
    """Verify that the label exists in the repo; raise if it does not."""
    result = run(
        ["gh", "label", "list", "--repo", ROCGDB_REPO,
         "--search", label, "--json", "name"],
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
        ["gh", "pr", "list", "--repo", ROCGDB_REPO, "--base", BASE_BRANCH,
         "--state", "open", "--label", UPDATE_LABEL,
         "--json", "url", "--jq", ".[0].url // empty"],
    )
    url = (result.stdout or "").strip()
    return url if url else None


# ---------------------------------------------------------------------------
# Fetch latest TheRock commit
# ---------------------------------------------------------------------------


def get_therock_commit() -> tuple[str, str]:
    """Return (commit_sha, today) for TheRock's main branch."""
    result = run(["git", "ls-remote", THEROCK_REPO_URL, "refs/heads/main"])
    parts = result.stdout.split()
    commit = parts[0] if parts else ""
    if not SHA_RE.match(commit):
        raise ValueError(f"Invalid TheRock SHA received: {commit!r}")
    return commit, date.today().isoformat()


# ---------------------------------------------------------------------------
# Fetch build container digest
# ---------------------------------------------------------------------------


def get_build_digest() -> tuple[str, str]:
    """Return (digest, today) for the latest build container image."""
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
# Find TheRock baseline run with usable artifacts
# ---------------------------------------------------------------------------


def _artifact_names_for_run(run_id: int) -> set[str]:
    """Return the set of artifact names available for a workflow run."""
    data = gh_api(f"repos/{THEROCK_REPO}/actions/runs/{run_id}/artifacts?per_page=100")
    return {a["name"] for a in data.get("artifacts", [])}


def _run_age_days(created_at: str) -> float:
    """Return the age in days of a run given its created_at ISO timestamp."""
    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return (datetime.now(tz=timezone.utc) - ts).total_seconds() / 86400


def _stages_available(artifact_names: set[str], families: tuple[str, ...]) -> bool:
    """Return True if artifacts exist for all required prebuilt stages and families."""
    for stage in PREBUILT_STAGES:
        for family in families:
            # Artifact names follow the pattern e.g. "multi-arch-release_linux_gfx94X_compiler-runtime"
            # Stage artifacts are generic (no family suffix) or per-family.
            # We check both generic and per-family patterns.
            generic = f"multi-arch-release_linux_{stage}"
            per_family = f"multi-arch-release_linux_{family}_{stage}"
            if not any(
                n == generic or n == per_family or n.startswith(f"{generic}_") or n.startswith(f"{per_family}_")
                for n in artifact_names
            ):
                # Not all stages are per-family; absence of per-family is fine if
                # the generic artifact is present. If neither is present, skip.
                if not any(n.startswith(generic) for n in artifact_names):
                    return False
    return True


def find_baseline_run(
    workflow_filename: str,
    variant_label: str,
) -> tuple[str, str, str] | None:
    """Find the newest TheRock main CI run with usable artifacts.

    Returns (run_id, head_sha, created_date) or None if no suitable run found.
    variant_label is used in artifact name matching (e.g. "release", "host-asan").
    """
    data = gh_api(
        f"repos/{THEROCK_REPO}/actions/workflows/{workflow_filename}/runs"
        f"?branch=main&status=success&per_page={BASELINE_SCAN_LIMIT}"
    )
    runs = data.get("workflow_runs", [])

    for run_info in runs:
        run_id = run_info["id"]
        head_sha = run_info["head_sha"]
        created_at = run_info["created_at"]

        age = _run_age_days(created_at)
        if age > MAX_ARTIFACT_AGE_DAYS:
            print(f"  run {run_id} ({head_sha[:12]}) too old ({age:.0f}d), stopping scan")
            break

        print(f"  checking run {run_id} ({head_sha[:12]}, {age:.1f}d old)...", end=" ")
        artifacts = _artifact_names_for_run(run_id)
        if not artifacts:
            print("no artifacts")
            continue

        if _stages_available(artifacts, REQUIRED_FAMILIES):
            created_date = created_at[:10]
            print(f"OK (found {len(artifacts)} artifacts)")
            return str(run_id), head_sha, created_date

        print(f"missing required artifacts ({len(artifacts)} found)")

    return None


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
    baseline: tuple[str, str, str] | None,
    baseline_asan: tuple[str, str, str] | None,
) -> None:
    """Write updated dependency pins to CONFIG_FILE."""
    updated = dict(pins)
    updated["therock_commit_ref"] = commit
    updated["therock_commit_date"] = commit_date
    updated["build_image"] = f"{BUILD_IMAGE}@{build_digest}"
    updated["build_image_date"] = build_date
    if baseline is not None:
        run_id, sha, run_date = baseline
        updated["therock_baseline_run_id"] = run_id
        updated["therock_baseline_run_ref"] = sha
        updated["therock_baseline_run_date"] = run_date
    if baseline_asan is not None:
        run_id, sha, run_date = baseline_asan
        updated["therock_baseline_run_id_asan"] = run_id
        updated["therock_baseline_run_ref_asan"] = sha
        updated["therock_baseline_run_date_asan"] = run_date
    CONFIG_FILE.write_text(json.dumps(updated, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Branch / commit / PR
# ---------------------------------------------------------------------------


def build_commit_message(
    pins: dict[str, str],
    commit: str,
    digest: str,
    baseline: tuple[str, str, str] | None,
    baseline_asan: tuple[str, str, str] | None,
) -> str:
    lines = ["Update TheRock dependencies and container images", ""]
    old_commit = pins.get("therock_commit_ref")
    if old_commit and old_commit != commit:
        lines.append(f"commit: {old_commit[:12]} -> {commit[:12]}")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    if old_digest and old_digest != digest:
        lines.append(f"build image: {old_digest[:19]}... -> {digest[:19]}...")
    if baseline is not None:
        run_id, sha, run_date = baseline
        old_run = pins.get("therock_baseline_run_id", "")
        if old_run != run_id:
            lines.append(f"baseline run (release): {old_run or 'none'} -> {run_id} ({sha[:12]}, {run_date})")
    if baseline_asan is not None:
        run_id, sha, run_date = baseline_asan
        old_run = pins.get("therock_baseline_run_id_asan", "")
        if old_run != run_id:
            lines.append(f"baseline run (asan): {old_run or 'none'} -> {run_id} ({sha[:12]}, {run_date})")
    return "\n".join(lines)


def open_update_pr(branch: str, commit: str) -> str:
    title = f"Update TheRock dependencies ({date.today().isoformat()})"
    body = (
        "## Automated TheRock dependency update\n\n"
        f"Bumps `therock_commit_ref` to `{commit[:12]}` and refreshes the "
        "`therock_build_manylinux_x86_64` build container digest and "
        "multi-arch CI baseline run IDs in `.github/configs.json`.\n\n"
        "Opened automatically by the TheRock dependency update workflow. "
        "Please kick CI manually (close and reopen, or push an empty commit), "
        "then merge."
    )
    ensure_label(UPDATE_LABEL)
    result = run(
        ["gh", "pr", "create", "--repo", ROCGDB_REPO, "--base", BASE_BRANCH,
         "--head", branch, "--title", title, "--body", body, "--label", UPDATE_LABEL],
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

    print("\n=== Scanning for usable TheRock baseline runs ===\n")
    print("Release (multi_arch_ci.yml):")
    baseline = find_baseline_run("multi_arch_ci.yml", "release")
    if baseline:
        print(f"  -> selected run {baseline[0]} at {baseline[1][:12]} ({baseline[2]})")
    else:
        print("  -> no usable baseline found; release prebuilt stages will be disabled")

    print("\nASAN (multi_arch_ci_asan.yml):")
    baseline_asan = find_baseline_run("multi_arch_ci_asan.yml", "host-asan")
    if baseline_asan:
        print(f"  -> selected run {baseline_asan[0]} at {baseline_asan[1][:12]} ({baseline_asan[2]})")
    else:
        print("  -> no usable ASAN baseline found; existing ASAN pins preserved")
        # Preserve existing ASAN fields if no new baseline is found.
        old_id = pins.get("therock_baseline_run_id_asan")
        old_ref = pins.get("therock_baseline_run_ref_asan")
        old_date = pins.get("therock_baseline_run_date_asan")
        if old_id and old_ref and old_date:
            baseline_asan = (old_id, old_ref, old_date)

    old_commit = pins.get("therock_commit_ref", "")
    old_image = pins.get("build_image", "")
    old_digest = old_image.split("@", 1)[1] if "@" in old_image else ""
    old_baseline = pins.get("therock_baseline_run_id", "")
    old_baseline_asan = pins.get("therock_baseline_run_id_asan", "")

    commit_changed = old_commit != commit
    digest_changed = old_digest != build_digest
    baseline_changed = baseline is not None and old_baseline != baseline[0]
    baseline_asan_changed = baseline_asan is not None and old_baseline_asan != baseline_asan[0]

    print("\n=== TheRock dependency update ===\n")
    if commit_changed:
        print(f"TheRock ref update:          {old_commit[:12]} -> {commit[:12]}")
    else:
        print(f"TheRock ref:                 unchanged ({commit[:12]})")
    if digest_changed:
        print(f"Build container digest:      {old_digest[:19]}... -> {build_digest[:19]}...")
    else:
        print(f"Build container digest:      unchanged ({build_digest[:19]}...)")
    if baseline_changed:
        print(f"Release baseline run:        {old_baseline or 'none'} -> {baseline[0]}")
    elif baseline:
        print(f"Release baseline run:        unchanged ({baseline[0]})")
    else:
        print(f"Release baseline run:        no usable run found")
    if baseline_asan_changed:
        print(f"ASAN baseline run:           {old_baseline_asan or 'none'} -> {baseline_asan[0]}")
    elif baseline_asan:
        print(f"ASAN baseline run:           unchanged ({baseline_asan[0]})")
    else:
        print(f"ASAN baseline run:           no usable run found")

    any_changed = commit_changed or digest_changed or baseline_changed or baseline_asan_changed

    if not any_changed:
        print("\nNo updates needed.")
        summary("not-needed")
        return "not-needed"

    if dry_run:
        changes = []
        if commit_changed:
            changes.append("TheRock ref")
        if digest_changed:
            changes.append("build container digest")
        if baseline_changed:
            changes.append("release baseline run")
        if baseline_asan_changed:
            changes.append("ASAN baseline run")
        print(f"\nDry run: would update {', '.join(changes)}.")
        summary("dry-run")
        return "dry-run"

    branch = f"{BRANCH_PREFIX}-{commit_date}-{commit[:8]}"
    message = build_commit_message(pins, commit, build_digest, baseline, baseline_asan)

    run(["git", "checkout", "-B", branch])
    write_deps(pins, commit, commit_date, build_digest, build_date, baseline, baseline_asan)
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
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        summary("error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
