#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sync ROCgdb upstream branch into the target branch.

Fetches the upstream branch, fast-forwards the origin mirror to match,
then merges cleanly into the target branch. On conflict, pushes a conflict
branch and opens a PR for manual resolution.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCEWARE_URL = "https://sourceware.org/git/binutils-gdb.git"
SOURCEWARE_REMOTE = "sourceware"

_rocgdb_repo = os.environ.get("GITHUB_REPOSITORY", "")
if not _rocgdb_repo:
    print(
        "Error: GITHUB_REPOSITORY environment variable is required.",
        file=sys.stderr,
    )
    sys.exit(1)
ROCGDB_REPO = _rocgdb_repo
ROCGDB_ORIGIN_URL = f"https://github.com/{ROCGDB_REPO}.git"

_target_branch = os.environ.get("AUTOMERGE_TARGET_BRANCH", "")
if not _target_branch:
    print(
        "Error: AUTOMERGE_TARGET_BRANCH environment variable is required.",
        file=sys.stderr,
    )
    sys.exit(1)
TARGET_BRANCH = _target_branch

_upstream_branch = os.environ.get("AUTOMERGE_UPSTREAM_BRANCH", "")
if not _upstream_branch:
    print(
        "Error: AUTOMERGE_UPSTREAM_BRANCH environment variable is required.",
        file=sys.stderr,
    )
    sys.exit(1)
UPSTREAM_BRANCH = _upstream_branch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Sync {SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH} into origin/{TARGET_BRANCH}."
    )
    p.add_argument(
        "--workspace",
        metavar="DIR",
        help="Root directory for the working clone (overrides $WORKSPACE). "
        "The clone is created at DIR/rocgdb_sync/binutils-gdb.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and probe as normal, but skip all pushes and PR creation.",
    )
    return p.parse_args()


_args = _parse_args()
DRY_RUN: bool = _args.dry_run
_workspace = _args.workspace or os.environ.get("WORKSPACE", "")
if _workspace:
    _workspace_path = Path(_workspace).resolve()
    if not _workspace_path.exists():
        print(
            f"Error: workspace '{_workspace}' is not a valid path.",
            file=sys.stderr,
        )
        sys.exit(1)
    WORK_DIR = _workspace_path / "rocgdb_sync"
else:
    WORK_DIR = Path.cwd() / "rocgdb_sync"

# Never pass secrets as CLI args — use env vars (e.g. GH_TOKEN) instead.

_branch_infix = f"{UPSTREAM_BRANCH}-to-{TARGET_BRANCH}".replace("/", "-")
CONFLICT_BRANCH_PREFIX = f"users/github/{_branch_infix}-conflict"
CONFLICT_FREE_BRANCH_PREFIX = f"users/github/{_branch_infix}-conflict-free"
CONFLICT_LABEL = "merge-conflict"
CONFLICT_FREE_LABEL = "merge-testing"
CI_SKIP_LABEL = "ci:skip"

# Sleep durations (seconds) between successive network attempts.
# Total attempts = len(RETRY_DELAYS) + 1.
RETRY_DELAYS = [60, 120]


# ---------------------------------------------------------------------------
# Subprocess / git helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    cwd=None,
    check: bool = True,
    env: dict[str, str] | None = None,
    redact: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, printing the command (with any redacted values masked)."""
    display_cmd = cmd
    if redact:
        display_cmd = [
            "<redacted>" if any(r in arg for r in redact) else arg for arg in cmd
        ]
    print(f"[RUN] {' '.join(shlex.quote(str(a)) for a in display_cmd)}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, env=merged_env
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0 and result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def _is_network_error(exc: subprocess.CalledProcessError) -> bool:
    stderr = (exc.stderr or "").lower()
    keywords = (
        "could not resolve",
        "unable to connect",
        "connection refused",
        "connection timed out",
        "network is unreachable",
        "no route to host",
        "fatal: unable to access",
        "ssh: connect",
    )
    return any(k in stderr for k in keywords)


def run_net(cmd: list[str], cwd=None) -> subprocess.CompletedProcess:
    """Run a network git command, retrying on transient connectivity errors."""
    max_attempts = len(RETRY_DELAYS) + 1
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return run(cmd, cwd=cwd)
        except subprocess.CalledProcessError as exc:
            if not _is_network_error(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                delay = RETRY_DELAYS[attempt - 1]
                print(
                    f"Network error (attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay}s…"
                )
                time.sleep(delay)
            else:
                print(f"Network error (attempt {attempt}/{max_attempts}), giving up.")
    raise ConnectivityError() from last_exc


class ConnectivityError(Exception):
    pass


# ---------------------------------------------------------------------------
# GitHub helpers (gh CLI)
# ---------------------------------------------------------------------------


def _ensure_label(label: str, color: str) -> None:
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
        run(["gh", "label", "create", label, "--repo", ROCGDB_REPO, "--color", color])


def open_conflict_pr(
    branch: str,
    merge_base_sha: str,
    conflict_commit: str,
    conflicted_files: list[str],
    title: str,
) -> str:
    compare_url = f"https://github.com/{ROCGDB_REPO}/compare/{merge_base_sha[:12]}...{conflict_commit[:12]}"
    file_list = "\n".join(f"- `{f}`" for f in conflicted_files)
    body = (
        f"## ROCgdb {UPSTREAM_BRANCH} → {TARGET_BRANCH} merge conflict\n\n"
        f"**Commits being merged:** [{merge_base_sha[:12]}...{conflict_commit[:12]}]({compare_url})\n\n"
        f"**Conflicted files:**\n{file_list}\n\n"
        f"Please resolve the conflicts, remove the `{CI_SKIP_LABEL}` label, "
        f"validate in CI, then merge."
    )

    _ensure_label(CONFLICT_LABEL, "e11d48")
    _ensure_label(CI_SKIP_LABEL, "f59e0b")

    result = run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            ROCGDB_REPO,
            "--base",
            TARGET_BRANCH,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
            "--label",
            CONFLICT_LABEL,
            "--label",
            CI_SKIP_LABEL,
        ]
    )
    pr_url = result.stdout.strip()
    print(f"Conflict PR opened: {pr_url}")
    return pr_url


def find_open_conflict_pr() -> str | None:
    """Return the URL of an open conflict PR if one exists, else None."""
    result = run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            ROCGDB_REPO,
            "--base",
            TARGET_BRANCH,
            "--state",
            "open",
            "--label",
            CONFLICT_LABEL,
            "--json",
            "url",
            "--jq",
            ".[0].url // empty",
        ],
    )
    url = (result.stdout or "").strip()
    return url if url else None


def open_conflict_free_pr(
    branch: str,
    first_commit: str,
    last_commit: str,
    title: str,
) -> str:
    compare_url = f"https://github.com/{ROCGDB_REPO}/compare/{first_commit[:12]}...{last_commit[:12]}"
    body = (
        f"## ROCgdb {UPSTREAM_BRANCH} → {TARGET_BRANCH} conflict-free merge\n\n"
        f"**Commits being merged:** [{first_commit[:12]}...{last_commit[:12]}]({compare_url})\n\n"
        f"This PR was opened automatically by the automerge workflow. "
        f"Please validate CI, then merge."
    )

    _ensure_label(CONFLICT_FREE_LABEL, "0075ca")

    result = run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            ROCGDB_REPO,
            "--base",
            TARGET_BRANCH,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
            "--label",
            CONFLICT_FREE_LABEL,
        ]
    )
    pr_url = result.stdout.strip()
    print(f"Conflict-free PR opened: {pr_url}")
    return pr_url


def find_open_conflict_free_pr() -> str | None:
    """Return the URL of an open conflict-free PR if one exists, else None."""
    result = run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            ROCGDB_REPO,
            "--base",
            TARGET_BRANCH,
            "--state",
            "open",
            "--label",
            CONFLICT_FREE_LABEL,
            "--json",
            "url",
            "--jq",
            ".[0].url // empty",
        ],
    )
    url = (result.stdout or "").strip()
    return url if url else None


# ---------------------------------------------------------------------------
# Core sync logic
# ---------------------------------------------------------------------------


def push_upstream_mirror(repo: Path) -> None:
    """Fast-forward origin/UPSTREAM_BRANCH to match SOURCEWARE_REMOTE/UPSTREAM_BRANCH."""
    print(
        f"Fast-forwarding origin/{UPSTREAM_BRANCH} to {SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}…"
    )
    run_net(
        [
            "git",
            "push",
            "origin",
            f"{SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}:refs/heads/{UPSTREAM_BRANCH}",
        ],
        cwd=repo,
    )


def probe_clean_prefix(
    repo: Path,
    commits: list[str],
) -> tuple[str | None, str | None]:
    """
    Walk commits oldest-to-newest on a throwaway branch off origin/TARGET_BRANCH.
    Returns (last_clean_commit, first_conflict_commit).
    """
    run(["git", "checkout", "-B", "probe", f"origin/{TARGET_BRANCH}"], cwd=repo)

    last_clean = None
    first_conflict = None

    try:
        for commit in commits:
            result = run(["git", "merge", "--no-edit", commit], cwd=repo, check=False)
            if result.returncode != 0:
                run(["git", "merge", "--abort"], cwd=repo, check=False)
                first_conflict = commit
                break
            last_clean = commit
    finally:
        run(["git", "checkout", "--detach", "HEAD"], cwd=repo, check=False)
        run(["git", "branch", "-D", "probe"], cwd=repo, check=False)

    return last_clean, first_conflict


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    repo = WORK_DIR / "binutils-gdb"

    try:
        # ------------------------------------------------------------------ #
        # 1. Clone or reuse.                                                   #
        # ------------------------------------------------------------------ #
        if (repo / ".git").exists():
            print(f"Reusing existing clone at {repo}…")
        else:
            print(f"Cloning {ROCGDB_REPO}…")
            run_net(["git", "clone", ROCGDB_ORIGIN_URL, str(repo)])

        remotes = run(["git", "remote"], cwd=repo).stdout.split()
        if SOURCEWARE_REMOTE not in remotes:
            run(["git", "remote", "add", SOURCEWARE_REMOTE, SOURCEWARE_URL], cwd=repo)

        # ------------------------------------------------------------------ #
        # 2. Fetch SOURCEWARE_REMOTE/UPSTREAM_BRANCH and origin/TARGET_BRANCH.  #
        # ------------------------------------------------------------------ #
        print(f"Fetching {SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}…")
        run_net(["git", "fetch", SOURCEWARE_REMOTE, UPSTREAM_BRANCH], cwd=repo)

        print(f"Fetching origin/{TARGET_BRANCH}…")
        run_net(["git", "fetch", "origin", TARGET_BRANCH], cwd=repo)

        if not DRY_RUN:
            push_upstream_mirror(repo)

        # ------------------------------------------------------------------ #
        # 3. Skip if a conflict or conflict-free PR is already open.          #
        # ------------------------------------------------------------------ #
        existing_conflict_pr = find_open_conflict_pr()
        if existing_conflict_pr:
            print(
                f"Open conflict PR found: {existing_conflict_pr}\n"
                "Skipping merge — resolve the PR first."
            )
            return

        existing_conflict_free_pr = find_open_conflict_free_pr()
        if existing_conflict_free_pr:
            print(
                f"Open conflict-free PR found: {existing_conflict_free_pr}\n"
                "Skipping merge — wait for the PR to be merged first."
            )
            return

        # ------------------------------------------------------------------ #
        # 4. Determine the commit range to merge.                             #
        # ------------------------------------------------------------------ #
        merge_base_sha = run(
            [
                "git",
                "merge-base",
                f"origin/{TARGET_BRANCH}",
                f"{SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}",
            ],
            cwd=repo,
        ).stdout.strip()

        log_out = run(
            [
                "git",
                "log",
                "--format=%H",
                "--reverse",
                f"{merge_base_sha}..{SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}",
            ],
            cwd=repo,
        ).stdout.strip()

        if not log_out:
            print(
                f"{TARGET_BRANCH} is already up to date with {SOURCEWARE_REMOTE}/{UPSTREAM_BRANCH}. Nothing to do."
            )
            return

        commits = log_out.splitlines()
        print(
            f"Found {len(commits)} new commit(s) to merge "
            f"({commits[0][:12]}..{commits[-1][:12]})."
        )

        # ------------------------------------------------------------------ #
        # 5. Set up local target branch (skipped in dry-run; probe uses the  #
        #    remote ref directly and we have no local branch to return to).   #
        # ------------------------------------------------------------------ #
        if not DRY_RUN:
            run(
                ["git", "checkout", "-B", TARGET_BRANCH, f"origin/{TARGET_BRANCH}"],
                cwd=repo,
            )

        # ------------------------------------------------------------------ #
        # 6. Probe for the largest clean prefix.                              #
        # ------------------------------------------------------------------ #
        print("Probing for clean merge prefix…")
        last_clean, first_conflict = probe_clean_prefix(repo, commits)

        # Probe leaves the repo on detached HEAD; return to the local branch.
        if not DRY_RUN:
            run(["git", "checkout", TARGET_BRANCH], cwd=repo)

        # ------------------------------------------------------------------ #
        # 7a. All commits apply cleanly → open a conflict-free PR.           #
        # ------------------------------------------------------------------ #
        if first_conflict is None:
            print(f"All {len(commits)} commit(s) merge cleanly.")
            conflict_free_branch = (
                f"{CONFLICT_FREE_BRANCH_PREFIX}-{date.today().isoformat()}-{commits[-1][:8]}"
            )
            conflict_free_title = (
                f"ROCgdb {UPSTREAM_BRANCH} → {TARGET_BRANCH} conflict-free: "
                f"{merge_base_sha[:12]}..{commits[-1][:12]}"
            )
            if DRY_RUN:
                print(
                    f"[DRY RUN] Would open conflict-free PR:\n"
                    f"  Branch : {conflict_free_branch}\n"
                    f"  Commits: {merge_base_sha[:12]}..{commits[-1][:12]}"
                    f" ({len(commits)} commit(s))\n"
                    f"  Title  : {conflict_free_title}"
                )
            else:
                run_net(
                    ["git", "push", "origin", f"{commits[-1]}:refs/heads/{conflict_free_branch}"],
                    cwd=repo,
                )
                open_conflict_free_pr(
                    branch=conflict_free_branch,
                    first_commit=merge_base_sha,
                    last_commit=commits[-1],
                    title=conflict_free_title,
                )
            return

        # ------------------------------------------------------------------ #
        # 7b. Partial clean prefix → open a conflict-free PR for clean range.#
        # ------------------------------------------------------------------ #
        if last_clean is not None:
            print(f"Clean prefix ends at {last_clean[:12]}. Opening conflict-free PR…")
            clean_count = commits.index(last_clean) + 1
            conflict_free_branch = (
                f"{CONFLICT_FREE_BRANCH_PREFIX}-{date.today().isoformat()}-{last_clean[:8]}"
            )
            conflict_free_title = (
                f"ROCgdb {UPSTREAM_BRANCH} → {TARGET_BRANCH} conflict-free: "
                f"{merge_base_sha[:12]}..{last_clean[:12]}"
            )
            if DRY_RUN:
                print(
                    f"[DRY RUN] Would open conflict-free PR:\n"
                    f"  Branch : {conflict_free_branch}\n"
                    f"  Commits: {merge_base_sha[:12]}..{last_clean[:12]}"
                    f" ({clean_count} commit(s))\n"
                    f"  Title  : {conflict_free_title}"
                )
            else:
                run_net(
                    ["git", "push", "origin", f"{last_clean}:refs/heads/{conflict_free_branch}"],
                    cwd=repo,
                )
                open_conflict_free_pr(
                    branch=conflict_free_branch,
                    first_commit=merge_base_sha,
                    last_commit=last_clean,
                    title=conflict_free_title,
                )
                return
        else:
            print("No clean commits to merge; first commit already conflicts.")

        # ------------------------------------------------------------------ #
        # 8. Create conflict branch and open PR.                              #
        # ------------------------------------------------------------------ #
        conflict_branch = (
            f"{CONFLICT_BRANCH_PREFIX}-{date.today().isoformat()}-{first_conflict[:8]}"
        )

        conflict_info = run(
            ["git", "log", "-1", "--format=%H %s (%aN <%aE>)", first_conflict],
            cwd=repo,
        ).stdout.strip()
        print(f"\nConflict commit: {conflict_info}")

        # Probe the conflicted files by attempting the merge on a scratch branch.
        run(
            ["git", "checkout", "-B", "probe-conflict", f"origin/{TARGET_BRANCH}"],
            cwd=repo,
        )
        run(["git", "merge", "--no-edit", first_conflict], cwd=repo, check=False)
        conflicted_files = (
            run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=repo)
            .stdout.strip()
            .splitlines()
        )
        run(["git", "merge", "--abort"], cwd=repo, check=False)
        run(["git", "checkout", "--detach", "HEAD"], cwd=repo, check=False)
        run(["git", "branch", "-D", "probe-conflict"], cwd=repo, check=False)

        conflict_title = (
            f"ROCgdb {UPSTREAM_BRANCH} → {TARGET_BRANCH} conflict: "
            f"{merge_base_sha[:12]}..{first_conflict[:12]}"
        )

        if DRY_RUN:
            files_str = "\n".join(f"    {f}" for f in conflicted_files)
            print(
                f"[DRY RUN] Would open conflict PR:\n"
                f"  Branch            : {conflict_branch}\n"
                f"  Conflicting commit: {conflict_info}\n"
                f"  Conflicted files  :\n{files_str}\n"
                f"  Title             : {conflict_title}"
            )
        else:
            # Checkout a fresh conflict branch from the current target branch state.
            run(["git", "checkout", "-B", conflict_branch], cwd=repo)
            run(["git", "merge", "--no-edit", first_conflict], cwd=repo, check=False)

            print("Conflicted files:")
            for f in conflicted_files:
                print(f"  {f}")

            # Stage and commit the conflict markers so the branch carries a diff
            # that gh pr create can open (the branch would otherwise be identical
            # to the target branch and gh would reject it with "no commits between...").
            run(["git", "add", "-A"], cwd=repo)
            run(
                [
                    "git",
                    "commit",
                    "--no-verify",
                    "-m",
                    f"Merge {first_conflict[:12]} into {TARGET_BRANCH} (unresolved conflicts)",
                ],
                cwd=repo,
            )

            run_net(["git", "push", "origin", conflict_branch], cwd=repo)
            open_conflict_pr(
                branch=conflict_branch,
                merge_base_sha=merge_base_sha,
                conflict_commit=first_conflict,
                conflicted_files=conflicted_files,
                title=conflict_title,
            )

    except ConnectivityError:
        print("Connectivity error: all retries exhausted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
