# Automerge Workflow

Automatically syncs the configured upstream branch into the target branch every
6 hours, opening GitHub PRs for CI validation or conflict resolution as needed.

## Files

- `.github/workflows/automerge.yml` — GitHub Actions workflow definition.
- `.github/scripts/automerge.py` — Python sync script invoked by the workflow.
- `.github/scripts/test_automerge.py` — Unit tests for automerge.py.

## Triggers

| Event               | Effect          |
| ------------------- | --------------- |
| Scheduled (6 hours) | Normal sync run |
| `workflow_dispatch` | Manual run; exposes a `dry_run` boolean input |

## Repository Variables

Both variables must be set in the repository settings before the workflow runs:

| Variable                    | Example       | Description                        |
| --------------------------- | ------------- | ---------------------------------- |
| `AUTOMERGE_TARGET_BRANCH`   | `amd-staging` | Branch to merge upstream commits into |
| `AUTOMERGE_UPSTREAM_BRANCH` | `master`      | Upstream branch to pull from       |

## Dry-Run Mode

Pass `--dry-run` on the CLI, or enable the `dry_run` input in a manual
`workflow_dispatch` run. In dry-run mode the script fetches and probes normally
but skips all pushes and PR creation, printing what it would do instead.

## Sync Logic

Each run follows these steps:

1. **Acquire repo** — use an existing clone passed via `--repo`, or clone/reuse
   one in `$WORKSPACE/rocgdb_sync/`. In GHA the checked-out workspace is passed
   as `--repo`, avoiding a redundant clone.
2. **Fetch upstream and update mirror** — fetch the upstream branch from
   sourceware, then fast-forward `origin/<upstream>` to keep the mirror current.
3. **Gate** — if a conflict-free or conflict PR is already open, skip and exit.
4. **Fetch target** — fetch `origin/<target>` (deferred until after the gate to
   avoid unnecessary work when a PR is already open).
5. **Compute** the commit range: `merge-base(TARGET, upstream)..upstream`.
6. **Probe** — walk commits oldest-to-newest on a throwaway branch off the target
   to find the largest clean-merging prefix.
7. **Act** based on the probe result:

### 7a. All commits merge cleanly

Opens a **conflict-free PR** covering the full range. CI must pass before anyone
merges it.

### 7b. Partial clean prefix

Opens a **conflict-free PR** for the clean prefix only. The conflict commit is
left for the next scheduled run.

### 7c. No clean commits

Opens a **conflict PR** directly against the target branch. The PR is labeled
`ci:skip` to suppress CI until conflicts are resolved manually.

## PR Types

### Conflict-free PR

- **Branch**: `users/github/<upstream>-to-<target>-conflict-free-YYYY-MM-DD-<hash>`
- **Label**: `merge-testing`
- **Base**: target branch
- **Purpose**: CI gate before landing upstream commits into the target branch.

### Conflict PR

- **Branch**: `users/github/<upstream>-to-<target>-conflict-YYYY-MM-DD-<hash>`
- **Labels**: `merge-conflict`, `ci:skip`
- **Base**: target branch
- **Purpose**: Signals a merge conflict requiring manual resolution.
  Remove the `ci:skip` label once conflicts are resolved to allow CI to run.

## Configuration

| Variable / flag             | Description                                                                      |
| --------------------------- | -------------------------------------------------------------------------------- |
| `AUTOMERGE_TARGET_BRANCH`   | Required. Branch to merge into.                                                  |
| `AUTOMERGE_UPSTREAM_BRANCH` | Required. Upstream branch to pull from.                                          |
| `GITHUB_REPOSITORY`         | Required. Set automatically by GitHub Actions (`owner/repo`).                    |
| `WORKSPACE` / `--workspace` | Root directory for the working clone. Defaults to `$PWD`. Clone is created at `DIR/rocgdb_sync/binutils-gdb`. The CLI flag takes precedence over the env var. |
| `--repo`                    | Path to an existing ROCgdb clone to use directly, skipping the clone step. Used in GHA to reuse the `actions/checkout` workspace. |
| `--dry-run`                 | Skip all pushes and PR creation; print what would happen instead.                |
