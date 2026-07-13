# Automerge Workflow

Automatically syncs `sourceware/master` into `origin/amd-staging` every 6 hours,
opening GitHub PRs for CI validation or conflict resolution as needed.

## Files

- `.github/workflows/automerge.yml` — GitHub Actions workflow definition.
- `.github/scripts/automerge.py` — Python sync script invoked by the workflow.

## Triggers

| Event                 | Condition                                             | Effect            |
| --------------------- | ----------------------------------------------------- | ----------------- |
| Scheduled             | Every 6 hours                                         | Normal sync run   |
| `workflow_dispatch`   | Manual                                                | Normal sync run   |
| `pull_request` closed | PR merged + label `merge-testing` or `merge-conflict` | Immediate re-sync |

## Sync Logic

Each run follows these steps:

1. **Clone or reuse** a local clone of `ROCm/ROCgdb` in `$WORKSPACE/rocgdb_sync/`.
2. **Fetch** `sourceware/master` and `origin/amd-staging`, then immediately fast-forward `origin/master` to `sourceware/master` to keep the mirror current.
3. **Gate** — if a fast-forward or conflict PR is already open, skip and exit.
4. **Compute** the commit range: `merge-base(amd-staging, sourceware/master)..sourceware/master`.
5. **Probe** — walk commits oldest-to-newest on a throwaway branch off `amd-staging`
   to find the largest clean-merging prefix.
6. **Act** based on the probe result:

### 6a. All commits merge cleanly

Opens a **fast-forward PR** covering the full range. CI must pass before anyone
merges it. Once merged, the `pull_request` closed trigger fires another sync run
to pick up any further upstream commits.

### 6b. Partial clean prefix

Opens a **fast-forward PR** for the clean prefix only. The conflict commit is left
for the next run — after the FF PR merges, the re-sync will find the conflict and
open a conflict PR then.

### 6c. No clean commits

Opens a **conflict PR** directly against `amd-staging`. The PR is labeled
`ci:skip` to suppress CI until conflicts are resolved manually. After the PR is
merged, the `pull_request` closed trigger fires another sync run.

## PR Types

### Fast-forward PR

- **Branch**: `users/github/master-to-amd-staging-ff-YYYY-MM-DD-<hash>`
- **Label**: `merge-testing`
- **Base**: `amd-staging`
- **Purpose**: CI gate before fast-forwarding upstream commits into staging.

### Conflict PR

- **Branch**: `users/github/master-to-amd-staging-conflict-YYYY-MM-DD-<hash>`
- **Labels**: `merge-conflict`, `ci:skip`
- **Base**: `amd-staging`
- **Purpose**: Signals a merge conflict that requires manual resolution.
  Remove the `ci:skip` label once conflicts are resolved to allow CI to run.

## Configuration

Environment variables:

| Variable / flag      | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `WORKSPACE` / `--workspace` | Root directory for the working clone. Defaults to `$PWD`. The clone is created at `DIR/rocgdb_sync/binutils-gdb`. The CLI flag takes precedence over the env var. |

Key constants at the top of `automerge.py`:

| Constant                 | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `ROCGDB_REPO`            | GitHub repo slug (`ROCm/ROCgdb`)                         |
| `SOURCEWARE_URL`         | Upstream GDB repository URL                              |
| `FF_BRANCH_PREFIX`       | Branch prefix for fast-forward PRs                       |
| `CONFLICT_BRANCH_PREFIX` | Branch prefix for conflict PRs                           |
| `FF_LABEL`               | Label applied to fast-forward PRs (`merge-testing`)      |
| `CONFLICT_LABEL`         | Label applied to conflict PRs (`merge-conflict`)         |
| `CI_SKIP_LABEL`          | Label applied to conflict PRs to suppress CI (`ci:skip`) |
| `RETRY_DELAYS`           | Sleep intervals (seconds) between network retries        |
