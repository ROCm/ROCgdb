# .github/configs.json field reference

`configs.json` holds the TheRock dependency pins used by CI. It is the
single source of truth for which TheRock commit and build container image
the CI workflow uses. The automated update workflow
(`.github/workflows/therock-deps-update.yml`) rewrites this file and opens
a PR when either pin changes.

## Fields

### `therock_commit_ref`

Full 40-character SHA of the TheRock commit pinned for CI. Read by the
`resolve` job in `therock-ci-linux.yml` and passed to the TheRock checkout
step as `THEROCK_COMMIT_REF`.

### `therock_commit_created`

ISO 8601 timestamp of when `therock_commit_ref` was committed, taken from
the GitHub Commits API (`commit.committer.date`). Not consumed by CI; used
by the update script to preserve the original timestamp when only the build
image changes.

### `build_image`

Full container image reference including registry, image name, and digest,
e.g. `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:<64-hex-chars>`.
Read by the `resolve` job and used as `container.image` for the build job.
Must include the `@sha256:` digest suffix; the resolve job validates this.

### `build_image_created`

ISO 8601 timestamp of when the build container image was created, taken
from the OCI config blob `created` field. Not consumed by CI; used by the
update script to preserve the original timestamp when only the TheRock
commit changes.
