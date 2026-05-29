# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Path filter patterns consumed by TheRock's detect_external_repo_config.py.

TheRock's multi-arch CI orchestrates job selection itself; this module's
only remaining job is to expose SKIPPABLE_PATH_PATTERNS for TheRock's
get_skip_patterns() helper.
"""

# Paths matching any of these patterns are considered to have no influence
# over build or test workflows. If every modified file in a commit/PR matches
# one of these patterns, related jobs can be skipped.
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*LICENSE*",
    "*NOTICES*",
    ".github/CODEOWNERS",
    ".github/*.md",
    ".github/dependabot.yml",
    ".azuredevops*",
]
