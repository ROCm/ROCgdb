# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Project matrix consumed by TheRock's detect_external_repo_config.py.

TheRock loads this module from external repos to discover which tests
to run. Each value in `project_map` describes a buildable project; the
`project_to_test` field lists test labels that TheRock's
fetch_test_configurations.py maps to test jobs.
"""

# Subtree → project mapping is unused for ROCgdb (TheRock's path-based
# project pruning isn't applied here) but the symbol is part of the
# convention shared with rocm-libraries / rocm-systems.
subtree_to_project_map = {}

project_map = {
    "rocgdb": {
        "cmake_options": [],
        "project_to_test": ["rocgdb-cpu", "rocgdb-gpu"],
    },
}
