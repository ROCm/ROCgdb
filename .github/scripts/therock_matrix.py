# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Maps ROCgdb source changes to TheRock build flags and tests.

ROCgdb is the source of truth for a single component (rocgdb, built in TheRock's
debug-tools stage). Unlike a monorepo, its files live at the repository root, so
subtree mapping is trivial: any change maps to the single "rocgdb" project.

TheRock's build_tools/github_actions/detect_external_repo_config.py imports this
module for two things:
  * _is_valid_repo_path(): requires this file (and therock_configure_ci.py) to
    exist for ROCgdb to be treated as a valid external repo.
  * get_test_list(): unions project_map[*]["project_to_test"] to know which tests
    the external repo defines. It reads the "project_to_test" key (list or CSV)
    and ignores cmake_options (multi-arch does full builds).
"""

# Root-level layout: the whole repository is the single rocgdb component.
subtree_to_project_map = {
    ".": "rocgdb",
}

project_map = {
    "rocgdb": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ],
        "project_to_test": ["rocgdb-cpu", "rocgdb-gpu", "rocgdb-corefile"],
    },
}

# ROCgdb multi-arch CI targets Linux only for now.
windows_only_subtrees = set()

trigger_windows_ci_for_subtrees_paths = []
