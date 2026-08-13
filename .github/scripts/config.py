# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read and validate fields from .github/configs.json.

Usage: python3 config.py <field> [<field> ...]

Validates each requested field and writes field=value to $GITHUB_OUTPUT.
Exits non-zero if the config file is missing, malformed, or any field
fails validation.
"""

import json
import os
import re
import sys
from pathlib import Path

# Maps each known field name to a (description, validator) pair.
_VALIDATORS = {
    "therock_commit_ref": (
        "40-character lowercase hex SHA",
        re.compile(r"^[0-9a-f]{40}$").match,
    ),
    "build_image": (
        "ghcr.io image reference with sha256 digest",
        re.compile(r"^ghcr\.io/[^@]+@sha256:[0-9a-f]{64}$").match,
    ),
}

CONFIGS_PATH = Path(".github/configs.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: config.py <field> [<field> ...]", file=sys.stderr)
        sys.exit(1)

    fields = sys.argv[1:]

    for field in fields:
        if field not in _VALIDATORS:
            known = ", ".join(sorted(_VALIDATORS))
            print(f"Unknown field {field!r}. Known fields: {known}", file=sys.stderr)
            sys.exit(1)

    try:
        d = json.loads(CONFIGS_PATH.read_text())
    except FileNotFoundError:
        print(f"{CONFIGS_PATH} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{CONFIGS_PATH} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for field in fields:
        description, validate = _VALIDATORS[field]
        value = d.get(field, "")
        if not isinstance(value, str) or not validate(value):
            print(
                f"Invalid {field!r} (expected {description}): {value!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        results[field] = value

    github_output = os.environ.get("GITHUB_OUTPUT")
    for field, value in results.items():
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"{field}={value}\n")
        print(f"{field}={value}")


if __name__ == "__main__":
    main()
