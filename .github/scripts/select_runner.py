# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Select a GitHub Actions runner label from therock-ci-config.

Usage: python select_runner.py <output_var> [ci_config_path]

Writes <output_var>=<label> to GITHUB_OUTPUT and prints it to stdout.
Falls back to aws-linux-scale-rocm-prod if the config is unavailable.
"""

import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: select_runner.py <output_var> [ci_config_path]", file=sys.stderr)
        sys.exit(1)

    output_var = sys.argv[1]
    ci_config = Path(sys.argv[2] if len(sys.argv) > 2 else "ci-config")
    default_runner = "aws-linux-scale-rocm-prod"

    label = default_runner
    if (ci_config / "runner-config.json").exists():
        try:
            sys.path.insert(0, str(ci_config))
            from ci_config_api import load_config_v1

            config = load_config_v1(ci_config)
            candidates = config.build_runners.get("linux", {}).get("default", [])
            best = max(candidates, key=lambda e: e["weight"], default=None)
            if best:
                label = best["label"]
        except Exception as e:
            print(
                f"Failed to load ci_config_api, using default: {e}", file=sys.stderr
            )
    else:
        print(
            f"ci-config not found, using default: {default_runner}", file=sys.stderr
        )

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{output_var}={label}\n")
    print(f"{output_var}={label}")


if __name__ == "__main__":
    main()
