#! /usr/bin/env python3

import subprocess
import sys
from itertools import chain

# These files are required to be identical.
required_list = [
    ["gdb/.dir-locals.el", "gdbserver/.dir-locals.el", "gdbsupport/.dir-locals.el"],
    ["gdb/.shellcheckrc", "gdbserver/.shellcheckrc", "gdbsupport/.shellcheckrc"],
]


def run_cmd(cmd, **kwargs):
    res = subprocess.run(cmd, **kwargs)
    if res.returncode != 0:
        raise RuntimeError(
            "command %s failed with exit status %s" % (cmd, res.returncode)
        )
    return res


files = sys.argv[1:]
always_check_files = list(chain.from_iterable(required_list))
cmd = ["git", "ls-files", "--stage"] + files + always_check_files
res = run_cmd(cmd, capture_output=True, text=True)

hash_dict = {}
for line in res.stdout.splitlines():
    parts = line.split(maxsplit=3)
    hash = parts[1]
    file = parts[3]
    hash_dict.setdefault(hash, []).append(file)

copies = []
for hash in hash_dict:
    files = hash_dict[hash]
    if len(files) == 1:
        continue
    copies.append(set(files))

required_set = [set(r) for r in required_list]

for required in required_set:
    if required in copies:
        continue
    print("No longer copies: %s" % required)
    sys.exit(1)

for copy in copies:
    if copy in required_set:
        continue
    print("Accidental copies found: %s" % copy)
    sys.exit(1)
