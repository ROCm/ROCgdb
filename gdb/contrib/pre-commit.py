#!/usr/bin/env python3

# Copyright (C) 2026 Free Software Foundation, Inc.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import subprocess
import sys

import yaml


def get_repos(cfg):
    with open(cfg, "r") as f:
        data = yaml.safe_load(f)
        repos = data.get("repos")
        if not repos:
            raise RuntimeError("repos missing")
        return repos


def config_check_repo(repo):
    name = repo.get("repo")
    if not name:
        raise RuntimeError("no repo")
    if name == "local":
        # Skip local repo, there's no revision to check.
        return True

    # Get the revision.
    rev = repo.get("rev")
    if not rev:
        raise RuntimeError("empty revision")

    # Normalize revision: skip 'v' prefix.
    if rev[0].lower() == "v":
        rev = rev[1:]

    # Check version number.  Don't allow pre-releases like 9.0.0b1.
    # We currently only need to support x.y.z, but that could change.
    if not re.fullmatch(r"\d+[.]\d+[.]\d+", rev):
        print("Revision %s for repo %s not allowed." % (rev, name))
        return False

    return True


def config_check(cfg):
    for repo in get_repos(cfg):
        if not config_check_repo(repo):
            sys.exit(1)


def run_cmd(cmd, **kwargs):
    res = subprocess.run(cmd, **kwargs)
    if res.returncode != 0:
        raise RuntimeError(
            "command %s failed with exit status %s" % (cmd, res.returncode)
        )
    return res


def is_clean(args):
    if not isinstance(args, list):
        args = [args]
    cmd = ["git", "status", "--porcelain"] + args
    res = run_cmd(cmd, capture_output=True, text=True)
    return res.stdout == ""


def autoupdate_repo(cfg, repo):
    name = repo["repo"]
    if name == "local":
        # Skip local repo, there's no revision to update.
        return

    cmd = ["pre-commit", "autoupdate", "--repo", name]
    run_cmd(cmd)

    if is_clean(cfg):
        # No autoupdate changes.
        return

    rev = repo["rev"]

    # Config has changed, refresh repo.
    found = False
    for new_repo in get_repos(cfg):
        if new_repo["repo"] == name:
            found = True
            break
    if not found:
        raise RuntimeError("Repo not found in updated %s" % cfg)
    repo = new_repo

    new_rev = repo["rev"]

    if not config_check_repo(repo):
        # Reject autoupdate for this repo.  Throwing away changes here is safe
        # because we checked in function autoupdate that cfg is clean.
        cmd = ["git", "checkout", "-f", cfg]
        # Capture and ignore output.
        run_cmd(cmd, capture_output=True, text=True)
        return

    # Commit autoupdate for this repo.
    try:
        name_for_msg = repo["hooks"][0]["id"]
    except (KeyError, IndexError):
        name_for_msg = name
    msg = 'pre-commit: Update %s: %s -> %s\n\nRan "pre-commit.py --autoupdate".' % (
        name_for_msg,
        rev,
        new_rev,
    )
    cmd = ["git", "commit", "-m", msg, cfg]
    run_cmd(cmd)


def autoupdate(cfg):
    if not is_clean(cfg):
        print("Not clean: %s" % cfg)
        sys.exit(1)

    for repo in get_repos(cfg):
        autoupdate_repo(cfg, repo)


def usage():
    print("Usage: pre-commit.py --config-check [<file>]")
    print("                     --autoupdate")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()

    cfg = ".pre-commit-config.yaml"

    if sys.argv[1] == "--config-check":
        if len(sys.argv) not in [2, 3]:
            usage()
        if len(sys.argv) == 3:
            cfg = sys.argv[2]
        config_check(cfg)
        return

    if sys.argv[1] == "--autoupdate":
        if len(sys.argv) != 2:
            usage()
        autoupdate(cfg)
        return

    usage()


main()
