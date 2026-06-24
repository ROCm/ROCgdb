# Copyright (C) 2026 Free Software Foundation, Inc.
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

# This file is part of GDB.

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

# Shared driver helpers for the gdb.rocm multi-inferior tests.  Tests
# source this file with:
#
#   source $srcdir/$subdir/multi-inferior.exp.tcl
#
# and drive a session built around the multi-inferior.cpp program (the
# "Break here" / kern / "Last break here" source markers).

# Drive a HIP multi-inferior session.  Load the program, turn on
# non-stop mode with detach-on-fork off and follow-fork parent, set
# breakpoints on the "Break here" / kern / "Last break here" markers,
# run the parent to the pre-fork breakpoint, resume everything in the
# background, and wait for one "kern" breakpoint stop per child.
#
#   arg_list   - arguments passed via "set args" ("" for none).
#   expected   - number of child stops to wait for.  If "", it is read
#                from the "num_devices" value at the pre-fork stop.
#
# Return the list of stopped GPU thread ids (one per child).

proc rocm_multi_inferior_run_to_kernels { {arg_list ""} {expected ""} } {
    clean_restart
    gdb_load $::binfile

    gdb_test_no_output "set non-stop on"
    gdb_test_no_output "set detach-on-fork off"
    gdb_test_no_output "set follow-fork parent"
    if { $arg_list ne "" } {
	gdb_test_no_output "set args $arg_list"
    }

    gdb_breakpoint [gdb_get_line_number "Break here"]
    gdb_breakpoint kern allow-pending
    gdb_breakpoint [gdb_get_line_number "Last break here"]

    # Run the parent up to the pre-fork sync point.  GDB prints the
    # source line of the stop after the breakpoint line, so the pattern
    # must account for breakpoint-hit + source-line before the prompt.
    gdb_test_multiple "run" "run to fork point" {
	-re -wrap "hit Breakpoint\[^\r\n\]*parent\[^\r\n\]*\r\n\[^\r\n\]+" {
	    pass $gdb_test_name
	}
    }

    if { $expected eq "" } {
	set expected [get_integer_valueof "num_devices" 0]
    }

    # Resume everything.  As children are forked and re-exec, each
    # launches its kernel and we expect one "hit Breakpoint ..., kern ()"
    # notification per child inferior.
    gdb_test_multiple "continue -a &" "continue -a in non-stop" {
	-re "Continuing\\.\r\n$::gdb_prompt " {
	    pass $gdb_test_name
	}
    }

    set seen [list]
    set threads [list]
    gdb_test_multiple "" "wait for all GPU stops" {
	-re "Thread ($::decimal)\\.($::decimal)\[^\r\n\]* hit Breakpoint\[^\r\n\]*, kern \\(\\)\[^\r\n\]*\r\n" {
	    set inf $expect_out(1,string)
	    if { [lsearch -exact $seen $inf] == -1 } {
		lappend seen $inf
		lappend threads "$inf.$expect_out(2,string)"
	    }
	    if { [llength $seen] < $expected } {
		exp_continue
	    } else {
		pass $gdb_test_name
	    }
	}
	timeout {
	    fail $gdb_test_name
	    verbose -log "only [llength $seen] of $expected inferiors stopped"
	}
    }

    return $threads
}

# Continue each stopped GPU inferior in THREADS to a clean exit, wait
# for the parent to reach its post-waitpid breakpoint, then run the
# parent to completion.  THREADS is the list returned by
# rocm_multi_inferior_run_to_kernels.

proc rocm_multi_inferior_drain { threads } {
    foreach thread $threads {
	set inf [lindex [split $thread .] 0]
	gdb_test "thread $thread" "Switching to thread.*" \
	    "switch to GPU thread in inferior $inf"
	gdb_test_multiple "continue" "continue inferior $inf to end" {
	    -re "\\\[Inferior $inf \[^\r\n\]* exited normally\\\]\r\n$::gdb_prompt " {
		pass $gdb_test_name
	    }
	    -re "\\\[Inferior $inf \[^\r\n\]* exited with code\[^\r\n\]*\\\]" {
		fail "$gdb_test_name (non-zero exit)"
	    }
	}
    }

    # Once all children have exited, the parent's waitpid loop falls
    # through to the post-fork breakpoint.
    gdb_test_multiple "" "parent reached post-waitpid breakpoint" {
	-re "hit Breakpoint\[^\r\n\]*parent\[^\r\n\]*" {
	    pass $gdb_test_name
	}
	timeout {
	    fail $gdb_test_name
	}
    }

    gdb_test "inferior 1" "Switching to inferior 1.*"
    gdb_continue_to_end "" "continue -a" 1
}
