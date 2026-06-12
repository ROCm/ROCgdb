# Copyright 2026 Free Software Foundation, Inc.
# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.

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

# Child driver for slot-pool-lock.exp.  Each invocation acquires one
# slot-pool lock, holds it for HOLD_MS milliseconds, releases, and
# exits.  Acquire / hold / release events are appended to LOG_FILE
# with microsecond timestamps so the parent can reason about
# concurrency.  When done, the child touches DONE_FILE.
#
# Usage:
#   tclsh slot-pool-lock-helper.tcl \
#       LOCK_DIR KIND HOLD_MS LOG_FILE DONE_FILE ID
#
# KIND is one of:
#   shared-<npools>-<perpool>
#   exclusive-<npools>-<perpool>
#   machine-<npools>-<perpool>

# Stubs for the bits of gdb-utils.exp that touch DejaGnu state.
proc verbose {args} {}
set ::subdir "gdb.testsuite"
set ::gdb_test_file_name "slot-pool-lock-helper"

set here [file dirname [file normalize [info script]]]
source [file join $here .. lib gdb-utils.exp]

if {[llength $argv] != 6} {
    puts stderr "usage: $argv0 LOCK_DIR KIND HOLD_MS LOG_FILE DONE_FILE ID"
    exit 2
}
lassign $argv lock_dir kind hold_ms log_file done_file id

if {![regexp {^(shared|exclusive|machine)-(\d+)-(\d+)$} $kind -> mode npools per_pool]} {
    puts stderr "bad KIND: $kind"
    exit 2
}
if {$npools <= 0 || $per_pool <= 0} {
    puts stderr "bad KIND: npools and per_pool must be > 0 (got $npools, $per_pool)"
    exit 2
}

proc ev {log id tag} {
    set ch [open $log a]
    puts $ch "[clock microseconds] $id $tag"
    close $ch
}

ev $log_file $id want

# Wrap acquire/release in catch so a regression in the lock layer
# surfaces as a clean stderr line + non-zero exit instead of a silent
# hang or an unhelpful Tcl stack trace.
if {[catch {
    switch -- $mode {
	shared {
	    set tok [lock_file_acquire_shared_multi $lock_dir slot \
			 $npools $per_pool barrier]
	}
	exclusive {
	    set tok [lock_file_acquire_exclusive_multi $lock_dir slot \
			 $npools $per_pool barrier]
	}
	machine {
	    set tok [lock_file_acquire_machine_multi $lock_dir slot \
			 $npools $per_pool barrier]
	}
    }
} msg]} {
    puts stderr "acquire failed: $msg"
    exit 1
}

# Token shapes:
#   {shared    <tok>         <pool> <slot>}
#   {exclusive <barrier_tok> <slot_toks>  <pool> 0}
#   {machine   <barrier_toks> <slot_toks>}
if {$mode eq "machine"} {
    set pool "all"
    set slot "all"
} else {
    set pool [lindex $tok end-1]
    set slot [lindex $tok end]
}
ev $log_file $id "hold pool=$pool slot=$slot"

after $hold_ms

ev $log_file $id release
if {[catch {lock_file_release_shared $tok} msg]} {
    puts stderr "release failed: $msg"
    exit 1
}

set ch [open $done_file w]
close $ch
