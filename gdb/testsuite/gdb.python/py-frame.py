# Copyright (C) 2026 Free Software Foundation, Inc.
#
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

# A class which is a GDB frame unwinder.  This frame unwinder can be
# used to corrupt the backtrace by tricking GDB into believing that
# there is a repeated frame-id in the backtrace.
#
# Sourcing this script creates a global called 'global_unwinder' and
# registers a global unwinder called 'stop-at-level'.  The unwinder
# needs to be primed with the frame-ids copied from the current stack,
# this can be done by enabling the unwinder using the GDB command:
#
# (gdb) enable unwinder global stop-at-level
#
# You then need to set the level at which to corrupt the stack, e.g.:
#
# (gdb) python global_unwinder.stop_at_level = 3
#
# This will allow the frames #0, #1, #2, and #3 to print correctly,
# but frame #4 will appear to be a repeat of frame #3, so GDB will
# terminate the backtrace.
#
# As the unwinder relies on cached frame-ids, then resuming the
# inferior for which the frame-ids were cached will cause the unwinder
# to auto-disable itself, after which you'll need to re-enable the
# next time the inferior stops.
#
# Changing the 'global_unwinder.stop_at_level' will flush the frame
# cache, so you can test breaking the stack at different levels.


import re
import gdb
from gdb.unwinder import Unwinder, FrameId


class corrupt_stack_unwinder(Unwinder):
    def __init__(self):
        # Is this unwinder enabled or not?  Accessed via the 'enabled'
        # property.
        self._enabled = True

        # List of FrameId instances, one for each stack frame.  This list is
        # populated when this file is sourced into GDB.
        self._frame_ids = []

        # The inferior for which we cached self._frame_ids.  The
        # unwinder will only work within this inferior.
        self._inferior = None

        # At what stack level should we break the unwind?
        self._stop_at_level = None

        Unwinder.__init__(self, "stop-at-level")

        if gdb.selected_inferior().pid > 0:
            self._update_frame_id_cache()
        else:
            self._enabled = False

        gdb.events.cont.connect(lambda ev: self._continued_event_handler(ev))

    # Assume that the current inferior has a stack.  Collect a FrameId
    # object for each level of the current stack.
    def _update_frame_id_cache(self):
        self._frame_ids = []
        self._inferior = gdb.selected_inferior()

        frame = gdb.newest_frame()
        while frame is not None:
            # Get the frame-id in a verbose text form.
            output = gdb.execute(
                "maint print frame-id %d" % frame.level(), to_string=True
            )

            # Parse the frame-id in OUTPUT, find the stack and code addresses.
            match = re.search(r"stack=(0x[0-9a-fA-F]+).*?code=(0x[0-9a-fA-F]+)", output)
            if not match:
                raise gdb.GdbError(
                    "Could not parse frame-id for frame #%d" % frame.level()
                )

            # Create the FrameId object.
            sp_addr = int(match.group(1), 16)
            pc_addr = int(match.group(2), 16)
            self._frame_ids.append(FrameId(sp_addr, pc_addr))

            frame = frame.older()

    # Called when an inferior continues.  If this is the inferior for
    # which this unwinder collected the stack, then discard the
    # collected FrameId objects and disable the unwinder.
    def _continued_event_handler(self, event):
        if gdb.selected_inferior() == self._inferior:
            self._enabled = False
            self._frame_ids = []
            self._inferior = None

    # To aid debugging, print the captured FrameId instances.
    def print_frame_ids(self):
        for level, fid in enumerate(self._frame_ids):
            print(
                "frame-id for frame #%s: {stack=0x%x,code=0x%x}"
                % (level, fid.sp, fid.pc)
            )

    def __call__(self, pending_frame):
        if self.stop_at_level is None or pending_frame.level() != self.stop_at_level:
            return None

        if gdb.selected_inferior() != self._inferior:
            return None

        if len(self._frame_ids) <= self.stop_at_level:
            raise gdb.GdbError("not enough parsed frame-ids")

        # Set the frame-id for this frame to its actual, expected
        # frame-id, which we captured in the FRAME_IDS list.
        unwinder = pending_frame.create_unwind_info(self._frame_ids[self.stop_at_level])

        # Provide the register values for the caller frame, that is,
        # the frame at 'STOP_AT_LEVEL + 1'.
        #
        # We forward all of the register values unchanged from this
        # frame.
        #
        # What this means is that, as far as GDB is concerned, the
        # caller frame will appear to be identical to this frame.  Of
        # particular importance, we send $pc and $sp unchanged to the
        # caller frame.
        #
        # Because the caller frame has the same $pc and $sp as this
        # frame, GDB will compute the same frame-id for the caller
        # frame as we just supplied for this frame (above).  This
        # creates the artificial frame cycle which is the whole point
        # of this test.
        #
        # NOTE: Forwarding all registers unchanged like this to the
        # caller frame is not how you'd normally write a frame
        # unwinder.  Some registers might indeed be unmodified between
        # frames, but we'd usually expect the $sp and/or the $pc to
        # change.  This test is deliberately doing something weird in
        # order to force a cycle, and so test GDB.
        for reg in pending_frame.architecture().registers("general"):
            val = pending_frame.read_register(reg)
            # Having unavailable registers leads to a fall back to the standard
            # unwinders.  Don't add unavailable registers to avoid this.
            if str(val) == "<unavailable>":
                continue
            unwinder.add_saved_register(reg, val)
        return unwinder

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, val):
        if val:
            self._enabled = False
            self._update_frame_id_cache()

        self._enabled = val

    @property
    def stop_at_level(self):
        return self._stop_at_level

    @stop_at_level.setter
    def stop_at_level(self, val):
        self._stop_at_level = val
        gdb.invalidate_cached_frames()


global_unwinder = corrupt_stack_unwinder()
gdb.unwinder.register_unwinder(None, global_unwinder, True)
