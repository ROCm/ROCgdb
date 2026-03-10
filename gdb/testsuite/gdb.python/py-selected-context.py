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

import gdb

event_throws_error = False


def gdb_selected_context_handler(event):
    assert isinstance(event, gdb.SelectedContextEvent)

    if event_throws_error:
        raise gdb.GdbError("error from gdb_selected_context_handler")
    else:
        print("event type: selected-context")
        assert isinstance(event.inferior, gdb.Inferior)
        print("  Inferior: %d" % (event.inferior.num))
        if event.thread is None:
            thr = "None"
        else:
            assert isinstance(event.thread, gdb.InferiorThread)
            thr = "%d.%d" % (event.thread.inferior.num, event.thread.num)
        print("    Thread: %s" % (thr))
        if event.frame is None:
            frame = "None"
        else:
            assert isinstance(event.frame, gdb.Frame)
            frame = "#%d" % (event.frame.level())
        print("     Frame: %s" % (frame))


class test_selected_context(gdb.Command):
    """Test GDB's Selected Context Event."""

    def __init__(self):
        gdb.Command.__init__(self, "test-selected-context-event", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        gdb.events.selected_context.connect(gdb_selected_context_handler)
        print("GDB selected-context event registered.")


test_selected_context()

print("DONE")
