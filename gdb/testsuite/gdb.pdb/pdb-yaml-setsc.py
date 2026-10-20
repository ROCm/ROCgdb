#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
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

"""Set a module's section contribution in a yaml2pdb-generated PDB.

llvm-pdbutil's yaml2pdb writes a zero module section contribution.  GDB's PDB
reader gives a module whose section contribution names an invalid section
(gdb/pdb/pdb.c: section_valid (module->sc_section)) no address range, and
drops the procedures such a module declares.  Setting the contribution's
1-based section index to a valid section of the companion PE (its .text is
section 1) lets them be read.  The offset and size describe the contribution
within that section; address resolution uses the PE's section table.  This
patch changes those three fields in the module's DBI module-info record
without growing any stream or creating a separate section-contribution
substream.

The "hide" form instead sets the DEBUG_S_IGNORE bit on a module's C13
subsections of KIND, so a reader skips them as if they were absent.  It
gives a PDB without one kind of subsection, which yaml2pdb cannot write
when other subsections refer to it.

Usage: pdb-yaml-setsc.py PDB ISECT OFFSET SIZE [MODULE]
       pdb-yaml-setsc.py PDB hide KIND [MODULE]
"""

import struct
import sys

DBI_STREAM = 3
DBI_HDR_SIZE = 64
MODI_HDR_SIZE = 64
MODI_SC_ISECT_OFFS = 4
MODI_SC_OFFSET_OFFS = 8
MODI_SC_SIZE_OFFS = 12
MODI_STREAM_OFFS = 34
MODI_SYM_BYTES_OFFS = 36
MODI_C11_BYTES_OFFS = 40
MODI_C13_BYTES_OFFS = 44
DEBUG_S_IGNORE = 0x80000000


def read_u32 (data, off):
    return struct.unpack_from ("<I", data, off)[0]


class Msf:
    def __init__ (self, path):
        self.path = path
        with open (path, "rb") as f:
            self.data = bytearray (f.read ())
        d = self.data
        self.block_size = read_u32 (d, 32)
        num_dir_bytes = read_u32 (d, 44)
        block_map_addr = read_u32 (d, 52)
        num_dir_blocks = ((num_dir_bytes + self.block_size - 1)
                          // self.block_size)
        dir_list_off = block_map_addr * self.block_size
        dir_blocks = [read_u32 (d, dir_list_off + 4 * i)
                      for i in range (num_dir_blocks)]
        directory = self._read_blocks (dir_blocks)[:num_dir_bytes]
        self.streams = self._parse_directory (directory)

    def _read_blocks (self, blocks):
        out = bytearray ()
        for b in blocks:
            out += self.data[b * self.block_size:(b + 1) * self.block_size]
        return out

    def _parse_directory (self, directory):
        num_streams = read_u32 (directory, 0)
        sizes = [read_u32 (directory, 4 + 4 * i)
                 for i in range (num_streams)]
        streams = []
        pos = 4 + 4 * num_streams
        for size in sizes:
            if size == 0xFFFFFFFF:
                streams.append ((0, []))
                continue
            nblocks = (size + self.block_size - 1) // self.block_size
            blocks = [read_u32 (directory, pos + 4 * i)
                      for i in range (nblocks)]
            pos += 4 * nblocks
            streams.append ((size, blocks))
        return streams

    def _file_off (self, stream_idx, logical_off):
        _, blocks = self.streams[stream_idx]
        return (blocks[logical_off // self.block_size] * self.block_size
                + logical_off % self.block_size)

    def _modi_offset (self, module):
        """Offset in the DBI stream of MODULE's module-info record.  Each
        record is a fixed header and two NUL-terminated names, padded to
        4 bytes."""
        size, blocks = self.streams[DBI_STREAM]
        dbi = self._read_blocks (blocks)[:size]
        off = DBI_HDR_SIZE
        for _ in range (module):
            end = dbi.index (0, off + MODI_HDR_SIZE)
            end = dbi.index (0, end + 1)
            off = (end + 1 + 3) & ~3
        return off

    def set_modi_sc (self, isect, offset, size, module=0):
        base = self._modi_offset (module)
        struct.pack_into ("<h", self.data,
                          self._file_off (DBI_STREAM,
                                          base + MODI_SC_ISECT_OFFS), isect)
        struct.pack_into ("<I", self.data,
                          self._file_off (DBI_STREAM,
                                          base + MODI_SC_OFFSET_OFFS), offset)
        struct.pack_into ("<I", self.data,
                          self._file_off (DBI_STREAM,
                                          base + MODI_SC_SIZE_OFFS), size)
        with open (self.path, "wb") as f:
            f.write (self.data)

    def hide_c13 (self, kind, module=0):
        """Set DEBUG_S_IGNORE on MODULE's C13 subsections of KIND.  The
        C13 subsections follow the module's symbols and C11 lines; each is
        a kind, a length and data padded to 4 bytes."""
        size, blocks = self.streams[DBI_STREAM]
        dbi = self._read_blocks (blocks)[:size]
        base = self._modi_offset (module)
        stream = struct.unpack_from ("<H", dbi, base + MODI_STREAM_OFFS)[0]
        off = (read_u32 (dbi, base + MODI_SYM_BYTES_OFFS)
               + read_u32 (dbi, base + MODI_C11_BYTES_OFFS))
        end = off + read_u32 (dbi, base + MODI_C13_BYTES_OFFS)
        msize, mblocks = self.streams[stream]
        data = self._read_blocks (mblocks)[:msize]
        while off + 8 <= end:
            sub_kind, length = struct.unpack_from ("<II", data, off)
            if sub_kind == kind:
                struct.pack_into ("<I", self.data,
                                  self._file_off (stream, off),
                                  kind | DEBUG_S_IGNORE)
            off += 8 + ((length + 3) & ~3)
        with open (self.path, "wb") as f:
            f.write (self.data)


def main (argv):
    if len (argv) in (4, 5) and argv[2] == "hide":
        module = int (argv[4]) if len (argv) == 5 else 0
        Msf (argv[1]).hide_c13 (int (argv[3], 0), module)
        return 0
    if len (argv) not in (5, 6):
        print ("usage: pdb-yaml-setsc.py PDB ISECT OFFSET SIZE [MODULE]\n"
               "       pdb-yaml-setsc.py PDB hide KIND [MODULE]",
               file=sys.stderr)
        return 2
    module = int (argv[5]) if len (argv) == 6 else 0
    Msf (argv[1]).set_modi_sc (int (argv[2]), int (argv[3]), int (argv[4]),
                               module)
    return 0


if __name__ == "__main__":
    sys.exit (main (sys.argv))
