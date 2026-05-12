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

"""Set module 0's section contribution in a yaml2pdb-generated PDB.

llvm-pdbutil's yaml2pdb writes a zero module section contribution.  GDB's PDB
reader skips any module whose section contribution names an invalid section
(gdb/pdb/pdb.c: section_valid (module->sc_section)), so a module's symbols are
never built.  Setting the contribution's 1-based section index to a valid
section of the companion PE (its .text is section 1) lets the module's symbols
be read.  The offset and size describe the contribution within that section;
address resolution uses the PE's section table.  This patch changes those
three fields in module 0's DBI module-info record without growing any stream
or creating a separate section-contribution substream.

Usage: pdb-yaml-setsc.py PDB ISECT OFFSET SIZE
"""

import struct
import sys

DBI_STREAM = 3
DBI_HDR_SIZE = 64
MODI_SC_ISECT_OFFS = 4
MODI_SC_OFFSET_OFFS = 8
MODI_SC_SIZE_OFFS = 12


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

    def set_modi_sc (self, isect, offset, size):
        base = DBI_HDR_SIZE
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


def main (argv):
    if len (argv) != 5:
        print ("usage: pdb-yaml-setsc.py PDB ISECT OFFSET SIZE",
               file=sys.stderr)
        return 2
    Msf (argv[1]).set_modi_sc (int (argv[2]), int (argv[3]), int (argv[4]))
    return 0


if __name__ == "__main__":
    sys.exit (main (sys.argv))
