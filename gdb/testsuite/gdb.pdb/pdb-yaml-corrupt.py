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

"""Corrupt one record of a yaml2pdb-generated PDB in place.

yaml2pdb derives counts, lengths and terminators from the YAML, so it cannot
express a malformed record.  This overwrites bytes inside an existing record
without changing any stream size.

STREAM selects the records: "tpi" for the TPI stream, "module" for the
symbol records of the first module, "globals" for the global symbol
records, "subsections" for the first module's C13 subsections, or
"filenames" for the DBI File Info names buffer, "names" for the /names
string data or "psgsi" for the whole PSGSI stream, each a single record of
kind 0.
KIND is the record's leaf, symbol or subsection kind and INDEX selects among
the records of that kind in stream order.  OFFSET counts from the record's
data, which follows its RecordLen and RecordKind fields (a subsection's
kind and length).

  set:    write VALUE as a little-endian integer of WIDTH bytes.  VALUE is
          a number, or "self" for a TPI record's own type index.
  fill:   write BYTE from OFFSET to the end of the record, which removes
          any NUL terminator there.
  shrink: cut a symbol record's data to BODY bytes.  The bytes cut off
          become an S_SKIP record, so the records that follow stay in place.

Usage: pdb-yaml-corrupt.py PDB STREAM set KIND INDEX OFFSET WIDTH VALUE
       pdb-yaml-corrupt.py PDB STREAM fill KIND INDEX OFFSET BYTE
       pdb-yaml-corrupt.py PDB STREAM shrink KIND INDEX BODY
"""

import struct
import sys

TPI_STREAM = 2
DBI_STREAM = 3
INFO_STREAM = 1

# The PDB Info stream's named stream map follows Version, Signature, Age
# and GUID.  The /names stream's string data follows its signature,
# version and data size.
INFO_NAMED_STREAM_MAP_OFFS = 28
NAMES_DATA_SIZE_OFFS = 8
NAMES_DATA_OFFS = 12

# DBI header fields and the first module's ModInfo fields.
DBI_PSGSI_STREAM_OFFS = 16
DBI_SYM_RECORD_STREAM_OFFS = 20
DBI_MOD_INFO_SIZE_OFFS = 24
DBI_SECTION_CONTRIBUTION_SIZE_OFFS = 28
DBI_SECTION_MAP_SIZE_OFFS = 32
DBI_SOURCE_INFO_SIZE_OFFS = 36
DBI_HEADER_SIZE = 64
MODI_STREAM_NUM_OFFS = 34
MODI_SYM_BYTES_OFFS = 36
MODI_C11_BYTES_OFFS = 40
MODI_C13_BYTES_OFFS = 44

# A module symbol stream starts with a 4-byte signature.
MODULE_SYMBOLS_OFFS = 4

S_SKIP = 0x0007


def read_u16(data, off):
    return struct.unpack_from("<H", data, off)[0]


def read_u32(data, off):
    return struct.unpack_from("<I", data, off)[0]


class Msf:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = bytearray(f.read())
        d = self.data
        self.block_size = read_u32(d, 32)
        num_dir_bytes = read_u32(d, 44)
        block_map_addr = read_u32(d, 52)
        num_dir_blocks = (num_dir_bytes + self.block_size - 1) // self.block_size
        dir_list_off = block_map_addr * self.block_size
        dir_blocks = [read_u32(d, dir_list_off + 4 * i) for i in range(num_dir_blocks)]
        directory = self._read_blocks(dir_blocks)[:num_dir_bytes]
        self.streams = self._parse_directory(directory)

    def _read_blocks(self, blocks):
        out = bytearray()
        for b in blocks:
            out += self.data[b * self.block_size : (b + 1) * self.block_size]
        return out

    def _parse_directory(self, directory):
        num_streams = read_u32(directory, 0)
        sizes = [read_u32(directory, 4 + 4 * i) for i in range(num_streams)]
        streams = []
        pos = 4 + 4 * num_streams
        for size in sizes:
            if size == 0xFFFFFFFF:
                streams.append((0, []))
                continue
            nblocks = (size + self.block_size - 1) // self.block_size
            blocks = [read_u32(directory, pos + 4 * i) for i in range(nblocks)]
            pos += 4 * nblocks
            streams.append((size, blocks))
        return streams

    def read_stream(self, index):
        size, blocks = self.streams[index]
        return self._read_blocks(blocks)[:size]

    def write(self, index, offset, data):
        size, blocks = self.streams[index]
        if offset < 0 or offset + len(data) > size:
            raise ValueError("write outside the logical stream")
        for i, byte in enumerate(data):
            pos = offset + i
            block = blocks[pos // self.block_size]
            self.data[block * self.block_size + pos % self.block_size] = byte

    def save(self):
        with open(self.path, "wb") as f:
            f.write(self.data)


def walk_records(stream, offset, end, first_ti=None):
    """Yield (TI, KIND, DATA_OFFSET, DATA_LENGTH) for each record in
    STREAM[OFFSET:END].  TI is None unless FIRST_TI is given."""
    ti = first_ti
    while offset + 4 <= end:
        length = read_u16(stream, offset)
        kind = read_u16(stream, offset + 2)
        yield ti, kind, offset + 4, length - 2
        offset += length + 2
        if ti is not None:
            ti += 1


def walk_subsections(stream, offset, end):
    """Yield (None, KIND, DATA_OFFSET, DATA_LENGTH) for each C13
    subsection in STREAM[OFFSET:END].  Subsections are 4-byte aligned."""
    while offset + 8 <= end:
        kind = read_u32(stream, offset)
        length = read_u32(stream, offset + 4)
        yield None, kind, offset + 8, length
        offset += 8 + (length + 3) // 4 * 4


def names_stream(msf):
    """Return the index of the /names stream.  The named stream map is a
    string buffer, then a hash table: size, capacity, the present and
    deleted bit vectors (each a word count and words), and one (string
    offset, stream index) pair per present bucket."""
    info = msf.read_stream(INFO_STREAM)
    strings_size = read_u32(info, INFO_NAMED_STREAM_MAP_OFFS)
    strings = INFO_NAMED_STREAM_MAP_OFFS + 4
    pos = strings + strings_size
    size = read_u32(info, pos)
    pos += 8
    for _ in range(2):
        pos += 4 + 4 * read_u32(info, pos)
    for i in range(size):
        key = read_u32(info, pos + 8 * i)
        end = info.index(0, strings + key)
        if info[strings + key : end] == b"/names":
            return read_u32(info, pos + 8 * i + 4)
    raise ValueError("no /names stream")


def stream_records(msf, name):
    """Return the index of stream NAME and the records it holds."""
    if name == "names":
        index = names_stream(msf)
        stream = msf.read_stream(index)
        size = read_u32(stream, NAMES_DATA_SIZE_OFFS)
        return index, [(None, 0, NAMES_DATA_OFFS, size)]
    if name == "tpi":
        stream = msf.read_stream(TPI_STREAM)
        offset = read_u32(stream, 4)
        first = read_u32(stream, 8)
        end = offset + read_u32(stream, 16)
        return TPI_STREAM, list(walk_records(stream, offset, end, first))

    dbi = msf.read_stream(DBI_STREAM)
    if name == "module":
        index = read_u16(dbi, DBI_HEADER_SIZE + MODI_STREAM_NUM_OFFS)
        end = read_u32(dbi, DBI_HEADER_SIZE + MODI_SYM_BYTES_OFFS)
        stream = msf.read_stream(index)
        return index, list(walk_records(stream, MODULE_SYMBOLS_OFFS, end))
    if name == "subsections":
        index = read_u16(dbi, DBI_HEADER_SIZE + MODI_STREAM_NUM_OFFS)
        offset = read_u32(dbi, DBI_HEADER_SIZE + MODI_SYM_BYTES_OFFS) + read_u32(
            dbi, DBI_HEADER_SIZE + MODI_C11_BYTES_OFFS
        )
        end = offset + read_u32(dbi, DBI_HEADER_SIZE + MODI_C13_BYTES_OFFS)
        stream = msf.read_stream(index)
        return index, list(walk_subsections(stream, offset, end))
    if name == "filenames":
        # File Info: module count, source file count, per-module first
        # file and file count, one name offset per file, then the names.
        offset = (
            DBI_HEADER_SIZE
            + read_u32(dbi, DBI_MOD_INFO_SIZE_OFFS)
            + read_u32(dbi, DBI_SECTION_CONTRIBUTION_SIZE_OFFS)
            + read_u32(dbi, DBI_SECTION_MAP_SIZE_OFFS)
        )
        end = offset + read_u32(dbi, DBI_SOURCE_INFO_SIZE_OFFS)
        modules = read_u16(dbi, offset)
        counts = offset + 4 + 2 * modules
        files = sum(read_u16(dbi, counts + 2 * i) for i in range(modules))
        names = counts + 2 * modules + 4 * files
        return DBI_STREAM, [(None, 0, names, end - names)]
    if name == "globals":
        index = read_u16(dbi, DBI_SYM_RECORD_STREAM_OFFS)
        stream = msf.read_stream(index)
        return index, list(walk_records(stream, 0, len(stream)))
    if name == "psgsi":
        index = read_u16(dbi, DBI_PSGSI_STREAM_OFFS)
        return index, [(None, 0, 0, msf.streams[index][0])]

    raise ValueError("unknown stream %s" % name)


def find_record(records, kind, index):
    matches = [r for r in records if r[1] == kind]
    if index >= len(matches):
        raise ValueError("no record %d of kind 0x%x" % (index, kind))
    return matches[index]


def main(argv):
    counts = {"set": 9, "fill": 8, "shrink": 7}
    if len(argv) < 4 or counts.get(argv[3]) != len(argv):
        print(__doc__, file=sys.stderr)
        return 2

    msf = Msf(argv[1])
    stream, records = stream_records(msf, argv[2])
    op = argv[3]
    ti, _, data, length = find_record(records, int(argv[4], 0), int(argv[5], 0))
    if op == "set":
        offset = int(argv[6], 0)
        width = int(argv[7], 0)
        value = ti if argv[8] == "self" else int(argv[8], 0)
        msf.write(stream, data + offset, value.to_bytes(width, "little"))
    elif op == "fill":
        offset = int(argv[6], 0)
        msf.write(stream, data + offset, bytes([int(argv[7], 0)]) * (length - offset))
    else:
        body = int(argv[6], 0)
        freed = length - body
        if argv[2] not in ("module", "globals") or freed < 4:
            raise ValueError("cannot shrink this record to %d bytes" % body)
        msf.write(stream, data - 4, struct.pack("<H", body + 2))
        msf.write(stream, data + body, struct.pack("<HH", freed - 2, S_SKIP))
    msf.save()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
