#!/usr/bin/env python3
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of GDB.
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

"""Add global symbol records and a GSI hash table to a YAML-built PDB.

llvm-pdbutil's yaml2pdb has no key for the globals stream: yamlToPdb feeds
the GSI builder only through addPublicSymbols(), which deserializes every
record as PublicSym32.  So a YAML fixture gets an empty GSI and cannot cover
`maint info pdb-gsi`, file-scope `S_UDT` aliases, or the cooked index's
GSI-seeded path.  This fills both streams in place afterwards.

The encoding follows create_globals_stream() in ld/pdb.c, which is this
tree's own PDB writer:

  header    16 bytes: signature, version, cbHr, cbBuckets
  records    8 bytes each, ascending bucket: 1-based stream offset, refcount
  bitmap   512 bytes, bucket b at byte b/8 bit b%8, then a 4-byte zero gap
  offsets    4 bytes per filled bucket, ascending: record index * 0xc

The 0xc is the size of Microsoft's in-memory hash_record, not the 8-byte
on-disk one; using 8 here produces a file that parses but misindexes.

Both streams are patched within their existing blocks, so the block map and
the free page map never change -- only the stream's size field in the
directory.  Run with --check to confirm the records still fit.
"""

import importlib.util
import os
import struct
import sys

S_CONSTANT = 0x1107
S_UDT = 0x1108
S_LDATA32 = 0x110C
S_GDATA32 = 0x110D
S_PROCREF = 0x1125
S_LPROCREF = 0x1127

DBI_STREAM = 3
DBI_HDR_GSI_STREAM_OFFS = 12
DBI_HDR_SYM_RECORD_STREAM_OFFS = 20

NUM_BUCKETS = 4096
GSI_SIGNATURE = 0xFFFFFFFF
GSI_VERSION_70 = 0xEFFE0000 + 19990810

# Size of Microsoft's in-memory hash record; see ld/pdb.c.
MS_HASH_RECORD_SIZE = 0xC


def load_msf_module():
    """Reuse the MSF reader that ships with pdb-yaml-setsc.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "pdb-yaml-setsc.py")
    spec = importlib.util.spec_from_file_location("pdb_yaml_setsc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def calc_hash(name):
    """calc_hash() from ld/pdb.c, over the name without its terminator."""
    data = name.encode("ascii")
    h = 0
    i = 0
    n = len(data)
    while n - i >= 4:
        h ^= data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        i += 4
    if n - i >= 2:
        h ^= data[i] | (data[i + 1] << 8)
        i += 2
    if n - i != 0:
        h ^= data[i]
    h &= 0xFFFFFFFF
    h |= 0x20202020
    h ^= h >> 11
    h ^= h >> 16
    return h & 0xFFFFFFFF


def _rec(kind, payload, name):
    """Wrap a payload as a CodeView record, 4-byte aligned."""
    body = payload + name.encode("ascii") + b"\0"
    pad = (-(len(body) + 4)) % 4
    body += b"\0" * pad
    return struct.pack("<HH", len(body) + 2, kind) + body


def make_data(kind, typind, seg, off, name):
    return _rec(kind, struct.pack("<IIH", typind, off, seg), name)


def make_constant(typind, value, name):
    if value < 0x8000:
        leaf = struct.pack("<H", value)
    else:
        leaf = struct.pack("<HI", 0x8003, value)
    return _rec(S_CONSTANT, struct.pack("<I", typind) + leaf, name)


def make_udt(typind, name):
    return _rec(S_UDT, struct.pack("<I", typind), name)


def make_procref(kind, sumname, ibsym, imod, name):
    return _rec(kind, struct.pack("<IIH", sumname, ibsym, imod), name)


def parse_spec(path):
    """One record per line: KIND key=value ...  '#' starts a comment."""
    out = []
    with open(path, "r") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            kind = parts[0]
            kw = {}
            for p in parts[1:]:
                k, _, v = p.partition("=")
                kw[k] = v
            try:
                name = kw["name"]
                if kind in ("S_GDATA32", "S_LDATA32"):
                    code = S_GDATA32 if kind == "S_GDATA32" else S_LDATA32
                    out.append((name, make_data(code, int(kw["ti"], 0),
                                                int(kw.get("seg", "1"), 0),
                                                int(kw.get("off", "0"), 0),
                                                name)))
                elif kind == "S_CONSTANT":
                    out.append((name, make_constant(int(kw["ti"], 0),
                                                    int(kw["val"], 0), name)))
                elif kind == "S_UDT":
                    out.append((name, make_udt(int(kw["ti"], 0), name)))
                elif kind in ("S_PROCREF", "S_LPROCREF"):
                    code = S_PROCREF if kind == "S_PROCREF" else S_LPROCREF
                    out.append((name, make_procref(code,
                                                   int(kw.get("sum", "0"), 0),
                                                   int(kw["ibsym"], 0),
                                                   int(kw["imod"], 0), name)))
                else:
                    raise KeyError(kind)
            except KeyError as e:
                sys.exit("%s:%d: bad record (%s): %s"
                         % (path, lineno, e, line))
    return out


def build_gsi(entries):
    """entries: list of (name, one-based offset into the symbol stream)."""
    buckets = {}
    for name, off in entries:
        buckets.setdefault(calc_hash(name) % NUM_BUCKETS, []).append(off)

    # Records go out in ascending bucket order; a bucket's stored value is
    # the index of its first record in that flat array.
    records = bytearray()
    first_index = {}
    index = 0
    for b in sorted(buckets):
        first_index[b] = index
        for off in buckets[b]:
            records += struct.pack("<II", off, 1)
            index += 1

    bitmap = bytearray(NUM_BUCKETS // 8)
    for b in buckets:
        bitmap[b // 8] |= 1 << (b % 8)

    offsets = bytearray()
    for b in sorted(buckets):
        offsets += struct.pack("<I", first_index[b] * MS_HASH_RECORD_SIZE)

    # The 4-byte gap after the bitmap is counted in cbBuckets; ld/pdb.c
    # computes 4096/8 + 4 + filled*4.
    bucket_data = bytes(bitmap) + struct.pack("<I", 0) + bytes(offsets)
    header = struct.pack("<IIII", GSI_SIGNATURE, GSI_VERSION_70,
                         len(records), len(bucket_data))
    return header + bytes(records) + bucket_data


class Patcher:
    def __init__(self, path):
        msf_mod = load_msf_module()
        self.msf = msf_mod.Msf(path)
        self.path = path
        d = self.msf.data
        self.block_size = self.msf.block_size
        self.num_dir_bytes = msf_mod.read_u32(d, 44)
        block_map_addr = msf_mod.read_u32(d, 52)
        n = (self.num_dir_bytes + self.block_size - 1) // self.block_size
        off = block_map_addr * self.block_size
        self.dir_blocks = [msf_mod.read_u32(d, off + 4 * i) for i in range(n)]

    def stream_bytes(self, idx):
        size, blocks = self.msf.streams[idx]
        return bytes(self.msf._read_blocks(blocks)[:size])

    def capacity(self, idx):
        _, blocks = self.msf.streams[idx]
        return len(blocks) * self.block_size

    def write_stream(self, idx, payload):
        size, blocks = self.msf.streams[idx]
        if len(payload) > len(blocks) * self.block_size:
            sys.exit("stream %d needs %d bytes but has %d allocated; "
                     "growing the block map is not implemented"
                     % (idx, len(payload), len(blocks) * self.block_size))
        for i, b in enumerate(blocks):
            chunk = payload[i * self.block_size:(i + 1) * self.block_size]
            if not chunk:
                break
            start = b * self.block_size
            self.msf.data[start:start + len(chunk)] = chunk
        self.msf.streams[idx] = (len(payload), blocks)
        self._set_dir_size(idx, len(payload))

    def _set_dir_size(self, idx, size):
        """Rewrite the stream's size field in the MSF directory."""
        directory = bytearray(self.msf._read_blocks(self.dir_blocks))
        struct.pack_into("<I", directory, 4 + 4 * idx, size)
        for i, b in enumerate(self.dir_blocks):
            start = b * self.block_size
            chunk = directory[i * self.block_size:(i + 1) * self.block_size]
            self.msf.data[start:start + len(chunk)] = chunk

    def dbi_u16(self, offs):
        base = self.msf._file_off(DBI_STREAM, offs)
        return struct.unpack_from("<H", self.msf.data, base)[0]

    def save(self):
        with open(self.path, "wb") as f:
            f.write(self.msf.data)


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: pdb-yaml-addglobals.py PDB SPEC [--check]")
    pdb_path, spec_path = argv[1], argv[2]
    check_only = "--check" in argv[3:]

    recs = parse_spec(spec_path)
    p = Patcher(pdb_path)

    sym_idx = p.dbi_u16(DBI_HDR_SYM_RECORD_STREAM_OFFS)
    gsi_idx = p.dbi_u16(DBI_HDR_GSI_STREAM_OFFS)

    existing = p.stream_bytes(sym_idx)
    if len(existing) % 4:
        sys.exit("symbol records stream is not 4-byte aligned")

    blob = bytearray(existing)
    entries = []
    for name, rec in recs:
        entries.append((name, len(blob) + 1))   # GSI offsets are 1-based
        blob += rec

    gsi = build_gsi(entries)

    if check_only:
        print("symbol records: %d -> %d bytes (capacity %d)"
              % (len(existing), len(blob), p.capacity(sym_idx)))
        print("gsi stream %d: %d -> %d bytes (capacity %d)"
              % (gsi_idx, len(p.stream_bytes(gsi_idx)), len(gsi),
                 p.capacity(gsi_idx)))
        return 0

    p.write_stream(sym_idx, bytes(blob))
    p.write_stream(gsi_idx, gsi)
    p.save()
    print("added %d global records to %s" % (len(recs), pdb_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
