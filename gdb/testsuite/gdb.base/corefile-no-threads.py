# Copyright 2025-2026 Free Software Foundation, Inc.
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

# Add a new GDB command which modifies a core file on disk prior to
# GDB loading it.  Usage:
#
# modify-core-file <corefile> <note type> [ <name regex> ]
#
# Find all notes in the core file <corefile> that have the type value
# <note type>.  Change the type of those notes to 0xffffffff, which
# should prevent GDB from processing the note.  The <corefile> is
# modified in place.
#
# The optional argument <name regex> is a regular expression to match
# against the name of the note in addition to the type value check.
# Only notes whose name matches <name regex> will be changed.
#
# If <name regex> is missing, or is the empty string, then any note
# with the required type will be modified.

import os
import re
import struct

import gdb

# Constants taken from the ELF spec.
ELF_CLASS_32 = 0x1
ELF_CLASS_64 = 0x2
ELF_DATA_2_LSB = 0x1
ELF_DATA_2_MSB = 0x2
ELF_VERSION_CURRENT = 0x1
ELF_TYPE_CORE = 0x4

# Program Header Type for notes.
PT_NOTE = 0x4


# Helper function to read a single value from DATA at byte OFFSET.
# The FMT specifies the format of the data (see struct.unpack).  A
# single value is extracted and returned.
def read_field(data, offset, fmt):
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError(
            "Read operation at 0x{:x}, length {}, exceeds file boundaries.".format(
                offset, size
            )
        )
    return struct.unpack(fmt, data[offset : offset + size])[0]


# Helper function to read the name of a note from DATA.  The name
# string starts at OFFSET and is SIZE bytes long.  The last byte of
# the name should be the NULL character.
#
# Return the name (excluding the NULL character).  Raises a ValueError
# if anything goes wrong.
def read_note_name(data, offset, size):
    if offset + size > len(data):
        raise ValueError(
            "Read operation at {:x}, length {}, exceeds file boundaries.".format(
                offset, size
            )
        )
    name = ""
    for i in range(size):
        b = struct.unpack("=b", data[offset + i : offset + i + 1])[0]
        if i == size - 1:
            if b != 0:
                raise ValueError(
                    "Last byte of name for note at 0x{:x} is not NULL".format(offset)
                )
        else:
            name = name + chr(b)
    return name


# Open core file CORE_FILEPATH and find all the PT_NOTE segments.
# Within these segments find the notes with type value TYPE_VAL and
# change the type of these notes to 0xffffffff, which should mean that
# BFD/GDB doesn't process them.
#
# If NAME_RE is not None, then it is a regexp that must match against
# the note's name in order for the note to be modified, this regexp
# check is in addition to the type check.  If NAME_RE is None, then
# the note's name is ignored, only TYPE_VAL is checked.
def invalidate_corefile_notes(core_filepath, type_val, name_re):
    # Value used as the replacement note type.
    CORRUPT_TYPE = 0xFFFFFFFF

    # Open the core file and read it into one huge data block.  This
    # will be fine so long as the core file isn't too large.
    try:
        with open(core_filepath, "r+b") as f:
            core_data = bytearray(f.read())
    except FileNotFoundError:
        raise gdb.GdbError("Error: File not found at '%s'" % (core_filepath))
    except Exception as e:
        raise gdb.GdbError("Error reading file: %s" % (str(e)))

    file_size = len(core_data)

    # Confirm that this is an ELF.
    elf_magic = [0x7F, ord("E"), ord("L"), ord("F")]
    for idx, magic_value in enumerate(elf_magic):
        v = read_field(core_data, idx, "=b")
        if v != magic_value:
            raise gdb.GdbError(
                "Unexpected magic ELF header value at offset %d: %d vs %d"
                % (idx, magic_value, v)
            )

    # Check that the ELF is of a format that we can understand.  The
    # restrictions encoded here could be relaxed by making this script
    # smarter.
    ei_class = read_field(core_data, 4, "=b")
    ei_data = read_field(core_data, 5, "=b")
    ei_version = read_field(core_data, 6, "=b")

    # Based on the endinanness of the core file, select a character to
    # use with the 'struct' module for unpacking multi-byte fields.
    if ei_data == ELF_DATA_2_LSB:
        endian_char = "<"
    elif ei_data == ELF_DATA_2_MSB:
        endian_char = ">"
    else:
        raise gdb.GdbError("Unsupported ELF data %d" % (ei_data))

    # Based on the ELF class, setup some constants.  We define some
    # offsets into the ELF header, and into the program header
    # structure.  We setup some format strings used with the 'struct'
    # module for unpacking bytes from the ELF.  And we define the size
    # of the program header structure.  All of these constants are
    # based on the ELF specification, and were created using the
    # structures found in include/elf/external.h.
    #
    # Note: ELF_NOTE_WORD_FMT is the same for 32 and 64 bit ELFs as
    # every part of the note header is a 4-byte word.
    if ei_class == ELF_CLASS_32:
        # Structure offsets.
        ELF_HDR_TYPE_OFFSET = 0x10
        ELF_HDR_PHOFF_OFFSET = 0x1C
        ELF_HDR_PHNUM_OFFSET = 0x2C
        PHDR_OFFSET_OF_OFFSET_FIELD = 0x04
        PHDR_OFFSET_OF_FILESZ_FIELD = 0x10

        # Format specifiers for unpacking fields.
        ELF_HALF_FMT = endian_char + "H"
        ELF_WORD_FMT = endian_char + "I"
        ELF_OFF_FMT = endian_char + "I"
        ELF_ADDR_FMT = ELF_OFF_FMT

        ELF_NOTE_WORD_FMT = endian_char + "I"

        # Program Header size assuming 32-bit ELF.
        PHDR_SIZE = 32
    elif ei_class == ELF_CLASS_64:
        # Structure offsets.
        ELF_HDR_TYPE_OFFSET = 0x10
        ELF_HDR_PHOFF_OFFSET = 0x20
        ELF_HDR_PHNUM_OFFSET = 0x38
        PHDR_OFFSET_OF_OFFSET_FIELD = 0x08
        PHDR_OFFSET_OF_FILESZ_FIELD = 0x20

        # Format specifiers for unpacking fields.
        ELF_HALF_FMT = endian_char + "H"
        ELF_WORD_FMT = endian_char + "I"
        ELF_OFF_FMT = endian_char + "Q"
        ELF_ADDR_FMT = ELF_OFF_FMT

        ELF_NOTE_WORD_FMT = endian_char + "I"

        # Program Header size assuming 64-bit ELF.
        PHDR_SIZE = 56
    else:
        raise gdb.GdbError("Unsupported ELF class %d" % (ei_class))

    if ei_version != ELF_VERSION_CURRENT:
        raise gdb.GdbError("Unsupported ELF version %d" % (ei_version))

    # Read the e_type field and check this is a core file.
    e_type = read_field(core_data, ELF_HDR_TYPE_OFFSET, ELF_HALF_FMT)
    if e_type != ELF_TYPE_CORE:
        raise gdb.GdbError("Unsuported ELF e_type %d" % (e_type))

    # Read the offset to the program header table, and the number of
    # entries in the program header table.
    e_phoff = read_field(core_data, ELF_HDR_PHOFF_OFFSET, ELF_OFF_FMT)
    e_phnum = read_field(core_data, ELF_HDR_PHNUM_OFFSET, ELF_HALF_FMT)

    # Iterate through the program header table looking for and
    # segments with type NOTE.  Record the offset and size of those
    # NOTE segments.
    note_segments = []

    for i in range(e_phnum):
        phdr_offset = e_phoff + i * PHDR_SIZE

        # First word of the program header entry is the type.
        p_type = read_field(core_data, phdr_offset, ELF_WORD_FMT)

        if p_type == PT_NOTE:
            # Read the file offset and file size.  The offsets used
            # here are valid for 64-bit ELF only.
            p_offset = read_field(
                core_data, phdr_offset + PHDR_OFFSET_OF_OFFSET_FIELD, ELF_ADDR_FMT
            )
            p_filesz = read_field(
                core_data, phdr_offset + PHDR_OFFSET_OF_FILESZ_FIELD, ELF_ADDR_FMT
            )
            note_segments.append((p_offset, p_filesz))

    # Maybe there were no NOTE segments?
    if len(note_segments) == 0:
        gdb.warning("No PT_NOTE segments founds")
        return

    # Iterate through the notes within the PT_NOTE segment.
    for note_start_offset, note_segment_size in note_segments:
        print(
            "Located PT_NOTE segment: Offset 0x%x, Size 0x%x"
            % (note_start_offset, note_segment_size)
        )

        current_offset = note_start_offset
        end_offset = note_start_offset + note_segment_size
        if end_offset > file_size:
            end_offset = file_size
        count = 0

        # This is the size of the first 3 fields within a note; the
        # namesz, descsz, and type.  These fields are all 4 bytes,
        # even for a 64-bit ELF.
        NOTE_HEADER_SIZE = 12

        while current_offset <= end_offset - NOTE_HEADER_SIZE:
            # Read namesz, descsz, and type.  These fields are all
            # 32-bit, even for 64-bit ELF format, as a result, the
            # offsets are all hard-coded here.
            namesz = read_field(core_data, current_offset, ELF_NOTE_WORD_FMT)
            descsz = read_field(core_data, current_offset + 4, ELF_NOTE_WORD_FMT)
            note_type = read_field(core_data, current_offset + 8, ELF_NOTE_WORD_FMT)

            is_matching_note = note_type == type_val

            if is_matching_note and name_re is not None:
                name_str = read_note_name(core_data, current_offset + 12, namesz)
                if not re.match(name_re, name_str):
                    is_matching_note = False

            # Check if this note type is of the required type.
            if is_matching_note:
                print(
                    "  Found note with type 0x%x at file offset 0x%x."
                    % (type_val, current_offset)
                )

                # Overwrite the 4 bytes starting at the type field offset (current_offset + 8)
                type_offset = current_offset + 8
                corrupted_bytes = struct.pack(ELF_NOTE_WORD_FMT, CORRUPT_TYPE)
                core_data[type_offset : type_offset + 4] = corrupted_bytes
                count += 1

            # Move to the next note record, accounting for 4-byte alignment.

            # Advance past header, this is namesz, descsz, and type.
            next_offset = current_offset + NOTE_HEADER_SIZE

            # Advance past name field, aligned up to the next 4-byte boundary.
            next_offset += (namesz + 3) & ~0x3

            # Advance past descriptor field, aligned up to the next 4-byte boundary.
            next_offset += (descsz + 3) & ~0x3

            # We've now found the next note entry.  Or we're at the
            # end of the segment.  The while loop condition should
            # figure this out for us.
            current_offset = next_offset

        if count > 0:
            # Write the modified bytearray back to the file.
            try:
                with open(core_filepath, "wb") as f:
                    f.write(core_data)
                    print(
                        "Successfully updated %d note(s) in '%s'."
                        % (count, core_filepath)
                    )
            except Exception as e:
                raise gdb.GdbError("Error writing to file: %s" % (str(e)))
        else:
            gdb.warning(
                "No notes with type 0x%x found within the segment. File was not modified."
                % (type_val)
            )


class modify_core_file(gdb.Command):
    """Update notes within a core file.

    Usage:
    modify-core-file COREFILE NOTE_TYPE [ NAME_REGEX ]

    Within COREFILE, find any notes matching NOTE_TYPE, which should
    be an integer.  Change the type of these notes to 0xffffffff.

    If NAME_REGEX is supplied, and is not the empty string, then only
    notes whose type matches NOTE_TYPE, and whose name matches
    NAME_REGEX, are modified."""

    def __init__(self):
        gdb.Command.__init__(self, "modify-core-file", gdb.COMMAND_USER)

    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)

        if len(argv) != 2 and len(argv) != 3:
            raise gdb.GdbError(
                "Invalid argument count.  Usage modify-core-file COREFILE TYPE [ REGEX ]"
            )

        filename = argv[0]
        type_str = argv[1]
        if len(argv) == 3:
            name_re_str = argv[2]
        else:
            name_re_str = ""

        if not os.path.isfile(filename):
            raise gdb.GdbError("Error: File '%s' doesn't exist." % (filename))

        if not os.access(filename, os.R_OK):
            raise gdb.GdbError(
                "Error: Cannot read '%s'. Check file permissions." % (filename)
            )

        if not os.access(filename, os.W_OK):
            raise gdb.GdbError(
                "Error: Cannot write to '%s'. Check file permissions." % (filename)
            )

        try:
            type_val = int(type_str, 0)
        except ValueError as e:
            raise gdb.GdbError(
                "Error: Unable to parse '%s' as number: %s" % (type_str, str(e))
            )

        if name_re_str != "":
            name_re = re.compile(name_re_str)
        else:
            name_re = None

        invalidate_corefile_notes(filename, type_val, name_re)


modify_core_file()
