/* Self tests for tracepoint-related code for GDB, the GNU debugger.

   Copyright (C) 2018-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include "gdbsupport/selftest.h"
#include "tracepoint.h"

#include <string>
#include <vector>

namespace selftests {
namespace tracepoint_tests {

/* Run PARSE and assert that it throws a gdb_exception_error.  CLEANUP
   always runs afterward.  */

template<typename Parse, typename Cleanup>
static void
expect_bad_definition (Parse parse, Cleanup cleanup)
{
  bool threw = false;

  try
    {
      parse ();
    }
  catch (const gdb_exception_error &ex)
    {
      SELF_CHECK (ex.reason == RETURN_ERROR);
      threw = true;
    }

  SELF_CHECK (threw);
  cleanup ();
}

/* Like the two-argument overload, with a no-op cleanup.  */

template<typename Parse>
static void
expect_bad_definition (Parse parse)
{
  expect_bad_definition (parse, [] () {});
}

static void
test_parse_static_tracepoint_marker_definition ()
{
  static_tracepoint_marker marker;
  const char def[] = ("1234:6d61726b657231:6578747261207374756666,"
		      "abba:6d61726b657232:,"
		      "cafe:6d61726b657233:6d6f72657374756666");
  const char *start = def;
  const char *end;

  parse_static_tracepoint_marker_definition (start, &end, &marker);

  SELF_CHECK (marker.address == 0x1234);
  SELF_CHECK (marker.str_id == "marker1");
  SELF_CHECK (marker.extra == "extra stuff");
  SELF_CHECK (end == strchr (start, ','));

  start = end + 1;
  parse_static_tracepoint_marker_definition (start, &end, &marker);

  SELF_CHECK (marker.address == 0xabba);
  SELF_CHECK (marker.str_id == "marker2");
  SELF_CHECK (marker.extra == "");
  SELF_CHECK (end == strchr (start, ','));

  start = end + 1;
  parse_static_tracepoint_marker_definition (start, &end, &marker);

  SELF_CHECK (marker.address == 0xcafe);
  SELF_CHECK (marker.str_id == "marker3");
  SELF_CHECK (marker.extra == "morestuff");
  SELF_CHECK (end == def + strlen (def));
}

/* Build a buffer that mimics a reused RSP packet buffer (rs->buf):
   PREFIX is a NUL-terminated definition, and STALE_HEX is appended
   immediately after that NUL -- as leftover bytes from a previous, longer
   reply (for example qTStatus) that a subsequent short reply overwrote.

   Returns a vector with the following layout:

     [ prefix bytes | NUL | stale_hex bytes | NUL ]

   The final NUL keeps strlen on the stale region well-defined.  */

static std::vector<char>
make_stale_tail_buffer (const char *prefix, const std::string &stale_hex)
{
  std::vector<char> buf (prefix, prefix + strlen (prefix) + 1);
  buf.insert (buf.end (), stale_hex.begin (), stale_hex.end ());
  buf.push_back ('\0');
  return buf;
}

/* Assert that LINE is rejected as a malformed TSV definition.  */

static void
expect_bad_tsv_definition (const char *line)
{
  uploaded_tsv *utsvs = nullptr;

  expect_bad_definition ([&] () { parse_tsv_definition (line, &utsvs); },
			 [&] () { free_uploaded_tsvs (&utsvs); });
}

/* Test parse_tsv_definition, including buffers with stale bytes after a
   short definition's terminating NUL, as when reusing an RSP packet
   buffer.  */

static void
test_parse_tsv_definition ()
{
  uploaded_tsv *utsvs = nullptr;

  /* Well-formed definition: name hex-encodes "foo".  */
  {
    const char *good = "1:0:0:666f6f";

    parse_tsv_definition (good, &utsvs);
    SELF_CHECK (utsvs != nullptr);
    SELF_CHECK (utsvs->number == 1);
    SELF_CHECK (utsvs->initial_value == 0);
    SELF_CHECK (utsvs->builtin == 0);
    SELF_CHECK (utsvs->name != nullptr && streq (utsvs->name, "foo"));
    free_uploaded_tsvs (&utsvs);
  }

  /* Short stale tail.  Too small to overflow the old alloca buffer, but
     still missing a trailing colon, so the parser must throw.  */
  {
    static const char prefix[] = "1:0:0000";
    std::vector<char> buf = make_stale_tail_buffer (prefix, "4444");
    expect_bad_tsv_definition (buf.data ());
  }

  /* Long stale tail.  Large enough to overflow the old alloca buffer; with
     the fix, the missing trailing colon must cause a throw instead.  */
  {
    static const char prefix[] = "1:0:0000";
    std::vector<char> buf
      = make_stale_tail_buffer (prefix, std::string (64, '4'));
    expect_bad_tsv_definition (buf.data ());
  }
}

/* Assert that LINE is rejected as a malformed tracepoint definition.  */

static void
expect_bad_tracepoint_definition (const char *line)
{
  uploaded_tp *utps = nullptr;

  expect_bad_definition ([&] () { parse_tracepoint_definition (line, &utps); },
			 [&] () { free_uploaded_tps (&utps); });
}

/* Test parse_tracepoint_definition, including missing-colon rejects and
   enable-flag rejects.  */

static void
test_parse_tracepoint_definition ()
{
  uploaded_tp *utps = nullptr;

  /* Well-formed T definition: T<num>:<addr>:<E|D>:<step>:<pass>.  */
  {
    const char *good = "T1:0:E:0:0";

    parse_tracepoint_definition (good, &utps);
    SELF_CHECK (utps != nullptr);
    SELF_CHECK (utps->number == 1);
    SELF_CHECK (utps->addr == 0);
    SELF_CHECK (utps->enabled);
    SELF_CHECK (utps->step == 0);
    SELF_CHECK (utps->pass == 0);
    free_uploaded_tps (&utps);
  }

  /* Well-formed disabled T definition.  */
  {
    const char *good = "T1:0:D:0:0";

    parse_tracepoint_definition (good, &utps);
    SELF_CHECK (utps != nullptr);
    SELF_CHECK (!utps->enabled);
    free_uploaded_tps (&utps);
  }

  expect_bad_tracepoint_definition ("T1");
  expect_bad_tracepoint_definition ("T1:0");
  expect_bad_tracepoint_definition ("T1:0::0:0");
  expect_bad_tracepoint_definition ("T1:0:X:0:0");
  expect_bad_tracepoint_definition ("T1:0:E");
  expect_bad_tracepoint_definition ("T1:0:E:0");

  /* Well-formed Z definition: at-string hex-encodes "foo".
     Encoded form is Z<num>:<addr>:<srctype>:<start>:<len>:<hex>.  */
  {
    const char *good = "Z1:0:at:0:3:666f6f";

    parse_tracepoint_definition (good, &utps);
    SELF_CHECK (utps != nullptr);
    SELF_CHECK (utps->number == 1);
    SELF_CHECK (utps->addr == 0);
    SELF_CHECK (utps->at_string != nullptr
		&& streq (utps->at_string.get (), "foo"));
    free_uploaded_tps (&utps);
  }

  expect_bad_tracepoint_definition ("Z1");
  expect_bad_tracepoint_definition ("Z1:0");

  /* Short stale tail.  Too small to overflow the old alloca buffer, but
     still missing a trailing colon, so the parser must throw.  */
  {
    static const char prefix[] = "Z1:0:at:0:0000";
    std::vector<char> buf = make_stale_tail_buffer (prefix, "4444");
    expect_bad_tracepoint_definition (buf.data ());
  }

  /* Long stale tail.  Large enough to overflow the old alloca buffer; with
     the fix, the missing trailing colon must cause a throw instead.  */
  {
    static const char prefix[] = "Z1:0:at:0:0000";
    std::vector<char> buf
      = make_stale_tail_buffer (prefix, std::string (64, '4'));
    expect_bad_tracepoint_definition (buf.data ());
  }
}

} /* namespace tracepoint_tests */
} /* namespace selftests */

INIT_GDB_FILE (tracepoint_selftests)
{
  selftests::register_test
    ("parse_static_tracepoint_marker_definition",
     selftests::tracepoint_tests::test_parse_static_tracepoint_marker_definition);
  selftests::register_test
    ("parse_tsv_definition",
     selftests::tracepoint_tests::test_parse_tsv_definition);
  selftests::register_test
    ("parse_tracepoint_definition",
     selftests::tracepoint_tests::test_parse_tracepoint_definition);
}
