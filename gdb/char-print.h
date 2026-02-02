/* Character and string printing

   Copyright (C) 2025, 2026 Free Software Foundation, Inc.

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

#ifndef GDB_CHAR_PRINT_H
#define GDB_CHAR_PRINT_H

#include "gdb_wchar.h"
#include "gdbtypes.h"
#include "ui-file.h"
#include "charset.h"
#include "gdbsupport/gdb_obstack.h"

/* A ui_file that writes wide characters to an obstack.  */
class obstack_wide_file : public ui_file
{
public:
  explicit obstack_wide_file (struct obstack *output)
    : m_output (output)
  {
  }

  ~obstack_wide_file () = default;

  void write (const char *buf, long length_buf) override
  {
    for (long i = 0; i < length_buf; ++i)
      {
	gdb_wchar_t w = gdb_btowc (buf[i]);
	obstack_grow (m_output, &w, sizeof (gdb_wchar_t));
      }
  }

  void write (gdb_wchar_t w)
  {
    obstack_grow (m_output, &w, sizeof (gdb_wchar_t));
  }

  void write (const gdb_wchar_t *str)
  {
    obstack_grow (m_output, str, sizeof (gdb_wchar_t) * gdb_wcslen (str));
  }

private:
  struct obstack *m_output;
};

/* A class that handles display of a character or string.  The object
   is printed as if it were a literal in the source language.  This
   class implements C-like syntax, but can be subclassed for other
   languages.  */
class wchar_printer
{
private:

  /* Helper function to get a default encoding based on CHTYPE.  */
  static const char *get_default_encoding (type *chtype);

public:

  /* Constructor.  CH_TYPE is the type of the underlying character.
     QUOTER is a (narrow) character that is the quote to print,
     e.g. '\'' or '"' for C.  ENCODING is the encoding to use for the
     contents; if NULL then a default is chosen based on CH_TYPE.  */
  wchar_printer (type *ch_type, int quoter, const char *encoding = nullptr)
    : m_encoding (encoding == nullptr
		  ? get_default_encoding (ch_type)
		  : encoding),
      m_byte_order (type_byte_order (ch_type)),
      m_file (&m_wchar_buf),
      m_quoter (quoter),
      m_width (ch_type->length ()),
      m_type (ch_type)
  {
  }

  /* Print a target character, C, to STREAM.  The character is printed
     as if it were a character literal; that is, surrounded by the
     appropriate quotes for the language, and escaped as needed.  */
  void print (int c, ui_file *stream);

  /* Print the character string STRING, printing at most LENGTH
     characters.  LENGTH is -1 if the string is nul terminated.
     OPTIONS holds the printing options; printing stops early if the
     number hits print_max_chars; repeat counts are printed as
     appropriate.  Print ellipses at the end if we had to stop before
     printing LENGTH characters, or if FORCE_ELLIPSES.  If
     C_STYLE_TERMINATOR is true, and the last character is 0, then it
     is omitted.  */
  void print (struct ui_file *stream, const gdb_byte *string,
	      unsigned int length, int force_ellipses,
	      int c_style_terminator,
	      const struct value_print_options *options);

protected:

  /* Return true if W is printable.  This must agree with do_print in
     the sense that if printable returns true, then do_print is
     assumed to not emit an escape sequence for that character.
     "Printable" means that either the character is printed as-is, or
     that it is printed as a known short escape sequence, like
     "\\".  */
  virtual bool printable (gdb_wchar_t w) const;

  /* Print a wide character W.  This is called for characters where
     'printable' returns true.  */
  virtual void print_char (gdb_wchar_t w);

  /* Print an escape sequence for a character.  ORIG is a pointer to
     the original (target) bytes representing the character, ORIG_LEN
     is the number of valid bytes.  W might be gdb_WEOF, in which case
     an escape sequence for the bytes given by ORIG must be
     printed.  */
  virtual void print_escape (const gdb_byte *orig, int orig_len);

  /* Maximum number of wchars returned from wchar_iterate.  */
  static constexpr int MAX_WCHARS = 4;

  /* A structure to encapsulate state information from iterated
     character conversions.  */
  struct converted_character
  {
    /* The number of characters converted.  */
    int num_chars;

    /* The result of the conversion.  See charset.h for more.  */
    enum wchar_iterate_result result;

    /* The (saved) converted character(s).  */
    gdb_wchar_t chars[MAX_WCHARS];

    /* The first converted target byte.  */
    const gdb_byte *buf;

    /* The number of bytes converted.  */
    size_t buflen;

    /* How many times this character(s) is repeated.  */
    int repeat_count;
  };

  /* Return the repeat count of the next character/byte in ITER,
     storing the result in VEC.  */
  int count_next_character (wchar_iterator *iter,
			    std::vector<converted_character> *vec);

  /* Print the characters in CHARS.  OPTIONS is the user's print
     options.  *FINISHED is set to 0 if we didn't print all the
     elements in CHARS.  */
  void print_converted_chars_to_obstack
       (const std::vector<converted_character> &chars,
	const struct value_print_options *options,
	int *finished);

private:

  /* Intermediate output is stored here.  */
  auto_obstack m_wchar_buf;
  /* The encoding.  */
  const char *m_encoding;

protected:

  /* Byte order for the character type.  */
  bfd_endian m_byte_order;
  /* The intermediate output file.  "print" implementations should
     write to this file.  */
  obstack_wide_file m_file;
  /* The current quote character.  */
  int m_quoter;
  /* The width of a single character.  Note that for multi-byte
     encodings, this is the width of a 'base' character; e.g., for
     UTF-8 it would be 1.  */
  int m_width;
  /* The type of character being processed.  */
  type *m_type;

  /* If a character was printed as an escape sequence, and this might
     force the next character to be an escape sequence as well, then
     this can be set.  This happens in C syntax when a hex escape is
     followed by a hex digit.  */
  bool m_need_escape = false;
};

#endif /* GDB_CHAR_PRINT_H */
