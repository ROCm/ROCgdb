/* Copyright (C) 2025, 2026 Free Software Foundation, Inc.

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

#ifndef GDB_LOGGING_FILE_H
#define GDB_LOGGING_FILE_H

#include "ui-file.h"

/* A ui_file implementation that optionally writes its output to a
   second logging stream.  Whether logging is actually done depends on
   the user's logging settings.  The precise underlying ui_file type
   is a template parameter, so that either owning or non-owning
   instances can be made.  */

template<typename T>
class logging_file : public ui_file
{
public:
  /* This wraps another stream.  Whether or not output actually goes
     to that stream depends on the redirection settings.  FOR_STDLOG
     should only be set for a stream intended by use as gdb_stdlog;
     this is used to implement the "debug redirect" feature.  */
  logging_file (T out, bool for_stdlog = false)
    : m_out (std::move (out)),
      m_for_stdlog (for_stdlog)
  {
  }

  void write (const char *buf, long length_buf) override;
  void write_async_safe (const char *buf, long length_buf) override;
  void puts (const char *) override;
  void flush () override;
  bool can_page () const override;
  void emit_style_escape (const ui_file_style &style) override;
  void puts_unfiltered (const char *str) override;

  bool isatty () override
  {
    /* Defer to the wrapped file.  */
    return m_out->isatty ();
  }

  bool term_out () override
  {
    /* Defer to the wrapped file.  */
    return m_out->term_out ();
  }

  bool can_emit_style_escape () override
  {
    /* Defer to the wrapped file.  */
    return m_out->can_emit_style_escape ();
  }

private:
  /* A helper function that returns true if output should go to
     M_OUT.  */
  bool ordinary_output () const;

  /* The underlying file.  */
  T m_out;

  /* True if this stream is used for gdb_stdlog.  This is used to
     implement the debug redirect feature.  */
  bool m_for_stdlog;
};

#endif /* GDB_LOGGING_FILE_H */
