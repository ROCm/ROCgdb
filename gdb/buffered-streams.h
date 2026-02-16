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

#ifndef GDB_BUFFERED_STREAMS_H
#define GDB_BUFFERED_STREAMS_H

#include <optional>
#include "ui-file.h"

struct buffered_streams;
class ui_out;

/* Organizes writes to a collection of buffered output streams
   so that when flushed, output is written to all streams in
   chronological order.  */

struct buffer_group
{
  buffer_group (ui_out *uiout);

  /* Flush all buffered writes to the underlying output streams.  */
  void flush () const;

  /* Record contents of BUF and associate it with STREAM.  */
  void write (const char *buf, long length_buf, ui_file *stream);

  /* Record a wrap_here and associate it with STREAM.  */
  void wrap_here (int indent, ui_file *stream);

  /* Record a call to flush and associate it with STREAM.  */
  void flush_here (ui_file *stream);

private:

  struct output_unit
  {
    output_unit (ui_file *stream, std::string msg, int wrap_hint = -1,
		 bool flush = false)
      : m_stream (stream), m_msg (msg), m_wrap_hint (wrap_hint),
	m_flush (flush)
    {}

    /* Write contents of this output_unit to the underlying stream.  */
    void flush () const;

    /* Underlying stream for which this output unit will be written to.  */
    ui_file *m_stream;

    /* String to be written to underlying buffer.  */
    std::string m_msg;

    /* Argument to wrap_here.  -1 indicates no wrap.  Used to call wrap_here
       during buffer flush.  */
    int m_wrap_hint;

    /* Indicate that the underlying output stream's flush should be called.  */
    bool m_flush;
  };

  /* Output_units to be written to buffered output streams.  */
  std::vector<output_unit> m_buffered_output;

  /* Buffered output streams.  */
  std::unique_ptr<buffered_streams> m_buffered_streams;
};

/* If FILE is a buffering_file, return its underlying stream.  */

extern ui_file *get_unbuffered (ui_file *file);

/* Buffer output to gdb_stdout and gdb_stderr for the duration of FUNC.  */

template<typename F, typename... Arg>
void
do_with_buffered_output (F func, ui_out *uiout, Arg... args)
{
  buffer_group g (uiout);

  try
    {
      func (uiout, std::forward<Arg> (args)...);
    }
  catch (gdb_exception &ex)
    {
      /* Ideally flush would be called in the destructor of buffer_group,
	 however flushing might cause an exception to be thrown.  Catch it
	 and ensure the first exception propagates.  */
      try
	{
	  g.flush ();
	}
      catch (const gdb_exception &)
	{
	}

      throw_exception (std::move (ex));
    }

  /* Try was successful.  Let any further exceptions propagate.  */
  g.flush ();
}

/* Accumulate writes to an underlying ui_file.  Output to the
   underlying file is deferred until required.  */

struct buffering_file : public ui_file
{
  buffering_file (buffer_group *group, ui_file *stream)
    : m_group (group),
      m_stream (stream)
  { /* Nothing.  */ }

  /* Return the underlying output stream.  */
  ui_file *stream () const
  {
    return m_stream;
  }

  /* Record the contents of BUF.  */
  void write (const char *buf, long length_buf) override
  {
    m_group->write (buf, length_buf, m_stream);
  }

  /* Record a wrap_here call with argument INDENT.  */
  void wrap_here (int indent) override
  {
    m_group->wrap_here (indent, m_stream);
  }

  /* Return true if the underlying stream is a tty.  */
  bool isatty () override
  {
    return m_stream->isatty ();
  }

  /* Return true if ANSI escapes can be used on the underlying stream.  */
  bool can_emit_style_escape () override
  {
    return m_stream->can_emit_style_escape ();
  }

  void emit_style_escape (const ui_file_style &style) override
  {
    if (can_emit_style_escape () && style != m_applied_style)
      {
	m_applied_style = style;
	ui_file::emit_style_escape (style);
      }
  }

  /* Flush the underlying output stream.  */
  void flush () override
  {
    return m_group->flush_here (m_stream);
  }

private:

  /* Coordinates buffering across multiple buffering_files.  */
  buffer_group *m_group;

  /* The underlying output stream.  */
  ui_file *m_stream;

  /* The currently applied style.  */
  ui_file_style m_applied_style;
};

/* Attaches and detaches buffers for each of the gdb_std* streams.  */

struct buffered_streams
{
  buffered_streams (buffer_group *group, ui_out *uiout);

  ~buffered_streams ()
  {
    this->remove_buffers ();
  }

  /* Remove buffering_files from all underlying streams.  */
  void remove_buffers ();

private:

  /* True if buffers are still attached to each underlying output stream.  */
  bool m_buffers_in_place;

  /* Buffers for each gdb_std* output stream.  */
  buffering_file m_buffered_stdout;
  buffering_file m_buffered_stderr;
  buffering_file m_buffered_stdlog;
  buffering_file m_buffered_stdtarg;

  /* Buffer for current_uiout's output stream.  */
  std::optional<buffering_file> m_buffered_current_uiout;

  /* Additional ui_out being buffered.  */
  ui_out *m_uiout;

  /* Buffer for m_uiout's output stream.  */
  std::optional<buffering_file> m_buffered_uiout;
};

#endif /* GDB_BUFFERED_STREAMS_H */
