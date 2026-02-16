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

#include "buffered-streams.h"
#include "ui-out.h"

/* See buffered-streams.h.  */

void
buffer_group::output_unit::flush () const
{
  if (!m_msg.empty ())
    m_stream->puts (m_msg.c_str ());

  if (m_wrap_hint >= 0)
    m_stream->wrap_here (m_wrap_hint);

  if (m_flush)
    m_stream->flush ();
}

/* See buffered-streams.h.  */

void
buffer_group::write (const char *buf, long length_buf, ui_file *stream)
{
  /* Record each line separately.  */
  for (size_t prev = 0, cur = 0; cur < length_buf; ++cur)
    if (buf[cur] == '\n' || cur == length_buf - 1)
      {
	std::string msg (buf + prev, cur - prev + 1);

	if (m_buffered_output.size () > 0
	    && m_buffered_output.back ().m_wrap_hint == -1
	    && m_buffered_output.back ().m_stream == stream
	    && m_buffered_output.back ().m_msg.size () > 0
	    && m_buffered_output.back ().m_msg.back () != '\n')
	  m_buffered_output.back ().m_msg.append (msg);
	else
	  m_buffered_output.emplace_back (stream, msg);
	prev = cur + 1;
      }
}

/* See buffered-streams.h.  */

void
buffer_group::wrap_here (int indent, ui_file *stream)
{
  m_buffered_output.emplace_back (stream, "", indent);
}

/* See buffered-streams.h.  */

void
buffer_group::flush_here (ui_file *stream)
{
  m_buffered_output.emplace_back (stream, "", -1, true);
}

/* See buffered-streams.h.  */

ui_file *
get_unbuffered (ui_file *stream)
{
  while (true)
    {
      buffering_file *buf = dynamic_cast<buffering_file *> (stream);

      if (buf == nullptr)
	return stream;

      stream = buf->stream ();
    }
}

buffered_streams::buffered_streams (buffer_group *group, ui_out *uiout)
  : m_buffered_stdout (group, *redirectable_stdout ()),
    m_buffered_stderr (group, *redirectable_stderr ()),
    m_buffered_stdlog (group, *redirectable_stdlog ()),
    m_buffered_stdtarg (group, *redirectable_stdtarg ()),
    m_uiout (uiout)
{
  *redirectable_stdout () = &m_buffered_stdout;
  *redirectable_stderr () = &m_buffered_stderr;
  *redirectable_stdlog () = &m_buffered_stdlog;
  *redirectable_stdtarg () = &m_buffered_stdtarg;

  ui_file *stream = current_uiout->current_stream ();
  if (stream != nullptr)
    {
      m_buffered_current_uiout.emplace (group, stream);
      current_uiout->redirect (&(*m_buffered_current_uiout));
    }

  stream = m_uiout->current_stream ();
  if (stream != nullptr && current_uiout != m_uiout)
    {
      m_buffered_uiout.emplace (group, stream);
      m_uiout->redirect (&(*m_buffered_uiout));
    }

  m_buffers_in_place = true;
}

/* See buffered-streams.h.  */

void
buffered_streams::remove_buffers ()
{
  if (!m_buffers_in_place)
    return;

  m_buffers_in_place = false;

  *redirectable_stdout () = m_buffered_stdout.stream ();
  *redirectable_stderr () = m_buffered_stderr.stream ();
  *redirectable_stdlog () = m_buffered_stdlog.stream ();
  *redirectable_stdtarg () = m_buffered_stdtarg.stream ();

  if (m_buffered_current_uiout.has_value ())
    current_uiout->redirect (nullptr);

  if (m_buffered_uiout.has_value ())
    m_uiout->redirect (nullptr);
}

buffer_group::buffer_group (ui_out *uiout)
  : m_buffered_streams (new buffered_streams (this, uiout))
{ /* Nothing.  */ }

/* See buffered-streams.h.  */

void
buffer_group::flush () const
{
  m_buffered_streams->remove_buffers ();

  for (const output_unit &ou : m_buffered_output)
    ou.flush ();
}
