/* The ptid_t type and common functions operating on it.

   Copyright (C) 1986-2026 Free Software Foundation, Inc.

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

#include "ptid.h"
#include "print-utils.h"
#include "rsp-low.h"

/* See ptid.h for these.  */

ptid_t const null_ptid = ptid_t::make_null ();
ptid_t const minus_one_ptid = ptid_t::make_minus_one ();

/* See ptid.h.  */

std::string
ptid_t::to_rsp_string (bool multi) const
{
  char result[50];
  char *buf = result;
  char *endbuf = &result[sizeof (result)];

  if (multi)
    {
      if (m_pid == -1)
	buf += xsnprintf (buf, endbuf - buf, "p-1.");
      else
	buf += xsnprintf (buf, endbuf - buf, "p%x.", (unsigned) m_pid);
    }
  if (m_lwp == -1)
    xsnprintf (buf, endbuf - buf, "-1");
  else
    xsnprintf (buf, endbuf - buf, "%lx", (unsigned long) m_lwp);

  return result;
}

/* See ptid.h.  */

std::string
ptid_t::to_string () const
{
  return string_printf ("%d.%ld.%s", m_pid, m_lwp, pulongest (m_tid));
}

/* See ptid.h.  */

static ULONGEST
hex_or_minus_one (const char *buf, const char **obuf, bool for_remote)
{
  ULONGEST ret;

  if (for_remote && startswith (buf, "-1"))
    {
      ret = (ULONGEST) -1;
      buf += 2;
    }
  else
    buf = unpack_varlen_hex (buf, &ret);

  if (obuf != nullptr)
    *obuf = buf;

  return ret;
}

ptid_t
ptid_t::parse (const char *buf, const char **obuf, bool for_remote,
	       gdb::function_view<pid_type ()> default_pid)
{
  const char *p = buf;
  const char *pp;
  ptid_t::pid_type pid = 0;
  ptid_t::lwp_type lwp = 0;
  ULONGEST hex;

  using unsigned_pid_type = std::make_unsigned<pid_type>::type;
  using unsigned_lwp_type = std::make_unsigned<lwp_type>::type;

  if (*p == 'p')
    {
      /* Multi-process ptid.  */
      pp = unpack_varlen_hex (p + 1, &hex);
      if (pp == (p + 1) || *pp != '.')
	error (_("invalid remote ptid: %s"), buf);

      pid = (ptid_t::pid_type) hex;
      if (hex != ((unsigned_pid_type) pid))
	error (_("invalid remote ptid: %s"), buf);

      p = pp + 1;
      hex = hex_or_minus_one (p, &pp, for_remote);
      if (pp == p)
	error (_("invalid remote ptid: %s"), buf);

      lwp = (ptid_t::lwp_type) hex;
      if (hex != ((unsigned_lwp_type) lwp))
	error (_("invalid remote ptid: %s"), buf);

      if (obuf != nullptr)
	*obuf = pp;

      return ptid_t (pid, lwp);
    }

  /* No multi-process.  Just a thread id.  */
  hex = hex_or_minus_one (p, &pp, for_remote);

  /* Handle special thread ids.  */
  if (hex == (ULONGEST) -1)
    return minus_one_ptid;

  if (hex == 0)
    return null_ptid;

  lwp = (ptid_t::lwp_type) hex;
  if (hex != ((unsigned_lwp_type) lwp))
    error (_("invalid remote ptid: %s"), buf);

  pid = default_pid ();

  if (obuf != nullptr)
    *obuf = pp;

  return ptid_t (pid, lwp);
}
