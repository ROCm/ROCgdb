/* Copyright (C) 2025 Free Software Foundation, Inc.

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

#ifndef GDB_FINISH_THREAD_STATE_H
#define GDB_FINISH_THREAD_STATE_H

#include "gdbsupport/gdb-checked-static-cast.h"
#include "gdbsupport/scope-exit.h"
#include "gdbthread.h"
#include "target.h"
#include "process-stratum-target.h"

namespace detail
{

/* Policy class for use with scope_exit_base.  Calls finish_thread_state on
   scope exit, unless release() is called to disengage.  The release
   mechanism is supplied by scope_exit_base.  */

struct scoped_finish_thread_state_policy
{
  /* Store TARG and PTID for a later call to finish_thread_state.  For the
     meaning of these arguments, see the comments on finish_thread_state.  */
  explicit scoped_finish_thread_state_policy (process_stratum_target *targ,
					      ptid_t ptid)
    : m_ptid (ptid)
  {
    if (targ != nullptr)
      m_target_ref = target_ops_ref::new_reference (targ);
  }

  /* Called on scope exit unless release was called, see scope_exit_base
     for details.  Calls finish_thread_state with stored target and ptid.  */
  void on_exit ()
  {
    target_ops *t = m_target_ref.get ();
    process_stratum_target *p_target
      = gdb::checked_static_cast<process_stratum_target *> (t);
    finish_thread_state (p_target, m_ptid);
  }

private:

  /* The process and target passed through to finish_thread_state.  For
     their use see the comments on that function.  */
  ptid_t m_ptid;
  target_ops_ref m_target_ref;
};

}

/* Calls finish_thread_state on scope exit, unless release() is called to
   disengage.  */
using scoped_finish_thread_state
	= scope_exit_base<detail::scoped_finish_thread_state_policy>;

#endif /* GDB_FINISH_THREAD_STATE_H */
