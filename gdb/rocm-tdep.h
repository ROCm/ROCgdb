/* Target-dependent code for ROCm.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _ROCM_TDEP_H
#define _ROCM_TDEP_H 1

#include "gdbsupport/common-defs.h"
#include "gdbsupport/observable.h"
#include "gdbsupport/traits.h"

#include <amd-dbgapi.h>

namespace detail
{

template <typename T>
using is_amd_dbgapi_handle
  = gdb::Or<std::is_same<T, amd_dbgapi_address_class_id_t>,
	    std::is_same<T, amd_dbgapi_address_space_id_t>,
	    std::is_same<T, amd_dbgapi_architecture_id_t>,
	    std::is_same<T, amd_dbgapi_agent_id_t>,
	    std::is_same<T, amd_dbgapi_breakpoint_id_t>,
	    std::is_same<T, amd_dbgapi_code_object_id_t>,
	    std::is_same<T, amd_dbgapi_dispatch_id_t>,
	    std::is_same<T, amd_dbgapi_displaced_stepping_id_t>,
	    std::is_same<T, amd_dbgapi_event_id_t>,
	    std::is_same<T, amd_dbgapi_process_id_t>,
	    std::is_same<T, amd_dbgapi_queue_id_t>,
	    std::is_same<T, amd_dbgapi_register_class_id_t>,
	    std::is_same<T, amd_dbgapi_register_id_t>,
	    std::is_same<T, amd_dbgapi_watchpoint_id_t>,
	    std::is_same<T, amd_dbgapi_wave_id_t>>;

} /* namespace detail */

template <typename T,
	  typename = gdb::Requires<detail::is_amd_dbgapi_handle<T>>>
bool
operator== (const T &lhs, const T &rhs)
{
  return lhs.handle == rhs.handle;
}

template <typename T,
	  typename = gdb::Requires<detail::is_amd_dbgapi_handle<T>>>
bool
operator!= (const T &lhs, const T &rhs)
{
  return !(lhs == rhs);
}

/* ROCm Debug API event observers.  */

extern gdb::observers::observable<> amd_dbgapi_runtime_loaded;
extern gdb::observers::observable<> amd_dbgapi_runtime_unloaded;
extern gdb::observers::observable<> amd_dbgapi_code_object_list_updated;

/* Return true if the given ptid is a GPU thread (wave) ptid.  */

static inline bool
ptid_is_gpu (ptid_t ptid)
{
  /* FIXME: Currently using values that are known not to conflict with other
     processes to indicate if it is a GPU thread.  ptid.pid 1 is the init
     process and is the only process that could have a ptid.lwp of 1.  The init
     process cannot have a GPU.  No other process can have a ptid.lwp of 1.
     The GPU wave ID is stored in the ptid.tid.  */
  return ptid.pid () != 1 && ptid.lwp () == 1;
}

/* Return true if this is the AMDGCN architecture.  */
extern bool rocm_is_amdgcn_gdbarch (struct gdbarch *gdbarch);

/* Return the current inferior's amd_dbgapi process id.  */
extern amd_dbgapi_process_id_t
get_amd_dbgapi_process_id (struct inferior *inferior = nullptr);

static inline amd_dbgapi_wave_id_t
get_amd_dbgapi_wave_id (ptid_t ptid)
{
  gdb_assert (ptid_is_gpu (ptid));
  return amd_dbgapi_wave_id_t{
    static_cast<decltype (amd_dbgapi_wave_id_t::handle)> (ptid.tid ())
  };
}

/* Convenience wrapper around amd_dbgapi_wave_get_info that avoids
   manually specifying RES's size and accepts a thread pointer instead
   of a wave id.  */

template<typename Res>
static amd_dbgapi_status_t
wave_get_info (thread_info *tp, amd_dbgapi_wave_info_t query, Res &res)
{
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (tp->ptid);

  return amd_dbgapi_wave_get_info (wave_id, query, sizeof (res), &res);
}

/* Like wave_get_info above, but throws an error if the dbgapi call
   fails.  */

template<typename Res>
static void
wave_get_info_throw (thread_info *tp, amd_dbgapi_wave_info_t query, Res &res)
{
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (tp->ptid);

  if (amd_dbgapi_wave_get_info (wave_id, query, sizeof (res), &res)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_wave_get_info for wave_%ld failed "),
	   wave_id.handle);
}

/* Convenience wrapper around amd_dbgapi_dispatch_get_info that avoids
   manually specifying RES's size and throws an error if the dbgapi
   call fails.  */

template<typename Res>
static void
dispatch_get_info_throw (amd_dbgapi_dispatch_id_t dispatch_id,
			 amd_dbgapi_dispatch_info_t query, Res &res)
{
  if (amd_dbgapi_dispatch_get_info (dispatch_id, query, sizeof (res), &res)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_dispatch_get_info for dispatch_%ld failed "),
	   dispatch_id.handle);
}

#endif /* rocm-tdep.h */
