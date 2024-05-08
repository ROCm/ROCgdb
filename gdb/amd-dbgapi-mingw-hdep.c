/* Host dependent utilities for the amd-dbgapi on mingw.

   Copyright (C) 2024 Free Software Foundation, Inc.
   Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "amd-dbgapi-hdep.h"

#include <winsock2.h>
#include <windows.h>

#include <io.h>

#include <cstdio>
#include <unordered_map>
#include <algorithm>

#include "serial.h"

/* See amd-dbgapi-hdep.h.  */
const amd_dbgapi_notifier_t null_amd_dbgapi_notifier = nullptr;

/* Use a custom implementation of serial for events from dbgapi.  This is
   really similar to what is implemented in ser-event.c, except that here:
   - The underlying event object is not managed by us but by dbgapi instead,
   - We keep a handy mapping from event object's handle to serial events for
     lookup.  */

struct amd_dbgapi_serial_event_state
{
  /* The Windows event handle, provided by dbgapi.  */
  HANDLE event;
};

/* Mapping from event object's handle to serial events for lookup.  */
static std::unordered_map<HANDLE, serial *> serial_event_cache;

/* serial_ops::open implementation for the amd-dbgapi serial event.  */

static void
amd_dbgapi_serial_event_state_open (struct serial *scb,
				    const char *name)
{
  amd_dbgapi_serial_event_state *state
    = new amd_dbgapi_serial_event_state;
  scb->state = state;

  HANDLE dummy_file = CreateFile ("nul", 0, 0, nullptr, OPEN_EXISTING, 0,
				  nullptr);
  scb->fd = _open_osfhandle ((intptr_t) dummy_file, 0);
};

/* serial_ops::close implementation for the amd-dbgapi serial event.  */

static void
amd_dbgapi_serial_event_state_close (struct serial *scb)
{
  scb->fd = -1;
  delete (amd_dbgapi_serial_event_state *) scb->state;
  scb->state = nullptr;
}

/* serial_ops::wait_handle implementation for the amd-dbgapi serial event.  */

static void
amd_dbgapi_serial_event_state_wait_handle (struct serial *scb,
					   HANDLE *read,
					   HANDLE *except)
{
  amd_dbgapi_serial_event_state *state
    = (amd_dbgapi_serial_event_state *) scb->state;
  *read = state->event;
}

static const struct serial_ops amd_dbgapi_serial_event_ops =
{
  "amd-dbgapi-event",
  amd_dbgapi_serial_event_state_open,
  amd_dbgapi_serial_event_state_close,
  nullptr, /* fdopen */
  nullptr, /* readchar */
  nullptr, /* write */
  nullptr, /* flush_output */
  nullptr, /* flush_input */
  nullptr, /* send_break */
  nullptr, /* go_raw */
  nullptr, /* get_tty_state */
  nullptr, /* copy_tty_state */
  nullptr, /* set_tty_state */
  nullptr, /* print_tty_state */
  nullptr, /* setbaudrate */
  nullptr, /* setstopbits */
  nullptr, /* setparity */
  nullptr, /* drain_output */
  nullptr, /* async */
  nullptr, /* read_prim */
  nullptr, /* write_prim */
  nullptr, /* avail */
  amd_dbgapi_serial_event_state_wait_handle,
  nullptr, /* done_wait_handle */
};

/* Return the serial object associated with EVENT_HANDLE, creating it if
   necessary.  */

static serial *
get_serial_event (HANDLE event_handle)
{
  auto it = serial_event_cache.find (event_handle);

  /* We already have a FD for this event.  */
  if (it != serial_event_cache.end ())
    return it->second;

  serial *scb = serial_open_ops (&amd_dbgapi_serial_event_ops);

  /* Set the underlying Windows event object.  */
  amd_dbgapi_serial_event_state *state
    = (amd_dbgapi_serial_event_state *) scb->state;
  state->event = event_handle;
  serial_event_cache.insert ({event_handle, scb});

  /* Keep one ref for the cache.  This will be released by
     amd_dbgapi_notifier_release.  */
  serial_ref (scb);

  return scb;
}

/* See amd-dbgapi-hdep.h.  */
void
amd_dbgapi_notifier_clear (amd_dbgapi_notifier_t notifier)
{
  ResetEvent (notifier);
}

/* See amd-dbgapi-hdep.h.  */
int
amd_dbgapi_notifier_get_fd (amd_dbgapi_notifier_t notifier)
{
  return get_serial_event (notifier)->fd;
}

/* See amd-dbgapi-hdep.h.  */
void
amd_dbgapi_notifier_release (amd_dbgapi_notifier_t notifier)
{
  /* Remove the serial from the cache.  */
  auto it = serial_event_cache.find (notifier);
  if (it != serial_event_cache.end ())
    {
      serial_event_cache.erase (it);
      serial_unref (get_serial_event (notifier));
    }
}

/* See amd-dbgapi-hdep.h.  */
const char *
get_dbgapi_library_file_path ()
{
  static char path[MAX_PATH];
  HMODULE m = nullptr;

  if (GetModuleHandleEx (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
			 | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			 (const char *) amd_dbgapi_get_version, &m)
      && GetModuleFileName (m, path, sizeof (path)))
    return path;

  return "";
}
