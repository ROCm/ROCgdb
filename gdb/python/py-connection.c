/* Python interface to inferiors.

   Copyright (C) 2009-2026 Free Software Foundation, Inc.

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

#include "python-internal.h"
#include "process-stratum-target.h"
#include "inferior.h"
#include "observable.h"
#include "target-connection.h"
#include "py-events.h"
#include "py-event.h"
#include "arch-utils.h"
#include "remote.h"
#include "charset.h"
#include "gdbsupport/unordered_map.h"

/* The Python object that represents a connection.  */

struct connection_object : public PyObject
{
  /* The process target that represents this connection.   When a
     connection_object is created this field will always point at a valid
     target.  Later, if GDB stops using this target (the target is popped
     from all target stacks) then this field is set to nullptr, which
     indicates that this Python object is now in the invalid state (see
     the is_valid() method below).  */
  struct process_stratum_target *target;

  /* Require that this object be valid.  */
  void require_valid () const;

  /* Implement is_valid method.  */
  bool is_valid () const
  {
    return target != nullptr;
  }

  /* Return the id number of this connection.  */
  int get_connection_num () const
  {
    require_valid ();
    return target->connection_number;
  }

  /* Return a string that gives the short name for this connection type.  */
  const char *get_connection_type () const
  {
    require_valid ();
    return target->shortname ();
  }

  /* Return a string that gives a longer description of this
     connection type.  */
  const char *get_description () const
  {
    require_valid ();
    return target->longname ();
  }

  /* Return a string that gives additional details about this
     connection, or None, if there are no additional details for this
     connection type.  */
  const char *get_connection_details () const
  {
    require_valid ();
    return target->connection_string ();
  }

  /* Implement repr() for gdb.TargetConnection.  */
  gdbpy_ref<> repr ();

  /* The send_packet method.  */
  gdbpy_ref<> send_packet (gdbpy_borrowed_ref<> args,
			   gdbpy_opt_borrowed_ref<> kw);
};

static_assert (gdb::is_python_allocatable_v<connection_object>);

extern PyTypeObject connection_object_type;

extern PyTypeObject remote_connection_object_type;

/* A map between process_stratum targets and the Python object representing
   them.  We actually hold a gdbpy_ref around the Python object so that
   reference counts are handled correctly when entries are deleted.  */
static gdb::unordered_map<process_stratum_target *,
			  gdbpy_ref<connection_object>> all_connection_objects;

void
connection_object::require_valid () const
{
  if (!is_valid ())
    gdbpy_err_set_string (PyExc_RuntimeError,
			  _("Connection no longer exists."));
}

/* Return a reference to a gdb.TargetConnection object for TARGET.  If
   TARGET is nullptr then a reference to None is returned.

   Previously created gdb.TargetConnection objects are cached, and
   additional references to the same connection object can be returned with
   later calls to this function.  */

gdbpy_ref<>
target_to_connection_object (process_stratum_target *target)
{
  if (target == nullptr)
    return py_none ();

  gdbpy_ref <connection_object> conn_obj;
  auto conn_obj_iter = all_connection_objects.find (target);
  if (conn_obj_iter == all_connection_objects.end ())
    {
      PyTypeObject *type;

      if (is_remote_target (target))
	type = &remote_connection_object_type;
      else
	type = &connection_object_type;

      conn_obj.reset (PyObject_New (connection_object, type));
      if (conn_obj == nullptr)
	return nullptr;
      conn_obj->target = target;
      all_connection_objects.emplace (target, conn_obj);
    }
  else
    conn_obj = conn_obj_iter->second;

  gdb_assert (conn_obj != nullptr);

  /* Repackage the result as a PyObject reference.  */
  return conn_obj;
}

/* Return a list of gdb.TargetConnection objects, one for each currently
   active connection.  The returned list is in no particular order.  */

gdbpy_ref<>
gdbpy_connections ()
{
  gdbpy_ref<> list = gdbpy_list_new (0);

  for (process_stratum_target *target : all_non_exited_process_targets ())
    {
      gdb_assert (target != nullptr);

      gdbpy_ref<> conn = target_to_connection_object (target);
      /* FIXME: Python safety.  target_to_connection_object should
	 throw on error.  */
      if (conn == nullptr)
	return nullptr;
      gdb_assert (conn.get () != Py_None);

      gdbpy_list_append (list, conn);
    }

  return list;
}

/* Emit a connection event for TARGET to REGISTRY.  Return 0 on success, or
   a negative value on error.  */

static int
emit_connection_event (process_stratum_target *target,
		       eventregistry_object *registry)
{
  gdbpy_ref<> event_obj
    = create_event_object (&connection_event_object_type);
  if (event_obj == nullptr)
    return -1;

  gdbpy_ref<> conn = target_to_connection_object (target);
  if (conn == nullptr
      || evpy_add_attribute (event_obj, "connection", conn) < 0)
    return -1;

  return evpy_emit_event (event_obj, registry);
}

/* Callback for the connection_removed observer.  */

static void
connpy_connection_removed (process_stratum_target *target)
{
  if (!gdb_python_initialized)
    return;

  gdbpy_enter enter_py;

  if (evregpy_has_listeners_p (gdb_py_events.connection_removed)
      && emit_connection_event (target, gdb_py_events.connection_removed) < 0)
    gdbpy_print_stack ();

  auto conn_obj_iter = all_connection_objects.find (target);
  if (conn_obj_iter != all_connection_objects.end ())
    {
      gdbpy_ref <connection_object> conn_obj = conn_obj_iter->second;
      conn_obj->target = nullptr;
      all_connection_objects.erase (target);
    }
}

/* Called when a gdb.TargetConnection object is deallocated.  */

static void
connpy_connection_dealloc (PyObject *obj)
{
  connection_object *conn_obj = (connection_object *) obj;

  /* As the all_connection_objects map holds a reference to each connection
     object we can only enter the dealloc function when the reference in
     all_connection_objects has been erased.

     As we always set the target pointer back to nullptr before we erase
     items from all_connection_objects then, when we get here, the target
     pointer must be nullptr.  */
  gdb_assert (conn_obj->target == nullptr);

  Py_TYPE (obj)->tp_free (obj);
}

/* Implement repr() for gdb.TargetConnection.  */

gdbpy_ref<>
connection_object::repr ()
{
  if (target == nullptr)
    /* FIXME: Python safety.  gdb_py_invalid_object_repr ought to
       throw on error, and return gdbpy_ref<>, but currently does
       not.  */
    return gdbpy_ref<> (gdb_py_invalid_object_repr (this));

  return (gdbpy_unicode_from_format
	  ("<%s num=%d, what=\"%s\">",
	   gdbpy_py_obj_tp_name (this).c_str (),
	   target->connection_number,
	   make_target_connection_string (target).c_str ()));
}

/* Python specific initialization for this file.  */

static int
gdbpy_initialize_connection ()
{
  if (gdbpy_type_ready (&connection_object_type) < 0)
    return -1;

  if (gdbpy_type_ready (&remote_connection_object_type) < 0)
    return -1;

  return 0;
}

/* Set of callbacks used to implement gdb.send_packet.  */

struct py_send_packet_callbacks : public send_remote_packet_callbacks
{
  /* Constructor, initialise the result to nullptr.  It is invalid to try
     and read the result before sending a packet and processing the
     reply.  */

  py_send_packet_callbacks ()
    : m_result (nullptr)
  { /* Nothing.  */ }

  /* There's nothing to do when the packet is sent.  */

  void sending (gdb::array_view<const char> &buf) override
  { /* Nothing.  */ }

  /* When the result is returned create a Python object and assign
     this into M_RESULT.  If for any reason we can't create a Python
     object to represent the result then an exception is thrown.  If
     the result we got back from the remote is empty then set the
     result to None.  */

  void received (gdb::array_view<const char> &buf) override
  {
    if (buf.size () > 0 && buf.data ()[0] != '\0')
      m_result = gdbpy_bytes_from_string_and_size (buf);
    else
      {
	/* We didn't get back any result data; set the result to None.  */
	m_result = py_none ();
      }
  }

  /* Return the resulting Python object.  It is invalid to call this
     before sending a packet to the remote and processing the reply.

     The result value is setup in the RECEIVED call above.  */

  gdbpy_ref<> &&result ()
  {
    return std::move (m_result);
  }

private:

  /* A reference to the result value.  */

  gdbpy_ref<> m_result;
};

/* Implement RemoteTargetConnection.send_packet function.  Send a packet to
   the target identified by SELF.  The connection must still be valid, and
   the packet to be sent must be non-empty, otherwise an exception will be
   thrown.  */

gdbpy_ref<>
connection_object::send_packet (gdbpy_borrowed_ref<> args,
				gdbpy_opt_borrowed_ref<> kw)
{
  require_valid ();

  static const char *keywords[] = {"packet", nullptr};
  PyObject *packet_obj;

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "O", keywords, &packet_obj);

  /* If the packet is a unicode string then convert it to a bytes object.  */
  gdbpy_ref<> ascii_object;
  if (PyUnicode_Check (packet_obj))
    {
      /* We encode the string to bytes using the ascii codec, if this fails
	 then a suitable error will have been set.  */
      ascii_object = gdbpy_unicode_as_ascii_string (packet_obj);
      packet_obj = ascii_object.get ();
    }

  /* Check the packet is now a bytes object.  */
  if (!PyBytes_Check (packet_obj))
    gdbpy_err_set_string (PyExc_TypeError, _("Packet is not a bytes object"));

  Py_ssize_t packet_len = 0;
  const char *packet_str = nullptr;
  gdbpy_bytes_as_string_and_size (packet_obj, &packet_str, &packet_len);

  if (packet_len == 0)
    gdbpy_err_set_string (PyExc_ValueError, _("Packet must not be empty"));

  scoped_restore_current_thread restore_thread;
  switch_to_target_no_thread (target);

  gdb::array_view<const char> view (packet_str, packet_len);
  py_send_packet_callbacks callbacks;
  send_remote_packet (view, &callbacks);
  return callbacks.result ();
}

/* Global initialization for this file.  */

INIT_GDB_FILE (py_connection)
{
  gdb::observers::connection_removed.attach (connpy_connection_removed,
					     "py-connection");
}

GDBPY_INITIALIZE_FILE (gdbpy_initialize_connection);

/* Methods for the gdb.TargetConnection object type.  */

static PyMethodDef connection_object_methods[] =
{
  noargs_method<connection_object, &connection_object::is_valid> ("is_valid",
    "is_valid () -> Boolean.\n\
Return true if this TargetConnection is valid, false if not."),
  { nullptr }
};

/* Methods for the gdb.RemoteTargetConnection object type.  */

static PyMethodDef remote_connection_object_methods[] =
{
  varargs_method<connection_object, &connection_object::send_packet>
   ("send_packet",
    "send_packet (PACKET) -> Bytes\n\
Send PACKET to a remote target, return the reply as a bytes array."),
  { nullptr }
};

/* Attributes for the gdb.TargetConnection object type.  */

static gdb_PyGetSetDef connection_object_getset[] =
{
  { "num",
    wrap_getter<connection_object, &connection_object::get_connection_num>,
    nullptr,
    "ID number of this connection, as assigned by GDB.",
    nullptr },
  { "type",
    wrap_getter<connection_object, &connection_object::get_connection_type>,
    nullptr,
    "A short string that is the name for this connection type.",
    nullptr },
  { "description",
    wrap_getter<connection_object, &connection_object::get_description>,
    nullptr,
    "A longer string describing this connection type.",
    nullptr },
  { "details",
    wrap_getter<connection_object, &connection_object::get_connection_details>,
    nullptr,
    "A string containing additional connection details.",
    nullptr },
  { nullptr }
};

/* Define the gdb.TargetConnection object type.  */

PyTypeObject connection_object_type =
{
  PyVarObject_HEAD_INIT (NULL, 0)
  "gdb.TargetConnection",	  /* tp_name */
  sizeof (connection_object),	  /* tp_basicsize */
  0,				  /* tp_itemsize */
  connpy_connection_dealloc,	  /* tp_dealloc */
  0,				  /* tp_print */
  0,				  /* tp_getattr */
  0,				  /* tp_setattr */
  0,				  /* tp_compare */
  wrap_tp_callback<connection_object, &connection_object::repr>, /* tp_repr */
  0,				  /* tp_as_number */
  0,				  /* tp_as_sequence */
  0,				  /* tp_as_mapping */
  0,				  /* tp_hash  */
  0,				  /* tp_call */
  0,				  /* tp_str */
  0,				  /* tp_getattro */
  0,				  /* tp_setattro */
  0,				  /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,	/* tp_flags */
  "GDB target connection object", /* tp_doc */
  0,				  /* tp_traverse */
  0,				  /* tp_clear */
  0,				  /* tp_richcompare */
  0,				  /* tp_weaklistoffset */
  0,				  /* tp_iter */
  0,				  /* tp_iternext */
  connection_object_methods,	  /* tp_methods */
  0,				  /* tp_members */
  connection_object_getset,	  /* tp_getset */
  0,				  /* tp_base */
  0,				  /* tp_dict */
  0,				  /* tp_descr_get */
  0,				  /* tp_descr_set */
  0,				  /* tp_dictoffset */
  0,				  /* tp_init */
  0				  /* tp_alloc */
};

/* Define the gdb.RemoteTargetConnection object type.  */

PyTypeObject remote_connection_object_type =
{
  PyVarObject_HEAD_INIT (NULL, 0)
  "gdb.RemoteTargetConnection",	  /* tp_name */
  sizeof (connection_object),	  /* tp_basicsize */
  0,				  /* tp_itemsize */
  connpy_connection_dealloc,	  /* tp_dealloc */
  0,				  /* tp_print */
  0,				  /* tp_getattr */
  0,				  /* tp_setattr */
  0,				  /* tp_compare */
  wrap_tp_callback<connection_object, &connection_object::repr>, /* tp_repr */
  0,				  /* tp_as_number */
  0,				  /* tp_as_sequence */
  0,				  /* tp_as_mapping */
  0,				  /* tp_hash  */
  0,				  /* tp_call */
  0,				  /* tp_str */
  0,				  /* tp_getattro */
  0,				  /* tp_setattro */
  0,				  /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,		  /* tp_flags */
  "GDB remote target connection object",	  /* tp_doc */
  0,				  /* tp_traverse */
  0,				  /* tp_clear */
  0,				  /* tp_richcompare */
  0,				  /* tp_weaklistoffset */
  0,				  /* tp_iter */
  0,				  /* tp_iternext */
  remote_connection_object_methods,	  /* tp_methods */
  0,				  /* tp_members */
  0,				  /* tp_getset */
  &connection_object_type,	  /* tp_base */
  0,				  /* tp_dict */
  0,				  /* tp_descr_get */
  0,				  /* tp_descr_set */
  0,				  /* tp_dictoffset */
  0,				  /* tp_init */
  0				  /* tp_alloc */
};
