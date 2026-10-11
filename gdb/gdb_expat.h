/* Slightly more portable version of <expat.h>.

   Copyright (C) 2006-2026 Free Software Foundation, Inc.

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

#ifndef GDB_GDB_EXPAT_H
#define GDB_GDB_EXPAT_H

#include <expat.h>

/* Expat 1.95.x does not define these; this is the definition
   recommended by the expat 2.0 headers.  */
#ifndef XML_STATUS_OK
# define XML_STATUS_OK    1
# define XML_STATUS_ERROR 0
#endif

/* Old versions of expat do not define this macro, so define it
   as void.  */
#ifndef XMLCALL
#define XMLCALL
#endif

/* Return the line number of PARSER's current position.

   Expat 2.9.0 deprecates XML_GetCurrentLineNumber in favor of
   XML_GetCurrentLineNumber64.  */

static inline std::uint64_t
gdb_xml_get_current_line_number (XML_Parser parser)
{
#if (XML_MAJOR_VERSION > 2 \
     || (XML_MAJOR_VERSION == 2 && XML_MINOR_VERSION >= 9))
  return XML_GetCurrentLineNumber64 (parser);
#else
  return XML_GetCurrentLineNumber (parser);
#endif
}

#endif /* GDB_GDB_EXPAT_H */
