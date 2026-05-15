/* Self tests for cp-support physname helpers.

   Copyright (C) 2026 Advanced Micro Devices, Inc.

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

#include "cp-support.h"
#include "gdbsupport/selftest.h"

#include <cstdlib>
#include <cstring>

namespace selftests {

/* Verify cp_class_name_from_physname / method_name_from_physname on a
   Itanium-mangled physname.  */

static void
itanium_physname_tests ()
{
  /* _ZN7MyClass3fooEv  ==  MyClass::foo()  */
  const char *physname = "_ZN7MyClass3fooEv";

  char *cls = cp_class_name_from_physname (physname);
  SELF_CHECK (cls != NULL && std::strcmp (cls, "MyClass") == 0);
  std::free (cls);

  char *mtd = method_name_from_physname (physname);
  SELF_CHECK (mtd != NULL && std::strcmp (mtd, "foo") == 0);
  std::free (mtd);
}

#ifdef HAVE_MSVC_DEMANGLER

/* Verify cp_class_name_from_physname / method_name_from_physname on a
   MSVC mangled physname.  */

static void
msvc_physname_tests ()
{
  /* Simple method: ?foo@MyClass@@QEAAXXZ ==
     void MyClass::foo(void).  */
  char *cls = cp_class_name_from_physname ("?foo@MyClass@@QEAAXXZ");
  SELF_CHECK (cls != NULL && std::strstr (cls, "MyClass") != NULL);
  std::free (cls);

  char *mtd = method_name_from_physname ("?foo@MyClass@@QEAAXXZ");
  SELF_CHECK (mtd != NULL && std::strcmp (mtd, "foo") == 0);
  std::free (mtd);

  /* Nested class: ?bar@Inner@Outer@@QEAAXXZ ==
     void Outer::Inner::bar(void).  */
  cls = cp_class_name_from_physname ("?bar@Inner@Outer@@QEAAXXZ");
  SELF_CHECK (cls != NULL && std::strstr (cls, "Outer") != NULL
	      && std::strstr (cls, "Inner") != NULL);
  std::free (cls);

  mtd = method_name_from_physname ("?bar@Inner@Outer@@QEAAXXZ");
  SELF_CHECK (mtd != NULL && std::strcmp (mtd, "bar") == 0);
  std::free (mtd);
}

#endif /* HAVE_MSVC_DEMANGLER */

} /* namespace selftests */

INIT_GDB_FILE (cp_support_selftests)
{
  selftests::register_test ("cp_class_name_from_physname-itanium",
			    selftests::itanium_physname_tests);
#ifdef HAVE_MSVC_DEMANGLER
  selftests::register_test ("cp_class_name_from_physname-msvc",
			    selftests::msvc_physname_tests);
#endif
}
