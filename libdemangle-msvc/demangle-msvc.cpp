/* MSVC demangling implementation for GDB and binutils.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.  */

#include <cctype>
#include <cstring>

#include "demangle.h"
#include "demangle-msvc.h"
#include "demangle-msvc-internal.h"
#include "llvm/Demangle/Demangle.h"

using namespace llvm;

extern "C" {

/* Main MSVC demangling function with proper buffer allocation.

   Handles:
   - Microsoft Visual C++ demangling
*/
char *
msvc_demangle (const char *mangled, int options)
{
  if (msvc_mangled_start (mangled) != mangled)
    return NULL;

  return llvm_msvc_demangle (mangled, options);
}

/* Return non-zero if NAME is an MSVC-mangled symbol ('?...') or one of
   the EH throw-info names built from mangled parts (_CT??_R0..., _CTA<n>...,
   _TI[C][V][U]<n>...).  */

static int
msvc_is_mangled (const char *name)
{
  if (name == NULL)
    return 0;
  if (name[0] == '?')
    return 1;
  if (std::strncmp (name, "_CT??_R0", 8) == 0)
    return 1;

  const char *p;
  if (std::strncmp (name, "_CTA", 4) == 0)
    p = name + 4;
  else if (std::strncmp (name, "_TI", 3) == 0)
    for (p = name + 3; *p == 'C' || *p == 'V' || *p == 'U'; ++p)
      ;
  else
    return 0;

  if (!std::isdigit (static_cast<unsigned char> (*p)))
    return 0;
  while (std::isdigit (static_cast<unsigned char> (*p)))
    ++p;
  return *p != '\0';
}

/* Return where the MSVC-mangled part of NAME starts: NAME itself, or the
   '?' after the tag of an EH or unwind table '$<tag>$?<mangled>' (e.g.
   $cppxdata$?f@@YAXXZ, $handlerMap$0$?f@@YAXXZ).  NULL if NAME is not
   MSVC-mangled.  */

const char *
msvc_mangled_start (const char *name)
{
  if (name == NULL)
    return NULL;
  if (msvc_is_mangled (name))
    return name;
  if (name[0] == '$')
    {
      const char *q = std::strchr (name, '?');
      if (q != NULL && q[-1] == '$' && msvc_is_mangled (q))
	return q;
    }
  return NULL;
}

/* Extract class name from a mangled method physname.  */

char *
msvc_class_name_from_physname (const char *physname)
{
  if (msvc_mangled_start (physname) != physname)
    return NULL;

  return llvm_msvc_class_name_from_physname (physname);
}

/* Extract unqualified method name from the mangled physname.  */

char *
msvc_method_name_from_physname (const char *physname)
{
  if (msvc_mangled_start (physname) != physname)
    return NULL;

  return llvm_msvc_method_name_from_physname (physname);
}

enum gnu_v3_ctor_kinds
msvc_mangled_ctor_kind (const char *mangled)
{
  enum gnu_v3_ctor_kinds ctor;
  enum gnu_v3_dtor_kinds dtor;

  if (msvc_mangled_start (mangled) != mangled)
    return (enum gnu_v3_ctor_kinds) 0;

  llvm_msvc_structor_kind (mangled, &ctor, &dtor);
  return ctor;
}

enum gnu_v3_dtor_kinds
msvc_mangled_dtor_kind (const char *mangled)
{
  enum gnu_v3_ctor_kinds ctor;
  enum gnu_v3_dtor_kinds dtor;

  if (msvc_mangled_start (mangled) != mangled)
    return (enum gnu_v3_dtor_kinds) 0;

  llvm_msvc_structor_kind (mangled, &ctor, &dtor);
  return dtor;
}

const struct msvc_demangler_ops msvc_demangler_ops = {
  msvc_mangled_start,
  msvc_demangle,
  msvc_mangled_ctor_kind,
  msvc_mangled_dtor_kind,
  msvc_class_name_from_physname,
  msvc_method_name_from_physname,
};

} /* extern "C" */
