/* MSVC demangling implementation for GDB and binutils.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception  */

#include "demangle.h"
#include "demangle-msvc.h"
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
  if (mangled == NULL || *mangled == '\0')
    return NULL;

  if (mangled[0] != '?')
    {
      return NULL;
    }

  /* Try MSVC demangling if it looks like MSVC */
  return llvm_msvc_demangle (mangled, options);
}

/* Free memory allocated by msvc_demangle.  */
void
msvc_demangle_free (char *demangled)
{
  std::free (demangled);
}

char *
msvc_class_name_from_physname (const char *physname)
{
  return llvm_msvc_class_name_from_physname (physname);
}

char *
msvc_method_name_from_physname (const char *physname)
{
  return llvm_msvc_method_name_from_physname (physname);
}

} /* extern "C" */
