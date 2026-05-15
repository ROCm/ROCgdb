/* MSVC demangling — public C interface.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.  */

#ifndef DEMANGLE_MSVC_H
#define DEMANGLE_MSVC_H

#include "demangle.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Public C entry points provided by libdemangle-msvc.  Tools that want
   MSVC demangling link this library and register msvc_demangler_ops
   via cplus_demangle_set_msvc_ops (declared in include/demangle.h).

   In addition to plain demangling, libdemangle-msvc exposes queries
   that GDB normally derives by walking the libiberty demangle_component_t
   AST: msvc_class_name_from_physname, msvc_method_name_from_physname,
   msvc_mangled_ctor_kind and msvc_mangled_dtor_kind.  */

extern const char *msvc_mangled_start (const char *name);
extern char *msvc_demangle (const char *mangled, int options);
extern char *msvc_class_name_from_physname (const char *physname);
extern char *msvc_method_name_from_physname (const char *physname);
extern enum gnu_v3_ctor_kinds msvc_mangled_ctor_kind (const char *mangled);
extern enum gnu_v3_dtor_kinds msvc_mangled_dtor_kind (const char *mangled);

extern const struct msvc_demangler_ops msvc_demangler_ops;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DEMANGLE_MSVC_H */
