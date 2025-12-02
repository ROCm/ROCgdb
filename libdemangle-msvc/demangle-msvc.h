/* MSVC demangling implementation for GDB and binutils.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception  */

#ifndef LLVM_DEMANGLE_WRAPPER_H
#define LLVM_DEMANGLE_WRAPPER_H

#include <memory>
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/MicrosoftDemangle.h"

/* Returns demangled string. */
char *llvm_msvc_demangle (const char *mangled_name, int c_demangle_flags);

/* Get class name from mangled name. Includes the class full scope. */
char *llvm_msvc_class_name_from_physname (const char *physname);

/* Get function name (unqualified name) from mangled name. */
char *llvm_msvc_method_name_from_physname (const char *physname);

std::string llvmNodeKindToString (llvm::ms_demangle::NodeKind K);

llvm::ms_demangle::SymbolNode *
llvm_demangle_symbol_node (const char *mangled_name, void **ctx);

using demangler_ctx_deleter = void (*) (void *);

struct demangler_context_deleter
{
  demangler_context_deleter () = default;
  explicit demangler_context_deleter (demangler_ctx_deleter fn) : fn (fn) {}

  void
  operator() (llvm::ms_demangle::Demangler *ptr) const
  {
    if (ptr && fn)
      fn (ptr);
  }

private:
  demangler_ctx_deleter fn = nullptr;
};

using demangler_context_ptr
  = std::unique_ptr<llvm::ms_demangle::Demangler, demangler_context_deleter>;

#endif // LLVM_DEMANGLE_WRAPPER_H
