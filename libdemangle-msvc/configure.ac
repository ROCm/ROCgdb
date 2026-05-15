dnl Autoconf configure script for libdemangle-msvc
dnl Copyright (C) 2026 Advanced Micro Devices, Inc.
dnl
dnl Licensed under the Apache License v2.0 with LLVM Exceptions.
dnl See https://llvm.org/LICENSE.txt for license information.
dnl SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

dnl Process this file with autoconf to produce a configure script.

AC_INIT([libdemangle-msvc], [1.0])
AC_CONFIG_SRCDIR(demangle-msvc.cpp)
AC_CONFIG_HEADERS([config.h:config.in])

# Set build, build_cpu, build_vendor and build_os.
AC_CANONICAL_BUILD

# Set host, host_cpu, host_vendor, and host_os.
AC_CANONICAL_HOST

# Set target, target_cpu, target_vendor, and target_os.
AC_CANONICAL_TARGET

AM_MAINTAINER_MODE
AC_CONFIG_AUX_DIR([..])
AM_INIT_AUTOMAKE
AM_SILENT_RULES([yes])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB

AC_USE_SYSTEM_EXTENSIONS
AC_SYS_LARGEFILE
AC_CHECK_HEADERS([libgen.h])

AC_CHECK_DECLS([basename])

# We require C++17
AX_CXX_COMPILE_STDCXX(17, , mandatory)

AC_ARG_ENABLE(msvc-demangle-debug,
[  --enable-msvc-demangle-debug
			  enable diagnostic output in SET_ERROR],
[case "${enableval}" in
  no|none)  ac_msvc_debug= ;;
  *)	    ac_msvc_debug=yes ;;
esac])dnl
if test x$ac_msvc_debug != x ; then
  AC_DEFINE(DEBUG_MSVC_DEMANGLE, 1, [Enable diagnostic output in MSVC demangler.])
fi

# Always build when configured.  The top-level configure skips this
# directory entirely when --without-msvc-demangler is passed.
# Whether consumers link this library is controlled by
# --with-msvc-demangler in gdb/binutils.
AM_CONDITIONAL([ENABLE_MSVC_DEMANGLER], [true])

AC_CONFIG_FILES([Makefile testsuite/Makefile])
AC_OUTPUT
