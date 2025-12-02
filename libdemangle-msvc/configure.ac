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

AC_CONFIG_FILES([Makefile testsuite/Makefile])
AC_OUTPUT
