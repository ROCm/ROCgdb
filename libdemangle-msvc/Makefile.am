## Process this file with automake to generate Makefile.in
#
#   Copyright (C) 2026 Free Software Foundation, Inc.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING3.  If not see
# <http://www.gnu.org/licenses/>.
#

AUTOMAKE_OPTIONS = no-dist foreign subdir-objects no-texinfo.tex info-in-builddir
ACLOCAL_AMFLAGS = -I . -I ../config

SUBDIRS = testsuite .

AM_CPPFLAGS = \
	-include $(top_builddir)/config.h \
	-I$(srcdir) \
	-I$(srcdir)/llvm/include \
	-I$(srcdir)/../include

info_TEXINFOS =

noinst_LIBRARIES = libdemangle-msvc.a

libdemangle_msvc_a_SOURCES = \
	demangle-msvc.cpp \
	DemangleWrapper.cpp \
	llvm/MicrosoftDemangle.cpp \
	llvm/MicrosoftDemangleNodes.cpp

include doc/local.mk
