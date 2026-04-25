AUTOMAKE_OPTIONS = no-dist foreign subdir-objects
ACLOCAL_AMFLAGS = -I .

SUBDIRS = testsuite

AM_CPPFLAGS = \
	-include $(top_builddir)/config.h \
	-I$(srcdir) \
	-I$(srcdir)/llvm/include \
	-I$(srcdir)/../include

noinst_LIBRARIES = libdemangle-msvc.a

libdemangle_msvc_a_SOURCES = \
	demangle-msvc.cpp \
	DemangleWrapper.cpp \
	llvm/MicrosoftDemangle.cpp \
	llvm/MicrosoftDemangleNodes.cpp
