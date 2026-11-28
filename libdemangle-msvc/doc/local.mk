if ENABLE_MSVC_DEMANGLER
info_TEXINFOS += %D%/libdemangle-msvc.texi
endif

AM_MAKEINFOFLAGS = --no-split -I "$(srcdir)/%D%"
TEXI2DVI = texi2dvi -I "$(srcdir)/%D%"

DISTCLEANFILES = %D%/libdemangle-msvc.?? %D%/libdemangle-msvc.???
