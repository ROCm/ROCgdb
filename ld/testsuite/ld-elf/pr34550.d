#source: pr34550a.s
#source: pr34550b.s
#target: [check_shared_lib_support]
#as:
#ld: -shared --version-script=pr34550.t
#error: multiple default versions of `fmod': `GLIBC_2.0' in tmpdir/pr34550a.o and `GLIBC_2.43' in tmpdir/pr34550b.o.
