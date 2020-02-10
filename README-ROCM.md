ROCm Debugger (ROCgdb)
======================

The ROCm Debugger (ROCgdb) is the ROCm source-level debugger for Linux, based on
the GNU Debugger (GDB). It enables heterogenous debugging on the ROCm platform
of an x86-based host architecture along with AMD GPU architectures supported by
the AMD Debugger API Library (ROCdbgapi). The AMD Debugger API Library
(ROCdbgapi) is included with the ROCm release.

The current ROCm Debugger (ROCgdb) is an initial prototype that focuses on
source line debugging and does not provide symbolic variable debugging
capabilities. The user guide presents features and commands that may be
implemented in future versions.

For more information about ROCm, see:

- https://github.com/RadeonOpenCompute/ROCm

You can use the standard GDB commands for both CPU and GPU code debugging. For
more information about ROCgdb, refer to the *ROCgdb User Guide* which is
installed at:

- ``/opt/rocm/share/info/gdb.info`` as a texinfo file
- ``/opt/rocm/share/doc/gdb/gdb.pdf`` as a PDF file

You can refer to the following chapters in the *ROCgdb User Guide* for more
specific information about debugging heterogenous programs on ROCm:

- *Debugging Heterogeneous Programs* provides general information about
  debugging heterogenous programs.
- *Configuration-Specific Information > Architectures > AMD GPU* provides
  specific information about debugging heterogenous programs on ROCm with
  supported AMD GPU chips. This section also lists the features, commands, and
  known issues that may be implemented and resolved in future releases.

For more information about the GNU Debugger (GDB), refer to the ``README`` file
in this folder or check the GNU Debugger (GDB) web site at:

- http://www.gnu.org/software/gdb

Build the ROCm Debugger
-----------------------

ROCgdb can be built on Ubuntu 16.04, Ubuntu 18.04, and Centos 7.6.

Building ROCgdb has the following prerequisites:

1. A C++11 compiler such as GCC 4.8 or Clang 3.3.

2. AMD Debugger API Library (ROCdbgapi) which can be installed as part of the
   ROCm release by the ``rocm-dbgapi`` package.

3. For Ubuntu 16.04 and Ubuntu 18.04 the following adds the needed packages:

   ````shell
   apt install bison flex gcc make ncurses-dev texinfo g++ \
     zlib1g-dev libexpat-dev libpython2.7-dev python2.7-minimal liblzma-dev \
     libbabeltrace-dev libbabeltrace-ctf-dev
   ````

4. For Centos 7.6 the following adds the needed packages:

   ````shell
   yum install -y epel-release centos-release-scl
   yum install -y bison flex gcc make texinfo gcc-c++ \
     zlib-devel expat-devel python-devel xz-devel \
     libbabeltrace-devel ncurses-devel
   ````

An example command-line to build ROCgdb on Linux is:

````shell
cd rocgdb
mkdir build
cd build
../configure --program-prefix=roc \
  --enable-64-bit-bfd --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
  --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
  --disable-gdbtk --disable-shared --with-expat --with-system-zlib \
  --without-guile --with-babeltrace --with-lzma --with-python
make
````

Specify ``--with-rocm-dbgapi=PATH`` if the the AMD Debugger API Library
(ROCdbgapi) is not installed in its default location. The ``configure`` script
looks in ``PATH/include`` and ``PATH/lib``. The default value for ``PATH`` is
``/opt/rocm``.

The built ROCgdb executable will be placed in:

- ``build/gdb/gdb``

The texinfo *User Manual* will be placed in:

- ``build/gdb/doc/gdb.info``

To install ROCgdb:

````shell
make install
````

The installed ROCgdb will be placed in:

- ``<prefix>/bin/rocgdb``

To execute ROCgdb, the ROCdbgapi library and its dependent ROCcomgr library must
be installed. These can be installed as part of the ROCm release by the
``rocm-dbgapi`` package:

- ``librocm-dbgapi.so.0``
- ``libamd_comgr.so.1``

The PDF *User Manual* can be generated with:

````shell
make pdf
````

The generated PDF will be placed in:

- ``build/gdb/doc/gdb.pdf``
