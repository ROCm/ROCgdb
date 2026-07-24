/* Table mapping between kernel xtregset and GDB register cache.
   Copyright (C) 2007-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */


struct xtensa_regtable_t
{
  int gdb_regnum;
  int gdb_offset;
  int ptrace_cp_offset;
  int ptrace_offset;
  int size;
  int coproc;
  int dbnum;
  const char *name;
};

#define XTENSA_ELF_XTREG_SIZE 4

const xtensa_regtable_t xtensa_regmap_table[] =
{
  /* codespell:ignore-begin.  */
  /* gnum,gofs,cpofs,ofs,siz,cp, dbnum,  name */
  /* codespell:ignore-end.  */
  {   44, 176,   0,   0,  4, -1, 0x020c, "scompare1" },
  { 0 }
};
