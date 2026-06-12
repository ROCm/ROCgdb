/* Native-dependent code for GNU/Linux MicroBlaze.
   Copyright (C) 2021-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include "regcache.h"
#include "gregset.h"
#include "linux-nat.h"
#include "microblaze-tdep.h"
#include "microblaze-linux-tdep.h"
#include "nat/gdb_ptrace.h"

/* ELF_NGREG from procfs.h (does not work) conflicts with asm/elf.h (works)
   just use MICROBLAZE_FSR_REGNUM as NGREG.  */

static const int microblaze_greg_begin = MICROBLAZE_R1_REGNUM;
static const int microblaze_greg_end = MICROBLAZE_FSR_REGNUM;

/* MicroBlaze Linux native additions to the default linux support.  */

class microblaze_linux_nat_target final : public linux_nat_target
{
public:
  /* Add our register access methods.  */
  void fetch_registers (struct regcache *regcache, int regnum) override;
  void store_registers (struct regcache *regcache, int regnum) override;

  /* Read suitable target description.  */
  const struct target_desc *read_description () override;
};

static microblaze_linux_nat_target the_microblaze_linux_nat_target;

/* Copy general purpose register REGNUM (or all gp regs if REGNUM == -1)
   from regset GREGS into REGCACHE.  */

static void
supply_gregset_regnum (struct regcache *regcache, const prgregset_t *gregs,
		    int regnum)
{
  const elf_greg_t *regp = *gregs;

  /* Access all registers */
  if (regnum == -1)
    {
      /* We fill the general purpose registers.  */
      for (int i = microblaze_greg_begin; i < microblaze_greg_end; i++)
	regcache->raw_supply (i, regp + i);

      /* Fill the inaccessible zero register with zero.  */
      regcache->raw_supply_zeroed (0);
    }
  else if (regnum == MICROBLAZE_R0_REGNUM)
    regcache->raw_supply_zeroed (0);
  else if (regnum >= microblaze_greg_begin && regnum < microblaze_greg_end)
    regcache->raw_supply (regnum, regp + regnum);
}

/* Copy all general purpose registers from regset GREGS into REGCACHE.  */

void
supply_gregset (struct regcache *regcache, const prgregset_t *gregs)
{
  supply_gregset_regnum (regcache, gregs, -1);
}

/* Copy general purpose register REGNUM (or all gp regs if REGNUM == -1)
   from REGCACHE into regset GREGS.  */

void
fill_gregset (const struct regcache *regcache, prgregset_t *gregs, int regnum)
{
  elf_greg_t *regp = *gregs;
  if (regnum == -1)
    {
      /* We fill the general purpose registers.  */
      for (int i = microblaze_greg_begin; i < microblaze_greg_end; i++)
	regcache->raw_collect (i, regp + i);
    }
  else if (regnum >= microblaze_greg_begin && regnum < microblaze_greg_end)
    regcache->raw_collect (regnum, regp + regnum);
}

/* Transfering floating-point registers between GDB, inferiors and cores.
   Since MicroBlaze floating-point registers are the same as GPRs these do
   nothing.  */

void
supply_fpregset (struct regcache *regcache, const gdb_fpregset_t *fpregs)
{
}

void
fill_fpregset (const struct regcache *regcache,
	       gdb_fpregset_t *fpregs, int regno)
{
}

/* Wrapper function around ptrace.  */

static void
fetch_target_gp_regs (int tid, elf_gregset_t *gregs)
{
  elf_greg_t *gregp = *gregs;

  /* Using PTRACE_PEEKUSER as PTRACE_GETREGS did not work.  */
  for (int i = microblaze_greg_begin; i < microblaze_greg_end; i++)
    {
      errno = 0;
      gregp[i] = ptrace (PTRACE_PEEKUSER, tid,
			 (PTRACE_TYPE_ARG3) (i * sizeof(elf_greg_t)), 0);
      if (errno != 0)
	{
	  perror_with_name (_("Couldn't get register"));
	}
    }
}


/* Wrapper function around ptrace.  */

static void
store_target_gp_regs ( int tid, elf_gregset_t *gregs)
{
  elf_greg_t *gregp = *gregs;

  /* Using PTRACE_POKEUSER as PTRACE_SETREGS did not work.  */
  for (int i = microblaze_greg_begin; i < microblaze_greg_end; i++)
    {
      long l;
      l = gregp[i];
      errno = 0;
      ptrace (PTRACE_POKEUSER, tid,
	      (PTRACE_TYPE_ARG3) (i * sizeof(elf_greg_t)), l);
      if (errno != 0)
	{
	  perror_with_name (_("Couldn't set register"));
	}
    }
}

/* Return a target description for the current target.  */

const struct target_desc *
microblaze_linux_nat_target::read_description ()
{
  return tdesc_microblaze_linux;
}

/* Fetch REGNUM (or all registers if REGNUM == -1) from the target
   into REGCACHE using PTRACE_GETREGSET.  */

void
microblaze_linux_nat_target::fetch_registers (struct regcache *regcache,
					      int regno)
{
  int tid;
  elf_gregset_t gregs;

  tid = get_ptrace_pid (regcache->ptid());

  fetch_target_gp_regs (tid, &gregs);

  supply_gregset_regnum (regcache, &gregs, regno);
}


/* Store REGNUM (or all registers if REGNUM == -1) to the target
   from REGCACHE using PTRACE_SETREGSET.  */

void
microblaze_linux_nat_target::store_registers (struct regcache *regcache,
					      int regno)
{
  int tid;
  elf_gregset_t gregs;

  tid = get_ptrace_pid (regcache->ptid());

  fetch_target_gp_regs (tid, &gregs);
  fill_gregset (regcache, &gregs, regno);
  store_target_gp_regs (tid, &gregs);
}

/* Initialize MicroBlaze Linux native support.  */

INIT_GDB_FILE (microblaze_linux_nat)
{
  /* Register the target.  */
  linux_target = &the_microblaze_linux_nat_target;
  add_inf_child_target (&the_microblaze_linux_nat_target);
}
