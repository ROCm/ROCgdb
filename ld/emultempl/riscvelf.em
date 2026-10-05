# This shell script emits a C file. -*- C -*-
#   Copyright (C) 2004-2026 Free Software Foundation, Inc.
#
# This file is part of the GNU Binutils.
#
# This program is free software; you can redistribute it and/or modify
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
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston,
# MA 02110-1301, USA.

fragment <<EOF

#include "ldmain.h"
#include "ldctor.h"
#include "elf/riscv.h"
#include "elfxx-riscv.h"

static struct riscv_elf_params params = { .relax_gp = 1,
					  .check_uleb128 = 0,
					  .zicfilp = RISCV_ZICFILP_IMPLICIT,
					  .zicfiss = RISCV_ZICFISS_IMPLICIT,
					  .zicfilp_unlabeled_report = RISCV_REPORT_NONE,
					  .zicfiss_report = RISCV_REPORT_NONE };

/* Parse the value of -z NAME=VALUE for the CFI report options.  */

static riscv_report_policy
riscv_parse_report_option (const char *option, const char *value)
{
  if (strcmp (value, "none") == 0)
    return RISCV_REPORT_NONE;
  if (strcmp (value, "warning") == 0)
    return RISCV_REPORT_WARNING;
  if (strcmp (value, "error") == 0)
    return RISCV_REPORT_ERROR;
  fatal (_("%P: error: unrecognized value '-z %s'\n"), option);
}
EOF

# Define some shell vars to insert bits of code into the standard elf
# parse_args and list_options functions.  */
PARSE_AND_LIST_LONGOPTS=${PARSE_AND_LIST_LONGOPTS}'
    { "relax-gp", no_argument, NULL, OPTION_RELAX_GP },
    { "no-relax-gp", no_argument, NULL, OPTION_NO_RELAX_GP },
    { "check-uleb128", no_argument, NULL, OPTION_CHECK_ULEB128 },
    { "no-check-uleb128", no_argument, NULL, OPTION_NO_CHECK_ULEB128 },
'

PARSE_AND_LIST_OPTIONS=${PARSE_AND_LIST_OPTIONS}'
  fprintf (file, _("  --relax-gp                  Perform GP relaxation\n"));
  fprintf (file, _("  --no-relax-gp               Don'\''t perform GP relaxation\n"));
  fprintf (file, _("  --check-uleb128             Check if SUB_ULEB128 has non-zero addend\n"));
  fprintf (file, _("  --no-check-uleb128          Don'\''t check if SUB_ULEB128 has non-zero addend\n"));
  fprintf (file, _("\
  -z zicfilp=[implicit|unlabeled|never]\n\
                              Control the Zicfilp marking of the output\n\
                                implicit (default): deduce from the inputs\n\
                                unlabeled: mark the output with CFI_LP_UNLABELED\n\
                                  and generate the landing pad PLT\n\
                                never: never mark the output with Zicfilp\n"));
  fprintf (file, _("\
  -z zicfilp-unlabeled-report=[none|warning|error]\n\
                              Report the inputs without CFI_LP_UNLABELED\n\
                                (default: none)\n"));
  fprintf (file, _("\
  -z zicfiss=[implicit|always|never]\n\
                              Control the Zicfiss marking of the output\n\
                                implicit (default): deduce from the inputs\n\
                                always: mark the output with CFI_SS\n\
                                never: never mark the output with CFI_SS\n"));
  fprintf (file, _("\
  -z zicfiss-report=[none|warning|error]\n\
                              Report the inputs without CFI_SS (default: none)\n"));
'

PARSE_AND_LIST_ARGS_CASE_Z_RISCV='
      else if (startswith (optarg, "zicfilp="))
	{
	  const char *value = optarg + strlen ("zicfilp=");
	  if (strcmp (value, "implicit") == 0)
	    params.zicfilp = RISCV_ZICFILP_IMPLICIT;
	  else if (strcmp (value, "unlabeled") == 0)
	    params.zicfilp = RISCV_ZICFILP_UNLABELED;
	  else if (strcmp (value, "never") == 0)
	    params.zicfilp = RISCV_ZICFILP_NEVER;
	  else
	    fatal (_("%P: error: unrecognized value '\''-z %s'\''\n"), optarg);
	}
      else if (startswith (optarg, "zicfilp-unlabeled-report="))
	params.zicfilp_unlabeled_report
	  = riscv_parse_report_option (optarg,
	      optarg + strlen ("zicfilp-unlabeled-report="));
      else if (startswith (optarg, "zicfiss="))
	{
	  const char *value = optarg + strlen ("zicfiss=");
	  if (strcmp (value, "implicit") == 0)
	    params.zicfiss = RISCV_ZICFISS_IMPLICIT;
	  else if (strcmp (value, "always") == 0)
	    params.zicfiss = RISCV_ZICFISS_ALWAYS;
	  else if (strcmp (value, "never") == 0)
	    params.zicfiss = RISCV_ZICFISS_NEVER;
	  else
	    fatal (_("%P: error: unrecognized value '\''-z %s'\''\n"), optarg);
	}
      else if (startswith (optarg, "zicfiss-report="))
	params.zicfiss_report
	  = riscv_parse_report_option (optarg,
	      optarg + strlen ("zicfiss-report="));
'

PARSE_AND_LIST_ARGS_CASE_Z="$PARSE_AND_LIST_ARGS_CASE_Z $PARSE_AND_LIST_ARGS_CASE_Z_RISCV"

PARSE_AND_LIST_ARGS_CASES=${PARSE_AND_LIST_ARGS_CASES}'
    case OPTION_RELAX_GP:
      params.relax_gp = 1;
      break;

    case OPTION_NO_RELAX_GP:
      params.relax_gp = 0;
      break;

    case OPTION_CHECK_ULEB128:
      params.check_uleb128 = 1;
      break;

    case OPTION_NO_CHECK_ULEB128:
      params.check_uleb128 = 0;
      break;
'

fragment <<EOF
static void
riscv_elf_before_allocation (void)
{
  gld${EMULATION_NAME}_before_allocation ();

  if (link_info.discard == discard_sec_merge)
    link_info.discard = discard_l;

  if (!bfd_link_relocatable (&link_info))
    {
      /* We always need at least some relaxation to handle code alignment.  */
      if (RELAXATION_DISABLED_BY_USER)
	TARGET_ENABLE_RELAXATION;
      else
	ENABLE_RELAXATION;
    }

  /* BFD picks the relax passes for this link.  */
  link_info.relax_pass = bfd_elf${ELFSIZE}_riscv_init_relax_passes (&link_info);
}

static void
gld${EMULATION_NAME}_after_allocation (void)
{
  int need_layout = 0;

  /* Don't attempt to discard unused .eh_frame sections until the final link,
     as we can't reliably tell if they're used until after relaxation.  */
  if (!bfd_link_relocatable (&link_info))
    {
      need_layout = bfd_elf_discard_info (&link_info);
      if (need_layout < 0)
	{
	  einfo (_("%X%P: .eh_frame/.stab edit: %E\n"));
	  return;
	}
    }

  /* PR 27566, if the phase of data segment is exp_seg_relro_adjust,
     that means we are still adjusting the relro, and shouldn't do the
     relaxations at this stage.  Otherwise, we will get the symbol
     values beofore handling the relro, and may cause truncated fails
     when the relax range crossing the data segment.  One of the solution
     is to monitor the data segment phase while relaxing, to know whether
     the relro has been handled or not.

     I think we probably need to record more information about data
     segment or alignments in the future, to make sure it is safe
     to doing relaxations.  */
  enum phase_enum *phase = &(expld.dataseg.phase);
  bfd_elf${ELFSIZE}_riscv_set_data_segment_info (&link_info, (int *) phase);

  ldelf_map_segments (need_layout);
}

/* This is a convenient point to tell BFD about target specific flags.
   After the output has been created, but before inputs are read.  */

static void
riscv_after_open_output (void)
{
  /* See PR 22920 for an example of why this is necessary.  */
  if (strstr (bfd_get_target (link_info.output_bfd), "riscv") == NULL)
    {
      /* The RISC-V backend needs special fields in the output hash structure.
	 These will only be created if the output format is a RISC-V format,
	 hence we do not support linking and changing output formats at the
	 same time.  Use a link followed by objcopy to change output formats.  */
      fatal (_("%P: error: cannot change output format"
	       " whilst linking %s binaries\n"), "RISC-V");
      return;
    }

  ldelf_after_open_output ();
  riscv_elf${ELFSIZE}_set_options (&link_info, &params);
}

EOF

LDEMUL_BEFORE_ALLOCATION=riscv_elf_before_allocation
LDEMUL_AFTER_ALLOCATION=gld${EMULATION_NAME}_after_allocation
LDEMUL_AFTER_OPEN_OUTPUT=riscv_after_open_output
