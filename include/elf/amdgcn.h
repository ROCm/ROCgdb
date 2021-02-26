/* AMDGCN ELF support for BFD.

   Copyright (C) 2019-2020 Free Software Foundation, Inc.
   Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

   This file is part of BFD, the Binary File Descriptor library.

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

#ifndef _ELF_AMDGCN_H
#define _ELF_AMDGCN_H

#include "elf/reloc-macros.h"

/* Bits in the e_flags field of the Elf64_Ehdr:  */

#define EF_AMDGPU_MACH      0x0ff /* Processor selection mask for EF_AMDGPU_MACH_* values. */
#define EF_AMDGPU_MACH_NONE 0x000 /* Not specified processor */

/* R600-based processors. */

/* Radeon HD 2000/3000 Series (R600). */
#define EF_AMDGPU_MACH_R600_R600  0x001
#define EF_AMDGPU_MACH_R600_R630  0x002
#define EF_AMDGPU_MACH_R600_RS880 0x003
#define EF_AMDGPU_MACH_R600_RV670 0x004
/* Radeon HD 4000 Series (R700). */
#define EF_AMDGPU_MACH_R600_RV710 0x005
#define EF_AMDGPU_MACH_R600_RV730 0x006
#define EF_AMDGPU_MACH_R600_RV770 0x007
/* Radeon HD 5000 Series (Evergreen). */
#define EF_AMDGPU_MACH_R600_CEDAR   0x008
#define EF_AMDGPU_MACH_R600_CYPRESS 0x009
#define EF_AMDGPU_MACH_R600_JUNIPER 0x00a
#define EF_AMDGPU_MACH_R600_REDWOOD 0x00b
#define EF_AMDGPU_MACH_R600_SUMO 0x00c
/* Radeon HD 6000 Series (Northern Islands). */
#define EF_AMDGPU_MACH_R600_BARTS  0x00d
#define EF_AMDGPU_MACH_R600_CAICOS 0x00e
#define EF_AMDGPU_MACH_R600_CAYMAN 0x00f
#define EF_AMDGPU_MACH_R600_TURKS  0x010

/* Reserved for R600-based processors. */
#define EF_AMDGPU_MACH_R600_RESERVED_FIRST 0x011
#define EF_AMDGPU_MACH_R600_RESERVED_LAST  0x01f

/* First/last R600-based processors. */
#define EF_AMDGPU_MACH_R600_FIRST EF_AMDGPU_MACH_R600_R600
#define EF_AMDGPU_MACH_R600_LAST EF_AMDGPU_MACH_R600_TURKS

/* AMDGCN-based processors. */

/* AMDGCN GFX6. */
#define EF_AMDGPU_MACH_AMDGCN_GFX600 0x020
#define EF_AMDGPU_MACH_AMDGCN_GFX601 0x021
/* AMDGCN GFX7. */
#define EF_AMDGPU_MACH_AMDGCN_GFX700 0x022
#define EF_AMDGPU_MACH_AMDGCN_GFX701 0x023
#define EF_AMDGPU_MACH_AMDGCN_GFX702 0x024
#define EF_AMDGPU_MACH_AMDGCN_GFX703 0x025
#define EF_AMDGPU_MACH_AMDGCN_GFX704 0x026
/* AMDGCN GFX8. */
#define EF_AMDGPU_MACH_AMDGCN_GFX801 0x028
#define EF_AMDGPU_MACH_AMDGCN_GFX802 0x029
#define EF_AMDGPU_MACH_AMDGCN_GFX803 0x02a
#define EF_AMDGPU_MACH_AMDGCN_GFX810 0x02b
/* AMDGCN GFX9. */
#define EF_AMDGPU_MACH_AMDGCN_GFX900 0x02c
#define EF_AMDGPU_MACH_AMDGCN_GFX902 0x02d
#define EF_AMDGPU_MACH_AMDGCN_GFX904 0x02e
#define EF_AMDGPU_MACH_AMDGCN_GFX906 0x02f
#define EF_AMDGPU_MACH_AMDGCN_GFX908 0x030
#define EF_AMDGPU_MACH_AMDGCN_GFX90A 0x03f
/* AMDGCN GFX10.  */
#define EF_AMDGPU_MACH_AMDGCN_GFX1010 0x033
#define EF_AMDGPU_MACH_AMDGCN_GFX1011 0x034
#define EF_AMDGPU_MACH_AMDGCN_GFX1012 0x035
#define EF_AMDGPU_MACH_AMDGCN_GFX1030 0x036
#define EF_AMDGPU_MACH_AMDGCN_GFX1031 0x037

/* Reserved for AMDGCN-based processors. */
#define EF_AMDGPU_MACH_AMDGCN_RESERVED0 0x027
#define EF_AMDGPU_MACH_AMDGCN_RESERVED1 0x030

/* First/last AMDGCN-based processors. */
#define EF_AMDGPU_MACH_AMDGCN_FIRST EF_AMDGPU_MACH_AMDGCN_GFX600
#define EF_AMDGPU_MACH_AMDGCN_LAST EF_AMDGPU_MACH_AMDGCN_GFX90A

/* Indicates if the "xnack" target feature is enabled for all code contained */
/* in the object. */
#define EF_AMDGPU_XNACK 0x100
/* Indicates if the "sram-ecc" target feature is enabled for all code */
/* contained in the object. */
#define EF_AMDGPU_SRAM_ECC 0x200


/* Additional symbol types for AMDGCN.  */

#define STT_AMDGPU_HSA_KERNEL 10 /* Symbol is a kernel descriptor */


/* Note segments. */

#define NT_AMDGPU_HSA_RESERVED_0          0
#define NT_AMDGPU_HSA_CODE_OBJECT_VERSION 1
#define NT_AMDGPU_HSA_HSAIL               2
#define NT_AMDGPU_HSA_ISA                 3
#define NT_AMDGPU_HSA_PRODUCER            4
#define NT_AMDGPU_HSA_PRODUCER_OPTIONS    5
#define NT_AMDGPU_HSA_EXTENSION           6
#define NT_AMDGPU_HSA_RESERVED_7          7
#define NT_AMDGPU_HSA_RESERVED_8          8
#define NT_AMDGPU_HSA_RESERVED_9          9

/* Code Object V2. */
/* Note types with values between 0 and 9 (inclusive) are reserved. */
#define NT_AMDGPU_HSA_METADATA            10
#define NT_AMDGPU_ISA                     11
#define NT_AMDGPU_PAL_METADATA            12

/* Code Object V3. */
/* Note types with values between 0 and 31 (inclusive) are reserved. */
#define NT_AMDGPU_METADATA                32

/* Other */
#define NT_AMDGPU_HSA_HLDEBUG_DEBUG       101
#define NT_AMDGPU_HSA_HLDEBUG_TARGET      102

/* Relocation types.  */

START_RELOC_NUMBERS (elf_amdgcn_reloc_type)
 RELOC_NUMBER (R_AMDGPU_NONE,           0)
 RELOC_NUMBER (R_AMDGPU_ABS32_LO,       1)
 RELOC_NUMBER (R_AMDGPU_ABS32_HI,       2)
 RELOC_NUMBER (R_AMDGPU_ABS64,          3)
 RELOC_NUMBER (R_AMDGPU_REL32,          4)
 RELOC_NUMBER (R_AMDGPU_REL64,          5)
 RELOC_NUMBER (R_AMDGPU_ABS32,          6)
 RELOC_NUMBER (R_AMDGPU_GOTPCREL,       7)
 RELOC_NUMBER (R_AMDGPU_GOTPCREL32_LO,  8)
 RELOC_NUMBER (R_AMDGPU_GOTPCREL32_HI,  9)
 RELOC_NUMBER (R_AMDGPU_REL32_LO,      10)
 RELOC_NUMBER (R_AMDGPU_REL32_HI,      11)
 RELOC_NUMBER (R_AMDGPU_RELATIVE64,    13)
END_RELOC_NUMBERS (R_AMDGPU_max)


#endif /* _ELF_AMDGCN_H */
