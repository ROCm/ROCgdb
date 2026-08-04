#as: -march=rv64ifc -gdwarf-2
#name: Dwarf line number info for .insn
#objdump: -WL -w
#source: insn.s

.*:[ 	]+file format .*

Contents of the .debug_line section:

CU: .*/insn.s:
File name +Line number +Starting address.*
insn.s +2 +0.*
insn.s +3 +0x4.*
insn.s +4 +0x8.*
insn.s +5 +0xc.*
insn.s +6 +0x10.*
insn.s +7 +0x14.*
insn.s +8 +0x18.*
insn.s +9 +0x1c.*
insn.s +10 +0x20.*
insn.s +11 +0x24.*
insn.s +13 +0x28.*
insn.s +14 +0x2a.*
insn.s +15 +0x2c.*
insn.s +16 +0x2e.*
insn.s +17 +0x30.*
insn.s +18 +0x32.*
insn.s +19 +0x34.*
insn.s +20 +0x36.*
insn.s +22 +0x38.*
insn.s +23 +0x3c.*
insn.s +24 +0x40.*
insn.s +25 +0x44.*
insn.s +26 +0x48.*
insn.s +27 +0x4c.*
insn.s +28 +0x50.*
insn.s +29 +0x54.*
insn.s +30 +0x58.*
insn.s +31 +0x5c.*
insn.s +33 +0x60.*
insn.s +34 +0x62.*
insn.s +35 +0x64.*
insn.s +36 +0x66.*
insn.s +37 +0x68.*
insn.s +38 +0x6a.*
insn.s +39 +0x6c.*
insn.s +40 +0x6e.*
insn.s +41 +0x70.*
insn.s +43 +0x72.*
insn.s +44 +0x76.*
insn.s +45 +0x7a.*
insn.s +46 +0x7e.*
insn.s +47 +0x82.*
insn.s +48 +0x86.*
insn.s +49 +0x8a.*
insn.s +50 +0x8e.*
insn.s +51 +0x92.*
insn.s +52 +0x96.*
insn.s +53 +0x9a.*
insn.s +54 +0x9e.*
insn.s +55 +0xa2.*
insn.s +57 +0xa6.*
insn.s +61 +0xaa.*
insn.s +62 +0xae.*
insn.s +63 +0xb2.*
insn.s +64 +0xb6.*
insn.s +65 +0xba.*
insn.s +66 +0xbe.*
insn.s +67 +0xc2.*
insn.s +68 +0xc6.*
insn.s +69 +0xca.*
insn.s +70 +0xce.*
insn.s +71 +0xd2.*
insn.s +72 +0xd6.*
insn.s +73 +0xda.*
insn.s +74 +0xde.*
insn.s +77 +0xe2.*
insn.s +78 +0xe4.*
insn.s +79 +0xe8.*
insn.s +80 +0xee.*
insn.s +81 +0xf6.*
insn.s +82 +0x100.*
insn.s +83 +0x10c.*
insn.s +84 +0x122.*
insn.s +85 +0x124.*
insn.s +86 +0x128.*
insn.s +87 +0x12e.*
insn.s +88 +0x136.*
insn.s +89 +0x140.*
insn.s +90 +0x14c.*
insn.s +92 +0x162.*
insn.s +93 +0x16c.*
insn.s +94 +0x176.*
insn.s +95 +0x18c.*
insn.s +96 +0x1a2.*
insn.s +97 +0x1b8.*
insn.s +98 +0x1ce.*
insn.s +99 +0x1e4.*
insn.s +101 +0x1fa.*
insn.s +- +0x1fe
#pass
