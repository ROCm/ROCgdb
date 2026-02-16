Relocation section '\.rela\.dyn' .* contains 1 entry:
.*
.* R_390_GLOB_DAT +0+ foo \+ 0

Relocation section '\.rela\.plt' .* contains 1 entry:
.*
.* R_390_JMP_SLOT +0+ bar \+ 0

Global Offset Table '\.got' contains 5 entries:
.*
.* 0
.* [1-9a-f][0-9a-f]*
.* 0
.* R_390_JMP_SLOT +bar \+ 0
.* R_390_GLOB_DAT +foo \+ 0
