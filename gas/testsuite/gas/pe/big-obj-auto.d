#objdump: -h
#name: PE big obj auto-promotion

.*: *file format pe-bigobj-.*

Sections:
#...
80002 +\.data\$a79999  .*
                  CONTENTS, ALLOC, LOAD, DATA
