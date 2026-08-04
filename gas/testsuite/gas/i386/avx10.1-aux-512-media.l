.*: Assembler messages:
.*:29: Error: `vdpphps' is not supported .*
.*:30: Error: `vdpphps' is not supported .*
.*:31: Error: `vdpphps' is not supported .*
.*:32: Error: `vdpphps' is not supported .*
.*:33: Error: `vdpphps' is not supported .*
.*:35: Error:.* `vmpsadbw'.*
.*:36: Error:.* `vmpsadbw'.*
.*:37: Error:.* `vmpsadbw'.*
.*:38: Error:.* `vmpsadbw'.*
.*:39: Error:.* `vmpsadbw'.*
.*:63: Error: `vdpphps' is not supported .*
.*:64: Error: `vdpphps' is not supported .*
.*:65: Error: `vdpphps' is not supported .*
.*:66: Error: `vdpphps' is not supported .*
.*:67: Error: `vdpphps' is not supported .*
.*:69: Error:.* `vmpsadbw'.*
.*:70: Error:.* `vmpsadbw'.*
.*:71: Error:.* `vmpsadbw'.*
.*:72: Error:.* `vmpsadbw'.*
.*:73: Error:.* `vmpsadbw'.*
[ 	]*[0-9]+[ 	]+# Check .*
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25748 50F4[ 	]+>>  vpdpbssd %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25748 51F4[ 	]+>>  vpdpbssds %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25648 50F4[ 	]+>>  vpdpbsud %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25648 51F4[ 	]+>>  vpdpbsuds %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25448 50F4[ 	]+>>  vpdpbuud %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25448 51F4[ 	]+>>  vpdpbuuds %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25648 D2F4[ 	]+>>  vpdpwsud %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25648 D3F4[ 	]+>>  vpdpwsuds %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25548 D2F4[ 	]+>>  vpdpwusd %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25548 D3F4[ 	]+>>  vpdpwusds %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+>  \.irp .*
[ 	]*[0-9]+[ 	]+\?+ 62F25448 D2F4[ 	]+>>  vpdpwuud %zmm4,%zmm5,%zmm6
#...
[ 	]*[0-9]+[ 	]+\?+ 62F25448 D3F4[ 	]+>>  vpdpwuuds %zmm4,%zmm5,%zmm6
#pass
