.section .tdata,"awT",@progbits
.globl tls_var
.type tls_var,@tls_object
tls_var:
    .dword 100
.size tls_var,.-tls_var

.text
.globl get_ld_pcaddi, get_gd_pcaddi, get_desc_pcaddi
.type get_ld_pcaddi,@function
.type get_gd_pcaddi,@function
.type get_desc_pcaddi,@function

get_ld_pcaddi:
    addi.d $sp,$sp,-16
    st.d $ra,$sp,8
    pcaddi $a0, %ld_pcrel_20(tls_var)
    bl %plt(__tls_get_addr)
    ld.d $ra,$sp,8
    addi.d $sp,$sp,16
    jr $ra
.size get_ld_pcaddi,.-get_ld_pcaddi

get_gd_pcaddi:
    addi.d $sp,$sp,-16
    st.d $ra,$sp,8
    pcaddi $a0, %gd_pcrel_20(tls_var)
    bl %plt(__tls_get_addr)
    ld.d $ra,$sp,8
    addi.d $sp,$sp,16
    jr $ra
.size get_gd_pcaddi,.-get_gd_pcaddi

get_desc_pcaddi:
    addi.d $sp,$sp,-16
    st.d $ra,$sp,8
    pcaddi $a0, %desc_pcrel_20(tls_var)
    ld.d   $ra, $a0, %desc_ld(tls_var)
    jirl   $ra, $ra, %desc_call(tls_var)
    add.d  $a0, $a0, $tp
    ld.d $ra,$sp,8
    addi.d $sp,$sp,16
    jr $ra
.size get_desc_pcaddi,.-get_desc_pcaddi
