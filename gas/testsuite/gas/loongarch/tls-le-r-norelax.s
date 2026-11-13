/* Test TLS_LE_R relocs with norelax.  */
.text
/* New TLS Insn.  */
lu12i.w $r4,%le_hi20_r(TLSL1)
add.d   $r5,$r5,$r4,%le_add_r(TLSL1)
st.w $r5,$r4,%le_lo12_r(TLSL1)

/* New TLS Insn with addend.  */
lu12i.w $r4,%le_hi20_r(TLSL1 + 0x8)
add.d   $r5,$r5,$r4,%le_add_r(TLSL1 + 0x8)
st.w $r5,$r4,%le_lo12_r(TLSL1 + 0x8)
