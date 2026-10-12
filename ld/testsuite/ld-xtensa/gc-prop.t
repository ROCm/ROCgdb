ENTRY(_start)
SECTIONS
{
  .text 0x1000 : { *(.literal .literal.* .text .text.*) }
  .xt.prop 0 : { *(.xt.prop .xt.prop.* .gnu.linkonce.prop.*) }
  .xt.lit 0 : { *(.xt.lit .xt.lit.* .gnu.linkonce.p.*) }
}
