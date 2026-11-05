.text
a:
  beq	$a0, $a1, undef_func
  beqz	$a0, undef_func
  b	undef_func
  bl	undef_func
  call  undef_func
