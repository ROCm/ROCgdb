.L1:
  # call .L1, r1(ra) temp register, r1(ra) return register.
  call36 .L1
  pcaddu18i $r1, %call36(.L1)
  jirl	    $r1, $r1, 0
  # tail .L1, r12(t0) temp register, r0(zero) return register.
  tail36 $r12, .L1
  pcaddu18i $r12, %call36(.L1)
  jirl	    $r0, $r12, 0
