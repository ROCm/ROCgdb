! Copyright 2026 Free Software Foundation, Inc.
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! This file is part of GDB.
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
! Small Fortran program offloading a loop to the GPU via
! '!$omp target'.

program omp_target_fortran
  implicit none
  integer, parameter :: n = 32
  integer :: a(n), b(n), c(n)
  integer :: i, total

  do i = 1, n
     a(i) = i
     b(i) = 2 * i
     c(i) = 0
  end do

  !$omp target map(to:a, b) map(tofrom:c)
  do i = 1, n
     c(i) = a(i) + b(i)             ! ftarget-line
  end do
  !$omp end target

  total = 0
  do i = 1, n
     total = total + c(i)
  end do

  print *, 'total=', total
end program omp_target_fortran
