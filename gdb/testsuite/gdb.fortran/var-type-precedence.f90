! Copyright 2026 Free Software Foundation, Inc.
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

! Source code for var-type-precedence test to verify that local Fortran
! variables take precedence over conflicting C type names from shared
! libraries.

program test
  use iso_c_binding
  implicit none

  interface
    subroutine fortran_var_type_order_test () bind (C)
    end subroutine fortran_var_type_order_test
  end interface

  ! Declare variables with names that conflict with types in C library.
  integer, dimension (-2:2) :: type_shadowing_var

  ! Call C library function to ensure it's linked.
  call fortran_var_type_order_test ()

  type_shadowing_var = 1

  print *, "" ! break-here
  print *, type_shadowing_var

end program test
