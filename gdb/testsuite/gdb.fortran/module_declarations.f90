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

module mod1
  implicit none
  integer :: var_i = 1
  integer :: global_var = 100
end module mod1

module mod2
  implicit none
  integer :: var_i = 2
end module mod2

subroutine sub1
  use mod1, var_i_alias=>var_i
  use mod2
  implicit none
  var_i_alias = 3
  var_i = 4
end subroutine  ! bp-sub1

subroutine sub2
  use mod1
  use mod2, var_i_alias=>var_i
  implicit none
  var_i_alias = 23
  var_i = 25
end subroutine ! bp-sub2

subroutine sub3
  use mod1, var_i_from_mod1=>var_i
  use mod2, var_i_from_mod2=>var_i
  implicit none
  var_i_from_mod1 = 31
  var_i_from_mod2 = 32
end subroutine ! bp-sub3

program main
  use mod1, global_alias=>global_var, global_var_i_from_mod1=>var_i
  implicit none
  global_alias = 200
  global_var_i_from_mod1 = 300
  call sub1  ! bp-main
  call sub2
  call sub3
end
