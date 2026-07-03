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

program test
  implicit none

  type :: arr
    integer, allocatable :: f (:)
  end type

  type(arr), dimension(2) :: keys
  allocate (keys(1)%f(2))
  allocate (keys(2)%f(3))
  keys(1)%f(1) = 1
  keys(1)%f(2) = 2
  keys(2)%f(1) = 3
  keys(2)%f(2) = 4
  keys(2)%f(3) = 5
  print *, keys(1)%f(1)         ! stop
  print *, keys(2)%f(1)

end program test
