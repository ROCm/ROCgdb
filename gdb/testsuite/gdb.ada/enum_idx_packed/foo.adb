--  Copyright 2012-2026 Free Software Foundation, Inc.
--
--  This program is free software; you can redistribute it and/or modify
--  it under the terms of the GNU General Public License as published by
--  the Free Software Foundation; either version 3 of the License, or
--  (at your option) any later version.
--
--  This program is distributed in the hope that it will be useful,
--  but WITHOUT ANY WARRANTY; without even the implied warranty of
--  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--  GNU General Public License for more details.
--
--  You should have received a copy of the GNU General Public License
--  along with this program.  If not, see <http://www.gnu.org/licenses/>.

with Pck; use Pck;

procedure Foo is
   Full : Full_Table := (False, True, False, True, False);
   Primary : Primary_Table := (False, True, False);
   Cold : Cold_Table := (False, True);
   Small : Small_Table := New_Small_Table (Low => Red, High => Green);
   Multi : Multi_Table := New_Multi_Table (Red, Green, Low, Medium);
   Multi_Multi : Multi_Multi_Table := New_Multi_Multi_Table (1, 2, 1, 7, 1, 10);
   Multi_Access : Multi_Dimension_Access
     := new Multi_Dimension'(True => (1, 1, 2, 3, 5),
                             False => (8, 13, 21, 34, 55));

   Confused_Array : Confused_Array_Type := (Red => (0, 1, 2),
                                            Green => (5, 6, 7),
                                            others => (others => 72));

   --  These variables ensure the bounds aren't elided by LLVM.
   Small_First : Color := Small'First;
   Small_Last : Color := Small'Last;
   Multi_First1 : Color := Multi'First (1);
   Multi_Last1 : Color := Multi'Last (1);
   Multi_First2 : Strength := Multi'First (2);
   Multi_Last2 : Strength := Multi'Last (2);

   MM_First1 : Positive := Multi_Multi'First (1);
   MM_Last1 : Positive := Multi_Multi'Last (1);
   MM_First2 : Positive := Multi_Multi'First (2);
   MM_Last2 : Positive := Multi_Multi'Last (2);
   MM_First3 : Positive := Multi_Multi'First (3);
   MM_Last3 : Positive := Multi_Multi'Last (3);

begin
   Do_Nothing (Full'Address);  -- STOP
   Do_Nothing (Primary'Address);
   Do_Nothing (Cold'Address);
   Do_Nothing (Small'Address);
   Do_Nothing (Small_First'Address);
   Do_Nothing (Small_Last'Address);
   Do_Nothing (Multi'Address);
   Do_Nothing (Multi_First1'Address);
   Do_Nothing (Multi_Last1'Address);
   Do_Nothing (Multi_First2'Address);
   Do_Nothing (Multi_Last2'Address);
   Do_Nothing (Multi_Multi'Address);
   Do_Nothing (Multi_Access'Address);
   Do_Nothing (MM_First1'Address);
   Do_Nothing (MM_Last1'Address);
   Do_Nothing (MM_First2'Address);
   Do_Nothing (MM_Last2'Address);
   Do_Nothing (MM_First3'Address);
   Do_Nothing (MM_Last3'Address);
end Foo;
