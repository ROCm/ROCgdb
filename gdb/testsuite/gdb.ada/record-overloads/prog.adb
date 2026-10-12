--  Copyright 2026 Free Software Foundation, Inc.
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

procedure Prog is

   R1 : Rec_1 := (X => 23);
   R2 : Rec_2 := (X => 23);
   RB : Rec_Base := (X => 23);
   RD : Rec_Derived := (X => 23);

   RCB : Rec_Base'Class := RB;
   RCD : Rec_Base'Class := RD;

   RCF : Rec_Dyn := (Cond => False, FV => 23);

begin
   null; -- START
end Prog;
