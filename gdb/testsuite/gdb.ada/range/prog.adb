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

procedure Prog is
   type Classic_Enum is (Alpha, Beta, Gamma, Epsilon);
   subtype Smaller_Enum is Classic_Enum range Beta .. Gamma;

   type A_T is array (Smaller_Enum, Classic_Enum) of Integer;

   SE : Smaller_Enum := Gamma;
   CE : Classic_Enum := Epsilon;

   A : A_T := (others => (others => 0));

begin
   null;                        --  START
end Prog;
