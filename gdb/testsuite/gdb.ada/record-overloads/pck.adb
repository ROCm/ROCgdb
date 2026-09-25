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

package body Pck is

   function Oload (C : Rec_1) return Integer is
   begin
      return 0;
   end Oload;

   function Oload (C : Rec_2) return Integer is
   begin
      return 1;
   end Oload;

   function Oload (C : Rec_Base) return Integer is
   begin
      return 2;
   end Oload;

   function Oload (C : Rec_Derived) return Integer is
   begin
      return 3;
   end Oload;

   function Oload (C : Rec_Dyn) return Integer is
   begin
      return 4;
   end Oload;

end Pck;
