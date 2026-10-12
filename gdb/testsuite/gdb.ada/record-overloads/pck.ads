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

package Pck is

   type Rec_1 is record
      X : Integer;
   end record;

   type Rec_2 is record
      X : Integer;
   end record;

   function Oload (C : Rec_1) return Integer;
   function Oload (C : Rec_2) return Integer;

   type Rec_Base is tagged record
      X : Integer;
   end record;

   function Oload (C : Rec_Base) return Integer;

   type Rec_Derived is new Rec_Base with null record;

   function Oload (C : Rec_Derived) return Integer;

   type Rec_Dyn (Cond : Boolean := True) is record
      case Cond is
         when True =>
            TV : Integer;
         when False =>
            FV : Integer;
      end case;
   end record;

   function Oload (C : Rec_Dyn) return Integer;

end Pck;
