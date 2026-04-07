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

procedure Nested is
   type Proc_Access is access procedure (Id: Integer);

   procedure Parent (My_Id : Integer; Call : access procedure (Id: Integer));

   procedure Do_Nothing (Id : Boolean);

   procedure Do_Nothing (Id : Boolean) is
   begin
      null;
   end Do_Nothing;

   procedure Parent (My_Id : Integer; Call : access procedure (Id: Integer)) is
      procedure Inner (Id : Integer);

      Outer_Id : Integer := My_Id;

      procedure Inner (Id : Integer) is
      begin
         Do_Nothing (Id = Outer_Id); -- BREAK
      end Inner;

   begin

      --  This setup ensures that when Inner is reached, the most
      --  recent invocation of Parent will not be the correct one for
      --  the purposes of finding "Outer_Id".
      if Call = null then
         Parent (My_Id + 5, Inner'Access);
      else
         Call (Outer_Id);
      end if;
   end Parent;

begin
   Parent (23, null);
end Nested;
