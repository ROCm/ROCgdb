/* Split a symbol name.

   Copyright (C) 2022-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include "split-name.h"
#include "cp-support.h"
#include "gdbsupport/enumerate.h"
#include "gdbsupport/selftest.h"

/* See split-name.h.  */

std::vector<std::string_view>
split_name (const char *name, split_style style)
{
  std::vector<std::string_view> result;
  unsigned int previous_len = 0;

  switch (style)
    {
    case split_style::CXX:
      for (unsigned int current_len = cp_find_first_component (name);
	   name[current_len] != '\0';
	   current_len += cp_find_first_component (name + current_len))
	{
	  gdb_assert (name[current_len] == ':');
	  result.emplace_back (&name[previous_len],
			       current_len - previous_len);
	  /* Skip the '::'.  */
	  current_len += 2;
	  previous_len = current_len;
	}
      break;

    case split_style::DOT_STYLE:
      /* D and Go-style names.  */
      for (const char *iter = strchr (name, '.');
	   iter != nullptr;
	   iter = strchr (iter, '.'))
	{
	  result.emplace_back (&name[previous_len],
			       iter - &name[previous_len]);
	  ++iter;
	  previous_len = iter - name;
	}
      break;

    default:
      break;
    }

  result.emplace_back (&name[previous_len]);
  return result;
}

#if GDB_SELF_TEST

namespace selftests {

/* Return a string representation of COMPONENTS, for debugging.  */

static std::string
components_to_string (const std::vector<std::string_view> &components)
{
  std::string str = "{";

  for (auto [i, component] : gdb::ranges::views::enumerate (components))
    {
      if (i > 0)
	str += ",";

      str += " \"";
      str += component;
      str += "\"";
    }

  str += " }";
  return str;
}

/* Check that splitting NAME using STYLE gives EXPECTED.  */

static void
check_split_name (const char *name, split_style style,
		  const std::vector<std::string_view> &expected)
{
  std::vector<std::string_view> result = split_name (name, style);

  if (result != expected)
    debug_printf ("split_name (\"%s\"): got %s, expected %s\n", name,
		  components_to_string (result).c_str (),
		  components_to_string (expected).c_str ());

  SELF_CHECK (result == expected);
}

static void
test_split_name_none ()
{
  /* No splitting is ever done.  */
  check_split_name ("", split_style::NONE, { "" });
  check_split_name ("foo", split_style::NONE, { "foo" });
  check_split_name ("a::b", split_style::NONE, { "a::b" });
  check_split_name ("a.b", split_style::NONE, { "a.b" });
  check_split_name ("a..b", split_style::NONE, { "a..b" });
  check_split_name ("::", split_style::NONE, { "::" });
  check_split_name (".", split_style::NONE, { "." });
}

static void
test_split_name_cxx ()
{
  /* Simple names.  */
  check_split_name ("", split_style::CXX, { "" });
  check_split_name ("foo", split_style::CXX, { "foo" });
  check_split_name ("a::b", split_style::CXX, { "a", "b" });
  check_split_name ("a::b::c", split_style::CXX, { "a", "b", "c" });

  /* Dots are not separators.  */
  check_split_name ("a.b", split_style::CXX, { "a.b" });
  check_split_name ("a..b::c", split_style::CXX, { "a..b", "c" });

  /* A single colon is not a separator.  */
  check_split_name ("a:b", split_style::CXX, { "a:b" });
  check_split_name ("a:::b", split_style::CXX, { "a", ":b" });

  /* Templates.  */
  check_split_name ("a<b::c>", split_style::CXX, { "a<b::c>" });
  check_split_name ("a<b::c>::d", split_style::CXX, { "a<b::c>", "d" });
  check_split_name ("a::b<c::d, e::f>::g", split_style::CXX,
		    { "a", "b<c::d, e::f>", "g" });
  check_split_name ("a<b::c<d::e>>::f", split_style::CXX,
		    { "a<b::c<d::e>>", "f" });
  check_split_name ("std::vector<int, std::allocator<int> >::size",
		    split_style::CXX,
		    { "std", "vector<int, std::allocator<int> >", "size" });

  /* Parentheses: function parameters, function-local names,
     anonymous namespaces.  */
  check_split_name ("a::b(c::d)", split_style::CXX, { "a", "b(c::d)" });
  check_split_name ("a::b(c::d) const", split_style::CXX,
		    { "a", "b(c::d) const" });
  check_split_name ("foo(int)::bar", split_style::CXX,
		    { "foo(int)", "bar" });
  check_split_name ("(anonymous namespace)::foo", split_style::CXX,
		    { "(anonymous namespace)", "foo" });
  check_split_name ("a::(anonymous namespace)::foo", split_style::CXX,
		    { "a", "(anonymous namespace)", "foo" });

  /* Operators, which can contain characters that would otherwise
     be treated specially.  */
  check_split_name ("A::operator<", split_style::CXX,
		    { "A", "operator<" });
  check_split_name ("A::operator<<", split_style::CXX,
		    { "A", "operator<<" });
  check_split_name ("A::operator>", split_style::CXX,
		    { "A", "operator>" });
  check_split_name ("A::operator>>", split_style::CXX,
		    { "A", "operator>>" });
  check_split_name ("A::operator->", split_style::CXX,
		    { "A", "operator->" });
  check_split_name ("A::operator()", split_style::CXX,
		    { "A", "operator()" });
  check_split_name ("A::operator< <int>", split_style::CXX,
		    { "A", "operator< <int>" });
  check_split_name ("A::operator<(A const&, A const&)", split_style::CXX,
		    { "A", "operator<(A const&, A const&)" });

  /* "operator" as part of an identifier is not an operator.  */
  check_split_name ("cooperator::x", split_style::CXX,
		    { "cooperator", "x" });
  check_split_name ("operator_foo::x", split_style::CXX,
		    { "operator_foo", "x" });

  /* Malformed names are returned as a single component.  */
  check_split_name ("a<b::c", split_style::CXX, { "a<b::c" });
  check_split_name ("a(b::c", split_style::CXX, { "a(b::c" });
  check_split_name ("a>b::c", split_style::CXX, { "a>b::c" });
  check_split_name ("a)b::c", split_style::CXX, { "a)b::c" });

  /* Leading, trailing or consecutive separators produce empty
     components.  */
  check_split_name ("::", split_style::CXX, { "", "" });
  check_split_name ("::a", split_style::CXX, { "", "a" });
  check_split_name ("a::", split_style::CXX, { "a", "" });
  check_split_name ("a::::b", split_style::CXX, { "a", "", "b" });
}

static void
test_split_name_dot_style ()
{
  /* Simple names.  */
  check_split_name ("", split_style::DOT_STYLE, { "" });
  check_split_name ("foo", split_style::DOT_STYLE, { "foo" });
  check_split_name ("a.b", split_style::DOT_STYLE, { "a", "b" });
  check_split_name ("a.b.c", split_style::DOT_STYLE, { "a", "b", "c" });
  check_split_name ("pkg.sub.func", split_style::DOT_STYLE,
		    { "pkg", "sub", "func" });

  /* "::" is not a separator.  */
  check_split_name ("a::b", split_style::DOT_STYLE, { "a::b" });
  check_split_name ("a::b.c", split_style::DOT_STYLE, { "a::b", "c" });

  /* Go method names.  */
  check_split_name ("main.(*T).Method", split_style::DOT_STYLE,
		    { "main", "(*T)", "Method" });

  /* Ada operator names.  */
  check_split_name ("pkg.\"+\"", split_style::DOT_STYLE,
		    { "pkg", "\"+\"" });

  /* Leading, trailing or consecutive dots produce empty components.  */
  check_split_name (".", split_style::DOT_STYLE, { "", "" });
  check_split_name ("..", split_style::DOT_STYLE, { "", "", "" });
  check_split_name (".a", split_style::DOT_STYLE, { "", "a" });
  check_split_name ("a.", split_style::DOT_STYLE, { "a", "" });
  check_split_name ("a..b", split_style::DOT_STYLE, { "a", "", "b" });
  check_split_name (".a.b", split_style::DOT_STYLE, { "", "a", "b" });
  check_split_name ("a.b.", split_style::DOT_STYLE, { "a", "b", "" });
  check_split_name ("a.b..c", split_style::DOT_STYLE,
		    { "a", "b", "", "c" });
  check_split_name ("a..b.c", split_style::DOT_STYLE,
		    { "a", "", "b", "c" });
  check_split_name ("a...b", split_style::DOT_STYLE, { "a", "", "", "b" });
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

INIT_GDB_FILE (split_name)
{
#if GDB_SELF_TEST
  selftests::register_test ("split_name_none",
			    selftests::test_split_name_none);
  selftests::register_test ("split_name_cxx",
			    selftests::test_split_name_cxx);
  selftests::register_test ("split_name_dot_style",
			    selftests::test_split_name_dot_style);
#endif
}
