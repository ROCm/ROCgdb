/* Copyright (C) 2017-2026 Free Software Foundation, Inc.

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

#ifndef GDBSUPPORT_SPAN_H
#define GDBSUPPORT_SPAN_H

#include "traits.h"
#include <algorithm>
#include <type_traits>
#include "gdbsupport/gdb_assert.h"

/* A span is an abstraction that provides a non-owning view
   over a sequence of contiguous objects.

   A way to put it is that span is to std::vector (and
   std::array and built-in arrays with rank==1) like std::string_view
   is to std::string.

   The main intent of span is to use it as function input
   parameter type, making it possible to pass in any sequence of
   contiguous objects, irrespective of whether the objects live on the
   stack or heap and what actual container owns them.  Implicit
   construction from the element type is supported too, making it easy
   to call functions that expect an array of elements when you only
   have one element (usually on the stack).  For example:

    struct A { .... };
    void function (gdb::span<A> as);

    std::vector<A> std_vec = ...;
    std::array<A, N> std_array = ...;
    A array[] = {...};
    A elem;

    function (std_vec);
    function (std_array);
    function (array);
    function (elem);

   Views can be either mutable or const.  A const view is simply
   created by specifying a const T as span template parameter,
   in which case operator[] of non-const span objects ends up
   returning const references.  Making the span itself const is
   analogous to making a pointer itself be const.  I.e., disables
   re-seating the view/pointer.

   Since span objects are small (pointer plus size), and
   designed to be trivially copyable, they should generally be passed
   around by value.

   You can find unit tests covering the whole API in
   unittests/span-selftests.c.  */

namespace gdb {

template <typename T>
class span
{
  /* True iff decayed T is the same as decayed U.  E.g., we want to
     say that 'T&' is the same as 'const T'.  */
  template <typename U>
  using IsDecayedT = typename std::is_same<typename std::decay<T>::type,
					   typename std::decay<U>::type>;

  /* True iff decayed T is the same as decayed U, and 'U *' is
     implicitly convertible to 'T *'.  This is a requirement for
     several methods.  */
  template <typename U>
  using DecayedConvertible = gdb::And<IsDecayedT<U>,
				      std::is_convertible<U *, T *>>;

public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = size_t;
  using const_iterator = const T *;
  using iterator = T *;

  /* Default construction creates an empty view.  */
  constexpr span () noexcept
    : m_array (nullptr), m_size (0)
  {}

  /* Create a span over a single object of the type of an
     span element.  The created view as size==1.  This is
     templated on U to allow constructing a span<const T> over a
     (non-const) T.  The "convertible" requirement makes sure that you
     can't create a span<T> over a const T.  */
  template<typename U,
	   typename = Requires<DecayedConvertible<U>>>
  constexpr span (U &elem) noexcept
    : m_array (&elem), m_size (1)
  {}

  /* Same as above, for rvalue references.  */
  template<typename U,
	   typename = Requires<DecayedConvertible<U>>>
  constexpr span (U &&elem) noexcept
    : m_array (&elem), m_size (1)
  {}

  /* Create a span from a pointer to an array and an element
     count.  */
  template<typename U,
	   typename = Requires<DecayedConvertible<U>>>
  constexpr span (U *array, size_t size) noexcept
    : m_array (array), m_size (size)
  {}

  /* Create a span from a range.  This is templated on both U
     an V to allow passing in a mix of 'const T *' and 'T *'.  */
  template<typename U, typename V,
	   typename = Requires<DecayedConvertible<U>>,
	   typename = Requires<DecayedConvertible<V>>>
  constexpr span (U *begin, V *end) noexcept
    : m_array (begin), m_size (end - begin)
  {}

  /* Create a span from an array.  */
  template<typename U, size_t Size,
	   typename = Requires<DecayedConvertible<U>>>
  constexpr span (U (&array)[Size]) noexcept
    : m_array (array), m_size (Size)
  {}

  /* Create a span from a contiguous container.  E.g.,
     std::vector and std::array.  */
  template<typename Container,
	   typename = Requires<gdb::Not<IsDecayedT<Container>>>,
	   typename
	     = Requires<DecayedConvertible
			<typename std::remove_pointer
			 <decltype (std::declval<Container> ().data ())
			 >::type>>,
	   typename
	     = Requires<std::is_convertible
			<decltype (std::declval<Container> ().size ()),
			 size_type>>>
  constexpr span (Container &&c) noexcept
    : m_array (c.data ()), m_size (c.size ())
  {}

  /* Observer methods.  */
  constexpr T *data () noexcept { return m_array; }
  constexpr const T *data () const noexcept { return m_array; }

  constexpr iterator begin () const noexcept { return m_array; }
  constexpr const_iterator cbegin () const noexcept { return m_array; }

  constexpr iterator end () const noexcept { return m_array + m_size; }
  constexpr const_iterator cend () const noexcept { return m_array + m_size; }

  constexpr reference operator[] (size_t index) noexcept
  {
#if defined(_GLIBCXX_DEBUG)
    gdb_assert (index < m_size);
#endif
    return m_array[index];
  }
  constexpr const_reference operator[] (size_t index) const noexcept
  {
#if defined(_GLIBCXX_DEBUG)
    gdb_assert (index < m_size);
#endif
    return m_array[index];
  }

  constexpr size_type size () const noexcept { return m_size; }
  constexpr bool empty () const noexcept { return m_size == 0; }

  /* Slice a span.  */

  /* Return a new span over SIZE elements starting at START.  */
  [[nodiscard]]
  constexpr span<T> slice (size_type start, size_type size) const noexcept
  {
#if defined(_GLIBCXX_DEBUG)
    gdb_assert (start + size <= m_size);
#endif
    return {m_array + start, size};
  }

  /* Return a new span over all the elements after START,
     inclusive.  */
  [[nodiscard]]
  constexpr span<T> slice (size_type start) const noexcept
  {
#if defined(_GLIBCXX_DEBUG)
    gdb_assert (start <= m_size);
#endif
    return {m_array + start, size () - start};
  }

private:
  T *m_array;
  size_type m_size;
};

/* Copy the contents referenced by the span SRC to the span DEST.

   The two spans must have the same length.  */

template <typename U, typename T>
void copy (gdb::span<U> src, gdb::span<T> dest)
{
  gdb_assert (dest.size () == src.size ());
  if (dest.data () < src.data ())
    std::copy (src.begin (), src.end (), dest.begin ());
  else if (dest.data () > src.data ())
    std::copy_backward (src.begin (), src.end (), dest.end ());
}

/* Compare LHS and RHS for (deep) equality.  That is, whether LHS and
   RHS have the same sizes, and whether each pair of elements of LHS
   and RHS at the same position compares equal.  */

template <typename T>
bool
operator== (const gdb::span<T> &lhs, const gdb::span<T> &rhs)
{
  if (lhs.size () != rhs.size ())
    return false;

  for (size_t i = 0; i < lhs.size (); i++)
    if (!(lhs[i] == rhs[i]))
      return false;

  return true;
}

/* Compare two spans for inequality.  */

template <typename T>
bool
operator!= (const gdb::span<T> &lhs, const gdb::span<T> &rhs)
{
  return !(lhs == rhs);
}

/* Create a span from a pointer to an array and an element
   count.

   This is useful as alternative to constructing a span using
   brace initialization when the size variable you have handy is of
   signed type, since otherwise without an explicit cast the code
   would be ill-formed.

   For example, with:

     extern void foo (int, int, gdb::span<value *>);

     value *args[2];
     int nargs;
     foo (1, 2, {values, nargs});

   You'd get:

     source.c:10: error: narrowing conversion of ‘nargs’ from ‘int’ to
     ‘size_t {aka long unsigned int}’ inside { } [-Werror=narrowing]

   You could fix it by writing the somewhat distracting explicit cast:

     foo (1, 2, {values, (size_t) nargs});

   Or by instantiating a span explicitly:

     foo (1, 2, gdb::span<value *>(values, nargs));

   Or, better, using make_span, which has the advantage of
   inferring the span element's type:

     foo (1, 2, gdb::make_span (values, nargs));
*/

template<typename U>
constexpr inline span<U>
make_array_view (U *array, size_t size) noexcept
{
  return {array, size};
}

/* Create a span from an array.  */

template <typename U, std::size_t Size>
constexpr inline span<U>
make_array_view (U (&array)[Size])
{
  return {array};
}

} /* namespace gdb */

#endif /* GDBSUPPORT_SPAN_H */
