/* TID parsing for GDB, the GNU debugger.

   Copyright (C) 2015-2024 Free Software Foundation, Inc.

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

#include "tid-parse.h"
#include "inferior.h"
#include "gdbthread.h"
#include <ctype.h>

/* See tid-parse.h.  */

[[noreturn]] void
invalid_thread_id_error (const char *string)
{
  error (_("Invalid thread ID: %s"), string);
}

[[noreturn]] void
invalid_lane_id_error (const char *string)
{
  error (_("Invalid lane ID: %s"), string);
}

/* Wrapper for get_number_trailer that throws an error if we get back
   a negative number.  We'll see a negative value if the number is
   stored in a negative convenience variable (e.g., $minus_one = -1).
   STRING is the parser string to be used in the error message if we
   do get back a negative number.  */

static std::optional<int>
get_non_negative_number_trailer (const char **pp, int trailer,
				 const char *string)
{
  std::optional<int> res = get_number_trailer (pp, trailer);
  if (res.has_value () && *res < 0)
    error (_("negative value: %s"), string);
  return res;
}

/* Parse TIDSTR as a per-inferior thread ID, in either INF_NUM.THR_NUM
   or THR_NUM form, and return a pair, the first item of the pair is
   INF_NUM and the second item is THR_NUM.

   If TIDSTR does not include an INF_NUM component, then the first item in
   the pair will be 0 (which is an invalid inferior number), this indicates
   that TIDSTR references the current inferior.

   This function does not validate the INF_NUM and THR_NUM are actually
   valid numbers, that is, they might reference inferiors or threads that
   don't actually exist; this function just splits the string into its
   component parts.

   If there is an error parsing TIDSTR then this function will raise an
   exception.  */

static std::pair<int, int>
parse_thread_id_1 (const char *tidstr, const char **end,
		   gdb::function_view<void (const char *)> invalid_id,
		   const char *tidstr_error)
{
  const char *p;
  int inf_num;
  const char *dot = strchr (tidstr, '.');
  if (dot != nullptr)
    {
      /* Parse number to the left of the dot.  */
      p = tidstr;
      std::optional<int> res
	= get_non_negative_number_trailer (&p, '.', tidstr_error);
      if (!res.has_value () || *res == 0)
	invalid_id (tidstr_error);
      inf_num = *res;
      p = dot + 1;
    }
  else
    {
      inf_num = 0;
      p = tidstr;
    }

  std::optional<int> res
    = get_non_negative_number_trailer (&p, '.', tidstr_error);
  if (!res.has_value () || *res == 0)
    invalid_id (tidstr_error);
  int thr_num = *res;

  if (end != nullptr)
    *end = p;

  return { inf_num, thr_num };
}

static thread_info *
resolve_thread_id (int inf_num, int thr_num)
{
  inferior *inf;
  bool explicit_inf_id = false;

  if (inf_num != 0)
    {
      inf = find_inferior_id (inf_num);
      if (inf == nullptr)
	error (_("No inferior number '%d'"), inf_num);
      explicit_inf_id = true;
    }
  else
    inf = current_inferior ();

  thread_info *tp = nullptr;
  for (thread_info *it : inf->threads ())
    if (it->per_inf_num == thr_num)
      {
	tp = it;
	break;
      }

  if (tp == nullptr)
    {
      if (show_inferior_qualified_tids () || explicit_inf_id)
	error (_("Unknown thread %d.%d."), inf->num, thr_num);
      else
	error (_("Unknown thread %d."), thr_num);
    }

  return tp;
}

/* See tid-parse.h.  */

struct thread_info *
parse_thread_id (const char *tidstr, const char **end)
{
  const auto [inf_num, thr_num]
    = parse_thread_id_1 (tidstr, end, invalid_thread_id_error, tidstr);
  return resolve_thread_id (inf_num, thr_num);
}

std::pair<std::string, std::string>
split_lane_id (const char *lidstr)
{
  const char *last_dot = strrchr (lidstr, '.');
  if (last_dot != nullptr)
    {
      std::string tidstr (lidstr, last_dot);
      std::string lanestr (last_dot + 1);
      return {tidstr, lanestr};
    }
  else
    return {"", lidstr};
}

static std::array<std::string_view, 3>
split_lane_id_parts (const char *input, const char **endp)
{
  std::string_view sv (input);

  /* Check if there's a space in the input and adjust the
     string_view accordingly.  */
  size_t space_pos = sv.find (' ');
  if (space_pos != std::string_view::npos)
    {
      sv = sv.substr (0, space_pos);
      *endp = input + space_pos;
    }
  else
    *endp = input + sv.size ();

  /* Hold the three parts (filled from right to left).  */
  std::array<std::string_view, 3> parts;

  if (sv.empty ())
    return parts;

  size_t count = 0;
  size_t end = sv.size ();

  /* Find dots from right to left.  */
  for (;;)
    {
      size_t dot_pos = sv.rfind ('.', end - 1);
      if (dot_pos == std::string_view::npos)
	break;

      if (count == 2 || dot_pos == 0 || dot_pos == end - 1)
	invalid_lane_id_error (input);

      parts[2 - count++] = sv.substr (dot_pos + 1, end - dot_pos - 1);
      end = dot_pos;
    }
  /* Capture the last remaining part.  */
  parts[2 - count++] = sv.substr (0, end);

  return parts;
}

void
lane_id_list_parser::init (const char *input)
{
  m_cursor = skip_spaces (input);
}

lane_id_list_parser::lane_id_range
lane_id_list_parser::get_id_range ()
{
  lane_id_range res;
  const char *end;
  auto str_res = split_lane_id_parts (m_cursor, &end);

  for (size_t i = 0; i < 3; i++)
    {
      auto &part = str_res[i];
      if (part.empty ())
	res[i] = missing_part;
      else if (part == "*")
	res[i] = star_part;
      else
	{
	  std::string str (str_res[i]);
	  number_or_range_parser parser (str.c_str ());
	  int range_start = parser.get_number ();
	  int range_end = (parser.in_range ()
			   ? parser.end_value ()
			   : range_start);
	  res[i] = {range_start, range_end};
	}
    }

  m_cursor = skip_spaces (end);

  return res;
}

bool
lane_id_list_parser::finished ()
{
  /* Parsing is finished when at end of string or null string, or we
     are not in a range and not in front of an integer, negative
     integer, convenience var or negative convenience var.  */
  return (*m_cursor == '\0'
	  || !(isdigit (*m_cursor)
	       || *m_cursor == '$'
	       || *m_cursor == '*'));
}

static lane_id
parse_lane_id_1 (const char *lidstr, const char **end)
{
  const char *lanenum_str;
  lane_id res;

  const char *last_dot = strrchr (lidstr, '.');
  if (last_dot != nullptr)
    {
      /* Parse [INF.]THR to the left of the dot.  */
      std::string tidstr (lidstr, last_dot);
      const char *tid_end;
      auto pair = parse_thread_id_1 (tidstr.c_str (), &tid_end,
				     invalid_lane_id_error, lidstr);
      if (*tid_end != '\0')
	error (_("whoops"));
      res[0] = std::get<0> (pair);
      res[1] = std::get<1> (pair);
      lanenum_str = last_dot + 1;
    }
  else
    {
      res[0] = 0;
      res[1] = 0;
      lanenum_str = lidstr;
    }

  const char *p = lanenum_str;
  std::optional<int> lanenum = get_non_negative_number_trailer (&p, 0, lidstr);
  /* Note that 0 is valid.  */
  if (!lanenum.has_value ())
    invalid_lane_id_error (lidstr);
  res[2] = *lanenum;

  if (end != nullptr)
    *end = p;

  return res;
}

std::pair<thread_info *, int>
parse_lane_id (const char *lidstr, const char **end)
{
  auto [inf_num, thr_num, lane_num] = parse_lane_id_1 (lidstr, end);

  thread_info *thr;
  if (thr_num == 0)
    {
      /* If the thread number is 0, it means the user didn't specify a
	 thread.  Fill it from current context.  */
      gdb_assert (inf_num == 0);
      thr = inferior_thread ();
    }
  else
    {
      /* resolve_thread_id handles the case of the user not specifying
	 an inferior number either.  */
      thr = resolve_thread_id (inf_num, thr_num);
    }

  return { thr, lane_num };
}

/* See tid-parse.h.  */

bool
is_thread_id (const char *tidstr, const char **end)
{
  try
    {
      (void) parse_thread_id_1 (tidstr, end, invalid_thread_id_error, tidstr);
      return true;
    }
  catch (const gdb_exception_error &)
    {
      return false;
    }
}

/* See tid-parse.h.  */

bool
is_lane_id (const char *lidstr, const char **end)
{
  try
    {
      (void) parse_lane_id_1 (lidstr, end);
      return true;
    }
  catch (const gdb_exception_error &)
    {
      return false;
    }
}

/* See tid-parse.h.  */

tid_range_parser::tid_range_parser (const char *tidlist,
				    int default_inferior)
{
  init (tidlist, default_inferior);
}

/* See tid-parse.h.  */

void
tid_range_parser::init (const char *tidlist, int default_inferior)
{
  m_state = STATE_INFERIOR;
  m_cur_tok = tidlist;
  m_inf_num = 0;
  m_qualified = false;
  m_default_inferior = default_inferior;
}

/* See tid-parse.h.  */

bool
tid_range_parser::finished () const
{
  switch (m_state)
    {
    case STATE_INFERIOR:
      /* Parsing is finished when at end of string or null string,
	 or we are not in a range and not in front of an integer, negative
	 integer, convenience var or negative convenience var.  */
      return (*m_cur_tok == '\0'
	      || !(isdigit (*m_cur_tok)
		   || *m_cur_tok == '$'
		   || *m_cur_tok == '*'));
    case STATE_THREAD_RANGE:
    case STATE_STAR_RANGE:
      return m_range_parser.finished ();
    }

  gdb_assert_not_reached ("unhandled state");
}

/* See tid-parse.h.  */

const char *
tid_range_parser::cur_tok () const
{
  switch (m_state)
    {
    case STATE_INFERIOR:
      return m_cur_tok;
    case STATE_THREAD_RANGE:
    case STATE_STAR_RANGE:
      return m_range_parser.cur_tok ();
    }

  gdb_assert_not_reached ("unhandled state");
}

void
tid_range_parser::skip_range ()
{
  gdb_assert (m_state == STATE_THREAD_RANGE
	      || m_state == STATE_STAR_RANGE);

  m_range_parser.skip_range ();
  init (m_range_parser.cur_tok (), m_default_inferior);
}

/* See tid-parse.h.  */

bool
tid_range_parser::tid_is_qualified () const
{
  return m_qualified;
}

/* Helper for tid_range_parser::get_tid and
   tid_range_parser::get_tid_range.  Return the next range if THR_END
   is non-NULL, return a single thread ID otherwise.  */

bool
tid_range_parser::get_tid_or_range (int *inf_num,
				    int *thr_start, int *thr_end)
{
  if (m_state == STATE_INFERIOR)
    {
      const char *p;
      const char *space;

      space = skip_to_space (m_cur_tok);

      p = m_cur_tok;
      while (p < space && *p != '.')
	p++;
      if (p < space)
	{
	  const char *dot = p;

	  /* Parse number to the left of the dot.  */
	  p = m_cur_tok;
	  std::optional<int> res
	    = get_non_negative_number_trailer (&p, '.', m_cur_tok);
	  if (!res.has_value () || *res == 0)
	    return 0;
	  m_inf_num = *res;

	  m_qualified = true;
	  p = dot + 1;

	  if (isspace (*p))
	    return false;
	}
      else
	{
	  m_inf_num = m_default_inferior;
	  m_qualified = false;
	  p = m_cur_tok;
	}

      m_range_parser.init (p);
      if (p[0] == '*' && (p[1] == '\0' || isspace (p[1])))
	{
	  /* Setup the number range parser to return numbers in the
	     whole [1,INT_MAX] range.  */
	  m_range_parser.setup_range (1, INT_MAX, skip_spaces (p + 1));
	  m_state = STATE_STAR_RANGE;
	}
      else
	m_state = STATE_THREAD_RANGE;
    }

  *inf_num = m_inf_num;
  *thr_start = m_range_parser.get_number ();
  if (*thr_start < 0)
    error (_("negative value: %s"), m_cur_tok);
  if (*thr_start == 0)
    {
      m_state = STATE_INFERIOR;
      return false;
    }

  /* If we successfully parsed a thread number or finished parsing a
     thread range, switch back to assuming the next TID is
     inferior-qualified.  */
  if (!m_range_parser.in_range ())
    {
      m_state = STATE_INFERIOR;
      m_cur_tok = m_range_parser.cur_tok ();

      if (thr_end != NULL)
	*thr_end = *thr_start;
    }

  /* If we're midway through a range, and the caller wants the end
     value, return it and skip to the end of the range.  */
  if (thr_end != NULL
      && (m_state == STATE_THREAD_RANGE
	  || m_state == STATE_STAR_RANGE))
    {
      *thr_end = m_range_parser.end_value ();

      skip_range ();
    }

  return (*inf_num != 0 && *thr_start != 0);
}

/* See tid-parse.h.  */

bool
tid_range_parser::get_tid_range (int *inf_num,
				 int *thr_start, int *thr_end)
{
  gdb_assert (inf_num != NULL && thr_start != NULL && thr_end != NULL);

  return get_tid_or_range (inf_num, thr_start, thr_end);
}

/* See tid-parse.h.  */

bool
tid_range_parser::get_tid (int *inf_num, int *thr_num)
{
  gdb_assert (inf_num != NULL && thr_num != NULL);

  return get_tid_or_range (inf_num, thr_num, NULL);
}

/* See tid-parse.h.  */

bool
tid_range_parser::in_star_range () const
{
  return m_state == STATE_STAR_RANGE;
}

bool
tid_range_parser::in_thread_range () const
{
  return m_state == STATE_THREAD_RANGE;
}

/* See tid-parse.h.  */

int
tid_is_in_list (const char *list, int default_inferior,
		int inf_num, int thr_num)
{
  if (list == NULL || *list == '\0')
    return 1;

  tid_range_parser parser (list, default_inferior);
  if (parser.finished ())
    invalid_thread_id_error (parser.cur_tok ());
  while (!parser.finished ())
    {
      int tmp_inf, tmp_thr_start, tmp_thr_end;

      if (!parser.get_tid_range (&tmp_inf, &tmp_thr_start, &tmp_thr_end))
	invalid_thread_id_error (parser.cur_tok ());
      if (tmp_inf == inf_num
	  && tmp_thr_start <= thr_num && thr_num <= tmp_thr_end)
	return 1;
    }
  return 0;
}
