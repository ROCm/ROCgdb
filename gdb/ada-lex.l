/* FLEX lexer for Ada expressions, for GDB. -*- c++ -*-
   Copyright (C) 1994-2026 Free Software Foundation, Inc.

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

/*----------------------------------------------------------------------*/

/* The converted version of this file is to be included in ada-exp-parser.y, */
/* the Ada parser for gdb.  The function yylex obtains characters from */
/* the global pointer lexptr.  It returns a syntactic category for */
/* each successive token and places a semantic value into yylval */
/* (ada-lval), defined by the parser.   */

DIG	[0-9]
NUM10	({DIG}({DIG}|_)*)
HEXDIG	[0-9a-f]
NUM16	({HEXDIG}({HEXDIG}|_)*)
OCTDIG	[0-7]
LETTER	[a-z_]
ID	({LETTER}({LETTER}|{DIG}|[\x80-\xff])*|"<"{LETTER}({LETTER}|{DIG})*">")
WHITE	[ \t\n]
TICK	("'"{WHITE}*)
GRAPHIC [a-z0-9 #&'()*+,-./:;<>=_|!$%?@\[\]\\^`{}~]
OPER    ([-+*/=<>&]|"<="|">="|"**"|"/="|"and"|"or"|"xor"|"not"|"mod"|"rem"|"abs")

EXP	(e[+-]{NUM10})
POSEXP  (e"+"?{NUM10})

/* This must agree with COMPLETION_CHAR below.  See the comment there
   for the explanation.  */
COMPLETE "\001"
NOT_COMPLETE [^\001]

%{

#include "diagnostics.h"

/* Some old versions of flex (2.5.x) generate code that uses the "register"
   keyword, which compilers warn about, because it is not allowed in ISO
   C++17.  */
DIAGNOSTIC_PUSH
DIAGNOSTIC_IGNORE_REGISTER

#define NUMERAL_WIDTH 256
#define LONGEST_SIGN ((ULONGEST) 1 << (sizeof(LONGEST) * HOST_CHAR_BIT - 1))

#undef YY_DECL
#define YY_DECL static int yylex ( void )

/* Flex generates a static function "input" which is not used.
   Defining YY_NO_INPUT comments it out.  */
#define YY_NO_INPUT

#undef YY_INPUT
#define YY_INPUT(BUF, RESULT, MAX_SIZE)					\
  if ( *pstate->lexptr == '\000' )					\
    {									\
      if (pstate->parse_completion && !ada_parser->returned_complete)	\
	{								\
	  ada_parser->returned_complete = true;				\
	  *(BUF) = COMPLETE_CHAR;					\
	  (RESULT) = 1;							\
	}								\
      else								\
	(RESULT) = YY_NULL;						\
    }									\
  else									\
    {									\
      *(BUF) = *pstate->lexptr == COMPLETE_CHAR ? ' ' : *pstate->lexptr; \
      (RESULT) = 1;							\
      pstate->lexptr += 1;						\
    }

%}

%option case-insensitive interactive nodefault noyywrap

%s BEFORE_QUAL_QUOTE

%%

{WHITE}		 { }

"--".*		 { yyterminate(); }

{NUM10}{POSEXP}  {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   char *e_ptr = strrchr (numbuf, 'e');
		   *e_ptr = '\0';
		   return processInt (pstate, nullptr, numbuf, e_ptr + 1);
		 }

{NUM10}          {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   return processInt (pstate, NULL, numbuf, NULL);
		 }

{NUM10}"#"{HEXDIG}({HEXDIG}|_)*"#"{POSEXP} {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   char *e_ptr = strrchr (numbuf, 'e');
		   *e_ptr = '\0';
		   return processInt (pstate, numbuf,
				      strchr (numbuf, '#') + 1,
				      e_ptr + 1);
		 }

	/* The "llf" is a gdb extension to allow a floating-point
	   constant to be written in some other base.  The
	   floating-point number is formed by reinterpreting the
	   bytes, allowing direct control over the bits.  */
{NUM10}(l{0,2}f)?"#"{HEXDIG}({HEXDIG}|_)*"#" {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   return processInt (pstate, numbuf, strchr (numbuf, '#') + 1,
				      NULL);
		 }

"0x"{HEXDIG}+	{
		  char numbuf[NUMERAL_WIDTH];
		  canonicalizeNumeral (numbuf, yytext+2);
		  return processInt (pstate, "16#", numbuf, NULL);
		}


{NUM10}"."{NUM10}{EXP} {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   return processReal (pstate, numbuf);
		}

{NUM10}"."{NUM10} {
		   char numbuf[NUMERAL_WIDTH];
		   canonicalizeNumeral (numbuf, yytext);
		   return processReal (pstate, numbuf);
		}

{NUM10}"#"{NUM16}"."{NUM16}"#"{EXP} {
		   error (_("Based real literals not implemented yet."));
		}

{NUM10}"#"{NUM16}"."{NUM16}"#" {
		   error (_("Based real literals not implemented yet."));
		}

<INITIAL>"'"({GRAPHIC}|\")"'" {
		   yylval.typed_char.val = yytext[1];
		   yylval.typed_char.type = type_for_char (pstate, yytext[1]);
		   return CHARLIT;
		}

<INITIAL>"'[\""{HEXDIG}{2,}"\"]'"   {
		   ULONGEST v = strtoulst (yytext+3, nullptr, 16);
		   yylval.typed_char.val = v;
		   yylval.typed_char.type = type_for_char (pstate, v);
		   return CHARLIT;
		}

	/* Note that we don't handle bracket sequences of more than 2
	   digits here.  Currently there's no support for wide or
	   wide-wide strings.  */
\"({GRAPHIC}|"[\""({HEXDIG}{2,}|\")"\"]")*\"   {
		   yylval.sval = processString (yytext+1, yyleng-2);
		   return STRING;
		}

\"              {
		   error (_("ill-formed or non-terminated string literal"));
		}


if		{
		  rewind_to_char ('i');
		  return 0;
		}

task            {
		  rewind_to_char ('t');
		  return 0;
		}

thread{WHITE}+{DIG} {
		  /* This keyword signals the end of the expression and
		     will be processed separately.  */
		  rewind_to_char ('t');
		  return 0;
		}

	/* ADA KEYWORDS */

abs		{ return ABS; }
and		{ return _AND_; }
delta		{ return DELTA; }
else		{ return ELSE; }
for		{ return FOR; }
in		{ return IN; }
mod		{ return MOD; }
new		{ return NEW; }
not		{ return NOT; }
null		{ return NULL_PTR; }
or		{ return OR; }
others          { return OTHERS; }
rem		{ return REM; }
then		{ return THEN; }
with		{ return WITH; }
xor		{ return XOR; }

	/* BOOLEAN "KEYWORDS" */

 /* True and False are not keywords in Ada, but rather enumeration constants.
    However, the boolean type is no longer represented as an enum, so True
    and False are no longer defined in symbol tables.  We compromise by
    making them keywords (when bare). */

true		{ return TRUEKEYWORD; }
false		{ return FALSEKEYWORD; }

	/* ATTRIBUTES */

{TICK}([a-z][a-z_]*)?{COMPLETE}? { BEGIN INITIAL; return processAttribute (yytext); }

	/* PUNCTUATION */

"=>"		{ return ARROW; }
".."		{ return DOTDOT; }
"**"		{ return STARSTAR; }
":="		{ return ASSIGN; }
"/="		{ return NOTEQUAL; }
"<="		{ return LEQ; }
">="		{ return GEQ; }

<BEFORE_QUAL_QUOTE>"'"/{NOT_COMPLETE} { BEGIN INITIAL; return '\''; }

[-&*+{}@/:<>=|;\[\]] { return yytext[0]; }

","		{ if (ada_parser->paren_depth == 0 && pstate->comma_terminates)
		    {
		      rewind_to_char (',');
		      return 0;
		    }
		  else
		    return ',';
		}

"("		{ ada_parser->paren_depth += 1; return '('; }
")"		{ if (ada_parser->paren_depth == 0)
		    {
		      rewind_to_char (')');
		      return 0;
		    }
		  else
		    {
		      ada_parser->paren_depth -= 1;
		      return ')';
		    }
		}

"."{WHITE}*{ID}{COMPLETE}? {
		  yylval.sval = processId (yytext+1, yyleng-1);
		  if (yytext[yyleng - 1] == COMPLETE_CHAR)
		    return DOT_COMPLETE;
		  return DOT_ID;
		}

"."{WHITE}*{COMPLETE} {
		  yylval.sval.ptr = "";
		  yylval.sval.length = 0;
		  return DOT_COMPLETE;
		}

{ID}({WHITE}*"."{WHITE}*({ID}|\"{OPER}\"))*(" "*"'"|{COMPLETE})?  {
		  int all_posn = find_dot_all (yytext);

		  if (all_posn == -1 && yytext[yyleng-1] == '\'')
		    {
		      BEGIN BEFORE_QUAL_QUOTE;
		      yyless (yyleng-1);
		    }
		  else if (all_posn >= 0)
		    yyless (all_posn);
		  bool is_completion = yytext[yyleng - 1] == COMPLETE_CHAR;
		  yylval.sval = processId (yytext, yyleng);
		  return is_completion ? NAME_COMPLETE : NAME;
	       }


	/* GDB EXPRESSION CONSTRUCTS  */

"'"[^']+"'"{WHITE}*:: {
		  yyless (yyleng - 2);
		  yylval.sval = processId (yytext, yyleng);
		  return NAME;
		}

"::"            { return COLONCOLON; }

	/* REGISTERS AND GDB CONVENIENCE VARIABLES */

"$"({LETTER}|{DIG}|"$")*  {
		  yylval.sval.ptr = yytext;
		  yylval.sval.length = yyleng;
		  return DOLLAR_VARIABLE;
		}

	/* CATCH-ALL ERROR CASE */

.		{ error (_("Invalid character '%s' in expression."), yytext); }
%%

namespace ada_exp_parser
{

/* See ada-exp-parser.h.  */

void
lexer_init (FILE *inp)
{
  BEGIN INITIAL;
  yyrestart (inp);
}

} /* namespace ada_exp_parser */

/* Dummy definition to suppress warnings about unused static definitions. */
typedef void (*dummy_function) ();
dummy_function ada_flex_use[] =
{
  (dummy_function) yyunput
};

DIAGNOSTIC_POP
