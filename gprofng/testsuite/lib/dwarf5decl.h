/* Header for dwarf5.exp: a function that is inlined into hot_function,
   so that the line table of the test program refers to a second file.  */
#ifndef DWARF5DECL_H
#define DWARF5DECL_H

static inline long
step (long v)
{
  return v * 3 + 1;
}

#endif
