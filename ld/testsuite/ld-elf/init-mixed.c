#include "config.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_INITFINI_ARRAY
static int order;

static void
init1005 ()
{
  if (order != 0)
    abort ();
  order = 1;
}
void (*const init_array1005[]) ()
  __attribute__ ((section (".init_array.01005"), aligned (sizeof (void *))))
  = { init1005 };
static void
fini1005 ()
{
  if (order != 1)
    abort ();
}
void (*const fini_array1005[]) ()
  __attribute__ ((section (".fini_array.01005"), aligned (sizeof (void *))))
  = { fini1005 };

static void
ctor1007a ()
{
  if (order != 1)
    abort ();
  order = 2;
}
static void
ctor1007b ()
{
  if (order != 2)
    abort ();
  order = 3;
}
void (*const ctors1007[]) ()
  __attribute__ ((section (".ctors.64528"), aligned (sizeof (void *))))
  = { ctor1007b, ctor1007a };
static void
dtor1007a ()
{
  if (order != 2)
    abort ();
  order = 1;
}
static void
dtor1007b ()
{
  if (order != 3)
    abort ();
  order = 2;
}
void (*const dtors1007[]) ()
  __attribute__ ((section (".dtors.64528"), aligned (sizeof (void *))))
  = { dtor1007b, dtor1007a };

static void
init65530 ()
{
  if (order != 3)
    abort ();
  order = 4;
}
void (*const init_array65530[]) ()
  __attribute__ ((section (".init_array.65530"), aligned (sizeof (void *))))
  = { init65530 };
static void
fini65530 ()
{
  if (order != 4)
    abort ();
  order = 3;
}
void (*const fini_array65530[]) ()
  __attribute__ ((section (".fini_array.65530"), aligned (sizeof (void *))))
  = { fini65530 };

static void
ctor65535a ()
{
  if (order != 4)
    abort ();
  order = 5;
}
static void
ctor65535b ()
{
  if (order != 5)
    abort ();
  order = 6;
}
void (*const ctors65535[]) ()
  __attribute__ ((section (".ctors"), aligned (sizeof (void *))))
  = { ctor65535b, ctor65535a };
static void
dtor65535b ()
{
  if (order != 6)
    abort ();
  order = 5;
}
static void
dtor65535a ()
{
  if (order != 5)
    abort ();
  order = 4;
}
void (*const dtors65535[]) ()
  __attribute__ ((section (".dtors"), aligned (sizeof (void *))))
  = { dtor65535b, dtor65535a };
#endif

int
main ()
{
  printf ("OK\n");
  return 0;
}
