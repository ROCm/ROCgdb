/* Make sure that a foreign-endian dict whose type section contains a
   record whose variable-length data runs off the end of the section is
   rejected at open time.  The only indication that anything is wrong is
   the vlen of the single type record: the record header lies entirely
   within the section, but the vbytes derived from it do not.  Without
   bound checks, endianness conversion read-modify-wrote right off the
   end of the buffer, and consumers then walked off the end as well.  */

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctf-api.h>
#include <ctf.h>

/* Store N into BUF so that reading it back in host endianness yields the
   byteswapped value: the section is foreign-endian on any host.  */

static uint32_t
swapu32 (uint32_t v)
{
  return ((v >> 24) | ((v >> 8) & 0xff00)
	  | ((v << 8) & 0xff0000) | (v << 24));
}

static void
putu16 (unsigned char *buf, uint16_t n)
{
  uint16_t v = (uint16_t) ((n >> 8) | (n << 8));
  memcpy (buf, &v, 2);
}

static void
putu32 (unsigned char *buf, uint32_t n)
{
  uint32_t v = swapu32 (n);
  memcpy (buf, &v, 4);
}

int
main (void)
{
  /* 52-byte header, one 12-byte ctf_stype_t (vlen claims 255 argument
     types, i.e. 1KiB of variant data), then a 4-byte strtab -- the type
     section ends right after this one record's header.  */
  unsigned char buf[52 + 12 + 4];
  unsigned char *p = buf;
  ctf_sect_t sect;
  ctf_dict_t *fp;
  ctf_next_t *it = NULL;
  char *foo;
  int err;

  memset (buf, 0, sizeof (buf));

  putu16 (p, CTF_MAGIC);        p += 2;  /* foreign-endian marker */
  *p++ = CTF_VERSION_3;
  *p++ = 0;                              /* flags */
  p += 8 * 4;                            /* parlabel..funcidxoff: zero */
  putu32 (p, 0);              p += 4;    /* cth_varoff */
  putu32 (p, 0);              p += 4;    /* cth_typeoff */
  putu32 (p, 12);             p += 4;    /* cth_stroff: after one record */
  putu32 (p, 4);              p += 4;    /* cth_strlen */

  putu32 (p, 1);          p += 4;         /* ctt_name -> "x" */
  putu32 (p, (CTF_K_FUNCTION << 26) | 255);
  p += 4;                                 /* ctt_info */
  putu32 (p, 4);                          /* ctt_size */

  buf[65] = 'x';                          /* strtab "\0x\0\0" */

  memset (&sect, 0, sizeof (sect));
  sect.cts_name = ".ctf";
  sect.cts_data = (const char *) buf;
  sect.cts_size = sizeof (buf);

  if ((fp = ctf_bufopen (&sect, NULL, NULL, &err)) != NULL)
    {
      fprintf (stderr, "dict with truncated type variable data opened OK?!\n");
      exit (1);
    }

  if (err != ECTF_CORRUPT)
    {
      fprintf (stderr, "dict rejected, but with wrong error: %s\n",
	       ctf_errmsg (err));
      exit (1);
    }

  /* Consume the accumulated error strings.  */
  while ((foo = ctf_errwarning_next (NULL, &it, NULL, NULL)) != NULL)
    free (foo);

  printf ("All OK.\n");
  exit (0);
}
