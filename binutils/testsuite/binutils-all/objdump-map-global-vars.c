char      a;
int       b;
long      c;
void     *d;

volatile int e;
const    int f;

enum _enum {
  E1,
  E2,
  E3
};

enum _enum g;

struct _struct {
  char       m1;
  enum _enum m2;
};

struct _struct h;

union _union {
  char        u1;
  const char *u2;
};

union _union i;

struct _node {
  int           value;
  struct _node *next;
};

typedef struct _node node;

node j;

node k[4][2];

char *l[2];

int
main (void)
{
  return 0;
}
