#include "mnist_utils.h"

// Check if system is little-endian
int is_little_endian()
{
    int i = 1;
    char *p = (char *)&i;

    if (p[0] == 1)
        return 0;
    else
        return 1;
}
