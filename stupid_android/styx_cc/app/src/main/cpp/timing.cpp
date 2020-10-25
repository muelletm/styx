#include "timing.h"

#include <time.h>

long currentTimeMillis(void) {
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return (long) (1.0e3 * res.tv_sec + (double) res.tv_nsec / 1.0e6);
}