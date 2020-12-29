#ifndef GRADIENT_H
#define GRADIENT_H

/* header files */
#include <random>
#include <cuComplex.h>

#include "precision_types.h"
#include "Network.h"
#include "Activation.h"

#ifdef __cplusplus
extern "C" {
#endif

void gradient_check(int argc, char **argv);

#ifdef __cplusplus
}
#endif
#endif /* GRADIENT_H */