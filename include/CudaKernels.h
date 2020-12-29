/**
 * File:   CudaKernels.h
 * Author: akirby
 *
 * Created on May 11, 2020, 11:27 AM
 */

#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

/* system header files */

/* header files */
#include "precision_types.h"
#include "math_utilities.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================ */
/* Public Functions */
/* ================ */
void CudaKernels_vecAdd(Real *d_a, Real *d_b, Real *d_c, int n);
void CudaKernels_memset(Real *d_a, int val, size_t count);
void CudaKernels_memset_async(Real *d_a, int val, size_t count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif /* CUDAKERNELS_H */