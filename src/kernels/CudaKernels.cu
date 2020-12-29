/**
 * File:   CudaKernels.cu
 * Author: akirby
 *
 * Created on May 11, 2020, 11:26 AM
 */

/* header files */
#include "CudaKernels.h"

/* ============ */
/* CUDA Kernels */
/* ============ */

/** CUDA kernel: addVec
 *  Each thread adds one element of the vectors.
 */
__global__ void vecAdd(Real *a, Real *b, Real *c, int n){
    /* global thread ID */
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if(id < n) c[id] = a[id] + b[id];
}

/* ===================== */
/* CUDA Kernels Wrappers */
/* ===================== */
void CudaKernels_vecAdd(Real *d_a, Real *d_b, Real *d_c, int n){
    int blockSize = 1024;
    vecAdd<<<RoundUp(n,blockSize),blockSize>>>(d_a, d_b, d_c, n);
}

void CudaKernels_memset(Real *d_a, int val, size_t count){
    cudaMemset(d_a,val,count);
}

void CudaKernels_memset_async(Real *d_a, int val, size_t count, cudaStream_t stream){
    cudaMemsetAsync(d_a,val,count,stream);
}