
/**
 * File:   gpu_kernel_defines.h
 * Author: akirby
 *
 * Created on May 11, 2020, 12:21 PM
 */

#ifndef GPU_KERNEL_DEFINES_H
#define GPU_KERNEL_DEFINES_H

#if defined (__DLMG_CUDA__)
#  include "CudaKernels.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** double/float data type overload */
#if defined (__DLMG_CUDA__)
#define g_blas_memset CudaKernels_memset
#define g_blas_memset_async CudaKernels_memset_async
//#define g_blas_memset_async(x,y,z,zz) CudaKernels_memset(x,y,z)

/* =================================== */
/* CUDA: HALF PRECISION FUNCTIONS      */
/* =================================== */
#   if defined (HALF_PRECISION)
/*  Level 0: */
#       define DNN_REAL CUDNN_DATA_HALF
/*  Level 1: */
#       define g_blas_copy cublasHcopy
#       define g_blas_scal cublasHscal
#       define g_blas_dot  cublasHdot
#       define g_blas_axpy cublasHaxpy
#       define g_blas_asum cublasHasum
#       define g_blas_amin cublasHamin
#       define g_blas_amax cublasHamax
#       define g_blas_nrm2 cublasHnrm2
/*  Level 2: */
#       define g_blas_gemv cublasHgemv
/*  Level 3: */
#       define g_blas_gemm cublasHgemm
/* ----------------------------------- */


/* =================================== */
/* CUDA: SINGLE PRECISION FUNCTIONS    */
/* =================================== */
#   elif defined (SINGLE_PRECISION)
/*  Level 0: */
#       define DNN_REAL CUDNN_DATA_FLOAT
/*  Level 1: */
#       define g_blas_copy cublasScopy
#       define g_blas_scal cublasSscal
#       define g_blas_dot  cublasSdot
#       define g_blas_axpy cublasSaxpy
#       define g_blas_asum cublasSasum
#       define g_blas_amin cublasSamin
#       define g_blas_amax cublasSamax
#       define g_blas_nrm2 cublasSnrm2
/*  Level 2: */
#       define g_blas_gemv cublasSgemv
/*  Level 3: */
#       define g_blas_gemm cublasSgemm
/* ----------------------------------- */

/* =================================== */
/* CUDA: DOUBLE PRECISION FUNCTIONS    */
/* =================================== */
#   elif defined (DOUBLE_PRECISION)
/*  Level 0: */
#       define DNN_REAL CUDNN_DATA_DOUBLE
/*  Level 1: */
#       define g_blas_copy cublasDcopy
#       define g_blas_scal cublasDscal
#       define g_blas_dot  cublasDdot
#       define g_blas_axpy cublasDaxpy
#       define g_blas_asum cublasDasum
#       define g_blas_amin cublasDamin
#       define g_blas_amax cublasDamax
#       define g_blas_nrm2 cublasDnrm2
/*  Level 2: */
#       define g_blas_gemv cublasDgemv
/*  Level 3: */
#       define g_blas_gemm cublasDgemm
/* ------------------------------------ */

/* =================================== */
/* CUDA: COMPLEX PRECISION FUNCTIONS   */
/* =================================== */
#   elif defined (COMPLEX_PRECISION)
/*  Level 0: */
#       define DNN_REAL CUDNN_DATA_FLOAT
/*  Level 1: */
#       define g_blas_copy cublasCcopy
#       define g_blas_scal cublasCscal
#       define g_blas_dot  cublasCdot
#       define g_blas_axpy cublasCaxpy
#       define g_blas_asum cublasCasum
#       define g_blas_amin cublasCamin
#       define g_blas_amax cublasCamax
#       define g_blas_nrm2 cublasCnrm2
/*  Level 2: */
#       define g_blas_gemv cublasCgemv
/*  Level 3: */
#       define g_blas_gemm cublasCgemm
/* ----------------------------------- */

/* ======================================== */
/* CUDA: DOUBLE COMPLEX PRECISION FUNCTIONS */
/* ======================================== */
#   elif defined (DOUBLECOMPLEX_PRECISION)
/*  Level 0: */
#       define DNN_REAL CUDNN_DATA_DOUBLE
/*  Level 1: */
#       define g_blas_copy cublasZcopy
#       define g_blas_scal cublasZscal
#       define g_blas_dot  cublasZdot
#       define g_blas_axpy cublasZaxpy
#       define g_blas_asum cublasZasum
#       define g_blas_amin cublasZamin
#       define g_blas_amax cublasZamax
#       define g_blas_nrm2 cublasZnrm2
/*  Level 2: */
#       define g_blas_gemv cublasZgemv
/*  Level 3: */
#       define g_blas_gemm cublasZgemm
/* ----------------------------------- */
#   endif /* DATA TYPES */

#elif defined (__DLMG_HIP__)
/* =================================== */
/* HIP: HALF PRECISION FUNCTIONS       */
/* =================================== */
#   if defined (HALF_PRECISION)
/*  Level 0: */
#       define DNN_REAL HIPDNN_DATA_HALF
/*  Level 1: */
#       define g_blas_copy hipblasHcopy
#       define g_blas_scal hipblasHscal
#       define g_blas_dot  hipblasHdot
#       define g_blas_axpy hipblasHaxpy
#       define g_blas_asum hipblasHasum
#       define g_blas_amin hipblasHamin
#       define g_blas_amax hipblasHamax
#       define g_blas_nrm2 hipblasHnrm2
/*  Level 2: */
#       define g_blas_gemv cublasHgemv
/*  Level 3: */
#       define g_blas_gemm cublasHgemm
/* ----------------------------------- */
/* =================================== */
/* HIP: SINGLE PRECISION FUNCTIONS     */
/* =================================== */
#   elif defined (SINGLE_PRECISION)
/*  Level 0: */
#       define DNN_REAL HIPDNN_DATA_FLOAT
/*  Level 1: */
#       define g_blas_copy hipblasScopy
#       define g_blas_scal hipblasSscal
#       define g_blas_dot  hipblasSdot
#       define g_blas_axpy hipblasSaxpy
#       define g_blas_asum hipblasSasum
#       define g_blas_amin hipblasSamin
#       define g_blas_amax hipblasSamax
#       define g_blas_nrm2 hipblasSnrm2
/*  Level 2: */
#       define g_blas_gemv hipblasSgemv
/*  Level 3: */
#       define g_blas_gemm hipblasSgemm
/* ------------------------------------ */

/* =================================== */
/* HIP: DOUBLE PRECISION FUNCTIONS     */
/* =================================== */
#   elif defined (DOUBLE_PRECISION)
/*  Level 0: */
#       define DNN_REAL HIPDNN_DATA_DOUBLE
/*  Level 1: */
#       define g_blas_copy hipblasDcopy
#       define g_blas_scal hipblasDscal
#       define g_blas_dot  hipblasDdot
#       define g_blas_axpy hipblasDaxpy
#       define g_blas_asum hipblasDasum
#       define g_blas_amin hipblasDamin
#       define g_blas_amax hipblasDamax
#       define g_blas_nrm2 hipblasDnrm2
/*  Level 2: */
#       define g_blas_gemv hipblasDgemv
/*  Level 3: */
#       define g_blas_gemm hipblasDgemm
/* ----------------------------------- */

/* =================================== */
/* HIP: COMPLEX PRECISION FUNCTIONS    */
/* =================================== */
#   elif defined (COMPLEX_PRECISION)
/*  Level 0: */
#       define DNN_REAL HIPDNN_DATA_FLOAT
/*  Level 1: */
#       define g_blas_copy hipblasCcopy
#       define g_blas_scal hipblasCscal
#       define g_blas_dot  hipblasCdot
#       define g_blas_axpy hipblasCaxpy
#       define g_blas_asum hipblasCasum
#       define g_blas_amin hipblasCamin
#       define g_blas_amax hipblasCamax
#       define g_blas_nrm2 hipblasCnrm2
/*  Level 2: */
#       define g_blas_gemv hipblasCgemv
/*  Level 3: */
#       define g_blas_gemm hipblasCgemm
/* ------------------------------------ */

/* ======================================= */
/* HIP: DOUBLE COMPLEX PRECISION FUNCTIONS */
/* ======================================= */
#   elif defined (DOUBLECOMPLEX_PRECISION)
/*  Level 0: */
#       define DNN_REAL HIPDNN_DATA_DOUBLE
/*  Level 1: */
#       define g_blas_copy hipblasZcopy
#       define g_blas_scal hipblasZscal
#       define g_blas_dot  hipblasZdot
#       define g_blas_axpy hipblasZaxpy
#       define g_blas_asum hipblasZasum
#       define g_blas_amin hipblasZamin
#       define g_blas_amax hipblasZamax
#       define g_blas_nrm2 hipblasZnrm2
/*  Level 2: */
#       define g_blas_gemv hipblasZgemv
/*  Level 3: */
#       define g_blas_gemm hipblasZgemm
/* ------------------------------------ */
#   endif /* DATA TYPES */
#endif /* DLMG_HIP */

#ifdef __cplusplus
}
#endif
#endif /* GPU_KERNEL_DEFINES_H */