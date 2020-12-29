/**
 * File:   precision_types.h
 * Author: akirby
 *
 * Created on April 23, 2020, 1:34 PM
 */

#ifndef PRECISION_TYPES_H
#define PRECISION_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/* default precision */
#  define Real float

/* ======================== */
/* HALF PRECISION FUNCTIONS */
/* ======================== */
#if defined (HALF_PRECISION)
#  undef  Real
#  define Real float
#  define MPI_DLREAL MPI_FLOAT

/* ========================== */
/* SINGLE PRECISION FUNCTIONS */
/* ========================== */
#elif defined (SINGLE_PRECISION)
#  undef  Real
#  define Real float
#  define MPI_DLREAL MPI_FLOAT

/* ========================== */
/* DOUBLE PRECISION FUNCTIONS */
/* ========================== */
#elif defined (DOUBLE_PRECISION)
#  undef  Real
#  define Real double
#  define MPI_DLREAL MPI_DOUBLE

/* =========================== */
/* COMPLEX PRECISION FUNCTIONS */
/* =========================== */
#elif defined (COMPLEX_PRECISION)
#  include <cuComplex.h>
#  undef  Real
#  define Real cuFloatComplex
#  define MPI_DLREAL MPI_COMPLEX

/* ================================== */
/* DOUBLE COMPLEX PRECISION FUNCTIONS */
/* ================================== */
#elif defined (DOUBLE_COMPLEX_PRECISION)
#  include <cuComplex.h>
#  undef  Real
#  define Real cuDoubleComplex
#  define MPI_DLREAL MPI_DOUBLE_COMPLEX
#endif

#ifndef UINT
#  define UINT unsigned int
#endif

#ifdef __cplusplus
}
#endif
#endif /* PRECISION_TYPES_H */