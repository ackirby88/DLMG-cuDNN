/**
 * File:   math_utilities.h
 * Author: akirby
 *
 * Created on May 11, 2020, 12:11 PM
 */

#ifndef MATH_UTILITIES_H
#define MATH_UTILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

/** Computes ceil(x / y) for integral nonnegative values */
static inline
unsigned int RoundUp(unsigned int nominator,unsigned int denominator){
    return (nominator + denominator - 1) / denominator;
}

#ifdef __cplusplus
}
#endif
#endif /* MATH_UTILITIES_H */