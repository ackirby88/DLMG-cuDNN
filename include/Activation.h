/**
 * File:   Activation.h
 * Author: akirby
 *
 * Created on April 14, 2020, 3:35 PM
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

/* system header files */
#include <stdio.h>
#include <ctype.h>

/* header files */
#include "CudaHelper.h"

/* Activation Function Declarations */
#define RELU     0x2000001
#define TANH     0x2000002
#define SIGMOID  0x2000003
#define IDENTITY 0x2000004

#ifdef __cplusplus
extern "C" {
#endif

static inline
cudnnActivationMode_t activationSelect(int act){
    switch (act) {
        case RELU:
            return CUDNN_ACTIVATION_RELU;
        case TANH:
            return CUDNN_ACTIVATION_TANH;
        case SIGMOID:
            return CUDNN_ACTIVATION_SIGMOID;
        case IDENTITY:
            return (cudnnActivationMode_t) NULL;
        default:
            return (cudnnActivationMode_t) NULL;
    }
}

#ifdef __cplusplus
}
#endif
#endif /* ACTIVATION_H */