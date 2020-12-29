/**
 * File:   dlmg_types.h
 * Author: akirby
 *
 * Created on April 23, 2020, 2:44 PM
 */

#ifndef DLMG_TYPES_H
#define DLMG_TYPES_H

/* system header files */
#include <mpi.h>
#include <omp.h>
#include <vector>

/* header files */
#include "precision_types.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DEBUG
#  define DEBUG_MESG(x)    x;
#else
#  define DEBUG_MESG(x)
#endif

/**
 * Define maximum number of multigrid levels
 */
#define MAX_LEVELS 20

enum layerType {
    FULLYCONNECTED,
    CONVOLUTION,
    MAXPOOL,
    SOFTMAX
};

/**
 * @brief MPI request related info for MPI_Waitall.
 */
typedef struct {
    int nrequest;
    MPI_Request *request;
}
ghost_exchange_t;

/**
 * @brief Data for solving the right-hand-side of equations including
 *        solution vector, residual vector, and ghost exchange information.
 */
typedef struct {
    ghost_exchange_t *exc;  /**< Ghost communication schedule */
    Real *soln;             /**< Solution vector pointer */
    Real *rhs;              /**< Residual rhs vector pointer */
    Real *src;              /**< Source term vector pointer */
}
rhs_dynamic_t;

/**
 * @brief Contains all MPI communication data.
 */
typedef struct {
    int nsend;              /**< MPI send count */
    int nrecv;              /**< MPI receive count */
    int *send_info;         /**< MPI send info: [0] = send quad id */
    int *recv_info;         /**< MPI receive info */
    int *data_counts;       /**< Number of data counts on each rank */
    int send_info_size;     /**< Number of entries in the mpi send */
    int recv_info_size;     /**< Number of entries in the mpi recv */
    Real *sendbuf;          /**< MPI data communication buffer */
    MPI_Request *request;   /**< MPI request buffer */
}
mpi_t;

typedef struct {
    mpi_t d_mpi;            /**< MPI communication data */
}
external_t;

typedef struct {
    int test_accuracy_interval; /**< Intervale to check test accuracy */
    Real *h_train_data;         /**< Training data host address */
    Real *h_train_labels;       /**< Training labels host address */
    Real *h_test_data;          /**< Testing data host address */
    uint8_t *h_test_labels;     /**< Testing labels host address */

    size_t width;               /**< Data width */
    size_t height;              /**< Data height */
    size_t channels;            /**< Data channels */
    size_t test_size;           /**< Testing sample count */
    size_t train_size;          /**< Training sample count */
}
dataset_t;

typedef struct {
    int nlayers_per_block;
    int nranks_per_model;
    int nbatch_groups;
    char sequential;
    cudaStream_t cudaStream;
}
partition_info_t;

/**
 * @brief Contains all context data of DLMG.
 *        This contains all the other data structures.
 */
typedef struct {
    int ngpus;              /**< Number of GPU devices visible to each rank */
    int rank;               /**< MPI rank within this communicator */
    int nranks;             /**< Number of ranks in this communicator */
    int group_id;           /**< Assigned group id */
    int group_range[1][3];  /**< MPI ranks range for this group: {start, end, stride} */
    MPI_Comm mpi_comm;      /**< MPI communicator */
    MPI_Group mpi_group;    /**< MPI group */
}
compute_ctx_t;

/**
 * @brief Contains all context data of DLMG.
 *        This contains all the other data structures.
 */
typedef struct {
    compute_ctx_t data_compute_ctx;
    compute_ctx_t model_compute_ctx;
    compute_ctx_t global_compute_ctx;
}
ctx_t;

typedef struct {
    int ncyc;       /**< Number of smoothing cycles on this multigrid level */
    int ncyc_post;  /**< Number of post-smoothing cycles on this multigrid level */
}
multigrid_level_t;

typedef struct {
    int cycle_count;                /**< Total multigrid cycles count */
    int scheme;                     /**< Multigrid cycle scheme */
    int ncyc_mg;                    /**< Maximum number of multigrid cycles per step */
    int nsweeps;                    /**< Number of smoothing sweeps on each level */
    int nsweeps_init;               /**< Number of smoothing sweeps on initial step */
    int nlevels;                    /**< Number of multigrid levels */
    int cfactor;                    /**< Coarseing factor */
    Real smooth_fac;                /**< Smoothing update factor (0.2-1.2; [0.8])*/
    multigrid_level_t *mg_levels;   /**< Multigrid data structures for each level */
}
multigrid_t;

#ifdef __cplusplus
}
#endif
#endif /* DLMG_TYPES_H */