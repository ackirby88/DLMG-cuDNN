/**
 * File:   DLMG.h
 * Author: akirby
 *
 * Created on April 23, 2020, 1:32 PM
 */

#ifndef DLMG_H
#define DLMG_H

/* data types */
#include "dlmg_types.h"

/* header files */
#include "CudaHelper.h"
#include "mpi_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Contains all data of DLMG.
 */
class DLMG {
  public:
    char multigrid_flag;
    ctx_t ctx;                /**< Context dg4est data */
    multigrid_t multigrid;    /**< Multigrid data */
    external_t *external;     /**< [nlevels]: External solver data */
    partition_info_t partition_info;

    /* ============ */
    /* Constructors */
    /* ============ */
    DLMG(int argc,char **argv);
   ~DLMG();

    /* ======= */
    /* Methods */
    /* ======= */
   int get_global_rank(){return ctx.global_compute_ctx.rank;}
   int get_model_rank(){return ctx.model_compute_ctx.rank;}
   int get_data_rank(){return ctx.data_compute_ctx.rank;}

  private:
    int initialize(int argc,char **argv);
    void initalize_cuda_handles();
    void form_model_groups();
    void form_data_groups();
};

#ifdef __cplusplus
}
#endif
#endif /* DLMG_H */