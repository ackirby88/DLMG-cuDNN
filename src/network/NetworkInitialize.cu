/**
 * File:   NetworkInitialize.cu
 * Author: akirby
 *
 * Created on April 23, 2020, 12:37 PM
 */

/* header files */
#include "Network.h"
#include <unistd.h>

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define GLOBAL_LOOP(level) \
    for (int i = 0; i < globalLayers[level].size(); ++i)

#define BLOCK_LOOP(level) \
    for (int i = 0; i < blocks[level].size(); ++i)

#define ROOT if(global_rank == 0)

void partition_layers(int nlayers, int nranks, int rank, int &lower_id, int &upper_id){
    int quo = nlayers/nranks;
    int rem = nlayers%nranks;

    int p = rank;
    lower_id = p*quo + (p < rem ? p : rem);

    p = rank+1;
    upper_id = p*quo + (p < rem ? p : rem) - 1;
}

void Network::add(Layer *layer){
    layer->setLevel(0);
    layer->setGlobalIdx(globalLayers_count++);
    globalLayers[0].push_back(layer);
}

void Network::initialize(int max_batch_size){
    /* ================================================================== */
    /* Partition network given the number of ranks (GPUs) per model:      */
    /* ------------------------------------------------------------       */
    /* The partitioning of a network is determined by splitting the       */
    /* number of blocks evenly across the ranks. Then (over)determining   */
    /* the number of layers to be computed by each rank.                  */
    /*                                                                    */
    /* The model is partitioned such that:                                */
    /*    -- first  k-layers are on model.rank = 0                        */
    /*    -- second k-layers are on model.rank = 1                        */
    /*       ....                                                         */
    /*    -- last remaining layers are on model.rank = nranks_per_model-1 */
    /* ================================================================== */
    compute_ctx_t &global = dlmg.ctx.global_compute_ctx;
    compute_ctx_t &model = dlmg.ctx.model_compute_ctx;
    compute_ctx_t &data = dlmg.ctx.data_compute_ctx;
    partition_info_t &pinfo = dlmg.partition_info;
    multigrid_t &mg = dlmg.multigrid;

    /* set MPI ranks for Network */
    global_nranks = global.nranks;
    model_nranks = model.nranks;
    data_nranks = data.nranks;

    global_rank = global.rank;
    model_rank = model.rank;
    data_rank = data.rank;

    gpu_id = global.rank % global.ngpus;

    /* save synchronous CUDA Handles */
    cudaStreamCreateWithFlags(&syncCudaStream,cudaStreamDefault);

    /* save batch_size */
    batch_size = max_batch_size;

    /* loop all global network layers on Level 0 */
    long int param_count = 0;
    GLOBAL_LOOP(0){
       /* ============================================== *
        * Initialize all layer host data:                *
        * This is done for model parallelism consistency *
        * ============================================== */
        globalLayers[0][i]->layerAllocateHost();
        param_count += globalLayers[0][i]->parametersGetCount();

        /* =============================================================== *
         * Initialize all layer CUDA handles:                              *
         * This model_rank layers are modified in LayerBlock construction. *
         * =============================================================== */
        globalLayers[0][i]->setCudaStream(&syncCudaStream);
    }

    ROOT{
        printf("********* MODEL STATISTICS: Layers = %lu, Parameters: %lu\n",globalLayers[0].size(),param_count);
    }
    //pinfo.nlayers_per_block = globalLayers[0].size() / global_nranks * 2;

    /* check nlayers_per_block */
    if(pinfo.nlayers_per_block < 1) pinfo.nlayers_per_block = globalLayers[0].size();
    if(pinfo.nlayers_per_block > globalLayers[0].size()) pinfo.nlayers_per_block = globalLayers[0].size();

    /* check model ranks count */
    if (model.nranks > globalLayers[0].size()) {
        printf("[DLMG] *** ERROR ***: "
               "Number of model ranks is greater than the number of layers in network! EXITING...\n");

        /* NOTE: the MPI communicators are already
         *       setup so we cant simply just change
         *       the number of model ranks.
         */
        exit(EXIT_FAILURE);
    }

    /* ======================== */
    /* Partition Network Layers */
    /* ======================== */
    asyncCudaStreams = (cudaStream_t **) malloc(globalLayers[0].size()*sizeof(cudaStream_t *));
    partition_layers(globalLayers[0].size(),model.nranks,model.rank,s_layer[0],e_layer[0]);

    DEBUG_MESG(
        printf("Model rank: %d, start layer: %d, end layer: %d, stride: %d\n",
                model.rank,s_layer[0],e_layer[0],pinfo.nlayers_per_block);
    );

    /* ================================== */
    /* Construct LayerBlocks Finest Level */
    /* ================================== */
    int nblocks = 0;
    for (int i = s_layer[0]; i <= e_layer[0]; i += pinfo.nlayers_per_block, ++nblocks) {
        /* if sequential, assign single CUDA handles set */
        LayerBlock *block = (pinfo.sequential) ?
                            (new LayerBlock(syncCudaStream)) :
                            (new LayerBlock());
        block->mpi_comm = model.mpi_comm;
        block->mpi_rank = model.rank;
        block->gpu_id = gpu_id;

        /* add layers to the block */
        std::vector<Layer *> &block_layers = block->layers;
        for (int j = 0; j < pinfo.nlayers_per_block; ++j) {
            if(i+j > e_layer[0]) break;

//            printf("Rank[%d] Pushing layer[%d]{GID:%d} to block[%d]:   ",
//                    model_rank,i+j,globalLayers[0][i+j]->getGlobalIdx(),nblocks);
//            globalLayers[0][i+j]->displayLayerType();
            block_layers.push_back(globalLayers[0][i+j]);
        }

        /* setup GPU memory for LayerBlock */
        block->setCudaStreams(); // cudaHandles for block are set above by constructor
        block->allocateLayersDevice(dlmg.multigrid_flag);

        /* push back into blocks vector of this rank's network */
        blocks[0].push_back(block);
    }

    /* ========================== */
    /* Construct Multigrid Levels */
    /* ========================== */
    for (int level = 1; level < mg.nlevels; ++level) {
        Real dt_scale = (Real) pinfo.nlayers_per_block;
        LayerBlock *block;
        int fine_level = level-1;

        s_layer[level] = s_layer[fine_level];
        e_layer[level] = s_layer[fine_level];

        int layer_count = 0;
        BLOCK_LOOP(fine_level) {
            LayerBlock *fblock = blocks[fine_level][i];

            Layer *flayer = fblock->layers[0];
            Layer *clayer = flayer->clone();

            /* save layer pointers */
            flayer->setCoarseLayer(clayer);
            clayer->setFineLayer(flayer);

            clayer->scale_dt(dt_scale);
            if (layer_count == 0) {
                /* build new block: sequential on coarsest level */
                block = (pinfo.sequential || level == mg.nlevels-1) ?
                        (new LayerBlock(syncCudaStream)) :
                        (new LayerBlock());

                /* copy coarse level mpi information */
                block->mpi_comm = fblock->mpi_comm;
                block->mpi_rank = fblock->mpi_rank;
                block->gpu_id = fblock->gpu_id;
            }

            /* push layer into global layers for level */
            clayer->setLevel(level);
            globalLayers[level].push_back(clayer);

            /* push layer onto block */
            block->layers.push_back(clayer);
            e_layer[level] = clayer->getGlobalIdx();

            /* update block layer counter */
            layer_count = (layer_count+1) % pinfo.nlayers_per_block;

            /* push block if full */
            if (layer_count == 0) {
                block->setCudaStreams();
                block->allocateLayersDevice(dlmg.multigrid_flag);
                blocks[level].push_back(block);
            }
        }
        /* push remainder block into vector */
        if(layer_count != 0) {
            block->setCudaStreams();
            block->allocateLayersDevice(dlmg.multigrid_flag);
            blocks[level].push_back(block);
        }
    }

    /* ==================== */
    /* Build Network Ghosts */
    /* ==================== */
    if (model_rank != 0) {
        for (int level = 0; level < mg.nlevels; ++level) {
            GhostLayerLeft *ghost = new GhostLayerLeft(globalLayers[0][s_layer[level]-1],gpu_id);
            leftGhost.push_back(ghost);
        }
    }
    if (model_rank != model.nranks-1) {
        for (int level = 0; level < mg.nlevels; ++level) {
            GhostLayerRight *ghost = new GhostLayerRight(globalLayers[0][e_layer[level]+1],gpu_id);
            rightGhost.push_back(ghost);
        }
    }

    /* allocate network output and adjoint host data for MPI communication */
    for (int level = 0; level < mg.nlevels; ++level) {
        h_out[level] = (Real *) malloc(sizeof(Real)*getOutSize(level)*batch_size);
        h_adjoint[level] = (Real *) malloc(sizeof(Real)*getInSize(level)*batch_size);
    }

    int nblocks_total = 0;
    MPI_Reduce(&nblocks,&nblocks_total,1,MPI_INT,MPI_SUM,0,model.mpi_comm);
    ROOT {
        printf("+=======================================+\n");
        printf("[DLMG] Network partitioned into %d blocks\n",nblocks_total);
        printf("+=======================================+\n");
    }
}