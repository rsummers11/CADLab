/*
 *  _reg_blockMatching_gpu.cu
 *  
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright 2009 UCL - CMIC. All rights reserved.
 *
 */

#ifndef _REG_BLOCKMATCHING_GPU_CU
#define _REG_BLOCKMATCHING_GPU_CU

#include "_reg_blockMatching_gpu.h"
#include "_reg_blockMatching_kernels.cu"
#include <fstream>

void block_matching_method_gpu(nifti_image *targetImage,
                               nifti_image *resultImage,
                               _reg_blockMatchingParam *params,
                               float **targetImageArray_d,
                               float **resultImageArray_d,
                               float **targetPosition_d,
                               float **resultPosition_d,
                               int **activeBlock_d)
{
    if(resultImage!=resultImage)
        printf("Useless lines to avoid a warning");

    // Copy some required parameters over to the device
    int3 bDim =make_int3(params->blockNumber[0], params->blockNumber[1], params->blockNumber[2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_BlockDim, &bDim, sizeof(int3)));

    // Image size
    int3 image_size= make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize, &image_size, sizeof(int3)));

    // Texture binding
    const int numBlocks = bDim.x*bDim.y*bDim.z;
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, targetImageArray_texture, *targetImageArray_d, targetImage->nvox*sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, resultImageArray_texture, *resultImageArray_d, targetImage->nvox*sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, activeBlock_texture, *activeBlock_d, numBlocks*sizeof(int)));

    // Copy the sform transformation matrix onto the device memort
    mat44 *xyz_mat;
    if(targetImage->sform_code>0)
        xyz_mat=&(targetImage->sto_xyz);
    else xyz_mat=&(targetImage->qto_xyz);
    float4 t_m_a_h = make_float4(xyz_mat->m[0][0],xyz_mat->m[0][1],xyz_mat->m[0][2],xyz_mat->m[0][3]);
    float4 t_m_b_h = make_float4(xyz_mat->m[1][0],xyz_mat->m[1][1],xyz_mat->m[1][2],xyz_mat->m[1][3]);
    float4 t_m_c_h = make_float4(xyz_mat->m[2][0],xyz_mat->m[2][1],xyz_mat->m[2][2],xyz_mat->m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_a, &t_m_a_h,sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_b, &t_m_b_h,sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(t_m_c, &t_m_c_h,sizeof(float4)));
    // We need to allocate some memory to keep track of overlap areas and values for blocks
    unsigned memSize = BLOCK_SIZE * params->activeBlockNumber;
    float * targetValues;NR_CUDA_SAFE_CALL(cudaMalloc(&targetValues, memSize * sizeof(float)));
    memSize = BLOCK_SIZE * params->activeBlockNumber;
    float * resultValues;NR_CUDA_SAFE_CALL(cudaMalloc(&resultValues, memSize * sizeof(float)));
    unsigned int Grid_block_matching = (unsigned int)ceil((float)params->activeBlockNumber/(float)Block_target_block);
    unsigned int Grid_block_matching_2 = 1;

    // We have hit the limit in one dimension
    if (Grid_block_matching > 65335) {
        Grid_block_matching_2 = (unsigned int)ceil((float)Grid_block_matching/65535.0f);
        Grid_block_matching = 65335;
    }

    dim3 B1(Block_target_block,1,1);
    dim3 G1(Grid_block_matching,Grid_block_matching_2,1);
    // process the target blocks
    process_target_blocks_gpu<<<G1, B1>>>(  *targetPosition_d,
                                          targetValues);
    NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] process_target_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G1.x,G1.y,G1.z,B1.x,B1.y,B1.z);
#endif

    unsigned int Result_block_matching = params->activeBlockNumber;
    unsigned int Result_block_matching_2 = 1;

    // We have hit the limit in one dimension
    if (Result_block_matching > 65335) {
        Result_block_matching_2 = (unsigned int)ceil((float)Result_block_matching/65535.0f);
        Result_block_matching = 65335;
    }

    dim3 B2(Block_result_block,1,1);
    dim3 G2(Result_block_matching,Result_block_matching_2,1);
    process_result_blocks_gpu<<<G2, B2>>>(*resultPosition_d, targetValues);
    NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifndef NDEBUG
    printf("[NiftyReg CUDA DEBUG] process_result_blocks_gpu kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n",
           cudaGetErrorString(cudaGetLastError()),G2.x,G2.y,G2.z,B2.x,B2.y,B2.z);
#endif
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(targetImageArray_texture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(resultImageArray_texture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(activeBlock_texture));
    NR_CUDA_SAFE_CALL(cudaFree(targetValues));
    NR_CUDA_SAFE_CALL(cudaFree(resultValues));

}

void optimize_gpu(	_reg_blockMatchingParam *blockMatchingParams,
                  mat44 *updateAffineMatrix,
                  float **targetPosition_d,
                  float **resultPosition_d,
                  bool affine)
{   
    // We will simply call the CPU version as this step is probably
    // not worth implementing on the GPU.
    // device to host copy
    int memSize = blockMatchingParams->activeBlockNumber * 3 * sizeof(float);
    NR_CUDA_SAFE_CALL(cudaMemcpy(blockMatchingParams->targetPosition, *targetPosition_d, memSize, cudaMemcpyDeviceToHost));
    NR_CUDA_SAFE_CALL(cudaMemcpy(blockMatchingParams->resultPosition, *resultPosition_d, memSize, cudaMemcpyDeviceToHost));
    // Cheat and call the CPU version.
    optimize(blockMatchingParams, updateAffineMatrix, affine);
}

#endif
