/*
 *  _reg_blockMatching_kernels.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_KERNELS_CU__
#define __REG_BLOCKMATCHING_KERNELS_CU__

// Some parameters that we need for the kernel execution.
// The caller is supposed to ensure that the values are set

// Number of blocks in each dimension
__device__ __constant__ int3 c_BlockDim;
__device__ __constant__ int c_StepSize;
__device__ __constant__ int3 c_ImageSize;
__device__ __constant__ float r1c1;

// Transformation matrix from nifti header
__device__ __constant__ float4 t_m_a;
__device__ __constant__ float4 t_m_b;
__device__ __constant__ float4 t_m_c;

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define OVERLAP_SIZE 3
#define STEP_SIZE 1

#include "_reg_blockMatching_gpu.h"

texture<float, 1, cudaReadModeElementType> targetImageArray_texture;
texture<float, 1, cudaReadModeElementType> resultImageArray_texture;
texture<int, 1, cudaReadModeElementType> activeBlock_texture;

// Apply the transformation matrix
__device__ inline void apply_affine(const float4 &pt, float * result)
{
        float4 mat = t_m_a;
        result[0] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
        mat = t_m_b;
        result[1] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
        mat = t_m_c;
        result[2] = (mat.x * pt.x) + (mat.y*pt.y) + (mat.z*pt.z) + (mat.w);
}

// CUDA kernel to process the target values
__global__ void process_target_blocks_gpu(float *targetPosition_d,
                                          float *targetValues)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    const int3 bDim = c_BlockDim;

    if (tid < bDim.x * bDim.y * bDim.z){
        const int currentBlockIndex = tex1Dfetch(activeBlock_texture,tid);
        if (currentBlockIndex >= 0){
            // Get the corresponding (i, j, k) indices
            int tempIndex = currentBlockIndex;
            const int k =(int)(tempIndex/(bDim.x * bDim.y));
            tempIndex -= k * bDim.x * bDim.y;
            const int j =(int)(tempIndex/(bDim.x));
            const int i = tempIndex - j * (bDim.x);
            const int offset = tid * BLOCK_SIZE;
            const int targetIndex_start_x = i * BLOCK_WIDTH;
            const int targetIndex_start_y = j * BLOCK_WIDTH;
            const int targetIndex_start_z = k * BLOCK_WIDTH;

            int targetIndex_end_x = targetIndex_start_x + BLOCK_WIDTH;
            int targetIndex_end_y = targetIndex_start_y + BLOCK_WIDTH;
            int targetIndex_end_z = targetIndex_start_z + BLOCK_WIDTH;
            const int3 imageSize = c_ImageSize;
            for (int count = 0; count < BLOCK_SIZE; ++count) targetValues[count + offset] = 0.0f;
            unsigned int index = 0;

            for(int z = targetIndex_start_z; z< targetIndex_end_z; ++z){
                if (z>=0 && z<imageSize.z) {
                    int indexZ = z * imageSize.x * imageSize.y;
                    for(int y = targetIndex_start_y; y < targetIndex_end_y; ++y){
                        if (y>=0 && y<imageSize.y) {
                            int indexXYZ = indexZ + y * imageSize.x + targetIndex_start_x;
                            for(int x = targetIndex_start_x; x < targetIndex_end_x; ++x){
                                if(x>=0 && x<imageSize.x) {
                                    targetValues[index + offset] = tex1Dfetch(targetImageArray_texture, indexXYZ);
                                }
                                indexXYZ++;
                                index++;
                            }
                        }
                        else index += BLOCK_WIDTH;
                    }
                }
                else index += BLOCK_WIDTH * BLOCK_WIDTH;
            }

            float4 targetPosition;
            targetPosition.x = i * BLOCK_WIDTH;
            targetPosition.y = j * BLOCK_WIDTH;
            targetPosition.z = k * BLOCK_WIDTH;
            apply_affine(targetPosition, &(targetPosition_d[tid * 3]));
        }
    }
}

// CUDA kernel to process the result blocks
__global__ void process_result_blocks_gpu(float *resultPosition_d,
                                          float *targetValues)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    const int3 bDim = c_BlockDim;
    int tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
    __shared__ int ctid;
    if (tempIndex == 0) ctid = (int)(tid / NUM_BLOCKS_TO_COMPARE);
    __syncthreads();
    //const int ctid = (int)(tid / NUM_BLOCKS_TO_COMPARE);
    __shared__ float4 localCC [NUM_BLOCKS_TO_COMPARE];
    __shared__ int3 indexes;
    localCC[tempIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    __shared__ int updateThreadID;
    updateThreadID = -1;
    if (ctid < bDim.x * bDim.y * bDim.z) {
        const int activeBlockIndex = tex1Dfetch(activeBlock_texture, ctid);
        tempIndex = activeBlockIndex;
        int k =(int)(tempIndex/(bDim.x * bDim.y));
        tempIndex -= k * bDim.x * bDim.y;
        int j =(int)(tempIndex/(bDim.x));
        int i = tempIndex - j * (bDim.x);
        tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
        if (tempIndex == 0) {
            indexes.x = i * BLOCK_WIDTH;
            indexes.y = j * BLOCK_WIDTH;
            indexes.z = k * BLOCK_WIDTH;
        }
        __syncthreads();

        if (activeBlockIndex >= 0) {
            const int block_offset = ctid * BLOCK_SIZE;
            const int3 imageSize = c_ImageSize;
            int k = (int)tempIndex /NUM_BLOCKS_TO_COMPARE_2D;
            tempIndex -= k * NUM_BLOCKS_TO_COMPARE_2D;
            int j = (int)tempIndex /NUM_BLOCKS_TO_COMPARE_1D;
            int i = tempIndex - j * NUM_BLOCKS_TO_COMPARE_1D;
            k -= OVERLAP_SIZE;
            j -= OVERLAP_SIZE;
            i -= OVERLAP_SIZE;
            tempIndex = tid % NUM_BLOCKS_TO_COMPARE;
            int resultIndex_start_z = indexes.z + k;
            int resultIndex_end_z = resultIndex_start_z + BLOCK_WIDTH;
            int resultIndex_start_y = indexes.y + j;
            int resultIndex_end_y = resultIndex_start_y + BLOCK_WIDTH;
            int resultIndex_start_x = indexes.x + i;
            int resultIndex_end_x = resultIndex_start_x + BLOCK_WIDTH;
            __shared__ float4 cc_vars [NUM_BLOCKS_TO_COMPARE];
            cc_vars[tempIndex].x = 0.0f;
            cc_vars[tempIndex].y = 0.0f;
            unsigned int index = 0;
            for(int z = resultIndex_start_z; z< resultIndex_end_z; ++z){
                if (z>=0 && z<imageSize.z) {
                    int indexZ = z * imageSize.y * imageSize.x;
                    for(int y = resultIndex_start_y; y < resultIndex_end_y; ++y){
                        if (y>=0 && y<imageSize.y) {
                            int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
                            for(int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
                                if (x>=0 && x<imageSize.x) {
                                    cc_vars[tempIndex].x = tex1Dfetch(resultImageArray_texture, indexXYZ);
                                    cc_vars[tempIndex].y = targetValues[block_offset + index];
                                    if (cc_vars[tempIndex].x != 0.0f && cc_vars[tempIndex].y != 0.0f) {
                                        localCC[tempIndex].x += cc_vars[tempIndex].x;
                                        localCC[tempIndex].y += cc_vars[tempIndex].y;
                                        ++localCC[tempIndex].z;
                                    }
                                }
                                ++indexXYZ;
                                ++index;
                            }
                        }
                        else index += BLOCK_WIDTH;
                    }
                }
                else index += BLOCK_WIDTH * BLOCK_WIDTH;
            }

            if (localCC[tempIndex].z > 0) {
                localCC[tempIndex].x /= localCC[tempIndex].z;
                localCC[tempIndex].y /= localCC[tempIndex].z;
            }
            cc_vars[tempIndex].z = 0.0f;
            cc_vars[tempIndex].w = 0.0f;
            index = 0;
            for(int z = resultIndex_start_z; z< resultIndex_end_z; ++z){
                if (z>=0 && z<imageSize.z) {
                    int indexZ = z * imageSize.y * imageSize.x;
                    for(int y = resultIndex_start_y; y < resultIndex_end_y; ++y){
                        if(y>=0 && y<imageSize.y) {
                            int indexXYZ = indexZ + y * imageSize.x + resultIndex_start_x;
                            for(int x = resultIndex_start_x; x < resultIndex_end_x; ++x){
                                if (x>=0 && x<imageSize.x) {
                                    cc_vars[tempIndex].x = tex1Dfetch(resultImageArray_texture, indexXYZ);
                                    cc_vars[tempIndex].y = targetValues[block_offset + index];
                                    if (cc_vars[tempIndex].x != 0.0f && cc_vars[tempIndex].y != 0.0f) {
                                        cc_vars[tempIndex].x -= localCC[tempIndex].x;
                                        cc_vars[tempIndex].y -= localCC[tempIndex].y;

                                        cc_vars[tempIndex].z += cc_vars[tempIndex].x * cc_vars[tempIndex].x;
                                        cc_vars[tempIndex].w += cc_vars[tempIndex].y * cc_vars[tempIndex].y;
                                        localCC[tempIndex].w += cc_vars[tempIndex].x * cc_vars[tempIndex].y;
                                    }
                                }
                                ++indexXYZ;
                                ++index;
                            }
                        }
                        else index += BLOCK_WIDTH;
                    }
                }
                else index += BLOCK_WIDTH * BLOCK_WIDTH;
            }

            if (localCC[tempIndex].z > (float)(BLOCK_SIZE/2)) {
                if (cc_vars[tempIndex].z > 0.0f && cc_vars[tempIndex].w > 0.0f) {
                    localCC[tempIndex].w = fabsf(localCC[tempIndex].w/sqrt(cc_vars[tempIndex].z * cc_vars[tempIndex].w));
                }
            }
            else { localCC[tempIndex].w = 0.0f; }

            localCC[tempIndex].x = i;
            localCC[tempIndex].y = j;
            localCC[tempIndex].z = k;

            // Just take ownership of updating the final value
            if (updateThreadID == -1) updateThreadID = tid;
        }
        __syncthreads();

        // Just let one thread do the final update
        if (tid == updateThreadID) {
            __shared__ float4 bestCC;
            bestCC = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = 0; i < NUM_BLOCKS_TO_COMPARE; ++i) {
                if (localCC[i].w > bestCC.w) {
                    bestCC.x = localCC[i].x;
                    bestCC.y = localCC[i].y;
                    bestCC.z = localCC[i].z;
                    bestCC.w = localCC[i].w;
                }
            }
            bestCC.x += indexes.x;
            bestCC.y += indexes.y;
            bestCC.z += indexes.z;
            apply_affine(bestCC, &(resultPosition_d[ctid * 3]));
        }
    }
}

#endif
