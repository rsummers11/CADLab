/*
 *  _reg_blocksize_gpu.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BLOCKSIZE_GPU_H
#define _REG_BLOCKSIZE_GPU_H

#include "nifti1_io.h"
#include "cuda_runtime.h"


#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
        struct __attribute__(aligned(4)) float4{
                float x,y,z,w;
        };
#endif

#if __CUDA_ARCH__ >= 120
    #if __CUDA_ARCH__ >= 200
/* ************************************************************************ */
#define Block_reg_affine_deformationField 512                       // 20 regs
#define Block_result_block 416                                      // 26 regs - 11016+00
#define Block_target_block 512                                      // 18 regs
/* ************************************************************************ */
#define Block_reg_bspline_getDeformationField 352                   // 44 regs - 11264+00
#define Block_reg_bspline_getApproxSecondDerivatives 352            // 43 regs - 00648+00
#define Block_reg_bspline_getApproxBendingEnergy 384                // 27 regs
#define Block_reg_bspline_getApproxBendingEnergyGradient 512        // 31 regs
#define Block_reg_bspline_getApproxJacobianValues 512               // 32 regs
#define Block_reg_bspline_getJacobianValues 288                     // 52 regs - 09216+0
#define Block_reg_bspline_logSquaredValues 512                      // 10 regs
#define Block_reg_bspline_computeApproxJacGradient 416              // 38 regs
#define Block_reg_bspline_computeJacGradient 416                    // 38 regs
#define Block_reg_bspline_approxCorrectFolding 416                  // 37 regs
#define Block_reg_bspline_correctFolding 416                        // 37 regs
#define Block_reg_defField_compose 512                              // 18 regs
#define Block_reg_defField_getJacobianMatrix 448                    // ?? regs
/* ************************************************************************ */
#define Block_reg_getVoxelBasedNMIGradientUsingPW 384               // 28 regs
#define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 288            // 52 regs
#define Block_reg_smoothJointHistogramX 512                         // 12 regs
#define Block_reg_smoothJointHistogramY 512                         // 11 regs
#define Block_reg_smoothJointHistogramZ 512                         // 12 regs
#define Block_reg_smoothJointHistogramW 512                         // 12 regs
#define Block_reg_marginaliseTargetX 512                            // 07 regs
#define Block_reg_marginaliseTargetXY 512                           // 07 regs
#define Block_reg_marginaliseResultX 512                            // 08 regs
#define Block_reg_marginaliseResultXY 512                           // 08 regs
/* ************************************************************************ */
#define Block_reg_resampleSourceImage 448                           // 23 regs
#define Block_reg_getSourceImageGradient 416                        // 26 regs
/* ************************************************************************ */
#define Block_reg_voxelCentric2NodeCentric 512                      // 10 regs
#define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 19 regs
#define Block_reg_initialiseConjugateGradient 512                   // 10 regs
#define Block_reg_GetConjugateGradient1 512                         // 15 regs
#define Block_reg_GetConjugateGradient2 512                         // 20 regs
#define Block_reg_getMaximalLength 512                              // 11 regs
#define Block_reg_updateControlPointPosition 512                    // 11 regs
#define Block_reg_ApplyConvolutionWindowAlongX 512                  // 15 regs
#define Block_reg_ApplyConvolutionWindowAlongY 512                  // 15 regs
#define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 16 regs
/* ************************************************************************ */
    #else // __CUDA_ARCH__ >= 200
/* ************************************************************************ */
#define Block_reg_affine_deformationField 512                       // 16 regs
#define Block_result_block 384                                      // 21 regs - 11032+16
#define Block_target_block 512                                      // 15 regs
/* ************************************************************************ */
#define Block_reg_bspline_getDeformationField 384                   // 37 regs - 12296+16
#define Block_reg_bspline_getApproxSecondDerivatives 448            // 35 regs - 00656+16
#define Block_reg_bspline_getApproxBendingEnergy 512                // 12 regs
#define Block_reg_bspline_getApproxBendingEnergyGradient 320        // 23 regs
#define Block_reg_bspline_getApproxJacobianValues 512               // 27 regs
#define Block_reg_bspline_getJacobianValues 384                     // 41 regs - 12304+16
#define Block_reg_bspline_logSquaredValues 512                      // 07 regs
#define Block_reg_bspline_computeApproxJacGradient 512              // 32 regs
#define Block_reg_bspline_computeJacGradient 512                    // 32 regs
#define Block_reg_bspline_approxCorrectFolding 512                  // 32 regs
#define Block_reg_bspline_correctFolding 512                        // 31 regs
#define Block_reg_defField_compose 448                              // 18 regs
#define Block_reg_defField_getJacobianMatrix 512                    // ?? regs
/* ************************************************************************ */
#define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 25 regs
#define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 384            // 42 regs
#define Block_reg_smoothJointHistogramX 512                         // 07 regs
#define Block_reg_smoothJointHistogramY 512                         // 11 regs
#define Block_reg_smoothJointHistogramZ 512                         // 11 regs
#define Block_reg_smoothJointHistogramW 512                         // 08 regs
#define Block_reg_marginaliseTargetX 512                            // 06 regs
#define Block_reg_marginaliseTargetXY 512                           // 06 regs
#define Block_reg_marginaliseResultX 512                            // 07 regs
#define Block_reg_marginaliseResultXY 512                           // 07 regs
/* ************************************************************************ */
#define Block_reg_getSourceImageGradient 320                        // 25 regs
#define Block_reg_resampleSourceImage 512                           // 16 regs
/* ************************************************************************ */
#define Block_reg_voxelCentric2NodeCentric 512                      // 11 regs
#define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 16 regs
#define Block_reg_initialiseConjugateGradient 512                   // 09 regs
#define Block_reg_GetConjugateGradient1 512                         // 12 regs
#define Block_reg_GetConjugateGradient2 512                         // 10 regs
#define Block_reg_getMaximalLength 384                              // 07 regs
#define Block_reg_updateControlPointPosition 512                    // 08 regs
#define Block_reg_ApplyConvolutionWindowAlongX 512                  // 14 regs
#define Block_reg_ApplyConvolutionWindowAlongY 512                  // 14 regs
#define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 15 regs
/* ************************************************************************ */
    #endif // __CUDA_ARCH__ >= 200
#else // __CUDA_ARCH__ >= 120
/* ************************************************************************ */
#define Block_reg_affine_deformationField 512                       // 16 regs
#define Block_result_block 384                                      // 21 regs - 11032+16
#define Block_target_block 512                                      // 15 regs
/* ************************************************************************ */
#define Block_reg_bspline_getDeformationField 192                   // 37 regs - 06152+16
#define Block_reg_bspline_getApproxSecondDerivatives 192            // 35 regs - 00656+16
#define Block_reg_bspline_getApproxBendingEnergy 320                // 12 regs
#define Block_reg_bspline_getApproxBendingEnergyGradient 256        // 27 regs
#define Block_reg_bspline_getApproxJacobianValues 256               // 27 regs
#define Block_reg_bspline_getJacobianValues 192                     // 41 regs - 09280+16
#define Block_reg_bspline_logSquaredValues 384                      // 07 regs
#define Block_reg_bspline_computeApproxJacGradient 256              // 32 regs
#define Block_reg_bspline_computeJacGradient 256                    // 32 regs
#define Block_reg_bspline_approxCorrectFolding 256                  // 32 regs
#define Block_reg_bspline_correctFolding 256                        // 31 regs
#define Block_reg_defField_compose 448                              // 18 regs
#define Block_reg_defField_getJacobianMatrix 512                    // 16 regs
/* ************************************************************************ */
#define Block_reg_getVoxelBasedNMIGradientUsingPW 320               // 25 regs
#define Block_reg_getVoxelBasedNMIGradientUsingPW2x2 192            // 42 regs
#define Block_reg_smoothJointHistogramX 512                         // 07 regs
#define Block_reg_smoothJointHistogramY 512                         // 11 regs
#define Block_reg_smoothJointHistogramZ 512                         // 11 regs
#define Block_reg_smoothJointHistogramW 512                         // 08 regs
#define Block_reg_marginaliseTargetX 512                            // 06 regs
#define Block_reg_marginaliseTargetXY 512                           // 06 regs
#define Block_reg_marginaliseResultX 512                            // 07 regs
#define Block_reg_marginaliseResultXY 512                           // 07 regs
/* ************************************************************************ */
#define Block_reg_getSourceImageGradient 320                        // 25 regs
#define Block_reg_resampleSourceImage 512                           // 16 regs
/* ************************************************************************ */
#define Block_reg_voxelCentric2NodeCentric 512                      // 11 regs
#define Block_reg_convertNMIGradientFromVoxelToRealSpace 512        // 16 regs
#define Block_reg_initialiseConjugateGradient 512                   // 09 regs
#define Block_reg_GetConjugateGradient1 512                         // 12 regs
#define Block_reg_GetConjugateGradient2 512                         // 10 regs
#define Block_reg_getMaximalLength 384                              // 07 regs
#define Block_reg_updateControlPointPosition 512                    // 08 regs
#define Block_reg_ApplyConvolutionWindowAlongX 512                  // 14 regs
#define Block_reg_ApplyConvolutionWindowAlongY 512                  // 14 regs
#define Block_reg_ApplyConvolutionWindowAlongZ 512                  // 15 regs
/* ************************************************************************ */
#endif // __CUDA_ARCH__ >= 200

#if CUDART_VERSION >= 3200
#define NR_CUDA_SAFE_CALL(call) { \
    call; \
    cudaError err = cudaPeekAtLastError(); \
    if( cudaSuccess != err) { \
        fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#define NR_CUDA_CHECK_KERNEL(grid,block) { \
    cudaThreadSynchronize(); \
    cudaError err = cudaPeekAtLastError(); \
    if( err != cudaSuccess) { \
        fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
        grid.x,grid.y,grid.z,block.x,block.y,block.z); \
        exit(EXIT_FAILURE); \
    } \
}
#else //CUDART_VERSION >= 3200
#define NR_CUDA_SAFE_CALL(call) { \
    call; \
    cudaError err = cudaThreadSynchronize(); \
    if( cudaSuccess != err) { \
        fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#define NR_CUDA_CHECK_KERNEL(grid,block) { \
    cudaError err = cudaThreadSynchronize(); \
    if( err != cudaSuccess) { \
        fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
        grid.x,grid.y,grid.z,block.x,block.y,block.z); \
        exit(EXIT_FAILURE); \
    } \
}
#endif //CUDART_VERSION >= 3200
#endif
