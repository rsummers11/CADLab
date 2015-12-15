/** @file _reg_blocksize_gpu.h
 * @author Marc Modat
 * @date 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_BLOCKSIZE_GPU_H
#define _REG_BLOCKSIZE_GPU_H

#include "nifti1_io.h"
#include "cuda_runtime.h"
#include "cuda.h"

/* ******************************** */
/* ******************************** */
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
struct __attribute__((aligned(4))) float4
{
   float x,y,z,w;
};
#endif
/* ******************************** */
/* ******************************** */
#if CUDART_VERSION >= 3200
#   define NR_CUDA_SAFE_CALL(call) { \
        call; \
        cudaError err = cudaPeekAtLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
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
#   define NR_CUDA_SAFE_CALL(call) { \
        call; \
        cudaError err = cudaThreadSynchronize(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
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
/* ******************************** */
/* ******************************** */
class NiftyReg_CudaBlock100
{
public:    /* _reg_blockMatching_gpu */
   size_t Block_target_block;
   size_t Block_result_block;
   /* _reg_mutualinformation_gpu */
   size_t Block_reg_smoothJointHistogramX;
   size_t Block_reg_smoothJointHistogramY;
   size_t Block_reg_smoothJointHistogramZ;
   size_t Block_reg_smoothJointHistogramW;
   size_t Block_reg_marginaliseTargetX;
   size_t Block_reg_marginaliseTargetXY;
   size_t Block_reg_marginaliseResultX;
   size_t Block_reg_marginaliseResultXY;
   size_t Block_reg_getVoxelBasedNMIGradientUsingPW2D;
   size_t Block_reg_getVoxelBasedNMIGradientUsingPW3D;
   size_t Block_reg_getVoxelBasedNMIGradientUsingPW2x2;
   /* _reg_globalTransformation_gpu */
   size_t Block_reg_affine_deformationField;
   /* _reg_localTransformation_gpu */
   size_t Block_reg_spline_getDeformationField2D;
   size_t Block_reg_spline_getDeformationField3D;
   size_t Block_reg_spline_getApproxSecondDerivatives2D;
   size_t Block_reg_spline_getApproxSecondDerivatives3D;
   size_t Block_reg_spline_getApproxBendingEnergy2D;
   size_t Block_reg_spline_getApproxBendingEnergy3D;
   size_t Block_reg_spline_getApproxBendingEnergyGradient2D;
   size_t Block_reg_spline_getApproxBendingEnergyGradient3D;
   size_t Block_reg_spline_getApproxJacobianValues2D;
   size_t Block_reg_spline_getApproxJacobianValues3D;
   size_t Block_reg_spline_getJacobianValues2D;
   size_t Block_reg_spline_getJacobianValues3D;
   size_t Block_reg_spline_logSquaredValues;
   size_t Block_reg_spline_computeApproxJacGradient2D;
   size_t Block_reg_spline_computeApproxJacGradient3D;
   size_t Block_reg_spline_computeJacGradient2D;
   size_t Block_reg_spline_computeJacGradient3D;
   size_t Block_reg_spline_approxCorrectFolding3D;
   size_t Block_reg_spline_correctFolding3D;
   size_t Block_reg_getDeformationFromDisplacement;
   size_t Block_reg_getDisplacementFromDeformation;
   size_t Block_reg_defField_compose2D;
   size_t Block_reg_defField_compose3D;
   size_t Block_reg_defField_getJacobianMatrix;
   /* _reg_optimiser_gpu */
   size_t Block_reg_initialiseConjugateGradient;
   size_t Block_reg_GetConjugateGradient1;
   size_t Block_reg_GetConjugateGradient2;
   size_t Block_reg_getEuclideanDistance;
   size_t Block_reg_updateControlPointPosition;
   /* _reg_ssd_gpu */
   size_t Block_reg_getSquaredDifference;
   size_t Block_reg_getSSDGradient;
   /* _reg_tools_gpu */
   size_t Block_reg_voxelCentric2NodeCentric;
   size_t Block_reg_convertNMIGradientFromVoxelToRealSpace;
   size_t Block_reg_ApplyConvolutionWindowAlongX;
   size_t Block_reg_ApplyConvolutionWindowAlongY;
   size_t Block_reg_ApplyConvolutionWindowAlongZ;
   size_t Block_reg_arithmetic;
   /* _reg_resampling_gpu */
   size_t Block_reg_resampleImage2D;
   size_t Block_reg_resampleImage3D;
   size_t Block_reg_getImageGradient2D;
   size_t Block_reg_getImageGradient3D;

   NiftyReg_CudaBlock100();
   ~NiftyReg_CudaBlock100()
   {
      ;
   }
};
/* ******************************** */
class NiftyReg_CudaBlock200 : public NiftyReg_CudaBlock100
{
public:
   NiftyReg_CudaBlock200();
   ~NiftyReg_CudaBlock200()
   {
      ;
   }
};
/* ******************************** */
class NiftyReg_CudaBlock300 : public NiftyReg_CudaBlock100
{
public:
   NiftyReg_CudaBlock300();
   ~NiftyReg_CudaBlock300()
   {
      ;
   }
};
/* ******************************** */
class NiftyReg_CudaBlock
{
public:
   static NiftyReg_CudaBlock100 * getInstance(int major)
   {
      if (instance) return instance;
      else
      {
         switch(major)
         {
         case 3:
            instance = new NiftyReg_CudaBlock300();
            break;
         case 2:
            instance = new NiftyReg_CudaBlock200();
            break;
         default:
            instance = new NiftyReg_CudaBlock100();
            break;
         }
      }
      return instance;
   }
private:
   static NiftyReg_CudaBlock100 * instance;
};
/* ******************************** */
/* ******************************** */

#endif
