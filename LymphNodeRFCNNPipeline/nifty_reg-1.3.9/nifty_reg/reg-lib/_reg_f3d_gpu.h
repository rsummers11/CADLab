/*
 *  _reg_f3d_gpu.h
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_GPU_H
#define _REG_F3D_GPU_H

#include "_reg_resampling_gpu.h"
#include "_reg_globalTransformation_gpu.h"
#include "_reg_localTransformation_gpu.h"
#include "_reg_mutualinformation_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_cudaCommon.h"
#include "_reg_f3d.h"

#include "cuda.h"

template <class T>
class reg_f3d_gpu : public reg_f3d<T>
{
  protected:

    // cuda variables
    cudaArray *currentReference_gpu;
    cudaArray *currentFloating_gpu;
    int *currentMask_gpu;
    float *warped_gpu;
    float4 *controlPointGrid_gpu;
    float4 *deformationFieldImage_gpu;
    float4 *warpedGradientImage_gpu;
    float4 *voxelBasedMeasureGradientImage_gpu;
    float4 *nodeBasedGradientImage_gpu;
    float4 *conjugateG_gpu;
    float4 *conjugateH_gpu;
    float4 *bestControlPointPosition_gpu;
    float *logJointHistogram_gpu;

    // cuda variable for multispectral registration
    cudaArray *currentReference2_gpu;
    cudaArray *currentFloating2_gpu;
    float *warped2_gpu;
    float4 *warpedGradientImage2_gpu;

    void AllocateCurrentInputImage();
    void ClearCurrentInputImage();
    void AllocateWarped();
    void ClearWarped();
    void AllocateDeformationField();
    void ClearDeformationField();
    void AllocateWarpedGradient();
    void ClearWarpedGradient();
    void AllocateVoxelBasedMeasureGradient();
    void ClearVoxelBasedMeasureGradient();
    void AllocateNodeBasedGradient();
    void ClearNodeBasedGradient();
    void AllocateConjugateGradientVariables();
    void ClearConjugateGradientVariables();
    void AllocateBestControlPointArray();
    void ClearBestControlPointArray();
    void AllocateJointHistogram();
    void ClearJointHistogram();

    void SaveCurrentControlPoint();
    void RestoreCurrentControlPoint();
    double ComputeJacobianBasedPenaltyTerm(int);
    double ComputeBendingEnergyPenaltyTerm();
    void GetDeformationField();
    void WarpFloatingImage(int);
    double ComputeSimilarityMeasure();
    void GetVoxelBasedGradient();
    void GetSimilarityMeasureGradient();
    void GetBendingEnergyGradient();
    void GetJacobianBasedGradient();
    void ComputeConjugateGradient();
    T GetMaximalGradientLength();
    void UpdateControlPointPosition(T);

public:
    reg_f3d_gpu(int refTimePoint,int floTimePoint);
    ~reg_f3d_gpu();
    void Run_f3d();
    int CheckMemoryMB_f3d();
};

#include "_reg_f3d_gpu.cpp"

#endif
