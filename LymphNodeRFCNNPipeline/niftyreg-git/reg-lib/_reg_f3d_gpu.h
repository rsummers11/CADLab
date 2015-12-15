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
#include "_reg_nmi_gpu.h"
#include "_reg_ssd_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_common_gpu.h"
#include "_reg_optimiser_gpu.h"
#include "_reg_f3d.h"

class reg_f3d_gpu : public reg_f3d<float>
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
   float4 *transformationGradient_gpu;

   // cuda variable for multispectral registration
   cudaArray *currentReference2_gpu;
   cudaArray *currentFloating2_gpu;
   float *warped2_gpu;
   float4 *warpedGradientImage2_gpu;

   // Measure related variables
   reg_ssd_gpu *measure_gpu_ssd;
   reg_kld_gpu *measure_gpu_kld;
   reg_dti_gpu *measure_gpu_dti;
   reg_lncc_gpu *measure_gpu_lncc;
   reg_nmi_gpu *measure_gpu_nmi;
   reg_multichannel_nmi_gpu *measure_gpu_multichannel_nmi;

   float InitialiseCurrentLevel();
   void ClearCurrentInputImage();
   void AllocateWarped();
   void ClearWarped();
   void AllocateDeformationField();
   void ClearDeformationField();
   void AllocateWarpedGradient();
   void ClearWarpedGradient();
   void AllocateVoxelBasedMeasureGradient();
   void ClearVoxelBasedMeasureGradient();
   void AllocateTransformationGradient();
   void ClearTransformationGradient();

   double ComputeJacobianBasedPenaltyTerm(int);
   double ComputeBendingEnergyPenaltyTerm();
   void GetDeformationField();
   void WarpFloatingImage(int);
   void GetVoxelBasedGradient();
   void GetSimilarityMeasureGradient();
   void GetBendingEnergyGradient();
   void GetJacobianBasedGradient();
   void GetApproximatedGradient();
   void UpdateParameters(float);
   void SetOptimiser();
   void SetGradientImageToZero();
   float NormaliseGradient();
   void InitialiseSimilarity();

public:
   void UseNMISetReferenceBinNumber(int,int);
   void UseNMISetFloatingBinNumber(int,int);
   void UseMultiChannelNMI(int timepointNumber, int *timepoint);
   void UseSSD(int timepoint);
   void UseKLDivergence(int timepoint);
   void UseDTI(int timepoint[6]);
   void UseLNCC(int timepoint, float stdDevKernel);

   reg_f3d_gpu(int refTimePoint,int floTimePoint);
   ~reg_f3d_gpu();
   int CheckMemoryMB();
};

#include "_reg_f3d_gpu.cpp"

#endif
