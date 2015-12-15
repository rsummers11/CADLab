/** @file _reg_blocksize_gpu.cu
 * @author Marc Modat
 * @date 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_BLOCKSIZE_GPU_CU
#define _REG_BLOCKSIZE_GPU_CU

#include "_reg_blocksize_gpu.h"

/* ******************************** */
/* ******************************** */
NiftyReg_CudaBlock100 * NiftyReg_CudaBlock::instance = NULL;
/* ******************************** */
/* ******************************** */
NiftyReg_CudaBlock100::NiftyReg_CudaBlock100()
{
    Block_target_block = 512; // 15 reg - 32 smem - 24 cmem
    Block_result_block = 384; // 21 reg - 11048 smem - 24 cmem
    /* _reg_mutualinformation_gpu */
    Block_reg_smoothJointHistogramX = 384; // 07 reg - 24 smem - 20 cmem
    Block_reg_smoothJointHistogramY = 320; // 11 reg - 24 smem - 20 cmem
    Block_reg_smoothJointHistogramZ = 320; // 11 reg - 24 smem - 20 cmem
    Block_reg_smoothJointHistogramW = 384; // 08 reg - 24 smem - 20 cmem
    Block_reg_marginaliseTargetX = 384; // 06 reg - 24 smem
    Block_reg_marginaliseTargetXY = 384; // 07 reg - 24 smem
    Block_reg_marginaliseResultX = 384; // 06 reg - 24 smem
    Block_reg_marginaliseResultXY = 384; // 07 reg - 24 smem
    Block_reg_getVoxelBasedNMIGradientUsingPW2D = 384; // 21 reg - 24 smem - 32 cmem
    Block_reg_getVoxelBasedNMIGradientUsingPW3D = 320; // 25 reg - 24 smem - 32 cmem
    Block_reg_getVoxelBasedNMIGradientUsingPW2x2 = 192; // 42 reg - 24 smem - 36 cmem
    /* _reg_globalTransformation_gpu */
    Block_reg_affine_deformationField = 512; // 16 reg - 24 smem
    /* _reg_localTransformation_gpu */
    Block_reg_spline_getDeformationField2D = 384; // 20 reg - 6168 smem - 28 cmem
    Block_reg_spline_getDeformationField3D = 192; // 37 reg - 6168 smem - 28 cmem
    Block_reg_spline_getApproxSecondDerivatives2D = 512; // 15 reg - 132 smem - 32 cmem
    Block_reg_spline_getApproxSecondDerivatives3D = 192; // 38 reg - 672 smem - 104 cmem
    Block_reg_spline_getApproxBendingEnergy2D = 384; // 07 reg - 24 smem
    Block_reg_spline_getApproxBendingEnergy3D = 320; // 12 reg - 24 smem
    Block_reg_spline_getApproxBendingEnergyGradient2D = 512; // 15 reg - 132 smem - 36 cmem
    Block_reg_spline_getApproxBendingEnergyGradient3D = 256; // 27 reg - 672 smem - 108 cmem
    Block_reg_spline_getApproxJacobianValues2D = 384; // 17 reg - 104 smem - 36 cmem
    Block_reg_spline_getApproxJacobianValues3D = 256; // 27 reg - 356 smem - 108 cmem
    Block_reg_spline_getJacobianValues2D = 256; // 29 reg - 32 smem - 16 cmem - 32 lmem
    Block_reg_spline_getJacobianValues3D = 192; // 41 reg - 6176 smem - 20 cmem - 32 lmem
    Block_reg_spline_logSquaredValues = 384; // 07 reg - 24 smem - 36 cmem
    Block_reg_spline_computeApproxJacGradient2D = 320; // 23 reg - 96 smem - 72 cmem
    Block_reg_spline_computeApproxJacGradient3D = 256; // 32 reg - 384 smem - 144 cmem
    Block_reg_spline_computeJacGradient2D = 384; // 21 reg - 24 smem - 64 cmem
    Block_reg_spline_computeJacGradient3D = 256; // 32 reg - 24 smem - 64 cmem
    Block_reg_spline_approxCorrectFolding3D = 256; // 32 reg - 24 smem - 24 cmem
    Block_reg_spline_correctFolding3D = 256; // 31 reg - 24 smem - 32 cmem
    Block_reg_getDeformationFromDisplacement = 384; // 09 reg - 24 smem
    Block_reg_getDisplacementFromDeformation = 384; // 09 reg - 24 smem
    Block_reg_defField_compose2D = 512; // 15 reg - 24 smem - 08 cmem - 16 lmem
    Block_reg_defField_compose3D = 384; // 21 reg - 24 smem - 08 cmem - 24 lmem
    Block_reg_defField_getJacobianMatrix = 512; // 16 reg - 24 smem - 04 cmem
    /* _reg_optimiser_gpu */
    Block_reg_initialiseConjugateGradient = 384; // 09 reg - 24 smem
    Block_reg_GetConjugateGradient1 = 320; // 12 reg - 24 smem
    Block_reg_GetConjugateGradient2 = 384; // 10 reg - 40 smem
    Block_reg_getEuclideanDistance = 384; // 04 reg - 24 smem
    Block_reg_updateControlPointPosition = 384; // 08 reg - 24 smem
    /* _reg_ssd_gpu */
    Block_reg_getSquaredDifference = 320; // 12 reg - 24 smem - 08 cmem
    Block_reg_getSSDGradient = 320; // 12 reg - 24 smem - 08 cmem
    /* _reg_tools_gpu */
    Block_reg_voxelCentric2NodeCentric = 320; // 11 reg - 24 smem - 16 cmem
    Block_reg_convertNMIGradientFromVoxelToRealSpace = 512; // 16 reg - 24 smem
    Block_reg_ApplyConvolutionWindowAlongX = 512; // 14 reg - 28 smem - 08 cmem
    Block_reg_ApplyConvolutionWindowAlongY = 512; // 14 reg - 28 smem - 08 cmem
    Block_reg_ApplyConvolutionWindowAlongZ = 512; // 15 reg - 28 smem - 08 cmem
    Block_reg_arithmetic = 384; // 5 reg - 24 smem
    /* _reg_resampling_gpu */
    Block_reg_resampleImage2D = 320; // 10 reg - 24 smem - 12 cmem
    Block_reg_resampleImage3D = 512; // 16 reg - 24 smem - 12 cmem
    Block_reg_getImageGradient2D = 512; // 16 reg - 24 smem - 20 cmem - 24 lmem
    Block_reg_getImageGradient3D = 320; // 24 reg - 24 smem - 16 cmem - 32 lmem
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] NiftyReg_CudaBlock100 constructor called\n");
#endif
}
/* ******************************** */
NiftyReg_CudaBlock200::NiftyReg_CudaBlock200()
{
//    Block_target_block = ; //
//    Block_result_block = ; //
//    /* _reg_mutualinformation_gpu */
//    Block_reg_smoothJointHistogramX = ; //
//    Block_reg_smoothJointHistogramY = ; //
//    Block_reg_smoothJointHistogramZ = ; //
//    Block_reg_smoothJointHistogramW = ; //
//    Block_reg_marginaliseTargetX = ; //
//    Block_reg_marginaliseTargetXY = ; //
//    Block_reg_marginaliseResultX = ; //
//    Block_reg_marginaliseResultXY = ; //
//    Block_reg_getVoxelBasedNMIGradientUsingPW2D = ; //
//    Block_reg_getVoxelBasedNMIGradientUsingPW3D = ; //
//    Block_reg_getVoxelBasedNMIGradientUsingPW2x2 = ; //
//    /* _reg_globalTransformation_gpu */
//    Block_reg_affine_deformationField = ; //
//    /* _reg_localTransformation_gpu */
//    Block_reg_spline_getDeformationField2D = ; //
//    Block_reg_spline_getDeformationField3D = ; //
//    Block_reg_spline_getApproxSecondDerivatives2D = ; //
//    Block_reg_spline_getApproxSecondDerivatives3D = ; //
//    Block_reg_spline_getApproxBendingEnergy2D = ; //
//    Block_reg_spline_getApproxBendingEnergy3D = ; //
//    Block_reg_spline_getApproxBendingEnergyGradient2D = ; //
//    Block_reg_spline_getApproxBendingEnergyGradient3D = ; //
//    Block_reg_spline_getApproxJacobianValues2D = ; //
//    Block_reg_spline_getApproxJacobianValues3D = ; //
//    Block_reg_spline_getJacobianValues2D = ; //
//    Block_reg_spline_getJacobianValues3D = ; //
//    Block_reg_spline_logSquaredValues = ; //
//    Block_reg_spline_computeApproxJacGradient2D = ; //
//    Block_reg_spline_computeApproxJacGradient3D = ; //
//    Block_reg_spline_computeJacGradient2D = ; //
//    Block_reg_spline_computeJacGradient3D = ; //
//    Block_reg_spline_approxCorrectFolding3D = ; //
//    Block_reg_spline_correctFolding3D = ; //
//    Block_reg_getDeformationFromDisplacement = ; //
//    Block_reg_getDisplacementFromDeformation = ; //
//    Block_reg_defField_compose2D = ; //
//    Block_reg_defField_compose3D = ; //
//    Block_reg_defField_getJacobianMatrix = ; //
//    /* _reg_optimiser_gpu */
//    Block_reg_initialiseConjugateGradient = ; //
//    Block_reg_GetConjugateGradient1 = ; //
//    Block_reg_GetConjugateGradient2 = ; //
//    Block_reg_getEuclideanDistance = ; //
//    Block_reg_updateControlPointPosition = ; //
//    /* _reg_ssd_gpu */
//    Block_reg_getSquaredDifference = ; //
//    Block_reg_getSSDGradient = ; //
//    /* _reg_tools_gpu */
//    Block_reg_voxelCentric2NodeCentric = ; //
//    Block_reg_convertNMIGradientFromVoxelToRealSpace = ; //
//    Block_reg_ApplyConvolutionWindowAlongX = ; //
//    Block_reg_ApplyConvolutionWindowAlongY = ; //
//    Block_reg_ApplyConvolutionWindowAlongZ = ; //
//    Block_reg_arithmetic = ; //
//    /* _reg_resampling_gpu */
//    Block_reg_resampleImage2D = ; //
//    Block_reg_resampleImage3D = ; //
//    Block_reg_getImageGradient2D = ; //
//    Block_reg_getImageGradient3D = ; //
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] NiftyReg_CudaBlock200 constructor called\n");
#endif
}
/* ******************************** */
NiftyReg_CudaBlock300::NiftyReg_CudaBlock300()
{
    Block_target_block = 640; // 45 reg
    Block_result_block = 640; // 47 reg - ????? smem
    /* _reg_mutualinformation_gpu */
    Block_reg_smoothJointHistogramX = 768; // 34 reg
    Block_reg_smoothJointHistogramY = 768; // 34 reg
    Block_reg_smoothJointHistogramZ = 768; // 34 reg
    Block_reg_smoothJointHistogramW = 768; // 34 reg
    Block_reg_marginaliseTargetX = 1024; // 24 reg
    Block_reg_marginaliseTargetXY = 1024; // 24 reg
    Block_reg_marginaliseResultX = 1024; // 24 reg
    Block_reg_marginaliseResultXY = 1024; // 24 reg
    Block_reg_getVoxelBasedNMIGradientUsingPW2D = 768; // 38 reg
    Block_reg_getVoxelBasedNMIGradientUsingPW3D = 640; // 45 reg
    Block_reg_getVoxelBasedNMIGradientUsingPW2x2 = 576; // 55 reg
    /* _reg_globalTransformation_gpu */
    Block_reg_affine_deformationField = 1024; // 23 reg
    /* _reg_localTransformation_gpu */
    Block_reg_spline_getDeformationField2D = 768; // 34 reg
    Block_reg_spline_getDeformationField3D = 768; // 34 reg
    Block_reg_spline_getApproxSecondDerivatives2D = 1024; // 25 reg
    Block_reg_spline_getApproxSecondDerivatives3D = 768; // 34 reg
    Block_reg_spline_getApproxBendingEnergy2D = 1024; // 23 reg
    Block_reg_spline_getApproxBendingEnergy3D = 1024; // 23 reg
    Block_reg_spline_getApproxBendingEnergyGradient2D = 1024; // 28 reg
    Block_reg_spline_getApproxBendingEnergyGradient3D = 768; // 33 reg
    Block_reg_spline_getApproxJacobianValues2D = 768; // 34 reg
    Block_reg_spline_getApproxJacobianValues3D = 640; // 46 reg
    Block_reg_spline_getJacobianValues2D = 768; // 34 reg
    Block_reg_spline_getJacobianValues3D = 768; // 34 reg
    Block_reg_spline_logSquaredValues = 1024; // 23 reg
    Block_reg_spline_computeApproxJacGradient2D = 768; // 34 reg
    Block_reg_spline_computeApproxJacGradient3D = 768; // 38 reg
    Block_reg_spline_computeJacGradient2D = 768; // 34 reg
    Block_reg_spline_computeJacGradient3D = 768; // 37 reg
    Block_reg_spline_approxCorrectFolding3D = 768; // 34 reg
    Block_reg_spline_correctFolding3D = 768; // 34 reg
    Block_reg_getDeformationFromDisplacement = 1024; // 18 reg
    Block_reg_getDisplacementFromDeformation = 1024; // 18 reg
    Block_reg_defField_compose2D = 1024; // 23 reg
    Block_reg_defField_compose3D = 1024; // 24 reg
    Block_reg_defField_getJacobianMatrix = 768; // 34 reg
    /* _reg_optimiser_gpu */
    Block_reg_initialiseConjugateGradient = 1024; // 20 reg
    Block_reg_GetConjugateGradient1 = 1024; // 22 reg
    Block_reg_GetConjugateGradient2 = 1024; // 25 reg
    Block_reg_getEuclideanDistance = 1024; // 20 reg
    Block_reg_updateControlPointPosition = 1024; // 22 reg
    /* _reg_ssd_gpu */
    Block_reg_getSquaredDifference = 768; // 34 reg
    Block_reg_getSSDGradient = 768; // 34 reg
    /* _reg_tools_gpu */
    Block_reg_voxelCentric2NodeCentric = 1024; // 23 reg
    Block_reg_convertNMIGradientFromVoxelToRealSpace = 1024; // 23 reg
    Block_reg_ApplyConvolutionWindowAlongX = 1024; // 25 reg
    Block_reg_ApplyConvolutionWindowAlongY = 1024; // 25 reg
    Block_reg_ApplyConvolutionWindowAlongZ = 1024; // 25 reg
    Block_reg_arithmetic = 1024; //
    /* _reg_resampling_gpu */
    Block_reg_resampleImage2D = 1024; // 23 reg
    Block_reg_resampleImage3D = 1024; // 24 reg
    Block_reg_getImageGradient2D = 768; // 34 reg
    Block_reg_getImageGradient3D = 768; // 34 reg
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] NiftyReg_CudaBlock300 constructor called\n");
#endif
}

#endif
