/*
 *  _reg_spline_gpu.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _reg_spline_GPU_CU
#define _reg_spline_GPU_CU

#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */
void reg_spline_getDeformationField_gpu(nifti_image *controlPointImage,
                                        nifti_image *reference,
                                        float4 **controlPointImageArray_d,
                                        float4 **positionFieldImageArray_d,
                                        int **mask_d,
                                        int activeVoxelNumber,
                                        bool bspline)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int useBSpline = static_cast<int>(bspline);

    const float3 controlPointVoxelSpacing = make_float3(
        controlPointImage->dx / reference->dx,
        controlPointImage->dy / reference->dy,
        controlPointImage->dz / reference->dz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_UseBSpline,&useBSpline,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))

    if(reference->nz>1){
        const unsigned int Grid_reg_spline_getDeformationField3D =
            (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(NR_BLOCK->Block_reg_spline_getDeformationField3D)));
        dim3 G1(Grid_reg_spline_getDeformationField3D,Grid_reg_spline_getDeformationField3D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getDeformationField3D,1,1);
        // 8 floats of shared memory are allocated per thread
        reg_spline_getDeformationField3D
                <<< G1, B1, NR_BLOCK->Block_reg_spline_getDeformationField3D*8*sizeof(float) >>>
                (*positionFieldImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const unsigned int Grid_reg_spline_getDeformationField2D =
            (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(NR_BLOCK->Block_reg_spline_getDeformationField2D)));
        dim3 G1(Grid_reg_spline_getDeformationField2D,Grid_reg_spline_getDeformationField2D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getDeformationField2D,1,1);
        // 4 floats of shared memory are allocated per thread
        reg_spline_getDeformationField2D
                <<< G1, B1, NR_BLOCK->Block_reg_spline_getDeformationField2D*4*sizeof(float) >>>
                   (*positionFieldImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
    return;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_spline_approxBendingEnergy_gpu(nifti_image *controlPointImage,
                                          float4 **controlPointImageArray_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    if(controlPointImage->nz>1){
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointGridMem))
        const unsigned int Grid_bspline_getApproxSecondDerivatives =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives3D)));
        dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives3D,1,1);
        reg_spline_getApproxSecondDerivatives3D <<< G1, B1 >>>(secondDerivativeValues_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 3*controlPointGridMem))
                const unsigned int Grid_bspline_getApproxSecondDerivatives =
                    (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives2D)));
        dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives2D,1,1);
        reg_spline_getApproxSecondDerivatives2D <<< G1, B1 >>>(secondDerivativeValues_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the bending energy from the second derivatives
    float *penaltyTerm_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&penaltyTerm_d, controlPointNumber*sizeof(float)))

    if(controlPointImage->nz>1){
        NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                          secondDerivativeValues_d,
                                          6*controlPointGridMem))
        const unsigned int Grid_reg_spline_ApproxBendingEnergy =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxBendingEnergy3D)));
        dim3 G2(Grid_reg_spline_ApproxBendingEnergy,Grid_reg_spline_ApproxBendingEnergy,1);
        dim3 B2(NR_BLOCK->Block_reg_spline_getApproxBendingEnergy3D,1,1);
        reg_spline_getApproxBendingEnergy3D_kernel <<< G2, B2 >>>(penaltyTerm_d);
        NR_CUDA_CHECK_KERNEL(G2,B2)
    }
    else{
        NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                          secondDerivativeValues_d,
                                          3*controlPointGridMem))
        const unsigned int Grid_reg_spline_ApproxBendingEnergy =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxBendingEnergy2D)));
        dim3 G2(Grid_reg_spline_ApproxBendingEnergy,Grid_reg_spline_ApproxBendingEnergy,1);
        dim3 B2(NR_BLOCK->Block_reg_spline_getApproxBendingEnergy2D,1,1);
        reg_spline_getApproxBendingEnergy2D_kernel <<< G2, B2 >>>(penaltyTerm_d);
        NR_CUDA_CHECK_KERNEL(G2,B2)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    // Compute the mean bending energy value
    double penaltyValue=reg_sumReduction_gpu(penaltyTerm_d,controlPointNumber);
    NR_CUDA_SAFE_CALL(cudaFree(penaltyTerm_d))

    return (float)(penaltyValue/(double)controlPointImage->nvox);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_approxBendingEnergyGradient_gpu(nifti_image *controlPointImage,
                                                float4 **controlPointImageArray_d,
                                                float4 **nodeGradientArray_d,
                                                float bendingEnergyWeight)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    if(controlPointImage->nz>1){
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointNumber*sizeof(float4)))
        const unsigned int Grid_bspline_getApproxSecondDerivatives =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives3D)));
        dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives3D,1,1);
        reg_spline_getApproxSecondDerivatives3D <<< G1, B1 >>>(secondDerivativeValues_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 3*controlPointNumber*sizeof(float4)))
        const unsigned int Grid_bspline_getApproxSecondDerivatives =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives2D)));
        dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxSecondDerivatives2D,1,1);
        reg_spline_getApproxSecondDerivatives2D <<< G1, B1 >>>(secondDerivativeValues_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the gradient
    bendingEnergyWeight *= 1.f / (float)controlPointNumber;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&bendingEnergyWeight,sizeof(float)))            
    if(controlPointImage->nz>1){
        NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                          secondDerivativeValues_d,
                                          6*controlPointNumber*sizeof(float4)))
        const unsigned int Grid_reg_spline_getApproxBendingEnergyGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxBendingEnergyGradient3D)));
        dim3 G2(Grid_reg_spline_getApproxBendingEnergyGradient,Grid_reg_spline_getApproxBendingEnergyGradient,1);
        dim3 B2(NR_BLOCK->Block_reg_spline_getApproxBendingEnergyGradient3D,1,1);
        reg_spline_getApproxBendingEnergyGradient3D_kernel <<< G2, B2 >>>(*nodeGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G2,B2)
    }
    else{
        NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                          secondDerivativeValues_d,
                                          3*controlPointNumber*sizeof(float4)))
        const unsigned int Grid_reg_spline_getApproxBendingEnergyGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxBendingEnergyGradient2D)));
        dim3 G2(Grid_reg_spline_getApproxBendingEnergyGradient,Grid_reg_spline_getApproxBendingEnergyGradient,1);
        dim3 B2(NR_BLOCK->Block_reg_spline_getApproxBendingEnergyGradient2D,1,1);
        reg_spline_getApproxBendingEnergyGradient2D_kernel <<< G2, B2 >>>(*nodeGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G2,B2)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_ComputeApproxJacobianValues(nifti_image *controlPointImage,
                                             float4 **controlPointImageArray_d,
                                             float **jacobianMatrices_d,
                                             float **jacobianDet_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorientation;
    if(controlPointImage->sform_code>0)
        reorientation=reg_mat44_to_mat33(&controlPointImage->sto_xyz);
    else reorientation=reg_mat44_to_mat33(&controlPointImage->qto_xyz);
    float3 temp=make_float3(reorientation.m[0][0],reorientation.m[0][1],reorientation.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[1][0],reorientation.m[1][1],reorientation.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[2][0],reorientation.m[2][1],reorientation.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // The Jacobian matrix is computed for every control point
	if(controlPointImage->nz>1){
		const unsigned int Grid_reg_spline_getApproxJacobianValues3D =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxJacobianValues3D)));
		dim3 G1(Grid_reg_spline_getApproxJacobianValues3D,Grid_reg_spline_getApproxJacobianValues3D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxJacobianValues3D,1,1);
		reg_spline_getApproxJacobianValues3D_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	else{
		const unsigned int Grid_reg_spline_getApproxJacobianValues2D =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_getApproxJacobianValues2D)));
		dim3 G1(Grid_reg_spline_getApproxJacobianValues2D,Grid_reg_spline_getApproxJacobianValues2D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getApproxJacobianValues2D,1,1);
		reg_spline_getApproxJacobianValues2D_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
void reg_spline_ComputeJacobianValues(nifti_image *controlPointImage,
                                       nifti_image *referenceImage,
                                       float4 **controlPointImageArray_d,
                                       float **jacobianMatrices_d,
                                       float **jacobianDet_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorientation;
    if(controlPointImage->sform_code>0)
        reorientation=reg_mat44_to_mat33(&controlPointImage->sto_xyz);
    else reorientation=reg_mat44_to_mat33(&controlPointImage->qto_xyz);
    float3 temp=make_float3(reorientation.m[0][0],reorientation.m[0][1],reorientation.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[1][0],reorientation.m[1][1],reorientation.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[2][0],reorientation.m[2][1],reorientation.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    // Bind some variables
    const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    const float3 controlPointVoxelSpacing = make_float3(
            controlPointImage->dx / referenceImage->dx,
            controlPointImage->dy / referenceImage->dy,
            controlPointImage->dz / referenceImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointNumber*sizeof(float4)))

    // The Jacobian matrix is computed for every voxel
	if(controlPointImage->nz>1){
		const unsigned int Grid_reg_spline_getJacobianValues3D =
            (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_spline_getJacobianValues3D)));
		dim3 G1(Grid_reg_spline_getJacobianValues3D,Grid_reg_spline_getJacobianValues3D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getJacobianValues3D,1,1);
        // 8 floats of shared memory are allocated per thread
        reg_spline_getJacobianValues3D_kernel
                <<< G1, B1, NR_BLOCK->Block_reg_spline_getJacobianValues3D*8*sizeof(float)>>>
                (*jacobianMatrices_d, *jacobianDet_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	else{
		const unsigned int Grid_reg_spline_getJacobianValues2D =
            (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_spline_getJacobianValues2D)));
		dim3 G1(Grid_reg_spline_getJacobianValues2D,Grid_reg_spline_getJacobianValues2D,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_getJacobianValues2D,1,1);
        reg_spline_getJacobianValues2D_kernel
                <<< G1, B1>>>
                (*jacobianMatrices_d, *jacobianDet_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
/* *************************************************************** */
double reg_spline_getJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
                                             nifti_image *controlPointImage,
                                             float4 **controlPointImageArray_d,
                                             bool approx
                                             )
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
		jacNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
		jacSum = (controlPointImage->nx-2)*(controlPointImage->ny-2);
		if(controlPointImage->nz>1){
			jacSum *= controlPointImage->nz-2;
			// Allocate array for 3x3 matrices
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
		}
		else{
			// Allocate array for 2x2 matrices
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,4*jacNumber*sizeof(float)))
		}
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
		reg_spline_ComputeApproxJacobianValues(controlPointImage,
											   controlPointImageArray_d,
											   &jacobianMatrices_d,
											   &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        jacSum=jacNumber;
		if(controlPointImage->nz>1){
			// Allocate array for 3x3 matrices
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
		}
		else{
			// Allocate array for 2x2 matrices
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,4*jacNumber*sizeof(float)))
		}
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
		reg_spline_ComputeJacobianValues(controlPointImage,
										 referenceImage,
										 controlPointImageArray_d,
										 &jacobianMatrices_d,
										 &jacobianDet_d);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))

    // The Jacobian determinant are squared and logged (might not be english but will do)
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    const unsigned int Grid_reg_spline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(NR_BLOCK->Block_reg_spline_logSquaredValues)));
    dim3 G1(Grid_reg_spline_logSquaredValues,Grid_reg_spline_logSquaredValues,1);
    dim3 B1(NR_BLOCK->Block_reg_spline_logSquaredValues,1,1);
    reg_spline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
	// Perform the reduction
	double penaltyTermValue = reg_sumReduction_gpu(jacobianDet_d,jacNumber);
	NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    return penaltyTermValue/jacSum;
}
/* *************************************************************** */
void reg_spline_getJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
                                                   nifti_image *controlPointImage,
                                                   float4 **controlPointImageArray_d,
                                                   float4 **nodeGradientArray_d,
                                                   float jacobianWeight,
                                                   bool approx)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
		if(controlPointImage->nz>1)
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
		else NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,4*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_spline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
		jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
		if(controlPointImage->nz>1)
			NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
		else NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,4*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_spline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorientation;
    if(controlPointImage->sform_code>0)
        reorientation=reg_mat44_to_mat33(&controlPointImage->sto_ijk);
    else reorientation=reg_mat44_to_mat33(&controlPointImage->qto_ijk);
    float3 temp=make_float3(reorientation.m[0][0],reorientation.m[0][1],reorientation.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[1][0],reorientation.m[1][1],reorientation.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[2][0],reorientation.m[2][1],reorientation.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
	if(controlPointImage->nz>1)
		NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
										  9*jacNumber*sizeof(float)))
	else NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
										   4*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    float3 weight=make_float3(
                referenceImage->dx*jacobianWeight / ((float)jacNumber*controlPointImage->dx),
                referenceImage->dy*jacobianWeight / ((float)jacNumber*controlPointImage->dy),
                referenceImage->dz*jacobianWeight / ((float)jacNumber*controlPointImage->dz));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight3,&weight,sizeof(float3)))
    if(approx){
		if(controlPointImage->nz>1){
			const unsigned int Grid_reg_spline_computeApproxJacGradient3D =
                (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_computeApproxJacGradient3D)));
			dim3 G1(Grid_reg_spline_computeApproxJacGradient3D,Grid_reg_spline_computeApproxJacGradient3D,1);
            dim3 B1(NR_BLOCK->Block_reg_spline_computeApproxJacGradient3D,1,1);
			reg_spline_computeApproxJacGradient3D_kernel<<< G1, B1>>>(*nodeGradientArray_d);
			NR_CUDA_CHECK_KERNEL(G1,B1)
		}
		else{
			const unsigned int Grid_reg_spline_computeApproxJacGradient2D =
                (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_computeApproxJacGradient2D)));
			dim3 G1(Grid_reg_spline_computeApproxJacGradient2D,Grid_reg_spline_computeApproxJacGradient2D,1);
            dim3 B1(NR_BLOCK->Block_reg_spline_computeApproxJacGradient2D,1,1);
			reg_spline_computeApproxJacGradient2D_kernel<<< G1, B1>>>(*nodeGradientArray_d);
			NR_CUDA_CHECK_KERNEL(G1,B1)
		}
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
		if(controlPointImage->nz>1){
			const unsigned int Grid_reg_spline_computeJacGradient3D =
                (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_computeJacGradient3D)));
			dim3 G1(Grid_reg_spline_computeJacGradient3D,Grid_reg_spline_computeJacGradient3D,1);
            dim3 B1(NR_BLOCK->Block_reg_spline_computeJacGradient3D,1,1);
			reg_spline_computeJacGradient3D_kernel<<< G1, B1>>>(*nodeGradientArray_d);
			NR_CUDA_CHECK_KERNEL(G1,B1)
		}
		else{
			const unsigned int Grid_reg_spline_computeJacGradient2D =
                (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_computeJacGradient2D)));
			dim3 G1(Grid_reg_spline_computeJacGradient2D,Grid_reg_spline_computeJacGradient2D,1);
            dim3 B1(NR_BLOCK->Block_reg_spline_computeJacGradient2D,1,1);
			reg_spline_computeJacGradient2D_kernel<<< G1, B1>>>(*nodeGradientArray_d);
			NR_CUDA_CHECK_KERNEL(G1,B1)
		}
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
}
/* *************************************************************** */
double reg_spline_correctFolding_gpu(nifti_image *referenceImage,
                                      nifti_image *controlPointImage,
                                      float4 **controlPointImageArray_d,
                                      bool approx)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum = (controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_spline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacSum=jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_spline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Check if the Jacobian determinant average
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    float *jacobianDet2_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet2_d,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet2_d,jacobianDet_d,jacNumber*sizeof(float),cudaMemcpyDeviceToDevice))
    const unsigned int Grid_reg_spline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(NR_BLOCK->Block_reg_spline_logSquaredValues)));
    dim3 G1(Grid_reg_spline_logSquaredValues,Grid_reg_spline_logSquaredValues,1);
    dim3 B1(NR_BLOCK->Block_reg_spline_logSquaredValues,1,1);
    reg_spline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet2_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet2_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet2_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i) penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    penaltyTermValue /= jacSum;
    if(penaltyTermValue==penaltyTermValue){
        NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
        NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
        return penaltyTermValue;
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorientation;
    if(controlPointImage->sform_code>0)
        reorientation=reg_mat44_to_mat33(&controlPointImage->sto_ijk);
    else reorientation=reg_mat44_to_mat33(&controlPointImage->qto_ijk);
    float3 temp=make_float3(reorientation.m[0][0],reorientation.m[0][1],reorientation.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[1][0],reorientation.m[1][1],reorientation.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[2][0],reorientation.m[2][1],reorientation.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianDeterminantTexture, jacobianDet_d,
                                      jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,jacobianMatricesTexture, jacobianMatrices_d,
                                      9*jacNumber*sizeof(float)))

    // Bind some variables
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const float3 controlPointSpacing = make_float3(controlPointImage->dx,controlPointImage->dy,controlPointImage->dz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointSpacing,&controlPointSpacing,sizeof(float3)))
    if(approx){
        const unsigned int Grid_reg_spline_approxCorrectFolding =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_approxCorrectFolding3D)));
        dim3 G1(Grid_reg_spline_approxCorrectFolding,Grid_reg_spline_approxCorrectFolding,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_approxCorrectFolding3D,1,1);
        reg_spline_approxCorrectFolding3D_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
        const int3 referenceImageDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
        const float3 controlPointVoxelSpacing = make_float3(
                controlPointImage->dx / referenceImage->dx,
                controlPointImage->dy / referenceImage->dy,
                controlPointImage->dz / referenceImage->dz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointVoxelSpacing,&controlPointVoxelSpacing,sizeof(float3)))
        const unsigned int Grid_reg_spline_correctFolding =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(NR_BLOCK->Block_reg_spline_correctFolding3D)));
        dim3 G1(Grid_reg_spline_correctFolding,Grid_reg_spline_correctFolding,1);
        dim3 B1(NR_BLOCK->Block_reg_spline_correctFolding3D,1,1);
        reg_spline_correctFolding3D_kernel<<< G1, B1>>>(*controlPointImageArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
    return std::numeric_limits<double>::quiet_NaN();
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFromDisplacement_gpu( nifti_image *image, float4 **imageArray_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // Bind the qform or sform
    mat44 temp_mat=image->qto_xyz;
    if(image->sform_code>0) temp_mat=image->sto_xyz;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    const int voxelNumber=image->nx*image->ny*image->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))

    const int3 imageDim=make_int3(image->nx,image->ny,image->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&imageDim,sizeof(int3)))

    const unsigned int Grid_reg_getDeformationFromDisplacement =
    (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_getDeformationFromDisplacement)));
    dim3 G1(Grid_reg_getDeformationFromDisplacement,Grid_reg_getDeformationFromDisplacement,1);
    dim3 B1(NR_BLOCK->Block_reg_getDeformationFromDisplacement,1,1);
    reg_getDeformationFromDisplacement3D_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDisplacementFromDeformation_gpu( nifti_image *image, float4 **imageArray_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    // Bind the qform or sform
    mat44 temp_mat=image->qto_xyz;
    if(image->sform_code>0) temp_mat=image->sto_xyz;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    const int voxelNumber=image->nx*image->ny*image->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))

    const int3 imageDim=make_int3(image->nx,image->ny,image->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&imageDim,sizeof(int3)))

    const unsigned int Grid_reg_getDisplacementFromDeformation =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_getDisplacementFromDeformation)));
    dim3 G1(Grid_reg_getDisplacementFromDeformation,Grid_reg_getDisplacementFromDeformation,1);
    dim3 B1(NR_BLOCK->Block_reg_getDisplacementFromDeformation,1,1);
    reg_getDisplacementFromDeformation3D_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
                                                 nifti_image *def_h,
                                                 float4 **cpp_gpu,
                                                 float4 **def_gpu)
{
    const int voxelNumber = def_h->nx * def_h->ny * def_h->nz;

    // Create a mask array where no voxel are excluded
    int *mask_gpu=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&mask_gpu, voxelNumber*sizeof(int)))
    reg_fillMaskArray_gpu(voxelNumber,&mask_gpu);

    // Define some variables for the deformation fields
    float4 *tempDef_gpu=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&tempDef_gpu,voxelNumber*sizeof(float4)))

    // The deformation field is computed
    reg_spline_getDeformationField_gpu(cpp_h,
                                       def_h,
                                       cpp_gpu,
                                       def_gpu,
                                       &mask_gpu,
                                       voxelNumber,
                                       true); // non-interpolant spline are used

    // The deformation field is converted into a displacement field
    reg_getDisplacementFromDeformation_gpu(def_h,def_gpu);

    // Scaling of the deformation field
    float scalingValue = pow(2.0f,fabs(cpp_h->intent_p1));
    if(cpp_h->intent_p1<0)
        // backward deformation field is scaled down
        reg_multiplyValue_gpu(voxelNumber,
                              def_gpu,
                              -1.f/scalingValue);
    else
        // forward deformation field is scaled down
        reg_multiplyValue_gpu(voxelNumber,
                              def_gpu,
                              1.f/scalingValue);

    // The displacement field is converted back into a deformation field
    reg_getDeformationFromDisplacement_gpu(def_h,def_gpu);


    // The deformation field is squared
    unsigned int squaringNumber = (unsigned int)fabs(cpp_h->intent_p1);
    for(unsigned int i=0;i<squaringNumber;++i){

        // The deformation field arrays are updated
        NR_CUDA_SAFE_CALL(cudaMemcpy(tempDef_gpu,*def_gpu,voxelNumber*sizeof(float4),cudaMemcpyDeviceToDevice))

        // The deformation fields are composed
        reg_defField_compose_gpu(def_h,
                                 &tempDef_gpu,
                                 def_gpu,
                                 &mask_gpu,
                                 voxelNumber);
    }

    NR_CUDA_SAFE_CALL(cudaFree(tempDef_gpu))
    NR_CUDA_SAFE_CALL(cudaFree(mask_gpu))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_compose_gpu(nifti_image *def,
                              float4 **def_gpu,
                              float4 **defOut_gpu,
                              int **mask_gpu,
                              int activeVoxel)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int voxelNumber=def->nx*def->ny*def->nz;

    // Bind the qform or sform
    mat44 temp_mat=def->qto_ijk;
    if(def->sform_code>0) temp_mat=def->sto_ijk;
    float4 temp;
    temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1b,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2b,&temp,sizeof(float4)))

    temp_mat=def->qto_xyz;
    if(def->sform_code>0) temp_mat=def->sto_xyz;
    temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0c,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[1][0],temp_mat.m[1][1],temp_mat.m[1][2],temp_mat.m[1][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1c,&temp,sizeof(float4)))
    temp=make_float4(temp_mat.m[2][0],temp_mat.m[2][1],temp_mat.m[2][2],temp_mat.m[2][3]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2c,&temp,sizeof(float4)))

    const int3 referenceImageDim=make_int3(def->nx,def->ny,def->nz);

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDeformationTexture,*def_gpu,activeVoxel*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,maskTexture,*mask_gpu,activeVoxel*sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int3)))

    if(def->nz>1){
        const unsigned int Grid_reg_defField_compose3D =
            (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_defField_compose3D)));
        dim3 G1(Grid_reg_defField_compose3D,Grid_reg_defField_compose3D,1);
        dim3 B1(NR_BLOCK->Block_reg_defField_compose3D,1,1);
        reg_defField_compose3D_kernel<<< G1, B1>>>(*defOut_gpu);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    else{
        const unsigned int Grid_reg_defField_compose2D =
            (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_defField_compose2D)));
        dim3 G1(Grid_reg_defField_compose2D,Grid_reg_defField_compose2D,1);
        dim3 B1(NR_BLOCK->Block_reg_defField_compose2D,1,1);
        reg_defField_compose2D_kernel<<< G1, B1>>>(*defOut_gpu);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDeformationTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_getJacobianMatrix_gpu(nifti_image *deformationField,
                                        float4 **deformationField_gpu,
                                        float **jacobianMatrices_gpu)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int3 referenceDim=make_int3(deformationField->nx,deformationField->ny,deformationField->nz);
    const float3 referenceSpacing=make_float3(deformationField->dx,deformationField->dy,deformationField->dz);
    const int voxelNumber = referenceDim.x*referenceDim.y*referenceDim.z;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceSpacing,&referenceSpacing,sizeof(float3)))

    mat33 reorientation;
    if(deformationField->sform_code>0)
        reorientation=reg_mat44_to_mat33(&deformationField->sto_xyz);
    else reorientation=reg_mat44_to_mat33(&deformationField->qto_xyz);
    float3 temp=make_float3(reorientation.m[0][0],reorientation.m[0][1],reorientation.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[1][0],reorientation.m[1][1],reorientation.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorientation.m[2][0],reorientation.m[2][1],reorientation.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDeformationTexture,*deformationField_gpu,voxelNumber*sizeof(float4)))

    const unsigned int Grid_reg_defField_getJacobianMatrix =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(NR_BLOCK->Block_reg_defField_getJacobianMatrix)));
    dim3 G1(Grid_reg_defField_getJacobianMatrix,Grid_reg_defField_getJacobianMatrix,1);
    dim3 B1(NR_BLOCK->Block_reg_defField_getJacobianMatrix);
    reg_defField_getJacobianMatrix3D_kernel<<<G1,B1>>>(*jacobianMatrices_gpu);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDeformationTexture))
}
/* *************************************************************** */
/* *************************************************************** */
#endif
