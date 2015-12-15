/*
 *  _reg_bspline_gpu.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BSPLINE_GPU_CU
#define _REG_BSPLINE_GPU_CU

#include "_reg_localTransformation_gpu.h"
#include "_reg_localTransformation_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_gpu(nifti_image *controlPointImage,
                     nifti_image *reference,
                     float4 **controlPointImageArray_d,
                     float4 **positionFieldImageArray_d,
                     int **mask_d,
                     int activeVoxelNumber,
                     bool bspline)
{
    const int voxelNumber = reference->nx * reference->ny * reference->nz;
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 referenceImageDim = make_int3(reference->nx, reference->ny, reference->nz);
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int useBSpline = bspline;

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

    const unsigned int Grid_reg_bspline_getDeformationField =
        (unsigned int)ceilf(sqrtf((float)activeVoxelNumber/(float)(Block_reg_bspline_getDeformationField)));
    dim3 G1(Grid_reg_bspline_getDeformationField,Grid_reg_bspline_getDeformationField,1);
    dim3 B1(Block_reg_bspline_getDeformationField,1,1);
    reg_bspline_getDeformationField <<< G1, B1 >>>(*positionFieldImageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
    return;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_bspline_ApproxBendingEnergy_gpu(nifti_image *controlPointImage,
                                          float4 **controlPointImageArray_d)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointGridMem))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives)));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the bending energy from the second derivatives
    float *penaltyTerm_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&penaltyTerm_d, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointGridMem))
    const unsigned int Grid_reg_bspline_ApproxBendingEnergy =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergy)));
    dim3 G2(Grid_reg_bspline_ApproxBendingEnergy,Grid_reg_bspline_ApproxBendingEnergy,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergy,1,1);
    reg_bspline_getApproxBendingEnergy_kernel <<< G2, B2 >>>(penaltyTerm_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    // Transfert the vales back to the CPU and average them
    float *penaltyTerm_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&penaltyTerm_h, controlPointNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(penaltyTerm_h, penaltyTerm_d, controlPointNumber*sizeof(float), cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(penaltyTerm_d))

    double penaltyValue=0.0;
    for(int i=0;i<controlPointNumber;i++)
            penaltyValue += penaltyTerm_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)penaltyTerm_h))
    return (float)(penaltyValue/(3.0*(double)controlPointNumber));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ApproxBendingEnergyGradient_gpu(nifti_image *referenceImage,
                                                 nifti_image *controlPointImage,
                                                 float4 **controlPointImageArray_d,
                                                 float4 **nodeNMIGradientArray_d,
                                                 float bendingEnergyWeight)
{
    const int controlPointNumber = controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
    const int3 controlPointImageDim = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
    const int controlPointGridMem = controlPointNumber*sizeof(float4);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointNumber,&controlPointNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&controlPointImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,controlPointTexture, *controlPointImageArray_d, controlPointGridMem))

    // First compute all the second derivatives
    float4 *secondDerivativeValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&secondDerivativeValues_d, 6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_bspline_getApproxSecondDerivatives =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxSecondDerivatives)));
    dim3 G1(Grid_bspline_getApproxSecondDerivatives,Grid_bspline_getApproxSecondDerivatives,1);
    dim3 B1(Block_reg_bspline_getApproxSecondDerivatives,1,1);
    reg_bspline_getApproxSecondDerivatives <<< G1, B1 >>>(secondDerivativeValues_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))

    // Compute the gradient
    bendingEnergyWeight *= referenceImage->nx*referenceImage->ny*referenceImage->nz /
                           (controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&bendingEnergyWeight,sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,secondDerivativesTexture,
                                      secondDerivativeValues_d,
                                      6*controlPointNumber*sizeof(float4)))
    const unsigned int Grid_reg_bspline_getApproxBendingEnergyGradient =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxBendingEnergyGradient)));
    dim3 G2(Grid_reg_bspline_getApproxBendingEnergyGradient,Grid_reg_bspline_getApproxBendingEnergyGradient,1);
    dim3 B2(Block_reg_bspline_getApproxBendingEnergyGradient,1,1);
    reg_bspline_getApproxBendingEnergyGradient_kernel <<< G2, B2 >>>(*nodeNMIGradientArray_d);
    NR_CUDA_CHECK_KERNEL(G2,B2)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(secondDerivativesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(secondDerivativeValues_d))

    return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_ComputeApproxJacobianValues(nifti_image *controlPointImage,
                                             float4 **controlPointImageArray_d,
                                             float **jacobianMatrices_d,
                                             float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
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
    const unsigned int Grid_reg_bspline_getApproxJacobianValues =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_getApproxJacobianValues)));
    dim3 G1(Grid_reg_bspline_getApproxJacobianValues,Grid_reg_bspline_getApproxJacobianValues,1);
    dim3 B1(Block_reg_bspline_getApproxJacobianValues,1,1);
    reg_bspline_getApproxJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianValues(nifti_image *controlPointImage,
                                       nifti_image *referenceImage,
                                       float4 **controlPointImageArray_d,
                                       float **jacobianMatrices_d,
                                       float **jacobianDet_d)
{
    // Need to reorient the Jacobian matrix using the header information - real to voxel conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
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
    const unsigned int Grid_reg_bspline_getJacobianValues =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_bspline_getJacobianValues)));
    dim3 G1(Grid_reg_bspline_getJacobianValues,Grid_reg_bspline_getJacobianValues,1);
    dim3 B1(Block_reg_bspline_getJacobianValues,1,1);
    reg_bspline_getJacobianValues_kernel<<< G1, B1>>>(*jacobianMatrices_d, *jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(controlPointTexture))
}
/* *************************************************************** */
/* *************************************************************** */
double reg_bspline_ComputeJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
                                                  nifti_image *controlPointImage,
                                                  float4 **controlPointImageArray_d,
                                                  bool approx
                                                  )
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    double jacSum;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        jacSum=(controlPointImage->nx-2)*(controlPointImage->ny-2)*(controlPointImage->nz-2);
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        jacSum=jacNumber;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))

    // The Jacobian determinant are squared and logged (might not be english but will do)
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&jacNumber,sizeof(int)))
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues)));
    dim3 G1(Grid_reg_bspline_logSquaredValues,Grid_reg_bspline_logSquaredValues,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    // Transfert the data back to the CPU
    float *jacobianDet_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&jacobianDet_h,jacNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(jacobianDet_h,jacobianDet_d,
                                 jacNumber*sizeof(float),
                                 cudaMemcpyDeviceToHost))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    double penaltyTermValue=0.;
    for(int i=0;i<jacNumber;++i)
        penaltyTermValue += jacobianDet_h[i];
    NR_CUDA_SAFE_CALL(cudaFreeHost(jacobianDet_h))
    return penaltyTermValue/jacSum;
}
/* *************************************************************** */
void reg_bspline_ComputeJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
                                                        nifti_image *controlPointImage,
                                                        float4 **controlPointImageArray_d,
                                                        float4 **nodeNMIGradientArray_d,
                                                        float jacobianWeight,
                                                        bool approx)
{
    // The Jacobian matrices and determinants are computed
    float *jacobianMatrices_d;
    float *jacobianDet_d;
    int jacNumber;
    if(approx){
        jacNumber=controlPointImage->nx*controlPointImage->ny*controlPointImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
                                          referenceImage,
                                          controlPointImageArray_d,
                                          &jacobianMatrices_d,
                                          &jacobianDet_d);
    }

    // Need to desorient the Jacobian matrix using the header information - voxel to real conversion
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
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
        float weight=jacobianWeight;
        weight = jacobianWeight * (float)(referenceImage->nx * referenceImage->ny * referenceImage->nz)
                 / (float)( controlPointImage->nx*controlPointImage->ny*controlPointImage->nz);
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeApproxJacGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_computeApproxJacGradient)));
        dim3 G1(Grid_reg_bspline_computeApproxJacGradient,Grid_reg_bspline_computeApproxJacGradient,1);
        dim3 B1(Block_reg_bspline_computeApproxJacGradient,1,1);
        reg_bspline_computeApproxJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
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
        NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&jacobianWeight,sizeof(float)))
        const unsigned int Grid_reg_bspline_computeJacGradient =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_computeJacGradient)));
        dim3 G1(Grid_reg_bspline_computeJacGradient,Grid_reg_bspline_computeJacGradient,1);
        dim3 B1(Block_reg_bspline_computeJacGradient,1,1);
        reg_bspline_computeJacGradient_kernel<<< G1, B1>>>(*nodeNMIGradientArray_d);
        NR_CUDA_CHECK_KERNEL(G1,B1)
    }
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianDeterminantTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(jacobianMatricesTexture))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianDet_d))
    NR_CUDA_SAFE_CALL(cudaFree(jacobianMatrices_d))
}
/* *************************************************************** */
double reg_bspline_correctFolding_gpu(nifti_image *referenceImage,
                                      nifti_image *controlPointImage,
                                      float4 **controlPointImageArray_d,
                                      bool approx)
{
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
        reg_bspline_ComputeApproxJacobianValues(controlPointImage,
                                                controlPointImageArray_d,
                                                &jacobianMatrices_d,
                                                &jacobianDet_d);
    }
    else{
        jacSum=jacNumber=referenceImage->nx*referenceImage->ny*referenceImage->nz;
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianMatrices_d,9*jacNumber*sizeof(float)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&jacobianDet_d,jacNumber*sizeof(float)))
        reg_bspline_ComputeJacobianValues(controlPointImage,
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
    const unsigned int Grid_reg_bspline_logSquaredValues =
        (unsigned int)ceilf(sqrtf((float)jacNumber/(float)(Block_reg_bspline_logSquaredValues)));
    dim3 G1(Grid_reg_bspline_logSquaredValues,Grid_reg_bspline_logSquaredValues,1);
    dim3 B1(Block_reg_bspline_logSquaredValues,1,1);
    reg_bspline_logSquaredValues_kernel<<< G1, B1>>>(jacobianDet2_d);
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
    mat33 reorient, desorient;
    reg_getReorientationMatrix(controlPointImage, &desorient, &reorient);
    float3 temp=make_float3(desorient.m[0][0],desorient.m[0][1],desorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[1][0],desorient.m[1][1],desorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(desorient.m[2][0],desorient.m[2][1],desorient.m[2][2]);
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
        const unsigned int Grid_reg_bspline_approxCorrectFolding =
            (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_approxCorrectFolding)));
        dim3 G1(Grid_reg_bspline_approxCorrectFolding,Grid_reg_bspline_approxCorrectFolding,1);
        dim3 B1(Block_reg_bspline_approxCorrectFolding,1,1);
        reg_bspline_approxCorrectFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
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
        const unsigned int Grid_reg_bspline_correctFolding =
        (unsigned int)ceilf(sqrtf((float)controlPointNumber/(float)(Block_reg_bspline_correctFolding)));
        dim3 G1(Grid_reg_bspline_correctFolding,Grid_reg_bspline_correctFolding,1);
        dim3 B1(Block_reg_bspline_correctFolding,1,1);
        reg_bspline_correctFolding_kernel<<< G1, B1>>>(*controlPointImageArray_d);
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
    (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(512)));
    dim3 G1(Grid_reg_getDeformationFromDisplacement,Grid_reg_getDeformationFromDisplacement,1);
    dim3 B1(512,1,1);
    reg_getDeformationFromDisplacement_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDisplacementFromDeformation_gpu( nifti_image *image, float4 **imageArray_d)
{
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
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(512)));
    dim3 G1(Grid_reg_getDisplacementFromDeformation,Grid_reg_getDisplacementFromDeformation,1);
    dim3 B1(512,1,1);
    reg_getDisplacementFromDeformation_kernel<<< G1, B1>>>(*imageArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
                                                 nifti_image *def_h,
                                                 float4 **cpp_gpu,
                                                 float4 **def_gpu,
                                                 float4 **interDef_gpu,
                                                 int **mask_gpu,
                                                 int activeVoxel,
                                                 bool approxComp)
{
    if(approxComp){
        fprintf(stderr, "[NiftyReg] reg_getDeformationFieldFromVelocityGrid_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR Approximation not implemented yet on the GPU\n");
        exit(1);
    }

    const int controlPointNumber = cpp_h->nx * cpp_h->ny * cpp_h->nz;
    const int voxelNumber = def_h->nx * def_h->ny * def_h->nz;

    if(voxelNumber != activeVoxel){
        fprintf(stderr, "[NiftyReg] reg_getDeformationFieldFromVelocityGrid_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR The mask must contains all voxel\n");
        exit(1);
    }

    // A scaled down velocity field is first store
    float4 *scaledVelocityField_d=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&scaledVelocityField_d,controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(scaledVelocityField_d,*cpp_gpu,controlPointNumber*sizeof(float4),cudaMemcpyDeviceToDevice))
    reg_getDisplacementFromDeformation_gpu(cpp_h, &scaledVelocityField_d);
    reg_multiplyValue_gpu(controlPointNumber,&scaledVelocityField_d,1.f/cpp_h->pixdim[5]);
    reg_getDeformationFromDisplacement_gpu(cpp_h, &scaledVelocityField_d);

    if(!approxComp){
        float4 *tempDef=NULL;
        float4 *currentDefPtr0=NULL;
        float4 *currentDefPtr1=NULL;
        if(interDef_gpu==NULL){
            NR_CUDA_SAFE_CALL(cudaMalloc(&tempDef,voxelNumber*sizeof(float4)))
            currentDefPtr0 = *def_gpu;
            currentDefPtr1 = tempDef;
        }
        else{
            currentDefPtr0 = interDef_gpu[0];
            currentDefPtr1 = interDef_gpu[1];
        }
        reg_bspline_gpu(cpp_h,
                        def_h,
                        &scaledVelocityField_d,
                        &currentDefPtr0,
                        mask_gpu,
                        activeVoxel,
                        true);

        for(unsigned int i=0;i<cpp_h->pixdim[5];++i){

            NR_CUDA_SAFE_CALL(cudaMemcpy(currentDefPtr1,currentDefPtr0,voxelNumber*sizeof(float4),cudaMemcpyDeviceToDevice))

            if(interDef_gpu==NULL){
                reg_defField_compose_gpu(def_h,
                                         &currentDefPtr1,
                                         &currentDefPtr0,
                                         mask_gpu,
                                         activeVoxel);
            }
            else{
                reg_defField_compose_gpu(def_h,
                                         &currentDefPtr0,
                                         &currentDefPtr1,
                                         mask_gpu,
                                         activeVoxel);
                if(i==cpp_h->pixdim[5]-2){
                    currentDefPtr0 = interDef_gpu[i+1];
                    currentDefPtr1 = *def_gpu;
                }
                else if(i<cpp_h->pixdim[5]-2){
                    currentDefPtr0 = interDef_gpu[i+1];
                    currentDefPtr1 = interDef_gpu[i+2];
                }
            }
        }
        if(tempDef!=NULL) NR_CUDA_SAFE_CALL(cudaFree(tempDef));
    }
    NR_CUDA_SAFE_CALL(cudaFree(scaledVelocityField_d))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getInverseDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
                                                        nifti_image *def_h,
                                                        float4 **cpp_gpu,
                                                        float4 **def_gpu,
                                                        float4 **interDef_gpu,
                                                        int **mask_gpu,
                                                        int activeVoxel,
                                                        bool approxComp)
{
    const int controlPointNumber = cpp_h->nx * cpp_h->ny * cpp_h->nz;
    // The CPP file is first negated
    float4 *invertedCpp_gpu=NULL;
    NR_CUDA_SAFE_CALL(cudaMalloc(&invertedCpp_gpu,controlPointNumber*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(invertedCpp_gpu,*cpp_gpu,controlPointNumber*sizeof(float4),cudaMemcpyDeviceToDevice))
    reg_getDisplacementFromDeformation_gpu(cpp_h, &invertedCpp_gpu);
    reg_multiplyValue_gpu(controlPointNumber,&invertedCpp_gpu,-1.f);
    reg_getDeformationFromDisplacement_gpu(cpp_h, &invertedCpp_gpu);

    reg_getDeformationFieldFromVelocityGrid_gpu(cpp_h,
                                                def_h,
                                                &invertedCpp_gpu,
                                                def_gpu,
                                                interDef_gpu,
                                                mask_gpu,
                                                activeVoxel,
                                                approxComp);
    NR_CUDA_SAFE_CALL(cudaFree(invertedCpp_gpu))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_compose_gpu(nifti_image *def,
                              float4 **def_gpu,
                              float4 **defOut_gpu,
                              int **mask_gpu,
                              int activeVoxel)
{
    const int voxelNumber=def->nx*def->ny*def->nz;
    if(voxelNumber != activeVoxel){
        fprintf(stderr, "[NiftyReg] reg_defField_compose_gpu\n");
        fprintf(stderr, "[NiftyReg] ERROR no mask can be used\n");
        exit(1);
    }

    // Bind the qform or sform
    mat44 temp_mat=def->qto_ijk;
    if(def->sform_code>0) temp_mat=def->sto_ijk;
    float4 temp=make_float4(temp_mat.m[0][0],temp_mat.m[0][1],temp_mat.m[0][2],temp_mat.m[0][3]);
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

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDisplacementTexture,*def_gpu,activeVoxel*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0,maskTexture,*mask_gpu,activeVoxel*sizeof(int)))

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceImageDim,sizeof(int)))

    const unsigned int Grid_reg_defField_compose =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_defField_compose)));
    dim3 G1(Grid_reg_defField_compose,Grid_reg_defField_compose,1);
    dim3 B1(Block_reg_defField_compose,1,1);
    reg_defField_compose_kernel<<< G1, B1>>>(*defOut_gpu);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDisplacementTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_defField_getJacobianMatrix_gpu(nifti_image *deformationField,
                                        float4 **deformationField_gpu,
                                        float **jacobianMatrices_gpu)
{
    const int3 referenceDim=make_int3(deformationField->nx,deformationField->ny,deformationField->nz);
    const float3 referenceSpacing=make_float3(deformationField->dx,deformationField->dy,deformationField->dz);
    const int voxelNumber = referenceDim.x*referenceDim.y*referenceDim.z;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceSpacing,&referenceSpacing,sizeof(float3)))

    mat33 reorient, desorient;
    reg_getReorientationMatrix(deformationField, &desorient, &reorient);
    float3 temp=make_float3(reorient.m[0][0],reorient.m[0][1],reorient.m[0][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix0,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[1][0],reorient.m[1][1],reorient.m[1][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix1,&temp,sizeof(float3)))
    temp=make_float3(reorient.m[2][0],reorient.m[2][1],reorient.m[2][2]);
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_AffineMatrix2,&temp,sizeof(float3)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0,voxelDisplacementTexture,*deformationField_gpu,voxelNumber*sizeof(float4)))

    const unsigned int Grid_reg_defField_getJacobianMatrix =
        (unsigned int)ceilf(sqrtf((float)voxelNumber/(float)(Block_reg_defField_getJacobianMatrix)));
    dim3 G1(Grid_reg_defField_getJacobianMatrix,Grid_reg_defField_getJacobianMatrix,1);
    dim3 B1(Block_reg_defField_getJacobianMatrix);
    reg_defField_getJacobianMatrix_kernel<<<G1,B1>>>(*jacobianMatrices_gpu);
    NR_CUDA_CHECK_KERNEL(G1,B1)

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(voxelDisplacementTexture))
}
/* *************************************************************** */
/* *************************************************************** */
#endif
