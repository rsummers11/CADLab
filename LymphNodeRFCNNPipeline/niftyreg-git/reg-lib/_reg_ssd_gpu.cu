/*
 * @file _reg_ssd_gpu.cu
 * @author Marc Modat
 * @date 14/11/2012
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_GPU_CU
#define _REG_SSD_GPU_CU

#include "_reg_ssd_gpu.h"
#include "_reg_ssd_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */
reg_ssd_gpu::reg_ssd_gpu()
	: reg_ssd::reg_ssd()
{
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_ssd_gpu constructor called\n");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_ssd_gpu::InitialiseMeasure(nifti_image *refImgPtr,
									nifti_image *floImgPtr,
									int *maskRefPtr,
									int activeVoxNum,
									nifti_image *warFloImgPtr,
									nifti_image *warFloGraPtr,
									nifti_image *forVoxBasedGraPtr,
									cudaArray **refDevicePtr,
									cudaArray **floDevicePtr,
									int **refMskDevicePtr,
									float **warFloDevicePtr,
									float4 **warFloGradDevicePtr,
									float4 **forVoxBasedGraDevicePtr)
{
	reg_ssd::InitialiseMeasure(refImgPtr,
							   floImgPtr,
							   maskRefPtr,
							   warFloImgPtr,
							   warFloGraPtr,
							   forVoxBasedGraPtr);
	// Check if a symmetric measure is required
	if(this->isSymmetric){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		fprintf(stderr,"[NiftyReg ERROR] Symmetric scheme is not yet supported on the GPU\n");
		reg_exit(1);
	}
	// Check that the input image are of type float
	if(this->referenceImagePointer->datatype!=NIFTI_TYPE_FLOAT32 ||
	   this->warpedFloatingImagePointer->datatype!=NIFTI_TYPE_FLOAT32){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		fprintf(stderr,"[NiftyReg ERROR] The input images are expected to be float\n");
		reg_exit(1);
	}
	// Check that the input images have only one time point
	if(this->referenceImagePointer->nt>1 || this->floatingImagePointer->nt>1){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		fprintf(stderr,"[NiftyReg ERROR] Both input images should have only one time point\n");
		reg_exit(1);
	}
	// Bind the required pointers
	this->referenceDevicePointer = *refDevicePtr;
	this->floatingDevicePointer = *floDevicePtr;
	this->referenceMaskDevicePointer = *refMskDevicePtr;
	this->activeVoxeNumber=activeVoxNum;
	this->warpedFloatingDevicePointer = *warFloDevicePtr;
	this->warpedFloatingGradientDevicePointer = *warFloGradDevicePtr;
	this->forwardVoxelBasedGradientDevicePointer = *forVoxBasedGraDevicePtr;
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_ssd_gpu::InitialiseMeasure()\n");
#endif
}
/* *************************************************************** */
float reg_getSSDValue_gpu(nifti_image *referenceImage,
						  cudaArray **reference_d,
						  float **warped_d,
						  int **mask_d,
						  int activeVoxelNumber
						  )
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	// Copy the constant memory variables
	int3 referenceDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
	int voxelNumber = referenceImage->nx * referenceImage->ny * referenceImage->nz;
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))
	// Bind the required textures
	referenceTexture.normalized = true;
	referenceTexture.filterMode = cudaFilterModeLinear;
	referenceTexture.addressMode[0] = cudaAddressModeWrap;
	referenceTexture.addressMode[1] = cudaAddressModeWrap;
	referenceTexture.addressMode[2] = cudaAddressModeWrap;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	NR_CUDA_SAFE_CALL(cudaBindTextureToArray(referenceTexture, *reference_d, channelDesc))
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, warpedTexture, *warped_d, voxelNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))
	// Create an array on the device to store the absolute difference values
	float *absoluteValues_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&absoluteValues_d, activeVoxelNumber*sizeof(float)))
	// Compute the absolute values
	const unsigned int Grid_reg_getSquaredDifference =
            (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getSquaredDifference));
    dim3 B1(NR_BLOCK->Block_reg_getSquaredDifference,1,1);
	dim3 G1(Grid_reg_getSquaredDifference,Grid_reg_getSquaredDifference,1);
	if(referenceDim.z>1)
		reg_getSquaredDifference3D_kernel <<< G1, B1 >>> (absoluteValues_d);
	else reg_getSquaredDifference2D_kernel <<< G1, B1 >>> (absoluteValues_d);
	NR_CUDA_CHECK_KERNEL(G1,B1)
	// Unbind the textures
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(referenceTexture))
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(warpedTexture))
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
	// Perform a reduction on the absolute values
    float ssd = (float)((double)reg_sumReduction_gpu(absoluteValues_d,activeVoxelNumber) / (double)activeVoxelNumber);
	// Free the absolute value array
	NR_CUDA_SAFE_CALL(cudaFree(absoluteValues_d))

	return ssd;
}
/* *************************************************************** */
/* *************************************************************** */
double reg_ssd_gpu::GetSimilarityMeasureValue()
{
	double SSDValue = reg_getSSDValue_gpu(this->referenceImagePointer,
										  &this->referenceDevicePointer,
										  &this->warpedFloatingDevicePointer,
										  &this->referenceMaskDevicePointer,
										  this->activeVoxeNumber
										  );
    return -SSDValue;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getVoxelBasedSSDGradient_gpu(nifti_image *referenceImage,
									  cudaArray **reference_d,
									  float **warped_d,
									  float4 **spaGradient_d,
									  float4 **ssdGradient_d,
									  float maxSD,
									  int **mask_d,
									  int activeVoxelNumber
									  )
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	// Copy the constant memory variables
	int3 referenceDim = make_int3(referenceImage->nx, referenceImage->ny, referenceImage->nz);
	int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ReferenceImageDim,&referenceDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NormalisationNumber,&maxSD,sizeof(float)))
	// Bind the required textures
	referenceTexture.normalized = true;
	referenceTexture.filterMode = cudaFilterModeLinear;
	referenceTexture.addressMode[0] = cudaAddressModeWrap;
	referenceTexture.addressMode[1] = cudaAddressModeWrap;
	referenceTexture.addressMode[2] = cudaAddressModeWrap;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	NR_CUDA_SAFE_CALL(cudaBindTextureToArray(referenceTexture, *reference_d, channelDesc))
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, warpedTexture, *warped_d, voxelNumber*sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, spaGradientTexture, *spaGradient_d, voxelNumber*sizeof(float4)))
	// Set the gradient image to zero
	NR_CUDA_SAFE_CALL(cudaMemset(*ssdGradient_d,0,voxelNumber*sizeof(float4)))
	const unsigned int Grid_reg_getSSDGradient =
            (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getSSDGradient));
    dim3 B1(NR_BLOCK->Block_reg_getSSDGradient,1,1);
	dim3 G1(Grid_reg_getSSDGradient,Grid_reg_getSSDGradient,1);
	if(referenceDim.z>1)
		reg_getSSDGradient3D_kernel <<< G1, B1 >>> (*ssdGradient_d);
	else reg_getSSDGradient2D_kernel <<< G1, B1 >>> (*ssdGradient_d);
	NR_CUDA_CHECK_KERNEL(G1,B1)
	// Unbind the textures
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(referenceTexture))
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(warpedTexture))
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(spaGradientTexture))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_ssd_gpu::GetVoxelBasedSimilarityMeasureGradient()
{
	reg_getVoxelBasedSSDGradient_gpu(this->referenceImagePointer,
									 &this->referenceDevicePointer,
									 &this->warpedFloatingDevicePointer,
									 &this->warpedFloatingGradientDevicePointer,
									 &this->forwardVoxelBasedGradientDevicePointer,
                                     1.0f,
									 &this->referenceMaskDevicePointer,
									 this->activeVoxeNumber
									 );
	return;
}
/* *************************************************************** */
/* *************************************************************** */
#endif
