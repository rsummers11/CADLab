/**
 * @file _reg_nmi_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_NMI_GPU_CU
#define _REG_NMI_GPU_CU

#include "_reg_nmi.h"
#include "_reg_nmi_gpu.h"
#include "_reg_nmi_kernels.cu"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_nmi_gpu::reg_nmi_gpu():
	reg_nmi::reg_nmi()
{
	this->forwardJointHistogramLog_device=NULL;
//	this->backwardJointHistogramLog_device=NULL;

#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_nmi_gpu::~reg_nmi_gpu()
{
	this->ClearHistogram();
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_nmi_gpu::ClearHistogram()
{
	if(this->forwardJointHistogramLog_device!=NULL){
		cudaFree(this->forwardJointHistogramLog_device);
	}
	this->forwardJointHistogramLog_device=NULL;
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu::ClearHistogram() called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_nmi_gpu::InitialiseMeasure(nifti_image *refImgPtr,
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
	this->ClearHistogram();
    reg_nmi::InitialiseMeasure(refImgPtr,
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
	// Check if the input images have multiple timepoints
	if(this->referenceTimePoint>1 ||
       this->floatingImagePointer->nt>1){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		fprintf(stderr,"[NiftyReg ERROR] This class can only be \n");
		reg_exit(1);
    }
    // Check that the input image are of type float
    if(this->referenceImagePointer->datatype!=NIFTI_TYPE_FLOAT32 ||
       this->warpedFloatingImagePointer->datatype!=NIFTI_TYPE_FLOAT32){
        fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
        fprintf(stderr,"[NiftyReg ERROR] This class can only be \n");
        reg_exit(1);
    }
	// Bind the required pointers
	this->referenceDevicePointer = *refDevicePtr;
	this->floatingDevicePointer = *floDevicePtr;
    this->referenceMaskDevicePointer = *refMskDevicePtr;
	this->activeVoxeNumber = activeVoxNum;
    this->warpedFloatingDevicePointer = *warFloDevicePtr;
    this->warpedFloatingGradientDevicePointer = *warFloGradDevicePtr;
    this->forwardVoxelBasedGradientDevicePointer = *forVoxBasedGraDevicePtr;
	// The reference and floating images have to be updated on the device
	if(cudaCommon_transferNiftiToArrayOnDevice<float>
			(&this->referenceDevicePointer, this->referenceImagePointer)){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		printf("[NiftyReg ERROR] Error when transfering the reference image.\n");
		reg_exit(1);
	}
	if(cudaCommon_transferNiftiToArrayOnDevice<float>
			(&this->floatingDevicePointer, this->floatingImagePointer)){
		fprintf(stderr,"[NiftyReg ERROR] reg_nmi_gpu::InitialiseMeasure\n");
		printf("[NiftyReg ERROR] Error when transfering the floating image.\n");
		reg_exit(1);
	}
	// Allocate the required joint histogram on the GPU
	cudaMalloc(&this->forwardJointHistogramLog_device,
			   this->totalBinNumber[0]*sizeof(float));

#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu::InitialiseMeasure called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_nmi_gpu::GetSimilarityMeasureValue()
{
	// The NMI computation is performed into the host for now
	// The relevant images have to be transfered from the device to the host
	cudaMemcpy(this->warpedFloatingImagePointer->data,
			   this->warpedFloatingDevicePointer,
			   this->warpedFloatingImagePointer->nvox *
			   this->warpedFloatingImagePointer->nbyper,
			   cudaMemcpyDeviceToHost
               );

    reg_getNMIValue<float>
            (this->referenceImagePointer,
			 this->warpedFloatingImagePointer,
             this->activeTimePoint,
             this->referenceBinNumber,
             this->floatingBinNumber,
             this->totalBinNumber,
             this->forwardJointHistogramLog,
             this->forwardJointHistogramPro,
             this->forwardEntropyValues,
             this->referenceMaskPointer
             );

    double nmi_value=0.;
    nmi_value += (this->forwardEntropyValues[0][0] + this->forwardEntropyValues[0][1] ) /
            this->forwardEntropyValues[0][2];

#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu::GetSimilarityMeasureValue called\n");
#endif
	return nmi_value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// Called when we only have one target and one source image
void reg_getVoxelBasedNMIGradient_gpu(nifti_image *referenceImage,
									  cudaArray **referenceImageArray_d,
									  float **warpedImageArray_d,
									  float4 **warpedGradientArray_d,
									  float **logJointHistogram_d,
									  float4 **voxelNMIGradientArray_d,
									  int **mask_d,
									  int activeVoxelNumber,
									  double *entropies,
									  int refBinning,
									  int floBinning)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	const int voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
	const int3 imageSize=make_int3(referenceImage->nx,referenceImage->ny,referenceImage->nz);
    const int binNumber = refBinning*floBinning+refBinning+floBinning;
	const float normalisedJE=(float)(entropies[2]*entropies[3]);
    const float NMI = (float)((entropies[0]+entropies[1])/entropies[2]);

    // Bind Symbols
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&voxelNumber,sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageSize,&imageSize,sizeof(int3)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstTargetBin,&refBinning,sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_firstResultBin,&floBinning,sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NormalisedJE,&normalisedJE,sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NMI,&NMI,sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)));

    // Texture bindingcurrentFloating
    //Bind target image array to a 3D texture
	firstreferenceImageTexture.normalized = true;
	firstreferenceImageTexture.filterMode = cudaFilterModeLinear;
	firstreferenceImageTexture.addressMode[0] = cudaAddressModeWrap;
	firstreferenceImageTexture.addressMode[1] = cudaAddressModeWrap;
	firstreferenceImageTexture.addressMode[2] = cudaAddressModeWrap;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	NR_CUDA_SAFE_CALL(cudaBindTextureToArray(firstreferenceImageTexture, *referenceImageArray_d, channelDesc))
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, firstwarpedImageTexture, *warpedImageArray_d, voxelNumber*sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaBindTexture(0, firstwarpedImageGradientTexture, *warpedGradientArray_d, voxelNumber*sizeof(float4)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, histogramTexture, *logJointHistogram_d, binNumber*sizeof(float)));
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)));
    NR_CUDA_SAFE_CALL(cudaMemset(*voxelNMIGradientArray_d, 0, voxelNumber*sizeof(float4)));

	if(referenceImage->nz>1){
		const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW3D =
            (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getVoxelBasedNMIGradientUsingPW3D));
        dim3 B1(NR_BLOCK->Block_reg_getVoxelBasedNMIGradientUsingPW3D,1,1);
		dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW3D,Grid_reg_getVoxelBasedNMIGradientUsingPW3D,1);
		reg_getVoxelBasedNMIGradientUsingPW3D_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	else{
		const unsigned int Grid_reg_getVoxelBasedNMIGradientUsingPW2D =
            (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getVoxelBasedNMIGradientUsingPW2D));
        dim3 B1(NR_BLOCK->Block_reg_getVoxelBasedNMIGradientUsingPW2D,1,1);
		dim3 G1(Grid_reg_getVoxelBasedNMIGradientUsingPW2D,Grid_reg_getVoxelBasedNMIGradientUsingPW2D,1);
		reg_getVoxelBasedNMIGradientUsingPW2D_kernel <<< G1, B1 >>> (*voxelNMIGradientArray_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(firstreferenceImageTexture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(firstwarpedImageTexture));
	NR_CUDA_SAFE_CALL(cudaUnbindTexture(firstwarpedImageGradientTexture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(histogramTexture));
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture));
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradient()
{
    // The latest joint histogram is transfered onto the GPU
    float *temp=(float *)malloc(this->totalBinNumber[0]*sizeof(float));
    for(unsigned short i=0;i<this->totalBinNumber[0]; ++i)
		temp[i]=static_cast<float>(this->forwardJointHistogramLog[0][i]);
    cudaMemcpy(this->forwardJointHistogramLog_device,
               temp,
               this->totalBinNumber[0]*sizeof(float),
               cudaMemcpyHostToDevice);
    free(temp);

    // THe gradient of the NMI is computed on the GPU
    reg_getVoxelBasedNMIGradient_gpu(this->referenceImagePointer,
									 &this->referenceDevicePointer,
									 &this->warpedFloatingDevicePointer,
									 &this->warpedFloatingGradientDevicePointer,
									 &this->forwardJointHistogramLog_device,
									 &this->forwardVoxelBasedGradientDevicePointer,
									 &this->referenceMaskDevicePointer,
                                     this->activeVoxeNumber,
									 this->forwardEntropyValues[0],
									 this->referenceBinNumber[0],
									 this->floatingBinNumber[0]);
#ifndef NDEBUG
		printf("[NiftyReg DEBUG] reg_nmi_gpu::GetVoxelBasedSimilarityMeasureGradient called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
