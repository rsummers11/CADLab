/*
 *  _reg_tools_gpu.cu
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_GPU_CU
#define _REG_TOOLS_GPU_CU

#include "_reg_common_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_tools_kernels.cu"


/* *************************************************************** */
/* *************************************************************** */
void reg_voxelCentric2NodeCentric_gpu(nifti_image *targetImage,
                                      nifti_image *controlPointImage,
                                      float4 **voxelNMIGradientArray_d,
                                      float4 **nodeNMIGradientArray_d,
                                      float weight)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
    const int voxelNumber = targetImage->nx * targetImage->ny * targetImage->nz;
    const int3 targetImageDim = make_int3(targetImage->nx, targetImage->ny, targetImage->nz);
    const int3 gridSize = make_int3(controlPointImage->nx, controlPointImage->ny, controlPointImage->nz);
	float3 voxelNodeRatio_h = make_float3(
            controlPointImage->dx / targetImage->dx,
            controlPointImage->dy / targetImage->dy,
            controlPointImage->dz / targetImage->dz);
	// Ensure that Z=0 if 2D images
	if(gridSize.z==1) voxelNodeRatio_h.z=0;

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_TargetImageDim,&targetImageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ControlPointImageDim,&gridSize,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNodeRatio,&voxelNodeRatio_h,sizeof(float3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&weight,sizeof(float)))

    NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *voxelNMIGradientArray_d, voxelNumber*sizeof(float4)))

    const unsigned int Grid_reg_voxelCentric2NodeCentric = (unsigned int)ceil(sqrtf((float)nodeNumber/(float)NR_BLOCK->Block_reg_voxelCentric2NodeCentric));
    dim3 B1(NR_BLOCK->Block_reg_voxelCentric2NodeCentric,1,1);
	dim3 G1(Grid_reg_voxelCentric2NodeCentric,Grid_reg_voxelCentric2NodeCentric,1);
    reg_voxelCentric2NodeCentric_kernel <<< G1, B1 >>> (*nodeNMIGradientArray_d);
	NR_CUDA_CHECK_KERNEL(G1,B1)

	NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(	mat44 *sourceMatrix_xyz,
                            nifti_image *controlPointImage,
                            float4 **nodeNMIGradientArray_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int nodeNumber = controlPointImage->nx * controlPointImage->ny * controlPointImage->nz;
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_NodeNumber,&nodeNumber,sizeof(int)))

    float4 *matrix_h;NR_CUDA_SAFE_CALL(cudaMallocHost(&matrix_h, 3*sizeof(float4)))
    matrix_h[0] = make_float4(sourceMatrix_xyz->m[0][0], sourceMatrix_xyz->m[0][1], sourceMatrix_xyz->m[0][2], sourceMatrix_xyz->m[0][3]);
    matrix_h[1] = make_float4(sourceMatrix_xyz->m[1][0], sourceMatrix_xyz->m[1][1], sourceMatrix_xyz->m[1][2], sourceMatrix_xyz->m[1][3]);
    matrix_h[2] = make_float4(sourceMatrix_xyz->m[2][0], sourceMatrix_xyz->m[2][1], sourceMatrix_xyz->m[2][2], sourceMatrix_xyz->m[2][3]);
    float4 *matrix_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&matrix_d, 3*sizeof(float4)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(matrix_d, matrix_h, 3*sizeof(float4), cudaMemcpyHostToDevice))
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)matrix_h))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, matrixTexture, matrix_d, 3*sizeof(float4)))

    const unsigned int Grid_reg_convertNMIGradientFromVoxelToRealSpace =
        (unsigned int)ceil(sqrtf((float)nodeNumber/(float)NR_BLOCK->Block_reg_convertNMIGradientFromVoxelToRealSpace));
    dim3 G1(Grid_reg_convertNMIGradientFromVoxelToRealSpace,Grid_reg_convertNMIGradientFromVoxelToRealSpace,1);
    dim3 B1(NR_BLOCK->Block_reg_convertNMIGradientFromVoxelToRealSpace,1,1);

    _reg_convertNMIGradientFromVoxelToRealSpace_kernel <<< G1, B1 >>> (*nodeNMIGradientArray_d);
    NR_CUDA_CHECK_KERNEL(G1,B1)
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(matrixTexture))
    NR_CUDA_SAFE_CALL(cudaFree(matrix_d))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_gaussianSmoothing_gpu( nifti_image *image,
                                float4 **imageArray_d,
                                float sigma,
                                bool smoothXYZ[8])

{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

	const unsigned int voxelNumber = image->nx * image->ny * image->nz;
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageDim, &imageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &voxelNumber,sizeof(int3)))

    bool axisToSmooth[8];
    if(smoothXYZ==NULL){
        for(int i=0; i<8; i++) axisToSmooth[i]=true;
    }
    else{
        for(int i=0; i<8; i++) axisToSmooth[i]=smoothXYZ[i];
    }

	for(int n=1; n<4; n++){
		if(axisToSmooth[n]==true && image->dim[n]>1){
            float currentSigma;
            if(sigma>0) currentSigma=sigma/image->pixdim[n];
            else currentSigma=fabs(sigma); // voxel based if negative value
            int radius=(int)ceil(currentSigma*3.0f);
            if(radius>0){
                int kernelSize = 1+radius*2;
                float *kernel_h;
                NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel_h, kernelSize*sizeof(float)))
                float kernelSum=0;
                for(int i=-radius; i<=radius; i++){
					kernel_h[radius+i]=(float)(exp( -((float)i*(float)i)/(2.0*currentSigma*currentSigma)) /
											   (currentSigma*2.506628274631));
					// 2.506... = sqrt(2*pi)
                    kernelSum += kernel_h[radius+i];
                }
				for(int i=0; i<kernelSize; i++)
					kernel_h[i] /= kernelSum;

                float *kernel_d;
                NR_CUDA_SAFE_CALL(cudaMalloc(&kernel_d, kernelSize*sizeof(float)))
                NR_CUDA_SAFE_CALL(cudaMemcpy(kernel_d, kernel_h, kernelSize*sizeof(float), cudaMemcpyHostToDevice))
                NR_CUDA_SAFE_CALL(cudaFreeHost(kernel_h))

                float4 *smoothedImage;
                NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage,voxelNumber*sizeof(float4)))

                NR_CUDA_SAFE_CALL(cudaBindTexture(0, convolutionKernelTexture, kernel_d, kernelSize*sizeof(float)))
                NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *imageArray_d, voxelNumber*sizeof(float4)))

				unsigned int Grid_reg_ApplyConvolutionWindow;
                dim3 B,G;
                switch(n){
                    case 1:
                        Grid_reg_ApplyConvolutionWindow =
                            (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongX));
                        B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongX,1,1);
                        G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                        _reg_ApplyConvolutionWindowAlongX_kernel <<< G, B >>> (smoothedImage, kernelSize);
                        NR_CUDA_CHECK_KERNEL(G,B)
                        break;
                    case 2:
                        Grid_reg_ApplyConvolutionWindow =
                            (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongY));
                        B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongY,1,1);
                        G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                        _reg_ApplyConvolutionWindowAlongY_kernel <<< G, B >>> (smoothedImage, kernelSize);
                        NR_CUDA_CHECK_KERNEL(G,B)
                        break;
                    case 3:
                        Grid_reg_ApplyConvolutionWindow =
                            (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongZ));
                        B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongZ,1,1);
                        G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                        _reg_ApplyConvolutionWindowAlongZ_kernel <<< G, B >>> (smoothedImage, kernelSize);
                        NR_CUDA_CHECK_KERNEL(G,B)
                        break;
                }
                NR_CUDA_SAFE_CALL(cudaUnbindTexture(convolutionKernelTexture))
                NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
                NR_CUDA_SAFE_CALL(cudaFree(kernel_d))
                NR_CUDA_SAFE_CALL(cudaMemcpy(*imageArray_d, smoothedImage, voxelNumber*sizeof(float4), cudaMemcpyDeviceToDevice))
                NR_CUDA_SAFE_CALL(cudaFree(smoothedImage))
            }
		}
	}
}
/* *************************************************************** */
void reg_smoothImageForCubicSpline_gpu( nifti_image *image,
                                        float4 **imageArray_d,
										float *spacingVoxel)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    const int voxelNumber = image->nx * image->ny * image->nz;
    const int3 imageDim = make_int3(image->nx, image->ny, image->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ImageDim, &imageDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber, &voxelNumber,sizeof(int)))

	for(int n=0; n<3; n++){
		if(spacingVoxel[n]>0 && image->dim[n+1]>1){
			int radius = static_cast<int>(reg_ceil(2.0*spacingVoxel[n]));
			int kernelSize = 1+radius*2;

            float *kernel_h;
            NR_CUDA_SAFE_CALL(cudaMallocHost(&kernel_h, kernelSize*sizeof(float)))

			float coeffSum=0;
			for(int it=-radius; it<=radius; it++){
				float coeff = (float)(fabs((float)(float)it/(float)spacingVoxel[0]));
				if(coeff<1.0) kernel_h[it+radius] = (float)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
				else if (coeff<2.0) kernel_h[it+radius] = (float)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
				else kernel_h[it+radius]=0;
				coeffSum += kernel_h[it+radius];
			}
			for(int it=0;it<kernelSize;it++) kernel_h[it] /= coeffSum;

            float *kernel_d;
            NR_CUDA_SAFE_CALL(cudaMalloc(&kernel_d, kernelSize*sizeof(float)))
            NR_CUDA_SAFE_CALL(cudaMemcpy(kernel_d, kernel_h, kernelSize*sizeof(float), cudaMemcpyHostToDevice))
            NR_CUDA_SAFE_CALL(cudaFreeHost(kernel_h))
            NR_CUDA_SAFE_CALL(cudaBindTexture(0, convolutionKernelTexture, kernel_d, kernelSize*sizeof(float)))

            float4 *smoothedImage_d;
            NR_CUDA_SAFE_CALL(cudaMalloc(&smoothedImage_d,voxelNumber*sizeof(float4)))

            NR_CUDA_SAFE_CALL(cudaBindTexture(0, gradientImageTexture, *imageArray_d, voxelNumber*sizeof(float4)))

            unsigned int Grid_reg_ApplyConvolutionWindow;
            dim3 B,G;
            switch(n){
                case 0:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongX));
                    B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongX,1,1);
                    G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                    _reg_ApplyConvolutionWindowAlongX_kernel <<< G, B >>> (smoothedImage_d, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G,B)
                    break;
                case 1:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongY));
                    B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongY,1,1);
                    G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                    _reg_ApplyConvolutionWindowAlongY_kernel <<< G, B >>> (smoothedImage_d, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G,B)
                    break;
                case 2:
                    Grid_reg_ApplyConvolutionWindow =
                        (unsigned int)ceil(sqrtf((float)voxelNumber/(float)NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongZ));
                    B=dim3(NR_BLOCK->Block_reg_ApplyConvolutionWindowAlongZ,1,1);
                    G=dim3(Grid_reg_ApplyConvolutionWindow,Grid_reg_ApplyConvolutionWindow,1);
                    _reg_ApplyConvolutionWindowAlongZ_kernel <<< G, B >>> (smoothedImage_d, kernelSize);
                    NR_CUDA_CHECK_KERNEL(G,B)
                    break;
            }
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(convolutionKernelTexture))
            NR_CUDA_SAFE_CALL(cudaUnbindTexture(gradientImageTexture))
            NR_CUDA_SAFE_CALL(cudaFree(kernel_d))
            NR_CUDA_SAFE_CALL(cudaMemcpy(*imageArray_d, smoothedImage_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToDevice))
            NR_CUDA_SAFE_CALL(cudaFree(smoothedImage_d))
        }
    }
}
/* *************************************************************** */
void reg_multiplyValue_gpu(int num, float4 **array_d, float value)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&num,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&value,sizeof(float)))

    const unsigned int Grid_reg_multiplyValues = (unsigned int)ceil(sqrtf((float)num/(float)NR_BLOCK->Block_reg_arithmetic));
    dim3 G=dim3(Grid_reg_multiplyValues,Grid_reg_multiplyValues,1);
    dim3 B=dim3(NR_BLOCK->Block_reg_arithmetic,1,1);
    reg_multiplyValue_kernel_float4<<<G,B>>>(*array_d);
    NR_CUDA_CHECK_KERNEL(G,B)
}
/* *************************************************************** */
void reg_addValue_gpu(int num, float4 **array_d, float value)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&num,sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Weight,&value,sizeof(float)))

    const unsigned int Grid_reg_addValues = (unsigned int)ceil(sqrtf((float)num/(float)NR_BLOCK->Block_reg_arithmetic));
    dim3 G=dim3(Grid_reg_addValues,Grid_reg_addValues,1);
    dim3 B=dim3(NR_BLOCK->Block_reg_arithmetic,1,1);
    reg_addValue_kernel_float4<<<G,B>>>(*array_d);
    NR_CUDA_CHECK_KERNEL(G,B)
}
/* *************************************************************** */
void reg_multiplyArrays_gpu(int num, float4 **array1_d, float4 **array2_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&num,sizeof(int)))

    const unsigned int Grid_reg_multiplyArrays = (unsigned int)ceil(sqrtf((float)num/(float)NR_BLOCK->Block_reg_arithmetic));
    dim3 G=dim3(Grid_reg_multiplyArrays,Grid_reg_multiplyArrays,1);
    dim3 B=dim3(NR_BLOCK->Block_reg_arithmetic,1,1);
    reg_multiplyArrays_kernel_float4<<<G,B>>>(*array1_d,*array2_d);
    NR_CUDA_CHECK_KERNEL(G,B)
}
/* *************************************************************** */
void reg_addArrays_gpu(int num, float4 **array1_d, float4 **array2_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&num,sizeof(int)))

    const unsigned int Grid_reg_addArrays = (unsigned int)ceil(sqrtf((float)num/(float)NR_BLOCK->Block_reg_arithmetic));
    dim3 G=dim3(Grid_reg_addArrays,Grid_reg_addArrays,1);
    dim3 B=dim3(NR_BLOCK->Block_reg_arithmetic,1,1);
    reg_addArrays_kernel_float4<<<G,B>>>(*array1_d,*array2_d);
    NR_CUDA_CHECK_KERNEL(G,B)
}
/* *************************************************************** */
void reg_fillMaskArray_gpu(int num, int **array1_d)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_VoxelNumber,&num,sizeof(int)))

    const unsigned int Grid_reg_fillMaskArray = (unsigned int)ceil(sqrtf((float)num/(float)NR_BLOCK->Block_reg_arithmetic));
    dim3 G=dim3(Grid_reg_fillMaskArray,Grid_reg_fillMaskArray,1);
    dim3 B=dim3(NR_BLOCK->Block_reg_arithmetic,1,1);
    reg_fillMaskArray_kernel<<<G,B>>>(*array1_d);
    NR_CUDA_CHECK_KERNEL(G,B)
}
/* *************************************************************** */
float reg_sumReduction_gpu(float *array_d,int size)
{
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr,dptr+size, 0.f, thrust::plus<float>());
}
/* *************************************************************** */
float reg_maxReduction_gpu(float *array_d,int size)
{
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr, dptr+size, 0.f, thrust::maximum<float>());
}
/* *************************************************************** */
float reg_minReduction_gpu(float *array_d,int size)
{
    thrust::device_ptr<float> dptr(array_d);
    return thrust::reduce(dptr, dptr+size, 0.f, thrust::minimum<float>());
}
/* *************************************************************** */
#endif

