/*
 *  _reg_resampling_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_GPU_CU
#define _REG_RESAMPLING_GPU_CU

#include "_reg_resampling_gpu.h"
#include "_reg_resampling_kernels.cu"

/* *************************************************************** */
/* *************************************************************** */
void reg_resampleImage_gpu(nifti_image *floatingImage,
                           float **warpedImageArray_d,
                           cudaArray **floatingImageArray_d,
                           float4 **deformationFieldImageArray_d,
                           int **mask_d,
                           int activeVoxelNumber,
                           float paddingValue)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_FloatingDim,&floatingDim,sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddingValue,&paddingValue,sizeof(float)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber,&activeVoxelNumber,sizeof(int)))

    //Bind floating image array to a 3D texture
    floatingTexture.normalized = false;
    floatingTexture.filterMode = cudaFilterModeLinear;
    floatingTexture.addressMode[0] = cudaAddressModeWrap;
    floatingTexture.addressMode[1] = cudaAddressModeWrap;
    floatingTexture.addressMode[2] = cudaAddressModeWrap;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    NR_CUDA_SAFE_CALL(cudaBindTextureToArray(floatingTexture, *floatingImageArray_d, channelDesc))

    //Bind deformationField to texture
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, deformationFieldTexture, *deformationFieldImageArray_d, activeVoxelNumber*sizeof(float4)))

    //Bind deformationField to texture
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, maskTexture, *mask_d, activeVoxelNumber*sizeof(int)))

    // Bind the real to voxel matrix to texture
    mat44 *floatingMatrix;
    if(floatingImage->sform_code>0)
        floatingMatrix=&(floatingImage->sto_ijk);
    else floatingMatrix=&(floatingImage->qto_ijk);
    float4 *floatingRealToVoxel_h;NR_CUDA_SAFE_CALL(cudaMallocHost(&floatingRealToVoxel_h, 3*sizeof(float4)))
    float4 *floatingRealToVoxel_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&floatingRealToVoxel_d, 3*sizeof(float4)))
    for(int i=0; i<3; i++){
        floatingRealToVoxel_h[i].x=floatingMatrix->m[i][0];
        floatingRealToVoxel_h[i].y=floatingMatrix->m[i][1];
        floatingRealToVoxel_h[i].z=floatingMatrix->m[i][2];
        floatingRealToVoxel_h[i].w=floatingMatrix->m[i][3];
    }
    NR_CUDA_SAFE_CALL(cudaMemcpy(floatingRealToVoxel_d, floatingRealToVoxel_h, 3*sizeof(float4), cudaMemcpyHostToDevice))
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)floatingRealToVoxel_h))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, floatingMatrixTexture, floatingRealToVoxel_d, 3*sizeof(float4)))

    if(floatingImage->nz>1){
        const unsigned int Grid_reg_resamplefloatingImage3D =
                (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_resampleImage3D));
        dim3 B1(NR_BLOCK->Block_reg_resampleImage3D,1,1);
        dim3 G1(Grid_reg_resamplefloatingImage3D,Grid_reg_resamplefloatingImage3D,1);
        reg_resampleImage3D_kernel <<< G1, B1 >>> (*warpedImageArray_d);
		cudaThreadSynchronize();
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	else{
        const unsigned int Grid_reg_resamplefloatingImage2D =
                (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_resampleImage2D));
        dim3 B1(NR_BLOCK->Block_reg_resampleImage2D,1,1);
        dim3 G1(Grid_reg_resamplefloatingImage2D,Grid_reg_resamplefloatingImage2D,1);
        reg_resampleImage2D_kernel <<< G1, B1 >>> (*warpedImageArray_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}

    NR_CUDA_SAFE_CALL(cudaUnbindTexture(floatingTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(deformationFieldTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(maskTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(floatingMatrixTexture))

    NR_CUDA_SAFE_CALL(cudaFree(floatingRealToVoxel_d))
}
/* *************************************************************** */
/* *************************************************************** */
void reg_getImageGradient_gpu(nifti_image *floatingImage,
                              cudaArray **floatingImageArray_d,
                              float4 **deformationFieldImageArray_d,
                              float4 **warpedGradientArray_d,
                              int activeVoxelNumber,
                              float paddingValue)
{
    // Get the BlockSize - The values have been set in _reg_common_gpu.h - cudaCommon_setCUDACard
    NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(0);

    int3 floatingDim = make_int3(floatingImage->nx, floatingImage->ny, floatingImage->nz);

    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_FloatingDim, &floatingDim, sizeof(int3)))
    NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_ActiveVoxelNumber, &activeVoxelNumber, sizeof(int)))
	NR_CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddingValue, &paddingValue, sizeof(float)))

    //Bind floating image array to a 3D texture
    floatingTexture.normalized = true;
    floatingTexture.filterMode = cudaFilterModeLinear;
    floatingTexture.addressMode[0] = cudaAddressModeWrap;
    floatingTexture.addressMode[1] = cudaAddressModeWrap;
    floatingTexture.addressMode[2] = cudaAddressModeWrap;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    NR_CUDA_SAFE_CALL(cudaBindTextureToArray(floatingTexture, *floatingImageArray_d, channelDesc))

    //Bind deformationField to texture
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, deformationFieldTexture, *deformationFieldImageArray_d, activeVoxelNumber*sizeof(float4)))

    // Bind the real to voxel matrix to texture
    mat44 *floatingMatrix;
    if(floatingImage->sform_code>0)
        floatingMatrix=&(floatingImage->sto_ijk);
    else floatingMatrix=&(floatingImage->qto_ijk);
    float4 *floatingRealToVoxel_h;NR_CUDA_SAFE_CALL(cudaMallocHost(&floatingRealToVoxel_h, 3*sizeof(float4)))
    float4 *floatingRealToVoxel_d;
    NR_CUDA_SAFE_CALL(cudaMalloc(&floatingRealToVoxel_d, 3*sizeof(float4)))
    for(int i=0; i<3; i++){
        floatingRealToVoxel_h[i].x=floatingMatrix->m[i][0];
        floatingRealToVoxel_h[i].y=floatingMatrix->m[i][1];
        floatingRealToVoxel_h[i].z=floatingMatrix->m[i][2];
        floatingRealToVoxel_h[i].w=floatingMatrix->m[i][3];
    }
    NR_CUDA_SAFE_CALL(cudaMemcpy(floatingRealToVoxel_d, floatingRealToVoxel_h, 3*sizeof(float4), cudaMemcpyHostToDevice))
    NR_CUDA_SAFE_CALL(cudaFreeHost((void *)floatingRealToVoxel_h))
    NR_CUDA_SAFE_CALL(cudaBindTexture(0, floatingMatrixTexture, floatingRealToVoxel_d, 3*sizeof(float4)))
    if(floatingImage->nz>1){
        const unsigned int Grid_reg_getImageGradient3D = (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getImageGradient3D));
        dim3 B1(NR_BLOCK->Block_reg_getImageGradient3D,1,1);
		dim3 G1(Grid_reg_getImageGradient3D,Grid_reg_getImageGradient3D,1);
        reg_getImageGradient3D_kernel <<< G1, B1 >>> (*warpedGradientArray_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
	else{
        const unsigned int Grid_reg_getImageGradient2D = (unsigned int)ceil(sqrtf((float)activeVoxelNumber/(float)NR_BLOCK->Block_reg_getImageGradient2D));
        dim3 B1(NR_BLOCK->Block_reg_getImageGradient2D,1,1);
		dim3 G1(Grid_reg_getImageGradient2D,Grid_reg_getImageGradient2D,1);
        reg_getImageGradient2D_kernel <<< G1, B1 >>> (*warpedGradientArray_d);
		NR_CUDA_CHECK_KERNEL(G1,B1)
	}
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(floatingTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(deformationFieldTexture))
    NR_CUDA_SAFE_CALL(cudaUnbindTexture(floatingMatrixTexture))

    cudaFree(floatingRealToVoxel_d);
}
/* *************************************************************** */
/* *************************************************************** */

#endif
