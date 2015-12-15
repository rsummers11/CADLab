/**
 * @file _reg_comon_gpu.cu
 * @author Marc Modat
 * @date 25/03/2009
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_COMMON_GPU_CU
#define _REG_COMMON_GPU_CU

#include "_reg_common_gpu.h"

/* ******************************** */
/* ******************************** */
int cudaCommon_setCUDACard(CUcontext *ctx,
                           bool verbose)
{
    // The CUDA card is setup
    cuInit(0);
    struct cudaDeviceProp deviceProp;
    int device_count=0;
    cudaGetDeviceCount( &device_count );
    if(verbose)
        printf("[NiftyReg CUDA] %i card(s) detected\n", device_count);
    // following code is from cutGetMaxGflopsDeviceId()
    int max_gflops_device = 0;
    int max_gflops = 0;
    int current_device = 0;
    while(current_device<device_count ){
        cudaGetDeviceProperties( &deviceProp, current_device );
        int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;
        if( gflops > max_gflops ){
            max_gflops = gflops;
            max_gflops_device = current_device;
        }
        ++current_device;
    }
    NR_CUDA_SAFE_CALL(cudaSetDevice(max_gflops_device));
    NR_CUDA_SAFE_CALL(cuCtxCreate(ctx, CU_CTX_SCHED_SPIN, max_gflops_device))
    NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, max_gflops_device));

    if(deviceProp.major<1){
        fprintf(stderr, "[NiftyReg ERROR CUDA] The specified graphical card does not exist.\n");
        return EXIT_FAILURE;
    }
    else{
        size_t free=0;
        size_t total=0;
        cuMemGetInfo(&free, &total);
        if(deviceProp.totalGlobalMem != total){
            fprintf(stderr,"[NiftyReg CUDA ERROR] The CUDA card %s does not seem to be available\n",
                   deviceProp.name);
            fprintf(stderr,"[NiftyReg CUDA ERROR] Expected total memory: %lu Mb - Recovered total memory: %lu Mb\n",
                    deviceProp.totalGlobalMem/(1024*1024), total/(1024*1024));
            return EXIT_FAILURE;
        }
        if(verbose){
            printf("[NiftyReg CUDA] The following device is used: %s\n",
                   deviceProp.name);
            printf("[NiftyReg CUDA] It has %lu Mb free out of %lu Mb\n",
                   (unsigned long int)(free/(1024*1024)),
                   (unsigned long int)(total/(1024*1024)));
            printf("[NiftyReg CUDA] Card compute capability: %i.%i\n",
                   deviceProp.major,
                   deviceProp.minor);
            printf("[NiftyReg CUDA] Shared memory size in bytes: %lu\n",
                   deviceProp.sharedMemPerBlock);
            printf("[NiftyReg CUDA] CUDA version %i\n",
                   CUDART_VERSION);
            printf("[NiftyReg CUDA] Card clock rate: %i MHz\n",
                   deviceProp.clockRate/1000);
            printf("[NiftyReg CUDA] Card has %i multiprocessor(s)\n",
                   deviceProp.multiProcessorCount);
        }
        NiftyReg_CudaBlock100 *NR_BLOCK = NiftyReg_CudaBlock::getInstance(deviceProp.major);
    }
    return EXIT_SUCCESS;
}
/* ******************************** */
void cudaCommon_unsetCUDACard(CUcontext *ctx)
{
//    cuCtxDetach(*ctx);
    cuCtxDestroy(*ctx);
}
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DTYPE);
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
			fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        const int voxelNumber = img->nx*img->ny*img->nz;
        for(int i=0; i<voxelNumber; i++)
            array_h[i].x= *niftiImgValues++;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].y= *niftiImgValues++;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].z= *niftiImgValues++;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].w= *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **, nifti_image *);
/* ******************************** */

template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DTYPE);
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h=&array_h[img->dim[1] * img->dim[2] * img->dim[3]];
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, memSize, cudaMemcpyHostToDevice));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
			fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        const int voxelNumber = img->nx*img->ny*img->nz;
        for(int i=0; i<voxelNumber; i++)
            array_h[i].x= *niftiImgValues++;
        for(int i=0; i<voxelNumber; i++)
            array2_h[i].x= *niftiImgValues++;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].y= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].y= *niftiImgValues++;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].z= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].z= *niftiImgValues++;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                array_h[i].w= *niftiImgValues++;
            for(int i=0; i<voxelNumber; i++)
                array2_h[i].w= *niftiImgValues++;
        }
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        NR_CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
        free(array_h);
        free(array2_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, array2_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **,float **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **,float4 **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) ){
			fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array_h[i].x= *niftiImgValues++;

        if(img->dim[5]>=2){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].y= *niftiImgValues++;
        }

        if(img->dim[5]>=3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].z= *niftiImgValues++;
        }

        if(img->dim[5]==3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].w= *niftiImgValues++;
        }
        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        copyParams.kind = cudaMemcpyHostToDevice;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams))
        free(array_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        NIFTI_TYPE *array_h = static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h = &array_h[img->dim[1]*img->dim[2]*img->dim[3]];

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray2_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
    if( sizeof(DTYPE)==sizeof(float4) ){
        if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) ){
			fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice\n");
            fprintf(stderr, "ERROR:\tThe specified image is not a single precision deformation field image\n");
            return 1;
        }
        float *niftiImgValues = static_cast<float *>(img->data);
        float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
        float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array_h[i].x= *niftiImgValues++;
        for(int i=0; i<img->nx*img->ny*img->nz; i++)
            array2_h[i].x= *niftiImgValues++;

        if(img->dim[5]>=2){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].y= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].y= *niftiImgValues++;
        }

        if(img->dim[5]>=3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].z= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].z= *niftiImgValues++;
        }

        if(img->dim[5]==3){
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array_h[i].w= *niftiImgValues++;
            for(int i=0; i<img->nx*img->ny*img->nz; i++)
                array2_h[i].w= *niftiImgValues++;
        }

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
        copyParams.kind = cudaMemcpyHostToDevice;
        // First timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array_h);
        // Second timepoint
        copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
                                                copyParams.extent.width*sizeof(DTYPE),
                                                copyParams.extent.width,
                                                copyParams.extent.height);
        copyParams.dstArray = *cuArray2_d;
        NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
        free(array2_h);
    }
    else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, cuArray2_d, img);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferNiftiToArrayOnDevice:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
    return 0;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, int *dim)
{
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    return 0;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, int *dim)
{
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
    NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray2_d, &texDesc, volumeSize));
    return 0;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **,cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, int *dim)
{
    const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    return 0;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(float4 **, int *); // for deformation field
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, int vox)
{
    const unsigned int memSize = vox * sizeof(DTYPE);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    return 0;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, int);
template int cudaCommon_allocateArrayToDevice<float4>(float4 **, int); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, DTYPE **array2_d, int *dim)
{
    const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
    NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
    NR_CUDA_SAFE_CALL(cudaMalloc(array2_d, memSize));
    return 0;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, float **, int *);
template int  cudaCommon_allocateArrayToDevice<float4>(float4 **, float4 **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, img->nvox*sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d)
{
    if(sizeof(DTYPE)==sizeof(float4)){
        // A nifti 5D volume is expected
        if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
            fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
            fprintf(stderr, "ERROR:\tThe nifti image is not a 5D volume.\n");
            return 1;
        }
        const int voxelNumber = img->nx*img->ny*img->nz;

        float4 *array_h;
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float *>(img->data);

        for(int i=0; i<voxelNumber; i++)
            *niftiImgValues++ = array_h[i].x;
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].y;
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].z;
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++)
                *niftiImgValues++ = array_h[i].w;
        }
        NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));

        return 0;
    }
    else{
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
    if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
        fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
        fprintf(stderr, "ERROR:\tThe host and device arrays are of different types.\n");
        return 1;
    }
    else{
        unsigned int voxelNumber=img->nx*img->ny*img->nz;
        NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
        NIFTI_TYPE *array2_h=&array_h[voxelNumber];
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (void *)*array2_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    return 0;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
    if(sizeof(DTYPE)==sizeof(float4)){
        // A nifti 5D volume is expected
        if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
            fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
            fprintf(stderr, "ERROR:\tThe nifti image is not a 5D volume.\n");
            return 1;
        }
        const int voxelNumber = img->nx*img->ny*img->nz;
        float4 *array_h=NULL;
        float4 *array2_h=NULL;
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMallocHost(&array2_h, voxelNumber*sizeof(float4)));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (const void *)*array2_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
        float *niftiImgValues = static_cast<float *>(img->data);
        for(int i=0; i<voxelNumber; i++){
            *niftiImgValues++ = array_h[i].x;
        }
        for(int i=0; i<voxelNumber; i++){
            *niftiImgValues++ = array2_h[i].x;
        }
        if(img->dim[5]>=2){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].y;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].y;
            }
        }
        if(img->dim[5]>=3){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].z;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].z;
            }
        }
        if(img->dim[5]>=4){
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array_h[i].w;
            }
            for(int i=0; i<voxelNumber; i++){
                *niftiImgValues++ = array2_h[i].w;
            }
        }
        NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));
        NR_CUDA_SAFE_CALL(cudaFreeHost(array2_h));

        return 0;
    }
    else{
        switch(img->datatype){
            case NIFTI_TYPE_FLOAT32:
                return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d, array2_d);
            default:
                fprintf(stderr, "ERROR:\tcudaCommon_transferFromDeviceToNifti:\n");
                fprintf(stderr, "ERROR:\tThe image data type is not supported\n");
                return 1;
        }
    }
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **, float **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
void cudaCommon_free(cudaArray **cuArray_d){
        NR_CUDA_SAFE_CALL(cudaFreeArray(*cuArray_d));
	return;
}
/* ******************************** */
/* ******************************** */
template <class DTYPE>
void cudaCommon_free(DTYPE **array_d){
    NR_CUDA_SAFE_CALL(cudaFree(*array_d));
	return;
}
template void cudaCommon_free<int>(int **);
template void cudaCommon_free<float>(float **);
template void cudaCommon_free<float4>(float4 **);
/* ******************************** */
/* ******************************** */
#endif
