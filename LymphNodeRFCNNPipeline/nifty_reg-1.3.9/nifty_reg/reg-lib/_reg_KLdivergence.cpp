/*
 *  _reg_KLdivergence.cpp
 *  
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_KLdivergence.h"

/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_getKLDivergence1(nifti_image *referenceImage,
                            nifti_image *warpedImage,
                            nifti_image *jacobianDetImg,
                            int *mask)
{
    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

#ifdef _WINDOWS
    int  voxel;
#else
    size_t  voxel;
#endif

    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);
    int *maskPtr=NULL;
    if(mask==NULL)
        maskPtr=(int *)calloc(voxelNumber,sizeof(int));
    else maskPtr = &mask[0];

    DTYPE *jacPtr=NULL;
    if(jacobianDetImg!=NULL)
        jacPtr=static_cast<DTYPE *>(jacobianDetImg->data);
    double measure=0., num=0.,tempRefValue,tempWarValue,tempValue;

    for(int time=0;time<referenceImage->nt;++time){

        DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
        DTYPE *currentWarPtr=&warPtr[time*voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,currentRefPtr, currentWarPtr, \
    maskPtr, jacobianDetImg, jacPtr) \
    private(voxel, tempRefValue, tempWarValue, tempValue) \
    reduction(+:measure) \
    reduction(+:num)
#endif
        for(voxel=0; voxel<voxelNumber; ++voxel){
            if(maskPtr[voxel]>-1){
                tempRefValue = currentRefPtr[voxel]+1e-16;
                tempWarValue = currentWarPtr[voxel]+1e-16;
                tempValue=tempRefValue*log(tempRefValue/tempWarValue);
                if(tempValue==tempValue &&
                   tempValue!=std::numeric_limits<double>::infinity()){
                    if(jacobianDetImg==NULL){
                        measure += tempValue;
                        num++;
                    }
                    else{
                        measure += tempValue * jacPtr[voxel];
                        num+=jacPtr[voxel];
                    }
                }
            }
        }
    }
    return measure/num;
}
/* *************************************************************** */
double reg_getKLDivergence(nifti_image *referenceImage,
                           nifti_image *warpedImage,
                           nifti_image *jacobianDetImg,
                           int *mask)
{
    // Check that both images are of the same type
    if(referenceImage->datatype!=warpedImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergence: both input images are expected to have the same type\n");
        exit(1);
    }
    // If the Jacobian determinant image if define, it checks it has the type of the referenceImage image
    if(jacobianDetImg!=NULL){
        if(referenceImage->datatype!=jacobianDetImg->datatype){
            fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergence: input images are expected to have the same type\n");
            exit(1);
        }
    }
    // Check that both input images have the same size
    for(int i=0;i<5;++i){
        if(referenceImage->dim[i] != warpedImage->dim[i]){
            fprintf(stderr,"[NiftyReg ERROR] reg_getSSD\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same dimension");
            exit(1);
        }
    }
    switch(referenceImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        return reg_getKLDivergence1<float>(referenceImage,warpedImage,jacobianDetImg,mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_getKLDivergence1<double>(referenceImage,warpedImage,jacobianDetImg,mask);
        break;
    default:
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergence: unsupported datatype\n");
        exit(1);
        break;
    }
    return 0.;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getKLDivergenceVoxelBasedGradient1(nifti_image *referenceImage,
                                            nifti_image *warpedImage,
                                            nifti_image *warpedImageGradient,
                                            nifti_image *KLdivGradient,
                                            nifti_image *jacobianDetImg,
                                            int *mask)
{
    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

#ifdef _WINDOWS
    int  voxel;
#else
    size_t  voxel;
#endif

    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);
    int *maskPtr=NULL;
    if(mask==NULL)
        maskPtr=(int *)calloc(voxelNumber,sizeof(int));
    else maskPtr = &mask[0];

    DTYPE *jacPtr=NULL;
    if(jacobianDetImg!=NULL)
        jacPtr=static_cast<DTYPE *>(jacobianDetImg->data);
    double tempValue, tempGradX, tempGradY, tempGradZ;

    // Create pointers to the spatial derivative of the warped image
    DTYPE *warGradPtr = static_cast<DTYPE *>(warpedImageGradient->data);

    // Create pointers to the voxel based gradient image - results
    DTYPE *kldGradPtrX = static_cast<DTYPE *>(KLdivGradient->data);
    DTYPE *kldGradPtrY = &kldGradPtrX[voxelNumber];
    DTYPE *kldGradPtrZ = NULL;

    if(referenceImage->nz>1)
        kldGradPtrZ = &kldGradPtrY[voxelNumber];

    // Set all the gradient values to zero
    for(voxel=0;voxel<KLdivGradient->nvox;++voxel)
        kldGradPtrX[voxel]=0;

    // Loop over the different time points
    for(int time=0;time<referenceImage->nt;++time){

        // Create some pointers to the current time point image to be accessed
        DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
        DTYPE *currentWarPtr=&warPtr[time*voxelNumber];
        // Create some pointers to the spatial gradient of the current warped volume
        DTYPE *currentGradPtrX=&warGradPtr[time*voxelNumber];
        DTYPE *currentGradPtrY=&currentGradPtrX[referenceImage->nt*voxelNumber];
        DTYPE *currentGradPtrZ=NULL;
        if(referenceImage->nz>1)
            currentGradPtrZ=&currentGradPtrY[referenceImage->nt*voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,currentRefPtr, currentWarPtr, \
    maskPtr, jacobianDetImg, jacPtr, referenceImage, \
    kldGradPtrX, kldGradPtrY, kldGradPtrZ, \
    currentGradPtrX, currentGradPtrY, currentGradPtrZ) \
    private(voxel, tempValue, tempGradX, tempGradY, tempGradZ)
#endif
        for(voxel=0; voxel<voxelNumber; ++voxel){
            // Check if the current voxel is in the mask
            if(maskPtr[voxel]>-1){
                // Read referenceImage and warpedImage probabilities and compute the ratio
                tempValue=(currentRefPtr[voxel]+1e-16)/(currentWarPtr[voxel]+1e-16);
                // Check if the intensity ratio is defined and different from zero
                if(tempValue==tempValue &&
                   tempValue!=std::numeric_limits<double>::infinity() &&
                   tempValue>0){

                    // Jacobian modulation if the Jacobian determinant image is defined
                    if(jacobianDetImg!=NULL)
                        tempValue *= jacPtr[voxel];

                    // Ensure that gradient of the warpedImage image along x-axis is not NaN
                    tempGradX=currentGradPtrX[voxel];
                    if(tempGradX==tempGradX)
                    // Update the gradient along the x-axis
                       kldGradPtrX[voxel] -= (DTYPE)(tempValue * tempGradX);

                    // Ensure that gradient of the warpedImage image along y-axis is not NaN
                    tempGradY=currentGradPtrY[voxel];
                    if(tempGradY==tempGradY)
                    // Update the gradient along the y-axis
                        kldGradPtrY[voxel] -= (DTYPE)(tempValue * tempGradY);

                    // Check if the current images are 3D
                    if(referenceImage->nz>1){
                        // Ensure that gradient of the warpedImage image along z-axis is not NaN
                        tempGradZ=currentGradPtrZ[voxel];
                        if(tempGradZ==tempGradZ)
                        // Update the gradient along the z-axis
                            kldGradPtrZ[voxel] -= (DTYPE)(tempValue * tempGradZ);
                    }
                }
            }
        }
    }
}
/* *************************************************************** */
void reg_getKLDivergenceVoxelBasedGradient(nifti_image *referenceImage,
                                           nifti_image *warpedImage,
                                           nifti_image *warpedImageGradient,
                                           nifti_image *KLdivGradient,
                                           nifti_image *jacobianDetImg,
                                           int *mask)
{
    if(referenceImage->datatype!=warpedImage->datatype ||
       referenceImage->datatype!=warpedImageGradient->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergenceVoxelBasedGradient: input images are expected to have the same type\n");
        exit(1);
    }
    if(jacobianDetImg!=NULL){
        if(referenceImage->datatype!=jacobianDetImg->datatype){
            fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergenceVoxelBasedGradient: input images are expected to have the same type\n");
            exit(1);
        }
    }
    if(referenceImage->nvox!=warpedImage->nvox){
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergenceVoxelBasedGradient: both input images have different size\n");
        exit(1);
    }
    if(referenceImage->nz>1 && warpedImageGradient->nu!=3 && KLdivGradient->nu!=3){
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergenceVoxelBasedGradient: check code\n");
        exit(1);
    }
    switch(referenceImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getKLDivergenceVoxelBasedGradient1<float>
                (referenceImage,warpedImage,warpedImageGradient,KLdivGradient,jacobianDetImg,mask);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getKLDivergenceVoxelBasedGradient1<double>
                (referenceImage,warpedImage,warpedImageGradient,KLdivGradient,jacobianDetImg,mask);
        break;
    default:
        fprintf(stderr, "[NiftyReg ERROR] reg_getKLDivergenceVoxelBasedGradient: unsupported datatype\n");
        exit(1);
        break;
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
