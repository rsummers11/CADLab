/*
 *  _reg_ssd.cpp
 *  
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ssd.h"

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getSSD1(nifti_image *referenceImage,
                   nifti_image *warpedImage,
                   nifti_image *jacobianDetImage,
                   int *mask
                   )
{
    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    // Create pointers to the reference and warped image data
    DTYPE *referencePtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warpedPtr=static_cast<DTYPE *>(warpedImage->data);
    // Create a pointer to the Jacobian determinant image if defined
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    // Create some variables to be use in the openmp loop
#ifdef _WINDOWS
    int  voxel;
#else
    size_t  voxel;
#endif
    
    double SSD=0.0, n=0.0;
    double targetValue, resultValue, diff;

    // Loop over the different time points
    for(int time=0;time<referenceImage->nt;++time){

        // Create pointers to the current time point of the reference and warped images
        DTYPE *currentRefPtr=&referencePtr[time*voxelNumber];
        DTYPE *currentWarPtr=&warpedPtr[time*voxelNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, currentRefPtr, currentWarPtr, mask, \
    jacobianDetImage, jacDetPtr, voxelNumber) \
    private(voxel, targetValue, resultValue, diff) \
    reduction(+:SSD) \
    reduction(+:n)
#endif
        for(voxel=0; voxel<voxelNumber;++voxel){
            // Check if the current voxel belongs to the mask
            if(mask[voxel]>-1){
                // Ensure that both ref and warped values are defined
                targetValue = (double)currentRefPtr[voxel];
                resultValue = (double)currentWarPtr[voxel];
                if(targetValue==targetValue && resultValue==resultValue){
                    diff = (targetValue-resultValue);
                    // Jacobian determinant modulation of the ssd if required
                    if(jacobianDetImage!=NULL){
                        SSD += diff * diff * jacDetPtr[voxel];
                        n += jacDetPtr[voxel];
                    }
                    else{
                        SSD += diff * diff;
                        n += 1.0;
                    }
                }
            }
        }
    }

    return SSD/n;
}
/* *************************************************************** */
double reg_getSSD(nifti_image *referenceImage,
                  nifti_image *warpedImage,
                  nifti_image *jacobianDetImage,
                  int *mask
                  )
{
    // Check that all input images are of the same type
    if(jacobianDetImage!=NULL){
        if(referenceImage->datatype != warpedImage->datatype || referenceImage->datatype != jacobianDetImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getSSD\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
            exit(1);
        }
    }
    else{
        if(referenceImage->datatype != warpedImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getSSD\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
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

    switch ( referenceImage->datatype ){
        case NIFTI_TYPE_FLOAT32:
            return reg_getSSD1<float>(referenceImage,warpedImage, jacobianDetImage, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getSSD1<double>(referenceImage,warpedImage, jacobianDetImage, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD computation function.\n");
            exit(1);
	}
	return 0.0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSSDGradient1(nifti_image *referenceImage,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedImageGradient,
                                   nifti_image *ssdGradientImage,
                                   nifti_image *jacobianDetImage,
                                   float maxSD,
                                   int *mask
                                   )
{
    // Create pointers to the reference and warped images
    size_t voxelNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;

#ifdef _WINDOWS
    int  voxel;
#else
    size_t  voxel;
#endif


    DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
    DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

    // Pointer to the warped image gradient
    DTYPE *spatialGradPtr=static_cast<DTYPE *>(warpedImageGradient->data);

    // Create pointers to the voxel based gradient image - results
    DTYPE *ssdGradPtrX=static_cast<DTYPE *>(ssdGradientImage->data);
    DTYPE *ssdGradPtrY = &ssdGradPtrX[voxelNumber];
    DTYPE *ssdGradPtrZ = NULL;

    // Create the z-axis pointers if the images are volume
    if(referenceImage->nz>1)
        ssdGradPtrZ = &ssdGradPtrY[voxelNumber];

    // Set all the gradient values to zero
    for(voxel=0;voxel<ssdGradientImage->nvox;++voxel)
        ssdGradPtrX[voxel]=0;

    // Create a pointer to the Jacobian determinant values if defined
    DTYPE *jacDetPtr=NULL;
    if(jacobianDetImage!=NULL)
        jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

    DTYPE gradX, gradY, gradZ;
    double JacDetValue, targetValue, resultValue, common;

    // Loop over the different time points
    for(int time=0;time<referenceImage->nt;++time){
        // Create some pointers to the current time point image to be accessed
        DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
        DTYPE *currentWarPtr=&warPtr[time*voxelNumber];
        // Create some pointers to the spatial gradient of the current warped volume
        DTYPE *currentGradPtrX=&spatialGradPtr[time*voxelNumber];
        DTYPE *currentGradPtrY=&currentGradPtrX[referenceImage->nt*voxelNumber];
        DTYPE *currentGradPtrZ=NULL;
        if(referenceImage->nz>1)
            currentGradPtrZ=&currentGradPtrY[referenceImage->nt*voxelNumber];

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(referenceImage, currentRefPtr, currentWarPtr, maxSD, mask, jacDetPtr, jacobianDetImage, \
    currentGradPtrX, currentGradPtrY, currentGradPtrZ, ssdGradPtrX, ssdGradPtrY, ssdGradPtrZ, voxelNumber) \
    private(voxel, targetValue, resultValue, common, gradX, gradY, gradZ, JacDetValue)
#endif
        for(voxel=0; voxel<voxelNumber;voxel++){
            if(mask[voxel]>-1){
                targetValue = currentRefPtr[voxel];
                resultValue = currentWarPtr[voxel];
                gradX=0;
                gradY=0;
                gradZ=0;
                if(targetValue==targetValue && resultValue==resultValue){
                    common = - 2.0 * (targetValue - resultValue)/maxSD;
                    gradX = (DTYPE)(common * currentGradPtrX[voxel]);
                    gradY = (DTYPE)(common * currentGradPtrY[voxel]);
                    if(referenceImage->nz>1)
                        gradZ = (DTYPE)(common * currentGradPtrZ[voxel]);
                    if(jacobianDetImage!=NULL){
                        JacDetValue = jacDetPtr[voxel];
                        gradX *= JacDetValue;
                        gradY *= JacDetValue;
                        if(referenceImage->nz>1)
                            gradZ *= JacDetValue;
                    }
                    ssdGradPtrX[voxel] += gradX;
                    ssdGradPtrY[voxel] += gradY;
                    if(referenceImage->nz>1)
                        ssdGradPtrZ[voxel] += gradZ;
                }
            }
        }
    }// loop over time points
}
/* *************************************************************** */
void reg_getVoxelBasedSSDGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  nifti_image *warpedImageGradient,
                                  nifti_image *ssdGradientImage,
                                  nifti_image *jacobianDeterminantImage,
                                  float maxSD,
                                  int *mask
                                  )
{
    if(referenceImage->datatype != warpedImage->datatype ||
       warpedImageGradient->datatype != ssdGradientImage->datatype ||
       referenceImage->datatype != warpedImageGradient->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedSSDGradient\n");
        fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
        exit(1);
    }
    if(jacobianDeterminantImage!=NULL){
        if(referenceImage->datatype != jacobianDeterminantImage->datatype){
            fprintf(stderr,"[NiftyReg ERROR] reg_getVoxelBasedSSDGradient\n");
            fprintf(stderr,"[NiftyReg ERROR] Input images are expected to have the same type\n");
            exit(1);
        }
    }
    switch ( referenceImage->datatype ){
        case NIFTI_TYPE_FLOAT32:
            reg_getVoxelBasedSSDGradient1<float>
                (referenceImage, warpedImage, warpedImageGradient, ssdGradientImage, jacobianDeterminantImage,maxSD, mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_getVoxelBasedSSDGradient1<double>
                (referenceImage, warpedImage, warpedImageGradient, ssdGradientImage, jacobianDeterminantImage,maxSD, mask);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Target pixel type unsupported in the SSD gradient computation function.\n");
            exit(1);
	}
}
/* *************************************************************** */
/* *************************************************************** */
