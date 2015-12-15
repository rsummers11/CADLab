/*
 *  _reg_tools.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_CPP
#define _REG_TOOLS_CPP

#include "_reg_tools.h"

/* *************************************************************** */
/* *************************************************************** */
void reg_checkAndCorrectDimension(nifti_image *image)
{
    // Ensure that no dimension is set to zero
    if(image->nx<1 || image->dim[1]<1) image->dim[1]=image->nx=1;
    if(image->ny<1 || image->dim[2]<1) image->dim[2]=image->ny=1;
    if(image->nz<1 || image->dim[3]<1) image->dim[3]=image->nz=1;
    if(image->nt<1 || image->dim[4]<1) image->dim[4]=image->nt=1;
    if(image->nu<1 || image->dim[5]<1) image->dim[5]=image->nu=1;
    if(image->nv<1 || image->dim[6]<1) image->dim[6]=image->nv=1;
    if(image->nw<1 || image->dim[7]<1) image->dim[7]=image->nw=1;
    // Set the slope to 1 if undefined
    if(image->scl_slope==0) image->scl_slope=1.f;
    // Ensure that no spacing is set to zero
    if(image->ny==1 && (image->dy==0 || image->pixdim[2]==0))
        image->dy=image->pixdim[2]=1;
    if(image->nz==1 && (image->dz==0 || image->pixdim[3]==0))
        image->dz=image->pixdim[3]=1;
    // Create the qform matrix if required
    if(image->qform_code==0 && image->sform_code==0){
        image->qto_xyz=nifti_quatern_to_mat44(image->quatern_b,
                                              image->quatern_c,
                                              image->quatern_d,
                                              image->qoffset_x,
                                              image->qoffset_y,
                                              image->qoffset_z,
                                              image->dx,
                                              image->dy,
                                              image->dz,
                                              image->qfac);
        image->qto_ijk=nifti_mat44_inverse(image->qto_xyz);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_intensityRescale2(nifti_image *image,
                           float *newMin,
                           float *newMax,
                           float *lowThr,
                           float *upThr
                           )
{
    DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
    unsigned int voxelNumber = image->nx*image->ny*image->nz;

    // The rescasling is done for each volume independtly
    for(int t=0;t<image->nt;t++){
        DTYPE *volumePtr = &imagePtr[t*voxelNumber];
        DTYPE currentMin=0;
        DTYPE currentMax=0;
        switch(image->datatype){
        case NIFTI_TYPE_UINT8:
            currentMin=(DTYPE)std::numeric_limits<unsigned char>::max();
            currentMax=0;
            break;
        case NIFTI_TYPE_INT8:
            currentMin=(DTYPE)std::numeric_limits<char>::max();
            currentMax=(DTYPE)-std::numeric_limits<char>::max();
            break;
        case NIFTI_TYPE_UINT16:
            currentMin=(DTYPE)std::numeric_limits<unsigned short>::max();
            currentMax=0;
            break;
        case NIFTI_TYPE_INT16:
            currentMin=(DTYPE)std::numeric_limits<char>::max();
            currentMax=-(DTYPE)std::numeric_limits<char>::max();
            break;
        case NIFTI_TYPE_UINT32:
            currentMin=(DTYPE)std::numeric_limits<unsigned int>::max();
            currentMax=0;
            break;
        case NIFTI_TYPE_INT32:
            currentMin=(DTYPE)std::numeric_limits<int>::max();
            currentMax=-(DTYPE)std::numeric_limits<int>::max();
            break;
        case NIFTI_TYPE_FLOAT32:
            currentMin=(DTYPE)std::numeric_limits<float>::max();
            currentMax=-(DTYPE)std::numeric_limits<float>::max();
            break;
        case NIFTI_TYPE_FLOAT64:
            currentMin=(DTYPE)std::numeric_limits<double>::max();
            currentMax=-(DTYPE)std::numeric_limits<double>::max();
            break;
        }

        // Extract the minimal and maximal values from the current volume
        if(image->scl_slope==0) image->scl_slope=1.0f;
        for(unsigned int index=0; index<voxelNumber; index++){
            DTYPE value = (DTYPE)(*volumePtr++ * image->scl_slope + image->scl_inter);
            if(value==value){
                currentMin=(currentMin<value)?currentMin:value;
                currentMax=(currentMax>value)?currentMax:value;
            }
        }

        // Check if the current extrama are outside of the user-specified threshold values
        if(currentMin<lowThr[t]) currentMin=(DTYPE)lowThr[t];
        if(currentMax>upThr[t]) currentMax=(DTYPE)upThr[t];

        // Compute constant values to rescale image intensities
        double currentDiff = (double)(currentMax-currentMin);
        double newDiff = (double)(newMax[t]-newMin[t]);

        // Set the image header information for appropriate display
        image->cal_min=newMin[t] * image->scl_slope + image->scl_inter;
        image->cal_max=newMax[t] * image->scl_slope + image->scl_inter;

        // Reset the volume pointer to the start of the current volume
        volumePtr = &imagePtr[t*voxelNumber];

        // Iterates over all voxels in the current volume
        for(unsigned int index=0; index<voxelNumber; index++){
            double value = (double)*volumePtr * image->scl_slope + image->scl_inter;
            // Check if the value is defined
            if(value==value){
                // Lower threshold is applied
                if(value<currentMin){
                    value = newMin[t];
                }
                // upper threshold is applied
                else if(value>currentMax){
                    value = newMax[t];
                }
                else{
                    // Normalise the value between 0 and 1
                    value = (value-(double)currentMin)/currentDiff;
                    // Rescale the value using the specified range
                    value = value * newDiff + newMin[t];
                }
            }
            *volumePtr++=(DTYPE)value;
        }
    }//t
    // The slope and offset information are cleared form the header
    image->scl_slope=1.f;
    image->scl_inter=0.f;
}
/* *************************************************************** */
void reg_intensityRescale(	nifti_image *image,
                            float *newMin,
                            float *newMax,
                            float *lowThr,
                            float *upThr
                            )
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_intensityRescale2<unsigned char>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT8:
        reg_intensityRescale2<char>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT16:
        reg_intensityRescale2<unsigned short>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT16:
        reg_intensityRescale2<short>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT32:
        reg_intensityRescale2<unsigned int>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT32:
        reg_intensityRescale2<int>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_intensityRescale2<float>(image, newMin, newMax, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_intensityRescale2<double>(image, newMin, newMax, lowThr, upThr);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_intensityRescale\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
//this function will threshold an image to the values provided,
//set the scl_slope and sct_inter of the image to 1 and 0 (SSD uses actual image data values),
//and sets cal_min and cal_max to have the min/max image data values
template<class T,class DTYPE>
void reg_thresholdImage2(	nifti_image *image,
                            T lowThr,
                            T upThr
                            )
{
    DTYPE *imagePtr = static_cast<DTYPE *>(image->data);
    T currentMin=std::numeric_limits<T>::max();
    T currentMax=-std::numeric_limits<T>::max();

    if(image->scl_slope==0)image->scl_slope=1.0;

    for(unsigned int index=0; index<image->nvox; index++){
        T value = (T)(*imagePtr * image->scl_slope + image->scl_inter);
        if(value==value){
            if(value<lowThr){
                value = lowThr;
            }
            else if(value>upThr){
                value = upThr;
            }
            currentMin=(currentMin<value)?currentMin:value;
            currentMax=(currentMax>value)?currentMax:value;
        }
        *imagePtr++=(DTYPE)value;
    }

    image->cal_min = currentMin;
    image->cal_max = currentMax;
}
/* *************************************************************** */
template<class T>
void reg_thresholdImage(	nifti_image *image,
                            T lowThr,
                            T upThr
                            )
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_thresholdImage2<T,unsigned char>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT8:
        reg_thresholdImage2<T,char>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT16:
        reg_thresholdImage2<T,unsigned short>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT16:
        reg_thresholdImage2<T,short>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_UINT32:
        reg_thresholdImage2<T,unsigned int>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_INT32:
        reg_thresholdImage2<T,int>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_thresholdImage2<T,float>(image, lowThr, upThr);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_thresholdImage2<T,double>(image, lowThr, upThr);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_thresholdImage\tThe image data type is not supported\n");
        exit(1);
    }
}
template void reg_thresholdImage<float>(nifti_image *, float, float);
template void reg_thresholdImage<double>(nifti_image *, double, double);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
void reg_tools_CubicSplineKernelConvolution1(nifti_image *image,
                                             int radius[]
                                             )
{
    DTYPE *imageArray = static_cast<DTYPE *>(image->data);

    /* a temp image array is first created */
    DTYPE *tempArray  = (DTYPE *)malloc(image->nvox * sizeof(DTYPE));

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    /* Smoothing along the X axis */
    int windowSize = 2*radius[0] + 1;
    PrecisionTYPE *window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    PrecisionTYPE coeffSum=0.0;
    for(int it=-radius[0]; it<=radius[0]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[0]));
        if(coeff<1.0) window[it+radius[0]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
        else if (coeff<2.0) window[it+radius[0]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
        else window[it+radius[0]]=0;
        coeffSum += window[it+radius[0]];
    }
//	for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int index, i, X, it, x, y, z;
            PrecisionTYPE finalValue, windowValue, t, c, temp;
            PrecisionTYPE currentCoeffSum;
            DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, X, it, x, y, z, finalValue, windowValue, \
    c, t, temp, imageValue, currentCoeffSum) \
    shared(image, readingValue, writtingValue, radius, windowSize, window, coeffSum)
#endif // _OPENMP
            for(z=0; z<image->nz; z++){
                i=z*image->nx*image->ny;
                for(y=0; y<image->ny; y++){
                    for(x=0; x<image->nx; x++){

                        finalValue=0.0;
                        currentCoeffSum=0.0;

                        index = i - radius[0];
                        X = x - radius[0];
                        // Kahan summation used here
                        c = (PrecisionTYPE)0;
                        for(it=0; it<windowSize; it++){
                            if(-1<X && X<image->nx){
                                imageValue = readingValue[index];
                                windowValue = window[it];
                                temp = (PrecisionTYPE)imageValue * windowValue - c;
                                t = finalValue + temp;
                                c = (t - finalValue) - temp;
                                finalValue = t;
                            }
                            else currentCoeffSum += window[it];
                            index++;
                            X++;
                        }
                        if(currentCoeffSum!=0)
                            finalValue *= coeffSum / (coeffSum-currentCoeffSum);
                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }

    /* Smoothing along the Y axis */
    windowSize = 2*radius[1] + 1;
    free(window);
    window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    coeffSum=0.0;
    for(int it=-radius[1]; it<=radius[1]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[1]));
        if(coeff<1.0) window[it+radius[1]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
        else if (coeff<2.0) window[it+radius[1]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
        else window[it+radius[1]]=0;
        coeffSum += window[it+radius[1]];
    }
//    for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int index, i, Y, it, x, y, z;
            PrecisionTYPE finalValue, windowValue, t, c, temp;
            PrecisionTYPE currentCoeffSum;
            DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, Y, it, x, y, z, finalValue, windowValue, \
    c, t, temp, imageValue, currentCoeffSum) \
    shared(image, readingValue, writtingValue, radius, windowSize, window, coeffSum)
#endif // _OPENMP
            for(z=0; z<image->nz; z++){
                i=z*image->nx*image->ny;
                for(y=0; y<image->ny; y++){
                    for(x=0; x<image->nx; x++){

                        finalValue=0.0;
                        currentCoeffSum=0.0;

                        index = i - image->nx*radius[1];
                        Y = y - radius[1];

                        // Kahan summation used here
                        c = (PrecisionTYPE)0;
                        for(it=0; it<windowSize; it++){
                            if(-1<Y && Y<image->ny){
                                imageValue = readingValue[index];
                                windowValue = window[it];
                                temp = (PrecisionTYPE)imageValue * windowValue - c;
                                t = finalValue + temp;
                                c = (t - finalValue) - temp;
                                finalValue = t;
                            }
                            else currentCoeffSum += window[it];
                            index+=image->nx;
                            Y++;
                        }
                        if(currentCoeffSum!=0)
                            finalValue *= coeffSum / (coeffSum-currentCoeffSum);
                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }
    if(image->nz>1){
        /* Smoothing along the Z axis */
        windowSize = 2*radius[2] + 1;
        free(window);
        window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
        coeffSum=0.0;
        for(int it=-radius[2]; it<=radius[2]; it++){
            PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[2]));
            if(coeff<1.0) window[it+radius[2]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
            else if (coeff<2.0) window[it+radius[2]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
            else window[it+radius[2]]=0;
            coeffSum += window[it+radius[2]];
        }
//	    for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
        for(int t=0;t<timePoint;t++){
            for(int u=0;u<field;u++){

                DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
                DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];

                int index, i, Z, it, x, y, z;
                PrecisionTYPE finalValue, windowValue, t, c, temp;
                PrecisionTYPE currentCoeffSum;
                DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, Z, it, x, y, z, finalValue, windowValue, \
    c, t, temp, imageValue, currentCoeffSum) \
    shared(image, readingValue, writtingValue, radius, windowSize, window, coeffSum)
#endif // _OPENMP

                for(z=0; z<image->nz; z++){
                    i=z*image->nx*image->ny;
                    for(y=0; y<image->ny; y++){
                        for(x=0; x<image->nx; x++){

                            finalValue=0.0;
                            currentCoeffSum=0.0;

                            index = i - image->nx*image->ny*radius[2];
                            Z = z - radius[2];

                            // Kahan summation used here
                            c = (PrecisionTYPE)0;
                            for(it=0; it<windowSize; it++){
                                if(-1<Z && Z<image->nz){
                                    imageValue = readingValue[index];
                                    windowValue = window[it];
                                    temp = (PrecisionTYPE)imageValue * windowValue - c;
                                    t = finalValue + temp;
                                    c = (t - finalValue) - temp;
                                    finalValue = t;
                                }
                                else currentCoeffSum += window[it];
                                index+=image->nx*image->ny;
                                Z++;
                            }
                            if(currentCoeffSum!=0)
                                finalValue *= coeffSum / (coeffSum-currentCoeffSum);
                            writtingValue[i++] = (DTYPE)finalValue;
                        }
                    }
                }
            }
        }
        memcpy(imageArray,tempArray,image->nvox * sizeof(DTYPE));
    }
    free(window);
    free(tempArray);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_tools_CubicSplineKernelConvolution(nifti_image *image,
                                            int radius[]
                                            )
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,unsigned char>(image, radius);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,char>(image, radius);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,unsigned short>(image, radius);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,short>(image, radius);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,unsigned int>(image, radius);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,int>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,float>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_CubicSplineKernelConvolution1<PrecisionTYPE,double>(image, radius);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_CubicSplineKernelConvolution\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
template void reg_tools_CubicSplineKernelConvolution<float>(nifti_image *, int[]);
template void reg_tools_CubicSplineKernelConvolution<double>(nifti_image *, int[]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
void reg_smoothNormImageForCubicSpline1(nifti_image *image,
                                    int radius[]
                                    )
{
    DTYPE *imageArray = static_cast<DTYPE *>(image->data);

    /* a temp image array is first created */
    DTYPE *tempArray  = (DTYPE *)malloc(image->nvox * sizeof(DTYPE));

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    /* Smoothing along the X axis */
    int windowSize = 2*radius[0] + 1;
    PrecisionTYPE *window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
        PrecisionTYPE coeffSum=0.0;
    for(int it=-radius[0]; it<=radius[0]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[0]));
        if(coeff<1.0)	window[it+radius[0]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
        else		window[it+radius[0]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
                coeffSum += window[it+radius[0]];
    }
        for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int index, i, X, it, x, y, z;
            PrecisionTYPE finalValue, windowValue, t, c, temp;
            DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, X, it, x, y, z, finalValue, windowValue, c, t, temp, imageValue) \
    shared(image, readingValue, writtingValue, radius, windowSize, window)
#endif // _OPENMP
            for(z=0; z<image->nz; z++){
                i=z*image->nx*image->ny;
                for(y=0; y<image->ny; y++){
                    for(x=0; x<image->nx; x++){


                        index = i - radius[0];
                        X = x - radius[0];

                        finalValue=0.0;
                        // Kahan summation used here
                        c = (PrecisionTYPE)0;
                        for(it=0; it<windowSize; it++){
                            if(-1<X && X<image->nx){
                                imageValue = readingValue[index];
                                windowValue = window[it];
                                temp = (PrecisionTYPE)imageValue * windowValue - c;
                                t = finalValue + temp;
                                c = (t - finalValue) - temp;
                                finalValue = t;
                            }
                            index++;
                            X++;
                        }
                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }

    /* Smoothing along the Y axis */
    windowSize = 2*radius[1] + 1;
    free(window);
    window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
        coeffSum=0.0;
    for(int it=-radius[1]; it<=radius[1]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[1]));
        if(coeff<1.0)	window[it+radius[1]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
        else		window[it+radius[1]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
                coeffSum += window[it+radius[1]];
    }
        for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int index, i, Y, it, x, y, z;
            PrecisionTYPE finalValue, windowValue, t, c, temp;
            DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, Y, it, x, y, z, finalValue, windowValue, c, t, temp, imageValue) \
    shared(image, readingValue, writtingValue, radius, windowSize, window)
#endif // _OPENMP
            for(z=0; z<image->nz; z++){
                i=z*image->nx*image->ny;
                for(y=0; y<image->ny; y++){
                    for(x=0; x<image->nx; x++){

                        finalValue=0.0;

                        index = i - image->nx*radius[1];
                        Y = y - radius[1];

                        // Kahan summation used here
                        c = (PrecisionTYPE)0;
                        for(it=0; it<windowSize; it++){
                            if(-1<Y && Y<image->ny){
                                imageValue = readingValue[index];
                                windowValue = window[it];
                                temp = (PrecisionTYPE)imageValue * windowValue - c;
                                t = finalValue + temp;
                                c = (t - finalValue) - temp;
                                finalValue = t;
                            }
                            index+=image->nx;
                            Y++;
                        }

                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }
    if(image->nz>1){
        /* Smoothing along the Z axis */
        windowSize = 2*radius[2] + 1;
        free(window);
        window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
            coeffSum=0.0;
        for(int it=-radius[2]; it<=radius[2]; it++){
            PrecisionTYPE coeff = (PrecisionTYPE)(fabs(2.0*(PrecisionTYPE)it/(PrecisionTYPE)radius[2]));
            if(coeff<1.0)	window[it+radius[2]] = (PrecisionTYPE)(2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff);
            else		window[it+radius[2]] = (PrecisionTYPE)(-(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0);
                        coeffSum += window[it+radius[2]];
        }
            for(int it=0;it<windowSize;it++)window[it] /= coeffSum;
        for(int t=0;t<timePoint;t++){
            for(int u=0;u<field;u++){

                DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
                DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];

                int index, i, Z, it, x, y, z;
                PrecisionTYPE finalValue, windowValue, t, c, temp;
                DTYPE imageValue;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, i, Z, it, x, y, z, finalValue, windowValue, c, t, temp, imageValue) \
    shared(image, readingValue, writtingValue, radius, windowSize, window)
#endif // _OPENMP

                for(z=0; z<image->nz; z++){
                    i=z*image->nx*image->ny;
                    for(y=0; y<image->ny; y++){
                        for(x=0; x<image->nx; x++){

                            finalValue=0.0;

                            index = i - image->nx*image->ny*radius[2];
                            Z = z - radius[2];

                            // Kahan summation used here
                            c = (PrecisionTYPE)0;
                            for(it=0; it<windowSize; it++){
                                if(-1<Z && Z<image->nz){
                                    imageValue = readingValue[index];
                                    windowValue = window[it];
                                    temp = (PrecisionTYPE)imageValue * windowValue - c;
                                    t = finalValue + temp;
                                    c = (t - finalValue) - temp;
                                    finalValue = t;
                                }
                                index+=image->nx*image->ny;
                                Z++;
                            }

                            writtingValue[i++] = (DTYPE)finalValue;
                        }
                    }
                }
            }
        }
        memcpy(imageArray,tempArray,image->nvox * sizeof(DTYPE));
    }
    free(window);
    free(tempArray);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_smoothNormImageForCubicSpline(	nifti_image *image,
                                    int radius[]
                                    )
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,unsigned char>(image, radius);
        break;
    case NIFTI_TYPE_INT8:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,char>(image, radius);
        break;
    case NIFTI_TYPE_UINT16:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,unsigned short>(image, radius);
        break;
    case NIFTI_TYPE_INT16:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,short>(image, radius);
        break;
    case NIFTI_TYPE_UINT32:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,unsigned int>(image, radius);
        break;
    case NIFTI_TYPE_INT32:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,int>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,float>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_smoothNormImageForCubicSpline1<PrecisionTYPE,double>(image, radius);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_smoothImage\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
template void reg_smoothNormImageForCubicSpline<float>(nifti_image *, int[]);
template void reg_smoothNormImageForCubicSpline<double>(nifti_image *, int[]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
void reg_smoothImageForTrilinear1(	nifti_image *image,
                                    int radius[]
                                    )
{
    DTYPE *imageArray = static_cast<DTYPE *>(image->data);

    /* a temp image array is first created */
    DTYPE *tempArray  = (DTYPE *)malloc(image->nvox * sizeof(DTYPE));

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    /* Smoothing along the X axis */
    int windowSize = 2*radius[0] + 1;
    // 	printf("window size along X: %i\n", windowSize);
    PrecisionTYPE *window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    PrecisionTYPE coeffSum=0.0;
    for(int it=-radius[0]; it<=radius[0]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
        window[it+radius[0]] = coeff;
        coeffSum += coeff;
    }
    for(int it=0;it<windowSize;it++){
        //printf("coeff[%i] = %g -> ", it, window[it]);
        window[it] /= coeffSum;
        //printf("%g\n", window[it]);
    }
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int i=0;
            for(int z=0; z<image->nz; z++){
                for(int y=0; y<image->ny; y++){
                    for(int x=0; x<image->nx; x++){

                        PrecisionTYPE finalValue=0.0;

                        int index = i - radius[0];
                        int X = x - radius[0];

                        for(int it=0; it<windowSize; it++){
                            if(-1<X && X<image->nx){
                                DTYPE imageValue = readingValue[index];
                                PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
                                    finalValue += (PrecisionTYPE)imageValue * windowValue;
                            }
                            index++;
                            X++;
                        }

                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }

    /* Smoothing along the Y axis */
    windowSize = 2*radius[1] + 1;
    // 	printf("window size along Y: %i\n", windowSize);
    free(window);
    window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    coeffSum=0.0;
    for(int it=-radius[1]; it<=radius[1]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
        window[it+radius[1]] = coeff;
        coeffSum += coeff;
    }
    for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int i=0;
            for(int z=0; z<image->nz; z++){
                for(int y=0; y<image->ny; y++){
                    for(int x=0; x<image->nx; x++){

                        PrecisionTYPE finalValue=0.0;

                        int index = i - image->nx*radius[1];
                        int Y = y - radius[1];

                        for(int it=0; it<windowSize; it++){
                            if(-1<Y && Y<image->ny){
                                DTYPE imageValue = readingValue[index];
                                PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
                                    finalValue += (PrecisionTYPE)imageValue * windowValue;
                            }
                            index+=image->nx;
                            Y++;
                        }

                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }

    /* Smoothing along the Z axis */
    windowSize = 2*radius[2] + 1;
    // 	printf("window size along Z: %i\n", windowSize);
    free(window);
    window = (PrecisionTYPE *)calloc(windowSize,sizeof(PrecisionTYPE));
    coeffSum=0.0;
    for(int it=-radius[2]; it<=radius[2]; it++){
        PrecisionTYPE coeff = (PrecisionTYPE)(fabs(1.0 -fabs((PrecisionTYPE)(it)/(PrecisionTYPE)radius[0] )));
        window[it+radius[2]] = coeff;
        coeffSum += coeff;
    }
    for(int it=0;it<windowSize;it++) window[it] /= coeffSum;
    for(int t=0;t<timePoint;t++){
        for(int u=0;u<field;u++){

            DTYPE *readingValue=&imageArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            DTYPE *writtingValue=&tempArray[(t+u*timePoint)*image->nx*image->ny*image->nz];
            int i=0;
            for(int z=0; z<image->nz; z++){
                for(int y=0; y<image->ny; y++){
                    for(int x=0; x<image->nx; x++){

                        PrecisionTYPE finalValue=0.0;

                        int index = i - image->nx*image->ny*radius[2];
                        int Z = z - radius[2];

                        for(int it=0; it<windowSize; it++){
                            if(-1<Z && Z<image->nz){
                                DTYPE imageValue = readingValue[index];
                                PrecisionTYPE windowValue = window[it];
                                if(windowValue==windowValue)
                                    finalValue += (PrecisionTYPE)imageValue * windowValue;
                            }
                            index+=image->nx*image->ny;
                            Z++;
                        }

                        writtingValue[i++] = (DTYPE)finalValue;
                    }
                }
            }
        }
    }
    free(window);
    memcpy(imageArray,tempArray,image->nvox * sizeof(DTYPE));
    free(tempArray);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_smoothImageForTrilinear(	nifti_image *image,
                                    int radius[]
                                    )
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_smoothImageForTrilinear1<PrecisionTYPE,unsigned char>(image, radius);
        break;
    case NIFTI_TYPE_INT8:
        reg_smoothImageForTrilinear1<PrecisionTYPE,char>(image, radius);
        break;
    case NIFTI_TYPE_UINT16:
        reg_smoothImageForTrilinear1<PrecisionTYPE,unsigned short>(image, radius);
        break;
    case NIFTI_TYPE_INT16:
        reg_smoothImageForTrilinear1<PrecisionTYPE,short>(image, radius);
        break;
    case NIFTI_TYPE_UINT32:
        reg_smoothImageForTrilinear1<PrecisionTYPE,unsigned int>(image, radius);
        break;
    case NIFTI_TYPE_INT32:
        reg_smoothImageForTrilinear1<PrecisionTYPE,int>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_smoothImageForTrilinear1<PrecisionTYPE,float>(image, radius);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_smoothImageForTrilinear1<PrecisionTYPE,double>(image, radius);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_smoothImage\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
template void reg_smoothImageForTrilinear<float>(nifti_image *, int[]);
template void reg_smoothImageForTrilinear<double>(nifti_image *, int[]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength2D(nifti_image *image)
{
    DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
    DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];

    PrecisionTYPE max=0.0;

    for(int i=0; i<image->nx*image->ny*image->nz; i++){
        PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
        PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
        PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY));
        max = (length>max)?length:max;
    }
    return max;
}
/* *************************************************************** */
template <class PrecisionTYPE, class DTYPE>
PrecisionTYPE reg_getMaximalLength3D(nifti_image *image)
{
    DTYPE *dataPtrX = static_cast<DTYPE *>(image->data);
    DTYPE *dataPtrY = &dataPtrX[image->nx*image->ny*image->nz];
    DTYPE *dataPtrZ = &dataPtrY[image->nx*image->ny*image->nz];

    PrecisionTYPE max=0.0;

    for(int i=0; i<image->nx*image->ny*image->nz; i++){
        PrecisionTYPE valX = (PrecisionTYPE)(*dataPtrX++);
        PrecisionTYPE valY = (PrecisionTYPE)(*dataPtrY++);
        PrecisionTYPE valZ = (PrecisionTYPE)(*dataPtrZ++);
        PrecisionTYPE length = (PrecisionTYPE)(sqrt(valX*valX + valY*valY + valZ*valZ));
        max = (length>max)?length:max;
    }
    return max;
}
/* *************************************************************** */
template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image)
{
    if(image->nz==1){
        switch(image->datatype){
        case NIFTI_TYPE_FLOAT32:
            return reg_getMaximalLength2D<PrecisionTYPE,float>(image);
            break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getMaximalLength2D<PrecisionTYPE,double>(image);
            break;
        }
    }
    else{
        switch(image->datatype){
        case NIFTI_TYPE_FLOAT32:
            return reg_getMaximalLength3D<PrecisionTYPE,float>(image);
            break;
        case NIFTI_TYPE_FLOAT64:
            return reg_getMaximalLength3D<PrecisionTYPE,double>(image);
            break;
        }
    }
    return 0;
}
/* *************************************************************** */
template float reg_getMaximalLength<float>(nifti_image *);
template double reg_getMaximalLength<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class NewTYPE, class DTYPE>
void reg_tools_changeDatatype1(nifti_image *image)
{
    // the initial array is saved and freeed
    DTYPE *initialValue = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
    memcpy(initialValue, image->data, image->nvox*sizeof(DTYPE));

    // the new array is allocated and then filled
    if(sizeof(NewTYPE)==sizeof(unsigned char)) image->datatype = NIFTI_TYPE_UINT8;
    else if(sizeof(NewTYPE)==sizeof(float)) image->datatype = NIFTI_TYPE_FLOAT32;
    else if(sizeof(NewTYPE)==sizeof(double)) image->datatype = NIFTI_TYPE_FLOAT64;
    else{
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_changeDatatype\tOnly change to unsigned char, float or double are supported\n");
        exit(1);
    }
    free(image->data);
    image->nbyper = sizeof(NewTYPE);
    image->data = (void *)calloc(image->nvox,sizeof(NewTYPE));
    NewTYPE *dataPtr = static_cast<NewTYPE *>(image->data);
    for(unsigned int i=0; i<image->nvox; i++) dataPtr[i] = (NewTYPE)(initialValue[i]);

    free(initialValue);
    return;
}
/* *************************************************************** */
template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image)
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_changeDatatype1<NewTYPE,unsigned char>(image);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_changeDatatype1<NewTYPE,char>(image);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_changeDatatype1<NewTYPE,unsigned short>(image);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_changeDatatype1<NewTYPE,short>(image);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_changeDatatype1<NewTYPE,unsigned int>(image);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_changeDatatype1<NewTYPE,int>(image);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_changeDatatype1<NewTYPE,float>(image);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_changeDatatype1<NewTYPE,double>(image);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_changeDatatype\tThe initial image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
template void reg_tools_changeDatatype<unsigned char>(nifti_image *);
template void reg_tools_changeDatatype<float>(nifti_image *);
template void reg_tools_changeDatatype<double>(nifti_image *);
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1, class TYPE2>
void reg_tools_addSubMulDivImages2( nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
    TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);
    TYPE2 *img2Ptr = static_cast<TYPE2 *>(img2->data);


    if(img1->scl_slope==0){
        img1->scl_slope=1.f;
        res->scl_slope=1.f;
    }
    if(img2->scl_slope==0)
        img2->scl_slope=1.f;

    switch(type){
    case 0:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) +
                                 ((double)*img2Ptr++ * (double)img2->scl_slope + (double)img2->scl_inter) -
                                 (double)img1->scl_inter)/(double)img1->scl_slope);
        //            *resPtr++ = (TYPE1)((double)*img1Ptr++ + (double)*img2Ptr++);
        break;
    case 1:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) -
                                 ((double)*img2Ptr++ * (double)img2->scl_slope + (double)img2->scl_inter) -
                                 (double)img1->scl_inter)/(double)img1->scl_slope);
        //            *resPtr++ = (TYPE1)((double)*img1Ptr++ - (double)*img2Ptr++);
        break;
    case 2:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) *
                                 ((double)*img2Ptr++ * (double)img2->scl_slope + (double)img2->scl_inter) -
                                 (double)img1->scl_inter)/(double)img1->scl_slope);
        //            *resPtr++ = (TYPE1)((double)*img1Ptr++ * (double)*img2Ptr++);
        break;
    case 3:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) /
                                 ((double)*img2Ptr++ * (double)img2->scl_slope + (double)img2->scl_inter) -
                                 (double)img1->scl_inter)/(double)img1->scl_slope);
        //            *resPtr++ = (TYPE1)((double)*img1Ptr++ / (double)*img2Ptr++);
        break;
    }
}
/* *************************************************************** */
template <class TYPE1>
void reg_tools_addSubMulDivImages1( nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    switch(img2->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_addSubMulDivImages2<TYPE1,unsigned char>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_addSubMulDivImages2<TYPE1,char>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_addSubMulDivImages2<TYPE1,unsigned short>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_addSubMulDivImages2<TYPE1,short>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_addSubMulDivImages2<TYPE1,unsigned int>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_addSubMulDivImages2<TYPE1,int>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_addSubMulDivImages2<TYPE1,float>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_addSubMulDivImages2<TYPE1,double>(img1, img2, res, type);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivImages1\tSecond image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
void reg_tools_addSubMulDivImages(  nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *res,
                                    int type)
{
    
    if(img1->dim[1]!=img2->dim[1] ||
            img1->dim[2]!=img2->dim[2] ||
            img1->dim[3]!=img2->dim[3] ||
            img1->dim[4]!=img2->dim[4] ||
            img1->dim[5]!=img2->dim[5] ||
            img1->dim[6]!=img2->dim[6] ||
            img1->dim[7]!=img2->dim[7]){
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivImages\tBoth images do not have the same dimension\n");
        exit(1);
    }

    if(img1->datatype != res->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivImages\tFirst and result image do not have the same data type\n");
        exit(1);
    }
    switch(img1->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_addSubMulDivImages1<unsigned char>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_addSubMulDivImages1<char>(img1, img1, res, type);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_addSubMulDivImages1<unsigned short>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_addSubMulDivImages1<short>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_addSubMulDivImages1<unsigned int>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_addSubMulDivImages1<int>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_addSubMulDivImages1<float>(img1, img2, res, type);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_addSubMulDivImages1<double>(img1, img2, res, type);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivImages1\tFirst image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1>
void reg_tools_addSubMulDivValue1(  nifti_image *img1,
                                    nifti_image *res,
                                    float val,
                                    int type)
{
    TYPE1 *img1Ptr = static_cast<TYPE1 *>(img1->data);
    TYPE1 *resPtr = static_cast<TYPE1 *>(res->data);

    if(img1->scl_slope==0){
        img1->scl_slope=1.f;
        res->scl_slope=1.f;
    }

    switch(type){
    case 0:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)(((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) +
                                  (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
        break;
    case 1:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)(((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) -
                                  (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
        break;
    case 2:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)(((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) *
                                  (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
        break;
    case 3:
        for(unsigned int i=0; i<res->nvox; i++)
            *resPtr++ = (TYPE1)(((((double)*img1Ptr++ * (double)img1->scl_slope + (double)img1->scl_inter) /
                                  (double)val) - (double)img1->scl_inter)/(double)img1->scl_slope);
        break;
    }
}
/* *************************************************************** */
void reg_tools_addSubMulDivValue(   nifti_image *img1,
                                    nifti_image *res,
                                    float val,
                                    int type)
{
    if(img1->datatype != res->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivValue\tInput and result image do not have the same data type\n");
        exit(1);
    }
    switch(img1->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_addSubMulDivValue1<unsigned char>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_addSubMulDivValue1<char>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_addSubMulDivValue1<unsigned short>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_addSubMulDivValue1<short>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_addSubMulDivValue1<unsigned int>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_addSubMulDivValue1<int>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_addSubMulDivValue1<float>
                (img1, res, val, type);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_addSubMulDivValue1<double>
                (img1, res, val, type);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_addSubMulDivImages1\tFirst image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_gaussianSmoothing1(nifti_image *image,
                            PrecisionTYPE sigma,
                            bool axisToSmooth[8])
{
    ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);

    int timePoint = image->nt;
    if(timePoint==0) timePoint=1;
    int field = image->nu;
    if(field==0) field=1;

    int voxelNumber = image->nx*image->ny*image->nz;

    int index, startingIndex, x, i, j, t, current, n, radius, increment;
    PrecisionTYPE value;

    // Loop over the dimension higher than 3
    for(t=0; t<timePoint*field; t++){
        ImageTYPE *timeImagePtr = &imagePtr[t * voxelNumber];
        PrecisionTYPE *resultValue=(PrecisionTYPE *)malloc(voxelNumber * sizeof(PrecisionTYPE));
        // Loop over the 3 dimensions
        for(n=1; n<4; n++){
            if(axisToSmooth[n]==true && image->dim[n]>1){
                // Define the Guassian kernel
                float currentSigma;
                if(sigma>0) currentSigma=sigma/image->pixdim[n];
                else currentSigma=fabs(sigma); // voxel based if negative value
                radius=(int)ceil(currentSigma*3.0f);
                if(radius>0){
                    PrecisionTYPE *kernel = new PrecisionTYPE[2*radius+1];
                    PrecisionTYPE kernelSum=0;
                    for(i=-radius; i<=radius; i++){
                        kernel[radius+i]=(PrecisionTYPE)(exp( -(i*i)/(2.0*currentSigma*currentSigma)) / (currentSigma*2.506628274631));
                        // 2.506... = sqrt(2*pi)
                        kernelSum += kernel[radius+i];
                    }
                    for(i=-radius; i<=radius; i++) kernel[radius+i] /= kernelSum;
#ifndef NDEBUG
                    printf("[NiftyReg DEBUG] smoothing dim[%i] radius[%i] kernelSum[%g]\n", n, radius, kernelSum);
#endif
                    // Define the variable to increment in the 1D array
                    increment=1;
                    switch(n){
                    case 1: increment=1;break;
                    case 2: increment=image->nx;break;
                    case 3: increment=image->nx*image->ny;break;
                    case 4: increment=image->nx*image->ny*image->nz;break;
                    case 5: increment=image->nx*image->ny*image->nz*image->nt;break;
                    case 6: increment=image->nx*image->ny*image->nz*image->nt*image->nu;break;
                    case 7: increment=image->nx*image->ny*image->nz*image->nt*image->nu*image->nv;break;
                    }
                    // Loop over the different voxel
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber,image,n,increment,radius,timeImagePtr,kernel,resultValue) \
    private(index, startingIndex,x,current,value,j)
#endif
                    for(index=0;index<voxelNumber;index+=image->dim[n]){
                        for(x=0; x<image->dim[n]; x++){
                            startingIndex=index+x;

                            current = startingIndex - increment*radius;
                            value=0;
                            // Check if the central voxel is a NaN
                            if(timeImagePtr[startingIndex]==timeImagePtr[startingIndex]){
                                for(j=-radius; j<=radius; j++){
                                    if(-1<current && current<(int)voxelNumber){
                                        if(timeImagePtr[current]==timeImagePtr[current])
                                            value += (PrecisionTYPE)(timeImagePtr[current]*kernel[j+radius]);
                                    }
                                    current += increment;
                                }
                                resultValue[startingIndex]=value;
                            }
                            else{
                                resultValue[startingIndex]=timeImagePtr[startingIndex];
                            }
                        }
                    }
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(voxelNumber, timeImagePtr, resultValue) \
    private(i)
#endif
                    for(i=0; i<voxelNumber; i++) timeImagePtr[i]=(ImageTYPE)resultValue[i];
                    delete[] kernel;
                }
            }
        }
        free(resultValue);
    }
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_gaussianSmoothing(	nifti_image *image,
                            PrecisionTYPE sigma,
                            bool smoothXYZ[8])
{
    bool axisToSmooth[8];
    if(smoothXYZ==NULL){
        for(int i=0; i<8; i++) axisToSmooth[i]=true;
    }
    else{
        for(int i=0; i<8; i++) axisToSmooth[i]=smoothXYZ[i];
    }

    if(sigma==0.0) return;
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_gaussianSmoothing1<PrecisionTYPE,unsigned char>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_INT8:
        reg_gaussianSmoothing1<PrecisionTYPE,char>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_UINT16:
        reg_gaussianSmoothing1<PrecisionTYPE,unsigned short>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_INT16:
        reg_gaussianSmoothing1<PrecisionTYPE,short>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_UINT32:
        reg_gaussianSmoothing1<PrecisionTYPE,unsigned int>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_INT32:
        reg_gaussianSmoothing1<PrecisionTYPE,int>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_gaussianSmoothing1<PrecisionTYPE,float>(image, sigma, axisToSmooth);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_gaussianSmoothing1<PrecisionTYPE,double>(image, sigma, axisToSmooth);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_smoothImage\tThe image data type is not supported\n");
        exit(1);
    }
}
template void reg_gaussianSmoothing<float>(nifti_image *, float, bool[8]);
template void reg_gaussianSmoothing<double>(nifti_image *, double, bool[8]);
/* *************************************************************** */
/* *************************************************************** */
template <class PrecisionTYPE, class ImageTYPE>
void reg_downsampleImage1(nifti_image *image, int type, bool downsampleAxis[8])
{
    if(type==1){
        /* the input image is first smooth */
        reg_gaussianSmoothing<float>(image, -0.7f, downsampleAxis);
    }

    /* the values are copied */
    ImageTYPE *oldValues = (ImageTYPE *)malloc(image->nvox * image->nbyper);
    ImageTYPE *imagePtr = static_cast<ImageTYPE *>(image->data);
    memcpy(oldValues, imagePtr, image->nvox*image->nbyper);
    free(image->data);

    // Keep the previous real to voxel qform
    mat44 real2Voxel_qform;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            real2Voxel_qform.m[i][j]=image->qto_ijk.m[i][j];
        }
    }

    // Update the axis dimension
    int oldDim[4];
    for(int i=1; i<4; i++){
        oldDim[i]=image->dim[i];
        if(image->dim[i]>1 && downsampleAxis[i]==true) image->dim[i]=(int)(image->dim[i]/2.0);
        if(image->pixdim[i]>0 && downsampleAxis[i]==true) image->pixdim[i]=image->pixdim[i]*2.0f;
    }
    image->nx=image->dim[1];
    image->ny=image->dim[2];
    image->nz=image->dim[3];
    image->dx=image->pixdim[1];
    image->dy=image->pixdim[2];
    image->dz=image->pixdim[3];
    if(image->nt<1 || image->dim[4]<1) image->nt=image->dim[4]=1;
    if(image->nu<1 || image->dim[5]<1) image->nu=image->dim[5]=1;
    if(image->nv<1 || image->dim[6]<1) image->nv=image->dim[6]=1;
    if(image->nw<1 || image->dim[7]<1) image->nw=image->dim[7]=1;

    // update the qform matrix
    image->qto_xyz=nifti_quatern_to_mat44(image->quatern_b,
                                          image->quatern_c,
                                          image->quatern_d,
                                          image->qoffset_x,
                                          image->qoffset_y,
                                          image->qoffset_z,
                                          image->dx,
                                          image->dy,
                                          image->dz,
                                          image->qfac);
    image->qto_ijk = nifti_mat44_inverse(image->qto_xyz);

    // update the sform matrix
    if(downsampleAxis[1]){
        image->sto_xyz.m[0][0] *= 2.f;image->sto_xyz.m[1][0] *= 2.f;image->sto_xyz.m[2][0] *= 2.f;
    }
    if(downsampleAxis[2]){
        image->sto_xyz.m[0][1] *= 2.f;image->sto_xyz.m[1][1] *= 2.f;image->sto_xyz.m[2][1] *= 2.f;
    }
    if(downsampleAxis[3]){
        image->sto_xyz.m[0][2] *= 2.f;image->sto_xyz.m[1][2] *= 2.f;image->sto_xyz.m[2][2] *= 2.f;
    }
    float origin_sform[3]={image->sto_xyz.m[0][3], image->sto_xyz.m[1][3], image->sto_xyz.m[2][3]};
    image->sto_xyz.m[0][3]=origin_sform[0];
    image->sto_xyz.m[1][3]=origin_sform[1];
    image->sto_xyz.m[2][3]=origin_sform[2];
    image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);

    // Reallocate the image
    image->nvox=image->nx*image->ny*image->nz*image->nt*image->nu*image->nv*image->nw;
    image->data=(void *)calloc(image->nvox, image->nbyper);
    imagePtr = static_cast<ImageTYPE *>(image->data);

    PrecisionTYPE real[3], position[3], relative, xBasis[2], yBasis[2], zBasis[2], intensity;
    int previous[3];

    // qform is used for resampling
    for(int tuvw=0; tuvw<image->nt*image->nu*image->nv*image->nw; tuvw++){
        ImageTYPE *valuesPtrTUVW = &oldValues[tuvw*oldDim[1]*oldDim[2]*oldDim[3]];
        for(int z=0; z<image->nz; z++){
            for(int y=0; y<image->ny; y++){
                for(int x=0; x<image->nx; x++){
                    // Extract the voxel coordinate in mm
                    real[0]=x*image->qto_xyz.m[0][0] +
                            y*image->qto_xyz.m[0][1] +
                            z*image->qto_xyz.m[0][2] +
                            image->qto_xyz.m[0][3];
                    real[1]=x*image->qto_xyz.m[1][0] +
                            y*image->qto_xyz.m[1][1] +
                            z*image->qto_xyz.m[1][2] +
                            image->qto_xyz.m[1][3];
                    real[2]=x*image->qto_xyz.m[2][0] +
                            y*image->qto_xyz.m[2][1] +
                            z*image->qto_xyz.m[2][2] +
                            image->qto_xyz.m[2][3];
                    // Extract the position in voxel in the old image;
                    position[0]=real[0]*real2Voxel_qform.m[0][0] + real[1]*real2Voxel_qform.m[0][1] + real[2]*real2Voxel_qform.m[0][2] + real2Voxel_qform.m[0][3];
                    position[1]=real[0]*real2Voxel_qform.m[1][0] + real[1]*real2Voxel_qform.m[1][1] + real[2]*real2Voxel_qform.m[1][2] + real2Voxel_qform.m[1][3];
                    position[2]=real[0]*real2Voxel_qform.m[2][0] + real[1]*real2Voxel_qform.m[2][1] + real[2]*real2Voxel_qform.m[2][2] + real2Voxel_qform.m[2][3];
                    /* trilinear interpolation */
                    previous[0] = (int)round(position[0]);
                    previous[1] = (int)round(position[1]);
                    previous[2] = (int)round(position[2]);

                    // basis values along the x axis
                    relative=position[0]-(PrecisionTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (PrecisionTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(PrecisionTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (PrecisionTYPE)(1.0-relative);
                    yBasis[1]= relative;
                    // basis values along the z axis
                    relative=position[2]-(PrecisionTYPE)previous[2];
                    if(relative<0) relative=0.0; // rounding error correction
                    zBasis[0]= (PrecisionTYPE)(1.0-relative);
                    zBasis[1]= relative;
                    intensity=0;
                    for(short c=0; c<2; c++){
                        short Z= previous[2]+c;
                        if(-1<Z && Z<oldDim[3]){
                            ImageTYPE *zPointer = &valuesPtrTUVW[Z*oldDim[1]*oldDim[2]];
                            PrecisionTYPE yTempNewValue=0.0;
                            for(short b=0; b<2; b++){
                                short Y= previous[1]+b;
                                if(-1<Y && Y<oldDim[2]){
                                    ImageTYPE *yzPointer = &zPointer[Y*oldDim[1]];
                                    ImageTYPE *xyzPointer = &yzPointer[previous[0]];
                                    PrecisionTYPE xTempNewValue=0.0;
                                    for(short a=0; a<2; a++){
                                        if(-1<(previous[0]+a) && (previous[0]+a)<oldDim[1]){
                                            const ImageTYPE coeff = *xyzPointer;
                                            xTempNewValue +=  (PrecisionTYPE)(coeff * xBasis[a]);
                                        }
                                        xyzPointer++;
                                    }
                                    yTempNewValue += (xTempNewValue * yBasis[b]);
                                }
                            }
                            intensity += yTempNewValue * zBasis[c];
                        }
                    }
                    switch(image->datatype){
                    case NIFTI_TYPE_FLOAT32:
                        (*imagePtr)=(ImageTYPE)intensity;
                        break;
                    case NIFTI_TYPE_FLOAT64:
                        (*imagePtr)=(ImageTYPE)intensity;
                        break;
                    case NIFTI_TYPE_UINT8:
                        (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                        break;
                    case NIFTI_TYPE_UINT16:
                        (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                        break;
                    case NIFTI_TYPE_UINT32:
                        (*imagePtr)=(ImageTYPE)(intensity>0?round(intensity):0);
                        break;
                    default:
                        (*imagePtr)=(ImageTYPE)round(intensity);
                        break;
                    }
                    imagePtr++;
                }
            }
        }
    }
    free(oldValues);
}
/* *************************************************************** */
template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image, int type, bool downsampleAxis[8])
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_downsampleImage1<PrecisionTYPE,unsigned char>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT8:
        reg_downsampleImage1<PrecisionTYPE,char>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_UINT16:
        reg_downsampleImage1<PrecisionTYPE,unsigned short>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT16:
        reg_downsampleImage1<PrecisionTYPE,short>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_UINT32:
        reg_downsampleImage1<PrecisionTYPE,unsigned int>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_INT32:
        reg_downsampleImage1<PrecisionTYPE,int>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_downsampleImage1<PrecisionTYPE,float>(image, type, downsampleAxis);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_downsampleImage1<PrecisionTYPE,double>(image, type, downsampleAxis);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_downsampleImage\tThe image data type is not supported\n");
        exit(1);
    }
}
template void reg_downsampleImage<float>(nifti_image *, int, bool[8]);
template void reg_downsampleImage<double>(nifti_image *, int, bool[8]);
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binarise_image1(nifti_image *image)
{
    DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
    image->scl_inter=0.f;
    image->scl_slope=1.f;
    for(unsigned i=0; i<image->nvox; i++){
        *dataPtr = (*dataPtr)!=0?(DTYPE)1:(DTYPE)0;
        dataPtr++;
    }
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image)
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_binarise_image1<unsigned char>(image);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binarise_image1<char>(image);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binarise_image1<unsigned short>(image);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binarise_image1<short>(image);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binarise_image1<unsigned int>(image);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binarise_image1<int>(image);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binarise_image1<float>(image);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binarise_image1<double>(image);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_binarise_image\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binarise_image1(nifti_image *image, float threshold)
{
    DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
    for(unsigned i=0; i<image->nvox; i++){
        *dataPtr = (*dataPtr)<threshold?(DTYPE)0:(DTYPE)1;
        dataPtr++;
    }
}
/* *************************************************************** */
void reg_tools_binarise_image(nifti_image *image, float threshold)
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_binarise_image1<unsigned char>(image, threshold);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binarise_image1<char>(image, threshold);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binarise_image1<unsigned short>(image, threshold);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binarise_image1<short>(image, threshold);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binarise_image1<unsigned int>(image, threshold);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binarise_image1<int>(image, threshold);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binarise_image1<float>(image, threshold);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binarise_image1<double>(image, threshold);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_binarise_image\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_tools_binaryImage2int1(nifti_image *image, int *array, int &activeVoxelNumber)
{
    // Active voxel are different from -1
    activeVoxelNumber=0;
    DTYPE *dataPtr=static_cast<DTYPE *>(image->data);
    for(int i=0; i<image->nx*image->ny*image->nz; i++){
        if(*dataPtr++ != 0){
            array[i]=1;
            activeVoxelNumber++;
        }
        else{
            array[i]=-1;
        }
    }
}
/* *************************************************************** */
void reg_tools_binaryImage2int(nifti_image *image, int *array, int &activeVoxelNumber)
{
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_tools_binaryImage2int1<unsigned char>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_INT8:
        reg_tools_binaryImage2int1<char>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_UINT16:
        reg_tools_binaryImage2int1<unsigned short>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_INT16:
        reg_tools_binaryImage2int1<short>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_UINT32:
        reg_tools_binaryImage2int1<unsigned int>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_INT32:
        reg_tools_binaryImage2int1<int>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_tools_binaryImage2int1<float>(image, array, activeVoxelNumber);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_tools_binaryImage2int1<double>(image, array, activeVoxelNumber);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_binarise_image\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class ATYPE,class BTYPE>
double reg_tools_getMeanRMS2(nifti_image *imageA, nifti_image *imageB)
{
    ATYPE *imageAPtrX = static_cast<ATYPE *>(imageA->data);
    BTYPE *imageBPtrX = static_cast<BTYPE *>(imageB->data);
    ATYPE *imageAPtrY=NULL;
    BTYPE *imageBPtrY=NULL;
    ATYPE *imageAPtrZ=NULL;
    BTYPE *imageBPtrZ=NULL;
    if(imageA->dim[5]>1){
        imageAPtrY = &imageAPtrX[imageA->nx*imageA->ny*imageA->nz];
        imageBPtrY = &imageBPtrX[imageA->nx*imageA->ny*imageA->nz];
    }
    if(imageA->dim[5]>2){
        imageAPtrZ = &imageAPtrY[imageA->nx*imageA->ny*imageA->nz];
        imageBPtrZ = &imageBPtrY[imageA->nx*imageA->ny*imageA->nz];
    }
    double sum=0.0f;
    double rms;
    double diff;
    for(int i=0; i<imageA->nx*imageA->ny*imageA->nz; i++){
        diff = (double)*imageAPtrX++ - (double)*imageBPtrX++;
        rms = diff * diff;
        if(imageA->dim[5]>1){
            diff = (double)*imageAPtrY++ - (double)*imageBPtrY++;
            rms += diff * diff;
        }
        if(imageA->dim[5]>2){
            diff = (double)*imageAPtrZ++ - (double)*imageBPtrZ++;
            rms += diff * diff;
        }
        sum += sqrt(rms);
    }
    return sum/(double)(imageA->nx*imageA->ny*imageA->nz);
}
/* *************************************************************** */
template <class ATYPE>
double reg_tools_getMeanRMS1(nifti_image *imageA, nifti_image *imageB)
{
    switch(imageB->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMeanRMS2<ATYPE,unsigned char>(imageA, imageB);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMeanRMS2<ATYPE,char>(imageA, imageB);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMeanRMS2<ATYPE,unsigned short>(imageA, imageB);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMeanRMS2<ATYPE,short>(imageA, imageB);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMeanRMS2<ATYPE,unsigned int>(imageA, imageB);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMeanRMS2<ATYPE,int>(imageA, imageB);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMeanRMS2<ATYPE,float>(imageA, imageB);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMeanRMS2<ATYPE,double>(imageA, imageB);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_getMeanRMS\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
double reg_tools_getMeanRMS(nifti_image *imageA, nifti_image *imageB)
{
    switch(imageA->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMeanRMS1<unsigned char>(imageA, imageB);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMeanRMS1<char>(imageA, imageB);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMeanRMS1<unsigned short>(imageA, imageB);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMeanRMS1<short>(imageA, imageB);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMeanRMS1<unsigned int>(imageA, imageB);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMeanRMS1<int>(imageA, imageB);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMeanRMS1<float>(imageA, imageB);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMeanRMS1<double>(imageA, imageB);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_getMeanRMS\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
int reg_createImagePyramid(nifti_image *inputImage, nifti_image **pyramid, int unsigned levelNumber, int unsigned levelToPerform)
{
    // FINEST LEVEL OF REGISTRATION
    pyramid[levelToPerform-1]=nifti_copy_nim_info(inputImage);
    pyramid[levelToPerform-1]->data = (void *)calloc(pyramid[levelToPerform-1]->nvox,
                                                     pyramid[levelToPerform-1]->nbyper);
    memcpy(pyramid[levelToPerform-1]->data, inputImage->data,
           pyramid[levelToPerform-1]->nvox* pyramid[levelToPerform-1]->nbyper);
    reg_tools_changeDatatype<DTYPE>(pyramid[levelToPerform-1]);

    // Images are downsampled if appropriate
    for(unsigned int l=levelToPerform; l<levelNumber; l++){
        bool downsampleAxis[8]={false,true,true,true,false,false,false,false};
        if((pyramid[levelToPerform-1]->nx/2) < 32) downsampleAxis[1]=false;
        if((pyramid[levelToPerform-1]->ny/2) < 32) downsampleAxis[2]=false;
        if((pyramid[levelToPerform-1]->nz/2) < 32) downsampleAxis[3]=false;
        reg_downsampleImage<DTYPE>(pyramid[levelToPerform-1], 1, downsampleAxis);
    }

    // Images for each subsequent levels are allocated and downsampled if appropriate
    for(int l=levelToPerform-2; l>=0; l--){
        // Allocation of the image
        pyramid[l]=nifti_copy_nim_info(pyramid[l+1]);
        pyramid[l]->data = (void *)calloc(pyramid[l]->nvox,
                                       pyramid[l]->nbyper);
        memcpy(pyramid[l]->data, pyramid[l+1]->data,
               pyramid[l]->nvox* pyramid[l]->nbyper);

        // Downsample the image if appropriate
        bool downsampleAxis[8]={false,true,true,true,false,false,false,false};
        if((pyramid[l]->nx/2) < 32) downsampleAxis[1]=false;
        if((pyramid[l]->ny/2) < 32) downsampleAxis[2]=false;
        if((pyramid[l]->nz/2) < 32) downsampleAxis[3]=false;
        reg_downsampleImage<DTYPE>(pyramid[l], 1, downsampleAxis);
    }
    return 0;
}
template int reg_createImagePyramid<float>(nifti_image *, nifti_image **, unsigned int , unsigned int);
template int reg_createImagePyramid<double>(nifti_image *, nifti_image **, unsigned int , unsigned int);
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
int reg_createMaskPyramid(nifti_image *inputMaskImage, int **maskPyramid, int unsigned levelNumber, int unsigned levelToPerform, int *activeVoxelNumber)
{
    // FINEST LEVEL OF REGISTRATION
    nifti_image **tempMaskImagePyramid=(nifti_image **)malloc(levelToPerform*sizeof(nifti_image *));
    tempMaskImagePyramid[levelToPerform-1]=nifti_copy_nim_info(inputMaskImage);
    tempMaskImagePyramid[levelToPerform-1]->data = (void *)calloc(tempMaskImagePyramid[levelToPerform-1]->nvox,
                                                                  tempMaskImagePyramid[levelToPerform-1]->nbyper);
    memcpy(tempMaskImagePyramid[levelToPerform-1]->data, inputMaskImage->data,
           tempMaskImagePyramid[levelToPerform-1]->nvox* tempMaskImagePyramid[levelToPerform-1]->nbyper);
    reg_tools_binarise_image(tempMaskImagePyramid[levelToPerform-1]);
    reg_tools_changeDatatype<unsigned char>(tempMaskImagePyramid[levelToPerform-1]);

    // Image is downsampled if appropriate
    for(unsigned int l=levelToPerform; l<levelNumber; l++){
        bool downsampleAxis[8]={false,true,true,true,false,false,false,false};
        if((tempMaskImagePyramid[levelToPerform-1]->nx/2) < 32) downsampleAxis[1]=false;
        if((tempMaskImagePyramid[levelToPerform-1]->ny/2) < 32) downsampleAxis[2]=false;
        if((tempMaskImagePyramid[levelToPerform-1]->nz/2) < 32) downsampleAxis[3]=false;
        reg_downsampleImage<DTYPE>(tempMaskImagePyramid[levelToPerform-1], 0, downsampleAxis);
    }
    activeVoxelNumber[levelToPerform-1]=tempMaskImagePyramid[levelToPerform-1]->nx *
                                        tempMaskImagePyramid[levelToPerform-1]->ny *
                                        tempMaskImagePyramid[levelToPerform-1]->nz;
    maskPyramid[levelToPerform-1]=(int *)malloc(activeVoxelNumber[levelToPerform-1] * sizeof(int));
    reg_tools_binaryImage2int(tempMaskImagePyramid[levelToPerform-1],
                             maskPyramid[levelToPerform-1],
                             activeVoxelNumber[levelToPerform-1]);

    // Images for each subsequent levels are allocated and downsampled if appropriate
    for(int l=levelToPerform-2; l>=0; l--){
        // Allocation of the reference image
        tempMaskImagePyramid[l]=nifti_copy_nim_info(tempMaskImagePyramid[l+1]);
        tempMaskImagePyramid[l]->data = (void *)calloc(tempMaskImagePyramid[l]->nvox,
                                                       tempMaskImagePyramid[l]->nbyper);
        memcpy(tempMaskImagePyramid[l]->data, tempMaskImagePyramid[l+1]->data,
               tempMaskImagePyramid[l]->nvox* tempMaskImagePyramid[l]->nbyper);

        // Downsample the image if appropriate
        bool downsampleAxis[8]={false,true,true,true,false,false,false,false};
        if((tempMaskImagePyramid[l]->nx/2) < 32) downsampleAxis[1]=false;
        if((tempMaskImagePyramid[l]->ny/2) < 32) downsampleAxis[2]=false;
        if((tempMaskImagePyramid[l]->nz/2) < 32) downsampleAxis[3]=false;
        reg_downsampleImage<DTYPE>(tempMaskImagePyramid[l], 0, downsampleAxis);

        activeVoxelNumber[l]=tempMaskImagePyramid[l]->nx *
                             tempMaskImagePyramid[l]->ny *
                             tempMaskImagePyramid[l]->nz;
        maskPyramid[l]=(int *)malloc(activeVoxelNumber[l] * sizeof(int));
        reg_tools_binaryImage2int(tempMaskImagePyramid[l],
                                 maskPyramid[l],
                                 activeVoxelNumber[l]);
    }
    for(unsigned int l=0; l<levelToPerform; ++l)
        nifti_image_free(tempMaskImagePyramid[l]);
    free(tempMaskImagePyramid);
    return 0;
}
template int reg_createMaskPyramid<float>(nifti_image *, int **, unsigned int , unsigned int , int *);
template int reg_createMaskPyramid<double>(nifti_image *, int **, unsigned int , unsigned int , int *);
/* *************************************************************** */
/* *************************************************************** */
template <class TYPE1, class TYPE2>
int reg_tools_nanMask_image2(nifti_image *image, nifti_image *maskImage, nifti_image *resultImage)
{
    TYPE1 *imagePtr = static_cast<TYPE1 *>(image->data);
    TYPE2 *maskPtr = static_cast<TYPE2 *>(maskImage->data);
    TYPE1 *resPtr = static_cast<TYPE1 *>(resultImage->data);
    for(unsigned int i=0; i<image->nvox; ++i){
        if(*maskPtr == 0)
            *resPtr=std::numeric_limits<TYPE1>::quiet_NaN();
        else *resPtr=*imagePtr;
        maskPtr++;
        imagePtr++;
        resPtr++;
    }
    return 0;
}
/* *************************************************************** */
template <class TYPE1>
int reg_tools_nanMask_image1(nifti_image *image, nifti_image *maskImage, nifti_image *resultImage)
{
    switch(maskImage->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_nanMask_image2<TYPE1,unsigned char>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT8:
        return reg_tools_nanMask_image2<TYPE1,char>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_UINT16:
        return reg_tools_nanMask_image2<TYPE1,unsigned short>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT16:
        return reg_tools_nanMask_image2<TYPE1,short>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_UINT32:
        return reg_tools_nanMask_image2<TYPE1,unsigned int>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT32:
        return reg_tools_nanMask_image2<TYPE1,int>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_nanMask_image2<TYPE1,float>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_nanMask_image2<TYPE1,double>
                (image, maskImage, resultImage);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_nanMask_image\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
int reg_tools_nanMask_image(nifti_image *image, nifti_image *maskImage, nifti_image *resultImage)
{
    // Check dimension
    if(image->nvox != maskImage->nvox || image->nvox != resultImage->nvox){
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_nanMask_image\tInput images have different size\n");
        exit(1);
    }
    // Check output data type
    if(image->datatype != resultImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_nanMask_image\tInput and result images have different data type\n");
        exit(1);
    }
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_nanMask_image1<unsigned char>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT8:
        return reg_tools_nanMask_image1<char>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_UINT16:
        return reg_tools_nanMask_image1<unsigned short>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT16:
        return reg_tools_nanMask_image1<short>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_UINT32:
        return reg_tools_nanMask_image1<unsigned int>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_INT32:
        return reg_tools_nanMask_image1<int>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_nanMask_image1<float>
                (image, maskImage, resultImage);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_nanMask_image1<double>
                (image, maskImage, resultImage);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_nanMask_image\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
float reg_tools_getMinValue1(nifti_image *image)
{
    // Create a pointer to the image data
    DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
    // Set a variable to store the minimal value
    float minValue=std::numeric_limits<DTYPE>::max();
    // Loop over all voxel to find the lowest value
    for(size_t i=0;i<image->nvox;++i){
        DTYPE currentVal = imgPtr[i] * image->scl_slope + image->scl_inter;
        minValue=currentVal<minValue?currentVal:minValue;
    }
    // The lowest value is returned
    return minValue;
}
/* *************************************************************** */
float reg_tools_getMinValue(nifti_image *image)
{
    // Check the image data type
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMinValue1<unsigned char>(image);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMinValue1<char>(image);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMinValue1<unsigned short>(image);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMinValue1<short>(image);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMinValue1<unsigned int>(image);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMinValue1<int>(image);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMinValue1<float>(image);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMinValue1<double>(image);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_getMinValue\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
float reg_tools_getMaxValue1(nifti_image *image)
{
    // Create a pointer to the image data
    DTYPE *imgPtr = static_cast<DTYPE *>(image->data);
    // Set a variable to store the maximal value
    float maxValue=-std::numeric_limits<DTYPE>::max();
    // Loop over all voxel to find the lowest value
    for(size_t i=0;i<image->nvox;++i){
        DTYPE currentVal = imgPtr[i] * image->scl_slope + image->scl_inter;
        maxValue=currentVal>maxValue?currentVal:maxValue;
    }
    // The lowest value is returned
    return maxValue;
}
/* *************************************************************** */
float reg_tools_getMaxValue(nifti_image *image)
{
    // Check the image data type
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        return reg_tools_getMaxValue1<unsigned char>(image);
    case NIFTI_TYPE_INT8:
        return reg_tools_getMaxValue1<char>(image);
    case NIFTI_TYPE_UINT16:
        return reg_tools_getMaxValue1<unsigned short>(image);
    case NIFTI_TYPE_INT16:
        return reg_tools_getMaxValue1<short>(image);
    case NIFTI_TYPE_UINT32:
        return reg_tools_getMaxValue1<unsigned int>(image);
    case NIFTI_TYPE_INT32:
        return reg_tools_getMaxValue1<int>(image);
    case NIFTI_TYPE_FLOAT32:
        return reg_tools_getMaxValue1<float>(image);
    case NIFTI_TYPE_FLOAT64:
        return reg_tools_getMaxValue1<double>(image);
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_tools_getMaxValue\tThe image data type is not supported\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_flippAxis_type(int nx,
                        int ny,
                        int nz,
                        int nt,
                        int nu,
                        int nv,
                        int nw,
                        void *inputArray,
                        void *outputArray,
                        std::string cmd
                        )
{
    // Allocate the outputArray if it is not allocated yet
    if(outputArray==NULL)
        outputArray=(void *)malloc(nx*ny*nz*nt*nu*nv*nw*sizeof(DTYPE));

    // Parse the cmd to check which axis have to be flipped
    char *axisName=(char *)"x\0y\0z\0t\0u\0v\0w\0";
    int increment[7]={1,1,1,1,1,1,1};
    int start[7]={0,0,0,0,0,0,0};
    int end[7]={nx,ny,nz,nt,nu,nv,nw};
    for(int i=0;i<7;++i){
        if(cmd.find(axisName[i*2])!=std::string::npos){
            increment[i]=-1;
            start[i]=end[i]-1;
        }
    }

    // Define the reading and writting pointers
    DTYPE *inputPtr=static_cast<DTYPE *>(inputArray);
    DTYPE *outputPtr=static_cast<DTYPE *>(outputArray);

    // Copy the data and flipp axis if required
    for(int w=0, w2=start[6];w<nw;++w, w2+=increment[6]){
        size_t index_w=w2*nx*ny*nz*nt*nu*nv;
        for(int v=0, v2=start[5];v<nv;++v, v2+=increment[5]){
            size_t index_v=index_w + v2*nx*ny*nz*nt*nu;
            for(int u=0, u2=start[4];u<nu;++u, u2+=increment[4]){
                size_t index_u=index_v + u2*nx*ny*nz*nt;
                for(int t=0, t2=start[3];t<nt;++t, t2+=increment[3]){
                    size_t index_t=index_u + t2*nx*ny*nz;
                    for(int z=0, z2=start[2];z<nz;++z, z2+=increment[2]){
                        size_t index_z=index_t + z2*nx*ny;
                        for(int y=0, y2=start[1];y<ny;++y, y2+=increment[1]){
                            size_t index_y=index_z + y2*nx;
                            for(int x=0, x2=start[0];x<nx;++x, x2+=increment[0]){
                                size_t index=index_y + x2;
                                *outputPtr++ = inputPtr[index];
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}
/* *************************************************************** */
void reg_flippAxis(nifti_image *image,
                   void *outputArray,
                   std::string cmd
                   )
{
    // Check the image data type
    switch(image->datatype){
    case NIFTI_TYPE_UINT8:
        reg_flippAxis_type<unsigned char>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT8:
        reg_flippAxis_type<char>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_UINT16:
        reg_flippAxis_type<unsigned short>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT16:
        reg_flippAxis_type<short>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_UINT32:
        reg_flippAxis_type<unsigned int>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_INT32:
        reg_flippAxis_type<int>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_flippAxis_type<float>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_flippAxis_type<double>
                (image->nx, image->ny, image->nz, image->nt, image->nu, image->nv, image->nw,
                 image->data, outputArray, cmd);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] reg_flippAxis\tThe image data type is not supported\n");
        exit(1);
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */


#endif
