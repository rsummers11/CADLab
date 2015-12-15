/*
 *  _reg_resampling.cpp
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_CPP
#define _REG_RESAMPLING_CPP

#include "_reg_resampling.h"

// No round() function available in windows.
#ifdef _WINDOWS
template<class DTYPE>
int round(DTYPE x)
{
    return static_cast<int>(x > 0.0 ? x + 0.5 : x - 0.5);
}
#endif

/* *************************************************************** */
template <class FieldTYPE>
void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis)
{
    if(ratio<0.0) ratio=0.0; //rounding error
    FieldTYPE FF= ratio*ratio;
    basis[0] = (FieldTYPE)((ratio * ((2.0-ratio)*ratio - 1.0))/2.0);
    basis[1] = (FieldTYPE)((FF * (3.0*ratio-5.0) + 2.0)/2.0);
    basis[2] = (FieldTYPE)((ratio * ((4.0-3.0*ratio)*ratio + 1.0))/2.0);
    basis[3] = (FieldTYPE)((ratio-1.0) * FF/2.0);
}
/* *************************************************************** */
template <class FieldTYPE>
void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis, FieldTYPE *derivative)
{
    interpolantCubicSpline<FieldTYPE>(ratio,basis);
    if(ratio<0.0) ratio=0.0; //rounding error
    FieldTYPE FF= ratio*ratio;
    derivative[0] = (FieldTYPE)((4.0*ratio - 3.0*FF - 1.0)/2.0);
    derivative[1] = (FieldTYPE)((9.0*ratio - 10.0) * ratio/2.0);
    derivative[2] = (FieldTYPE)((8.0*ratio - 9.0*FF + 1)/2.0);
    derivative[3] = (FieldTYPE)((3.0*ratio - 2.0) * ratio/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage3D(nifti_image *sourceImage,
                                      nifti_image *deformationField,
                                      nifti_image *resultImage,
                                      int *mask)
{
    // The spline decomposition assumes a background set to 0 the bgValue variable is thus not use here

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int resultVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[resultVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[resultVoxelNumber];


    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultImage->nt*resultImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D Cubic spline resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*resultVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], zBasis[4], relative;
        int a, b, c, Y, Z, previous[3], index;
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, resultVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, sourceImage)
#endif // _OPENMP
        for(index=0;index<resultVoxelNumber; index++){

            intensity=(FieldTYPE)(0.0);

            if((maskPtr[index])>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = static_cast<int>(floor(position[0]));
                previous[1] = static_cast<int>(floor(position[1]));
                previous[2] = static_cast<int>(floor(position[2]));

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                interpolantCubicSpline<FieldTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                interpolantCubicSpline<FieldTYPE>(relative, yBasis);
                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                interpolantCubicSpline<FieldTYPE>(relative, zBasis);

                --previous[0];--previous[1];--previous[2];

                for(c=0; c<4; c++){
                    Z= previous[2]+c;
                    if(-1<Z && Z<sourceImage->nz){
                        zPointer = &sourceIntensity[Z*sourceImage->nx*sourceImage->ny];
                        yTempNewValue=0.0;
                        for(b=0; b<4; b++){
                            Y= previous[1]+b;
                            yzPointer = &zPointer[Y*sourceImage->nx];
                            if(-1<Y && Y<sourceImage->ny){
                                xyzPointer = &yzPointer[previous[0]];
                                xTempNewValue=0.0;
                                for(a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                        xTempNewValue +=  (FieldTYPE)*xyzPointer * xBasis[a];
                                    }
                                    xyzPointer++;
                                }
                                yTempNewValue += (xTempNewValue * yBasis[b]);
                            }
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
                }
            }

            switch(sourceImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)round(intensity);
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void CubicSplineResampleSourceImage2D(  nifti_image *sourceImage,
                                      nifti_image *deformationField,
                                      nifti_image *resultImage,
                                      int *mask)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int targetVoxelNumber = resultImage->nx*resultImage->ny;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=sourceImage->sto_ijk;
    else sourceIJKMatrix=sourceImage->qto_ijk;

    for(int t=0; t<resultImage->nt*resultImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D Cubic spline resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[4], yBasis[4], relative;
        int a, b, Y, previous[2], index;
        SourceTYPE *yPointer, *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], position[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, relative, \
    a, b, Y, yPointer, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, sourceImage)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            intensity=0.0;

            if((maskPtr[index])>-1){

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix.m[0][0] + world[1]*sourceIJKMatrix.m[0][1] +
                        sourceIJKMatrix.m[0][3];
                position[1] = world[0]*sourceIJKMatrix.m[1][0] + world[1]*sourceIJKMatrix.m[1][1] +
                        sourceIJKMatrix.m[1][3];

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                interpolantCubicSpline<FieldTYPE>(relative, xBasis);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                interpolantCubicSpline<FieldTYPE>(relative, yBasis);

                previous[0]--;previous[1]--;

                for(b=0; b<4; b++){
                    Y= previous[1]+b;
                    yPointer = &sourceIntensity[Y*sourceImage->nx];
                    if(-1<Y && Y<sourceImage->ny){
                        xyPointer = &yPointer[previous[0]];
                        xTempNewValue=0.0;
                        for(a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                xTempNewValue +=  (FieldTYPE)*xyPointer * xBasis[a];
                            }
                            xyPointer++;
                        }
                        intensity += (xTempNewValue * yBasis[b]);
                    }
                }
            }

            switch(sourceImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)round(intensity);
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void LinearResampleSourceImage(  nifti_image *sourceImage,
                               nifti_image *deformationField,
                               nifti_image *resultImage,
                               int *mask,
                               FieldTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];
    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    for(int t=0; t<resultImage->nt*resultImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], zBasis[2], relative;
        int a, b, c, Y, Z, previous[3], index;
        SourceTYPE *zPointer, *xyzPointer;
        FieldTYPE xTempNewValue, yTempNewValue, intensity, world[3], position[3];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, sourceImage, bgValue)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            intensity=0.0;

            if(maskPtr[index]>-1){

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                if( position[0]>=0.f && position[0]<(FieldTYPE)(sourceImage->nx-1) &&
                    position[1]>=0.f && position[1]<(FieldTYPE)(sourceImage->ny-1) &&
                    position[2]>=0.f && position[2]<(FieldTYPE)(sourceImage->nz-1) ){

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    previous[2] = (int)position[2];
                    // basis values along the x axis
                    relative=position[0]-(FieldTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (FieldTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(FieldTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (FieldTYPE)(1.0-relative);
                    yBasis[1]= relative;
                    // basis values along the z axis
                    relative=position[2]-(FieldTYPE)previous[2];
                    if(relative<0) relative=0.0; // rounding error correction
                    zBasis[0]= (FieldTYPE)(1.0-relative);
                    zBasis[1]= relative;

                    for(c=0; c<2; c++){
                        Z= previous[2]+c;
                        zPointer = &sourceIntensity[Z*sourceImage->nx*sourceImage->ny];
                        yTempNewValue=0.0;
                        for(b=0; b<2; b++){
                            Y= previous[1]+b;
                            xyzPointer = &zPointer[Y*sourceImage->nx+previous[0]];
                            xTempNewValue=0.0;
                            for(a=0; a<2; a++){
                                xTempNewValue +=  (FieldTYPE)*xyzPointer * xBasis[a];
                                xyzPointer++;
                            }
                            yTempNewValue += (xTempNewValue * yBasis[b]);
                        }
                        intensity += yTempNewValue * zBasis[c];
                    }
                }
                else intensity = bgValue;
            }

            switch(sourceImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)round(intensity);
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void LinearResampleSourceImage2D(nifti_image *sourceImage,
                                 nifti_image *deformationField,
                                 nifti_image *resultImage,
                                 int *mask,
                                 FieldTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    int targetVoxelNumber = resultImage->nx*resultImage->ny;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    for(int t=0; t<resultImage->nt*resultImage->nu;t++){

#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE xBasis[2], yBasis[2], relative;
        int a, b, Y, previous[3], index;
        SourceTYPE *xyPointer;
        FieldTYPE xTempNewValue, intensity, world[2], voxel[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, voxel, previous, xBasis, yBasis, relative, \
    a, b, Y, xyPointer, xTempNewValue) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, sourceImage, bgValue)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            intensity=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];

                /* real -> voxel; source space */
                voxel[0] = world[0]*sourceIJKMatrix->m[0][0] + world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                voxel[1] = world[0]*sourceIJKMatrix->m[1][0] + world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                if( voxel[0]>=0.0f && voxel[0]<(FieldTYPE)(sourceImage->nx-1) &&
                        voxel[1]>=0.0f && voxel[1]<(FieldTYPE)(sourceImage->ny-1)) {

                    previous[0] = (int)voxel[0];
                    previous[1] = (int)voxel[1];
                    // basis values along the x axis
                    relative=voxel[0]-(FieldTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (FieldTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=voxel[1]-(FieldTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (FieldTYPE)(1.0-relative);
                    yBasis[1]= relative;

                    for(b=0; b<2; b++){
                        Y= previous[1]+b;
                        xyPointer = &sourceIntensity[Y*sourceImage->nx+previous[0]];
                        xTempNewValue=0.0;
                        for(a=0; a<2; a++){
                            xTempNewValue +=  (FieldTYPE)*xyPointer * xBasis[a];
                            xyPointer++;
                        }
                        intensity += (xTempNewValue * yBasis[b]);
                    }
                }
                else intensity = bgValue;
            }

            switch(sourceImage->datatype){
            case NIFTI_TYPE_FLOAT32:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_FLOAT64:
                resultIntensity[index]=(SourceTYPE)intensity;
                break;
            case NIFTI_TYPE_UINT8:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                resultIntensity[index]=(SourceTYPE)(intensity>0?round(intensity):0);
                break;
            default:
                resultIntensity[index]=(SourceTYPE)round(intensity);
                break;
            }
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage(nifti_image *sourceImage,
                                        nifti_image *deformationField,
                                        nifti_image *resultImage,
                                        int *mask,
                                        FieldTYPE bgValue)
{
    // The resampling scheme is applied along each time
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int targetVoxelNumber = resultImage->nx*resultImage->ny*resultImage->nz;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    for(int t=0; t<resultImage->nt*resultImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D nearest neighbor resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        SourceTYPE intensity;
        FieldTYPE world[3];
        FieldTYPE position[3];
        int previous[3];
        int index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, sourceImage, bgValue)
#endif // _OPENMP
        for(index=0; index<targetVoxelNumber; index++){

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = (int)round(position[0]);
                previous[1] = (int)round(position[1]);
                previous[2] = (int)round(position[2]);

                if( -1<previous[2] && previous[2]<sourceImage->nz &&
                        -1<previous[1] && previous[1]<sourceImage->ny &&
                        -1<previous[0] && previous[0]<sourceImage->nx){
                    intensity = sourceIntensity[(previous[2]*sourceImage->ny+previous[1]) *
                            sourceImage->nx+previous[0]];
                    resultIntensity[index]=intensity;
                }
                else resultIntensity[index]=(SourceTYPE)bgValue;
            }
            else resultIntensity[index]=(SourceTYPE)bgValue;
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class FieldTYPE>
void NearestNeighborResampleSourceImage2D(nifti_image *sourceImage,
                                          nifti_image *deformationField,
                                          nifti_image *resultImage,
                                          int *mask,
                                          FieldTYPE bgValue)
{
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    SourceTYPE *resultIntensityPtr = static_cast<SourceTYPE *>(resultImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    int targetVoxelNumber = resultImage->nx*resultImage->ny;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    for(int t=0; t<resultImage->nt*resultImage->nu;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D nearest neighbor resampling of volume number %i\n",t);
#endif

        SourceTYPE *resultIntensity = &resultIntensityPtr[t*targetVoxelNumber];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        SourceTYPE intensity;
        FieldTYPE world[2];
        FieldTYPE position[2];
        int previous[2];
        int index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, intensity, world, position, previous) \
    shared(sourceIntensity, resultIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, sourceImage, bgValue)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            if((*maskPtr++)>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix->m[0][0] + world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                position[1] = world[0]*sourceIJKMatrix->m[1][0] + world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                previous[0] = (int)round(position[0]);
                previous[1] = (int)round(position[1]);

                if( -1<previous[1] && previous[1]<sourceImage->ny &&
                        -1<previous[0] && previous[0]<sourceImage->nx){
                    intensity = sourceIntensity[previous[1]*sourceImage->nx+previous[0]];
                    resultIntensity[index]=intensity;
                }
                else resultIntensity[index]=(SourceTYPE)bgValue;
            }
            else resultIntensity[index]=(SourceTYPE)bgValue;
        }
    }
}
/* *************************************************************** */

/** This function resample a source image into the referential
 * of a target image by applying an affine transformation and
 * a deformation field. The affine transformation has to be in
 * real coordinate and the deformation field is in mm in the space
 * of the target image.
 * interp can be either 0, 1 or 3 meaning nearest neighbor, linear
 * or cubic spline interpolation.
 * every voxel which is not fully in the source image takes the
 * background value.
 */
template <class FieldTYPE, class SourceTYPE>
void reg_resampleSourceImage2(	nifti_image *targetImage,
                              nifti_image *sourceImage,
                              nifti_image *resultImage,
                              nifti_image *deformationFieldImage,
                              int *mask,
                              int interp,
                              FieldTYPE bgValue
                              )
{
    /* The deformation field contains the position in the real world */
    if(interp==3){
        if(targetImage->nz>1){
            CubicSplineResampleSourceImage3D<SourceTYPE,FieldTYPE>( sourceImage,
                                                                   deformationFieldImage,
                                                                   resultImage,
                                                                   mask);
        }
        else
        {
            CubicSplineResampleSourceImage2D<SourceTYPE,FieldTYPE>(  sourceImage,
                                                                   deformationFieldImage,
                                                                   resultImage,
                                                                   mask);
        }
    }
    else if(interp==0){ // Nearest neighbor interpolation
        if(targetImage->nz>1){
            NearestNeighborResampleSourceImage<SourceTYPE, FieldTYPE>( sourceImage,
                                                                      deformationFieldImage,
                                                                      resultImage,
                                                                      mask,
                                                                      bgValue);
        }
        else
        {
            NearestNeighborResampleSourceImage2D<SourceTYPE, FieldTYPE>( sourceImage,
                                                                        deformationFieldImage,
                                                                        resultImage,
                                                                        mask,
                                                                        bgValue);
        }

    }
    else{ // trilinear interpolation [ by default ]
        if(targetImage->nz>1){
            LinearResampleSourceImage<SourceTYPE, FieldTYPE>( sourceImage,
                                                             deformationFieldImage,
                                                             resultImage,
                                                             mask,
                                                             bgValue);
        }
        else{
            LinearResampleSourceImage2D<SourceTYPE, FieldTYPE>( sourceImage,
                                                               deformationFieldImage,
                                                               resultImage,
                                                               mask,
                                                               bgValue);
        }
    }
}

/* *************************************************************** */
void reg_resampleSourceImage(	nifti_image *targetImage,
                             nifti_image *sourceImage,
                             nifti_image *resultImage,
                             nifti_image *deformationField,
                             int *mask,
                             int interp,
                             float bgValue)
{
    if(sourceImage->datatype != resultImage->datatype){
        printf("[NiftyReg ERROR] reg_resampleSourceImage\tSource and result image should have the same data type\n");
        printf("[NiftyReg ERROR] reg_resampleSourceImage\tNothing has been done\n");
        exit(1);
    }

    if(sourceImage->nt != resultImage->nt){
        printf("[NiftyReg ERROR] reg_resampleSourceImage\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_resampleSourceImage\tNothing has been done\n");
        exit(1);
    }

    // a mask array is created if no mask is specified
    bool MrPropreRules = false;
    if(mask==NULL){
        // voxels in the background are set to -1 so 0 will do the job here
        mask=(int *)calloc(targetImage->nx*targetImage->ny*targetImage->nz,sizeof(int));
        MrPropreRules = true;
    }

    switch ( deformationField->datatype ){
    case NIFTI_TYPE_FLOAT32:
        switch ( sourceImage->datatype ){
        case NIFTI_TYPE_UINT8:
            reg_resampleSourceImage2<float,unsigned char>(	targetImage,
                                                          sourceImage,
                                                          resultImage,
                                                          deformationField,
                                                          mask,
                                                          interp,
                                                          bgValue);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleSourceImage2<float,char>(	targetImage,
                                                 sourceImage,
                                                 resultImage,
                                                 deformationField,
                                                 mask,
                                                 interp,
                                                 bgValue);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleSourceImage2<float,unsigned short>(	targetImage,
                                                           sourceImage,
                                                           resultImage,
                                                           deformationField,
                                                           mask,
                                                           interp,
                                                           bgValue);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleSourceImage2<float,short>(	targetImage,
                                                  sourceImage,
                                                  resultImage,
                                                  deformationField,
                                                  mask,
                                                  interp,
                                                  bgValue);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleSourceImage2<float,unsigned int>(	targetImage,
                                                         sourceImage,
                                                         resultImage,
                                                         deformationField,
                                                         mask,
                                                         interp,
                                                         bgValue);
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleSourceImage2<float,int>(	targetImage,
                                                sourceImage,
                                                resultImage,
                                                deformationField,
                                                mask,
                                                interp,
                                                bgValue);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleSourceImage2<float,float>(	targetImage,
                                                  sourceImage,
                                                  resultImage,
                                                  deformationField,
                                                  mask,
                                                  interp,
                                                  bgValue);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleSourceImage2<float,double>(	targetImage,
                                                   sourceImage,
                                                   resultImage,
                                                   deformationField,
                                                   mask,
                                                   interp,
                                                   bgValue);
            break;
        default:
            printf("Source pixel type unsupported.");
            break;
        }
        break;
    case NIFTI_TYPE_FLOAT64:
        switch ( sourceImage->datatype ){
        case NIFTI_TYPE_UINT8:
            reg_resampleSourceImage2<double,unsigned char>(	targetImage,
                                                           sourceImage,
                                                           resultImage,
                                                           deformationField,
                                                           mask,
                                                           interp,
                                                           bgValue);
            break;
        case NIFTI_TYPE_INT8:
            reg_resampleSourceImage2<double,char>(	targetImage,
                                                  sourceImage,
                                                  resultImage,
                                                  deformationField,
                                                  mask,
                                                  interp,
                                                  bgValue);
            break;
        case NIFTI_TYPE_UINT16:
            reg_resampleSourceImage2<double,unsigned short>(	targetImage,
                                                            sourceImage,
                                                            resultImage,
                                                            deformationField,
                                                            mask,
                                                            interp,
                                                            bgValue);
            break;
        case NIFTI_TYPE_INT16:
            reg_resampleSourceImage2<double,short>(	targetImage,
                                                   sourceImage,
                                                   resultImage,
                                                   deformationField,
                                                   mask,
                                                   interp,
                                                   bgValue);
            break;
        case NIFTI_TYPE_UINT32:
            reg_resampleSourceImage2<double,unsigned int>(	targetImage,
                                                          sourceImage,
                                                          resultImage,
                                                          deformationField,
                                                          mask,
                                                          interp,
                                                          bgValue);
            break;
        case NIFTI_TYPE_INT32:
            reg_resampleSourceImage2<double,int>(	targetImage,
                                                 sourceImage,
                                                 resultImage,
                                                 deformationField,
                                                 mask,
                                                 interp,
                                                 bgValue);
            break;
        case NIFTI_TYPE_FLOAT32:
            reg_resampleSourceImage2<double,float>(	targetImage,
                                                   sourceImage,
                                                   resultImage,
                                                   deformationField,
                                                   mask,
                                                   interp,
                                                   bgValue);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleSourceImage2<double,double>(	targetImage,
                                                    sourceImage,
                                                    resultImage,
                                                    deformationField,
                                                    mask,
                                                    interp,
                                                    bgValue);
            break;
        default:
            printf("Source pixel type unsupported.");
            break;
        }
        break;
    default:
        printf("Deformation field pixel type unsupported.");
        break;
    }
    if(MrPropreRules==true){ free(mask);mask=NULL;}
}
/* *************************************************************** */
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage(  nifti_image *sourceImage,
                                  nifti_image *deformationField,
                                  nifti_image *resultGradientImage,
                                  int *mask)
{
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;
    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D linear gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, c, Z, b, Y, a;
        FieldTYPE position[3], xBasis[2], yBasis[2], zBasis[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE relative, world[3], grad[3], coeff;
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, relative, grad, coeff, \
    a, b, c, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, sourceImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;
            grad[2]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                if( position[0]>=0.0f && position[0]<(FieldTYPE)(sourceImage->nx-1) &&
                        position[1]>=0.0f && position[1]<(FieldTYPE)(sourceImage->ny-1) &&
                        position[2]>=0.0f && position[2]<(FieldTYPE)(sourceImage->nz-1) ){

                    previous[0] = (int)position[0];
                    previous[1] = (int)position[1];
                    previous[2] = (int)position[2];
                    // basis values along the x axis
                    relative=position[0]-(FieldTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (FieldTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=position[1]-(FieldTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (FieldTYPE)(1.0-relative);
                    yBasis[1]= relative;
                    // basis values along the z axis
                    relative=position[2]-(FieldTYPE)previous[2];
                    if(relative<0) relative=0.0; // rounding error correction
                    zBasis[0]= (FieldTYPE)(1.0-relative);
                    zBasis[1]= relative;

                    for(c=0; c<2; c++){
                        Z= previous[2]+c;
                        zPointer = &sourceIntensity[Z*sourceImage->nx*sourceImage->ny];
                        xxTempNewValue=0.0;
                        yyTempNewValue=0.0;
                        zzTempNewValue=0.0;
                        for(b=0; b<2; b++){
                            Y= previous[1]+b;
                            xyzPointer = &zPointer[Y*sourceImage->nx+previous[0]];
                            xTempNewValue=0.0;
                            yTempNewValue=0.0;
                            for(a=0; a<2; a++){
                                coeff = (FieldTYPE)*xyzPointer;
                                xTempNewValue +=  coeff * deriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                                xyzPointer++;
                            }
                            xxTempNewValue += xTempNewValue * yBasis[b];
                            yyTempNewValue += yTempNewValue * deriv[b];
                            zzTempNewValue += yTempNewValue * yBasis[b];
                        }
                        grad[0] += xxTempNewValue * zBasis[c];
                        grad[1] += yyTempNewValue * zBasis[c];
                        grad[2] += zzTempNewValue * deriv[c];
                    }
                }
            }

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
            resultGradientPtrZ[index] = (GradientTYPE)grad[2];
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearGradientResultImage2D(	nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultGradientImage,
                                    int *mask)
{
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=sourceImage->sto_ijk;
    else sourceIJKMatrix=sourceImage->qto_ijk;

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D linear gradient computation of volume number %i\n",t);
#endif
        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        FieldTYPE voxel[3], xBasis[2], yBasis[2], relative, world[2], grad[2];
        FieldTYPE deriv[2];deriv[0]=-1;deriv[1]=1;
        FieldTYPE coeff, xTempNewValue, yTempNewValue;
        int previous[3], index, b, Y, a;
        SourceTYPE *xyPointer;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, voxel, previous, xBasis, yBasis, relative, grad, coeff, \
    a, b, Y,xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, sourceImage, resultGradientPtrX, resultGradientPtrY)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];

                /* real -> voxel; source space */
                voxel[0] = world[0]*sourceIJKMatrix.m[0][0] + world[1]*sourceIJKMatrix.m[0][1] +
                        sourceIJKMatrix.m[0][3];
                voxel[1] = world[0]*sourceIJKMatrix.m[1][0] + world[1]*sourceIJKMatrix.m[1][1] +
                        sourceIJKMatrix.m[1][3];

                if( voxel[0]>=0.0f && voxel[0]<(FieldTYPE)(sourceImage->nx-1) &&
                        voxel[1]>=0.0f && voxel[1]<(FieldTYPE)(sourceImage->ny-1) ){

                    previous[0] = (int)voxel[0];
                    previous[1] = (int)voxel[1];
                    // basis values along the x axis
                    relative=voxel[0]-(FieldTYPE)previous[0];
                    if(relative<0) relative=0.0; // rounding error correction
                    xBasis[0]= (FieldTYPE)(1.0-relative);
                    xBasis[1]= relative;
                    // basis values along the y axis
                    relative=voxel[1]-(FieldTYPE)previous[1];
                    if(relative<0) relative=0.0; // rounding error correction
                    yBasis[0]= (FieldTYPE)(1.0-relative);
                    yBasis[1]= relative;

                    for(b=0; b<2; b++){
                        Y= previous[1]+b;
                        xyPointer = &sourceIntensity[Y*sourceImage->nx+previous[0]];
                        xTempNewValue=0.0;
                        yTempNewValue=0.0;
                        for(a=0; a<2; a++){
                            coeff = (FieldTYPE)*xyPointer;
                            xTempNewValue +=  coeff* deriv[a];
                            yTempNewValue +=  coeff * xBasis[a];
                            xyPointer++;
                        }
                        grad[0] += xTempNewValue * yBasis[b];
                        grad[1] += yTempNewValue * deriv[b];
                    }
                }
            }

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage(nifti_image *sourceImage,
                                    nifti_image *deformationField,
                                    nifti_image *resultGradientImage,
                                    int *mask)
{
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny*resultGradientImage->nz;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny*sourceImage->nz;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);


    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 3D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        GradientTYPE *resultGradientPtrZ = &resultGradientPtrY[gradientOffSet];

        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, c, Z, b, Y, a; bool bg;
        FieldTYPE xBasis[4], yBasis[4], zBasis[4], xDeriv[4], yDeriv[4], zDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[3];
        FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
        SourceTYPE *zPointer, *yzPointer, *xyzPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, zBasis, xDeriv, yDeriv, zDeriv, relative, grad, coeff, bg, \
    a, b, c, Y, Z, zPointer, yzPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, maskPtr, \
    sourceIJKMatrix, sourceImage, resultGradientPtrX, resultGradientPtrY, resultGradientPtrZ)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[0]=0.0;
            grad[1]=0.0;
            grad[2]=0.0;

            if((*maskPtr++)>-1){

                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];
                world[2]=(FieldTYPE) deformationFieldPtrZ[index];

                /* real -> voxel; source space */
                reg_mat44_mul(sourceIJKMatrix, world, position);

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);
                previous[2] = (int)floor(position[2]);

                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                interpolantCubicSpline<FieldTYPE>(relative, xBasis, xDeriv);

                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                interpolantCubicSpline<FieldTYPE>(relative, yBasis, yDeriv);

                // basis values along the z axis
                relative=position[2]-(FieldTYPE)previous[2];
                interpolantCubicSpline<FieldTYPE>(relative, zBasis, zDeriv);

                previous[0]--;previous[1]--;previous[2]--;

                bg=false;
                for(c=0; c<4; c++){
                    Z= previous[2]+c;
                    if(-1<Z && Z<sourceImage->nz){
                        zPointer = &sourceIntensity[Z*sourceImage->nx*sourceImage->ny];
                        xxTempNewValue=0.0;
                        yyTempNewValue=0.0;
                        zzTempNewValue=0.0;
                        for(b=0; b<4; b++){
                            Y= previous[1]+b;
                            yzPointer = &zPointer[Y*sourceImage->nx];
                            if(-1<Y && Y<sourceImage->ny){
                                xyzPointer = &yzPointer[previous[0]];
                                xTempNewValue=0.0;
                                yTempNewValue=0.0;
                                for(a=0; a<4; a++){
                                    if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                        coeff = (FieldTYPE)*xyzPointer;
                                        xTempNewValue +=  coeff * xDeriv[a];
                                        yTempNewValue +=  coeff * xBasis[a];
                                    }
                                    else bg=true;
                                    xyzPointer++;
                                }
                                xxTempNewValue += (xTempNewValue * yBasis[b]);
                                yyTempNewValue += (yTempNewValue * yDeriv[b]);
                                zzTempNewValue += (yTempNewValue * yBasis[b]);
                            }
                            else bg=true;
                        }
                        grad[0] += xxTempNewValue * zBasis[c];
                        grad[1] += yyTempNewValue * zBasis[c];
                        grad[2] += zzTempNewValue * zDeriv[c];
                    }
                    else bg=true;
                }

                if(bg==true){
                    grad[0]=0.0;
                    grad[0]=0.0;
                    grad[0]=0.0;
                }
            }

            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
            resultGradientPtrZ[index] = (GradientTYPE)grad[2];
        }
    }
}
/* *************************************************************** */
template<class SourceTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineGradientResultImage2D(nifti_image *sourceImage,
                                      nifti_image *deformationField,
                                      nifti_image *resultGradientImage,
                                      int *mask)
{
    int targetVoxelNumber = resultGradientImage->nx*resultGradientImage->ny;
    int sourceVoxelNumber = sourceImage->nx*sourceImage->ny;
    unsigned int gradientOffSet = targetVoxelNumber*resultGradientImage->nt;

    SourceTYPE *sourceIntensityPtr = static_cast<SourceTYPE *>(sourceImage->data);
    GradientTYPE *resultGradientImagePtr = static_cast<GradientTYPE *>(resultGradientImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[targetVoxelNumber];

    int *maskPtr = &mask[0];

    mat44 *sourceIJKMatrix;
    if(sourceImage->sform_code>0)
        sourceIJKMatrix=&(sourceImage->sto_ijk);
    else sourceIJKMatrix=&(sourceImage->qto_ijk);

    // Iteration over the different volume along the 4th axis
    for(int t=0; t<resultGradientImage->nt;t++){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] 2D cubic spline gradient computation of volume number %i\n",t);
#endif

        GradientTYPE *resultGradientPtrX = &resultGradientImagePtr[targetVoxelNumber*t];
        GradientTYPE *resultGradientPtrY = &resultGradientPtrX[gradientOffSet];
        SourceTYPE *sourceIntensity = &sourceIntensityPtr[t*sourceVoxelNumber];

        int previous[3], index, b, Y, a; bool bg;
        FieldTYPE xBasis[4], yBasis[4], xDeriv[4], yDeriv[4];
        FieldTYPE coeff, position[3], relative, world[3], grad[3];
        FieldTYPE xTempNewValue, yTempNewValue;
        SourceTYPE *yPointer, *xyPointer;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(index, world, position, previous, xBasis, yBasis, xDeriv, yDeriv, relative, grad, coeff, bg, \
    a, b, Y, yPointer, xyPointer, xTempNewValue, yTempNewValue) \
    shared(sourceIntensity, targetVoxelNumber, sourceVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, maskPtr, \
    sourceIJKMatrix, sourceImage, resultGradientPtrX, resultGradientPtrY)
#endif // _OPENMP
        for(index=0;index<targetVoxelNumber; index++){

            grad[index]=0.0;
            grad[index]=0.0;

            if(maskPtr[index]>-1){
                world[0]=(FieldTYPE) deformationFieldPtrX[index];
                world[1]=(FieldTYPE) deformationFieldPtrY[index];

                /* real -> voxel; source space */
                position[0] = world[0]*sourceIJKMatrix->m[0][0] + world[1]*sourceIJKMatrix->m[0][1] +
                        sourceIJKMatrix->m[0][3];
                position[1] = world[0]*sourceIJKMatrix->m[1][0] + world[1]*sourceIJKMatrix->m[1][1] +
                        sourceIJKMatrix->m[1][3];

                previous[0] = (int)floor(position[0]);
                previous[1] = (int)floor(position[1]);
                // basis values along the x axis
                relative=position[0]-(FieldTYPE)previous[0];
                interpolantCubicSpline<FieldTYPE>(relative, xBasis, xDeriv);
                // basis values along the y axis
                relative=position[1]-(FieldTYPE)previous[1];
                interpolantCubicSpline<FieldTYPE>(relative, yBasis, yDeriv);

                previous[0]--;previous[1]--;

                bg=false;
                for(b=0; b<4; b++){
                    Y= previous[1]+b;
                    yPointer = &sourceIntensity[Y*sourceImage->nx];
                    if(-1<Y && Y<sourceImage->ny){
                        xyPointer = &yPointer[previous[0]];
                        xTempNewValue=0.0;
                        yTempNewValue=0.0;
                        for(a=0; a<4; a++){
                            if(-1<(previous[0]+a) && (previous[0]+a)<sourceImage->nx){
                                coeff = (FieldTYPE)*xyPointer;
                                xTempNewValue +=  coeff * xDeriv[a];
                                yTempNewValue +=  coeff * xBasis[a];
                            }
                            else bg=true;
                            xyPointer++;
                        }
                        grad[0] += (xTempNewValue * yBasis[b]);
                        grad[1] += (yTempNewValue * yDeriv[b]);
                    }
                    else bg=true;
                }

                if(bg==true){
                    grad[0]=0.0;
                    grad[1]=0.0;
                }
            }
            resultGradientPtrX[index] = (GradientTYPE)grad[0];
            resultGradientPtrY[index] = (GradientTYPE)grad[1];
        }
    }
}
/* *************************************************************** */
template <class FieldTYPE, class SourceTYPE, class GradientTYPE>
void reg_getSourceImageGradient3(   nifti_image *targetImage,
                                 nifti_image *sourceImage,
                                 nifti_image *resultGradientImage,
                                 nifti_image *deformationField,
                                 int *mask,
                                 int interp)
{
    /* The deformation field contains the position in the real world */

    if(interp==3){
        if(targetImage->nz>1){
            CubicSplineGradientResultImage
                    <SourceTYPE,GradientTYPE,FieldTYPE>(sourceImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask);
        }
        else{
            CubicSplineGradientResultImage2D
                    <SourceTYPE,GradientTYPE,FieldTYPE>(sourceImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask);
        }
    }
    else{ // trilinear interpolation [ by default ]
        if(targetImage->nz>1){
            TrilinearGradientResultImage
                    <SourceTYPE,GradientTYPE,FieldTYPE>(sourceImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask);
        }
        else{
            TrilinearGradientResultImage2D
                    <SourceTYPE,GradientTYPE,FieldTYPE>(sourceImage,
                                                        deformationField,
                                                        resultGradientImage,
                                                        mask);
        }
    }
}
/* *************************************************************** */
template <class FieldTYPE, class SourceTYPE>
void reg_getSourceImageGradient2(nifti_image *targetImage,
                                 nifti_image *sourceImage,
                                 nifti_image *resultGradientImage,
                                 nifti_image *deformationField,
                                 int *mask,
                                 int interp
                                 )
{
    switch(resultGradientImage->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getSourceImageGradient3<FieldTYPE,SourceTYPE,float>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getSourceImageGradient3<FieldTYPE,SourceTYPE,double>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
        return;
    }
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_getSourceImageGradient1(nifti_image *targetImage,
                                 nifti_image *sourceImage,
                                 nifti_image *resultGradientImage,
                                 nifti_image *deformationField,
                                 int *mask,
                                 int interp
                                 )
{
    switch(sourceImage->datatype){
    case NIFTI_TYPE_UINT8:
        reg_getSourceImageGradient2<FieldTYPE,unsigned char>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_INT8:
        reg_getSourceImageGradient2<FieldTYPE,char>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_UINT16:
        reg_getSourceImageGradient2<FieldTYPE,unsigned short>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_INT16:
        reg_getSourceImageGradient2<FieldTYPE,short>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_UINT32:
        reg_getSourceImageGradient2<FieldTYPE,unsigned int>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_INT32:
        reg_getSourceImageGradient2<FieldTYPE,int>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_getSourceImageGradient2<FieldTYPE,float>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getSourceImageGradient2<FieldTYPE,double>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getVoxelBasedNMIGradientUsingPW\tThe result image data type is not supported\n");
        return;
    }
}
/* *************************************************************** */
void reg_getSourceImageGradient(nifti_image *targetImage,
                                nifti_image *sourceImage,
                                nifti_image *resultGradientImage,
                                nifti_image *deformationField,
                                int *mask,
                                int interp
                                )
{
    // a mask array is created if no mask is specified
    bool MrPropreRule=false;
    if(mask==NULL){
        mask=(int *)calloc(targetImage->nx*targetImage->ny*targetImage->nz,sizeof(int)); // voxels in the background are set to -1 so 0 will do the job here
        MrPropreRule=true;
    }

    // Check if the dimension are correct
    if(sourceImage->nt != resultGradientImage->nt){
        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tThe source and result images have different dimension along the time axis\n");
        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tNothing has been done\n");
        return;
    }

    switch(deformationField->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_getSourceImageGradient1<float>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getSourceImageGradient1<double>
                (targetImage,sourceImage,resultGradientImage,deformationField,mask,interp);
        break;
    default:
        printf("[NiftyReg ERROR] reg_getSourceImageGradient\tDeformation field pixel type unsupported.\n");
        break;
    }
    if(MrPropreRule==true) free(mask);
}
/* *************************************************************** */
/* *************************************************************** */

#endif
