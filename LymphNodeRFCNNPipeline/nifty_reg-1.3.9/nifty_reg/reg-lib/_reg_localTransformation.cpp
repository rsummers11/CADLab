/*
 *  _reg_bspline.cpp
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANSFORMATION_CPP
#define _REG_TRANSFORMATION_CPP

#include "_reg_localTransformation.h"

/* *************************************************************** */
/* *************************************************************** */

template<class DTYPE>
int reg_round(DTYPE x)
{
#ifdef _WINDOWS
    return int(x > 0.0 ? x + 0.5 : x - 0.5);
#else
    return int(round(x));
#endif
}
/* *************************************************************** */
template<class DTYPE>
int reg_floor(DTYPE x)
{
    //    return int(x > 0.0 ? (int)x : (int)(x-1));
    return int(floor(x));
}
/* *************************************************************** */
template<class DTYPE>
int reg_ceil(DTYPE x)
{
    //   int casted = (int)x;
    //   return int((x-casted)==0 ? casted : (casted+1));
    return int(ceil(x));
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis, DTYPE *values)
{
    DTYPE FF= basis*basis;
    DTYPE FFF= FF*basis;
    DTYPE MF=(DTYPE)(1.0-basis);
    values[0] = (DTYPE)((MF)*(MF)*(MF)/(6.0));
    values[1] = (DTYPE)((3.0*FFF - 6.0*FF + 4.0)/6.0);
    values[2] = (DTYPE)((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
    values[3] = (DTYPE)(FFF/6.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
{
    Get_BSplineBasisValues<DTYPE>(basis, values);
    first[3]= (DTYPE)(basis * basis / 2.0);
    first[0]= (DTYPE)(basis - 1.0/2.0 - first[3]);
    first[2]= (DTYPE)(1.0 + first[0] - 2.0*first[3]);
    first[1]= - first[0] - first[2] - first[3];
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
{
    Get_BSplineBasisValues<DTYPE>(basis, values, first);
    second[3]= basis;
    second[0]= (DTYPE)(1.0 - second[3]);
    second[2]= (DTYPE)(second[0] - 2.0*second[3]);
    second[1]= - second[0] - second[2] - second[3];
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values)
{
    DTYPE FF= basis*basis;
    values[0] = (DTYPE)((basis * ((2.0-basis)*basis - 1.0))/2.0);
    values[1] = (DTYPE)((FF * (3.0*basis-5.0) + 2.0)/2.0);
    values[2] = (DTYPE)((basis * ((4.0-3.0*basis)*basis + 1.0))/2.0);
    values[3] = (DTYPE)((basis-1.0) * FF/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
{
    Get_SplineBasisValues<DTYPE>(basis,values);
    DTYPE FF= basis*basis;
    first[0] = (DTYPE)((4.0*basis - 3.0*FF - 1.0)/2.0);
    first[1] = (DTYPE)((9.0*basis - 10.0) * basis/2.0);
    first[2] = (DTYPE)((8.0*basis - 9.0*FF + 1)/2.0);
    first[3] = (DTYPE)((3.0*basis - 2.0) * basis/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
{
    Get_SplineBasisValues<DTYPE>(basis, values, first);
    second[0] = (DTYPE)(2.0 - 3.0*basis);
    second[1] = (DTYPE)(9.0*basis - 5.0);
    second[2] = (DTYPE)(4.0 - 9.0*basis);
    second[3] = (DTYPE)(3.0*basis - 1.0);
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    bool affineInit)
{
    unsigned int index;
    unsigned int coord=0;
    DTYPE *xxPtr=NULL, *yyPtr=NULL;
    if(affineInit){

        mat44 *voxel2realMatrix=NULL;
        if(splineControlPoint->sform_code>0)
            voxel2realMatrix=&(splineControlPoint->sto_xyz);
        else voxel2realMatrix=&(splineControlPoint->qto_xyz);

        for(int Y=startY; Y<startY+4; Y++){
            bool out=false;
            if(Y>-1 && Y<splineControlPoint->ny){
                index = Y*splineControlPoint->nx;
                xxPtr = &splineX[index];
                yyPtr = &splineY[index];
            }
            else out=true;
            for(int X=startX; X<startX+4; X++){
                if(X>-1 && X<splineControlPoint->nx && out==false){
                    dispX[coord] = (DTYPE)xxPtr[X];
                    dispY[coord] = (DTYPE)yyPtr[X];
                }
                else{
                    dispX[coord] = X*voxel2realMatrix->m[0][0]
                            + Y*voxel2realMatrix->m[0][1]
                            + voxel2realMatrix->m[0][3];
                    dispY[coord] = X*voxel2realMatrix->m[1][0]
                            + Y*voxel2realMatrix->m[1][1]
                            + voxel2realMatrix->m[1][3];
                }
                coord++;
            }
        }
    }
    else{
        memset(dispX,0,16*sizeof(DTYPE));
        memset(dispY,0,16*sizeof(DTYPE));
        for(int Y=startY; Y<startY+4; Y++){
            if(Y>-1 && Y<splineControlPoint->ny){
                index = Y*splineControlPoint->nx;
                xxPtr = &splineX[index];
                yyPtr = &splineY[index];
                for(int X=startX; X<startX+4; X++){
                    if(X>-1 && X<splineControlPoint->nx){
                        dispX[coord] = (DTYPE)xxPtr[X];
                        dispY[coord] = (DTYPE)yyPtr[X];
                    }
                    coord++;
                }
            }
            else coord+=4;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValuesApprox(int startX,
                          int startY,
                          nifti_image *splineControlPoint,
                          DTYPE *splineX,
                          DTYPE *splineY,
                          DTYPE *dispX,
                          DTYPE *dispY,
                          bool affineInit)
{
    unsigned int index;
    unsigned int coord=0;
    DTYPE *xxPtr=NULL, *yyPtr=NULL;
    if(affineInit){

        mat44 *voxel2realMatrix=NULL;
        if(splineControlPoint->sform_code>0) voxel2realMatrix=&(splineControlPoint->sto_xyz);
        else voxel2realMatrix=&(splineControlPoint->qto_xyz);

        for(int Y=startY; Y<startY+3; Y++){
            bool out=false;
            if(Y>-1 && Y<splineControlPoint->ny){
                index = Y*splineControlPoint->nx;
                xxPtr = &splineX[index];
                yyPtr = &splineY[index];
            }
            else out=true;
            for(int X=startX; X<startX+3; X++){
                if(X>-1 && X<splineControlPoint->nx && out==false){
                    dispX[coord] = (DTYPE)xxPtr[X];
                    dispY[coord] = (DTYPE)yyPtr[X];
                }
                else{
                    dispX[coord] = X*voxel2realMatrix->m[0][0]
                            + Y*voxel2realMatrix->m[0][1]
                            + voxel2realMatrix->m[0][3];
                    dispY[coord] = X*voxel2realMatrix->m[1][0]
                            + Y*voxel2realMatrix->m[1][1]
                            + voxel2realMatrix->m[1][3];
                }
                coord++;
            }
        }
    }
    else{
        memset(dispX,0,9*sizeof(DTYPE));
        memset(dispY,0,9*sizeof(DTYPE));
        for(int Y=startY; Y<startY+3; Y++){
            if(Y>-1 && Y<splineControlPoint->ny){
                index = Y*splineControlPoint->nx;
                xxPtr = &splineX[index];
                yyPtr = &splineY[index];
                for(int X=startX; X<startX+3; X++){
                    if(X>-1 && X<splineControlPoint->nx){
                        dispX[coord] = (DTYPE)xxPtr[X];
                        dispY[coord] = (DTYPE)yyPtr[X];
                    }
                    coord++;
                }
            }
            else coord+=3;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    int startZ,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *splineZ,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    DTYPE *dispZ,
                    bool affineInit)
{
    unsigned int index;
    unsigned int coord=0;
    DTYPE *xPtr=NULL, *yPtr=NULL, *zPtr=NULL;
    DTYPE *xxPtr=NULL, *yyPtr=NULL, *zzPtr=NULL;
    if(affineInit){
        mat44 *voxel2realMatrix=NULL;
        if(splineControlPoint->sform_code>0) voxel2realMatrix=&(splineControlPoint->sto_xyz);
        else voxel2realMatrix=&(splineControlPoint->qto_xyz);
        for(int Z=startZ; Z<startZ+4; Z++){
            bool out=false;
            if(Z>-1 && Z<splineControlPoint->nz){
                index=Z*splineControlPoint->nx*splineControlPoint->ny;
                xPtr = &splineX[index];
                yPtr = &splineY[index];
                zPtr = &splineZ[index];
            }
            else out=true;
            for(int Y=startY; Y<startY+4; Y++){
                if(Y>-1 && Y<splineControlPoint->ny && out==false){
                    index = Y*splineControlPoint->nx;
                    xxPtr = &xPtr[index];
                    yyPtr = &yPtr[index];
                    zzPtr = &zPtr[index];
                }
                else out=true;
                for(int X=startX; X<startX+4; X++){
                    if(X>-1 && X<splineControlPoint->nx &&out==false){
                        dispX[coord] = (DTYPE)xxPtr[X];
                        dispY[coord] = (DTYPE)yyPtr[X];
                        dispZ[coord] = (DTYPE)zzPtr[X];
                    }
                    else{
                        dispX[coord] = X*voxel2realMatrix->m[0][0]
                                + Y*voxel2realMatrix->m[0][1]
                                + Z*voxel2realMatrix->m[0][2]
                                + voxel2realMatrix->m[0][3];
                        dispY[coord] = X*voxel2realMatrix->m[1][0]
                                + Y*voxel2realMatrix->m[1][1]
                                + Z*voxel2realMatrix->m[1][2]
                                + voxel2realMatrix->m[1][3];
                        dispZ[coord] = X*voxel2realMatrix->m[2][0]
                                + Y*voxel2realMatrix->m[2][1]
                                + Z*voxel2realMatrix->m[2][2]
                                + voxel2realMatrix->m[2][3];
                    }
                    coord++;
                }
            }
        }
    }
    else{
        memset(dispX,0,64*sizeof(DTYPE));
        memset(dispY,0,64*sizeof(DTYPE));
        memset(dispZ,0,64*sizeof(DTYPE));
        for(int Z=startZ; Z<startZ+4; Z++){
            if(Z>-1 && Z<splineControlPoint->nz){
                index = Z*splineControlPoint->nx*splineControlPoint->ny;
                xPtr = &splineX[index];
                yPtr = &splineY[index];
                zPtr = &splineZ[index];
                for(int Y=startY; Y<startY+4; Y++){
                    if(Y>-1 && Y<splineControlPoint->ny){
                        index = Y*splineControlPoint->nx;
                        xxPtr = &xPtr[index];
                        yyPtr = &yPtr[index];
                        zzPtr = &zPtr[index];
                        for(int X=startX; X<startX+4; X++){
                            if(X>-1 && X<splineControlPoint->nx){
                                dispX[coord] = (DTYPE)xxPtr[X];
                                dispY[coord] = (DTYPE)yyPtr[X];
                                dispZ[coord] = (DTYPE)zzPtr[X];
                            }
                            coord++;
                        }
                    }
                    else coord+=4;
                }
            }
            else coord+=16;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValuesApprox(int startX,
                          int startY,
                          int startZ,
                          nifti_image *splineControlPoint,
                          DTYPE *splineX,
                          DTYPE *splineY,
                          DTYPE *splineZ,
                          DTYPE *dispX,
                          DTYPE *dispY,
                          DTYPE *dispZ,
                          bool affineInit)
{
    unsigned int index;
    unsigned int coord=0;
    DTYPE *xPtr=NULL, *yPtr=NULL, *zPtr=NULL;
    DTYPE *xxPtr=NULL, *yyPtr=NULL, *zzPtr=NULL;
    if(affineInit){
        mat44 *voxel2realMatrix=NULL;
        if(splineControlPoint->sform_code>0) voxel2realMatrix=&(splineControlPoint->sto_xyz);
        else voxel2realMatrix=&(splineControlPoint->qto_xyz);
        for(int Z=startZ; Z<startZ+3; Z++){
            bool out=false;
            if(Z>-1 && Z<splineControlPoint->nz){
                index=Z*splineControlPoint->nx*splineControlPoint->ny;
                xPtr = &splineX[index];
                yPtr = &splineY[index];
                zPtr = &splineZ[index];
            }
            else out=true;
            for(int Y=startY; Y<startY+3; Y++){
                if(Y>-1 && Y<splineControlPoint->ny && out==false){
                    index = Y*splineControlPoint->nx;
                    xxPtr = &xPtr[index];
                    yyPtr = &yPtr[index];
                    zzPtr = &zPtr[index];
                }
                else out=true;
                for(int X=startX; X<startX+3; X++){
                    if(X>-1 && X<splineControlPoint->nx &&out==false){
                        dispX[coord] = (DTYPE)xxPtr[X];
                        dispY[coord] = (DTYPE)yyPtr[X];
                        dispZ[coord] = (DTYPE)zzPtr[X];
                    }
                    else{
                        dispX[coord] = X*voxel2realMatrix->m[0][0]
                                + Y*voxel2realMatrix->m[0][1]
                                + Z*voxel2realMatrix->m[0][2]
                                + voxel2realMatrix->m[0][3];
                        dispY[coord] = X*voxel2realMatrix->m[1][0]
                                + Y*voxel2realMatrix->m[1][1]
                                + Z*voxel2realMatrix->m[1][2]
                                + voxel2realMatrix->m[1][3];
                        dispZ[coord] = X*voxel2realMatrix->m[2][0]
                                + Y*voxel2realMatrix->m[2][1]
                                + Z*voxel2realMatrix->m[2][2]
                                + voxel2realMatrix->m[2][3];
                    }
                    coord++;
                }
            }
        }
    }
    else{
        memset(dispX,0,27*sizeof(DTYPE));
        memset(dispY,0,27*sizeof(DTYPE));
        memset(dispZ,0,27*sizeof(DTYPE));
        for(int Z=startZ; Z<startZ+3; Z++){
            if(Z>-1 && Z<splineControlPoint->nz){
                index = Z*splineControlPoint->nx*splineControlPoint->ny;
                xPtr = &splineX[index];
                yPtr = &splineY[index];
                zPtr = &splineZ[index];
                for(int Y=startY; Y<startY+3; Y++){
                    if(Y>-1 && Y<splineControlPoint->ny){
                        index = Y*splineControlPoint->nx;
                        xxPtr = &xPtr[index];
                        yyPtr = &yPtr[index];
                        zzPtr = &zPtr[index];
                        for(int X=startX; X<startX+3; X++){
                            if(X>-1 && X<splineControlPoint->nx){
                                dispX[coord] = (DTYPE)xxPtr[X];
                                dispY[coord] = (DTYPE)yyPtr[X];
                                dispZ[coord] = (DTYPE)zzPtr[X];
                            }
                            coord++;
                        }
                    }
                    else coord+=3;
                }
            }
            else coord+=9;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_createControlPointGrid(nifti_image **controlPointGridImage,
                                nifti_image *referenceImage,
                                float *spacingMillimeter)
{
    // Define the control point grid dimension
    int dim_cpp[8];
    dim_cpp[0]=5;
    dim_cpp[1]=(int)floor(referenceImage->nx*referenceImage->dx/spacingMillimeter[0])+5;
    dim_cpp[2]=(int)floor(referenceImage->ny*referenceImage->dy/spacingMillimeter[1])+5;
    dim_cpp[3]=1;
    dim_cpp[5]=2;
    if(referenceImage->nz>1){
        dim_cpp[3]=(int)floor(referenceImage->nz*referenceImage->dz/spacingMillimeter[2])+5;
        dim_cpp[5]=3;
    }
    dim_cpp[4]=dim_cpp[6]=dim_cpp[7]=1;

    // Create the new control point grid image and allocate its space
    if(sizeof(DTYPE)==4)
        *controlPointGridImage = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT32, true);
    else *controlPointGridImage = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT64, true);

    // Fill the header information
    (*controlPointGridImage)->cal_min=0;
    (*controlPointGridImage)->cal_max=0;
    (*controlPointGridImage)->pixdim[0]=1.0f;
    (*controlPointGridImage)->pixdim[1]=(*controlPointGridImage)->dx=spacingMillimeter[0];
    (*controlPointGridImage)->pixdim[2]=(*controlPointGridImage)->dy=spacingMillimeter[1];
    if(referenceImage->nz==1){
        (*controlPointGridImage)->pixdim[3]=(*controlPointGridImage)->dz=1.0f;
    }
    else (*controlPointGridImage)->pixdim[3]=(*controlPointGridImage)->dz=spacingMillimeter[2];
    (*controlPointGridImage)->pixdim[4]=(*controlPointGridImage)->dt=1.0f;
    (*controlPointGridImage)->pixdim[5]=(*controlPointGridImage)->du=1.0f;
    (*controlPointGridImage)->pixdim[6]=(*controlPointGridImage)->dv=1.0f;
    (*controlPointGridImage)->pixdim[7]=(*controlPointGridImage)->dw=1.0f;

    // Reproduce the orientation of the reference image and add a one voxel shift
    if(referenceImage->qform_code+referenceImage->sform_code>0){
        (*controlPointGridImage)->qform_code=referenceImage->qform_code;
        (*controlPointGridImage)->sform_code=referenceImage->sform_code;
    }
    else{
        (*controlPointGridImage)->qform_code=1;
        (*controlPointGridImage)->sform_code=0;
    }

    // The qform (and sform) are set for the control point position image
    (*controlPointGridImage)->quatern_b=referenceImage->quatern_b;
    (*controlPointGridImage)->quatern_c=referenceImage->quatern_c;
    (*controlPointGridImage)->quatern_d=referenceImage->quatern_d;
    (*controlPointGridImage)->qoffset_x=referenceImage->qoffset_x;
    (*controlPointGridImage)->qoffset_y=referenceImage->qoffset_y;
    (*controlPointGridImage)->qoffset_z=referenceImage->qoffset_z;
    (*controlPointGridImage)->qfac=referenceImage->qfac;
    (*controlPointGridImage)->qto_xyz = nifti_quatern_to_mat44((*controlPointGridImage)->quatern_b,
                                                             (*controlPointGridImage)->quatern_c,
                                                             (*controlPointGridImage)->quatern_d,
                                                             (*controlPointGridImage)->qoffset_x,
                                                             (*controlPointGridImage)->qoffset_y,
                                                             (*controlPointGridImage)->qoffset_z,
                                                             (*controlPointGridImage)->dx,
                                                             (*controlPointGridImage)->dy,
                                                             (*controlPointGridImage)->dz,
                                                             (*controlPointGridImage)->qfac);

    // Origin is shifted from 1 control point in the qform
    float originIndex[3];
    float originReal[3];
    originIndex[0] = -1.0f;
    originIndex[1] = -1.0f;
    originIndex[2] = 0.0f;
    if(referenceImage->nz>1) originIndex[2] = -1.0f;
    reg_mat44_mul(&((*controlPointGridImage)->qto_xyz), originIndex, originReal);
    (*controlPointGridImage)->qto_xyz.m[0][3] = (*controlPointGridImage)->qoffset_x = originReal[0];
    (*controlPointGridImage)->qto_xyz.m[1][3] = (*controlPointGridImage)->qoffset_y = originReal[1];
    (*controlPointGridImage)->qto_xyz.m[2][3] = (*controlPointGridImage)->qoffset_z = originReal[2];

    (*controlPointGridImage)->qto_ijk = nifti_mat44_inverse((*controlPointGridImage)->qto_xyz);

    // Update the sform if required
    if((*controlPointGridImage)->sform_code>0){
        float scalingRatio[3];
        scalingRatio[0]= (*controlPointGridImage)->dx / referenceImage->dx;
        scalingRatio[1]= (*controlPointGridImage)->dy / referenceImage->dy;
        scalingRatio[2]= (*controlPointGridImage)->dz / referenceImage->dz;

        (*controlPointGridImage)->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
        (*controlPointGridImage)->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
        (*controlPointGridImage)->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
        (*controlPointGridImage)->sto_xyz.m[3][0]=referenceImage->sto_xyz.m[3][0];
        (*controlPointGridImage)->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
        (*controlPointGridImage)->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
        (*controlPointGridImage)->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
        (*controlPointGridImage)->sto_xyz.m[3][1]=referenceImage->sto_xyz.m[3][1];
        (*controlPointGridImage)->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
        (*controlPointGridImage)->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
        (*controlPointGridImage)->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
        (*controlPointGridImage)->sto_xyz.m[3][2]=referenceImage->sto_xyz.m[3][2];
        (*controlPointGridImage)->sto_xyz.m[0][3]=referenceImage->sto_xyz.m[0][3];
        (*controlPointGridImage)->sto_xyz.m[1][3]=referenceImage->sto_xyz.m[1][3];
        (*controlPointGridImage)->sto_xyz.m[2][3]=referenceImage->sto_xyz.m[2][3];
        (*controlPointGridImage)->sto_xyz.m[3][3]=referenceImage->sto_xyz.m[3][3];

        // Origin is shifted from 1 control point in the sform
        reg_mat44_mul(&((*controlPointGridImage)->sto_xyz), originIndex, originReal);
        (*controlPointGridImage)->sto_xyz.m[0][3] = originReal[0];
        (*controlPointGridImage)->sto_xyz.m[1][3] = originReal[1];
        (*controlPointGridImage)->sto_xyz.m[2][3] = originReal[2];
        (*controlPointGridImage)->sto_ijk = nifti_mat44_inverse((*controlPointGridImage)->sto_xyz);
    }

    (*controlPointGridImage)->intent_code=NIFTI_INTENT_VECTOR;
    memset((*controlPointGridImage)->intent_name, 0, 16);
    strcpy((*controlPointGridImage)->intent_name,"NREG_CPP_FILE");
}
template void reg_createControlPointGrid<float>(nifti_image **, nifti_image *, float *);
template void reg_createControlPointGrid<double>(nifti_image **, nifti_image *, float *);
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_spline_getDeformationField2D(nifti_image *splineControlPoint,
                                      nifti_image *referenceImage,
                                      nifti_image *deformationField,
                                      int *mask,
                                      bool composition,
                                      bool bspline)
{

#if _USE_SSE
    union{
        __m128 m;
        float f[4];
    } val;
    __m128 tempCurrent, tempX, tempY;
#ifdef _WINDOWS
    __declspec(align(16)) DTYPE temp[4];
    __declspec(align(16)) DTYPE yBasis[4];
    union{__m128 m[16];__declspec(align(16)) DTYPE f[16];} xControlPointCoordinates;
    union{__m128 m[16];__declspec(align(16)) DTYPE f[16];} yControlPointCoordinates;
    union u1{__m128 m[4]; __declspec(align(16)) DTYPE f[16];}xyBasis;
#else // _WINDOWS
    DTYPE temp[4] __attribute__((aligned(16)));
    DTYPE yBasis[4] __attribute__((aligned(16)));
    union{__m128 m[16];DTYPE f[16] __attribute__((aligned(16)));} xControlPointCoordinates;
    union{__m128 m[16];DTYPE f[16] __attribute__((aligned(16)));} yControlPointCoordinates;
    union u1{__m128 m[4]; DTYPE f[16] __attribute__((aligned(16)));}xyBasis;
#endif // _WINDOWS
#else // _USE_SSE
    DTYPE temp[4];
    DTYPE yBasis[4];
    DTYPE xyBasis[16];
    DTYPE xControlPointCoordinates[16];
    DTYPE yControlPointCoordinates[16];
#endif // _USE_SSE


    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    DTYPE *fieldPtrX=static_cast<DTYPE *>(deformationField->data);
    DTYPE *fieldPtrY=&fieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];

    DTYPE gridVoxelSpacing[2];
    gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;

    DTYPE basis, xReal, yReal, xVoxel, yVoxel;
    int x, y, a, b, xPre, yPre, oldXpre, oldYpre, index, coord;

    if(composition){ // Composition of deformation fields

        // read the ijk sform or qform, as appropriate
        mat44 *targetMatrix_real_to_voxel;
        if(splineControlPoint->sform_code>0)
            targetMatrix_real_to_voxel=&(splineControlPoint->sto_ijk);
        else targetMatrix_real_to_voxel=&(splineControlPoint->qto_ijk);

        for(y=0; y<deformationField->ny; y++){
            index=y*deformationField->nx;
            oldXpre=oldYpre=99999999;
            for(x=0; x<deformationField->nx; x++){

                // The previous position at the current pixel position is read
                xReal = (DTYPE)(fieldPtrX[index]);
                yReal = (DTYPE)(fieldPtrY[index]);

                // From real to pixel position in the CPP
                xVoxel = targetMatrix_real_to_voxel->m[0][0]*xReal
                        + targetMatrix_real_to_voxel->m[0][1]*yReal + targetMatrix_real_to_voxel->m[0][3];
                yVoxel = targetMatrix_real_to_voxel->m[1][0]*xReal
                        + targetMatrix_real_to_voxel->m[1][1]*yReal + targetMatrix_real_to_voxel->m[1][3];

                // The spline coefficients are computed
                xPre=(int)floor(xVoxel);
                basis=xVoxel-(DTYPE)xPre;--xPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, temp);
                else Get_SplineBasisValues<DTYPE>(basis, temp);

                yPre=(int)floor(yVoxel);
                basis=yVoxel-(DTYPE)yPre;--yPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, yBasis);
                else Get_SplineBasisValues<DTYPE>(basis, yBasis);


                if(xVoxel>=0 && xVoxel<=referenceImage->nx-1 &&
                   yVoxel>=0 && yVoxel<=referenceImage->ny-1){

                    // The control point postions are extracted
                    if(oldXpre!=xPre || oldYpre!=yPre){
#ifdef _USE_SSE
                        get_GridValues<DTYPE>(xPre,
                                              yPre,
                                              splineControlPoint,
                                              controlPointPtrX,
                                              controlPointPtrY,
                                              xControlPointCoordinates.f,
                                              yControlPointCoordinates.f,
                                              false);
#else // _USE_SSE
                        get_GridValues<DTYPE>(xPre,
                                              yPre,
                                              splineControlPoint,
                                              controlPointPtrX,
                                              controlPointPtrY,
                                              xControlPointCoordinates,
                                              yControlPointCoordinates,
                                              false);
#endif // _USE_SSE
                        oldXpre=xPre;oldYpre=yPre;
                    }
                    xReal=0.0;
                    yReal=0.0;

                    if(mask[index]>-1){
#if _USE_SSE
                        coord=0;
                        for(b=0; b<4; b++){
                            for(a=0; a<4; a++){
                                xyBasis.f[coord++] = temp[a] * yBasis[b];
                            }
                        }

                        tempX =  _mm_set_ps1(0.0);
                        tempY =  _mm_set_ps1(0.0);
                        //addition and multiplication of the 16 basis value and CP position for each axis
                        for(a=0; a<4; a++){
                            tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                            tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY );
                        }
                        //the values stored in SSE variables are transfered to normal float
                        val.m = tempX;
                        xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = tempY;
                        yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        for(b=0; b<4; b++){
                            for(a=0; a<4; a++){
                                DTYPE tempValue = temp[a] * yBasis[b];
                                xReal += xControlPointCoordinates[b*4+a] * tempValue;
                                yReal += yControlPointCoordinates[b*4+a] * tempValue;
                            }
                        }
#endif
                    }

                    fieldPtrX[index] = (DTYPE)xReal;
                    fieldPtrY[index] = (DTYPE)yReal;
                }
                index++;
            }
        }
    }
    else{ // starting deformation field is blank - !composition

#ifdef _OPENMP
#ifdef _USE_SSE
#pragma  omp parallel for default(none) \
        shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
               controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
        private(x, y, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, \
                val, temp, yBasis, tempCurrent, xyBasis, tempX, tempY, \
                xControlPointCoordinates, yControlPointCoordinates)
#else // _USE_SSE
#pragma  omp parallel for default(none) \
        shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
               controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
        private(x, y, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, coord, \
                temp, yBasis, xyBasis, xControlPointCoordinates, yControlPointCoordinates)
#endif // _USE_SEE
#endif // _OPENMP
        for( y=0; y<deformationField->ny; y++){
            index=y*deformationField->nx;
            oldXpre=oldYpre=9999999;

            yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
            basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) Get_BSplineBasisValues<DTYPE>(basis, yBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);

            for(x=0; x<deformationField->nx; x++){

                xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
                basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, temp);
                else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE
                val.f[0] = temp[0];
                val.f[1] = temp[1];
                val.f[2] = temp[2];
                val.f[3] = temp[3];
                tempCurrent=val.m;
                for(a=0;a<4;a++){
                    val.m=_mm_set_ps1(yBasis[a]);
                    xyBasis.m[a]=_mm_mul_ps(tempCurrent,val.m);
                }
#else
                coord=0;
                for(a=0;a<4;a++){
                    xyBasis[coord++]=temp[0]*yBasis[a];
                    xyBasis[coord++]=temp[1]*yBasis[a];
                    xyBasis[coord++]=temp[2]*yBasis[a];
                    xyBasis[coord++]=temp[3]*yBasis[a];
                }
#endif
                if(oldXpre!=xPre || oldYpre!=yPre){
#ifdef _USE_SSE
                    get_GridValues<DTYPE>(xPre,
                                          yPre,
                                          splineControlPoint,
                                          controlPointPtrX,
                                          controlPointPtrY,
                                          xControlPointCoordinates.f,
                                          yControlPointCoordinates.f,
                                          false);
#else // _USE_SSE
                    get_GridValues<DTYPE>(xPre,
                                          yPre,
                                          splineControlPoint,
                                          controlPointPtrX,
                                          controlPointPtrY,
                                          xControlPointCoordinates,
                                          yControlPointCoordinates,
                                          false);
#endif // _USE_SSE
                    oldXpre=xPre; oldYpre=yPre;
                }

                xReal=0.0;
                yReal=0.0;

                if(mask[index]>-1){
#if _USE_SSE
                    tempX =  _mm_set_ps1(0.0);
                    tempY =  _mm_set_ps1(0.0);
                    //addition and multiplication of the 64 basis value and CP displacement for each axis
                    for(a=0; a<4; a++){
                        tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY );
                    }
                    //the values stored in SSE variables are transfered to normal float
                    val.m=tempX;
                    xReal=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                    val.m=tempY;
                    yReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                    for(a=0; a<16; a++){
                        xReal += xControlPointCoordinates[a] * xyBasis[a];
                        yReal += yControlPointCoordinates[a] * xyBasis[a];
                    }
#endif
                }// mask
                fieldPtrX[index] = (DTYPE)xReal;
                fieldPtrY[index] = (DTYPE)yReal;
                index++;
            } // x
        } // y
    } // composition

    return;
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_getDeformationField3D(nifti_image *splineControlPoint,
                                      nifti_image *referenceImage,
                                      nifti_image *deformationField,
                                      int *mask,
                                      bool composition,
                                      bool bspline
                                      )
{
#if _USE_SSE
    union{
        __m128 m;
        float f[4];
    } val;
    __m128 tempX, tempY, tempZ, tempCurrent;
    __m128 xBasis_sse, yBasis_sse, zBasis_sse, temp_basis_sse, basis_sse;

#ifdef _WINDOWS
    __declspec(align(16)) DTYPE temp[4];
    __declspec(align(16)) DTYPE zBasis[4];
    union{__m128 m[16];__declspec(align(16)) DTYPE f[16];} xControlPointCoordinates;
    union{__m128 m[16];__declspec(align(16)) DTYPE f[16];} yControlPointCoordinates;
    union{__m128 m[16];__declspec(align(16)) DTYPE f[16];} zControlPointCoordinates;
#else // _WINDOWS
    DTYPE temp[4] __attribute__((aligned(16)));
    DTYPE zBasis[4] __attribute__((aligned(16)));
    union{__m128 m[16];DTYPE f[16] __attribute__((aligned(16)));} xControlPointCoordinates;
    union{__m128 m[16];DTYPE f[16] __attribute__((aligned(16)));} yControlPointCoordinates;
    union{__m128 m[16];DTYPE f[16] __attribute__((aligned(16)));} zControlPointCoordinates;
#endif // _WINDOWS
#else // _USE_SSE
    DTYPE temp[4];
    DTYPE zBasis[4];
    DTYPE xControlPointCoordinates[64];
    DTYPE yControlPointCoordinates[64];
    DTYPE zControlPointCoordinates[64];
    int coord;
#endif // _USE_SSE


    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    
    DTYPE *fieldPtrX=static_cast<DTYPE *>(deformationField->data);
    DTYPE *fieldPtrY=&fieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];
    DTYPE *fieldPtrZ=&fieldPtrY[deformationField->nx*deformationField->ny*deformationField->nz];

    DTYPE gridVoxelSpacing[3];
    gridVoxelSpacing[0] = splineControlPoint->dx / referenceImage->dx;
    gridVoxelSpacing[1] = splineControlPoint->dy / referenceImage->dy;
    gridVoxelSpacing[2] = splineControlPoint->dz / referenceImage->dz;

    DTYPE basis, oldBasis=(DTYPE)(1.1);

    int x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, index;
    DTYPE real[3];

    if(composition){ // Composition of deformation fields

        // read the ijk sform or qform, as appropriate
        mat44 *targetMatrix_real_to_voxel;
        if(splineControlPoint->sform_code>0)
            targetMatrix_real_to_voxel=&(splineControlPoint->sto_ijk);
        else targetMatrix_real_to_voxel=&(splineControlPoint->qto_ijk);
#ifdef _USE_SSE
#ifdef _WINDOWS
        __declspec(align(16)) DTYPE xBasis[4];
        __declspec(align(16)) DTYPE yBasis[4];
#else
        DTYPE xBasis[4] __attribute__((aligned(16)));
        DTYPE yBasis[4] __attribute__((aligned(16)));
#endif
#else // _USE_SSE
        DTYPE xBasis[4], yBasis[4];
#endif // _USE_SSE

        DTYPE voxel[3];

#ifdef _OPENMP
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
    index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates,  \
    tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
    temp_basis_sse, basis_sse, val) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, targetMatrix_real_to_voxel, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
    splineControlPoint, mask)
#else
#pragma omp parallel for default(none) \
    private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
    index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates, coord) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, targetMatrix_real_to_voxel, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
    splineControlPoint, mask)
#endif // _USE_SSE
#endif // _OPENMP
        for(z=0; z<deformationField->nz; z++){

            index=z*deformationField->nx*deformationField->ny;
            oldPreX=-99; oldPreY=-99; oldPreZ=-99;
            for(y=0; y<deformationField->ny; y++){
                for(x=0; x<deformationField->nx; x++){

                    if(mask[index]>-1){
                        // The previous position at the current pixel position is read
                        real[0] = fieldPtrX[index];
                        real[1] = fieldPtrY[index];
                        real[2] = fieldPtrZ[index];

                        // From real to pixel position in the control point space
                        reg_mat44_mul(targetMatrix_real_to_voxel, real, voxel);

                        // The spline coefficients are computed
                        xPre=(int)floor(voxel[0]);
                        basis=voxel[0]-(DTYPE)xPre;--xPre;
                        if(basis<0.0) basis=0.0; //rounding error
                        if(bspline) Get_BSplineBasisValues<DTYPE>(basis, xBasis);
                        else Get_SplineBasisValues<DTYPE>(basis, xBasis);

                        yPre=(int)floor(voxel[1]);
                        basis=voxel[1]-(DTYPE)yPre;--yPre;
                        if(basis<0.0) basis=0.0; //rounding error
                        if(bspline) Get_BSplineBasisValues<DTYPE>(basis, yBasis);
                        else Get_SplineBasisValues<DTYPE>(basis, yBasis);

                        zPre=(int)floor(voxel[2]);
                        basis=voxel[2]-(DTYPE)zPre;--zPre;
                        if(basis<0.0) basis=0.0; //rounding error
                        if(bspline) Get_BSplineBasisValues<DTYPE>(basis, zBasis);
                        else Get_SplineBasisValues<DTYPE>(basis, zBasis);

                        // The control point postions are extracted
                        if(xPre!=oldPreX || yPre!=oldPreY || zPre!=oldPreZ){
#ifdef _USE_SSE
                            get_GridValues<DTYPE>(xPre,
                                                  yPre,
                                                  zPre,
                                                  splineControlPoint,
                                                  controlPointPtrX,
                                                  controlPointPtrY,
                                                  controlPointPtrZ,
                                                  xControlPointCoordinates.f,
                                                  yControlPointCoordinates.f,
                                                  zControlPointCoordinates.f,
                                                  true);
#else // _USE_SSE
                            get_GridValues<DTYPE>(xPre,
                                                  yPre,
                                                  zPre,
                                                  splineControlPoint,
                                                  controlPointPtrX,
                                                  controlPointPtrY,
                                                  controlPointPtrZ,
                                                  xControlPointCoordinates,
                                                  yControlPointCoordinates,
                                                  zControlPointCoordinates,
                                                  true);
#endif // _USE_SSE
                            oldPreX=xPre;
                            oldPreY=yPre;
                            oldPreZ=zPre;
                        }

#if _USE_SSE
                        tempX =  _mm_set_ps1(0.0);
                        tempY =  _mm_set_ps1(0.0);
                        tempZ =  _mm_set_ps1(0.0);
                        val.f[0] = xBasis[0];
                        val.f[1] = xBasis[1];
                        val.f[2] = xBasis[2];
                        val.f[3] = xBasis[3];
                        xBasis_sse = val.m;

                        //addition and multiplication of the 16 basis value and CP position for each axis
                        for(c=0; c<4; c++){
                            for(b=0; b<4; b++){
                                yBasis_sse  = _mm_set_ps1(yBasis[b]);
                                zBasis_sse  = _mm_set_ps1(zBasis[c]);
                                temp_basis_sse = _mm_mul_ps(yBasis_sse, zBasis_sse);
                                basis_sse = _mm_mul_ps(temp_basis_sse, xBasis_sse);

                                tempX = _mm_add_ps(_mm_mul_ps(basis_sse, xControlPointCoordinates.m[c*4+b]), tempX );
                                tempY = _mm_add_ps(_mm_mul_ps(basis_sse, yControlPointCoordinates.m[c*4+b]), tempY );
                                tempZ = _mm_add_ps(_mm_mul_ps(basis_sse, zControlPointCoordinates.m[c*4+b]), tempZ );
                            }
                        }
                        //the values stored in SSE variables are transfered to normal float
                        val.m = tempX;
                        real[0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = tempY;
                        real[1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m = tempZ;
                        real[2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        real[0]=0.0;
                        real[1]=0.0;
                        real[2]=0.0;
                        coord=0;
                        for(c=0; c<4; c++){
                            for(b=0; b<4; b++){
                                for(a=0; a<4; a++){
                                    DTYPE tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                                    real[0] += xControlPointCoordinates[coord] * tempValue;
                                    real[1] += yControlPointCoordinates[coord] * tempValue;
                                    real[2] += zControlPointCoordinates[coord] * tempValue;
                                    coord++;
                                }
                            }
                        }
#endif
                        fieldPtrX[index] = real[0];
                        fieldPtrY[index] = real[1];
                        fieldPtrZ[index] = real[2];
                    }
                    index++;
                }
            }
        }
    }//Composition of deformation
    else{ // !composition
#ifdef _USE_SSE
    #ifdef _WINDOWS
            union u1{__m128 m[4];__declspec(align(16)) DTYPE f[16];}yzBasis;
            union u2{__m128 m[16];__declspec(align(16)) DTYPE f[64];}xyzBasis;
    #else // _WINDOWS
            union u1{__m128 m[4];DTYPE f[16] __attribute__((aligned(16)));}yzBasis;
            union u2{__m128 m[16];DTYPE f[64] __attribute__((aligned(16)));}xyzBasis;
    #endif // _WINDOWS
#else // _USE_SSE
            DTYPE yzBasis[16], xyzBasis[64];
#endif // _USE_SSE

#ifdef _OPENMP
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
    index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates, oldBasis, \
    tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
    temp_basis_sse, basis_sse, val, tempCurrent) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#else //  _USE_SSE
#pragma omp parallel for default(none) \
    private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
    index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
    yControlPointCoordinates, zControlPointCoordinates, oldBasis, coord) \
    shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
    gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#endif // _USE_SSE
#endif // _OPENMP
        for(z=0; z<deformationField->nz; z++){

            index=z*deformationField->nx*deformationField->ny;
            oldBasis=1.1;

            zPre=(int)((DTYPE)z/gridVoxelSpacing[2]);
            basis=(DTYPE)z/gridVoxelSpacing[2]-(DTYPE)zPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) Get_BSplineBasisValues<DTYPE>(basis, zBasis);
            else Get_SplineBasisValues<DTYPE>(basis, zBasis);

            for(y=0; y<deformationField->ny; y++){

                yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
                basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, temp);
                else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE
                val.f[0] = temp[0];
                val.f[1] = temp[1];
                val.f[2] = temp[2];
                val.f[3] = temp[3];
                tempCurrent=val.m;
                for(a=0;a<4;a++){
                    val.m=_mm_set_ps1(zBasis[a]);
                    yzBasis.m[a] = _mm_mul_ps(tempCurrent,val.m);
                }
#else
                coord=0;
                for(a=0;a<4;a++){
                    yzBasis[coord++]=temp[0]*zBasis[a];
                    yzBasis[coord++]=temp[1]*zBasis[a];
                    yzBasis[coord++]=temp[2]*zBasis[a];
                    yzBasis[coord++]=temp[3]*zBasis[a];
                }           
#endif

                for(x=0; x<deformationField->nx; x++){

                    xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
                    basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
                    if(basis<0.0) basis=0.0; //rounding error
                    if(bspline) Get_BSplineBasisValues<DTYPE>(basis, temp);
                    else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE

                    val.f[0] = temp[0];
                    val.f[1] = temp[1];
                    val.f[2] = temp[2];
                    val.f[3] = temp[3];
                    tempCurrent=val.m;
                    for(a=0;a<16;++a){
                        val.m=_mm_set_ps1(yzBasis.f[a]);
                        xyzBasis.m[a]=_mm_mul_ps(tempCurrent,val.m);
                    }
#else
                    coord=0;
                    for(a=0;a<16;a++){
                        xyzBasis[coord++]=temp[0]*yzBasis[a];
                        xyzBasis[coord++]=temp[1]*yzBasis[a];
                        xyzBasis[coord++]=temp[2]*yzBasis[a];
                        xyzBasis[coord++]=temp[3]*yzBasis[a];
                    }
#endif
                    if(basis<=oldBasis || x==0){
#ifdef _USE_SSE
                        get_GridValues<DTYPE>(xPre,
                                              yPre,
                                              zPre,
                                              splineControlPoint,
                                              controlPointPtrX,
                                              controlPointPtrY,
                                              controlPointPtrZ,
                                              xControlPointCoordinates.f,
                                              yControlPointCoordinates.f,
                                              zControlPointCoordinates.f,
                                              false);
#else // _USE_SSE
                        get_GridValues<DTYPE>(xPre,
                                              yPre,
                                              zPre,
                                              splineControlPoint,
                                              controlPointPtrX,
                                              controlPointPtrY,
                                              controlPointPtrZ,
                                              xControlPointCoordinates,
                                              yControlPointCoordinates,
                                              zControlPointCoordinates,
                                              false);
#endif // _USE_SSE
                    }
                    oldBasis=basis;

                    real[0]=0.0;
                    real[1]=0.0;
                    real[2]=0.0;

                    if(mask[index]>-1){
#if _USE_SSE
                        tempX =  _mm_set_ps1(0.0);
                        tempY =  _mm_set_ps1(0.0);
                        tempZ =  _mm_set_ps1(0.0);
                        //addition and multiplication of the 64 basis value and CP displacement for each axis
                        for(a=0; a<16; a++){
                            tempX = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                            tempY = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], yControlPointCoordinates.m[a]), tempY );
                            tempZ = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], zControlPointCoordinates.m[a]), tempZ );
                        }
                        //the values stored in SSE variables are transfered to normal float
                        val.m=tempX;
                        real[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m=tempY;
                        real[1]= val.f[0]+val.f[1]+val.f[2]+val.f[3];
                        val.m=tempZ;
                        real[2]= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                        for(a=0; a<64; a++){
                            real[0] += xControlPointCoordinates[a] * xyzBasis[a];
                            real[1] += yControlPointCoordinates[a] * xyzBasis[a];
                            real[2] += zControlPointCoordinates[a] * xyzBasis[a];
                        }
#endif
                    }// mask
                    fieldPtrX[index] = (DTYPE)real[0];
                    fieldPtrY[index] = (DTYPE)real[1];
                    fieldPtrZ[index] = (DTYPE)real[2];
                    index++;
                } // x
            } // y
        } // z
    }// from a deformation field
    
    return;
}
/* *************************************************************** */
void reg_spline_getDeformationField(nifti_image *splineControlPoint,
                                    nifti_image *referenceImage,
                                    nifti_image *deformationField,
                                    int *mask,
                                    bool composition,
                                    bool bspline)
{
    if(splineControlPoint->datatype != deformationField->datatype){
        fprintf(stderr,"[NiftyReg ERROR] The spline control point image and the deformation field image are expected to be the same type\n");
        fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
        exit(1);
    }

#if _USE_SSE
    if(splineControlPoint->datatype != NIFTI_TYPE_FLOAT32){
        fprintf(stderr,"[NiftyReg ERROR] SSE computation has only been implemented for single precision.\n");
        fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
        exit(1);
    }
#endif

    bool MrPropre=false;
    if(mask==NULL){
        // Active voxel are all superior to -1, 0 thus will do !
        MrPropre=true;
        mask=(int *)calloc(deformationField->nx*deformationField->ny*deformationField->nz, sizeof(int));
    }

    if(splineControlPoint->nz==1){
        switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_spline_getDeformationField2D<float>(splineControlPoint, referenceImage, deformationField, mask, composition, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_getDeformationField2D<double>(splineControlPoint, referenceImage, deformationField, mask, composition, bspline);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for deformation field\n");
            fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
            exit(1);
        }
    }
    else{
        switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_spline_getDeformationField3D<float>(splineControlPoint, referenceImage, deformationField, mask, composition, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_getDeformationField3D<double>(splineControlPoint, referenceImage, deformationField, mask, composition, bspline);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for deformation field\n");
            fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
            exit(1);
        }
    }
    if(MrPropre==true) free(mask);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_voxelCentric2NodeCentric2D(nifti_image *nodeImage,
                                    nifti_image *voxelImage,
                                    float weight,
                                    bool update
                                    )
{
    DTYPE *nodePtrX = static_cast<DTYPE *>(nodeImage->data);
    DTYPE *nodePtrY = &nodePtrX[nodeImage->nx*nodeImage->ny];

    DTYPE *voxelPtrX = static_cast<DTYPE *>(voxelImage->data);
    DTYPE *voxelPtrY = &voxelPtrX[voxelImage->nx*voxelImage->ny];

    float ratio[2];
    ratio[0] = nodeImage->dx / voxelImage->dx;
    ratio[1] = nodeImage->dy / voxelImage->dy;

    for(int y=0;y<nodeImage->ny; y++){
        int Y = (int)reg_round((float)(y-1) * ratio[1]);
        Y=Y<0?0:Y;
        Y=Y>voxelImage->ny-1?voxelImage->ny-1:Y;
        DTYPE *yVoxelPtrX=&voxelPtrX[Y*voxelImage->nx];
        DTYPE *yVoxelPtrY=&voxelPtrY[Y*voxelImage->nx];
        for(int x=0;x<nodeImage->nx; x++){
            int X = (int)reg_round((float)(x-1) * ratio[0]);
            X=X<0?0:X;
            X=X>voxelImage->nx-1?voxelImage->nx-1:X;
//            if( -1<Y && Y<voxelImage->ny && -1<X && X<voxelImage->nx){
                if(update){
                    *nodePtrX += (DTYPE)(yVoxelPtrX[X] * weight);
                    *nodePtrY += (DTYPE)(yVoxelPtrY[X] * weight);
                }
                else{
                    *nodePtrX = (DTYPE)(yVoxelPtrX[X] * weight);
                    *nodePtrY = (DTYPE)(yVoxelPtrY[X] * weight);
                }
//            }
//            else{
//                if(!update){
//                    *nodePtrX = 0.0;
//                    *nodePtrY = 0.0;
//                }
//            }
                ++nodePtrX;++nodePtrY;
        }
    }
}
/* *************************************************************** */
template<class DTYPE>
void reg_voxelCentric2NodeCentric3D(nifti_image *nodeImage,
                                    nifti_image *voxelImage,
                                    float weight,
                                    bool update
                                    )
{
    DTYPE *nodePtrX = static_cast<DTYPE *>(nodeImage->data);
    DTYPE *nodePtrY = &nodePtrX[nodeImage->nx*nodeImage->ny*nodeImage->nz];
    DTYPE *nodePtrZ = &nodePtrY[nodeImage->nx*nodeImage->ny*nodeImage->nz];

    DTYPE *voxelPtrX = static_cast<DTYPE *>(voxelImage->data);
    DTYPE *voxelPtrY = &voxelPtrX[voxelImage->nx*voxelImage->ny*voxelImage->nz];
    DTYPE *voxelPtrZ = &voxelPtrY[voxelImage->nx*voxelImage->ny*voxelImage->nz];

    float ratio[3];
    ratio[0] = nodeImage->dx / voxelImage->dx;
    ratio[1] = nodeImage->dy / voxelImage->dy;
    ratio[2] = nodeImage->dz / voxelImage->dz;

    for(int z=0;z<nodeImage->nz; z++){
        int Z = (int)reg_round((float)(z-1) * ratio[2]);
        // sliding condition-ish along the Z-axis
        Z=Z<0?0:Z;
        Z=Z>=voxelImage->nz?voxelImage->nz-1:Z;
        DTYPE *zvoxelPtrX=&voxelPtrX[Z*voxelImage->nx*voxelImage->ny];
        DTYPE *zvoxelPtrY=&voxelPtrY[Z*voxelImage->nx*voxelImage->ny];
        DTYPE *zvoxelPtrZ=&voxelPtrZ[Z*voxelImage->nx*voxelImage->ny];
        for(int y=0;y<nodeImage->ny; y++){
            int Y = (int)reg_round((float)(y-1) * ratio[1]);
            // sliding condition-ish along the Y-axis
            Y=Y<0?0:Y;
            Y=Y>=voxelImage->ny?voxelImage->ny-1:Y;
            DTYPE *yzvoxelPtrX=&zvoxelPtrX[Y*voxelImage->nx];
            DTYPE *yzvoxelPtrY=&zvoxelPtrY[Y*voxelImage->nx];
            DTYPE *yzvoxelPtrZ=&zvoxelPtrZ[Y*voxelImage->nx];
            for(int x=0;x<nodeImage->nx; x++){
                int X = (int)reg_round((float)(x-1) * ratio[0]);
                // sliding condition-ish along the X-axis
                X=X<0?0:X;
                X=X>=voxelImage->nx?voxelImage->nx-1:X;
//                if(-1<Z && Z<voxelImage->nz && -1<Y && Y<voxelImage->ny && -1<X && X<voxelImage->nx){
                    if(update){
                        *nodePtrX += (DTYPE)(yzvoxelPtrX[X]*weight);
                        *nodePtrY += (DTYPE)(yzvoxelPtrY[X]*weight);
                        *nodePtrZ += (DTYPE)(yzvoxelPtrZ[X]*weight);
                    }
                    else{
                        *nodePtrX = (DTYPE)(yzvoxelPtrX[X]*weight);
                        *nodePtrY = (DTYPE)(yzvoxelPtrY[X]*weight);
                        *nodePtrZ = (DTYPE)(yzvoxelPtrZ[X]*weight);
                    }
//                }
//                else{
//                    if(!update){
//                        *nodePtrX = 0.0;
//                        *nodePtrY = 0.0;
//                        *nodePtrZ = 0.0;
//                    }
//                }
                ++nodePtrX;++nodePtrY;++nodePtrZ;
            }
        }
    }
}
/* *************************************************************** */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight,
                                  bool update
                                  )
{
    if(nodeImage->datatype!=voxelImage->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_voxelCentric2NodeCentric\n");
        fprintf(stderr, "[NiftyReg ERROR] Both input images do not have the same type\n");
        exit(1);
    }
    // it is assumed than node[000] and voxel[000] are aligned.
    if(nodeImage->nz==1){
        switch(nodeImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_voxelCentric2NodeCentric2D<float>(nodeImage, voxelImage, weight, update);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_voxelCentric2NodeCentric2D<double>(nodeImage, voxelImage, weight, update);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_voxelCentric2NodeCentric:\tdata type not supported\n");
            exit(1);
        }
    }
    else{
        switch(nodeImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_voxelCentric2NodeCentric3D<float>(nodeImage, voxelImage, weight, update);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_voxelCentric2NodeCentric3D<double>(nodeImage, voxelImage, weight, update);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_voxelCentric2NodeCentric:\tdata type not supported\n");
            exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
SplineTYPE GetValue(SplineTYPE *array, int *dim, int x, int y, int z)
{
    if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
        return 0.0;
    return array[(z*dim[2]+y)*dim[1]+x];
}
/* *************************************************************** */
template<class SplineTYPE>
void SetValue(SplineTYPE *array, int *dim, int x, int y, int z, SplineTYPE value)
{
    if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
        return;
    array[(z*dim[2]+y)*dim[1]+x] = value;
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_bspline_refineControlPointGrid2D(  nifti_image *targetImage,
                                          nifti_image *splineControlPoint)
{
    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=1;

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = 1.0f;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    //    splineControlPoint->dim[1]=splineControlPoint->nx=(int)ceil(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    //    splineControlPoint->dim[2]=splineControlPoint->ny=(int)ceil(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
    splineControlPoint->dim[3]=1;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);

    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]];

    for(int y=0; y<oldDim[2]; y++){
        int Y=2*y-1;
        if(Y<splineControlPoint->ny){
            for(int x=0; x<oldDim[1]; x++){
                int X=2*x-1;
                if(X<splineControlPoint->nx){

                    /* X Axis */
                    // 0 0
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrX,oldDim,x-1,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrX,oldDim,x-1,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                                        GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) )
                              + 36.0f * GetValue(oldGridPtrX,oldDim,x,y,0) ) / 64.0f);
                    // 1 0
                    SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, 0,
                             (GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) ) ) / 16.0f);
                    // 0 1
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, 0,
                             (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x-1,y+1,0) +
                              GetValue(oldGridPtrX,oldDim,x+1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) ) ) / 16.0f);
                    // 1 1
                    SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, 0,
                             (GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                              GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0) ) / 4.0f);

                    /* Y Axis */
                    // 0 0
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrY,oldDim,x-1,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrY,oldDim,x-1,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                                        GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) )
                              + 36.0f * GetValue(oldGridPtrY,oldDim,x,y,0) ) / 64.0f);
                    // 1 0
                    SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, 0,
                             (GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) ) ) / 16.0f);
                    // 0 1
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, 0,
                             (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x-1,y+1,0) +
                              GetValue(oldGridPtrY,oldDim,x+1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) ) ) / 16.0f);
                    // 1 1
                    SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, 0,
                             (GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                              GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0) ) / 4.0f);

                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_bspline_refineControlPointGrid3D(nifti_image *targetImage,
                                          nifti_image *splineControlPoint)
{

    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[0]=splineControlPoint->dim[0];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=splineControlPoint->dim[3];

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = splineControlPoint->pixdim[3] = splineControlPoint->dz / 2.0f;

    //    splineControlPoint->dim[1]=splineControlPoint->nx=(int)ceil(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    //    splineControlPoint->dim[2]=splineControlPoint->ny=(int)ceil(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
    //    splineControlPoint->dim[3]=splineControlPoint->nz=(int)ceil(targetImage->nz*targetImage->dz/splineControlPoint->dz)+4;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    splineControlPoint->dim[3]=splineControlPoint->nz=(int)floor(targetImage->nz*targetImage->dz/splineControlPoint->dz)+5;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);
    
    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *gridPtrZ = &gridPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]*oldDim[3]];
    SplineTYPE *oldGridPtrZ = &oldGridPtrY[oldDim[1]*oldDim[2]*oldDim[3]];


    for(int z=0; z<oldDim[3]; z++){
        int Z=2*z-1;
        if(Z<splineControlPoint->nz){
            for(int y=0; y<oldDim[2]; y++){
                int Y=2*y-1;
                if(Y<splineControlPoint->ny){
                    for(int x=0; x<oldDim[1]; x++){
                        int X=2*x-1;
                        if(X<splineControlPoint->nx){

                            /* X Axis */
                            // 0 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrX,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrX,oldDim,x,y,z) ) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z)) ) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x-1,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z)) ) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) ) ) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y,z+1)) ) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) ) ) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) ) ) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)) / 8.0f);
                            

                            /* Y Axis */
                            // 0 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrY,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrY,oldDim,x,y,z) ) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z)) ) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x-1,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z)) ) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) ) ) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y,z+1)) ) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) ) ) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) ) ) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)) / 8.0f);

                            /* Z Axis */
                            // 0 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrZ,oldDim,x,y,z) ) / 512.0f);
                            
                            // 1 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z)) ) / 128.0f);
                            
                            // 0 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z)) ) / 128.0f);
                            
                            // 1 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) ) ) / 32.0f);
                            
                            // 0 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y,z+1)) ) / 128.0f);
                            
                            // 1 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) ) ) / 32.0f);
                            
                            // 0 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) ) ) / 32.0f);
                            
                            // 1 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)) / 8.0f);
                        }
                    }
                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
extern "C++"
void reg_bspline_refineControlPointGrid(nifti_image *referenceImage,
                                        nifti_image *controlPointGrid)
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Starting the refine the control point grid\n");
#endif
    if(controlPointGrid->nz==1){
        switch(controlPointGrid->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_refineControlPointGrid2D<float>(referenceImage,controlPointGrid);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_refineControlPointGrid2D<double>(referenceImage,controlPointGrid);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
            exit(1);
        }
    }else{
        switch(controlPointGrid->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_refineControlPointGrid3D<float>(referenceImage,controlPointGrid);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_refineControlPointGrid3D<double>(referenceImage,controlPointGrid);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
            exit(1);
        }
    }
    // Compute the new control point header
    // The qform (and sform) are set for the control point position image
    controlPointGrid->quatern_b=referenceImage->quatern_b;
    controlPointGrid->quatern_c=referenceImage->quatern_c;
    controlPointGrid->quatern_d=referenceImage->quatern_d;
    controlPointGrid->qoffset_x=referenceImage->qoffset_x;
    controlPointGrid->qoffset_y=referenceImage->qoffset_y;
    controlPointGrid->qoffset_z=referenceImage->qoffset_z;
    controlPointGrid->qfac=referenceImage->qfac;
    controlPointGrid->qto_xyz = nifti_quatern_to_mat44(controlPointGrid->quatern_b,
                                                       controlPointGrid->quatern_c,
                                                       controlPointGrid->quatern_d,
                                                       controlPointGrid->qoffset_x,
                                                       controlPointGrid->qoffset_y,
                                                       controlPointGrid->qoffset_z,
                                                       controlPointGrid->dx,
                                                       controlPointGrid->dy,
                                                       controlPointGrid->dz,
                                                       controlPointGrid->qfac);

    // Origin is shifted from 1 control point in the qform
    float originIndex[3];
    float originReal[3];
    originIndex[0] = -1.0f;
    originIndex[1] = -1.0f;
    originIndex[2] = 0.0f;
    if(referenceImage->nz>1) originIndex[2] = -1.0f;
    reg_mat44_mul(&(controlPointGrid->qto_xyz), originIndex, originReal);
    if(controlPointGrid->qform_code==0) controlPointGrid->qform_code=1;
    controlPointGrid->qto_xyz.m[0][3] = controlPointGrid->qoffset_x = originReal[0];
    controlPointGrid->qto_xyz.m[1][3] = controlPointGrid->qoffset_y = originReal[1];
    controlPointGrid->qto_xyz.m[2][3] = controlPointGrid->qoffset_z = originReal[2];

    controlPointGrid->qto_ijk = nifti_mat44_inverse(controlPointGrid->qto_xyz);

    if(controlPointGrid->sform_code>0){
        float scalingRatio[3];
        scalingRatio[0]= controlPointGrid->dx / referenceImage->dx;
        scalingRatio[1]= controlPointGrid->dy / referenceImage->dy;
        scalingRatio[2]= controlPointGrid->dz / referenceImage->dz;

        controlPointGrid->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[3][0]=0.f;
        controlPointGrid->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[3][1]=0.f;
        controlPointGrid->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[3][2]=0.f;
        controlPointGrid->sto_xyz.m[0][3]=referenceImage->sto_xyz.m[0][3];
        controlPointGrid->sto_xyz.m[1][3]=referenceImage->sto_xyz.m[1][3];
        controlPointGrid->sto_xyz.m[2][3]=referenceImage->sto_xyz.m[2][3];
        controlPointGrid->sto_xyz.m[3][3]=1.f;

        // The origin is shifted by one compare to the reference image
        float originIndex[3];originIndex[0]=originIndex[1]=originIndex[2]=-1;
        if(referenceImage->nz<=1) originIndex[2]=0;
        reg_mat44_mul(&(controlPointGrid->sto_xyz), originIndex, originReal);
        controlPointGrid->sto_xyz.m[0][3] = originReal[0];
        controlPointGrid->sto_xyz.m[1][3] = originReal[1];
        controlPointGrid->sto_xyz.m[2][3] = originReal[2];
        controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] The control point grid has been refined\n");
#endif
    return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_initialiseControlPointGridWithAffine2D(mat44 *affineTransformation,
                                                        nifti_image *controlPointImage)
{
    DTYPE *CPPX=static_cast<DTYPE *>(controlPointImage->data);
    DTYPE *CPPY=&CPPX[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];

    mat44 *cppMatrix;
    if(controlPointImage->sform_code>0)
        cppMatrix=&(controlPointImage->sto_xyz);
    else cppMatrix=&(controlPointImage->qto_xyz);

    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, cppMatrix);

    float index[3];
    float position[3];
    index[2]=0;
    for(int y=0; y<controlPointImage->ny; y++){
        index[1]=(float)y;
        for(int x=0; x<controlPointImage->nx; x++){
            index[0]=(float)x;

            reg_mat44_mul(&voxelToRealDeformed, index, position);

            *CPPX++ = position[0];
            *CPPY++ = position[1];
        }
    }
}
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_initialiseControlPointGridWithAffine3D(	mat44 *affineTransformation,
							nifti_image *controlPointImage)
{
    DTYPE *CPPX=static_cast<DTYPE *>(controlPointImage->data);
    DTYPE *CPPY=&CPPX[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];
    DTYPE *CPPZ=&CPPY[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];

    mat44 *cppMatrix;
    if(controlPointImage->sform_code>0)
        cppMatrix=&(controlPointImage->sto_xyz);
    else cppMatrix=&(controlPointImage->qto_xyz);

    mat44 voxelToRealDeformed = reg_mat44_mul(affineTransformation, cppMatrix);

    float index[3];
    float position[3];
    for(int z=0; z<controlPointImage->nz; z++){
        index[2]=(float)z;
        for(int y=0; y<controlPointImage->ny; y++){
            index[1]=(float)y;
            for(int x=0; x<controlPointImage->nx; x++){
                index[0]=(float)x;

                reg_mat44_mul(&voxelToRealDeformed, index, position);

                *CPPX++ = position[0];
                *CPPY++ = position[1];
                *CPPZ++ = position[2];
            }
        }
    }
}
/* *************************************************************** */
int reg_bspline_initialiseControlPointGridWithAffine(mat44 *affineTransformation,
                                                     nifti_image *controlPointImage)
{
    if(controlPointImage->nz==1){
        switch(controlPointImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_initialiseControlPointGridWithAffine2D<float>(affineTransformation, controlPointImage);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_initialiseControlPointGridWithAffine2D<double>(affineTransformation, controlPointImage);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_bspline_initialiseControlPointGridWithAffine\n");
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
            exit(1);
        }
    }
    else{
        switch(controlPointImage->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_initialiseControlPointGridWithAffine3D<float>(affineTransformation, controlPointImage);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_initialiseControlPointGridWithAffine3D<double>(affineTransformation, controlPointImage);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_bspline_initialiseControlPointGridWithAffine\n");
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the control point image\n");
            exit(1);
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_2D(nifti_image *splineControlPoint)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    int x, y,  index;
    DTYPE xInit, yInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        shared(splineControlPoint, splineMatrix, \
        controlPointPtrX, controlPointPtrY) \
        private(x, y, index, xInit, yInit)
#endif
    for(y=0; y<splineControlPoint->ny; y++){
        index=y*splineControlPoint->nx;
        for(x=0; x<splineControlPoint->nx; x++){

            // Get the initial control point position
            xInit = splineMatrix->m[0][0]*(DTYPE)x
                    + splineMatrix->m[0][1]*(DTYPE)y
                    + splineMatrix->m[0][3];
            yInit = splineMatrix->m[1][0]*(DTYPE)x
                    + splineMatrix->m[1][1]*(DTYPE)y
                    + splineMatrix->m[1][3];

            // The initial position is subtracted from every values
            controlPointPtrX[index] -= xInit;
            controlPointPtrY[index] -= yInit;
            index++;
        }
    }
}
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_3D(nifti_image *splineControlPoint)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    int x, y, z, index;
    DTYPE xInit, yInit, zInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        shared(splineControlPoint, splineMatrix, \
        controlPointPtrX, controlPointPtrY, controlPointPtrZ) \
        private(x, y, z, index, xInit, yInit, zInit)
#endif
    for(z=0; z<splineControlPoint->nz; z++){
        index=z*splineControlPoint->nx*splineControlPoint->ny;
        for(y=0; y<splineControlPoint->ny; y++){
            for(x=0; x<splineControlPoint->nx; x++){

                // Get the initial control point position
                xInit = splineMatrix->m[0][0]*(DTYPE)x
                        + splineMatrix->m[0][1]*(DTYPE)y
                        + splineMatrix->m[0][2]*(DTYPE)z
                        + splineMatrix->m[0][3];
                yInit = splineMatrix->m[1][0]*(DTYPE)x
                        + splineMatrix->m[1][1]*(DTYPE)y
                        + splineMatrix->m[1][2]*(DTYPE)z
                        + splineMatrix->m[1][3];
                zInit = splineMatrix->m[2][0]*(DTYPE)x
                        + splineMatrix->m[2][1]*(DTYPE)y
                        + splineMatrix->m[2][2]*(DTYPE)z
                        + splineMatrix->m[2][3];

                // The initial position is subtracted from every values
                controlPointPtrX[index] -= xInit;
                controlPointPtrY[index] -= yInit;
                controlPointPtrZ[index] -= zInit;
                index++;
            }
        }
    }
}
/* *************************************************************** */
int reg_getDisplacementFromDeformation(nifti_image *splineControlPoint)
{
    if(splineControlPoint->datatype==NIFTI_TYPE_FLOAT32){
        switch(splineControlPoint->nu){
        case 2:
            reg_getDisplacementFromDeformation_2D<float>(splineControlPoint);
            break;
        case 3:
            reg_getDisplacementFromDeformation_3D<float>(splineControlPoint);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getDisplacementFromPosition<float>\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for 5D image\n");
            fprintf(stderr,"[NiftyReg ERROR] with 2 or 3 components in the fifth dimension\n");
            return 1;
        }
    }
    else if(splineControlPoint->datatype==NIFTI_TYPE_FLOAT64){
        switch(splineControlPoint->nu){
        case 2:
            reg_getDisplacementFromDeformation_2D<double>(splineControlPoint);
            break;
        case 3:
            reg_getDisplacementFromDeformation_3D<double>(splineControlPoint);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getDisplacementFromPosition<double>\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for 5D image\n");
            fprintf(stderr,"[NiftyReg ERROR] with 2 or 3 components in the fifth dimension\n");
            return 1;
        }
    }
    else{
        fprintf(stderr,"[NiftyReg ERROR] reg_getDisplacementFromPosition\n");
        fprintf(stderr,"[NiftyReg ERROR] Only single or double floating precision have been implemented. EXIT\n");
        exit(1);
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_2D(nifti_image *splineControlPoint)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);


    int x, y, index;
    DTYPE xInit, yInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        shared(splineControlPoint, splineMatrix, \
        controlPointPtrX, controlPointPtrY) \
        private(x, y, index, xInit, yInit)
#endif
    for(y=0; y<splineControlPoint->ny; y++){
        index=y*splineControlPoint->nx;
        for(x=0; x<splineControlPoint->nx; x++){

            // Get the initial control point position
            xInit = splineMatrix->m[0][0]*(DTYPE)x
                    + splineMatrix->m[0][1]*(DTYPE)y
                    + splineMatrix->m[0][3];
            yInit = splineMatrix->m[1][0]*(DTYPE)x
                    + splineMatrix->m[1][1]*(DTYPE)y
                    + splineMatrix->m[1][3];

            // The initial position is added from every values
            controlPointPtrX[index] += xInit;
            controlPointPtrY[index] += yInit;
            index++;
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_3D(nifti_image *splineControlPoint)
{
    DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

    mat44 *splineMatrix;
    if(splineControlPoint->sform_code>0) splineMatrix=&(splineControlPoint->sto_xyz);
    else splineMatrix=&(splineControlPoint->qto_xyz);

    int x, y, z, index;
    DTYPE xInit, yInit, zInit;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
        shared(splineControlPoint, splineMatrix, \
        controlPointPtrX, controlPointPtrY, controlPointPtrZ) \
        private(x, y, z, index, xInit, yInit, zInit)
#endif
    for(z=0; z<splineControlPoint->nz; z++){
        index=z*splineControlPoint->nx*splineControlPoint->ny;
        for(y=0; y<splineControlPoint->ny; y++){
            for(x=0; x<splineControlPoint->nx; x++){

                // Get the initial control point position
                xInit = splineMatrix->m[0][0]*(DTYPE)x
                        + splineMatrix->m[0][1]*(DTYPE)y
                        + splineMatrix->m[0][2]*(DTYPE)z
                        + splineMatrix->m[0][3];
                yInit = splineMatrix->m[1][0]*(DTYPE)x
                        + splineMatrix->m[1][1]*(DTYPE)y
                        + splineMatrix->m[1][2]*(DTYPE)z
                        + splineMatrix->m[1][3];
                zInit = splineMatrix->m[2][0]*(DTYPE)x
                        + splineMatrix->m[2][1]*(DTYPE)y
                        + splineMatrix->m[2][2]*(DTYPE)z
                        + splineMatrix->m[2][3];

                // The initial position is subtracted from every values
                controlPointPtrX[index] += xInit;
                controlPointPtrY[index] += yInit;
                controlPointPtrZ[index] += zInit;
                index++;
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
int reg_getDeformationFromDisplacement(nifti_image *splineControlPoint)
{
    if(splineControlPoint->datatype==NIFTI_TYPE_FLOAT32){
        switch(splineControlPoint->nu){
        case 2:
            reg_getDeformationFromDisplacement_2D<float>(splineControlPoint);
            break;
        case 3:
            reg_getDeformationFromDisplacement_3D<float>(splineControlPoint);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getDeformationFromDisplacement\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for 2 or 3D deformation fields. EXIT\n");
            exit(1);
        }
    }
    else if(splineControlPoint->datatype==NIFTI_TYPE_FLOAT64){
        switch(splineControlPoint->nu){
        case 2:
            reg_getDeformationFromDisplacement_2D<double>(splineControlPoint);
            break;
        case 3:
            reg_getDeformationFromDisplacement_3D<double>(splineControlPoint);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_getDeformationFromDisplacement\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for 2 or 3D deformation fields. EXIT\n");
            exit(1);
        }
    }
    else{
        fprintf(stderr,"[NiftyReg ERROR] reg_getPositionFromDisplacement\n");
        fprintf(stderr,"[NiftyReg ERROR] Only single or double floating precision have been implemented. EXIT\n");
        exit(1);
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_defField_compose2D(nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            int *mask)
{
    int DFVoxelNumber=deformationField->nx*deformationField->ny;
    int resVoxelNumber=dfToUpdate->nx*dfToUpdate->ny;
    DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *defPtrY = &defPtrX[DFVoxelNumber];

    DTYPE *resPtrX = static_cast<DTYPE *>(dfToUpdate->data);
    DTYPE *resPtrY = &resPtrX[resVoxelNumber];

    mat44 *df_real2Voxel=NULL;
    mat44 *df_voxel2Real=NULL;
    if(deformationField->sform_code>0){
        df_real2Voxel=&(dfToUpdate->sto_ijk);
        df_voxel2Real=&(deformationField->sto_xyz);
    }
    else{
        df_real2Voxel=&(dfToUpdate->qto_ijk);
        df_voxel2Real=&(deformationField->qto_xyz);
    }

    int i, a, b, index, pre[2];
    DTYPE realDefX, realDefY, voxelX, voxelY;
    DTYPE defX, defY, relX[2], relY[2], basis;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(resVoxelNumber, mask, df_real2Voxel, df_voxel2Real, \
    deformationField, defPtrX, defPtrY, resPtrX, resPtrY) \
    private(i, a, b, index, pre,realDefX, realDefY, voxelX, voxelY, \
    defX, defY, relX, relY, basis)
#endif
    for(i=0;i<resVoxelNumber;++i){
        if(mask[i]>-1){
            realDefX = resPtrX[i];
            realDefY = resPtrY[i];

            // Conversion from real to voxel in the deformation field
            voxelX = realDefX * df_real2Voxel->m[0][0]
                    + realDefY * df_real2Voxel->m[0][1]
                    + df_real2Voxel->m[0][3];
            voxelY = realDefX * df_real2Voxel->m[1][0]
                    + realDefY * df_real2Voxel->m[1][1]
                    + df_real2Voxel->m[1][3];

            // Linear interpolation to compute the new deformation
            pre[0]=(int)floor(voxelX);
            pre[1]=(int)floor(voxelY);
            relX[1]=voxelX-(DTYPE)pre[0];relX[0]=1.f-relX[1];
            relY[1]=voxelY-(DTYPE)pre[1];relY[0]=1.f-relY[1];
            realDefX=realDefY=0.f;
            for(b=0;b<2;++b){
                for(a=0;a<2;++a){
                    basis = relX[a] * relY[b];
                    if(pre[0]+a>-1 && pre[0]+a<deformationField->nx &&
                            pre[1]+b>-1 && pre[1]+b<deformationField->ny){
                        // Uses the deformation field if voxel is in its space
                        index=(pre[1]+b)*deformationField->nx+pre[0]+a;
                        defX = defPtrX[index];
                        defY = defPtrY[index];
                    }
                    else{
                        // Uses the deformation field affine transformation
                        defX = (pre[0]+a) * df_voxel2Real->m[0][0]
                                + (pre[1]+b) * df_voxel2Real->m[0][1]
                                + df_voxel2Real->m[0][3];
                        defY = (pre[0]+a) * df_voxel2Real->m[1][0]
                                + (pre[1]+b) * df_voxel2Real->m[1][1]
                                + df_voxel2Real->m[1][3];
                    }
                    realDefX += defX * basis;
                    realDefY += defY * basis;
                }
            }
            resPtrX[i]=realDefX;
            resPtrY[i]=realDefY;
        }// mask
    }// loop over every voxel
}
/* *************************************************************** */
template <class DTYPE>
void reg_defField_compose3D(nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            int *mask)
{
    const int DefFieldDim[3]={deformationField->nx,deformationField->ny,deformationField->nz};
    const size_t DFVoxelNumber=DefFieldDim[0]*DefFieldDim[1]*DefFieldDim[2];
    size_t resVoxelNumber=dfToUpdate->nx*dfToUpdate->ny*dfToUpdate->nz;

    DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
    DTYPE *defPtrY = &defPtrX[DFVoxelNumber];
    DTYPE *defPtrZ = &defPtrY[DFVoxelNumber];

    DTYPE *resPtrX = static_cast<DTYPE *>(dfToUpdate->data);
    DTYPE *resPtrY = &resPtrX[resVoxelNumber];
    DTYPE *resPtrZ = &resPtrY[resVoxelNumber];

    mat44 df_real2Voxel;
    mat44 df_voxel2Real;
    if(deformationField->sform_code>0){
        df_real2Voxel=deformationField->sto_ijk;
        df_voxel2Real=deformationField->sto_xyz;
    }
    else{
        df_real2Voxel=deformationField->qto_ijk;
        df_voxel2Real=deformationField->qto_xyz;
    }
#ifdef _WIN32
    long i;
#else
    size_t i;
#endif

    short a, b, c;
    int currentX, currentY, currentZ, pre[3], index[3];
    DTYPE realDefX, realDefY, realDefZ, voxelX, voxelY, voxelZ, tempBasis;
    DTYPE defX, defY, defZ, relX[2], relY[2], relZ[2], basis;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(resVoxelNumber, mask, df_real2Voxel, df_voxel2Real, DefFieldDim, \
    defPtrX, defPtrY, defPtrZ, resPtrX, resPtrY, resPtrZ) \
    private(i, a, b, c, currentX, currentY, currentZ, index, pre, \
    realDefX, realDefY, realDefZ, voxelX, voxelY, voxelZ, tempBasis, \
    defX, defY, defZ, relX, relY, relZ, basis)
#endif
    for(i=0;i<resVoxelNumber;++i){
        if(mask[i]>-1){
            realDefX = resPtrX[i];
            realDefY = resPtrY[i];
            realDefZ = resPtrZ[i];

            // Conversion from real to voxel in the deformation field
            voxelX = realDefX * df_real2Voxel.m[0][0]
                    + realDefY * df_real2Voxel.m[0][1]
                    + realDefZ * df_real2Voxel.m[0][2]
                    + df_real2Voxel.m[0][3];
            voxelY = realDefX * df_real2Voxel.m[1][0]
                    + realDefY * df_real2Voxel.m[1][1]
                    + realDefZ * df_real2Voxel.m[1][2]
                    + df_real2Voxel.m[1][3];
            voxelZ = realDefX * df_real2Voxel.m[2][0]
                    + realDefY * df_real2Voxel.m[2][1]
                    + realDefZ * df_real2Voxel.m[2][2]
                    + df_real2Voxel.m[2][3];

            // Linear interpolation to compute the new deformation
            pre[0]=(int)floor(voxelX);
            pre[1]=(int)floor(voxelY);
            pre[2]=(int)floor(voxelZ);
            relX[1]=voxelX-(DTYPE)pre[0];relX[0]=1.-relX[1];
            relY[1]=voxelY-(DTYPE)pre[1];relY[0]=1.-relY[1];
            relZ[1]=voxelZ-(DTYPE)pre[2];relZ[0]=1.-relZ[1];
            realDefX=realDefY=realDefZ=0.;
            for(c=0;c<2;++c){
                currentZ = pre[2]+c;
                index[0]=currentZ*DefFieldDim[0]*DefFieldDim[1];
                for(b=0;b<2;++b){
                    currentY = pre[1]+b;
                    index[1]=index[0]+currentY*DefFieldDim[0];
                    tempBasis= relY[b] * relZ[c];
                    for(a=0;a<2;++a){
                        currentX = pre[0]+a;
                        if(currentX>-1 && currentX<DefFieldDim[0] &&
                           currentY>-1 && currentY<DefFieldDim[1] &&
                           currentZ>-1 && currentZ<DefFieldDim[2]){
                            // Uses the deformation field if voxel is in its space
                            index[2]=index[1]+currentX;
                            defX = defPtrX[index[2]];
                            defY = defPtrY[index[2]];
                            defZ = defPtrZ[index[2]];
                        }
                        else{
                            // Uses the deformation field affine transformation
                            defX = df_voxel2Real.m[0][0] * currentX
                                    + df_voxel2Real.m[0][1] * currentY
                                    + df_voxel2Real.m[0][2] * currentZ
                                    + df_voxel2Real.m[0][3];
                            defY = df_voxel2Real.m[1][0] * currentX
                                    + df_voxel2Real.m[1][1] * currentY
                                    + df_voxel2Real.m[1][2] * currentZ
                                    + df_voxel2Real.m[1][3];
                            defZ = df_voxel2Real.m[2][0] * currentX
                                    + df_voxel2Real.m[2][1] * currentY
                                    + df_voxel2Real.m[2][2] * currentZ
                                    + df_voxel2Real.m[2][3];
                        }
                        basis = relX[a] * tempBasis;
                        realDefX += defX * basis;
                        realDefY += defY * basis;
                        realDefZ += defZ * basis;
                    }
                }
            }
            resPtrX[i]=realDefX;
            resPtrY[i]=realDefY;
            resPtrZ[i]=realDefZ;
        }// mask
    }// loop over every voxel

}
/* *************************************************************** */
void reg_defField_compose(nifti_image *deformationField,
                          nifti_image *dfToUpdate,
                          int *mask)
{
    if(deformationField->datatype != dfToUpdate->datatype){
        fprintf(stderr, "[NiftyReg ERROR] reg_composeDefField\n");
        fprintf(stderr, "[NiftyReg ERROR] Both deformation fields are expected to have the same type. Exit\n");
        exit(1);
    }

    bool freeMask=false;
    if(mask==NULL){
        mask=(int *)calloc(dfToUpdate->nx*
                           dfToUpdate->ny*
                           dfToUpdate->nz,
                           sizeof(int));
        freeMask=true;
    }

    if(dfToUpdate->nu==2){
        switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_defField_compose2D<float>(deformationField,dfToUpdate,mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_defField_compose2D<double>(deformationField,dfToUpdate,mask);
            break;
        default:
            printf("[NiftyReg ERROR] reg_composeDefField2D\tDeformation field pixel type unsupported.");
            exit(1);
        }
    }
    else{
        switch(deformationField->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_defField_compose3D<float>(deformationField,dfToUpdate,mask);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_defField_compose3D<double>(deformationField,dfToUpdate,mask);
            break;
        default:
            printf("[NiftyReg ERROR] reg_composeDefField3D\tDeformation field pixel type unsupported.");
            exit(1);
        }
    }

    if(freeMask==true) free(mask);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_2D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline)
{
    // REMINDER Grid2(x)=Grid1(Grid2(x))

#if _USE_SSE
    union{
        __m128 m;
        float f[4];
    } val;
#endif // _USE_SSE

    DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
    DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny];

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny];

    DTYPE basis;

#ifdef _WINDOWS
    __declspec(align(16)) DTYPE xBasis[4];
    __declspec(align(16)) DTYPE yBasis[4];
    #if _USE_SSE
        __declspec(align(16)) DTYPE xyBasis[16];
    #endif  //_USE_SSE

    __declspec(align(16)) DTYPE xControlPointCoordinates[16];
    __declspec(align(16)) DTYPE yControlPointCoordinates[16];
#else // _WINDOWS
    DTYPE xBasis[4] __attribute__((aligned(16)));
    DTYPE yBasis[4] __attribute__((aligned(16)));
    #if _USE_SSE
        DTYPE xyBasis[16] __attribute__((aligned(16)));
    #endif  //_USE_SSE

    DTYPE xControlPointCoordinates[16] __attribute__((aligned(16)));
    DTYPE yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif // _WINDOWS

    unsigned int coord;

    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel1=NULL;
    mat44 *matrix_voxel_to_real2=NULL;
    if(grid1->sform_code>0)
        matrix_real_to_voxel1=&(grid1->sto_ijk);
    else matrix_real_to_voxel1=&(grid1->qto_ijk);
    if(grid2->sform_code>0)
        matrix_voxel_to_real2=&(grid2->sto_xyz);
    else matrix_voxel_to_real2=&(grid2->qto_xyz);

    for(int y=0; y<grid2->ny; y++){
        for(int x=0; x<grid2->nx; x++){

            // Get the control point actual position
            DTYPE xReal = *outCPPPtrX;
            DTYPE yReal = *outCPPPtrY;
            DTYPE initialX=xReal;
            DTYPE initialY=yReal;
            if(displacement2){
                xReal +=
                       matrix_voxel_to_real2->m[0][0]*x
                       + matrix_voxel_to_real2->m[0][1]*y
                       + matrix_voxel_to_real2->m[0][3];
                yReal +=
                       matrix_voxel_to_real2->m[1][0]*x
                       + matrix_voxel_to_real2->m[1][1]*y
                       + matrix_voxel_to_real2->m[1][3];
            }

            // Get the voxel based control point position in grid1
            DTYPE xVoxel = matrix_real_to_voxel1->m[0][0]*xReal
                    + matrix_real_to_voxel1->m[0][1]*yReal
                    + matrix_real_to_voxel1->m[0][3];
            DTYPE yVoxel = matrix_real_to_voxel1->m[1][0]*xReal
                    + matrix_real_to_voxel1->m[1][1]*yReal
                    + matrix_real_to_voxel1->m[1][3];

            // The spline coefficients are computed
            int xPre=(int)(floor(xVoxel));
            basis=(DTYPE)xVoxel-(DTYPE)xPre;xPre--;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) Get_BSplineBasisValues<DTYPE>(basis, xBasis);
            else Get_SplineBasisValues<DTYPE>(basis, xBasis);

            int yPre=(int)(floor(yVoxel));
            basis=(DTYPE)yVoxel-(DTYPE)yPre;yPre--;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) Get_BSplineBasisValues<DTYPE>(basis, yBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);

            // The control points are stored
            if(displacement1){
                get_GridValues<DTYPE>(xPre,
                                      yPre,
                                      grid1,
                                      controlPointPtrX,
                                      controlPointPtrY,
                                      xControlPointCoordinates,
                                      yControlPointCoordinates,
                                      false);
            }
            else{
                get_GridValues<DTYPE>(xPre,
                                      yPre,
                                      grid1,
                                      controlPointPtrX,
                                      controlPointPtrY,
                                      xControlPointCoordinates,
                                      yControlPointCoordinates,
                                      true);
            }
            xReal=0.0;
            yReal=0.0;
#if _USE_SSE
            coord=0;
            for(unsigned int b=0; b<4; b++){
                for(unsigned int a=0; a<4; a++){
                    xyBasis[coord++] = xBasis[a] * yBasis[b];
                }
            }

            __m128 tempX =  _mm_set_ps1(0.0);
            __m128 tempY =  _mm_set_ps1(0.0);
            __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
            __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
            __m128 *ptrBasis   = (__m128 *) &xyBasis[0];
            //addition and multiplication of the 16 basis value and CP position for each axis
            for(unsigned int a=0; a<4; a++){
                tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
                tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
                ptrBasis++;
                ptrX++;
                ptrY++;
            }
            //the values stored in SSE variables are transfered to normal float
            val.m = tempX;
            xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
            val.m = tempY;
            yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
            coord=0;
            for(unsigned int b=0; b<4; b++){
                for(unsigned int a=0; a<4; a++){
                    DTYPE tempValue = xBasis[a] * yBasis[b];
                    xReal += xControlPointCoordinates[coord] * tempValue;
                    yReal += yControlPointCoordinates[coord] * tempValue;
                    coord++;
                }
            }
#endif
            if(displacement1){
                xReal += initialX;
                yReal += initialY;
            }
            *outCPPPtrX++ = xReal;
            *outCPPPtrY++ = yReal;
        }
    }
    return;
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_3D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline)
{
    // REMINDER Grid2(x)=Grid1(Grid2(x))
#if _USE_SSE
    union{
        __m128 m;
        float f[4];
    } val;
    __m128 _xBasis_sse;
    __m128 tempX;
    __m128 tempY;
    __m128 tempZ;
    __m128 *ptrX;
    __m128 *ptrY;
    __m128 *ptrZ;
    __m128 _yBasis_sse;
    __m128 _zBasis_sse;
    __m128 _temp_basis;
    __m128 _basis;
#else
    int a, b, c, coord;
    DTYPE tempValue;
#endif

    DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
    DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny*grid2->nz];
    DTYPE *outCPPPtrZ = &outCPPPtrY[grid2->nx*grid2->ny*grid2->nz];

    DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
    DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny*grid1->nz];
    DTYPE *controlPointPtrZ = &controlPointPtrY[grid1->nx*grid1->ny*grid1->nz];

    DTYPE basis;

#ifdef _WINDOWS
    __declspec(align(16)) DTYPE xBasis[4];
    __declspec(align(16)) DTYPE yBasis[4];
    __declspec(align(16)) DTYPE zBasis[4];
    __declspec(align(16)) DTYPE xControlPointCoordinates[64];
    __declspec(align(16)) DTYPE yControlPointCoordinates[64];
    __declspec(align(16)) DTYPE zControlPointCoordinates[64];
#else
    DTYPE xBasis[4] __attribute__((aligned(16)));
    DTYPE yBasis[4] __attribute__((aligned(16)));
    DTYPE zBasis[4] __attribute__((aligned(16)));
    DTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
    DTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
    DTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif

    int xPre, xPreOld, yPre, yPreOld, zPre, zPreOld;
    int x, y, z;
    size_t index;
    DTYPE xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ;
    DTYPE xVoxel, yVoxel, zVoxel;

    // read the xyz/ijk sform or qform, as appropriate
    mat44 *matrix_real_to_voxel1=NULL;
    mat44 *matrix_voxel_to_real2=NULL;
    if(grid1->sform_code>0)
        matrix_real_to_voxel1=&(grid1->sto_ijk);
    else matrix_real_to_voxel1=&(grid1->qto_ijk);
    if(grid2->sform_code>0)
        matrix_voxel_to_real2=&(grid2->sto_xyz);
    else matrix_voxel_to_real2=&(grid2->qto_xyz);

#ifdef _OPENMP
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
    shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
    outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
    private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, val, index, \
    x, y, z, xVoxel, yVoxel, zVoxel, basis, xBasis, yBasis, zBasis, \
    xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
    _xBasis_sse, tempX, tempY, tempZ, ptrX, ptrY, ptrZ, _yBasis_sse, _zBasis_sse, _temp_basis, _basis, \
    xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#else
#pragma omp parallel for default(none) \
    shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
    outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
    private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, index, \
    x, y, z, xVoxel, yVoxel, zVoxel, a, b, c, coord, basis, tempValue, xBasis, yBasis, zBasis, \
    xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
    xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#endif
#endif
    for(z=0; z<grid2->nz; z++){
        xPreOld=99999;
        yPreOld=99999;
        zPreOld=99999;
        index=z*grid2->nx*grid2->ny;
        for(y=0; y<grid2->ny; y++){
            for(x=0; x<grid2->nx; x++){
                // Get the control point actual position
                xReal = outCPPPtrX[index];
                yReal = outCPPPtrY[index];
                zReal = outCPPPtrZ[index];
                initialPositionX=0;
                initialPositionY=0;
                initialPositionZ=0;
                if(displacement2){
                    xReal += initialPositionX =
                           matrix_voxel_to_real2->m[0][0]*x
                           + matrix_voxel_to_real2->m[0][1]*y
                           + matrix_voxel_to_real2->m[0][2]*z
                           + matrix_voxel_to_real2->m[0][3];
                    yReal += initialPositionY =
                           matrix_voxel_to_real2->m[1][0]*x
                           + matrix_voxel_to_real2->m[1][1]*y
                           + matrix_voxel_to_real2->m[1][2]*z
                           + matrix_voxel_to_real2->m[1][3];
                    zReal += initialPositionZ =
                           matrix_voxel_to_real2->m[2][0]*x
                           + matrix_voxel_to_real2->m[2][1]*y
                           + matrix_voxel_to_real2->m[2][2]*z
                           + matrix_voxel_to_real2->m[2][3];
                }

                // Get the voxel based control point position in grid1
                xVoxel =
                        matrix_real_to_voxel1->m[0][0]*xReal
                        + matrix_real_to_voxel1->m[0][1]*yReal
                        + matrix_real_to_voxel1->m[0][2]*zReal
                        + matrix_real_to_voxel1->m[0][3];
                yVoxel =
                        matrix_real_to_voxel1->m[1][0]*xReal
                        + matrix_real_to_voxel1->m[1][1]*yReal
                        + matrix_real_to_voxel1->m[1][2]*zReal
                        + matrix_real_to_voxel1->m[1][3];
                zVoxel =
                        matrix_real_to_voxel1->m[2][0]*xReal
                        + matrix_real_to_voxel1->m[2][1]*yReal
                        + matrix_real_to_voxel1->m[2][2]*zReal
                        + matrix_real_to_voxel1->m[2][3];

                // The spline coefficients are computed
                xPre=(int)(floor(xVoxel));
                basis=(DTYPE)xVoxel-(DTYPE)xPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, xBasis);
                else Get_SplineBasisValues<DTYPE>(basis, xBasis);

                yPre=(int)(floor(yVoxel));
                basis=(DTYPE)yVoxel-(DTYPE)yPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, yBasis);
                else Get_SplineBasisValues<DTYPE>(basis, yBasis);

                zPre=(int)(floor(zVoxel));
                basis=(DTYPE)zVoxel-(DTYPE)zPre;
                if(basis<0.0) basis=0.0; //rounding error
                if(bspline) Get_BSplineBasisValues<DTYPE>(basis, zBasis);
                else Get_SplineBasisValues<DTYPE>(basis, zBasis);

                --xPre;--yPre;--zPre;

                // The control points are stored
                if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld){
                    if(displacement1){
                        get_GridValues(xPre,
                                       yPre,
                                       zPre,
                                       grid1,
                                       controlPointPtrX,
                                       controlPointPtrY,
                                       controlPointPtrZ,
                                       xControlPointCoordinates,
                                       yControlPointCoordinates,
                                       zControlPointCoordinates,
                                       false);
                    }
                    else{
                        get_GridValues(xPre,
                                       yPre,
                                       zPre,
                                       grid1,
                                       controlPointPtrX,
                                       controlPointPtrY,
                                       controlPointPtrZ,
                                       xControlPointCoordinates,
                                       yControlPointCoordinates,
                                       zControlPointCoordinates,
                                       false);
                    }
                    xPreOld=xPre;
                    yPreOld=yPre;
                    zPreOld=zPre;
                }
                xReal=0.0;
                yReal=0.0;
                zReal=0.0;
#if _USE_SSE
                val.f[0] = xBasis[0];
                val.f[1] = xBasis[1];
                val.f[2] = xBasis[2];
                val.f[3] = xBasis[3];
                _xBasis_sse = val.m;

                tempX =  _mm_set_ps1(0.0);
                tempY =  _mm_set_ps1(0.0);
                tempZ =  _mm_set_ps1(0.0);
                ptrX = (__m128 *) &xControlPointCoordinates[0];
                ptrY = (__m128 *) &yControlPointCoordinates[0];
                ptrZ = (__m128 *) &zControlPointCoordinates[0];

                for(unsigned int c=0; c<4; c++){
                    for(unsigned int b=0; b<4; b++){
                        _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                        _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                        _temp_basis   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                        _basis       = _mm_mul_ps(_temp_basis, _xBasis_sse);
                        tempX = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), tempY );
                        tempZ = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), tempZ );
                        ptrX++;
                        ptrY++;
                        ptrZ++;
                    }
                }
                //the values stored in SSE variables are transfered to normal float
                val.m = tempX;
                xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempY;
                yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                val.m = tempZ;
                zReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                coord=0;
                for(c=0; c<4; c++){
                    for(b=0; b<4; b++){
                        for(a=0; a<4; a++){
                            tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                            xReal += xControlPointCoordinates[coord] * tempValue;
                            yReal += yControlPointCoordinates[coord] * tempValue;
                            zReal += zControlPointCoordinates[coord] * tempValue;
                            coord++;
                        }
                    }
                }
#endif
                if(displacement2){
                    xReal -= initialPositionX;
                    yReal -= initialPositionY;
                    zReal -= initialPositionZ;
                }
                outCPPPtrX[index] = xReal;
                outCPPPtrY[index] = yReal;
                outCPPPtrZ[index] = zReal;
                index++;
            }
        }
    }
    return;
}
/* *************************************************************** */
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline)
{
    // REMINDER Grid2(x)=Grid1(Grid2(x))

    if(grid1->datatype != grid2->datatype){
        fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition\n");
        fprintf(stderr,"[NiftyReg ERROR] Both input images do not have the same type\n");
        exit(1);
    }

#if _USE_SSE
    if(grid1->datatype != NIFTI_TYPE_FLOAT32){
        fprintf(stderr,"[NiftyReg ERROR] SSE computation has only been implemented for single precision.\n");
        fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
        exit(1);
    }
#endif

    if(grid1->nz>1){
        switch(grid1->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_spline_cppComposition_3D<float>
                    (grid1, grid2, displacement1, displacement2, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_cppComposition_3D<double>
                    (grid1, grid2, displacement1, displacement2, bspline);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 3D\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double floating images\n");
            return 1;
        }
    }
    else{
        switch(grid1->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_spline_cppComposition_2D<float>
                    (grid1, grid2, displacement1, displacement2, bspline);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_cppComposition_2D<double>
                    (grid1, grid2, displacement1, displacement2, bspline);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 2D\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
            return 1;
        }
    }
    return 0;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_bspline_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                     nifti_image *deformationFieldImage)
{
    // Check first if the velocity field is actually a velocity field
    if( velocityFieldGrid->intent_code!=NIFTI_INTENT_VECTOR ||
        strcmp(velocityFieldGrid->intent_name,"NREG_VEL_STEP")!=0 ){
        fprintf(stderr, "[NiftyReg ERROR] reg_bspline_getDeformationFieldFromVelocityGrid - the provide grid is not a velocity field\n");
        exit(1);
    }

    /*
    // Euler integration for testing
    {
        printf("Euler integration, %i step(s)\n", (int)pow(2.f,fabs(velocityFieldGrid->intent_p1)));

        nifti_image *scaledControlPointGrid = nifti_copy_nim_info(velocityFieldGrid);
        scaledControlPointGrid->data=(void *)malloc(scaledControlPointGrid->nvox*scaledControlPointGrid->nbyper);
        memcpy(scaledControlPointGrid->data,
               velocityFieldGrid->data,
               scaledControlPointGrid->nvox*scaledControlPointGrid->nbyper);
        reg_getDisplacementFromDeformation(scaledControlPointGrid);
        if(velocityFieldGrid->intent_p1<0) // backward deformation field
            reg_tools_addSubMulDivValue(scaledControlPointGrid,
                                        scaledControlPointGrid,
                                        -pow(2.f,fabs(velocityFieldGrid->intent_p1)),
                                        3);
        else // forward deformation field
            reg_tools_addSubMulDivValue(scaledControlPointGrid,
                                        scaledControlPointGrid,
                                        pow(2.f,fabs(velocityFieldGrid->intent_p1)),
                                        3);
        reg_getDeformationFromDisplacement(scaledControlPointGrid);

        reg_spline_getDeformationField(scaledControlPointGrid,
                                       deformationFieldImage,
                                       deformationFieldImage,
                                       NULL,
                                       false,//composition?
                                       true//bspline?
                                       );

        for(size_t i=1;i<(size_t)pow(2.0f,fabs(velocityFieldGrid->intent_p1));++i){
            reg_spline_getDeformationField(scaledControlPointGrid,
                                           deformationFieldImage,
                                           deformationFieldImage,
                                           NULL,
                                           true,//composition?
                                           true//bspline?
                                           );
        }
        nifti_image_free(scaledControlPointGrid);
    }
    return;
    */

    // The initial deformation is generated using cubic B-Spline parametrisation
    nifti_image *tempDEFImage = NULL;
    tempDEFImage = nifti_copy_nim_info(deformationFieldImage);
    tempDEFImage->data=(void *)malloc(tempDEFImage->nvox*tempDEFImage->nbyper);
    reg_spline_getDeformationField(velocityFieldGrid,
                                   tempDEFImage,
                                   tempDEFImage,
                                   NULL, // mask
                                   false, //composition
                                   true // bspline
                                   );

    // The deformation field is converted from deformation field to displacement field
    reg_getDisplacementFromDeformation(tempDEFImage);

    // The deformation field is scaled
    float scalingValue = pow(2.0f,fabs(velocityFieldGrid->intent_p1));
    if(velocityFieldGrid->intent_p1<0)
        // backward deformation field is scaled down
        reg_tools_addSubMulDivValue(tempDEFImage,
                                    tempDEFImage,
                                    -scalingValue,
                                    3);
    else
        // forward deformation field is scaled down
        reg_tools_addSubMulDivValue(tempDEFImage,
                                    tempDEFImage,
                                    scalingValue,
                                    3);

    // The displacement field is converted back into a deformation field
    reg_getDeformationFromDisplacement(tempDEFImage);

    // The computed scaled deformation field is copied over
    memcpy(deformationFieldImage->data, tempDEFImage->data,
           deformationFieldImage->nvox*deformationFieldImage->nbyper);

    // The deformation field is squared
    size_t squaringNumber = (size_t)fabs(velocityFieldGrid->intent_p1);
    for(size_t i=0;i<squaringNumber;++i){
        // The deformation field is applied to itself
        reg_defField_compose(deformationFieldImage,
                             tempDEFImage,
                             NULL);
        // The computed scaled deformation field is copied over
        memcpy(deformationFieldImage->data, tempDEFImage->data,
               deformationFieldImage->nvox*deformationFieldImage->nbyper);
    }
    // Free the temporary allocated deformation field
    nifti_image_free(tempDEFImage);
}
/* *************************************************************** */
/* *************************************************************** */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void compute_lie_bracket(nifti_image *img1,
                         nifti_image *img2,
                         nifti_image *res,
                         bool use_jac
                         )
{

    // Lie bracket using Jacobian for testing
    if(use_jac){

//        nifti_image *img1Disp = nifti_copy_nim_info(img1);
//        nifti_image *img2Disp = nifti_copy_nim_info(img2);
//        img1Disp->data=(void *)calloc(img1Disp->nvox, img1Disp->nbyper);
//        img2Disp->data=(void *)calloc(img2Disp->nvox, img2Disp->nbyper);
//        reg_spline_cppComposition(img1,
//                                  img1Disp,
//                                  true, // displacement1?
//                                  true, // displacement2?
//                                  true // bspline?
//                                  );
//        reg_spline_cppComposition(img2,
//                                  img2Disp,
//                                  true, // displacement1?
//                                  true, // displacement2?
//                                  true // bspline?
//                                  );

        size_t voxNumber = img1->nx*img1->ny*img1->nz;
        mat33 *jacImg1=(mat33 *)malloc(voxNumber*sizeof(mat33));
        mat33 *jacImg2=(mat33 *)malloc(voxNumber*sizeof(mat33));

        reg_getDeformationFromDisplacement(img1);
        reg_getDeformationFromDisplacement(img2);
        reg_bspline_GetJacobianMatrixFull(img1,img1,jacImg1);
        reg_bspline_GetJacobianMatrixFull(img2,img2,jacImg2);
        reg_getDisplacementFromDeformation(img1);
        reg_getDisplacementFromDeformation(img2);

        DTYPE *resPtrX=static_cast<DTYPE *>(res->data);
        DTYPE *resPtrY=&resPtrX[voxNumber];
        DTYPE *img1DispPtrX=static_cast<DTYPE *>(img1->data);
        DTYPE *img1DispPtrY=&img1DispPtrX[voxNumber];
        DTYPE *img2DispPtrX=static_cast<DTYPE *>(img2->data);
        DTYPE *img2DispPtrY=&img1DispPtrX[voxNumber];
        if(img1->nz>1){
            DTYPE *resPtrZ=&resPtrY[voxNumber];
            DTYPE *img1DispPtrZ=&img1DispPtrY[voxNumber];
            DTYPE *img2DispPtrZ=&img1DispPtrY[voxNumber];

            for(size_t i=0;i<voxNumber;++i){
                resPtrX[i]=
                           (jacImg2[i].m[0][0]*img1DispPtrX[i] +
                            jacImg2[i].m[0][1]*img1DispPtrY[i] +
                            jacImg2[i].m[0][2]*img1DispPtrZ[i] )
                           -
                           (jacImg1[i].m[0][0]*img2DispPtrX[i] +
                            jacImg1[i].m[0][1]*img2DispPtrY[i] +
                            jacImg1[i].m[0][2]*img2DispPtrZ[i] );
                resPtrY[i]=
                           (jacImg2[i].m[1][0]*img1DispPtrX[i] +
                            jacImg2[i].m[1][1]*img1DispPtrY[i] +
                            jacImg2[i].m[1][2]*img1DispPtrZ[i] )
                           -
                           (jacImg1[i].m[1][0]*img2DispPtrX[i] +
                            jacImg1[i].m[1][1]*img2DispPtrY[i] +
                            jacImg1[i].m[1][2]*img2DispPtrZ[i] );
                resPtrZ[i]=
                           (jacImg2[i].m[2][0]*img1DispPtrX[i] +
                            jacImg2[i].m[2][1]*img1DispPtrY[i] +
                            jacImg2[i].m[2][2]*img1DispPtrZ[i] )
                           -
                           (jacImg1[i].m[2][0]*img2DispPtrX[i] +
                            jacImg1[i].m[2][1]*img2DispPtrY[i] +
                            jacImg1[i].m[2][2]*img2DispPtrZ[i] );
            }
        }
        else{
            for(size_t i=0;i<voxNumber;++i){
                resPtrX[i]=
                           (jacImg2[i].m[0][0]*img1DispPtrX[i] +
                            jacImg2[i].m[0][1]*img1DispPtrY[i] )
                           -
                           (jacImg1[i].m[0][0]*img2DispPtrX[i] +
                            jacImg1[i].m[0][1]*img2DispPtrY[i] );
                resPtrY[i]=
                           (jacImg2[i].m[1][0]*img1DispPtrX[i] +
                            jacImg2[i].m[1][1]*img1DispPtrY[i] )
                           -
                           (jacImg1[i].m[1][0]*img2DispPtrX[i] +
                            jacImg1[i].m[1][1]*img2DispPtrY[i] );
            }
        }
        free(jacImg1);
        free(jacImg2);
        return;
    }


    // Allocate two temporary nifti images
    nifti_image *one_two = nifti_copy_nim_info(img2);
    nifti_image *two_one = nifti_copy_nim_info(img1);
    // Set the temporary images to zero displacement
    one_two->data=(void *)calloc(one_two->nvox, one_two->nbyper);
    two_one->data=(void *)calloc(two_one->nvox, two_one->nbyper);
    // Compute the displacement from img1
    reg_spline_cppComposition(img1,
                              two_one,
                              true, // displacement1?
                              true, // displacement2?
                              true // bspline?
                              );
    // Compute the displacement from img2
    reg_spline_cppComposition(img2,
                              one_two,
                              true, // displacement1?
                              true, // displacement2?
                              true // bspline?
                              );
    // Compose both transformations
    reg_spline_cppComposition(img1,
                              one_two,
                              true, // displacement1?
                              true, // displacement2?
                              true // bspline?
                              );
    // Compose both transformations
    reg_spline_cppComposition(img2,
                              two_one,
                              true, // displacement1?
                              true, // displacement2?
                              true // bspline?
                              );
    // Create the data pointers
    DTYPE *resPtr=static_cast<DTYPE *>(res->data);
    DTYPE *one_twoPtr=static_cast<DTYPE *>(one_two->data);
    DTYPE *two_onePtr=static_cast<DTYPE *>(two_one->data);
    // Compute the lie bracket value using difference of composition

#ifdef _WINDOWS
    int i;
#else
    size_t i;
#endif

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(res, resPtr, one_twoPtr, two_onePtr) \
    private(i)
#endif
    for(i=0;i<res->nvox;++i)
        resPtr[i]=one_twoPtr[i]-two_onePtr[i];
    // Free the temporary nifti images
    nifti_image_free(one_two);
    nifti_image_free(two_one);
//    reg_spline_GetDeconvolvedCoefficents(res);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void compute_BCH_update1(nifti_image *img1, // current field
                         nifti_image *img2, // gradient
                         int type)
{
    DTYPE *res=(DTYPE *)malloc(img1->nvox*sizeof(DTYPE));

#ifdef _WINDOWS
    int i;
#else
    size_t i;
#endif
    
    bool use_jac=false;

    // r <- 2 + 1
    DTYPE *img1Ptr=static_cast<DTYPE *>(img1->data);
    DTYPE *img2Ptr=static_cast<DTYPE *>(img2->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img1,img1Ptr,img2Ptr, res) \
    private(i)
#endif
    for(i=0; i<img1->nvox;++i)
        res[i] = img1Ptr[i] + img2Ptr[i];

    if(type>0){
        // Convert the deformation field into a displacement field
        reg_getDisplacementFromDeformation(img1);

        // r <- 2 + 1 + 0.5[2,1]
        nifti_image *lie_bracket_img2_img1=nifti_copy_nim_info(img1);
        lie_bracket_img2_img1->data=(void *)malloc(lie_bracket_img2_img1->nvox*lie_bracket_img2_img1->nbyper);
        compute_lie_bracket<DTYPE>(img2, img1, lie_bracket_img2_img1, use_jac);
        DTYPE *lie_bracket_img2_img1Ptr=static_cast<DTYPE *>(lie_bracket_img2_img1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img1, res, lie_bracket_img2_img1Ptr) \
    private(i)
#endif
        for(i=0; i<img1->nvox;++i)
            res[i] += 0.5 * lie_bracket_img2_img1Ptr[i];

        if(type>1){
            // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12
            nifti_image *lie_bracket_img2_lie1=nifti_copy_nim_info(lie_bracket_img2_img1);
            lie_bracket_img2_lie1->data=(void *)malloc(lie_bracket_img2_lie1->nvox*lie_bracket_img2_lie1->nbyper);
            compute_lie_bracket<DTYPE>(img2, lie_bracket_img2_img1, lie_bracket_img2_lie1, use_jac);
            DTYPE *lie_bracket_img2_lie1Ptr=static_cast<DTYPE *>(lie_bracket_img2_lie1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img1, res, lie_bracket_img2_lie1Ptr) \
    private(i)
#endif
            for(i=0; i<img1->nvox;++i)
                res[i] += lie_bracket_img2_lie1Ptr[i]/12.0;

            if(type>2){
                // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12
                nifti_image *lie_bracket_img1_lie1=nifti_copy_nim_info(lie_bracket_img2_img1);
                lie_bracket_img1_lie1->data=(void *)malloc(lie_bracket_img1_lie1->nvox*lie_bracket_img1_lie1->nbyper);
                compute_lie_bracket<DTYPE>(img1, lie_bracket_img2_img1, lie_bracket_img1_lie1, use_jac);
                DTYPE *lie_bracket_img1_lie1Ptr=static_cast<DTYPE *>(lie_bracket_img1_lie1->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img1, res, lie_bracket_img1_lie1Ptr) \
    private(i)
#endif
                for(i=0; i<img1->nvox;++i)
                    res[i] -= lie_bracket_img1_lie1Ptr[i]/12.0;
                nifti_image_free(lie_bracket_img1_lie1);

                if(type>3){
                    // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12 - [1,[2,[2,1]]]/24
                    nifti_image *lie_bracket_img1_lie2=nifti_copy_nim_info(lie_bracket_img2_lie1);
                    lie_bracket_img1_lie2->data=(void *)malloc(lie_bracket_img1_lie2->nvox*lie_bracket_img1_lie2->nbyper);
                    compute_lie_bracket<DTYPE>(img1, lie_bracket_img2_lie1, lie_bracket_img1_lie2, use_jac);
                    DTYPE *lie_bracket_img1_lie2Ptr=static_cast<DTYPE *>(lie_bracket_img1_lie2->data);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(img1, res, lie_bracket_img1_lie2Ptr) \
    private(i)
#endif
                    for(i=0; i<img1->nvox;++i)
                        res[i] -= lie_bracket_img1_lie2Ptr[i]/24.0;
                    nifti_image_free(lie_bracket_img1_lie2);
                }// >3
            }// >2
            nifti_image_free(lie_bracket_img2_lie1);
        }// >1
        nifti_image_free(lie_bracket_img2_img1);
    }// >0

    // update the deformation field
    memcpy(img1->data, res, img1->nvox*img1->nbyper);
    free(res);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void compute_BCH_update(nifti_image *img1, // current field
                         nifti_image *img2, // gradient
                         int type)
{
    if(img1->datatype!=img2->datatype){
        fprintf(stderr,"[NiftyReg ERROR] compute_BCH_update\n");
        fprintf(stderr,"[NiftyReg ERROR] Both input images are expected to be of similar type\n");
        exit(1);
    }
    switch(img1->datatype){
        case NIFTI_TYPE_FLOAT32:
            compute_BCH_update1<float>(img1, img2, type);
            break;
        case NIFTI_TYPE_FLOAT64:
            compute_BCH_update1<double>(img1, img2, type);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] compute_BCH_update\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
            exit(1);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void extractLine(int start, int end, int increment,const DTYPE *image, DTYPE *values)
{
    size_t index = 0;
    for(int i=start; i<end; i+=increment) values[index++] = image[i];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void restoreLine(int start, int end, int increment, DTYPE *image, const DTYPE *values)
{
    size_t index = 0;
    for(int i=start; i<end; i+=increment) image[i] = values[index++];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void intensitiesToSplineCoefficients(DTYPE *values, int number)
{
    // Border are set to zero
    DTYPE pole = sqrt(3.0) - 2.0;
    DTYPE currentPole = pole;
    DTYPE currentOpposite = pow(pole,(DTYPE)(2.0*(DTYPE)number-1.0));
    DTYPE sum=0.0;
    for(int i=1; i<number; i++){
        sum += (currentPole - currentOpposite) * values[i];
        currentPole *= pole;
        currentOpposite /= pole;
    }
    values[0] = (DTYPE)((values[0] - pole*pole*(values[0] + sum)) / (1.0 - pow(pole,(DTYPE)(2.0*(double)number+2.0))));

    //other values forward
    for(int i=1; i<number; i++){
        values[i] += pole * values[i-1];
    }

    DTYPE ipp=(DTYPE)(1.0-pole); ipp*=ipp;

    //last value
    values[number-1] = ipp * values[number-1];

    //other values backward
    for(int i=number-2; 0<=i; i--){
        values[i] = pole * values[i+1] + ipp*values[i];
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void reg_spline_GetDeconvolvedCoefficents1(nifti_image *img)
{
    double *coeff=(double *)malloc(img->nvox*sizeof(double));
    DTYPE *imgPtr=static_cast<DTYPE *>(img->data);
    for(size_t i=0;i<img->nvox;++i)
        coeff[i]=imgPtr[i];
    for(int u=0;u<img->nu;++u){
        for(int t=0;t<img->nt;++t){
            double *coeffPtr=&coeff[(u*img->nt+t)*img->nx*img->ny*img->nz];

            // Along the X axis
            int number = img->nx;
            double *values=new double[number];
            int increment = 1;
            for(int i=0;i<img->ny*img->nz;i++){
                int start = i*img->nx;
                int end = start + img->nx;
                extractLine<double>(start,end,increment,coeffPtr,values);
                intensitiesToSplineCoefficients<double>(values, number);
                restoreLine<double>(start,end,increment,coeffPtr,values);
            }
            delete[] values;values=NULL;

            // Along the Y axis
            number = img->ny;
            values=new double[number];
            increment = img->nx;
            for(int i=0;i<img->nx*img->nz;i++){
                int start = i + i/img->nx * img->nx * (img->ny - 1);
                int end = start + img->nx*img->ny;
                extractLine<double>(start,end,increment,coeffPtr,values);
                intensitiesToSplineCoefficients<double>(values, number);
                restoreLine<double>(start,end,increment,coeffPtr,values);
            }
            delete[] values;values=NULL;

            // Along the Y axis
            if(img->nz>1){
                number = img->nz;
                values=new double[number];
                increment = img->nx*img->ny;
                for(int i=0;i<img->nx*img->ny;i++){
                    int start = i;
                    int end = start + img->nx*img->ny*img->nz;
                    extractLine<double>(start,end,increment,coeffPtr,values);
                    intensitiesToSplineCoefficients<double>(values, number);
                    restoreLine<double>(start,end,increment,coeffPtr,values);
                }
                delete[] values;values=NULL;
            }
        }//t
    }//u

    for(size_t i=0;i<img->nvox;++i)
        imgPtr[i]=coeff[i];
    free(coeff);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_spline_GetDeconvolvedCoefficents(nifti_image *img)
{

    switch(img->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_spline_GetDeconvolvedCoefficents1<float>(img);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_spline_GetDeconvolvedCoefficents1<double>(img);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] reg_spline_GetDeconvolvedCoefficents1\n");
            fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
            exit(1);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#include "_reg_localTransformation_jac.cpp"
#include "_reg_localTransformation_be.cpp"

#endif
