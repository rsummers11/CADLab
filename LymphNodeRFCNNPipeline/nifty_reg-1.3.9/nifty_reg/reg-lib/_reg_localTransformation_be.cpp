/*
 *  _reg_localTransformation_be.cpp
 *  
 *
 *  Created by Marc Modat on 10/05/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_localTransformation.h"

/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_bendingEnergyApproxValue2D(nifti_image *splineControlPoint)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same
    SplineTYPE basisXX[9], basisYY[9], basisXY[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    int coord = 0;
    for(int b=0; b<3; b++){
        for(int a=0; a<3; a++){
            basisXX[coord] = second[a] * normal[b];
            basisYY[coord] = normal[a] * second[b];
            basisXY[coord] = first[a] * first[b];
            coord++;
        }
    }

    SplineTYPE constraintValue=0.0;

    SplineTYPE xControlPointCoordinates[9];
    SplineTYPE yControlPointCoordinates[9];
    int x, y, a;
    SplineTYPE XX_x, YY_x, XY_x;
    SplineTYPE XX_y, YY_y, XY_y;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, controlPointPtrX, controlPointPtrY,  basisXX, basisYY, basisXY) \
    private(XX_x, YY_x, XY_x, XX_y, YY_y, XY_y, x, y, a, \
    xControlPointCoordinates, yControlPointCoordinates) \
    reduction(+:constraintValue)
#endif
    for(y=1;y<splineControlPoint->ny-1;y++){
        for(x=1;x<splineControlPoint->nx-1;x++){

            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            XX_x=0.0;
            YY_x=0.0;
            XY_x=0.0;
            XX_y=0.0;
            YY_y=0.0;
            XY_y=0.0;

            for(a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + 2.0*XY_x*XY_x);
            constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + 2.0*XY_y*XY_y);
        }
    }
    return constraintValue/(double)(2.0*splineControlPoint->nx*splineControlPoint->ny);
}
/* *************************************************************** */
template<class SplineTYPE>
double reg_bspline_bendingEnergyApproxValue3D(nifti_image *splineControlPoint)
{
    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>
            (splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>
            (&controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>
            (&controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz]);

    // As the contraint is only computed at the control point positions, the basis value of the spline are always the same
    SplineTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    // There are six different values taken into account
    SplineTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];
    int coord=0;
    for(int c=0; c<3; c++){
        for(int b=0; b<3; b++){
            tempXX[coord]=normal[c]*normal[b];  // z * y
            tempYY[coord]=normal[c]*second[b];  // z * y"
            tempZZ[coord]=second[c]*normal[b];  // z"* y
            tempXY[coord]=normal[c]*first[b];   // z * y'
            tempYZ[coord]=first[c]*first[b];    // z'* y'
            tempXZ[coord]=first[c]*normal[b];   // z'* y
            coord++;
        }
    }
    coord=0;
    for(int bc=0; bc<9; bc++){
        for(int a=0; a<3; a++){
            basisXX[coord]=tempXX[bc]*second[a];    // z * y * x"
            basisYY[coord]=tempYY[bc]*normal[a];    // z * y"* x
            basisZZ[coord]=tempZZ[bc]*normal[a];    // z"* y * x
            basisXY[coord]=tempXY[bc]*first[a];     // z * y'* x'
            basisYZ[coord]=tempYZ[bc]*normal[a];    // z'* y'* x
            basisXZ[coord]=tempXZ[bc]*first[a];     // z'* y * x'
            coord++;
        }
    }

    double constraintValue=0.0;

    SplineTYPE xControlPointCoordinates[27];
    SplineTYPE yControlPointCoordinates[27];
    SplineTYPE zControlPointCoordinates[27];
    int x, y, z, a;
    SplineTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
    SplineTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
    SplineTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
    private(XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y, \
    XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z, x, y, z, a, \
    xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates) \
    reduction(+:constraintValue)
#endif
    for(z=1;z<splineControlPoint->nz-1;z++){
        for(y=1;y<splineControlPoint->ny-1;y++){
            for(x=1;x<splineControlPoint->nx-1;x++){

                get_GridValuesApprox<SplineTYPE>(x-1,
                                                 y-1,
                                                 z-1,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates,
                                                 yControlPointCoordinates,
                                                 zControlPointCoordinates,
                                                 false);

                XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
                XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
                XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
                XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
                XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
                XY_z=0.0, YZ_z=0.0, XZ_z=0.0;

                for(a=0; a<27; a++){
                    XX_x += basisXX[a]*xControlPointCoordinates[a];
                    YY_x += basisYY[a]*xControlPointCoordinates[a];
                    ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
                    XY_x += basisXY[a]*xControlPointCoordinates[a];
                    YZ_x += basisYZ[a]*xControlPointCoordinates[a];
                    XZ_x += basisXZ[a]*xControlPointCoordinates[a];

                    XX_y += basisXX[a]*yControlPointCoordinates[a];
                    YY_y += basisYY[a]*yControlPointCoordinates[a];
                    ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
                    XY_y += basisXY[a]*yControlPointCoordinates[a];
                    YZ_y += basisYZ[a]*yControlPointCoordinates[a];
                    XZ_y += basisXZ[a]*yControlPointCoordinates[a];

                    XX_z += basisXX[a]*zControlPointCoordinates[a];
                    YY_z += basisYY[a]*zControlPointCoordinates[a];
                    ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
                    XY_z += basisXY[a]*zControlPointCoordinates[a];
                    YZ_z += basisYZ[a]*zControlPointCoordinates[a];
                    XZ_z += basisXZ[a]*zControlPointCoordinates[a];
                }

                constraintValue += (double)(XX_x*XX_x + YY_x*YY_x + ZZ_x*ZZ_x + 2.0*(XY_x*XY_x + YZ_x*YZ_x + XZ_x*XZ_x));
                constraintValue += (double)(XX_y*XX_y + YY_y*YY_y + ZZ_y*ZZ_y + 2.0*(XY_y*XY_y + YZ_y*YZ_y + XZ_y*XZ_y));
                constraintValue += (double)(XX_z*XX_z + YY_z*YY_z + ZZ_z*ZZ_z + 2.0*(XY_z*XY_z + YZ_z*YZ_z + XZ_z*XZ_z));
            }
        }
    }

    return constraintValue/(double)(3.0*splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
}
/* *************************************************************** */
extern "C++"
double reg_bspline_bendingEnergy(nifti_image *splineControlPoint)
{
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
        case NIFTI_TYPE_FLOAT32:
            return reg_bspline_bendingEnergyApproxValue2D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_bspline_bendingEnergyApproxValue2D<double>(splineControlPoint);
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy is not computed\n");
            exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
        case NIFTI_TYPE_FLOAT32:
            return reg_bspline_bendingEnergyApproxValue3D<float>(splineControlPoint);
        case NIFTI_TYPE_FLOAT64:
            return reg_bspline_bendingEnergyApproxValue3D<double>(splineControlPoint);
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy is not computed\n");
            exit(1);
        }

    }
}
/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient2D(nifti_image *splineControlPoint,
                                               nifti_image *targetImage,
                                               nifti_image *gradientImage,
                                               float weight)
{
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisXX[9], basisYY[9], basisXY[9];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    int coord=0, x, y, a, b, X, Y, index;
    for(b=0; b<3; b++){
        for(a=0; a<3; a++){
            basisXX[coord] = second[a] * normal[b];
            basisYY[coord] = normal[a] * second[b];
            basisXY[coord] = first[a] * first[b];
            coord++;
        }
    }

    int nodeNumber = splineControlPoint->nx*splineControlPoint->ny;
    SplineTYPE *derivativeValues = (SplineTYPE *)calloc(6*nodeNumber, sizeof(SplineTYPE));

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = &(controlPointPtrX[nodeNumber]);

    SplineTYPE xControlPointCoordinates[9];
    SplineTYPE yControlPointCoordinates[9];

    SplineTYPE *derivativeValuesPtr, XX_x, YY_x, XY_x, XX_y, YY_y, XY_y;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, controlPointPtrX, controlPointPtrY, \
    basisXX, basisYY, basisXY, derivativeValues) \
    private(x, y, a, index, xControlPointCoordinates, yControlPointCoordinates, \
    derivativeValuesPtr, XX_x, YY_x, XY_x, XX_y, YY_y, XY_y)
#endif
    for(y=1;y<splineControlPoint->ny-1;y++){
        index=y*splineControlPoint->nx+1;
        for(x=1;x<splineControlPoint->nx-1;x++){

            get_GridValuesApprox<SplineTYPE>(x-1,
                                             y-1,
                                             splineControlPoint,
                                             controlPointPtrX,
                                             controlPointPtrY,
                                             xControlPointCoordinates,
                                             yControlPointCoordinates,
                                             false);

            XX_x=0.0;
            YY_x=0.0;
            XY_x=0.0;
            XX_y=0.0;
            YY_y=0.0;
            XY_y=0.0;

            for(a=0; a<9; a++){
                XX_x += basisXX[a]*xControlPointCoordinates[a];
                YY_x += basisYY[a]*xControlPointCoordinates[a];
                XY_x += basisXY[a]*xControlPointCoordinates[a];

                XX_y += basisXX[a]*yControlPointCoordinates[a];
                YY_y += basisYY[a]*yControlPointCoordinates[a];
                XY_y += basisXY[a]*yControlPointCoordinates[a];
            }

            derivativeValuesPtr = &derivativeValues[6*index];
            index++;

            derivativeValuesPtr[0] = (SplineTYPE)(2.0*XX_x);
            derivativeValuesPtr[1] = (SplineTYPE)(2.0*YY_x);
            derivativeValuesPtr[2] = (SplineTYPE)(4.0*XY_x);
            derivativeValuesPtr[3] = (SplineTYPE)(2.0*XX_y);
            derivativeValuesPtr[4] = (SplineTYPE)(2.0*YY_y);
            derivativeValuesPtr[5] = (SplineTYPE)(4.0*XY_y);
        }
    }

    SplineTYPE *gradientXPtr = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientYPtr = &(gradientXPtr[nodeNumber]);

    SplineTYPE approxRatio= weight * (SplineTYPE)(targetImage->nx*targetImage->ny)
            / (SplineTYPE)nodeNumber;

    SplineTYPE gradientValue[2];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, basisXX, basisYY, basisXY, derivativeValues, \
    gradientXPtr, gradientYPtr, approxRatio) \
    private(x, y, X, Y, index, coord, derivativeValuesPtr, gradientValue)
#endif

    for(y=0;y<splineControlPoint->ny;y++){
        index=y*splineControlPoint->nx;
        for(x=0;x<splineControlPoint->nx;x++){

            gradientValue[0]=gradientValue[1]=0.0;

            coord=0;
            for(Y=y-1; Y<y+2; Y++){
                for(X=x-1; X<x+2; X++){
                    if(-1<X && -1<Y && X<splineControlPoint->nx && Y<splineControlPoint->ny){
                        derivativeValuesPtr = &derivativeValues[6 * (Y*splineControlPoint->nx + X)];
                        gradientValue[0] += derivativeValuesPtr[0] * basisXX[coord] +
                                derivativeValuesPtr[1] * basisYY[coord] +
                                derivativeValuesPtr[2] * basisXY[coord];

                        gradientValue[1] += derivativeValuesPtr[3] * basisXX[coord] +
                                derivativeValuesPtr[4] * basisYY[coord] +
                                derivativeValuesPtr[5] * basisXY[coord];
                    }
                    ++coord;
                }
            }
            // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)

            gradientXPtr[index] += (SplineTYPE)(approxRatio*gradientValue[0]);
            gradientYPtr[index] += (SplineTYPE)(approxRatio*gradientValue[1]);
            index++;
        }
    }
    free(derivativeValues);
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_bspline_approxBendingEnergyGradient3D( nifti_image *splineControlPoint,
                                               nifti_image *targetImage,
                                               nifti_image *gradientImage,
                                               float weight)
{
    int a, b, c, bc, x, y, z, X, Y, Z;
    // As the contraint is only computed at the voxel position, the basis value of the spline are always the same
    SplineTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    SplineTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    SplineTYPE first[3]={-0.5, 0, 0.5};
    SplineTYPE second[3]={1.0, -2.0, 1.0};
    // There are six different values taken into account
    SplineTYPE tempXX[9], tempYY[9], tempZZ[9], tempXY[9], tempYZ[9], tempXZ[9];
    int coord=0;
    for(c=0; c<3; c++){
        for(b=0; b<3; b++){
            tempXX[coord]=normal[c]*normal[b];  // z * y
            tempYY[coord]=normal[c]*second[b];  // z * y"
            tempZZ[coord]=second[c]*normal[b];  // z"* y
            tempXY[coord]=normal[c]*first[b];   // z * y'
            tempYZ[coord]=first[c]*first[b];    // z'* y'
            tempXZ[coord]=first[c]*normal[b];   // z'* y
            coord++;
        }
    }
    coord=0;
    for(bc=0; bc<9; bc++){
        for(a=0; a<3; a++){
            basisXX[coord]=tempXX[bc]*second[a];    // z * y * x"
            basisYY[coord]=tempYY[bc]*normal[a];    // z * y"* x
            basisZZ[coord]=tempZZ[bc]*normal[a];    // z"* y * x
            basisXY[coord]=tempXY[bc]*first[a];     // z * y'* x'
            basisYZ[coord]=tempYZ[bc]*normal[a];    // z'* y'* x
            basisXZ[coord]=tempXZ[bc]*first[a];     // z'* y * x'
            coord++;
        }
    }

    SplineTYPE nodeNumber = (SplineTYPE)(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);
    SplineTYPE *derivativeValues = (SplineTYPE *)calloc(18*(int)nodeNumber, sizeof(SplineTYPE));
    SplineTYPE *derivativeValuesPtr;

    SplineTYPE *controlPointPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *controlPointPtrY = static_cast<SplineTYPE *>(&controlPointPtrX[(unsigned int)nodeNumber]);
    SplineTYPE *controlPointPtrZ = static_cast<SplineTYPE *>(&controlPointPtrY[(unsigned int)nodeNumber]);

    SplineTYPE xControlPointCoordinates[27];
    SplineTYPE yControlPointCoordinates[27];
    SplineTYPE zControlPointCoordinates[27];
    SplineTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
    SplineTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
    SplineTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint,controlPointPtrX,controlPointPtrY,controlPointPtrZ, derivativeValues, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ) \
    private(a, x, y, z, derivativeValuesPtr, xControlPointCoordinates, yControlPointCoordinates, \
    zControlPointCoordinates, XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x, XX_y, YY_y, \
    ZZ_y, XY_y, YZ_y, XZ_y, XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z)
#endif
    for(z=1;z<splineControlPoint->nz-1;z++){
        for(y=1;y<splineControlPoint->ny-1;y++){
            derivativeValuesPtr = &derivativeValues[18*((z*splineControlPoint->ny+y)*splineControlPoint->nx+1)];
            for(x=1;x<splineControlPoint->nx-1;x++){

                get_GridValuesApprox<SplineTYPE>(x-1,
                                                 y-1,
                                                 z-1,
                                                 splineControlPoint,
                                                 controlPointPtrX,
                                                 controlPointPtrY,
                                                 controlPointPtrZ,
                                                 xControlPointCoordinates,
                                                 yControlPointCoordinates,
                                                 zControlPointCoordinates,
                                                 false);

                XX_x=0.0, YY_x=0.0, ZZ_x=0.0;
                XY_x=0.0, YZ_x=0.0, XZ_x=0.0;
                XX_y=0.0, YY_y=0.0, ZZ_y=0.0;
                XY_y=0.0, YZ_y=0.0, XZ_y=0.0;
                XX_z=0.0, YY_z=0.0, ZZ_z=0.0;
                XY_z=0.0, YZ_z=0.0, XZ_z=0.0;

                for(a=0; a<27; a++){
                    XX_x += basisXX[a]*xControlPointCoordinates[a];
                    YY_x += basisYY[a]*xControlPointCoordinates[a];
                    ZZ_x += basisZZ[a]*xControlPointCoordinates[a];
                    XY_x += basisXY[a]*xControlPointCoordinates[a];
                    YZ_x += basisYZ[a]*xControlPointCoordinates[a];
                    XZ_x += basisXZ[a]*xControlPointCoordinates[a];

                    XX_y += basisXX[a]*yControlPointCoordinates[a];
                    YY_y += basisYY[a]*yControlPointCoordinates[a];
                    ZZ_y += basisZZ[a]*yControlPointCoordinates[a];
                    XY_y += basisXY[a]*yControlPointCoordinates[a];
                    YZ_y += basisYZ[a]*yControlPointCoordinates[a];
                    XZ_y += basisXZ[a]*yControlPointCoordinates[a];

                    XX_z += basisXX[a]*zControlPointCoordinates[a];
                    YY_z += basisYY[a]*zControlPointCoordinates[a];
                    ZZ_z += basisZZ[a]*zControlPointCoordinates[a];
                    XY_z += basisXY[a]*zControlPointCoordinates[a];
                    YZ_z += basisYZ[a]*zControlPointCoordinates[a];
                    XZ_z += basisXZ[a]*zControlPointCoordinates[a];
                }
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*XX_z);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*YY_z);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(2.0*ZZ_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XY_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*YZ_z);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_x);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_y);
                *derivativeValuesPtr++ = (SplineTYPE)(4.0*XZ_z);
            }
        }
    }

    SplineTYPE *gradientX = static_cast<SplineTYPE *>(gradientImage->data);
    SplineTYPE *gradientY = &gradientX[(int)nodeNumber];
    SplineTYPE *gradientZ = &gradientY[(int)nodeNumber];
    SplineTYPE *gradientXPtr = &gradientX[0];
    SplineTYPE *gradientYPtr = &gradientY[0];
    SplineTYPE *gradientZPtr = &gradientZ[0];

    SplineTYPE approxRatio = weight * (SplineTYPE)(targetImage->nx*targetImage->ny*targetImage->nz)
            / (SplineTYPE)(splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz);

    SplineTYPE gradientValue[3];

    int index;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, derivativeValues, gradientXPtr, gradientYPtr, gradientZPtr, \
    basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ, approxRatio) \
    private(index, X, Y, Z, x, y, z, derivativeValuesPtr, coord, gradientValue)
#endif
    for(z=0;z<splineControlPoint->nz;z++){
        index=z*splineControlPoint->nx*splineControlPoint->ny;
        for(y=0;y<splineControlPoint->ny;y++){
            for(x=0;x<splineControlPoint->nx;x++){

                gradientValue[0]=gradientValue[1]=gradientValue[2]=0.0;

                coord=0;
                for(Z=z-1; Z<z+2; Z++){
                    for(Y=y-1; Y<y+2; Y++){
                        for(X=x-1; X<x+2; X++){
                            if(-1<X && -1<Y && -1<Z && X<splineControlPoint->nx && Y<splineControlPoint->ny && Z<splineControlPoint->nz){
                                derivativeValuesPtr = &derivativeValues[18 * ((Z*splineControlPoint->ny + Y)*splineControlPoint->nx + X)];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXX[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXX[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXX[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYY[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYY[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYY[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisZZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisZZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisZZ[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXY[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXY[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXY[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisYZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisYZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisYZ[coord];

                                gradientValue[0] += (*derivativeValuesPtr++) * basisXZ[coord];
                                gradientValue[1] += (*derivativeValuesPtr++) * basisXZ[coord];
                                gradientValue[2] += (*derivativeValuesPtr++) * basisXZ[coord];
                            }
                            coord++;
                        }
                    }
                }
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way (NMI, BE, JAC)
                gradientXPtr[index] += (SplineTYPE)(approxRatio*gradientValue[0]);
                gradientYPtr[index] += (SplineTYPE)(approxRatio*gradientValue[1]);
                gradientZPtr[index] += (SplineTYPE)(approxRatio*gradientValue[2]);
                index++;
            }
        }
    }

    free(derivativeValues);
}
/* *************************************************************** */
extern "C++"
void reg_bspline_bendingEnergyGradient(nifti_image *splineControlPoint,
                                       nifti_image *targetImage,
                                       nifti_image *gradientImage,
                                       float weight)
{
    if(splineControlPoint->datatype != gradientImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] The spline control point image and the gradient image were expected to have the same datatype\n");
        fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
        exit(1);
    }
    if(splineControlPoint->nz==1){
        switch(splineControlPoint->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_approxBendingEnergyGradient2D<float>
                    (splineControlPoint, targetImage, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_approxBendingEnergyGradient2D<double>
                    (splineControlPoint, targetImage, gradientImage, weight);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not been computed\n");
            exit(1);
        }
    }
    else{
        switch(splineControlPoint->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_approxBendingEnergyGradient3D<float>
                    (splineControlPoint, targetImage, gradientImage, weight);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_approxBendingEnergyGradient3D<double>
                    (splineControlPoint, targetImage, gradientImage, weight);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not been computed\n");
            exit(1);
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_linearEnergyApproxValue1(nifti_image *splineControlPoint, double *constraintValue)
{
    size_t nodeNumber = splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
    int index, x, y, z;

    DTYPE *jacDeterminant=(DTYPE *)malloc(nodeNumber*sizeof(DTYPE));
    mat33 *jacobianMatrices=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    if(splineControlPoint->nz>1){
        reg_bspline_computeApproximateJacobianMatrices_3D(splineControlPoint,
                                                          jacobianMatrices,
                                                          jacDeterminant);
    }
    else{
        reg_bspline_computeApproximateJacobianMatrices_2D(splineControlPoint,
                                                          jacobianMatrices,
                                                          jacDeterminant);
    }

    double constraintValue0, constraintValue1;
    constraintValue0=constraintValue1=0;
    mat33 jacobianMatrix;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, jacobianMatrices) \
    private(x, y, z, index, jacobianMatrix) \
    reduction(+:constraintValue0) \
    reduction(+:constraintValue1)
#endif
    for(z=1;z<splineControlPoint->nz-1;z++){
        for(y=1;y<splineControlPoint->ny-1;y++){
            index=(z*splineControlPoint->ny+y)*splineControlPoint->nx+1;
            for(x=1;x<splineControlPoint->nx-1;x++){

                if(splineControlPoint->nz>1){
                    jacobianMatrix=jacobianMatrices[index];
                    jacobianMatrix.m[0][0]--;
                    jacobianMatrix.m[1][1]--;
                    jacobianMatrix.m[2][2]--;
                    constraintValue0 += (double).5 * ( POW2(jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) +
                                                       POW2(jacobianMatrix.m[0][2]+jacobianMatrix.m[2][0]) +
                                                       POW2(jacobianMatrix.m[1][2]+jacobianMatrix.m[2][1]) ) +
                                                       POW2(jacobianMatrix.m[0][0]) +
                                                       POW2(jacobianMatrix.m[1][1]) +
                                                       POW2(jacobianMatrix.m[2][2]);
                    constraintValue1 += (double)POW2(jacobianMatrix.m[0][0] +
                                                     jacobianMatrix.m[1][1]+
                                                     jacobianMatrix.m[2][2]);
                }
                else{
                    jacobianMatrix=jacobianMatrices[index];
                    jacobianMatrix.m[0][0]--;
                    jacobianMatrix.m[1][1]--;
                    constraintValue0 += (double).5 * ( POW2(jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) ) +
                                                       POW2(jacobianMatrix.m[0][0]) +
                                                       POW2(jacobianMatrix.m[1][1]);
                    constraintValue1 += (double)POW2(jacobianMatrix.m[0][0] +
                                                     jacobianMatrix.m[1][1]);
                }
                index++;
            }
        }
    }
    constraintValue[0] = constraintValue0/(double)(splineControlPoint->nvox);
    constraintValue[1] = constraintValue1/(double)(splineControlPoint->nvox);

    free(jacobianMatrices);
}
/* *************************************************************** */
void reg_bspline_linearEnergy(nifti_image *splineControlPoint, double *val)
{
    switch(splineControlPoint->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_bspline_linearEnergyApproxValue1<float>(splineControlPoint, val);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_bspline_linearEnergyApproxValue1<double>(splineControlPoint, val);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the linear energy\n");
        fprintf(stderr,"[NiftyReg ERROR] The linear energy is not computed\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
double reg_bspline_L2norm_displacement1(nifti_image *splineControlPoint)
{
    size_t nodeNumber = splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
    size_t index;
    int x,y,z;

    nifti_image *dispControlPoint=nifti_copy_nim_info(splineControlPoint);
    dispControlPoint->data=(void *)malloc(dispControlPoint->nvox*dispControlPoint->nbyper);
    memcpy(dispControlPoint->data,splineControlPoint->data,dispControlPoint->nvox*dispControlPoint->nbyper);
    reg_getDisplacementFromDeformation(dispControlPoint);

    DTYPE *dispPointPtrX = static_cast<DTYPE *>(dispControlPoint->data);
    DTYPE *dispPointPtrY = &dispPointPtrX[nodeNumber];
    DTYPE *dispPointPtrZ = NULL;
    if(splineControlPoint->nz>1)
        dispPointPtrZ = &dispPointPtrY[nodeNumber];

    double constraintValue=0;

    for(z=0;z<splineControlPoint->nz;z++){
        for(y=0;y<splineControlPoint->ny;y++){
            index=(z*splineControlPoint->ny+y)*splineControlPoint->nx;
            for(x=0;x<splineControlPoint->nx;x++){
                constraintValue += POW2((double)dispPointPtrX[index]);
                constraintValue += POW2((double)dispPointPtrY[index]);
                if(dispPointPtrZ!=NULL)
                    constraintValue += POW2((double)dispPointPtrZ[index]);

                ++index;
            }
        }
    }
    nifti_image_free(dispControlPoint);

    return constraintValue/(double)(splineControlPoint->nvox);
}
/* *************************************************************** */
double reg_bspline_L2norm_displacement(nifti_image *splineControlPoint)
{
    switch(splineControlPoint->datatype){
    case NIFTI_TYPE_FLOAT32:
        return reg_bspline_L2norm_displacement1<float>(splineControlPoint);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_bspline_L2norm_displacement1<double>(splineControlPoint);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for reg_bspline_L2norm_displacement\n");
        fprintf(stderr,"[NiftyReg ERROR] The reg_bspline_L2norm_displacement is not computed\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_L2norm_dispGradient1(nifti_image *splineControlPoint,
                                      nifti_image *referenceImage,
                                      nifti_image *gradientImage,
                                      float weight)
{    int nodeNumber = splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
     int voxNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
     int a, b, c, x, y, z, index, coord, currentIndex;

     DTYPE *gradPtrX = static_cast<DTYPE *>(gradientImage->data);
     DTYPE *gradPtrY = &gradPtrX[nodeNumber];
     DTYPE *gradPtrZ = NULL;
     if(splineControlPoint->nz>1)
         gradPtrZ = &gradPtrY[nodeNumber];

     DTYPE approxRatio = 2.0 * (double)weight * (double)(voxNumber) / (double)nodeNumber;

     DTYPE basis[27], basisX[27], basisY[27], basisZ[27];
     DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
     DTYPE first[3]={-0.5, 0, 0.5};
     coord = 0;

    nifti_image *dispControlPoint=nifti_copy_nim_info(splineControlPoint);
    dispControlPoint->data=(void *)malloc(dispControlPoint->nvox*dispControlPoint->nbyper);
    memcpy(dispControlPoint->data,splineControlPoint->data,dispControlPoint->nvox*dispControlPoint->nbyper);
    reg_getDisplacementFromDeformation(dispControlPoint);
    DTYPE *dispPointPtrX = static_cast<DTYPE *>(dispControlPoint->data);
    DTYPE *dispPointPtrY = &dispPointPtrX[nodeNumber];
    DTYPE *dispPointPtrZ = NULL;
    if(splineControlPoint->nz>1)
        dispPointPtrZ = &dispPointPtrY[nodeNumber];

     if(splineControlPoint->nz>1){
         for(c=2; c>-1; --c){
             for(b=2; b>-1; --b){
                 for(a=2; a>-1; --a){
                     basis[coord]  = normal[a] * normal[b] * normal[c];
                     basisX[coord] = first[a] * normal[b] * normal[c];
                     basisY[coord] = normal[a] * first[b] * normal[c];
                     basisZ[coord] = normal[a] * normal[b] * first[c];
                     coord++;
                 }
             }
         }
     }
     else{
         for(b=2; b>-1; --b){
             for(a=2; a>-1; --a){
                 basis[coord]  = normal[a] * normal[b];
                 basisX[coord] = first[a] * normal[b];
                 basisY[coord] = normal[a] * first[b];
                 coord++;
             }
         }
     }

     DTYPE gradX, gradY, gradZ;
 #ifdef _OPENMP
 #pragma omp parallel for default(none) \
     shared(splineControlPoint, gradPtrX, gradPtrY, gradPtrZ, approxRatio, \
     basis, basisX, basisY, basisZ, dispPointPtrX, dispPointPtrY, dispPointPtrZ) \
     private(x, y, z, a, b, c, index, currentIndex, coord, \
     gradX, gradY, gradZ)
 #endif
     for(z=0;z<splineControlPoint->nz;z++){
         index=z*splineControlPoint->nx*splineControlPoint->ny;
         for(y=0;y<splineControlPoint->ny;y++){
             for(x=0;x<splineControlPoint->nx;x++){
                 gradX=0;
                 gradY=0;
                 gradZ=0;
                 coord=0;
                 if(splineControlPoint->nz>1){
                     for(c=z-1;c<z+2;c++){
                         for(b=y-1;b<y+2;b++){
                             currentIndex= (c*splineControlPoint->ny+b)*splineControlPoint->nx+x-1;
                             for(a=x-1;a<x+2;a++){

                                 if(c>-1 && b>-1 && a>-1 && c<splineControlPoint->nz && b<splineControlPoint->ny && a<splineControlPoint->nx){

                                     gradX += 2.0 * basis[coord] * dispPointPtrX[currentIndex];
                                     gradY += 2.0 * basis[coord] * dispPointPtrY[currentIndex];
                                     gradZ += 2.0 * basis[coord] * dispPointPtrZ[currentIndex];
                                 }

                                 currentIndex++;
                                 coord++;
                             }
                         }
                     }
                     gradPtrX[index] += approxRatio * gradX;
                     gradPtrY[index] += approxRatio * gradY;
                     gradPtrZ[index] += approxRatio * gradZ;
                 }
                 else{
                     for(b=y-1;b<y+2;b++){
                         currentIndex= b*splineControlPoint->nx+x-1;
                         for(a=x-1;a<x+2;a++){

                             if(b>-1 && a>-1 && b<splineControlPoint->ny && a<splineControlPoint->nx){
                                 gradX += 2.0 * basis[coord] * dispPointPtrX[currentIndex];
                                 gradY += 2.0 * basis[coord] * dispPointPtrY[currentIndex];
                             }

                             currentIndex++;
                             coord++;
                         }
                     }
                     gradPtrX[index] += approxRatio * gradX;
                     gradPtrY[index] += approxRatio * gradY;
                 }
                 index++;
             }
         }
     }
    nifti_image_free(dispControlPoint);
}
/* *************************************************************** */
void reg_bspline_L2norm_dispGradient(nifti_image *splineControlPoint,
                                     nifti_image *referenceImage,
                                     nifti_image *gradientImage,
                                     float weight)
{

    switch(splineControlPoint->datatype){
    case NIFTI_TYPE_FLOAT32:
        return reg_bspline_L2norm_dispGradient1<float>
                (splineControlPoint, referenceImage, gradientImage, weight);
        break;
    case NIFTI_TYPE_FLOAT64:
        return reg_bspline_L2norm_dispGradient1<double>
                (splineControlPoint, referenceImage, gradientImage, weight);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for reg_bspline_L2norm_dispGradient");
        fprintf(stderr,"[NiftyReg ERROR] The reg_bspline_L2norm_dispGradient is not computed\n");
        exit(1);
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_bspline_approxLinearEnergyGradient1(nifti_image *splineControlPoint,
                                              nifti_image *referenceImage,
                                              nifti_image *gradientImage,
                                              float weight0,
                                              float weight1
                                              )
{
    int nodeNumber = splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz;
    int voxNumber = referenceImage->nx*referenceImage->ny*referenceImage->nz;
    int a, b, c, x, y, z, index, coord, currentIndex;

    DTYPE *jacobianDet=(DTYPE *)malloc(nodeNumber*sizeof(DTYPE));
    mat33 *jacobianMatrices=(mat33 *)malloc(nodeNumber*sizeof(mat33));
    if(splineControlPoint->nz>1){
        reg_bspline_computeApproximateJacobianMatrices_3D(splineControlPoint,
                                                          jacobianMatrices,
                                                          jacobianDet);
    }
    else{
        reg_bspline_computeApproximateJacobianMatrices_2D(splineControlPoint,
                                                          jacobianMatrices,
                                                          jacobianDet);
    }
    free(jacobianDet);

    DTYPE *gradPtrX = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradPtrY = &gradPtrX[nodeNumber];
    DTYPE *gradPtrZ = NULL;
    if(splineControlPoint->nz>1)
        gradPtrZ = &gradPtrY[nodeNumber];

    DTYPE approxRatio0 = 2.0 * (double)weight0 * (double)(voxNumber) / (double)nodeNumber;
    DTYPE approxRatio1 = 2.0 * (double)weight1 * (double)(voxNumber) / (double)nodeNumber;

    DTYPE basis[27], basisX[27], basisY[27], basisZ[27], common;
    DTYPE normal[3]={1.0/6.0, 2.0/3.0, 1.0/6.0};
    DTYPE first[3]={-0.5, 0, 0.5};
    coord = 0;

    if(splineControlPoint->nz>1){
        for(c=2; c>-1; --c){
            for(b=2; b>-1; --b){
                for(a=2; a>-1; --a){
                    basis[coord]  = normal[a] * normal[b] * normal[c];
                    basisX[coord] = first[a] * normal[b] * normal[c];
                    basisY[coord] = normal[a] * first[b] * normal[c];
                    basisZ[coord] = normal[a] * normal[b] * first[c];
                    coord++;
                }
            }
        }
    }
    else{
        for(b=2; b>-1; --b){
            for(a=2; a>-1; --a){
                basis[coord]  = normal[a] * normal[b];
                basisX[coord] = first[a] * normal[b];
                basisY[coord] = normal[a] * first[b];
                coord++;
            }
        }
    }

    DTYPE gradX0, gradX1, gradX2, gradY0, gradY1, gradY2, gradZ0, gradZ1, gradZ2;
    mat33 reorient, desorient, jacobianMatrix;
    reg_getReorientationMatrix(splineControlPoint, &desorient, &reorient);
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(splineControlPoint, jacobianMatrices, \
    gradPtrX, gradPtrY, gradPtrZ, desorient, approxRatio0, approxRatio1, \
    basis, basisX, basisY, basisZ) \
    private(x, y, z, a, b, c, index, currentIndex, jacobianMatrix, coord, common, \
    gradX0, gradX1, gradX2, gradY0, gradY1, gradY2, gradZ0, gradZ1, gradZ2)
#endif
    for(z=0;z<splineControlPoint->nz;z++){
        index=z*splineControlPoint->nx*splineControlPoint->ny;
        for(y=0;y<splineControlPoint->ny;y++){
            for(x=0;x<splineControlPoint->nx;x++){
                gradX0=0; gradX1=0; gradX2=0;
                gradY0=0; gradY1=0; gradY2=0;
                gradZ0=0; gradZ1=0; gradZ2=0;
                coord=0;
                if(splineControlPoint->nz>1){
                    for(c=z-1;c<z+2;c++){
                        for(b=y-1;b<y+2;b++){
                            currentIndex= (c*splineControlPoint->ny+b)*splineControlPoint->nx+x-1;
                            for(a=x-1;a<x+2;a++){

                                if(c>-1 && b>-1 && a>-1 && c<splineControlPoint->nz && b<splineControlPoint->ny && a<splineControlPoint->nx){
                                    jacobianMatrix=jacobianMatrices[currentIndex];
                                    jacobianMatrix.m[0][0]--;
                                    jacobianMatrix.m[1][1]--;
                                    jacobianMatrix.m[2][2]--;

                                    gradX0 +=  basisY[coord] * (jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) +
                                            basisZ[coord] * (jacobianMatrix.m[0][2]+jacobianMatrix.m[2][0]) +
                                            2.0*basisX[coord] * jacobianMatrix.m[0][0];
                                    gradY0 +=  basisX[coord] * (jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) +
                                            basisZ[coord] * (jacobianMatrix.m[2][1]+jacobianMatrix.m[1][2]) +
                                            2.0*basisY[coord] * jacobianMatrix.m[1][1];
                                    gradZ0 +=  basisX[coord] * (jacobianMatrix.m[0][2]+jacobianMatrix.m[2][0]) +
                                            basisY[coord] * (jacobianMatrix.m[2][1]+jacobianMatrix.m[1][2]) +
                                            2.0*basisZ[coord] * jacobianMatrix.m[2][2];

                                    common=jacobianMatrix.m[0][0]+jacobianMatrix.m[1][1]+jacobianMatrix.m[2][2];
                                    gradX1 += 2.0 * basisX[coord] * common;
                                    gradY1 += 2.0 * basisY[coord] * common;
                                    gradZ1 += 2.0 * basisZ[coord] * common;
                                }

                                currentIndex++;
                                coord++;
                            }
                        }
                    }
                    gradPtrX[index] += approxRatio0 * (desorient.m[0][0]*gradX0 + desorient.m[0][1]*gradY0 + desorient.m[0][2]*gradZ0) +
                            approxRatio1 * (desorient.m[0][0]*gradX1 + desorient.m[0][1]*gradY1 + desorient.m[0][2]*gradZ1);
                    gradPtrY[index] += approxRatio0 * (desorient.m[1][0]*gradX0 + desorient.m[1][1]*gradY0 + desorient.m[1][2]*gradZ0) +
                            approxRatio1 * (desorient.m[1][0]*gradX1 + desorient.m[1][1]*gradY1 + desorient.m[1][2]*gradZ1);
                    gradPtrZ[index] += approxRatio0 * (desorient.m[2][0]*gradX0 + desorient.m[2][1]*gradY0 + desorient.m[2][2]*gradZ0) +
                            approxRatio1 * (desorient.m[2][0]*gradX1 + desorient.m[2][1]*gradY1 + desorient.m[2][2]*gradZ1);
                }
                else{
                    for(b=y-1;b<y+2;b++){
                        currentIndex= (c*splineControlPoint->ny+b)*splineControlPoint->nx+x-1;
                        for(a=x-1;a<x+2;a++){

                            if(b>-1 && a>-1 && b<splineControlPoint->ny && a<splineControlPoint->nx){
                                jacobianMatrix=jacobianMatrices[currentIndex];
                                jacobianMatrix.m[0][0]--;
                                jacobianMatrix.m[1][1]--;

                                gradX0 += basisY[coord] * (jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) +
                                          2.0*basisX[coord] * jacobianMatrix.m[0][0];
                                gradY0 += basisX[coord] * (jacobianMatrix.m[0][1]+jacobianMatrix.m[1][0]) +
                                          2.0*basisY[coord] * jacobianMatrix.m[1][1];

                                common=jacobianMatrix.m[0][0]+jacobianMatrix.m[1][1];
                                gradX1 += 2.0 * basisX[coord] * common;
                                gradY1 += 2.0 * basisY[coord] * common;
                            }

                            currentIndex++;
                            coord++;
                        }
                    }
                    gradPtrX[index] += approxRatio0 * (desorient.m[0][0]*gradX0 + desorient.m[0][1]*gradY0) +
                                       approxRatio1 * (desorient.m[0][0]*gradX1 + desorient.m[0][1]*gradY1) ;
                    gradPtrY[index] += approxRatio0 * (desorient.m[1][0]*gradX0 + desorient.m[1][1]*gradY0) +
                                       approxRatio1 * (desorient.m[1][0]*gradX1 + desorient.m[1][1]*gradY1) ;
                }
                index++;
            }
        }
    }
    free(jacobianMatrices);
}
/* *************************************************************** */
void reg_bspline_linearEnergyGradient(nifti_image *splineControlPoint,
                                      nifti_image *targetImage,
                                      nifti_image *gradientImage,
                                      float weight0,
                                      float weight1
                                      )
{
    if(splineControlPoint->datatype != gradientImage->datatype){
        fprintf(stderr,"[NiftyReg ERROR] The spline control point image and the gradient image were expected to have the same datatype\n");
        fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
        exit(1);
    }
    switch(splineControlPoint->datatype){
    case NIFTI_TYPE_FLOAT32:
        reg_bspline_approxLinearEnergyGradient1<float>
                (splineControlPoint, targetImage, gradientImage, weight0, weight1);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_bspline_approxLinearEnergyGradient1<double>
                (splineControlPoint, targetImage, gradientImage, weight0, weight1);
        break;
    default:
        fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
        fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not been computed\n");
        exit(1);
    }
}
/* *************************************************************** */
